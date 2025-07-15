import os
import cv2
import time
import torch
import numpy as np
import seaborn as sns
from tqdm import tqdm
import concurrent.futures
import matplotlib.pyplot as plt


class Events_to_voxelgrid:
    def __init__(self, width=256, height=256, device='cuda', eps=1e-6):
        """
        初始化体素网格转换器
        
        参数:
        width (int): 宽度
        height (int): 高度
        device (str): 计算设备 ('cpu' or 'cuda')
        eps (float): 防止除零的小常数
        """
        self.width = width
        self.height = height
        self.device = device
        self.eps = eps

    def events_to_voxel_grid(self, events, num_bins, eps=1e-6, normalize_voxel=False):
        """
        将事件数据转换为体素网格并进行归一化
        
        参数:
        events (np.ndarray): 事件数据 [t, x, y, p]
        num_bins (int): 时间通道数
        eps (float): 防止除零的小常数
        normalize_voxel (bool): 是否归一化体素网格
        
        返回:
        voxel_grid (Tensor): 归一化后的体素网格
        """
        H, W = self.height, self.width
        
        events = torch.as_tensor(events, dtype=torch.float32, device=self.device)
        
        # 如果没有事件数据，返回零体素网格
        if events.numel() == 0:
            return torch.zeros((1, num_bins * 2, H, W), device=self.device)
        
        ts, xs, ys, pols = events[:, 0], events[:, 1], events[:, 2], events[:, 3]
        
        # 坐标裁剪
        xs = torch.clamp(xs, 0, W - 1).long()
        ys = torch.clamp(ys, 0, H - 1).long()
        pols = (pols > 0).float()

        # 时间归一化到 [0, num_bins - 1]
        t_min, t_max = ts.min(), ts.max()
        delta_t = max(t_max - t_min, 1e-8)
        norm_ts = (ts - t_min) * (num_bins - 1) / delta_t

        # 计算bin值
        tis = torch.floor(norm_ts).long()
        dts = norm_ts - tis
        vals_left = 1.0 - dts
        vals_right = dts

        # 创建空的体素网格
        voxel_grid = torch.zeros(num_bins * 2, H, W, device=self.device)
        hw = H * W

        # 按极性分通道存储
        for polarity_val, channel_offset in [(1.0, 0), (0.0, num_bins)]:
            mask = (pols == polarity_val)
            if not mask.any():
                continue

            # 仅处理有效的事件
            xs_p = xs[mask]
            ys_p = ys[mask]
            tis_p = tis[mask]
            vals_l = vals_left[mask]
            vals_r = vals_right[mask]

            # 计算空间索引
            spatial_idx = ys_p * W + xs_p
            bin_l = tis_p + channel_offset
            bin_r = tis_p + 1 + channel_offset

            # 仅处理有效的bin
            valid_l = (bin_l >= 0) & (bin_l < num_bins * 2)
            valid_r = (bin_r >= 0) & (bin_r < num_bins * 2)

            # 索引计算
            idx_left = bin_l[valid_l] * hw + spatial_idx[valid_l]
            idx_right = bin_r[valid_r] * hw + spatial_idx[valid_r]

            # 使用index_add_进行累加
            voxel_grid.view(-1).index_add_(0, idx_left, vals_l[valid_l])
            voxel_grid.view(-1).index_add_(0, idx_right, vals_r[valid_r])

        # 优化裁剪过程
        self.optimize_voxel_grid(voxel_grid, num_bins)

        # 添加batch维度并进行归一化
        voxel_grid = voxel_grid.unsqueeze(0)  # [1, C, H, W]
        
        if normalize_voxel:
            mean = voxel_grid.mean(dim=(2, 3), keepdim=True)
            std = voxel_grid.std(dim=(2, 3), keepdim=True)
            voxel_grid = (voxel_grid - mean) / (std + eps)

        return voxel_grid

    def optimize_voxel_grid(self, voxel_grid, num_bins):
        """
        优化体素网格裁剪过程
        
        通过使用分位数来避免完全排序，提高速度。
        """
        with torch.no_grad():
            for c in range(num_bins * 2):
                img = voxel_grid[c]
                flat_img = img.view(-1)
                
                # 使用高效的分位数估计
                sorted_vals, _ = torch.topk(flat_img, k=min(100, len(flat_img)), largest=True)
                top_val = sorted_vals[-1] if len(sorted_vals) > 0 else 0
                
                # 如果最大值很小则跳过裁剪
                if top_val > 1e-3:
                    voxel_grid[c] = torch.clamp(img, max=top_val)
                else:
                    voxel_grid[c] = img


    def segment_to_voxel_grids(self, prev_events, last_events, num_bins, normalize_voxel=False):
        """
        获取两个时间段的体素网格
        返回: prev_voxel, last_voxel
        """
        prev_voxel = self.events_to_voxel_grid(prev_events, num_bins, normalize_voxel=False)
        last_voxel = self.events_to_voxel_grid(last_events, num_bins, normalize_voxel=False)
        return prev_voxel, last_voxel


class Optical_Flow:
    def __init__(self, config=None):
        self.default_config = {
            'pyr_scale': 0.5,
            'levels': 4,
            'winsize': 25,
            'iterations': 3,
            'poly_n': 7,
            'poly_sigma': 1.5,
            'clahe_clip_limit': 2.0,
        }

        self.config = self.default_config.copy()
        if config is not None:
            self.config.update(config)

        self.device = torch.device("cpu")  # 使用 CPU
        self.clahe = cv2.createCLAHE(clipLimit=self.config['clahe_clip_limit'], tileGridSize=(8, 8))

        print(f"[INFO] Using CPU for Farneback")

    def enhance_voxel_channel(self, channel):
        norm = cv2.normalize(channel, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        # clahe = self.clahe.apply(norm)
        smooth = cv2.GaussianBlur(norm, (5, 5), 1.5)
        return smooth

    def calc_flow(self, img1, img2, flow_init=None):
        # 如果提供了初始光流，直接使用
        if flow_init is not None:
            return cv2.calcOpticalFlowFarneback(
                img1, img2, flow_init,
                pyr_scale=self.config['pyr_scale'],
                levels=self.config['levels'],
                winsize=self.config['winsize'],
                iterations=self.config['iterations'],
                poly_n=self.config['poly_n'],
                poly_sigma=self.config['poly_sigma'],
                flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN
            )
        else:
            return cv2.calcOpticalFlowFarneback(
                img1, img2, None,
                pyr_scale=self.config['pyr_scale'],
                levels=self.config['levels'],
                winsize=self.config['winsize'],
                iterations=self.config['iterations'],
                poly_n=self.config['poly_n'],
                poly_sigma=self.config['poly_sigma'],
                flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN
            )

    def compute_flow_for_channel(self, v1, v2, flow_u, flow_v, c, flow_init=None):
        # 计算单通道的光流
        # todo blur(i1,i2).cv2.blur
        i1 = self.enhance_voxel_channel(v1[c])
        i2 = self.enhance_voxel_channel(v2[c])
        flow = self.calc_flow(i1, i2, flow_init)
        flow_u[c], flow_v[c] = flow[..., 0], flow[..., 1]

    def estimate_flow_single_parallel(self, voxel1, voxel2, prev_flow_u=None, prev_flow_v=None):
        # 估计光流：单向，从 voxel1 到 voxel2，使用并行处理
        # 输入: voxel1, voxel2: torch.Tensor, shape = (1, C, H, W)
        # 输出: u, v: np.ndarray, shape = (H, W)
        
        _, C, H, W = voxel1.shape
        v1, v2 = voxel1[0].cpu().numpy(), voxel2[0].cpu().numpy()

        flow_u = np.zeros((C, H, W), dtype=np.float32)
        flow_v = np.zeros((C, H, W), dtype=np.float32)

        # 使用并行处理每个通道的光流计算
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for c in range(C):
                # 使用上次计算的光流作为初始值
                flow_init = None
                if prev_flow_u is not None and prev_flow_v is not None:
                    flow_init = np.stack([prev_flow_u[c], prev_flow_v[c]], axis=-1)
                
                futures.append(executor.submit(self.compute_flow_for_channel, v1, v2, flow_u, flow_v, c, flow_init))
            
            # 等待所有线程完成
            for future in futures:
                future.result()

        # 计算最终的平均光流
        return np.mean(flow_u, axis=0), np.mean(flow_v, axis=0)

    def estimate_flow(self, voxel1, voxel2, prev_flow_u=None, prev_flow_v=None):
        u_fwd, v_fwd = self.estimate_flow_single_parallel(voxel1, voxel2, prev_flow_u, prev_flow_v)
        
        # 反向光流可以根据需要计算，但目前这里只计算正向光流
        u = u_fwd
        v = v_fwd
        return u, v

class OfflineEventProcessor:
    def __init__(self, width=1280, height=720, num_bins=2, duration=10, interval=2,
                 flow_config=None, parallel_segments=32):
        """
        离线事件处理器 - 从预加载的事件数组计算光流
        
        参数:
            width, height: 图像分辨率
            num_bins: 体素网格的时间通道数
            duration: 每个事件段持续时间(秒)
            interval: 连续事件段之间的时间间隔(秒)
            parallel_segments: 同时处理的事件段数量
            flow_config: 光流估计器配置参数
        """
        # 初始化参数
        self.width = width
        self.height = height
        self.num_bins = num_bins
        self.duration = duration
        self.interval = interval
        self.parallel_segments = parallel_segments
        
        # 初始化处理模块
        self.voxel_converter = Events_to_voxelgrid(
            width=width, 
            height=height,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        self.flow_estimator = Optical_Flow(config=flow_config)
        
        # 处理状态跟踪
        self.frame_counter = 0
        self.compute_times = []
        
        # 时间记录
        self.event_extraction_time = 0
        self.voxel_construction_time = 0
        self.flow_computation_time = 0
        
    def process_events(self, evts):
        """
        处理整个事件数组并返回光流结果
        
        返回:
            光流结果数组，形状为(n, 2, h, w)，其中：
            - n: 有效光流帧数
            - 2: 通道维度（0为u方向，1为v方向）
            - h, w: 图像高度和宽度
        """
        total_start = time.time()
        
        event_start = time.time()  # 事件数据提取开始时间
        events_new = evts.copy()
        events_new = events_new[:, [2, 0, 1, 3]]  # [t, x, y, p]
        events_new[:, 0] = (events_new[:, 0] - events_new[0, 0])  # 时间相对化
        events_new = events_new[np.argsort(events_new[:, 0])]  # 按时间排序
        self.event_extraction_time += time.time() - event_start  # 累积事件提取时间
        
        # 获取时间范围
        t_min, t_max = events_new[0, 0], events_new[-1, 0]
        
        # 计算总段数
        total_segments = int((t_max - t_min - self.duration) // (self.interval + self.duration))
        print(f"Total segments to process: {total_segments}")
        
        # 使用tqdm创建进度条
        pbar = tqdm(total=total_segments, desc="Processing events")
        
        # 存储所有有效光流结果（过滤None值）
        all_flow_results = []
        
        # 主处理循环
        current_time = t_min
        while current_time + self.duration + self.interval <= t_max:
            # 计算批处理的时间范围
            start_times = [current_time + i * (self.interval + self.duration)
                          for i in range(self.parallel_segments)]
            
            # 确保不超过时间范围
            if start_times[-1] + self.duration + self.interval > t_max:
                start_times = [t for t in start_times 
                              if t + self.duration + self.interval <= t_max]
                if not start_times:
                    break
            
            # 处理当前批次并收集结果
            batch_results = self._process_batch(events_new, start_times)
            # 过滤None值并添加到总结果
            valid_batch = [flow for flow in batch_results if flow is not None]
            all_flow_results.extend(valid_batch)
            
            # 更新时间指针和进度条
            current_time = start_times[-1] + self.interval + self.duration
            pbar.update(len(start_times))
        
        pbar.close()
        
        # 将列表转换为(n, 2, h, w)的numpy数组
        if all_flow_results:
            flow_array = np.stack(all_flow_results, axis=0)
        else:
            flow_array = np.array([])  # 空数组表示无有效结果
        
        # 性能报告
        print(f"\nProcessing summary (compute time only):")
        print(f"Total valid flow frames: {len(all_flow_results)}")
        if self.compute_times:
            total_compute_time = sum(self.compute_times)
            avg_time = total_compute_time / len(all_flow_results) if len(all_flow_results) > 0 else 0
            fps = len(all_flow_results) / total_compute_time if total_compute_time > 0 else 0
            print(f"Average compute time per frame: {avg_time:.4f} sec/frame")
            print(f"Compute FPS: {fps:.1f}")
            print(f"Total compute time: {total_compute_time:.2f} seconds")
        
        # 总处理时间
        total_processing_time = time.time() - total_start
        print(f"Total processing time: {total_processing_time:.2f} seconds")

        # 输出分部分时间统计
        print(f"\nTime Breakdown:")
        print(f"Event Extraction Time: {self.event_extraction_time:.2f} seconds")
        print(f"Voxel Construction Time: {self.voxel_construction_time:.2f} seconds")
        print(f"Flow Computation Time: {self.flow_computation_time:.2f} seconds")
        
        return flow_array
        
    def _process_batch(self, evts, start_times):
        """处理一批连续的事件段，返回该批次的光流结果"""
        batch_start = time.time()
        
        results = []  # 存储当前批次的光流结果（每个为(2, h, w)）
        
        # 提取事件段
        segment_events = []
        next_segment_events = []
        
        for t_start in start_times:
            t_end = t_start + self.duration
            mask1 = (evts[:, 0] >= t_start) & (evts[:, 0] < t_end)
            seg_events = evts[mask1]
            
            t_next_start = t_start + self.interval
            t_next_end = t_end + self.interval
            mask2 = (evts[:, 0] >= t_next_start) & (evts[:, 0] < t_next_end)
            next_events = evts[mask2]
            
            segment_events.append(seg_events)
            next_segment_events.append(next_events)
        
        # 并行处理
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # 构建体素网格
            voxel_futures = []
            next_voxel_futures = []
            
            voxel_start = time.time()  # 体素网格构建开始时间
            for i in range(len(start_times)):
                if len(segment_events[i]) > 100 and len(next_segment_events[i]) > 100:
                    voxel_futures.append(executor.submit(
                        self.voxel_converter.events_to_voxel_grid, 
                        segment_events[i], self.num_bins
                    ))
                    next_voxel_futures.append(executor.submit(
                        self.voxel_converter.events_to_voxel_grid, 
                        next_segment_events[i], self.num_bins
                    ))
                else:
                    voxel_futures.append(None)
                    next_voxel_futures.append(None)
            self.voxel_construction_time += time.time() - voxel_start  # 累积体素构建时间
            
            # 计算光流
            flow_futures = []
            flow_start = time.time()  # 光流计算开始时间
            for i in range(len(start_times)):
                if voxel_futures[i] and next_voxel_futures[i]:
                    try:
                        voxel1 = voxel_futures[i].result()
                        voxel2 = next_voxel_futures[i].result()
                        flow_futures.append(executor.submit(
                            self.flow_estimator.estimate_flow, 
                            voxel1, voxel2
                        ))
                    except Exception as e:
                        print(f"Error in flow estimation: {e}")
                        flow_futures.append(None)
                else:
                    flow_futures.append(None)
            self.flow_computation_time += time.time() - flow_start  # 累积光流计算时间
            
            # 收集结果并调整形状为(2, h, w)
            for i in range(len(start_times)):
                if flow_futures[i]:
                    try:
                        u, v = flow_futures[i].result()
                        # 确保u和v是二维数组 (h, w)
                        if u.ndim == 2 and v.ndim == 2:
                            # 堆叠为(2, h, w)，0通道为u，1通道为v
                            flow_2hw = np.stack([u, v], axis=0)
                            results.append(flow_2hw)
                            self.frame_counter += 1
                        else:
                            print(f"Invalid flow shape: u={u.shape}, v={v.shape}")
                            results.append(None)
                    except Exception as e:
                        print(f"Error collecting flow result: {e}")
                        results.append(None)
                else:
                    results.append(None)
        
        # 记录纯计算时间
        batch_compute_time = time.time() - batch_start
        self.compute_times.append(batch_compute_time)
        
        return results
