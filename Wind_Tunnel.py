import numpy as np
from collections import deque
import torch
import cv2
import time
from skimage.registration import optical_flow_ilk
from concurrent.futures import ThreadPoolExecutor


class EventBufferManager:
    """
    事件缓冲区管理器：
    - 接收输入的事件流（形状 Nx4），缓存至固定大小块（默认 400 万）
    - 自动分块，一旦满块即排入 chunk 队列供后续处理
    """

    def __init__(self, chunk_size=4_000_000):
        self.chunk_size = chunk_size
        self.buffer_index = 0
        self.buffer = np.empty((chunk_size, 4), dtype=np.float32)  # 当前正在填充的缓冲块
        self.chunk_queue = deque()  # 存放完整块的队列

    def append(self, events_new: np.ndarray):
        """
        将新事件追加到缓冲区，并自动分块
        参数:
            events_new: 新输入的事件 (N, 4)，每行为 (x, y, t, p)
        """
        i = 0
        while i < len(events_new):
            space_left = self.chunk_size - self.buffer_index
            num_copy = min(space_left, len(events_new) - i)

            # 拷贝事件数据到缓冲区
            self.buffer[self.buffer_index:self.buffer_index + num_copy] = events_new[i:i + num_copy]
            self.buffer_index += num_copy
            i += num_copy

            # 若当前块已满，则加入队列
            if self.buffer_index == self.chunk_size:
                self.chunk_queue.append(self.buffer.copy())
                self.buffer_index = 0

    def has_next_chunk(self):
        """是否有完整块可以处理"""
        return len(self.chunk_queue) > 0

    def get_next_chunk(self):
        """弹出一个完整事件块 (chunk_size, 4)"""
        return self.chunk_queue.popleft() if self.has_next_chunk() else None
        

class VoxelGridBuilder:
    """
    体素网格构建器，支持重叠体素层（Overlapping Bins）：
    - 事件均分划分，滑动窗口重叠构建体素层
    """

    def __init__(self, num_bins, height, width, overlap=0.3, device='cuda'):
        """
        参数：
            num_bins: 划分体素层数量
            height, width: 图像尺寸
            device: torch设备
            overlap: 重叠比例，0 表示无重叠，0.5 表示 50% 重叠
        """
        assert 0 <= overlap < 1, "overlap需在[0,1)范围"
        self.num_bins = num_bins
        self.height = height
        self.width = width
        self.device = device
        self.overlap = overlap

    def build_voxel_grid(self, events: np.ndarray):
        """
        构建体素网格，事件均分，滑动窗口重叠

        参数:
            events: ndarray (N,4) 事件 (x,y,t,p)

        返回:
            voxel_grid: tensor (num_bins_eff, H, W)
            duration: 事件时间跨度
        """
        if len(events) == 0:
            raise ValueError("事件为空，无法构建体素网格")

        events_torch = torch.from_numpy(events).to(self.device)
        x = events_torch[:, 0].long().clamp(0, self.width - 1)
        y = events_torch[:, 1].long().clamp(0, self.height - 1)
        p = events_torch[:, 3].float()

        t0 = events_torch[:, 2].min()
        t1 = events_torch[:, 2].max()
        duration = t1 - t0

        N = len(events)
        window_size = int(N // self.num_bins)
        stride = int(window_size * (1 - self.overlap))
        if stride < 1:
            stride = 1  # 避免步长为0

        # 计算有效体素层数量
        num_bins_eff = max(1, (N - window_size) // stride + 1)

        voxel_grid = torch.zeros((num_bins_eff, self.height, self.width), dtype=torch.float32, device=self.device)

        for b in range(num_bins_eff):
            s = b * stride
            e = s + window_size
            if e > N:
                e = N
            xb = x[s:e]
            yb = y[s:e]
            pb = p[s:e]

            voxel_grid[b].index_put_((yb, xb), pb, accumulate=True)

        return voxel_grid, duration


class OpticalFlowCalculator:
    """
    光流计算器：基于 ILK 算法计算连续层之间的光流
    """

    def __init__(self, resize_factor=2, max_workers=4, device='cpu',
                 radius=16, num_warp=1, ksize=(7, 7)):
        """
        参数:
            resize_factor: 下采样比例
            max_workers: 并行线程数
            device: 返回张量设备
            radius: ILK 半径
            num_warp: ILK warp 次数
            ksize: 模糊核大小（降噪）
        """
        self.resize_factor = resize_factor
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.device = device
        self.radius = radius
        self.num_warp = num_warp
        self.ksize = ksize

    def compute_from_voxel(self, voxel_grid):
        """计算体素连续层之间的光流"""
        num_bins = voxel_grid.shape[0]
        if num_bins < 2:
            return None

        flow_tensors = []
        futures = []

        for j in range(num_bins - 1):
            img1 = voxel_grid[j].detach().cpu().numpy()
            img2 = voxel_grid[j + 1].detach().cpu().numpy()
            futures.append(self.executor.submit(self._compute_ilk_flow_tensor, img1, img2))

        for future in futures:
            flow = future.result()
            if flow is not None:
                flow_tensors.append(flow)

        if flow_tensors:
            return torch.stack(flow_tensors, dim=0).to(self.device)
        return None

    def _compute_ilk_flow_tensor(self, img1, img2):
        """使用 ILK 计算光流 (2, H, W)"""
        h, w = img1.shape
        img1 = cv2.blur(img1, self.ksize)
        img2 = cv2.blur(img2, self.ksize)

        new_size = (int(w / self.resize_factor), int(h / self.resize_factor))
        img1 = cv2.resize(img1, new_size).astype(np.float32)
        img2 = cv2.resize(img2, new_size).astype(np.float32)

        u, v = optical_flow_ilk(
            img1, img2,
            radius=self.radius,
            num_warp=self.num_warp,
            gaussian=False,
            prefilter=True
        )

        u = cv2.resize(u, (w, h))
        v = cv2.resize(v, (w, h))

        return torch.from_numpy(np.stack([u, v], axis=0)).float()



class EventProcessingPipeline:
    """
    主处理流水线：
    - 管理事件缓存、体素构建和光流估计
    - 提供 batch 级光流估计接口
    """

    def __init__(self,
                 event_chunk_size=4_000_000,
                 voxel_params={'num_bins': 4, 'height': 720, 'width': 1280, 'overlap':0.1},
                 flow_params={'resize_factor': 2, 'max_workers': 4},
                 device='cuda'):
        self.buffer_manager = EventBufferManager(chunk_size=event_chunk_size)
        self.voxel_builder = VoxelGridBuilder(
            num_bins=voxel_params['num_bins'],
            height=voxel_params['height'],
            width=voxel_params['width'],
            overlap = voxel_params['overlap'],
            device=device
        )
        self.flow_calculator = OpticalFlowCalculator(
            resize_factor=flow_params['resize_factor'],
            max_workers=flow_params['max_workers'],
            radius=flow_params['radius'],
            num_warp=flow_params['num_warp'],
            ksize=flow_params['ksize'],
            device=device
        )


        self.device = device
        self.last_event_time = None

        self.stats = {
            'events_processed': 0,
            'voxels_generated': 0,
            'flows_generated': 0,
            'processing_time': 0.0
        }

    def process_events(self, events: np.ndarray):
        """
        主处理函数：传入事件，自动分块 → 构建体素 → 计算光流
        参数:
            events: ndarray(N, 4)
        返回:
            full_flow: (T, 2, H, W) or None
        """
        start_time = time.time()

        if len(events) > 0:
            self.last_event_time = np.max(events[:, 2])

        self.buffer_manager.append(events)
        self.stats['events_processed'] += len(events)

        all_flows = []

        while self.buffer_manager.has_next_chunk():
            event_chunk = self.buffer_manager.get_next_chunk()
            voxel_grid, _ = self.voxel_builder.build_voxel_grid(event_chunk)
            self.stats['voxels_generated'] += 1

            flow_tensors = self.flow_calculator.compute_from_voxel(voxel_grid)
            if flow_tensors is not None:
                all_flows.append(flow_tensors)
                self.stats['flows_generated'] += flow_tensors.shape[0]

            # 每个完整块（400万事件）打印一次进度
            if self.stats['events_processed'] % self.buffer_manager.chunk_size == 0:
                self.stats['processing_time'] += time.time() - start_time
                fps = (self.stats['flows_generated'] / self.stats['processing_time']
                       if self.stats['processing_time'] > 0 else 0)
                print(f"[进度] 已处理事件: {self.stats['events_processed']} | "
                      f"体素块: {self.stats['voxels_generated']} | "
                      f"光流帧: {self.stats['flows_generated']} | "
                      f"平均帧率: {fps:.2f} fps")
                start_time = time.time()

        self.stats['processing_time'] += time.time() - start_time

        if all_flows:
            return torch.cat(all_flows, dim=0)
        return None

    def get_stats(self):
        """获取处理统计信息（含每秒事件数、光流数）"""
        stats = self.stats.copy()
        t = stats['processing_time']
        if t > 0:
            stats['events_per_sec'] = stats['events_processed'] / t
            stats['voxels_per_sec'] = stats['voxels_generated'] / t
            stats['flows_per_sec'] = stats['flows_generated'] / t
        return stats

    def reset(self):
        """重置状态与统计信息"""
        self.last_event_time = None
        self.stats = {
            'events_processed': 0,
            'voxels_generated': 0,
            'flows_generated': 0,
            'processing_time': 0.0
        }
