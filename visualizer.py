# visualizer.py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import cv2

def plot_piv_image_pair(img1, img2, output_file=None):
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)           # 1行2列，第一张图
    plt.imshow(img1, cmap='gray')
    plt.title('Image 1')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)           # 1行2列，第二张图
    plt.imshow(img2, cmap='gray')
    plt.title('Image 2')
    plt.axis('off')

    if output_file is not None:
        plt.savefig(output_file, bbox_inches='tight', pad_inches=0.1)
        print(f"Figure saved to {output_file}")
    plt.show()


def save_piv_video(image_seq, output_file="output/output.mp4", axis=0, fps=None):
    """
    将生成的图像帧序列导出为 MP4 视频文件。
    
    参数：
    - output_file: 输出视频文件路径（默认 "output.mp4"）
    
    要求：
    - self.imgs: 已生成的图像帧列表，形状为 [H, W]，灰度图
    """
    fps = int(len(image_seq)/15) if fps is None else fps  # 帧率，可根据模拟需要调整

    if len(image_seq) == 0:
        raise ValueError("图像帧未生成")

    # 自动获取帧大小（支持任意宽高）
    frame_height, frame_width = image_seq[0].shape[:2]
    video_size = (frame_width, frame_height)

    # 初始化视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_file, fourcc, fps, video_size)

    for frame in image_seq:
        # 确保类型和通道数为 uint8 & RGB
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        if frame.ndim == 2:
            frame = np.stack([frame] * 3, axis=-1)  # 灰度转RGB

        video_writer.write(frame)

    video_writer.release()
    print(f"[INFO] 视频保存至 {output_file}")

def plot_events_3d(events, output_file=None, startN=-10000, endN=-1):
    x = np.asarray(events[startN:endN,0])
    y = np.asarray(events[startN:endN,1])
    t = np.asarray(events[startN:endN,2])
    p = np.asarray(events[startN:endN,3])


    def expand_range(data, ratio=0.05):
        d_min, d_max = np.min(data), np.max(data)
        d_range = d_max - d_min
        if d_range == 0:
            d_range = 1e-6  # 防止为0时仍然是线
        return d_min - ratio * d_range, d_max + ratio * d_range
    
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(projection='3d')
    ax.scatter(x, y, t, marker='.', c=p)
    # 自动扩展三个坐标轴范围
    # ax.set_xlim(*expand_range(x))
    # ax.set_ylim(*expand_range(y))
    # ax.set_zlim(*expand_range(t))
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('T')
    if output_file is not None:
        plt.savefig(output_file, bbox_inches='tight', pad_inches=0.1)
        print(f"Figure saved to {output_file}")
    plt.show()


def plot_event_time2d(events, output_file=None, vmin=None, vmax=None):
    """
        events: numpy.ndarray, 形状 (N,4)，列为 x, y, t, p
        vmin, vmax: 时间映射的颜色范围（可选）
    """
    # 分割事件数组
    x, y, t, p = np.split(events, 4, axis=1)
    x = x.flatten().astype(int)
    y = y.flatten().astype(int)
    t = t.flatten()
    
    # 计算坐标偏移与图像大小
    x0, y0 = np.min(x), np.min(y)
    x_sz = np.max(x) - x0 + 1
    y_sz = np.max(y) - y0 + 1
    
    # 创建空图像，行对应y，列对应x
    img = np.zeros((x_sz, y_sz))+0.0
    
    # 赋值：注意索引顺序是 img[y, x]
    for xi, yi, ti in zip(x, y, t):
        yi_idx = yi - y0
        xi_idx = xi - x0
        img[xi_idx, yi_idx] = max(img[xi_idx, yi_idx], ti)  # 同点多事件取最大时间
    
    # 设置vmin/vmax默认值
    if vmin is None:
        vmin = np.min(t)
    if vmax is None:
        vmax = np.max(t)

    plt.figure(figsize=(6, 5))
    plt.imshow(img, cmap='jet', vmin=vmin, vmax=vmax)
    plt.colorbar(label='Timestamp (ms)')
    # plt.axis('off')
    plt.title('2D Event Time Visualization')
    
    if output_file is not None:
        plt.savefig(output_file, bbox_inches='tight', pad_inches=0.1)
        print(f"Figure saved to {output_file}")
    plt.show()


def save_voxel_grid_video(events, output_file="output/voxelgrids.mp4",img_size=(256,256), fps=10):
    def _event2voxelgrid3(events, img_size, num_bins=100, t_min=None, t_max=None):
        """
        将事件数据转换为3通道体素网格：正极性、负极性、总和。
    
        参数：
            events: np.ndarray, shape (N, 4)，每列为 (x, y, t, p)
            img_size: tuple (H, W)，图像空间尺寸
            num_bins: int，体素时间层数
            t_min: float, 时间最小值，None时使用events中最小t
            t_max: float, 时间最大值，None时使用events中最大t
    
        返回：
            voxel_grid: np.ndarray, shape (num_bins, H, W, 3)
        """
        x, y, t, p = events[:, 0], events[:, 1], events[:, 2], events[:, 3]
        H, W = img_size
    
        if t_min is None:
            t_min = t.min()
        if t_max is None:
            t_max = t.max()
        t_range = t_max - t_min
        if t_range == 0:
            t_range = 1e-9
    
        t_norm = (t - t_min) / t_range
        bin_idx = (t_norm * (num_bins - 1)).astype(int)
        bin_idx = np.clip(bin_idx, 0, num_bins - 1)
    
        # 初始化3个通道
        voxel_grid = np.zeros((num_bins, H, W, 3), dtype=np.float32)
    
        x_idx = x.astype(int)
        y_idx = y.astype(int)
    
        # 越界事件过滤
        mask = (x_idx >= 0) & (x_idx < H) & (y_idx >= 0) & (y_idx < W)
        x_idx = x_idx[mask]
        y_idx = y_idx[mask]
        bin_idx = bin_idx[mask]
        p = p[mask]
    
        for b, yi, xi, pol in zip(bin_idx, y_idx, x_idx, p):
            if pol > 0:
                voxel_grid[b, xi, yi, 2] += 1  # 正事件通道
            else:
                voxel_grid[b, xi, yi, 0] += 1  # 负事件通道
            # voxel_grid[b, xi, yi, 2] += pol   # 正负总和通道
    
        return voxel_grid
    # voxel_grid = _event2voxelgrid(events, img_size, num_bins=10, t_min=None, t_max=None)
    voxel_grid = _event2voxelgrid3(events, img_size, num_bins=10, t_min=None, t_max=None)
    voxel_grid_visual = np.clip(voxel_grid*512/np.max(voxel_grid),a_min=0,a_max=255)
    mask = np.all(voxel_grid_visual == 0, axis=-1)
    voxel_grid_visual[mask] = 255
    save_piv_video(voxel_grid_visual, output_file=output_file, axis=0, fps=fps)
    return True
    

def plot_flow_field(u, v, output_file=None, magFlag=True, slFlag=True,step=1,quiver_step=16):
    """
    可视化速度场：
    - 背景：速度大小的热力图
    - 前景：流线图 + 稀疏箭头图（quiver）

    参数：
    - u, v: 速度场的两个分量，形状 [H, W]
    - step: 背景图采样间隔（默认=1，表示全分辨率）
    - quiver_step: 箭头稀疏采样间隔（默认=10）
    """
    H, W = u.shape
    x = np.arange(0, H)
    y = np.arange(0, W)
    xx, yy = np.meshgrid(x, y, indexing='ij')

    # 速度大小
    magnitude = np.sqrt(u**2 + v**2)

    plt.figure(figsize=(6, 5))
    # plt.figure(figsize=(12, 10))
    
    # 背景热力图（速度大小）
    if magFlag:
        plt.imshow(magnitude[::step, ::step], cmap='viridis', origin='lower')
        plt.colorbar(label='Speed Magnitude')
    else:
        plt.imshow(0*magnitude[::step, ::step], cmap='viridis', origin='lower')
        plt.colorbar(label='')

    # # 流线图（streamplot）
    if slFlag:
        plt.streamplot(yy,xx,v,u, color='white', linewidth=0.8, density=1.0)

    # 稀疏箭头图（quiver）
    plt.quiver(yy[::quiver_step, ::quiver_step],
               xx[::quiver_step, ::quiver_step],
               v[::quiver_step, ::quiver_step],
               -u[::quiver_step, ::quiver_step],
               color='red', scale=None)

    plt.title("Vector Field Visualization")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.xlim([-0.1,W+0.1])
    plt.ylim([-0.1,H+0.1])
    plt.gca().invert_yaxis()
    plt.tight_layout()
    if output_file is not None:
        plt.savefig(output_file, bbox_inches='tight', pad_inches=0.1)
        print(f"Figure saved to {output_file}")
            
    plt.show()