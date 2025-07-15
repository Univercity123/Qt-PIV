import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from PyQt5.QtCore import QThread, pyqtSignal, QMutex
from PyQt5 import QtGui

class FlowVisualizer(QThread):
    """独立的流场可视化线程"""
    visualization_ready = pyqtSignal(object)  # 信号：可视化结果已准备好
    
    def __init__(self, u, v, target_width, target_height):
        super().__init__()
        self.u = u
        self.v = v
        self.target_width = target_width  # 目标显示宽度
        self.target_height = target_height  # 目标显示高度
        self.mutex = QMutex()
        self.active = True
    
    def run(self):
        if not self.active or self.u is None or self.v is None:
            return
            
        try:
            # 计算速度大小
            magnitude = np.sqrt(self.u**2 + self.v**2)
            
            # 使用目标尺寸创建图形
            dpi = 100
            fig_width = self.target_width / dpi
            fig_height = self.target_height / dpi
            fig = plt.figure(figsize=(fig_width, fig_height), dpi=dpi)
            ax = fig.add_subplot(111)
            
            # 显示速度大小热力图
            im = ax.imshow(magnitude, cmap='viridis', origin='upper', 
                          extent=[0, self.u.shape[1], 0, self.u.shape[0]])
            plt.colorbar(im, ax=ax, label='Speed')
            
            # 稀疏箭头图
            step = max(1, int(min(self.u.shape[0], self.u.shape[1]) / 20))  # 动态计算步长
            h, w = magnitude.shape
            quiver_x = np.arange(0, w, step)
            quiver_y = np.arange(0, h, step)
            X, Y = np.meshgrid(quiver_x, quiver_y)
            
            # 获取箭头位置的速度
            U = self.u[Y.astype(int), X.astype(int)]
            V = self.v[Y.astype(int), X.astype(int)]
            
            # 过滤掉非常小的向量
            speed_threshold = 0.1
            mask = np.sqrt(U**2 + V**2) > speed_threshold
            X = X[mask]
            Y = Y[mask]
            U = U[mask]
            V = V[mask]
            
            # 绘制箭头
            if len(X) > 0:
                ax.quiver(X, Y, U, V, 
                          color='red', scale=20, scale_units='inches', 
                          angles='xy', width=0.002, headwidth=3)
            
            ax.set_title("Flow Field")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            
            # 转换为QPixmap
            canvas = FigureCanvasAgg(fig)
            canvas.draw()
            buf = canvas.buffer_rgba()
            image = np.asarray(buf)
            
            # 转换为QImage
            height, width, channel = image.shape
            bytes_per_line = 4 * width
            q_image = QtGui.QImage(image.data, width, height, bytes_per_line, QtGui.QImage.Format_RGBA8888)
            
            # 转换为QPixmap
            pixmap = QtGui.QPixmap.fromImage(q_image)
            
            # 发送结果
            self.visualization_ready.emit(pixmap)
            
            # 清理资源
            plt.close(fig)
            
        except Exception as e:
            print(f"流场可视化错误: {e}")
    
    def stop(self):
        self.active = False
        self.wait()