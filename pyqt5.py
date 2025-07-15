import sys
import os
import cv2
import numpy as np
import time
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QThread, pyqtSignal, pyqtSlot, QTimer, QMutex, QMutexLocker
from Ui_PIV import Ui_MainWindow  # 导入上传文件中的UI类

# 解决 Qt 插件问题
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = ""
os.environ["QT_QPA_PLATFORM"] = "xcb"

class CameraThread(QThread):
    change_pixmap_signal = pyqtSignal(object)
    camera_status_signal = pyqtSignal(str)
    connection_status_signal = pyqtSignal(bool)
    fps_signal = pyqtSignal(float)
    flow_field_signal = pyqtSignal(np.ndarray, np.ndarray)

    def __init__(self, camera_index=0):
        super().__init__()
        self.camera_index = camera_index
        self.running = False
        self.cap = None
        self.connected = False
        self.opened = False
        self.last_frame = None
        self.prev_frame = None
        self.prev_gray = None
        self.frame_count = 0
        self.start_time = 0
        self.fps = 0.0
        self.mutex = QMutex()  # 添加互斥锁
        self.flow_enabled = True  # 控制是否计算光流
        self.flow_counter = 0  # 控制光流计算频率
        self.flow_interval = 3  # 每3帧计算一次光流

    def run(self):
        self.running = True
        
        if not self.connected:
            self.camera_status_signal.emit("摄像头未连接")
            return
            
        try:
            self.cap = cv2.VideoCapture(self.camera_index)
            if not self.cap.isOpened():
                self.camera_status_signal.emit("无法打开摄像头")
                return
                
            self.opened = True
            self.camera_status_signal.emit("摄像头已打开")
            
            self.frame_count = 0
            self.start_time = time.time()
            self.prev_frame = None
            self.prev_gray = None
            
            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    break
                    
                with QMutexLocker(self.mutex):  # 使用互斥锁保护共享资源
                    self.last_frame = frame
                    self.frame_count += 1
                    
                    # 计算帧率
                    current_time = time.time()
                    elapsed_time = current_time - self.start_time
                    if elapsed_time > 1.0:
                        self.fps = self.frame_count / elapsed_time
                        self.fps_signal.emit(self.fps)
                        self.frame_count = 0
                        self.start_time = current_time
                    
                    # 发送当前帧
                    self.change_pixmap_signal.emit(frame)
                    
                    # 计算光流（每flow_interval帧计算一次）
                    if self.flow_enabled and self.prev_frame is not None:
                        self.flow_counter += 1
                        if self.flow_counter % self.flow_interval == 0:
                            # 计算光流
                            u, v = self.calculate_optical_flow(self.prev_frame, frame)
                            if u is not None and v is not None:
                                self.flow_field_signal.emit(u, v)
                    
                    # 更新前一帧
                    self.prev_frame = frame.copy()
                    self.prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    
            self.cap.release()
            self.opened = False
            self.camera_status_signal.emit("摄像头已关闭")
            
        except Exception as e:
            self.camera_status_signal.emit(f"摄像头错误: {str(e)}")

    def stop(self):
        self.running = False
        self.wait(500)  # 等待最多500毫秒

    def calculate_optical_flow(self, prev_frame, current_frame):
        """计算两帧之间的光流（优化版）"""
        try:
            # 如果已有灰度图，直接使用
            if self.prev_gray is None:
                prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            else:
                prev_gray = self.prev_gray
                
            current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
            
            # 缩小图像以提高性能
            scale = 0.5  # 缩小到一半大小
            small_prev = cv2.resize(prev_gray, (0, 0), fx=scale, fy=scale)
            small_current = cv2.resize(current_gray, (0, 0), fx=scale, fy=scale)
            
            # 使用Farneback方法计算稠密光流
            flow = cv2.calcOpticalFlowFarneback(
                small_prev, 
                small_current, 
                None,
                0.5,   # pyr_scale
                2,     # levels (减少层数)
                15,    # winsize
                2,     # iterations (减少迭代次数)
                5,     # poly_n
                1.1,   # poly_sigma
                0      # flags
            )
            
            # 分离u和v分量
            u = flow[..., 0]
            v = flow[..., 1]
            
            # 放大回原始尺寸
            u = cv2.resize(u, (prev_gray.shape[1], prev_gray.shape[0]), interpolation=cv2.INTER_LINEAR)
            v = cv2.resize(v, (prev_gray.shape[1], prev_gray.shape[0]), interpolation=cv2.INTER_LINEAR)
            
            # 调整比例
            u /= scale
            v /= scale
            
            return u, v
            
        except Exception as e:
            print(f"光流计算错误: {e}")
            return None, None

    def connect_camera(self):
        """模拟连接摄像头的过程"""
        self.connected = True
        self.connection_status_signal.emit(True)
        self.camera_status_signal.emit("摄像头已连接")

    def disconnect_camera(self):
        """断开摄像头连接"""
        if self.opened:
            self.stop()
        self.connected = False
        self.connection_status_signal.emit(False)
        self.camera_status_signal.emit("摄像头已断开")

    def set_camera_settings(self):    
        """设置摄像头参数"""
        if not self.connected:
                self.camera_status_signal.emit("请先连接摄像头")
        return
             
        # 这里可以添加实际的摄像头设置代码
        self.camera_status_signal.emit("摄像头设置已应用")

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


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        
        # 初始化摄像头线程
        self.camera_thread = CameraThread()
        self.camera_index = 0
        self.zoom_factor = 1.0
        self.fps = 0.0
        self.is_fullscreen = False
        
        # 流场相关变量
        self.flow_u = None
        self.flow_v = None
        self.flow_visualizer = None  # 流场可视化线程
        
        # 性能优化
        self.last_flow_time = 0
        self.flow_min_interval = 0.3  # 流场更新最小间隔（秒）
        
        # 连接信号和槽
        self.connect_actions()
        self.setup_camera_signals()
        
        # 初始化显示
        self.clear_display("摄像头未连接")
        self.clear_flow_display("流场图未生成")
        
        # 设置初始状态
        self.ui.statusbar.showMessage("就绪")
        self.update_camera_buttons_state(False)
        self.ui.label_5.installEventFilter(self)
        
        # 设置Matplotlib使用英文字体
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica', 'sans-serif']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 设置流场标签的尺寸策略（关键修复）
        self.ui.label_6.setSizePolicy(
            QtWidgets.QSizePolicy.Preferred,
            QtWidgets.QSizePolicy.Preferred
        )
        self.ui.label_6.setMinimumSize(100, 100)  # 最小尺寸
        self.ui.label_6.setMaximumSize(1920, 1080)  # 最大尺寸

        # 新增：为摄像头标签设置相同的尺寸策略
        self.ui.label_5.setSizePolicy(
            QtWidgets.QSizePolicy.Preferred,
            QtWidgets.QSizePolicy.Preferred
        )
        self.ui.label_5.setMinimumSize(100, 100)  # 最小尺寸
        self.ui.label_5.setMaximumSize(1920, 1080)  # 最大尺寸

        # 添加以下两行
        self.last_camera_frame = None  # 保存最后一帧摄像头图像
        self.last_flow_pixmap = None   # 保存最后一个流场图像
        
        # 添加一个标志，防止递归调用
        self.resizing = False

    def label5_resized(self, event):
        """label5尺寸变化时触发"""
        if self.last_camera_frame is not None:
            self.update_display(self.last_camera_frame)
        else:
            self.clear_display("摄像头未连接")
    
    def label6_resized(self, event):
        """label6尺寸变化时触发"""
        if self.last_flow_pixmap is not None:
            self.update_flow_display(self.last_flow_pixmap)
        else:
            self.clear_flow_display("流场图未生成")

    def resizeEvent(self, event):
        """窗口大小变化时触发 - 修复无限变大问题"""
        # 防止递归调用
        if self.resizing:
            return
            
        self.resizing = True
        super().resizeEvent(event)
        
        # 更新摄像头显示
        if self.last_camera_frame is not None:
            # 使用定时器延迟更新，避免连续触发
            QtCore.QTimer.singleShot(50, lambda: self.update_display(self.last_camera_frame))
        else:
            self.clear_display("摄像头未连接")
        
        # 更新流场显示
        if self.last_flow_pixmap is not None:
            # 使用定时器延迟更新，避免连续触发
            QtCore.QTimer.singleShot(50, lambda: self.update_flow_display(self.last_flow_pixmap))
        elif self.flow_u is not None and self.flow_v is not None:
            # 如果流场数据存在但图像不存在，重新生成流场图
            self.update_real_flow_field(self.flow_u, self.flow_v)
        else:
            self.clear_flow_display("流场图未生成")
            
        self.resizing = False
    
    def eventFilter(self, source, event):
        """事件过滤器，用于检测鼠标双击事件"""
        # 主窗口标签双击进入全屏
        if source == self.ui.label_5 and event.type() == QtCore.QEvent.MouseButtonDblClick and not self.is_fullscreen:
            self.enter_fullscreen()
            return True
            
        # 全屏标签双击退出全屏
        if self.is_fullscreen and source == self.fullscreen_label and event.type() == QtCore.QEvent.MouseButtonDblClick:
            self.exit_fullscreen()
            return True
            
        return super().eventFilter(source, event)

    def connect_actions(self):
        # 摄像头相关操作
        self.ui.actionNew_Experiment_5.triggered.connect(self.link_camera)
        self.ui.actionNew_Experiment_6.triggered.connect(self.open_camera)
        self.ui.actionNew_Experiment_7.triggered.connect(self.set_camera)
        self.ui.actionNew_Experiment_8.triggered.connect(self.close_camera)
        
        # 显示操作
        self.ui.actionNew_Experiment_9.triggered.connect(self.zoom_out)
        self.ui.actionNew_Experiment_10.triggered.connect(self.zoom_in)
        self.ui.actionNew_Experiment_11.triggered.connect(self.zoom_fit)
        self.ui.actionNew_Experiment_16.triggered.connect(self.toggle_fullscreen)  # 全屏切换
        
        # 图像导航
        self.ui.actionNew_Experiment_12.triggered.connect(self.first_image)
        self.ui.actionNew_Experiment_13.triggered.connect(self.previous_image)
        self.ui.actionNew_Experiment_14.triggered.connect(self.next_image)
        self.ui.actionNew_Experiment_15.triggered.connect(self.last_image)
        
        # 其他按钮
        self.ui.pushButton_3.clicked.connect(self.search_adapters)
        self.ui.pushButton_4.clicked.connect(self.open_adapter)
        self.ui.pushButton_7.clicked.connect(self.laser_on)
        self.ui.pushButton_8.clicked.connect(self.laser_off)
        self.ui.pushButton_9.clicked.connect(self.capture)
        self.ui.pushButton_10.clicked.connect(self.grab_stop)
        self.ui.pushButton_11.clicked.connect(self.delete_setup)
        self.ui.pushButton_12.clicked.connect(self.save_setup)

    def setup_camera_signals(self):
        # 连接摄像头线程的信号
        self.camera_thread.change_pixmap_signal.connect(self.update_display)
        self.camera_thread.camera_status_signal.connect(self.update_status)
        self.camera_thread.connection_status_signal.connect(self.update_camera_buttons_state)
        self.camera_thread.fps_signal.connect(self.update_fps)
        self.camera_thread.flow_field_signal.connect(self.update_real_flow_field)  # 连接流场信号

    def update_real_flow_field(self, u, v):
        """更新真实流场数据（性能优化版）"""
        current_time = time.time()
        
        # 限制流场更新频率
        if current_time - self.last_flow_time < self.flow_min_interval:
            return
            
        self.last_flow_time = current_time
        
        # 确保之前的线程已完全停止
        if self.flow_visualizer is not None:
            self.flow_visualizer.stop()
            self.flow_visualizer.wait(500)  # 等待线程结束
            self.flow_visualizer = None
        
        self.flow_u = u
        self.flow_v = v
        
        # 获取标签的目标尺寸
        target_width = max(100, self.ui.label_6.width())
        target_height = max(100, self.ui.label_6.height())
        
        # 使用目标尺寸创建可视化线程
        self.flow_visualizer = FlowVisualizer(u, v, target_width, target_height)
        self.flow_visualizer.visualization_ready.connect(self.update_flow_display)
        self.flow_visualizer.start()
        
        # 在状态栏显示流场信息
        if u is not None and v is not None:
            avg_u = np.mean(u)
            avg_v = np.mean(v)
            max_speed = np.max(np.sqrt(u**2 + v**2))
            self.ui.statusbar.showMessage(f"流场更新 | 平均速度: U={avg_u:.2f}, V={avg_v:.2f} | 最大速度: {max_speed:.2f}")

    def update_flow_display(self, pixmap):
        """更新流场显示 - 拉伸填充模式"""
        if pixmap and not pixmap.isNull():
            # 保存当前流场图像用于后续调整
            self.last_flow_pixmap = pixmap
            
            # 获取标签的当前尺寸
            label_width = max(1, self.ui.label_6.width())
            label_height = max(1, self.ui.label_6.height())
            
            # 避免尺寸为0
            new_width = max(1, label_width)
            new_height = max(1, label_height)
            
            # 缩放图像 - 拉伸填充整个标签
            scaled_pixmap = pixmap.scaled(
                new_width, 
                new_height,
                QtCore.Qt.IgnoreAspectRatio,  # 忽略宽高比
                QtCore.Qt.SmoothTransformation
            )
            
            # 设置标签的pixmap
            self.ui.label_6.setPixmap(scaled_pixmap)
        else:
            self.clear_flow_display("流场数据无效")

    def link_camera(self):
        """连接摄像头"""
        if not self.camera_thread.connected:
            self.ui.statusbar.showMessage("正在连接摄像头...")
            self.camera_thread.connect_camera()
        else:
            self.ui.statusbar.showMessage("摄像头已连接")

    def open_camera(self):
        """打开摄像头"""
        if not self.camera_thread.connected:
            self.ui.statusbar.showMessage("请先连接摄像头")
            return
            
        if self.camera_thread.isRunning():
            self.camera_thread.stop()
            
        # 重置前一帧
        self.camera_thread.prev_frame = None
        
        self.camera_thread.start()
        self.ui.statusbar.showMessage("正在打开摄像头...")
        
        # 清除流场显示
        self.clear_flow_display("计算流场中...")

    def set_camera(self):
        """设置摄像头"""
        if not self.camera_thread.connected:
            self.ui.statusbar.showMessage("请先连接摄像头")
            return
            
        self.camera_thread.set_camera_settings()
        self.ui.statusbar.showMessage("正在设置摄像头...")

    def update_camera_buttons_state(self, connected):
        """根据摄像头连接状态更新按钮状态"""
        # 连接/断开按钮状态
        self.ui.actionNew_Experiment_5.setEnabled(not connected)  # 连接按钮
        self.ui.actionNew_Experiment_8.setEnabled(connected)      # 断开按钮
        
        # 相机操作按钮状态
        self.ui.actionNew_Experiment_6.setEnabled(connected)  # 打开相机
        self.ui.actionNew_Experiment_7.setEnabled(connected)  # 设置相机
        
        # 其他相关按钮状态
        self.ui.pushButton_4.setEnabled(connected)  # 打开适配器
        self.ui.pushButton_9.setEnabled(connected)   # 捕获按钮

    def update_display(self, frame):
        """更新显示画面 - 拉伸填充模式"""
        if frame is None:
            return
            
        # 保存当前帧用于后续调整
        self.last_camera_frame = frame
        
        # 确保接收到的是 numpy 数组
        if not isinstance(frame, np.ndarray):
            return
            
        try:
            # 将BGR图像转换为RGB
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 在图像左上角添加帧率信息
            if self.fps > 0:
                fps_text = f"FPS: {self.fps:.1f}"
                # 在左上角添加黑色背景的文本
                cv2.putText(rgb_image, fps_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
                # 添加白色文本
                cv2.putText(rgb_image, fps_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            
            # 获取图像的尺寸
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            
            # 创建QImage
            qt_image = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
            
            # 创建QPixmap
            pixmap = QtGui.QPixmap.fromImage(qt_image)
            
            # 获取标签的当前尺寸
            label_width = max(1, self.ui.label_5.width())
            label_height = max(1, self.ui.label_5.height())
            
            # 避免尺寸为0
            new_width = max(1, label_width)
            new_height = max(1, label_height)
            
            # 缩放图像 - 拉伸填充整个标签
            scaled_pixmap = pixmap.scaled(
                new_width, 
                new_height,
                QtCore.Qt.IgnoreAspectRatio,  # 忽略宽高比
                QtCore.Qt.SmoothTransformation
            )
            
            # 设置主窗口的label
            self.ui.label_5.setPixmap(scaled_pixmap)
            
            # 如果全屏窗口存在，更新全屏窗口
            if self.is_fullscreen and self.fullscreen_label:
                # 全屏模式仍然保持宽高比
                self.update_fullscreen_display(pixmap)
            
        except Exception as e:
            print(f"图像处理错误: {e}")
            self.ui.statusbar.showMessage(f"图像处理错误: {e}")


    def clear_display(self, message="摄像头未连接"):
        """清除显示画面并显示消息 - 拉伸填充模式"""
        # 清除保存的帧
        self.last_camera_frame = None
        
        # 获取label的尺寸
        label_width = max(1, self.ui.label_5.width())
        label_height = max(1, self.ui.label_5.height())
        
        # 创建背景
        background = QtGui.QPixmap(label_width, label_height)
        background.fill(QtGui.QColor(0, 0, 0))  # 黑色背景
        
        # 创建并配置文本
        painter = QtGui.QPainter(background)
        painter.setPen(QtGui.QColor(255, 255, 255))  # 白色文字
        
        # 根据标签大小动态调整字体
        font_size = max(10, min(label_height // 15, 24))
        painter.setFont(QtGui.QFont("Arial", font_size))
        
        # 居中绘制文本
        text_rect = painter.fontMetrics().boundingRect(message)
        x = (label_width - text_rect.width()) // 2
        y = (label_height + text_rect.height()) // 2
        painter.drawText(x, y, message)
        painter.end()
        
        # 设置label的pixmap
        self.ui.label_5.setPixmap(background)
        
    def clear_flow_display(self, message="流场图未生成"):
        """清除流场显示画面并显示消息 - 拉伸填充模式"""
        # 清除保存的流场图像
        self.last_flow_pixmap = None
        
        # 获取标签的当前尺寸
        label_width = max(1, self.ui.label_6.width())
        label_height = max(1, self.ui.label_6.height())
        
        # 创建背景
        background = QtGui.QPixmap(label_width, label_height)
        background.fill(QtGui.QColor(0, 0, 0))  # 黑色背景
        
        # 创建并配置文本
        painter = QtGui.QPainter(background)
        painter.setPen(QtGui.QColor(255, 255, 255))  # 白色文字
        
        # 根据标签大小动态调整字体
        font_size = max(10, min(label_height // 15, 24))
        painter.setFont(QtGui.QFont("Arial", font_size))
        
        # 居中绘制文本
        text_rect = painter.fontMetrics().boundingRect(message)
        x = (label_width - text_rect.width()) // 2
        y = (label_height + text_rect.height()) // 2
        painter.drawText(x, y, message)
        painter.end()
        
        # 设置标签的pixmap
        self.ui.label_6.setPixmap(background)

    def update_status(self, message):
        """更新状态信息"""
        self.ui.statusbar.showMessage(message)
        
        # 如果状态是连接成功但未打开，更新显示
        if "已连接" in message and not self.camera_thread.opened:
            self.clear_display("摄像头已连接，请点击'打开相机'")

    def update_fps(self, fps):
        """更新帧率信息"""
        self.fps = fps

    def zoom_out(self):
        """缩小画面"""
        self.zoom_factor = max(0.5, self.zoom_factor * 0.8)
        self.ui.statusbar.showMessage(f"缩放: {self.zoom_factor*100:.0f}%")
        if self.camera_thread.isRunning() and self.camera_thread.last_frame is not None:
            # 强制刷新当前帧
            self.update_display(self.camera_thread.last_frame)

    def zoom_in(self):
        """放大画面"""
        self.zoom_factor = min(2.0, self.zoom_factor * 1.2)
        self.ui.statusbar.showMessage(f"缩放: {self.zoom_factor*100:.0f}%")
        if self.camera_thread.isRunning() and self.camera_thread.last_frame is not None:
            # 强制刷新当前帧
            self.update_display(self.camera_thread.last_frame)

    def zoom_fit(self):
        """自适应画面大小"""
        self.zoom_factor = 1.0
        self.ui.statusbar.showMessage("缩放: 100%")
        if self.camera_thread.isRunning() and self.camera_thread.last_frame is not None:
            # 强制刷新当前帧
            self.update_display(self.camera_thread.last_frame)

    def toggle_fullscreen(self):
        """切换全屏模式 - 由菜单项触发"""
        if self.is_fullscreen:
            self.exit_fullscreen()
        else:
            self.enter_fullscreen()

    def enter_fullscreen(self):
        """进入全屏模式"""
        if self.is_fullscreen:
            return
            
        # 保存当前缩放比例
        self.pre_fullscreen_zoom = self.zoom_factor
        
        # 创建全屏窗口和标签
        self.fullscreen_window = QtWidgets.QWidget()
        self.fullscreen_window.setWindowFlags(
            QtCore.Qt.Window | 
            QtCore.Qt.CustomizeWindowHint | 
            QtCore.Qt.FramelessWindowHint |
            QtCore.Qt.WindowStaysOnTopHint
        )
        self.fullscreen_window.setStyleSheet("background-color: black;")
        
        # 创建全屏布局
        layout = QtWidgets.QVBoxLayout(self.fullscreen_window)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # 创建全屏标签
        self.fullscreen_label = QtWidgets.QLabel()
        self.fullscreen_label.setAlignment(QtCore.Qt.AlignCenter)
        self.fullscreen_label.setStyleSheet("background-color: black;")
        layout.addWidget(self.fullscreen_label)
        
        # 设置全屏窗口内容
        if self.ui.label_5.pixmap() and not self.ui.label_5.pixmap().isNull():
            self.update_fullscreen_display(self.ui.label_5.pixmap())
        else:
            self.fullscreen_label.setText("全屏显示")
            self.fullscreen_label.setStyleSheet("background-color: black; color: white; font-size: 24px;")
        
        # 显示全屏窗口
        self.fullscreen_window.showFullScreen()
        self.is_fullscreen = True
        self.ui.statusbar.showMessage("已进入全屏模式")
        
        # 安装事件过滤器
        self.fullscreen_label.installEventFilter(self)

    def exit_fullscreen(self):
        """退出全屏模式"""
        if not self.is_fullscreen:
            return
            
        # 关闭全屏窗口
        if self.fullscreen_window:
            self.fullscreen_window.close()
            self.fullscreen_window.deleteLater()
            self.fullscreen_window = None
            self.fullscreen_label = None
            
        # 恢复原始状态
        self.is_fullscreen = False
        self.zoom_factor = self.pre_fullscreen_zoom
        
        # 强制刷新主窗口
        self.ui.label_5.show()
        self.ui.centralwidget.update()
        
        # 更新状态
        self.ui.statusbar.showMessage(f"退出全屏，恢复缩放: {self.zoom_factor*100:.0f}%")
        
        # 恢复显示
        if self.camera_thread.isRunning() and self.camera_thread.last_frame is not None:
            self.update_display(self.camera_thread.last_frame)
        else:
            self.update_camera_display_state()

    def update_fullscreen_display(self, pixmap):
        """更新全屏显示"""
        if not self.is_fullscreen or not self.fullscreen_label:
            return
            
        # 获取屏幕尺寸
        screen_size = QtWidgets.QApplication.primaryScreen().size()
        screen_width = screen_size.width()
        screen_height = screen_size.height()
        
        # 计算保持宽高比的缩放比例
        pixmap_size = pixmap.size()
        if pixmap_size.width() > 0 and pixmap_size.height() > 0:
            width_ratio = screen_width / pixmap_size.width()
            height_ratio = screen_height / pixmap_size.height()
            scale_ratio = min(width_ratio, height_ratio)
            
            # 计算新尺寸
            new_width = int(pixmap_size.width() * scale_ratio)
            new_height = int(pixmap_size.height() * scale_ratio)
            
            # 避免尺寸为0
            new_width = max(1, new_width)
            new_height = max(1, new_height)
            
            # 缩放图像
            scaled_pixmap = pixmap.scaled(
                new_width, 
                new_height,
                QtCore.Qt.KeepAspectRatio,
                QtCore.Qt.SmoothTransformation
            )
            
            # 设置全屏标签的pixmap
            self.fullscreen_label.setPixmap(scaled_pixmap)

    def update_camera_display_state(self):
        """根据摄像头状态更新显示"""
        if self.camera_thread.connected:
            if self.camera_thread.opened:
                self.clear_display("摄像头已关闭")
            else:
                self.clear_display("摄像头已连接，请点击'打开相机'")
        else:
            self.clear_display("摄像头未连接")

    def close_camera(self):
        """关闭摄像头 - 关键功能实现点"""
        if self.camera_thread.isRunning():
            # 停止线程
            self.camera_thread.stop()
            
            # 清除画面
            self.clear_display("摄像头已关闭")
            self.clear_flow_display("流场图未生成")
            self.ui.statusbar.showMessage("摄像头已关闭")
            
            # 清除流场数据
            self.flow_u = None
            self.flow_v = None
            
            # 停止流场可视化线程
            if self.flow_visualizer is not None:
                self.flow_visualizer.stop()
                self.flow_visualizer = None
        else:
            self.ui.statusbar.showMessage("摄像头未打开")
            self.clear_display("摄像头未打开")
            self.clear_flow_display("流场图未生成")
            
        self.update_camera_buttons_state(self.camera_thread.connected)
        
        if self.is_fullscreen:
            self.exit_fullscreen()

        # 清除保存的帧
        self.last_camera_frame = None
        self.last_flow_pixmap = None

    def closeEvent(self, event):
        """窗口关闭时停止所有线程"""
        if self.camera_thread.isRunning():
            self.camera_thread.stop()
            
        if self.flow_visualizer is not None:
            self.flow_visualizer.stop()
            
        if self.is_fullscreen:
            self.exit_fullscreen()
            
        # 清除画面
        self.clear_display("应用程序已关闭")
        self.clear_flow_display("流场图未生成")
        event.accept()

    def first_image(self):
        """导航到第一张图像"""
        self.ui.statusbar.showMessage("导航到第一张图像")

    def previous_image(self):
        """导航到上一张图像"""
        self.ui.statusbar.showMessage("导航到上一张图像")

    def next_image(self):
        """导航到下一张图像"""
        self.ui.statusbar.showMessage("导航到下一张图像")

    def last_image(self):
        """导航到最后一张图像"""
        self.ui.statusbar.showMessage("导航到最后张图像")

    def search_adapters(self):
        """搜索适配器"""
        self.ui.statusbar.showMessage("正在搜索适配器...")
        # 模拟搜索过程
        self.ui.comboBox_4.clear()
        self.ui.comboBox_4.addItems(["Adapter 1", "Adapter 2", "Adapter 3"])
        self.ui.statusbar.showMessage("找到3个适配器")

    def open_adapter(self):
        """打开适配器"""
        adapter = self.ui.comboBox_4.currentText()
        if adapter:
            self.ui.statusbar.showMessage(f"已打开适配器: {adapter}")
        else:
            self.ui.statusbar.showMessage("请先选择适配器")

    def laser_on(self):
        """打开激光"""
        self.ui.statusbar.showMessage("激光已打开")

    def laser_off(self):
        """关闭激光"""
        self.ui.statusbar.showMessage("激光已关闭")

    def capture(self):
        """捕获图像"""
        self.ui.statusbar.showMessage("正在捕获图像...")
        # 这里可以添加保存当前帧和流场数据的代码
        if self.camera_thread.isRunning() and self.camera_thread.last_frame is not None:
            # 保存当前帧
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            frame_filename = f"capture_{timestamp}.png"
            cv2.imwrite(frame_filename, self.camera_thread.last_frame)
            
            # 保存流场数据（如果可用）
            if self.flow_u is not None and self.flow_v is not None:
                flow_filename = f"flowfield_{timestamp}.npz"
                np.savez(flow_filename, u=self.flow_u, v=self.flow_v)
                
                # 保存流场图像
                flow_img_filename = f"flowfield_{timestamp}.png"
                if self.ui.label_6.pixmap():
                    self.ui.label_6.pixmap().save(flow_img_filename)
            
            self.ui.statusbar.showMessage(f"已保存捕获: {frame_filename} 和流场数据")
        else:
            self.ui.statusbar.showMessage("没有可捕获的图像")

    def grab_stop(self):
        """停止抓取"""
        self.ui.statusbar.showMessage("已停止抓取图像")

    def delete_setup(self):
        """删除设置"""
        self.ui.statusbar.showMessage("设置已删除")

    def save_setup(self):
        """保存设置"""
        self.ui.statusbar.showMessage("设置已保存")

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    mainWindow = MainWindow()
    mainWindow.setWindowTitle("PIV - 粒子图像测速系统")
    mainWindow.show()
    sys.exit(app.exec_())