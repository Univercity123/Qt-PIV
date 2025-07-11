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
from PyQt5.QtCore import QThread, pyqtSignal, pyqtSlot, QTimer
from Ui_PIV import Ui_MainWindow  # 导入上传文件中的UI类

# 解决 Qt 插件问题
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = ""
os.environ["QT_QPA_PLATFORM"] = "xcb"

class CameraThread(QThread):
    change_pixmap_signal = pyqtSignal(object)
    camera_status_signal = pyqtSignal(str)
    connection_status_signal = pyqtSignal(bool)
    fps_signal = pyqtSignal(float)  # 新增：帧率信号

    def __init__(self, camera_index=0):
        super().__init__()
        self.camera_index = camera_index
        self.running = False
        self.cap = None
        self.connected = False
        self.opened = False
        self.last_frame = None
        self.frame_count = 0
        self.start_time = 0
        self.fps = 0.0

    def run(self):
        self.running = True
        
        # 检查摄像头是否已连接
        if not self.connected:
            self.camera_status_signal.emit("摄像头未连接")
            return
            
        # 尝试打开摄像头
        try:
            self.cap = cv2.VideoCapture(self.camera_index)
            if not self.cap.isOpened():
                self.camera_status_signal.emit("无法打开摄像头")
                return
                
            self.opened = True
            self.camera_status_signal.emit("摄像头已打开")
            
            # 初始化帧率计算
            self.frame_count = 0
            self.start_time = time.time()
            
            while self.running:
                ret, frame = self.cap.read()
                if ret:
                    self.last_frame = frame
                    self.frame_count += 1
                    
                    # 计算帧率
                    current_time = time.time()
                    elapsed_time = current_time - self.start_time
                    if elapsed_time > 1.0:  # 每秒更新一次帧率
                        self.fps = self.frame_count / elapsed_time
                        self.fps_signal.emit(self.fps)
                        self.frame_count = 0
                        self.start_time = current_time
                    
                    self.change_pixmap_signal.emit(frame)
                else:
                    break
                    
            self.cap.release()
            self.opened = False
            self.camera_status_signal.emit("摄像头已关闭")
            
        except Exception as e:
            self.camera_status_signal.emit(f"摄像头错误: {str(e)}")

    def stop(self):
        self.running = False
        self.wait()

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

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        
        # 初始化摄像头线程
        self.camera_thread = CameraThread()
        self.camera_index = 0
        self.zoom_factor = 1.0
        self.fps = 0.0  # 当前帧率
        self.is_fullscreen = False  # 全屏状态标志
        
        # 流场动画相关变量
        self.flow_time = 0.0  # 流场时间变量
        self.flow_timer = QTimer(self)  # 用于更新流场的定时器
        self.flow_timer.timeout.connect(self.update_flow_field)
        self.flow_update_interval = 100  # 流场更新间隔(毫秒)
        
        # 连接信号和槽
        self.connect_actions()
        self.setup_camera_signals()
        
        # 初始化显示
        self.clear_display("摄像头未连接")
        self.clear_flow_display("流场图未生成")
        
        # 设置初始状态
        self.ui.statusbar.showMessage("就绪")
        
        # 初始禁用相机相关按钮
        self.update_camera_buttons_state(False)
        
        # 安装事件过滤器用于全屏功能
        self.ui.label_5.installEventFilter(self)

    def eventFilter(self, source, event):
        """事件过滤器，用于检测鼠标双击事件"""
        if source == self.ui.label_5 and event.type() == QtCore.QEvent.MouseButtonDblClick:
            self.toggle_fullscreen()
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
        self.ui.actionNew_Experiment_16.triggered.connect(self.toggle_fullscreen)  # 新增：全屏切换
        
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
        self.camera_thread.fps_signal.connect(self.update_fps)  # 连接帧率信号

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
            
        self.camera_thread.start()
        self.ui.statusbar.showMessage("正在打开摄像头...")
        
        # 在label6上显示流场箭头图
        self.generate_flow_field()
        
        # 启动流场更新定时器
        self.flow_timer.start(self.flow_update_interval)

    def set_camera(self):
        """设置摄像头"""
        if not self.camera_thread.connected:
            self.ui.statusbar.showMessage("请先连接摄像头")
            return
            
        self.camera_thread.set_camera_settings()
        self.ui.statusbar.showMessage("正在设置摄像头...")

    def close_camera(self):
        """关闭摄像头 - 关键功能实现点"""
        if self.camera_thread.isRunning():
            # 停止线程
            self.camera_thread.stop()
            # 停止流场更新定时器
            self.flow_timer.stop()
            # 立即清除画面
            self.clear_display("摄像头已关闭")
            self.clear_flow_display("流场图未生成")
            self.ui.statusbar.showMessage("摄像头已关闭")
        else:
            self.ui.statusbar.showMessage("摄像头未打开")
            self.clear_display("摄像头未打开")
            self.clear_flow_display("流场图未生成")

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
        """更新显示画面"""
        if frame is None:
            return
            
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
            
            # 应用缩放 - 确保使用整数尺寸
            new_width = int(pixmap.width() * self.zoom_factor)
            new_height = int(pixmap.height() * self.zoom_factor)
            
            # 避免尺寸为0
            if new_width <= 0:
                new_width = 1
            if new_height <= 0:
                new_height = 1
                
            scaled_pixmap = pixmap.scaled(
                new_width, 
                new_height,
                QtCore.Qt.KeepAspectRatio,
                QtCore.Qt.SmoothTransformation
            )
            
            # 设置label的pixmap
            self.ui.label_5.setPixmap(scaled_pixmap)
            
        except Exception as e:
            print(f"图像处理错误: {e}")
            self.ui.statusbar.showMessage(f"图像处理错误: {e}")

    def clear_display(self, message="摄像头未连接"):
        """清除显示画面并显示消息 - 关键功能实现点"""
        # 获取label的尺寸
        label_width = max(1, self.ui.label_5.width())
        label_height = max(1, self.ui.label_5.height())
        
        # 创建黑色背景
        background = QtGui.QPixmap(label_width, label_height)
        background.fill(QtGui.QColor(0, 0, 0))  # 黑色背景
        
        # 创建并配置文本
        painter = QtGui.QPainter(background)
        painter.setPen(QtGui.QColor(255, 255, 255))  # 白色文字
        painter.setFont(QtGui.QFont("Arial", 16))
        
        # 居中绘制文本
        text_rect = painter.fontMetrics().boundingRect(message)
        x = (label_width - text_rect.width()) // 2
        y = (label_height + text_rect.height()) // 2
        painter.drawText(x, y, message)
        painter.end()
        
        # 设置label的pixmap
        self.ui.label_5.setPixmap(background)
        
    def clear_flow_display(self, message="流场图未生成"):
        """清除流场显示画面并显示消息"""
        # 获取label的尺寸
        label_width = max(1, self.ui.label_6.width())
        label_height = max(1, self.ui.label_6.height())
        
        # 创建黑色背景
        background = QtGui.QPixmap(label_width, label_height)
        background.fill(QtGui.QColor(0, 0, 0))  # 黑色背景
        
        # 创建并配置文本
        painter = QtGui.QPainter(background)
        painter.setPen(QtGui.QColor(255, 255, 255))  # 白色文字
        painter.setFont(QtGui.QFont("Arial", 16))
        
        # 居中绘制文本
        text_rect = painter.fontMetrics().boundingRect(message)
        x = (label_width - text_rect.width()) // 2
        y = (label_height + text_rect.height()) // 2
        painter.drawText(x, y, message)
        painter.end()
        
        # 设置label的pixmap
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
        """切换全屏模式"""
        if self.is_fullscreen:
            # 退出全屏
            self.ui.label_5.setWindowFlags(QtCore.Qt.Widget)
            self.ui.label_5.showNormal()
            self.show()
            self.is_fullscreen = False
        else:
            # 进入全屏
            self.ui.label_5.setWindowFlags(
                QtCore.Qt.Window | 
                QtCore.Qt.CustomizeWindowHint | 
                QtCore.Qt.FramelessWindowHint |
                QtCore.Qt.WindowStaysOnTopHint
            )
            self.ui.label_5.showFullScreen()
            self.is_fullscreen = True

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
        # 捕获时重新生成流场图
        self.generate_flow_field()

    def grab_stop(self):
        """停止抓取"""
        self.ui.statusbar.showMessage("已停止抓取图像")

    def delete_setup(self):
        """删除设置"""
        self.ui.statusbar.showMessage("设置已删除")

    def save_setup(self):
        """保存设置"""
        self.ui.statusbar.showMessage("设置已保存")
        
    def plot_flow_field(self, u, v, output_file=None, magFlag=True, slFlag=True, step=1, quiver_step=16):
        """
        可视化速度场：
        - 背景：速度大小的热力图
        - 前景：流线图 + 稀疏箭头图（quiver）

        参数：
        - u, v: 速度场的两个分量，形状 [H, W]
        - step: 背景图采样间隔（默认=1，表示全分辨率）
        - quiver_step: 箭头稀疏采样间隔（默认=16）
        """
        H, W = u.shape
        x = np.arange(0, H)
        y = np.arange(0, W)
        xx, yy = np.meshgrid(x, y, indexing='ij')

        # 速度大小
        magnitude = np.sqrt(u**2 + v**2)

        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111)
        
        # 背景热力图（速度大小）
        if magFlag:
            im = ax.imshow(magnitude[::step, ::step], cmap='viridis', origin='lower')
            plt.colorbar(im, ax=ax, label='Speed Magnitude')
        else:
            im = ax.imshow(0*magnitude[::step, ::step], cmap='viridis', origin='lower')
            plt.colorbar(im, ax=ax, label='')

        # 流线图（streamplot）
        if slFlag:
            ax.streamplot(yy, xx, v, u, color='white', linewidth=0.8, density=1.0)

        # 稀疏箭头图（quiver）
        ax.quiver(yy[::quiver_step, ::quiver_step],
                  xx[::quiver_step, ::quiver_step],
                  v[::quiver_step, ::quiver_step],
                  -u[::quiver_step, ::quiver_step],
                  color='red', scale=None)

        ax.set_title("Vector Field Visualization")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_xlim([-0.1, W+0.1])
        ax.set_ylim([-0.1, H+0.1])
        ax.invert_yaxis()
        plt.tight_layout()
        
        return fig

    def generate_flow_field(self):
        """生成流场箭头图并显示在label6上"""
        try:
            # 创建随时间变化的流场
            size = 64  # 流场大小
            x = np.linspace(-2, 2, size)
            y = np.linspace(-2, 2, size)
            X, Y = np.meshgrid(x, y)
            
            # 创建一个随时间变化的涡旋流场
            # 涡旋中心位置随时间变化
            center_x = 0.5 * np.sin(self.flow_time)
            center_y = 0.5 * np.cos(self.flow_time)
            
            # 涡旋强度随时间变化
            strength = 1.0 + 0.3 * np.sin(self.flow_time * 2)
            
            # 计算到涡旋中心的距离
            R = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
            
            # 创建涡旋流场
            theta = np.arctan2(Y - center_y, X - center_x)
            u = -strength * np.sin(theta) / (R + 0.1)  # 避免除以零
            v = strength * np.cos(theta) / (R + 0.1)  # 避免除以零
            
            # 添加全局流动
            u += 0.3 * np.sin(self.flow_time * 0.5)
            v += 0.2 * np.cos(self.flow_time * 0.3)
            
            # 添加一些随机扰动
            u += np.random.randn(size, size) * 0.1
            v += np.random.randn(size, size) * 0.1
            
            # 使用plot_flow_field函数绘制流场图
            fig = self.plot_flow_field(u, v, 
                                      magFlag=True, 
                                      slFlag=True,
                                      step=1,
                                      quiver_step=8)
            
            # 将图形转换为QPixmap
            canvas = FigureCanvasAgg(fig)
            canvas.draw()
            
            # 获取图像数据
            buf = canvas.buffer_rgba()
            image = np.asarray(buf)
            
            # 转换为QImage
            height, width, channel = image.shape
            bytes_per_line = 4 * width
            q_image = QtGui.QImage(image.data, width, height, bytes_per_line, QtGui.QImage.Format_RGBA8888)
            
            # 转换为QPixmap并显示
            pixmap = QtGui.QPixmap.fromImage(q_image)
            self.ui.label_6.setPixmap(pixmap)
            
            # 清理资源
            plt.close(fig)
            
        except Exception as e:
            print(f"生成流场图错误: {e}")
            self.ui.statusbar.showMessage(f"生成流场图错误: {e}")

    def update_flow_field(self):
        """更新流场图 - 随时间变化"""
        # 更新时间变量
        self.flow_time += 0.1
        
        # 更新流场图
        self.generate_flow_field()
        
        # 在状态栏显示流场更新时间
        self.ui.statusbar.showMessage(f"流场图更新时间: {self.flow_time:.1f}s")

    def closeEvent(self, event):
        """窗口关闭时停止摄像头线程"""
        if self.camera_thread.isRunning():
            self.camera_thread.stop()
            # 清除画面
            self.clear_display("应用程序已关闭")
            self.clear_flow_display("流场图未生成")
        # 停止流场更新定时器
        if self.flow_timer.isActive():
            self.flow_timer.stop()
        # 如果全屏，退出全屏
        if self.is_fullscreen:
            self.toggle_fullscreen()
        event.accept()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    
    # 设置应用程序样式
    app.setStyle("Fusion")
    
    # 创建主窗口
    mainWindow = MainWindow()
    mainWindow.setWindowTitle("PIV - 粒子图像测速系统")
    mainWindow.show()
    
    # 启动应用程序
    sys.exit(app.exec_())