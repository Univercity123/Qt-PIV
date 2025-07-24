from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QThread, pyqtSignal, pyqtSlot, QTimer, QMutex, QMutexLocker
from event_camera_thread import EventCameraThread
from flow_visualizer import FlowVisualizer
from Ui_PIV import Ui_MainWindow
import cv2
import os
import sys
import numpy as np
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        
        # 初始化事件相机线程
        self.camera_thread = EventCameraThread(
            input_event_file=None,  # 可以传入事件文件路径
            camera_serial=None,    # 可以传入相机序列号
            slicing_mode='N_US',   # 切片模式
            delta_ts=10000,        # 切片时间（微秒）
            delta_n_events=10000   # 切片事件数
        )
        
        self.camera_index = 0
        self.zoom_factor = 1.0
        self.fps = 0.0
        self.is_fullscreen = False
        
        # 流场相关变量
        self.flow_u = None
        self.flow_v = None
        self.flow_visualizer = None
        self.last_flow_time = 0
        self.flow_min_interval = 0.3
        
        # 连接信号和槽
        self.setup_camera_signals()
        self.connect_actions()
        
        # 初始化显示
        self.clear_display("事件相机未连接")
        self.clear_flow_display("流场图未生成")
        
        # 设置初始状态
        self.ui.statusbar.showMessage("就绪")
        self.update_camera_buttons_state(False)
        self.ui.label_5.installEventFilter(self)
        
        # 设置Matplotlib使用英文字体
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica', 'sans-serif']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 设置流场标签的尺寸策略
        self.ui.label_6.setSizePolicy(
            QtWidgets.QSizePolicy.Preferred,
            QtWidgets.QSizePolicy.Preferred
        )
        self.ui.label_6.setMinimumSize(100, 100)
        self.ui.label_6.setMaximumSize(1920, 1080)
        
        # 设置摄像头标签的尺寸策略
        self.ui.label_5.setSizePolicy(
            QtWidgets.QSizePolicy.Preferred,
            QtWidgets.QSizePolicy.Preferred
        )
        self.ui.label_5.setMinimumSize(100, 100)
        self.ui.label_5.setMaximumSize(1920, 1080)

        # 保存最后一帧
        self.last_camera_frame = None
        self.last_flow_pixmap = None
        self.resizing = False


    def fix_qt_plugin_issues(self):
        """修复Qt平台插件加载问题"""
        # 清除可能冲突的环境变量
        if "QT_QPA_PLATFORM_PLUGIN_PATH" in os.environ:
            del os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"]
        
        # 设置正确的平台插件
        os.environ["QT_QPA_PLATFORM"] = "xcb"
    
        # 添加系统Qt插件路径
        possible_plugin_paths = [
            "/usr/lib/x86_64-linux-gnu/qt5/plugins",
            "/usr/lib/qt/plugins",
            "/usr/local/lib/qt/plugins",
            os.path.join(sys.prefix, "plugins"),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "qt_plugins")
        ]
    
        # 查找有效的插件路径
        valid_plugin_path = None
        for path in possible_plugin_paths:
            xcb_plugin = os.path.join(path, "platforms", "libqxcb.so")
            if os.path.exists(xcb_plugin):
                valid_plugin_path = path
                break
    
        if valid_plugin_path:
            os.environ["QT_PLUGIN_PATH"] = valid_plugin_path
        else:
            print("警告: 未找到有效的Qt平台插件路径")
        
        # 调试信息
        print(f"设置 QT_PLUGIN_PATH = {os.environ.get('QT_PLUGIN_PATH', '未设置')}")
        print(f"设置 QT_QPA_PLATFORM = {os.environ.get('QT_QPA_PLATFORM', '未设置')}")

    def setup_camera_signals(self):
        """连接相机信号"""
        self.camera_thread.change_pixmap_signal.connect(self.update_display)
        self.camera_thread.camera_status_signal.connect(self.update_status)
        self.camera_thread.connection_status_signal.connect(self.update_camera_buttons_state)
        self.camera_thread.fps_signal.connect(self.update_fps)
        self.camera_thread.flow_field_signal.connect(self.update_real_flow_field)
        self.camera_thread.finished.connect(self.on_camera_finished)
        self.camera_thread.error_occurred.connect(self.on_camera_error)

    def connect_worker_signals(self):
        """当worker创建后连接信号"""
        if self.camera_thread.worker:
            worker = self.camera_thread.worker
            worker.change_pixmap_signal.connect(self.update_display)
            worker.camera_status_signal.connect(self.update_status)
            worker.connection_status_signal.connect(self.update_camera_buttons_state)
            worker.fps_signal.connect(self.update_fps)
            worker.flow_field_signal.connect(self.update_real_flow_field)
            worker.finished.connect(self.on_camera_finished)
    
    def on_camera_finished(self):
        """当相机线程完成时更新状态"""
        self.update_camera_buttons_state(False)
        self.clear_display("事件相机已关闭")
        self.clear_flow_display("流场图未生成")
        self.ui.statusbar.showMessage("事件相机已关闭")

    def on_camera_error(self, error_msg):
        """处理相机错误"""
        self.ui.statusbar.showMessage(error_msg)
        self.clear_display("相机错误")
        self.clear_flow_display("流场计算失败")
        self.update_camera_buttons_state(False)
        
        # 尝试恢复
        QtCore.QTimer.singleShot(2000, lambda: self.ui.statusbar.showMessage("尝试重新连接..."))
        QtCore.QTimer.singleShot(3000, self.link_camera)

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
        """连接事件相机"""
        if not self.camera_thread.is_connected():
            self.ui.statusbar.showMessage("正在连接事件相机...")
            # 启动线程
            if not self.camera_thread.isRunning():
                self.camera_thread.start()
        else:
            self.ui.statusbar.showMessage("事件相机已连接")

    def open_camera(self):
        """打开事件相机"""
        if not self.camera_thread.is_connected():
            self.ui.statusbar.showMessage("请先连接事件相机")
            return
            
        if not self.camera_thread.isRunning():
            # 启动线程
            self.camera_thread.start()
            self.ui.statusbar.showMessage("正在打开事件相机...")
        else:
            self.ui.statusbar.showMessage("事件相机已打开")
        
        # 清除流场显示
        self.clear_flow_display("计算流场中...")

    def close_camera(self):
        """关闭事件相机"""
        if self.camera_thread.isRunning():
            # 停止线程
            self.camera_thread.stop()
            
            # 清除画面
            self.clear_display("事件相机已关闭")
            self.clear_flow_display("流场图未生成")
            self.ui.statusbar.showMessage("事件相机已关闭")
            
            # 清除流场数据
            self.flow_u = None
            self.flow_v = None
            
            # 停止流场可视化线程
            if self.flow_visualizer is not None:
                self.flow_visualizer.stop()
                self.flow_visualizer = None
        else:
            self.ui.statusbar.showMessage("事件相机未打开")
            self.clear_display("事件相机未打开")
            self.clear_flow_display("流场图未生成")
            
        self.update_camera_buttons_state(self.camera_thread.is_connected())
        
        if self.is_fullscreen:
            self.exit_fullscreen()

        # 清除保存的帧
        self.last_camera_frame = None
        self.last_flow_pixmap = None


    def set_camera(self):
        """设置摄像头"""
        if not self.camera_thread.connected:
            self.ui.statusbar.showMessage("请先连接摄像头")
            return
            
        self.camera_thread.set_camera_settings()
        self.ui.statusbar.showMessage("正在设置摄像头...")

    def update_camera_buttons_state(self, connected):
        """根据相机状态更新按钮状态"""
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