import sys
import os
import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QThread, pyqtSignal

# 解决 Qt 插件问题
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = ""
os.environ["QT_QPA_PLATFORM"] = "xcb"

class CameraThread(QThread):
    change_pixmap_signal = pyqtSignal(object)

    def __init__(self, camera_index=0):
        super().__init__()
        self.camera_index = camera_index
        self.running = False
        self.cap = None

    def run(self):
        self.running = True
        self.cap = cv2.VideoCapture(self.camera_index)

        # 检查摄像头是否成功打开
        if not self.cap.isOpened():
            print(f"无法打开摄像头 {self.camera_index}")
            self.change_pixmap_signal.emit(None)  # 发送空信号表示失败
            return

        while self.running:
            ret, frame = self.cap.read()
            if ret:
                self.change_pixmap_signal.emit(frame)
            else:
                break
        self.cap.release()

    def stop(self):
        self.running = False
        self.wait()

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        
        # 使用垂直布局管理控件
        self.verticalLayout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout.setObjectName("verticalLayout")
        
        # 添加一个容器用于居中显示图像
        self.imageContainer = QtWidgets.QWidget(self.centralwidget)
        self.imageContainer.setObjectName("imageContainer")
        self.imageLayout = QtWidgets.QVBoxLayout(self.imageContainer)
        self.imageLayout.setContentsMargins(0, 0, 0, 0)
        self.imageLayout.setObjectName("imageLayout")
        
        self.label = QtWidgets.QLabel(self.imageContainer)
        self.label.setMinimumSize(QtCore.QSize(640, 480))
        self.label.setAlignment(QtCore.Qt.AlignCenter)  # 设置内容居中
        self.label.setStyleSheet("""
            background-color: rgb(0, 0, 0);
            color: white;
            font-size: 16px;
        """)
        self.label.setText("摄像头未启动")
        self.label.setObjectName("label")
        self.imageLayout.addWidget(self.label)
        
        self.verticalLayout.addWidget(self.imageContainer, 1)  # 添加图像容器并设置拉伸因子为1
        
        # 底部控制区域
        self.controlWidget = QtWidgets.QWidget(self.centralwidget)
        self.controlWidget.setObjectName("controlWidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.controlWidget)
        self.horizontalLayout.setObjectName("horizontalLayout")
        
        self.pushButton = QtWidgets.QPushButton(self.controlWidget)
        self.pushButton.setMinimumSize(QtCore.QSize(100, 30))
        self.pushButton.setObjectName("pushButton")
        self.horizontalLayout.addWidget(self.pushButton)
        
        self.comboBox = QtWidgets.QComboBox(self.controlWidget)
        self.comboBox.setMinimumSize(QtCore.QSize(150, 30))
        self.comboBox.setObjectName("comboBox")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.horizontalLayout.addWidget(self.comboBox)
        
        self.pushButton_2 = QtWidgets.QPushButton(self.controlWidget)
        self.pushButton_2.setMinimumSize(QtCore.QSize(100, 30))
        self.pushButton_2.setObjectName("pushButton_2")
        self.horizontalLayout.addWidget(self.pushButton_2)
        
        self.verticalLayout.addWidget(self.controlWidget)
        
        MainWindow.setCentralWidget(self.centralwidget)
        
        # 菜单栏
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 28))
        self.menubar.setObjectName("menubar")
        self.menu = QtWidgets.QMenu(self.menubar)
        self.menu.setObjectName("menu")
        MainWindow.setMenuBar(self.menubar)
        
        # 状态栏
        self.statusBar = QtWidgets.QStatusBar(MainWindow)
        self.statusBar.setObjectName("statusBar")
        MainWindow.setStatusBar(self.statusBar)
        
        # 工具栏
        self.toolBar = QtWidgets.QToolBar(MainWindow)
        self.toolBar.setObjectName("toolBar")
        MainWindow.addToolBar(QtCore.Qt.TopToolBarArea, self.toolBar)
        
        self.menubar.addAction(self.menu.menuAction())
        self.toolBar.addSeparator()

        self.camera_thread = None
        self.pushButton.clicked.connect(self.start_camera)
        self.pushButton_2.clicked.connect(self.stop_camera)
        self.comboBox.currentIndexChanged.connect(self.change_camera)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "摄像头应用"))
        self.pushButton.setText(_translate("MainWindow", "打开摄像头"))
        self.pushButton_2.setText(_translate("MainWindow", "关闭摄像头"))
        self.comboBox.setItemText(0, _translate("MainWindow", "内置摄像头"))
        self.comboBox.setItemText(1, _translate("MainWindow", "外接摄像头"))
        self.menu.setTitle(_translate("MainWindow", "文件"))
        self.toolBar.setWindowTitle(_translate("MainWindow", "工具栏"))

    def start_camera(self):
        # 确保之前的线程已停止
        if self.camera_thread is not None:
            self.camera_thread.stop()
            self.camera_thread.wait()
            self.camera_thread = None
            
        # 创建新线程
        camera_index = self.comboBox.currentIndex()
        self.camera_thread = CameraThread(camera_index)
        
        # 连接信号和槽
        self.camera_thread.change_pixmap_signal.connect(self.update_image)
        
        # 启动线程
        self.camera_thread.start()
        
        # 更新状态
        self.clear_camera_view("正在启动摄像头...")

    def stop_camera(self):
        if self.camera_thread is not None:
            # 停止线程
            self.camera_thread.stop()
            self.camera_thread.wait()
            self.camera_thread = None
            
            # 立即清除画面
            self.clear_camera_view("摄像头已关闭")
        else:
            # 即使没有运行中的摄像头线程，也清除画面
            self.clear_camera_view("摄像头未启动")
    
    def clear_camera_view(self, message="摄像头未启动"):
        """清除摄像头画面并显示提示信息"""
        # 创建黑色背景
        background = QtGui.QPixmap(self.label.width(), self.label.height())
        background.fill(QtGui.QColor(0, 0, 0))  # 黑色背景
        
        # 创建并配置文本
        painter = QtGui.QPainter(background)
        painter.setPen(QtGui.QColor(255, 255, 255))  # 白色文字
        painter.setFont(QtGui.QFont("Arial", 16))
        
        # 居中绘制文本
        text_rect = painter.fontMetrics().boundingRect(message)
        x = (background.width() - text_rect.width()) // 2
        y = (background.height() + text_rect.height()) // 2
        painter.drawText(x, y, message)
        painter.end()
        
        # 设置label的pixmap
        self.label.setPixmap(background)

    def change_camera(self):
        if self.camera_thread is not None and self.camera_thread.isRunning():
            self.start_camera()

    def update_image(self, frame):
        """在主线程中更新图像"""
        if frame is None:
            self.clear_camera_view("无法打开摄像头，请检查设备连接！")
            return

        # 确保接收到的是 numpy 数组
        if not isinstance(frame, np.ndarray):
            self.clear_camera_view("接收到无效的图像数据")
            return

        try:
            # 将BGR图像转换为RGB
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 获取图像的尺寸
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            
            # 创建QImage
            qt_image = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
            
            # 创建QPixmap
            pixmap = QtGui.QPixmap.fromImage(qt_image)
            
            # 获取label的尺寸
            label_width = self.label.width()
            label_height = self.label.height()
            
            # 计算缩放比例并保持宽高比
            scaled_pixmap = pixmap.scaled(
                label_width, 
                label_height, 
                QtCore.Qt.KeepAspectRatio, 
                QtCore.Qt.SmoothTransformation
            )
            
            # 创建新的QPixmap作为背景，大小与label相同
            background = QtGui.QPixmap(label_width, label_height)
            background.fill(QtGui.QColor(0, 0, 0))  # 黑色背景
            
            # 计算居中位置
            x = (label_width - scaled_pixmap.width()) // 2
            y = (label_height - scaled_pixmap.height()) // 2
            
            # 在背景上绘制居中的图像
            painter = QtGui.QPainter(background)
            painter.drawPixmap(x, y, scaled_pixmap)
            painter.end()
            
            # 设置label的pixmap
            self.label.setPixmap(background)
            
        except Exception as e:
            print(f"图像处理错误: {e}")
            self.clear_camera_view("图像处理错误")

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())