import cv2
import numpy as np
import time
from PyQt5.QtCore import QThread, pyqtSignal, QMutex, QMutexLocker

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