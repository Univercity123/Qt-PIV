import numpy as np
import time
import cv2
from PyQt5.QtCore import QThread, pyqtSignal, QMutex, QMutexLocker, QObject
from metavision_sdk_core import BaseFrameGenerationAlgorithm
from metavision_sdk_stream import Camera, CameraStreamSlicer, FileConfigHints, SliceCondition
from metavision_sdk_ui import EventLoop

class EventCameraWorker(QObject):
    # 信号定义
    change_pixmap_signal = pyqtSignal(object)
    camera_status_signal = pyqtSignal(str)
    connection_status_signal = pyqtSignal(bool)
    fps_signal = pyqtSignal(float)
    flow_field_signal = pyqtSignal(np.ndarray, np.ndarray)
    finished = pyqtSignal()
    error_occurred = pyqtSignal(str)

    def __init__(self, input_event_file=None, camera_serial=None, slicing_mode='N_US', delta_ts=10000, delta_n_events=100000):
        super().__init__()
        self.input_event_file = input_event_file
        self.camera_serial = camera_serial
        self.slicing_mode = slicing_mode
        self.delta_ts = delta_ts
        self.delta_n_events = delta_n_events
        
        self.running = False
        self.camera = None
        self.slicer = None
        self.connected = False
        self.opened = False
        self.last_frame = None
        self.prev_frame = None
        self.prev_gray = None
        self.frame_count = 0
        self.start_time = 0
        self.fps = 0.0
        self.mutex = QMutex()
        self.flow_enabled = True
        self.flow_counter = 0
        self.flow_interval = 3
        self.width = 0
        self.height = 0

    def init_camera(self):
        """初始化相机 - 必须在工作线程中调用"""
        try:
            # 连接事件相机
            if self.camera_serial:
                self.camera = Camera.from_serial(self.camera_serial)
            elif self.input_event_file:
                hints = FileConfigHints()
                hints.real_time_playback(True)  # 实时播放
                self.camera = Camera.from_file(self.input_event_file, hints)
            else:
                self.camera = Camera.from_first_available()
                
            if not self.camera:
                self.camera_status_signal.emit("无法连接事件相机")
                return False
                
            self.width = self.camera.width()
            self.height = self.camera.height()
            self.connected = True
            self.connection_status_signal.emit(True)
            self.camera_status_signal.emit(f"事件相机已连接，分辨率: {self.width}x{self.height}")
            
            # 设置切片条件
            if self.slicing_mode == 'N_EVENTS':
                slice_condition = SliceCondition.make_n_events(self.delta_n_events)
            elif self.slicing_mode == 'N_US':
                slice_condition = SliceCondition.make_n_us(self.delta_ts)
            elif self.slicing_mode == 'MIXED':
                slice_condition = SliceCondition.make_mixed(self.delta_ts, self.delta_n_events)
            else:
                slice_condition = SliceCondition.make_n_us(self.delta_ts)
                
            # 创建切片器
            self.slicer = CameraStreamSlicer(self.camera.move(), slice_condition)
            self.opened = True
            self.camera_status_signal.emit("事件相机已打开")
            
            return True
            
        except Exception as e:
            self.error_occurred.emit(f"初始化事件相机错误: {str(e)}")
            return False

    def start_capture(self):
        """在目标线程中启动捕获过程"""
        if self.running:
            return
            
        # 初始化相机
        if not self.init_camera():
            return
            
        self.running = True
        
        try:
            self.frame_count = 0
            self.start_time = time.time()
            self.prev_frame = None
            self.prev_gray = None
            
            # 创建空白帧缓冲区
            frame = np.zeros((self.height, self.width, 3), np.uint8)
            
            while self.running:
                # 获取事件切片
                for slice in self.slicer:
                    if not self.running:
                        break
                    
                    # 生成事件帧
                    BaseFrameGenerationAlgorithm.generate_frame(slice.events, frame)
                    
                    # 转换为BGR格式
                    bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    
                    with QMutexLocker(self.mutex):
                        self.last_frame = bgr_frame.copy()
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
                        self.change_pixmap_signal.emit(self.last_frame)
                        
                        # 计算光流
                        if self.flow_enabled and self.prev_frame is not None:
                            self.flow_counter += 1
                            if self.flow_counter % self.flow_interval == 0:
                                u, v = self.calculate_optical_flow(self.prev_frame, self.last_frame)
                                if u is not None and v is not None:
                                    self.flow_field_signal.emit(u, v)
                        
                        # 更新前一帧
                        self.prev_frame = self.last_frame.copy()
                        self.prev_gray = cv2.cvtColor(self.last_frame, cv2.COLOR_BGR2GRAY)
                    
                    EventLoop.poll_and_dispatch()
                    if not self.running:
                        break
            
            self.opened = False
            self.camera_status_signal.emit("事件相机已关闭")
            
        except Exception as e:
            error_msg = f"事件相机捕获错误: {str(e)}"
            self.error_occurred.emit(error_msg)
        finally:
            self.running = False
            self.cleanup()
            self.finished.emit()

    def cleanup(self):
        """清理资源"""
        if self.slicer:
            self.slicer = None
            
        if self.camera:
            self.camera = None

    def stop_capture(self):
        """停止捕获过程"""
        self.running = False

    def calculate_optical_flow(self, prev_frame, current_frame):
        """计算光流"""
        try:
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
            
            # 缩小图像以提高性能
            scale = 0.5
            small_prev = cv2.resize(prev_gray, (0, 0), fx=scale, fy=scale)
            small_current = cv2.resize(current_gray, (0, 0), fx=scale, fy=scale)
            
            # 使用Farneback方法计算稠密光流
            flow = cv2.calcOpticalFlowFarneback(
                small_prev, small_current, None,
                0.5, 2, 15, 2, 5, 1.1, 0
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

    def is_connected(self):
        """检查相机是否已连接"""
        return self.connected and self.camera is not None

    def is_running(self):
        """检查捕获是否正在运行"""
        return self.running


class EventCameraThread(QThread):
    """事件相机线程管理类"""
    
    # 信号定义
    change_pixmap_signal = pyqtSignal(object)
    camera_status_signal = pyqtSignal(str)
    connection_status_signal = pyqtSignal(bool)
    fps_signal = pyqtSignal(float)
    flow_field_signal = pyqtSignal(np.ndarray, np.ndarray)
    finished = pyqtSignal()
    error_occurred = pyqtSignal(str)
    
    def __init__(self, input_event_file=None, camera_serial=None, slicing_mode='N_US', delta_ts=10000, delta_n_events=100000):
        super().__init__()
        self.input_event_file = input_event_file
        self.camera_serial = camera_serial
        self.slicing_mode = slicing_mode
        self.delta_ts = delta_ts
        self.delta_n_events = delta_n_events
        
        # 创建worker对象，但不在主线程中初始化相机
        self.worker = EventCameraWorker(
            input_event_file,
            camera_serial,
            slicing_mode,
            delta_ts,
            delta_n_events
        )
        
        # 将worker移动到线程
        self.worker.moveToThread(self)
        
        # 连接worker信号到线程信号
        self.worker.change_pixmap_signal.connect(self.change_pixmap_signal)
        self.worker.camera_status_signal.connect(self.camera_status_signal)
        self.worker.connection_status_signal.connect(self.connection_status_signal)
        self.worker.fps_signal.connect(self.fps_signal)
        self.worker.flow_field_signal.connect(self.flow_field_signal)
        self.worker.finished.connect(self.finished)
        self.worker.error_occurred.connect(self.error_occurred)
        
        # 连接线程信号到worker槽
        self.started.connect(self.worker.start_capture)
        
    def run(self):
        """线程入口点 - 只需要启动事件循环"""
        # 启动事件循环
        self.exec_()
        
    def stop(self):
        """停止线程"""
        if self.worker:
            self.worker.stop_capture()
            
        if self.isRunning():
            self.quit()
            self.wait(500)
            
    def is_connected(self):
        """检查相机是否已连接"""
        if self.worker:
            return self.worker.is_connected()
        return False
        
    def is_running(self):
        """检查捕获是否正在运行"""
        if self.worker:
            return self.worker.is_running()
        return False