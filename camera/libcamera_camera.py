"""
LibCamera摄像头实现类
基于libcamera库实现摄像头功能
"""

import numpy as np
import time
from typing import Tuple, Any, Optional
import threading

from .camera_interface import CameraInterface

try:
    import libcamera
    from libcamera import controls
    LIBCAMERA_AVAILABLE = True
except ImportError:
    LIBCAMERA_AVAILABLE = False


class LibCameraCamera(CameraInterface):
    """LibCamera摄像头实现类"""
    
    def __init__(self, resolution: Tuple[int, int] = (640, 480), fps: int = 30, **options):
        """
        初始化LibCamera摄像头
        
        Args:
            resolution: 分辨率，格式为(宽, 高)
            fps: 帧率
            options: 其他摄像头特定选项
        """
        super().__init__(resolution, fps, **options)
        self.camera = None
        self.camera_manager = None
        self.config = None
        self.stream = None
        self.current_request = None
        self.last_frame = None
        self.frame_lock = threading.Lock()
        self.capturing = False
        self.capture_thread = None
        
        if not LIBCAMERA_AVAILABLE:
            print("警告: libcamera库未安装，无法使用LibCamera摄像头")
    
    def _capture_worker(self):
        """后台捕获线程函数"""
        while self.capturing:
            try:
                # 准备请求
                request = self.camera.create_request()
                
                # 分配缓冲区
                stream = next(s for s in self.camera.streams if s.configuration.pixel_format == libcamera.formats.RGB888)
                buffer = self.camera.allocate_buffer(stream)
                request.add_buffer(stream, buffer)
                
                # 提交请求
                self.camera.queue_request(request)
                
                # 等待完成
                event = self.camera.wait()
                if event.type == libcamera.Request.Event.RequestComplete:
                    # 获取结果
                    completed_request = event.request
                    result_buffer = completed_request.buffers[stream]
                    
                    # 将缓冲区转换为numpy数组
                    mmap = result_buffer.mmap()
                    array = np.frombuffer(mmap, dtype=np.uint8)
                    frame = array.reshape((self.height, self.width, 3))
                    
                    # 保存帧
                    with self.frame_lock:
                        self.last_frame = frame.copy()
                        
                    # 释放缓冲区
                    completed_request.release()
                    
                # 短暂休眠以限制帧率
                time.sleep(1.0 / self.fps)
                
            except Exception as e:
                print(f"LibCamera捕获错误: {e}")
                time.sleep(0.1)
                
    def open(self) -> bool:
        """
        打开摄像头
        
        Returns:
            是否成功打开
        """
        if not LIBCAMERA_AVAILABLE:
            return False
            
        try:
            # 创建摄像头管理器
            self.camera_manager = libcamera.CameraManager()
            self.camera_manager.start()
            
            # 获取第一个可用摄像头
            camera_id = next(iter(self.camera_manager.cameras), None)
            if camera_id is None:
                print("未找到可用的摄像头")
                return False
                
            # 获取并配置摄像头
            self.camera = self.camera_manager.get(camera_id)
            self.camera.acquire()
            
            # 创建配置
            self.config = libcamera.CameraConfiguration()
            stream_config = libcamera.StreamConfiguration()
            stream_config.size = libcamera.Size(self.width, self.height)
            stream_config.pixel_format = libcamera.formats.RGB888
            stream_config.buffer_count = 4  # 缓冲区数量
            
            self.config.add_stream(stream_config)
            self.camera.configure(self.config)
            
            # 设置控制参数
            controls_map = {}
            
            if 'brightness' in self.options:
                controls_map[controls.Brightness] = float(self.options['brightness'])
                
            if 'contrast' in self.options:
                controls_map[controls.Contrast] = float(self.options['contrast'])
                
            if 'saturation' in self.options:
                controls_map[controls.Saturation] = float(self.options['saturation'])
                
            # 曝光时间控制帧率
            if self.fps > 0:
                exposure_time = int(1000000 / self.fps)  # 微秒
                controls_map[controls.ExposureTime] = exposure_time
            
            if controls_map:
                self.camera.controls.set(controls_map)
            
            # 分配缓冲区并启动摄像头
            self.camera.allocate()
            self.camera.start()
            
            # 启动捕获线程
            self.capturing = True
            self.capture_thread = threading.Thread(target=self._capture_worker)
            self.capture_thread.daemon = True
            self.capture_thread.start()
            
            time.sleep(0.5)  # 给摄像头一些启动时间
            return True
        except Exception as e:
            print(f"打开LibCamera摄像头失败: {e}")
            self.close()
            return False
    
    def close(self) -> None:
        """关闭摄像头"""
        # 停止捕获线程
        self.capturing = False
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(1.0)  # 等待线程结束，最多1秒
            
        # 释放摄像头资源
        if self.camera:
            try:
                self.camera.stop()
                self.camera.release()
            except Exception as e:
                print(f"关闭LibCamera摄像头时出错: {e}")
                
        if self.camera_manager:
            try:
                self.camera_manager.stop()
            except Exception:
                pass
                
        self.camera = None
        self.camera_manager = None
        self.config = None
        self.last_frame = None
        
    def is_opened(self) -> bool:
        """
        检查摄像头是否已打开
        
        Returns:
            摄像头是否已打开
        """
        return self.camera is not None and self.capturing
        
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        读取一帧图像
        
        Returns:
            (成功读取, 图像数据) 的元组，如果读取失败，图像数据为None
        """
        if not self.is_opened():
            return False, None
            
        with self.frame_lock:
            frame = self.last_frame
            
        if frame is None:
            return False, None
            
        return True, frame.copy()
        
    def get_property(self, property_id: int) -> Any:
        """
        获取摄像头属性
        
        Args:
            property_id: 属性ID
            
        Returns:
            属性值
        """
        if not self.camera or not hasattr(controls, property_id):
            return None
            
        try:
            return self.camera.controls.get(getattr(controls, property_id))
        except Exception as e:
            print(f"获取LibCamera属性失败: {e}")
            return None
            
    def set_property(self, property_id: int, value: Any) -> bool:
        """
        设置摄像头属性
        
        Args:
            property_id: 属性ID
            value: 属性值
            
        Returns:
            是否设置成功
        """
        if not self.camera or not hasattr(controls, property_id):
            return False
            
        try:
            self.camera.controls.set({getattr(controls, property_id): value})
            return True
        except Exception as e:
            print(f"设置LibCamera属性失败: {e}")
            return False
            
    def set_resolution(self, width: int, height: int) -> bool:
        """
        设置分辨率
        
        Args:
            width: 宽度
            height: 高度
            
        Returns:
            是否设置成功
        """
        # 需要重新启动摄像头才能更改分辨率
        if not self.camera:
            super().set_resolution(width, height)
            return False
            
        # 保存当前状态
        was_running = self.capturing
        
        # 停止摄像头
        self.close()
        
        # 更新分辨率
        super().set_resolution(width, height)
        
        # 如果之前在运行，则重新启动
        result = True
        if was_running:
            result = self.open()
            
        return result
        
    def set_fps(self, fps: int) -> bool:
        """
        设置帧率
        
        Args:
            fps: 帧率
            
        Returns:
            是否设置成功
        """
        if not self.camera:
            super().set_fps(fps)
            return False
            
        try:
            # 设置曝光时间以控制帧率
            exposure_time = int(1000000 / fps)  # 微秒
            self.camera.controls.set({controls.ExposureTime: exposure_time})
            
            # 更新内部帧率
            super().set_fps(fps)
            return True
        except Exception as e:
            print(f"设置LibCamera帧率失败: {e}")
            return False 