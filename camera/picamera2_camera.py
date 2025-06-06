"""
Picamera2摄像头实现类
基于Raspberry Pi Picamera2库实现摄像头功能
"""

import numpy as np
from typing import Tuple, Any, Optional
import time

from .camera_interface import CameraInterface

try:
    from picamera2 import Picamera2
    PICAMERA2_AVAILABLE = True
except ImportError:
    PICAMERA2_AVAILABLE = False


class PiCamera2Camera(CameraInterface):
    """Picamera2摄像头实现类"""
    
    def __init__(self, resolution: Tuple[int, int] = (640, 480), fps: int = 30, **options):
        """
        初始化Picamera2摄像头
        
        Args:
            resolution: 分辨率，格式为(宽, 高)
            fps: 帧率
            options: 其他摄像头特定选项
        """
        super().__init__(resolution, fps, **options)
        self.camera = None
        self.camera_config = None
        
        if not PICAMERA2_AVAILABLE:
            print("警告: Picamera2库未安装，无法使用Picamera2摄像头")
            
    def open(self) -> bool:
        """
        打开摄像头
        
        Returns:
            是否成功打开
        """
        if not PICAMERA2_AVAILABLE:
            return False
            
        try:
            # 创建Picamera2对象
            self.camera = Picamera2()
            
            # 配置摄像头
            self.camera_config = self.camera.create_still_configuration(
                main={"size": (self.width, self.height)},
                controls={"FrameDurationLimits": (1000000 // self.fps, 1000000 // self.fps)}
            )
            
            # 应用配置
            self.camera.configure(self.camera_config)
            
            # 应用额外选项
            for key, value in self.options.items():
                if hasattr(self.camera, key):
                    setattr(self.camera, key, value)
            
            # 启动摄像头
            self.camera.start()
            time.sleep(0.5)  # 给摄像头一些启动时间
            return True
        except Exception as e:
            print(f"打开Picamera2摄像头失败: {e}")
            return False
            
    def close(self) -> None:
        """关闭摄像头"""
        if self.camera:
            try:
                self.camera.stop()
                self.camera.close()
            except Exception as e:
                print(f"关闭Picamera2摄像头失败: {e}")
            finally:
                self.camera = None
                
    def is_opened(self) -> bool:
        """
        检查摄像头是否已打开
        
        Returns:
            摄像头是否已打开
        """
        return self.camera is not None
        
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        读取一帧图像
        
        Returns:
            (成功读取, 图像数据) 的元组，如果读取失败，图像数据为None
        """
        if not self.camera:
            return False, None
            
        try:
            frame = self.camera.capture_array()
            return True, frame
        except Exception as e:
            print(f"读取Picamera2图像失败: {e}")
            return False, None
            
    def get_property(self, property_id: int) -> Any:
        """
        获取摄像头属性
        
        Args:
            property_id: 属性ID
            
        Returns:
            属性值
        """
        if not self.camera:
            return None
            
        try:
            # Picamera2使用字典方式访问属性
            controls = self.camera.camera_controls
            return controls.get(property_id)
        except Exception as e:
            print(f"获取Picamera2属性失败: {e}")
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
        if not self.camera:
            return False
            
        try:
            # 设置Picamera2控制参数
            self.camera.set_controls({property_id: value})
            return True
        except Exception as e:
            print(f"设置Picamera2属性失败: {e}")
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
        if not self.camera:
            super().set_resolution(width, height)
            return False
            
        try:
            # 需要重新配置摄像头
            self.camera.stop()
            
            # 更新分辨率
            super().set_resolution(width, height)
            
            # 重新创建配置
            self.camera_config = self.camera.create_still_configuration(
                main={"size": (self.width, self.height)},
                controls={"FrameDurationLimits": (1000000 // self.fps, 1000000 // self.fps)}
            )
            
            # 应用配置
            self.camera.configure(self.camera_config)
            self.camera.start()
            
            return True
        except Exception as e:
            print(f"设置Picamera2分辨率失败: {e}")
            return False
            
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
            # 更新帧率
            super().set_fps(fps)
            
            # 设置帧率限制
            self.camera.set_controls({"FrameDurationLimits": (1000000 // self.fps, 1000000 // self.fps)})
            
            return True
        except Exception as e:
            print(f"设置Picamera2帧率失败: {e}")
            return False 