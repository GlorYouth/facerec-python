"""
OpenCV摄像头实现类
基于OpenCV实现摄像头功能
"""

import numpy as np
from typing import Tuple, Any, Optional

from .camera_interface import CameraInterface

try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False


class OpenCVCamera(CameraInterface):
    """OpenCV摄像头实现类"""
    
    def __init__(self, resolution: Tuple[int, int] = (640, 480), fps: int = 30, device_id: int = 0, **options):
        """
        初始化OpenCV摄像头
        
        Args:
            resolution: 分辨率，格式为(宽, 高)
            fps: 帧率
            device_id: 设备ID
            options: 其他摄像头特定选项
        """
        super().__init__(resolution, fps, **options)
        self.device_id = device_id
        self.cap = None
        
        if not OPENCV_AVAILABLE:
            print("警告: OpenCV库未安装，无法使用OpenCV摄像头")
            
    def open(self) -> bool:
        """
        打开摄像头
        
        Returns:
            是否成功打开
        """
        if not OPENCV_AVAILABLE:
            return False
            
        try:
            # 创建VideoCapture对象
            self.cap = cv2.VideoCapture(self.device_id)
            
            if not self.cap.isOpened():
                print(f"无法打开设备ID为{self.device_id}的摄像头")
                return False
                
            # 设置分辨率
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            
            # 设置帧率
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            
            # 应用额外选项
            for key, value in self.options.items():
                if hasattr(cv2, key):
                    prop_id = getattr(cv2, key)
                    self.cap.set(prop_id, value)
                    
            return True
        except Exception as e:
            print(f"打开OpenCV摄像头失败: {e}")
            return False
            
    def close(self) -> None:
        """关闭摄像头"""
        if self.cap:
            self.cap.release()
            self.cap = None
                
    def is_opened(self) -> bool:
        """
        检查摄像头是否已打开
        
        Returns:
            摄像头是否已打开
        """
        return self.cap is not None and self.cap.isOpened()
        
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        读取一帧图像
        
        Returns:
            (成功读取, 图像数据) 的元组，如果读取失败，图像数据为None
        """
        if not self.is_opened():
            return False, None
            
        ret, frame = self.cap.read()
        return ret, frame
            
    def get_property(self, property_id: int) -> Any:
        """
        获取摄像头属性
        
        Args:
            property_id: 属性ID
            
        Returns:
            属性值
        """
        if not self.is_opened():
            return None
            
        return self.cap.get(property_id)
            
    def set_property(self, property_id: int, value: Any) -> bool:
        """
        设置摄像头属性
        
        Args:
            property_id: 属性ID
            value: 属性值
            
        Returns:
            是否设置成功
        """
        if not self.is_opened():
            return False
            
        return self.cap.set(property_id, value)
            
    def set_resolution(self, width: int, height: int) -> bool:
        """
        设置分辨率
        
        Args:
            width: 宽度
            height: 高度
            
        Returns:
            是否设置成功
        """
        # 更新内部属性
        super().set_resolution(width, height)
        
        if not self.is_opened():
            return False
            
        # 设置OpenCV分辨率
        width_success = self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        height_success = self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        # 验证设置是否成功
        actual_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        
        if abs(actual_width - width) > 1 or abs(actual_height - height) > 1:
            print(f"警告: 无法设置精确分辨率，请求 {width}x{height}，实际 {actual_width}x{actual_height}")
            
        return width_success and height_success
        
    def set_fps(self, fps: int) -> bool:
        """
        设置帧率
        
        Args:
            fps: 帧率
            
        Returns:
            是否设置成功
        """
        # 更新内部属性
        super().set_fps(fps)
        
        if not self.is_opened():
            return False
            
        # 设置OpenCV帧率
        success = self.cap.set(cv2.CAP_PROP_FPS, fps)
        
        # 验证设置是否成功
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        if abs(actual_fps - fps) > 1:
            print(f"警告: 无法设置精确帧率，请求 {fps}，实际 {actual_fps}")
            
        return success 