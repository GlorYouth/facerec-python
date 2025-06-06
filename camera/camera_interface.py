"""
摄像头接口基类
定义摄像头操作的抽象接口
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple, Dict, Any, Optional


class CameraInterface(ABC):
    """
    摄像头接口抽象基类
    所有具体摄像头实现类都应该继承此类
    """
    
    def __init__(self, resolution: Tuple[int, int] = (640, 480), fps: int = 30, **options):
        """
        初始化摄像头接口
        
        Args:
            resolution: 分辨率，格式为(宽, 高)
            fps: 帧率
            options: 其他摄像头特定选项
        """
        self.width, self.height = resolution
        self.fps = fps
        self.options = options
        self.is_running = False
        
    @abstractmethod
    def open(self) -> bool:
        """
        打开摄像头
        
        Returns:
            是否成功打开
        """
        pass
        
    @abstractmethod
    def close(self) -> None:
        """关闭摄像头"""
        pass
        
    @abstractmethod
    def is_opened(self) -> bool:
        """
        检查摄像头是否已打开
        
        Returns:
            摄像头是否已打开
        """
        pass
        
    @abstractmethod
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        读取一帧图像
        
        Returns:
            (成功读取, 图像数据) 的元组，如果读取失败，图像数据为None
        """
        pass
        
    @abstractmethod
    def get_property(self, property_id: int) -> Any:
        """
        获取摄像头属性
        
        Args:
            property_id: 属性ID
            
        Returns:
            属性值
        """
        pass
        
    @abstractmethod
    def set_property(self, property_id: int, value: Any) -> bool:
        """
        设置摄像头属性
        
        Args:
            property_id: 属性ID
            value: 属性值
            
        Returns:
            是否设置成功
        """
        pass
        
    def start(self) -> bool:
        """
        启动摄像头视频流
        
        Returns:
            是否启动成功
        """
        result = self.open()
        if result:
            self.is_running = True
        return result
        
    def stop(self) -> None:
        """停止摄像头视频流"""
        self.is_running = False
        self.close()
        
    def get_resolution(self) -> Tuple[int, int]:
        """
        获取当前分辨率
        
        Returns:
            (宽, 高) 元组
        """
        return (self.width, self.height)
        
    def set_resolution(self, width: int, height: int) -> bool:
        """
        设置分辨率
        
        Args:
            width: 宽度
            height: 高度
            
        Returns:
            是否设置成功
        """
        self.width = width
        self.height = height
        return True
        
    def get_fps(self) -> int:
        """
        获取当前帧率
        
        Returns:
            帧率
        """
        return self.fps
        
    def set_fps(self, fps: int) -> bool:
        """
        设置帧率
        
        Args:
            fps: 帧率
            
        Returns:
            是否设置成功
        """
        self.fps = fps
        return True 