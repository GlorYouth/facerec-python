"""
摄像头工厂类
负责创建不同类型摄像头的实例
"""

import importlib
from typing import Dict, Any, Optional, Type, Tuple

from .camera_interface import CameraInterface


class CameraFactory:
    """摄像头工厂类，用于创建不同类型的摄像头实例"""
    
    # 可用摄像头类型与对应实现类的映射
    _camera_types = {
        'picamera2': 'camera.picamera2_camera.PiCamera2Camera',
        'opencv': 'camera.opencv_camera.OpenCVCamera',
        'libcamera': 'camera.libcamera_camera.LibCameraCamera',
    }
    
    @classmethod
    def create_camera(
        cls, 
        camera_type: str, 
        resolution: Tuple[int, int] = (640, 480), 
        fps: int = 30, 
        **options
    ) -> Optional[CameraInterface]:
        """
        创建指定类型的摄像头实例
        
        Args:
            camera_type: 摄像头类型
            resolution: 分辨率，格式为(宽, 高)
            fps: 帧率
            options: 其他摄像头特定选项
            
        Returns:
            摄像头实例，如果类型不支持则返回None
        """
        if camera_type not in cls._camera_types:
            print(f"不支持的摄像头类型: {camera_type}")
            return None
            
        try:
            # 动态导入摄像头实现类
            module_path, class_name = cls._camera_types[camera_type].rsplit('.', 1)
            module = importlib.import_module(module_path)
            camera_class = getattr(module, class_name)
            
            # 创建实例
            camera = camera_class(resolution=resolution, fps=fps, **options)
            return camera
        except (ImportError, AttributeError) as e:
            print(f"创建摄像头失败: {e}")
            return None
            
    @classmethod
    def get_available_camera_types(cls) -> list:
        """
        获取所有可用的摄像头类型
        
        Returns:
            可用摄像头类型列表
        """
        return list(cls._camera_types.keys())
        
    @classmethod
    def register_camera_type(cls, camera_type: str, implementation: str) -> None:
        """
        注册新的摄像头类型及其实现类
        
        Args:
            camera_type: 摄像头类型名称
            implementation: 实现类的完整路径（格式如：'module.submodule.ClassName'）
        """
        cls._camera_types[camera_type] = implementation 