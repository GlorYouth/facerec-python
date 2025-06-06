"""
摄像头模块初始化文件
提供不同类型摄像头的统一接口
"""

from .camera_interface import CameraInterface
from .camera_factory import CameraFactory

__all__ = ['CameraInterface', 'CameraFactory'] 