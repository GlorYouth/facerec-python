"""
配置管理类
负责加载、访问和保存系统配置
"""

import os
import yaml
import argparse
from pathlib import Path
from typing import Any, Dict, Optional


class ConfigManager:
    """配置管理器，处理系统配置的加载、访问和保存"""

    def __init__(self, config_file: str = None):
        """
        初始化配置管理器
        
        Args:
            config_file: 配置文件路径，如未指定则使用默认配置
        """
        self.config_file = config_file
        self.config = {}
        self.default_config_file = os.path.join(os.path.dirname(__file__), 'default_config.yaml')
        
        # 加载默认配置
        self.load_default_config()
        
        # 如果指定了配置文件，加载用户配置
        if config_file and os.path.exists(config_file):
            self.load_config(config_file)

    def load_default_config(self) -> None:
        """加载默认配置"""
        try:
            with open(self.default_config_file, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
        except Exception as e:
            print(f"无法加载默认配置: {e}")
            self.config = {}

    def load_config(self, config_file: str) -> None:
        """
        加载用户配置文件并合并到当前配置
        
        Args:
            config_file: 配置文件路径
        """
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                user_config = yaml.safe_load(f)
                if user_config:
                    # 递归合并配置
                    self._merge_config(self.config, user_config)
        except Exception as e:
            print(f"加载配置文件 {config_file} 失败: {e}")

    def _merge_config(self, base: Dict, override: Dict) -> None:
        """
        递归合并配置字典
        
        Args:
            base: 基础配置字典
            override: 覆盖配置字典
        """
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_config(base[key], value)
            else:
                base[key] = value

    def get(self, key_path: str, default: Any = None) -> Any:
        """
        通过点分隔的键路径获取配置值
        
        Args:
            key_path: 点分隔的键路径，如 'camera.resolution.width'
            default: 如果键不存在时的默认值
            
        Returns:
            对应的配置值，如果不存在则返回默认值
        """
        parts = key_path.split('.')
        config = self.config
        
        for part in parts:
            if isinstance(config, dict) and part in config:
                config = config[part]
            else:
                return default
                
        return config

    def set(self, key_path: str, value: Any) -> None:
        """
        设置配置值
        
        Args:
            key_path: 点分隔的键路径，如 'camera.resolution.width'
            value: 要设置的值
        """
        parts = key_path.split('.')
        config = self.config
        
        # 导航到最后一个键之前的所有部分
        for part in parts[:-1]:
            if part not in config or not isinstance(config[part], dict):
                config[part] = {}
            config = config[part]
                
        # 设置值
        config[parts[-1]] = value

    def save(self, file_path: Optional[str] = None) -> None:
        """
        保存当前配置到文件
        
        Args:
            file_path: 要保存到的文件路径，如未指定则使用初始化时的配置文件
        """
        save_path = file_path or self.config_file
        
        if not save_path:
            raise ValueError("未指定保存路径")
            
        # 确保目录存在
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            
        try:
            with open(save_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
        except Exception as e:
            print(f"保存配置到 {save_path} 失败: {e}")

    def get_all(self) -> Dict:
        """
        获取整个配置字典
        
        Returns:
            配置字典的副本
        """
        return self.config.copy()

    @staticmethod
    def parse_args() -> argparse.Namespace:
        """
        解析命令行参数
        
        Returns:
            解析后的参数命名空间
        """
        parser = argparse.ArgumentParser(description='树莓派人脸识别监控系统')
        parser.add_argument('-c', '--config', help='配置文件路径')
        parser.add_argument('--camera-type', help='摄像头类型 (picamera2, opencv, libcamera)')
        parser.add_argument('--resolution', help='摄像头分辨率，格式为 宽x高，例如 640x480')
        parser.add_argument('--detection-fps', type=int, help='人脸检测帧率')
        parser.add_argument('--web-port', type=int, help='Web界面端口')
        parser.add_argument('--log-file', help='日志文件路径')
        parser.add_argument('--verbose', action='store_true', help='显示详细信息')
        
        return parser.parse_args()

    def apply_command_line_args(self, args: argparse.Namespace) -> None:
        """
        应用命令行参数到配置
        
        Args:
            args: 解析后的命令行参数
        """
        if args.camera_type:
            self.set('camera.type', args.camera_type)
            
        if args.resolution:
            try:
                width, height = map(int, args.resolution.split('x'))
                self.set('camera.resolution.width', width)
                self.set('camera.resolution.height', height)
            except ValueError:
                print(f"无效的分辨率格式: {args.resolution}，应为 宽x高")
                
        if args.detection_fps is not None:
            self.set('face_recognition.detection_fps', args.detection_fps)
            
        if args.web_port is not None:
            self.set('web_interface.port', args.web_port)
            
        if args.log_file:
            self.set('monitoring.log_file', args.log_file)
            
        if args.verbose:
            self.set('console_interface.verbose', True) 