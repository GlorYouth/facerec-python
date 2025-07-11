#!/usr/bin/env python3
"""
树莓派人脸识别监控系统
主程序入口
"""

import os
import logging
import signal
import time

from config.config_manager import ConfigManager
from monitor.monitor import FaceMonitor
from interface.web.web_server import WebServer
from utils.file_writer import FileWriter
from utils.logger import setup_logger
from utils.recognition_recorder import RecognitionRecorder


class Application:
    """应用程序主类"""
    
    def __init__(self):
        """初始化应用程序"""
        # 解析命令行参数
        self.args = ConfigManager.parse_args()
        
        # 初始化配置管理器
        config_file = self.args.config
        self.config = ConfigManager(config_file)
        
        # 应用命令行参数
        self.config.apply_command_line_args(self.args)
        
        # 创建必要的目录
        self._create_directories()
        
        # 配置日志
        self._setup_logging()
        
        # 初始化文件写入器
        self.file_writer = FileWriter()

        # 初始化识别记录器
        self.recorder = RecognitionRecorder(self.config)
        
        # 初始化监控系统
        self.monitor = FaceMonitor(self.config, self.file_writer, self.recorder)
        
        # 初始化Web服务器
        self.web_server = None
        if self.config.get('web_interface.enabled', True):
            self.web_server = WebServer(self.config, self.monitor)
        
        # 是否正在运行
        self.running = False
        
    def _create_directories(self) -> None:
        """创建必要的目录"""
        # 日志目录
        log_config = self.config.get('logging')
        if log_config and 'file' in log_config:
            os.makedirs(os.path.dirname(log_config['file']), exist_ok=True)

        # 已知人脸目录
        known_faces_dir = self.config.get('face_recognition.known_faces_dir', './data/known_faces')
        os.makedirs(known_faces_dir, exist_ok=True)
        
        # 未知人脸目录
        if self.config.get('face_recognition.save_unknown_faces', True):
            unknown_faces_dir = self.config.get('face_recognition.unknown_faces_dir', './data/unknown_faces')
            os.makedirs(unknown_faces_dir, exist_ok=True)
            
        # 检测图像目录
        if self.config.get('monitoring.actions.save_image', True):
            detected_images_dir = self.config.get('monitoring.actions.images_dir', './data/detected_images')
            os.makedirs(detected_images_dir, exist_ok=True)
        
    def _setup_logging(self) -> None:
        """配置日志系统"""
        log_config = self.config.get('logging')
        if not log_config:
            print("Warning: Logging configuration not found. Using basic config.")
            logging.basicConfig(level=logging.INFO)
            return

        log_level = log_config.get('level', 'INFO').upper()
        log_file = log_config.get('file', 'logs/facerec.log')
        max_size = log_config.get('max_size', 10485760)
        backup_count = log_config.get('backup_count', 5)

        # 命令行--verbose覆盖等级
        if self.args.verbose:
            log_level = 'DEBUG'

        setup_logger(
            level=log_level,
            filename=log_file,
            max_bytes=max_size,
            backup_count=backup_count
        )
        
    def _setup_signal_handlers(self) -> None:
        """设置信号处理器"""
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
    def _signal_handler(self, sig, frame) -> None:
        """信号处理函数"""
        logging.info(f"收到信号 {sig}，正在停止...")
        self.stop()
        
    def start(self) -> None:
        """启动应用程序"""
        # 设置信号处理
        self._setup_signal_handlers()
        
        logging.info("正在启动树莓派人脸识别监控系统...")
        
        # 设置运行状态
        self.running = True
        
        # 启动文件写入器
        if self.file_writer:
            self.file_writer.start()

        # 启动识别记录器
        if self.recorder:
            self.recorder.start()
            
        # 启动Web服务器
        if self.web_server:
            self.web_server.start()
            
        # 判断是否自动启动监控
        auto_start = self.config.get('monitoring.auto_start', True)
        if auto_start:
            logging.info("正在自动启动监控...")
            self.monitor.start()
        
    def run(self) -> None:
        """运行应用程序"""
        self.start()
        
        try:
            # 保持主线程运行
            while self.running:
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            logging.info("收到键盘中断，正在停止...")
            
        finally:
            self.stop()
            
    def stop(self) -> None:
        """停止应用程序"""
        if not self.running:
            return
            
        logging.info("正在停止应用程序...")
        
        # 停止监控
        if self.monitor:
            self.monitor.stop()
            
        # 停止Web服务器
        if self.web_server:
            self.web_server.stop()
            
        # 停止识别记录器
        if self.recorder:
            self.recorder.stop()
            
        # 停止文件写入器
        if self.file_writer:
            self.file_writer.stop()
            
        # 更新状态
        self.running = False
        
        logging.info("应用程序已停止")
        

if __name__ == "__main__":
    # 创建并运行应用程序
    app = Application()
    app.run() 