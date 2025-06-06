"""
日志工具模块
"""

import logging
from logging.handlers import RotatingFileHandler

def setup_logger(level: str = "INFO", 
                filename: str = "logs/facerec.log",
                max_bytes: int = 10485760,
                backup_count: int = 5) -> None:
    """
    配置日志系统
    
    Args:
        level: 日志级别
        filename: 日志文件路径
        max_bytes: 单个日志文件最大字节数
        backup_count: 保留的日志文件数量
    """
    # 创建处理器
    handler = RotatingFileHandler(
        filename,
        maxBytes=max_bytes,
        backupCount=backup_count
    )
    
    # 设置格式
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)
    
    # 配置根日志器
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.addHandler(handler)
    
    # 同时输出到控制台
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    root_logger.addHandler(console) 