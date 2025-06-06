"""
异步文件写入类
使用后台线程处理文件写入任务，避免阻塞主线程。
"""

import threading
import queue
import logging
import numpy as np

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

class FileWriter:
    """
    通过后台线程异步写入图像文件，避免阻塞。
    """
    def __init__(self):
        """初始化文件写入器。"""
        if not CV2_AVAILABLE:
            raise ImportError("OpenCV (cv2) is not available, FileWriter cannot function.")
            
        self.write_queue = queue.Queue()
        self.is_running = False
        self.worker_thread = None
        
    def _worker(self):
        """后台工作线程，从队列中获取任务并写入文件。"""
        logging.info("文件写入工作线程已启动。")
        while self.is_running or not self.write_queue.empty():
            try:
                # 从队列中获取任务，设置超时以允许线程在空闲时退出
                image_data, file_path = self.write_queue.get(timeout=1)
                
                if image_data is None:  # 哨兵值，用于退出
                    continue
                
                cv2.imwrite(file_path, image_data)
                logging.debug(f"成功保存文件: {file_path}")
                self.write_queue.task_done()
                
            except queue.Empty:
                # 队列为空，继续循环检查 is_running 状态
                continue
            except Exception as e:
                logging.error(f"写入文件失败 {file_path}: {e}")
        
        logging.info("文件写入工作线程已停止。")
        
    def start(self):
        """启动文件写入工作线程。"""
        if self.is_running:
            logging.warning("文件写入器已在运行。")
            return
            
        self.is_running = True
        self.worker_thread = threading.Thread(target=self._worker)
        self.worker_thread.daemon = True
        self.worker_thread.start()
        
    def stop(self):
        """停止文件写入工作线程。"""
        if not self.is_running:
            return
            
        logging.info("正在停止文件写入器...")
        self.is_running = False
        # 等待所有任务完成
        self.write_queue.join()
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=2.0)
            
    def save(self, image_data: np.ndarray, file_path: str):
        """
        将保存任务添加到队列中。
        
        Args:
            image_data: 要保存的图像数据 (numpy array, BGR格式)。
            file_path: 文件保存路径。
        """
        if not self.is_running:
            logging.warning("文件写入器未运行，无法保存。")
            return
        
        # 将任务放入队列
        self.write_queue.put((image_data, file_path)) 