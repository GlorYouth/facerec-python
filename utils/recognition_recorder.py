#!/usr/bin/env python3
"""
人脸识别结果记录器
"""

import logging
import threading
from queue import Queue, Empty
from typing import Any, Dict, Tuple, List
import time


class RecognitionRecorder:
    """
    负责人脸识别结果的缓存、状态更新和异步写入磁盘。
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化记录器。

        :param config: 应用程序配置字典
        """
        self.config = config
        self._queue = Queue()
        self._thread = None
        self._running = False
        self._last_seen = {} # 记录每个人的最后出现时间
        self._face_cache = {} # 缓存每个人的近期照片
        
        # 显示相关状态
        display_config = self.config.get('display', {})
        self.draw_cooldown = display_config.get('draw_cooldown', 5)
        self.box_retention = display_config.get('box_retention', 3)
        self._last_drawn_timestamp = {} # 记录每个人最后一次被绘制的时间
        self._faces_to_display = {} # 需要显示的人脸: {name: {'box': (t,r,b,l), 'expiry': timestamp}}

    def _process_queue(self):
        """
        处理队列中的数据，在后台运行。
        """
        while self._running:
            try:
                # 非阻塞地从队列获取数据
                name, face_image = self._queue.get(timeout=1)
                
                # 更新状态
                timestamp = time.time()
                self._last_seen[name] = timestamp
                self._face_cache[name] = (face_image, timestamp)
                
                logging.info(f"记录到人脸: {name}")

                # 在这里可以添加异步写入磁盘的逻辑

            except Empty:
                # 队列为空时，继续循环
                continue
            except Exception as e:
                logging.error(f"处理识别结果时出错: {e}")
                
    def record(self, name: str, face_image: Any):
        """
        记录一次新的人脸识别结果。

        :param name: 识别出的人名
        :param face_image: 人脸的图像数据 (e.g., a numpy array)
        """
        if not self._running:
            logging.warning("记录器未运行，忽略此次记录。")
            return
        
        self._queue.put((name, face_image))

    def process_for_display(self, name: str, box: Tuple[int, int, int, int]) -> bool:
        """
        处理一个识别到的人脸，决定是否需要更新其显示状态。
        检查冷却时间，如果通过，则更新此人脸的显示过期时间。

        :param name: 识别出的人名
        :param box: 人脸框位置
        :return: 如果这个人脸是新开始绘制的（即冷却时间通过），返回 True
        """
        current_time = time.time()
        last_drawn_time = self._last_drawn_timestamp.get(name)

        should_draw = False
        if last_drawn_time is None or (current_time - last_drawn_time) > self.draw_cooldown:
            self._last_drawn_timestamp[name] = current_time
            should_draw = True
        
        # 无论是否冷却通过，都更新人脸框位置和过期时间
        self._faces_to_display[name] = {
            'box': box,
            'expiry': current_time + self.box_retention
        }
        
        return should_draw

    def get_faces_to_draw(self) -> List[Tuple[Tuple[int, int, int, int], str]]:
        """
        获取所有应在当前帧绘制的人脸框。
        此方法会清理掉已过期的条目。

        :return: 一个列表，每个元素是 ((box), name)
        """
        current_time = time.time()
        active_faces = []
        
        # 使用 list(keys()) 来避免在迭代时修改字典
        for name in list(self._faces_to_display.keys()):
            face_data = self._faces_to_display[name]
            if current_time > face_data['expiry']:
                # 如果已过期，从字典中移除
                del self._faces_to_display[name]
            else:
                # 否则，添加到返回列表
                active_faces.append((face_data['box'], name))
                
        return active_faces

    def get_status(self) -> Dict[str, float]:
        """
        获取当前所有已识别人脸的最后出现时间。

        :return: 一个字典，键是人名，值是最后出现的时间戳
        """
        return self._last_seen.copy()

    def start(self):
        """
        启动后台处理线程。
        """
        if self._running:
            logging.warning("记录器已在运行中。")
            return
            
        logging.info("正在启动识别结果记录器...")
        self._running = True
        self._thread = threading.Thread(target=self._process_queue)
        self._thread.daemon = True
        self._thread.start()

    def stop(self):
        """
        停止后台处理线程。
        """
        if not self._running:
            return
            
        logging.info("正在停止识别结果记录器...")
        self._running = False
        if self._thread:
            self._thread.join()
        logging.info("识别结果记录器已停止。") 