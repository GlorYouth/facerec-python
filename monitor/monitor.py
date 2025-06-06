"""
人脸监控类
整合摄像头和人脸识别功能，实现人脸监控
"""

import os
import time
import logging
import threading
import numpy as np
from typing import Callable, Dict, List, Tuple, Optional, Any
from datetime import datetime

from config.config_manager import ConfigManager
from camera.camera_interface import CameraInterface
from camera.camera_factory import CameraFactory
from face.face_recognizer import FaceRecognizer


class FaceMonitor:
    """人脸监控类，整合摄像头和人脸识别功能"""
    
    def __init__(self, config: ConfigManager):
        """
        初始化人脸监控
        
        Args:
            config: 配置管理器实例
        """
        self.config = config
        self.camera = None
        self.face_recognizer = None
        self.is_running = False
        self.monitor_thread = None
        self.last_frame = None
        self.last_detection_results = []
        self.frame_lock = threading.Lock()
        self.detection_count = 0
        
        # 运动检测频率控制
        self.last_motion_detection_time = 0
        
        # 配置日志
        log_file = config.get('monitoring.log_file', './logs/detections.log')
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        # 初始化组件
        self._init_components()
        
    def _init_components(self) -> None:
        """初始化监控所需的组件"""
        try:
            # 创建摄像头
            camera_type = self.config.get('camera.type', 'picamera2')
            camera_width = self.config.get('camera.resolution.width', 640)
            camera_height = self.config.get('camera.resolution.height', 480)
            camera_fps = self.config.get('camera.fps', 30)
            camera_device_id = self.config.get('camera.device_id', 0)
            camera_options = self.config.get('camera.options', {})
            
            # 创建摄像头对象
            self.camera = CameraFactory.create_camera(
                camera_type=camera_type,
                resolution=(camera_width, camera_height),
                fps=camera_fps,
                device_id=camera_device_id,
                **camera_options
            )
            
            if not self.camera:
                logging.error(f"无法创建类型为 {camera_type} 的摄像头")
                return
                
            # 创建人脸识别器
            face_model = self.config.get('face_recognition.model', 'hog')
            face_tolerance = self.config.get('face_recognition.tolerance', 0.6)
            known_faces_dir = self.config.get('face_recognition.known_faces_dir', './data/known_faces')
            detection_fps = self.config.get('face_recognition.detection_fps', 5)
            save_unknown_faces = self.config.get('face_recognition.save_unknown_faces', True)
            unknown_faces_dir = self.config.get('face_recognition.unknown_faces_dir', './data/unknown_faces')
            
            # 确保目录存在
            os.makedirs(known_faces_dir, exist_ok=True)
            if save_unknown_faces:
                os.makedirs(unknown_faces_dir, exist_ok=True)
                
            # 创建人脸识别器
            self.face_recognizer = FaceRecognizer(
                known_faces_dir=known_faces_dir,
                model=face_model,
                tolerance=face_tolerance,
                detection_fps=detection_fps,
                save_unknown_faces=save_unknown_faces,
                unknown_faces_dir=unknown_faces_dir
            )
            
        except Exception as e:
            logging.error(f"初始化监控组件失败: {e}")
            
    def start(self) -> bool:
        """
        启动监控系统
        
        Returns:
            是否成功启动
        """
        if self.is_running:
            logging.info("监控系统已经在运行")
            return True
            
        if not self.camera:
            logging.error("摄像头未初始化，无法启动监控")
            return False
            
        if not self.face_recognizer:
            logging.error("人脸识别器未初始化，无法启动监控")
            return False
            
        # 启动摄像头
        if not self.camera.start():
            logging.error("无法启动摄像头")
            return False
            
        # 设置运行状态
        self.is_running = True
        
        # 启动监控线程
        self.monitor_thread = threading.Thread(target=self._monitor_worker)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
        logging.info("监控系统已启动")
        return True
        
    def stop(self) -> None:
        """停止监控系统"""
        self.is_running = False
        
        # 等待线程结束
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(2.0)
            
        # 关闭摄像头
        if self.camera:
            self.camera.stop()
            
        logging.info("监控系统已停止")
        
    def _monitor_worker(self) -> None:
        """监控线程工作函数"""
        logging.info("监控线程已启动")
        
        enable_motion_detection = self.config.get('monitoring.enable_motion_detection', True)
        log_all_detections = self.config.get('monitoring.log_all_detections', True)
        motion_sensitivity = self.config.get('monitoring.motion_sensitivity', 50) / 100.0
        
        # 获取运动检测帧率配置
        motion_detection_fps = self.config.get('monitoring.motion_detection_fps', 5)
        motion_detection_interval = 1.0 / motion_detection_fps if motion_detection_fps > 0 else 0

        save_detected_images = self.config.get('monitoring.actions.save_image', True)
        detected_images_dir = self.config.get('monitoring.actions.images_dir', './data/detected_images')
        
        # 确保目录存在
        if save_detected_images:
            os.makedirs(detected_images_dir, exist_ok=True)
            
        prev_frame = None
        motion_detected = False
        
        while self.is_running:
            try:
                # 读取一帧
                ret, frame = self.camera.read()
                
                if not ret or frame is None:
                    time.sleep(0.01)
                    continue
                    
                # 保存当前帧（用于界面显示）
                with self.frame_lock:
                    self.last_frame = frame.copy()
                
                # 运动检测（如果启用，并按指定帧率执行）
                current_time = time.time()
                if enable_motion_detection:
                    if prev_frame is not None:
                        if current_time - self.last_motion_detection_time >= motion_detection_interval:
                            self.last_motion_detection_time = current_time
                            motion_detected = self._detect_motion(prev_frame, frame, motion_sensitivity)
                            # 仅在执行检测时更新用于比较的上一帧
                            prev_frame = frame.copy()
                    else:
                        # 初始化第一帧
                        prev_frame = frame.copy()
                        self.last_motion_detection_time = current_time
                else:
                    motion_detected = True  # 如果未启用运动检测，则总是进行人脸检测
                
                # 如果检测到运动或未启用运动检测，则进行人脸识别
                if motion_detected:
                    # 人脸识别
                    face_results = self.face_recognizer.recognize_faces(frame)
                    
                    # 如果检测到人脸
                    if face_results:
                        self.detection_count += 1
                        
                        # 保存检测结果（用于界面显示）
                        with self.frame_lock:
                            self.last_detection_results = face_results
                            
                        # 记录检测结果
                        if log_all_detections:
                            self._log_detection(face_results)
                            
                        # 保存检测到的图像
                        if save_detected_images:
                            self._save_detection_image(frame, face_results)
                
                # 限制帧率，减少CPU使用
                time.sleep(0.01)
                
            except Exception as e:
                logging.error(f"监控线程错误: {e}")
                time.sleep(0.1)
                
        logging.info("监控线程已结束")
        
    def _detect_motion(self, prev_frame: np.ndarray, curr_frame: np.ndarray, sensitivity: float) -> bool:
        """
        简单的运动检测
        
        Args:
            prev_frame: 前一帧
            curr_frame: 当前帧
            sensitivity: 灵敏度 (0-1)
            
        Returns:
            是否检测到运动
        """
        try:
            import cv2
            
            # 转换为灰度
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
            
            # 计算帧差
            frame_diff = cv2.absdiff(prev_gray, curr_gray)
            
            # 应用阈值
            threshold = int(25 * sensitivity)
            _, thresh = cv2.threshold(frame_diff, threshold, 255, cv2.THRESH_BINARY)
            
            # 计算变化像素百分比
            change_percent = np.count_nonzero(thresh) / thresh.size
            
            # 判断是否有足够的变化
            motion_threshold = 0.01 * sensitivity
            return change_percent > motion_threshold
            
        except Exception as e:
            logging.error(f"运动检测错误: {e}")
            return True  # 出错时默认认为有运动
            
    def _log_detection(self, face_results: List[Tuple[Tuple[int, int, int, int], str]]) -> None:
        """
        记录人脸检测结果
        
        Args:
            face_results: 人脸检测结果
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for i, (_, name) in enumerate(face_results):
            logging.info(f"检测到人脸 #{i+1}: {name} ({timestamp})")
            
    def _save_detection_image(
        self, 
        frame: np.ndarray, 
        face_results: List[Tuple[Tuple[int, int, int, int], str]]
    ) -> None:
        """
        保存检测到的人脸图像
        
        Args:
            frame: 原始图像
            face_results: 人脸检测结果
        """
        try:
            import cv2
            
            # 绘制标记的图像
            marked_frame = self.face_recognizer.draw_faces(frame, face_results)
            
            # 生成文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            detected_images_dir = self.config.get('monitoring.actions.images_dir', './data/detected_images')
            filename = f"detected_{timestamp}.jpg"
            filepath = os.path.join(detected_images_dir, filename)
            
            # 保存图像
            cv2.imwrite(filepath, cv2.cvtColor(marked_frame, cv2.COLOR_RGB2BGR))
            logging.debug(f"已保存检测图像: {filepath}")
            
        except Exception as e:
            logging.error(f"保存检测图像失败: {e}")
            
    def get_latest_frame(self) -> Optional[np.ndarray]:
        """
        获取最近一帧图像
        
        Returns:
            最近一帧图像，如果没有则返回None
        """
        with self.frame_lock:
            if self.last_frame is not None:
                return self.last_frame.copy()
        return None
        
    def get_latest_detection(self) -> Tuple[Optional[np.ndarray], List]:
        """
        获取最近一次检测结果
        
        Returns:
            (带标记的图像, 人脸结果列表) 元组，如果没有则图像为None
        """
        with self.frame_lock:
            if self.last_frame is not None and self.last_detection_results:
                marked_frame = self.face_recognizer.draw_faces(
                    self.last_frame, 
                    self.last_detection_results
                )
                return marked_frame, self.last_detection_results
                
        return None, []
        
    def add_face(self, person_name: str) -> bool:
        """
        添加当前帧中的人脸到数据库
        
        Args:
            person_name: 人名
            
        Returns:
            是否成功添加
        """
        if not self.face_recognizer:
            logging.error("人脸识别器未初始化")
            return False
            
        # 获取当前帧
        frame = self.get_latest_frame()
        if frame is None:
            logging.error("无法获取当前帧")
            return False
            
        # 添加人脸
        return self.face_recognizer.add_face(frame, person_name)
        
    def is_active(self) -> bool:
        """
        检查监控系统是否处于活动状态
        
        Returns:
            是否处于活动状态
        """
        return self.is_running 