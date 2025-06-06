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
from face.face_tracker import FaceTracker
from utils.file_writer import FileWriter


class FaceMonitor:
    """人脸监控类，整合摄像头和人脸识别功能"""
    
    def __init__(self, config: ConfigManager, file_writer: FileWriter):
        """
        初始化人脸监控
        
        Args:
            config: 配置管理器实例
            file_writer: 文件写入器实例
        """
        self.config = config
        self.file_writer = file_writer
        self.camera = None
        self.face_recognizer = None
        self.face_tracker = None
        self.is_running = False
        self.monitor_thread = None
        self.last_frame = None
        self.last_detection_results = []
        self.frame_lock = threading.Lock()
        self.detection_count = 0
        
        # 运动检测频率控制
        self.last_motion_detection_time = 0
        
        # 人脸识别频率控制
        self.last_recognition_time = 0
        
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
                unknown_faces_dir=unknown_faces_dir,
                file_writer=self.file_writer
            )
            
            # 创建人脸跟踪器
            max_disappeared = self.config.get('face_tracking.max_disappeared', 30)
            min_distance = self.config.get('face_tracking.min_distance', 0.6)
            min_iou = self.config.get('face_tracking.min_iou', 0.3)
            
            self.face_tracker = FaceTracker(
                max_disappeared=max_disappeared,
                min_distance=min_distance,
                min_iou=min_iou
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

        # 获取人脸识别帧率配置
        recognition_fps = self.config.get('face_recognition.detection_fps', 5)
        recognition_interval = 1.0 / recognition_fps if recognition_fps > 0 else 0
        
        # 获取跟踪配置
        enable_face_tracking = self.config.get('face_tracking.enable', True)

        save_detected_images = self.config.get('monitoring.actions.save_image', True)
        detected_images_dir = self.config.get('monitoring.actions.images_dir', './data/detected_images')
        
        # 确保目录存在
        if save_detected_images:
            os.makedirs(detected_images_dir, exist_ok=True)
            
        prev_frame = None
        motion_detected = False
        
        # 最后一次进行完整识别的时间
        self.last_recognition_time = 0
        
        # 跟踪的人脸结果
        tracked_faces = {}
        
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
                
                # 处理逻辑：基于帧率和运动检测结果决定是进行完整识别还是只进行跟踪
                if motion_detected:
                    # 确定是否需要进行完整的人脸识别（基于帧率）
                    need_full_recognition = current_time - self.last_recognition_time >= recognition_interval
                    
                    if need_full_recognition:
                        # 进行完整的人脸识别
                        self.last_recognition_time = current_time
                        face_results = self.face_recognizer.recognize_faces(frame)
                        
                        # 转换识别结果格式，添加编码信息
                        full_face_results = []
                        for (bbox, name) in face_results:
                            # 从原始图像中获取人脸区域
                            top, right, bottom, left = bbox
                            face_image = frame[top:bottom, left:right]
                            
                            # 尝试提取编码
                            encoding = None
                            try:
                                import face_recognition
                                encodings = face_recognition.face_encodings(face_image)
                                if encodings:
                                    encoding = encodings[0]
                            except Exception as e:
                                logging.debug(f"无法提取人脸编码：{e}")
                            
                            full_face_results.append((bbox, name, encoding))
                        
                        # 更新跟踪器
                        if enable_face_tracking:
                            tracked_faces = self.face_tracker.update(full_face_results)
                    elif enable_face_tracking and tracked_faces:
                        # 仅使用跟踪结果，不进行完整识别
                        pass
                    else:
                        # 如果没有跟踪结果且不需要完整识别，跳过
                        continue
                    
                    # 准备显示结果
                    if enable_face_tracking and tracked_faces:
                        # 使用跟踪结果
                        display_results = [(track_info['bbox'], track_info['name']) 
                                        for track_id, track_info in tracked_faces.items()]
                    else:
                        # 使用识别结果
                        display_results = face_results if need_full_recognition else []
                    
                    # 如果有人脸结果（识别或跟踪）
                    if display_results:
                        self.detection_count += 1
                        
                        # 保存检测结果（用于界面显示）
                        with self.frame_lock:
                            self.last_detection_results = display_results
                            
                        # 记录检测结果
                        if log_all_detections:
                            self._log_detection(display_results)
                            
                        # 保存检测到的图像
                        if save_detected_images:
                            self._save_detection_image(frame, display_results)
                
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
            
            # 异步保存图像
            self.file_writer.save(marked_frame, filepath)
            logging.debug(f"已将检测图像任务提交到队列: {filepath}")
            
        except Exception as e:
            logging.error(f"提交检测图像保存任务失败: {e}")
            
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