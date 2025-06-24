"""
人脸监控类
整合摄像头和人脸识别功能，实现人脸监控
"""

import os
import time
import logging
import threading
import cv2
import numpy as np
from typing import List, Tuple, Optional, Any
from PIL import Image, ImageDraw, ImageFont

from config.config_manager import ConfigManager
from camera.camera_interface import CameraInterface
from camera.camera_factory import CameraFactory
from face.face_recognizer import FaceRecognizer
from utils.file_writer import FileWriter
from utils.recognition_recorder import RecognitionRecorder


class FaceMonitor:
    """人脸监控类，整合摄像头和人脸识别功能"""

    def __init__(self, config: ConfigManager, file_writer: FileWriter, recorder: RecognitionRecorder):
        """
        初始化人脸监控
        
        Args:
            config: 配置管理器实例
            file_writer: 文件写入器实例
            recorder: 识别记录器实例
        """
        self.config = config
        self.file_writer = file_writer
        self.recorder = recorder
        self.camera: Optional[CameraInterface] = None
        self.face_recognizer: Optional[FaceRecognizer] = None
        self.font: Optional[ImageFont.FreeTypeFont] = None
        self.is_running = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.last_frame: Optional[np.ndarray] = None
        self.frame_lock = threading.Lock()
        self.detection_count = 0
        self.last_motion_detection_time = 0
        self.last_recognition_time = 0
        
        self._init_components()

    def _init_components(self) -> None:
        """初始化监控所需的组件"""
        try:
            # 加载字体
            self._load_font()

            # 创建摄像头
            camera_type = self.config.get('camera.type', 'picamera2')
            camera_config = self.config.get('camera', {})
            self.camera = CameraFactory.create_camera(
                camera_type=camera_type,
                resolution=(camera_config.get('resolution', {}).get('width', 640), camera_config.get('resolution', {}).get('height', 480)),
                fps=camera_config.get('fps', 30),
                device_id=camera_config.get('device_id', 0),
                **camera_config.get('options', {})
            )
            if not self.camera:
                logging.error(f"无法创建类型为 {camera_type} 的摄像头")
                return

            # 创建人脸识别器
            fr_config = self.config.get('face_recognition', {})
            known_faces_dir = fr_config.get('known_faces_dir', './data/known_faces')
            os.makedirs(known_faces_dir, exist_ok=True)
            if fr_config.get('save_unknown_faces', True):
                os.makedirs(fr_config.get('unknown_faces_dir', './data/unknown_faces'), exist_ok=True)

            self.face_recognizer = FaceRecognizer(
                known_faces_dir=known_faces_dir,
                model=fr_config.get('model', 'hog'),
                tolerance=fr_config.get('tolerance', 0.6),
                detection_fps=fr_config.get('detection_fps', 5),
                save_unknown_faces=fr_config.get('save_unknown_faces', True),
                unknown_faces_dir=fr_config.get('unknown_faces_dir', './data/unknown_faces'),
                file_writer=self.file_writer,
                font=self.font
            )
            
        except Exception as e:
            logging.error(f"初始化监控组件失败: {e}", exc_info=True)

    def start(self) -> bool:
        """启动监控系统"""
        if self.is_running:
            logging.info("监控系统已经在运行")
            return True
        if not self.camera or not self.face_recognizer:
            logging.error("摄像头或人脸识别器未初始化，无法启动监控")
            return False

        if not self.camera.start():
            logging.error("无法启动摄像头")
            return False
            
        self.is_running = True
        self.monitor_thread = threading.Thread(target=self._monitor_worker)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        logging.info("监控系统已启动")
        return True

    def stop(self) -> None:
        """停止监控系统"""
        self.is_running = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(2.0)
        if self.camera:
            self.camera.stop()
        logging.info("监控系统已停止")

    def _load_font(self) -> None:
        """加载用于绘制中文文本的字体"""
        font_path = self.config.get('monitoring.font_path', 'assets/fonts/wqy-zenhei.ttf')
        font_size = self.config.get('monitoring.font_size', 15)
        try:
            self.font = ImageFont.truetype(font_path, font_size)
            logging.info(f"成功加载字体: {font_path}")
        except IOError:
            self.font = None
            logging.warning(f"无法加载中文字体 '{font_path}'。将使用默认字体，中文可能无法正常显示。")
            logging.warning("请下载一个支持中文的TTF字体文件（如 '文泉驿正黑' 或 '思源黑体'）并放置在指定路径，然后在 config.yaml 中配置 'monitoring.font_path'。")

    def _monitor_worker(self) -> None:
        """监控线程工作函数"""
        logging.info("监控线程已启动")
        
        mon_config = self.config.get('monitoring', {})
        enable_motion_detection = mon_config.get('enable_motion_detection', True)
        motion_fps = mon_config.get('motion_detection_fps', 5)
        motion_interval = 1.0 / motion_fps if motion_fps > 0 else 0
        
        rec_fps = self.config.get('face_recognition.detection_fps', 5)
        recognition_interval = 1.0 / rec_fps if rec_fps > 0 else 0

        prev_frame = None
        motion_detected = False

        while self.is_running:
            try:
                ret, frame = self.camera.read()
                if not ret or frame is None:
                    time.sleep(0.01)
                    continue

                current_time = time.time()
                
                # 1. 运动检测
                if enable_motion_detection:
                    if prev_frame is not None and current_time - self.last_motion_detection_time >= motion_interval:
                        motion_detected = self._detect_motion(prev_frame, frame)
                        self.last_motion_detection_time = current_time
                    prev_frame = frame.copy()
                else:
                    motion_detected = True

                # 2. 人脸识别
                if motion_detected and current_time - self.last_recognition_time >= recognition_interval:
                    self.last_recognition_time = current_time
                    
                    # 使用副本进行识别，避免在绘制时影响原图
                    processing_frame = frame.copy()
                    face_results = self.face_recognizer.recognize_faces(processing_frame)
                    
                    if face_results:
                        # 3. 处理识别结果 (更新状态)
                        for (box, name) in face_results:
                            # 提取人脸图像并记录，用于后台和"最近识别"列表
                            face_image = processing_frame[box[0]:box[2], box[3]:box[1]]
                            self.recorder.record(name, face_image)

                            # 处理显示逻辑，并根据结果增加计数器
                            is_newly_drawn = self.recorder.process_for_display(name, box)
                            if is_newly_drawn:
                                self.detection_count += 1
                
                # 4. 在每一帧都进行绘制
                faces_to_draw = self.recorder.get_faces_to_draw()
                for (box, name) in faces_to_draw:
                    self._draw_face(frame, box, name)

                # 5. 更新用于Web界面显示的最终帧
                with self.frame_lock:
                    self.last_frame = frame

            except Exception as e:
                logging.error(f"监控工作线程出错: {e}", exc_info=True)
                time.sleep(1)

    def _draw_face(self, frame: np.ndarray, box: Tuple[int, int, int, int], name: str) -> None:
        """在图像上绘制单个人脸框和姓名，支持中文。"""
        (top, right, bottom, left) = box
        
        # 绘制人脸框
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        # 检查字体是否加载成功
        if self.font:
            try:
                # Pillow绘制中文需要将图像从OpenCV格式(BGR)转换为Pillow格式(RGB)
                pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(pil_img)

                label = name
                # Pillow 10.0.0+ 使用 textbbox 获取文本尺寸
                if hasattr(draw, 'textbbox'):
                    text_bbox = draw.textbbox((left, top), label, font=self.font)
                    text_width = text_bbox[2] - text_bbox[0]
                    text_height = text_bbox[3] - text_bbox[1]
                else:
                    # 旧版Pillow使用 textsize
                    text_width, text_height = draw.textsize(label, font=self.font)

                # 文本绘制位置
                text_y = top - text_height - 5
                # 如果文本框超出顶部，则显示在人脸框内部
                if text_y < 0:
                    text_y = top + 5
                
                # 绘制文本背景框
                draw.rectangle([(left, text_y), (left + text_width, text_y + text_height)], fill=(255, 255, 255))
                # 绘制文本
                draw.text((left, text_y), label, font=self.font, fill=(0, 0, 0, 255))
                
                # 将Pillow图像转换回OpenCV格式并更新原图
                frame[:] = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            except Exception as e:
                logging.warning(f"使用Pillow绘制文本时出错: {e}。回退到OpenCV绘制。")
                cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        else:
            # 如果字体未加载，使用OpenCV的默认英文字体
            label = name
            label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            top_label = max(top, label_size[1])
            cv2.rectangle(frame, (left, top_label - label_size[1]), (left + label_size[0], top_label + base_line), (255, 255, 255), cv2.FILLED)
            cv2.putText(frame, label, (left, top_label), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    def _detect_motion(self, prev_frame: np.ndarray, curr_frame: np.ndarray) -> bool:
        """检测两帧之间的运动"""
        sensitivity = self.config.get('monitoring.motion_sensitivity', 70)
        
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        prev_gray = cv2.GaussianBlur(prev_gray, (21, 21), 0)
        
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.GaussianBlur(curr_gray, (21, 21), 0)
        
        frame_delta = cv2.absdiff(prev_gray, curr_gray)
        thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
        
        # 通过像素变化百分比判断运动
        non_zero_count = np.count_nonzero(thresh)
        total_pixels = thresh.shape[0] * thresh.shape[1]
        motion_percentage = (non_zero_count / total_pixels) * 100
        
        # motion_sensitivity (1-100), 越高越不灵敏, 把它转换成一个0-1的阈值
        threshold = 1.01 - (sensitivity / 100.0)

        return motion_percentage > threshold

    def get_latest_frame(self) -> Optional[np.ndarray]:
        """获取最新的视频帧"""
        with self.frame_lock:
            if self.last_frame is None:
                return None
            return self.last_frame.copy()

    def get_detection_count(self) -> int:
        """获取总检测人次"""
        return self.detection_count

    def add_face(self, person_name: str) -> bool:
        """添加新的人脸"""
        if not self.face_recognizer:
            logging.error("人脸识别器未初始化，无法添加人脸")
            return False
        
        frame = self.get_latest_frame()
        if frame is None:
            logging.error("无法获取当前帧，无法添加人脸")
            return False
        
        return self.face_recognizer.add_known_face_from_frame(frame, person_name)

    def delete_face(self, person_name: str) -> bool:
        """删除一个已知的人脸"""
        if not self.face_recognizer:
            logging.error("人脸识别器未初始化，无法删除人脸")
            return False
        
        success = self.face_recognizer.delete_known_face(person_name)
        if success:
            logging.info(f"已请求删除人脸: {person_name}")
        else:
            logging.error(f"请求删除人脸失败: {person_name}")
        return success

    def is_active(self) -> bool:
        """检查监控是否正在运行"""
        return self.is_running and self.monitor_thread is not None and self.monitor_thread.is_alive() 