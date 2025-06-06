"""
人脸识别类
提供人脸检测和识别功能
"""

import os
import time
import uuid
import numpy as np
import threading
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import logging
from PIL import Image, ImageDraw, ImageFont

try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False


class FaceRecognizer:
    """人脸识别类，提供人脸检测和识别功能"""
    
    def __init__(
        self,
        known_faces_dir: str,
        model: str = "hog",
        tolerance: float = 0.6,
        detection_fps: int = 5,
        save_unknown_faces: bool = True,
        unknown_faces_dir: str = None,
    ):
        """
        初始化人脸识别器
        
        Args:
            known_faces_dir: 已知人脸数据库目录
            model: 人脸检测模型，可选 'hog' (CPU) 或 'cnn' (GPU，需要 CUDA)
            tolerance: 识别容差，值越小越严格
            detection_fps: 检测帧率
            save_unknown_faces: 是否保存未知人脸
            unknown_faces_dir: 未知人脸保存目录
        """
        self.known_faces_dir = known_faces_dir
        self.model = model
        self.tolerance = tolerance
        self.detection_interval = 1.0 / detection_fps if detection_fps > 0 else 0
        self.save_unknown_faces = save_unknown_faces
        self.unknown_faces_dir = unknown_faces_dir or "./data/unknown_faces"
        
        # 已知人脸数据
        self.known_face_encodings = []
        self.known_face_names = []
        
        # 线程安全锁
        self.detection_lock = threading.Lock()
        
        # 上次检测时间
        self.last_detection_time = 0
        
        # 检查依赖
        if not FACE_RECOGNITION_AVAILABLE:
            logging.error("face_recognition 库未安装，无法进行人脸识别")
            
        # 创建必要的目录
        os.makedirs(self.known_faces_dir, exist_ok=True)
        if self.save_unknown_faces:
            os.makedirs(self.unknown_faces_dir, exist_ok=True)
        
        # 加载已知人脸
        self.load_known_faces()
        
    def load_known_faces(self) -> None:
        """加载已知人脸数据库"""
        self.known_face_encodings = []
        self.known_face_names = []
        
        if not os.path.exists(self.known_faces_dir):
            logging.warning(f"已知人脸目录不存在: {self.known_faces_dir}")
            return
            
        logging.info(f"正在加载已知人脸数据库: {self.known_faces_dir}")
        
        # 遍历人脸目录
        for person_name in os.listdir(self.known_faces_dir):
            person_dir = os.path.join(self.known_faces_dir, person_name)
            
            if not os.path.isdir(person_dir):
                continue
                
            # 遍历每个人的图片
            for image_file in os.listdir(person_dir):
                image_path = os.path.join(person_dir, image_file)
                
                # 检查文件是否为图片
                if not image_path.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue
                    
                try:
                    # 加载并编码人脸
                    face_image = face_recognition.load_image_file(image_path)
                    face_encodings = face_recognition.face_encodings(face_image)
                    
                    # 如果检测到人脸，添加到数据库
                    if face_encodings:
                        self.known_face_encodings.append(face_encodings[0])
                        self.known_face_names.append(person_name)
                        logging.debug(f"已加载人脸: {person_name} ({image_file})")
                    else:
                        logging.warning(f"未在图片中检测到人脸: {image_path}")
                except Exception as e:
                    logging.error(f"加载人脸失败 {image_path}: {e}")
        
        logging.info(f"已加载 {len(self.known_face_encodings)} 个已知人脸")
        
    def add_face(self, face_image: np.ndarray, person_name: str) -> bool:
        """
        添加新的人脸到数据库
        
        Args:
            face_image: 人脸图像
            person_name: 人名
            
        Returns:
            是否成功添加
        """
        try:
            # 检测人脸
            face_locations = face_recognition.face_locations(face_image, model=self.model)
            if not face_locations:
                logging.error("未检测到人脸，无法添加")
                return False
                
            # 创建人名目录
            person_dir = os.path.join(self.known_faces_dir, person_name)
            os.makedirs(person_dir, exist_ok=True)
            
            # 生成文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            image_filename = f"{person_name}_{timestamp}.jpg"
            image_path = os.path.join(person_dir, image_filename)
            
            # 保存图片
            import cv2
            cv2.imwrite(image_path, cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR))
            
            # 重新加载人脸库
            self.load_known_faces()
            
            logging.info(f"已添加人脸: {person_name}")
            return True
        except Exception as e:
            logging.error(f"添加人脸失败: {e}")
            return False
            
    def need_detection(self) -> bool:
        """
        是否需要进行检测（基于帧率限制）
        
        Returns:
            是否应该进行检测
        """
        current_time = time.time()
        if current_time - self.last_detection_time >= self.detection_interval:
            self.last_detection_time = current_time
            return True
        return False
        
    def detect_faces(
        self, frame: np.ndarray
    ) -> List[Tuple[Tuple[int, int, int, int], str]]:
        """
        检测图像中的人脸位置
        
        Args:
            frame: 输入图像
            
        Returns:
            人脸位置列表，每个元素为 ((top, right, bottom, left), 标签) 形式
        """
        if not FACE_RECOGNITION_AVAILABLE:
            return []
            
        with self.detection_lock:
            # 检测人脸位置
            face_locations = face_recognition.face_locations(frame, model=self.model)
            return [(location, "") for location in face_locations]
            
    def recognize_faces(
        self, frame: np.ndarray
    ) -> List[Tuple[Tuple[int, int, int, int], str]]:
        """
        识别图像中的人脸
        
        Args:
            frame: 输入图像
            
        Returns:
            识别结果列表，每个元素为 ((top, right, bottom, left), 人名) 形式
        """
        if not FACE_RECOGNITION_AVAILABLE or not self.need_detection():
            return []
            
        with self.detection_lock:
            # 检测人脸位置
            face_locations = face_recognition.face_locations(frame, model=self.model)
            
            if not face_locations:
                return []
                
            # 提取人脸特征
            face_encodings = face_recognition.face_encodings(frame, face_locations)
            
            results = []
            for i, (face_location, face_encoding) in enumerate(zip(face_locations, face_encodings)):
                # 与已知人脸比较
                matches = face_recognition.compare_faces(
                    self.known_face_encodings, face_encoding, tolerance=self.tolerance
                )
                
                name = "未知"
                
                if True in matches:
                    # 找出所有匹配的人脸中距离最小的
                    face_distances = face_recognition.face_distance(
                        self.known_face_encodings, face_encoding
                    )
                    best_match_index = np.argmin(face_distances)
                    
                    if matches[best_match_index]:
                        name = self.known_face_names[best_match_index]
                else:
                    # 处理未知人脸
                    if self.save_unknown_faces:
                        self._save_unknown_face(frame, face_location)
                        
                results.append((face_location, name))
                
            return results
            
    def _save_unknown_face(self, frame: np.ndarray, face_location: Tuple[int, int, int, int]) -> None:
        """
        保存未知人脸
        
        Args:
            frame: 原始图像
            face_location: 人脸位置 (top, right, bottom, left)
        """
        try:
            # 确保目录存在
            os.makedirs(self.unknown_faces_dir, exist_ok=True)
            
            # 提取人脸区域
            top, right, bottom, left = face_location
            # 扩大一点区域，确保包含整个人脸
            top = max(0, top - 20)
            bottom = min(frame.shape[0], bottom + 20)
            left = max(0, left - 20)
            right = min(frame.shape[1], right + 20)
            
            face_image = frame[top:bottom, left:right]
            
            # 生成文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"unknown_{timestamp}_{uuid.uuid4().hex[:8]}.jpg"
            filepath = os.path.join(self.unknown_faces_dir, filename)
            
            # 保存图片
            import cv2
            cv2.imwrite(filepath, cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR))
            logging.debug(f"已保存未知人脸: {filepath}")
        except Exception as e:
            logging.error(f"保存未知人脸失败: {e}")
            
    def draw_faces(
        self, 
        frame: np.ndarray, 
        face_results: List[Tuple[Tuple[int, int, int, int], str]]
    ) -> np.ndarray:
        """
        在图像上绘制人脸框和名字
        
        Args:
            frame: 原始图像
            face_results: 人脸识别结果列表
            
        Returns:
            绘制后的图像
        """
        try:
            import cv2
            
            # 转换为PIL图像以支持中文
            frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(frame_pil)
            
            # 尝试加载字体
            font = None
            font_paths = [
                "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",  # 文泉驿微米黑
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # DejaVu Sans
                "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",    # 文泉驿正黑
                "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"  # Droid Sans
            ]
            
            for font_path in font_paths:
                try:
                    if os.path.exists(font_path):
                        font = ImageFont.truetype(font_path, 20)
                        logging.debug(f"使用字体: {font_path}")
                        break
                except Exception as e:
                    logging.warning(f"加载字体 {font_path} 失败: {e}")
                    continue
            
            # 如果所有字体都加载失败，使用默认字体
            if font is None:
                logging.warning("所有字体加载失败，使用默认字体")
                font = ImageFont.load_default()
            
            # 绘制每个人脸
            for (top, right, bottom, left), name in face_results:
                # 绘制人脸框
                draw.rectangle([(left, top), (right, bottom)], outline=(0, 255, 0), width=2)
                
                # 准备文本
                text = name if name != "未知" else "Unknown"
                
                try:
                    # 使用新的API计算文本大小
                    bbox = draw.textbbox((0, 0), text, font=font)
                    text_width = bbox[2] - bbox[0]
                    text_height = bbox[3] - bbox[1]
                    
                    # 绘制文本背景
                    draw.rectangle(
                        [(left, bottom), (left + text_width, bottom + text_height + 4)],
                        fill=(0, 255, 0)
                    )
                    
                    # 绘制文本
                    draw.text((left, bottom), text, fill=(255, 255, 255), font=font)
                except Exception as e:
                    logging.error(f"绘制文本失败: {e}")
                    # 如果文本绘制失败，至少绘制一个简单的标签
                    draw.rectangle(
                        [(left, bottom), (left + 100, bottom + 30)],
                        fill=(0, 255, 0)
                    )
                    draw.text((left + 5, bottom + 5), "Face", fill=(255, 255, 255), font=font)
            
            # 转换回OpenCV格式
            frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)
            
        except Exception as e:
            logging.error(f"绘制人脸框失败: {e}")
            
        return frame 