"""
人脸识别类
提供人脸检测和识别功能
"""

import os
import time
import uuid
import cv2
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

from utils.file_writer import FileWriter


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
        file_writer: Optional[FileWriter] = None
    ):
        """
        初始化人脸识别器
        
        Args:
            known_faces_dir: 已知人脸数据库目录
            model: 人脸检测模型, 可选 'hog' (CPU) 或 'cnn' (GPU), 主要用于 'add_face' 等高精度场景。实时检测将使用速度更快的 Haar Cascade。
            tolerance: 识别容差，值越小越严格
            detection_fps: 检测帧率
            save_unknown_faces: 是否保存未知人脸
            unknown_faces_dir: 未知人脸保存目录
            file_writer: 异步文件写入器实例
        """
        self.known_faces_dir = known_faces_dir
        self.model = model
        self.tolerance = tolerance
        self.detection_interval = 1.0 / detection_fps if detection_fps > 0 else 0
        self.save_unknown_faces = save_unknown_faces
        self.unknown_faces_dir = unknown_faces_dir or "./data/unknown_faces"
        self.file_writer = file_writer
        
        # 已知人脸数据
        self.known_face_encodings = []
        self.known_face_names = []
        
        # 线程安全锁
        self.detection_lock = threading.Lock()
        
        # 上次检测时间
        self.last_detection_time = 0
        
        # 字体
        self.font = None
        
        # Haar-cascade 分类器
        self.haar_cascade = None
        
        # 检查依赖
        if not FACE_RECOGNITION_AVAILABLE:
            logging.error("face_recognition 库未安装，无法进行人脸识别")
            
        # 创建必要的目录
        os.makedirs(self.known_faces_dir, exist_ok=True)
        if self.save_unknown_faces:
            os.makedirs(self.unknown_faces_dir, exist_ok=True)
        
        # 加载 Haar 分类器
        self._load_haar_cascade()
        
        # 加载已知人脸
        self.load_known_faces()
        
        # 加载字体
        self._load_font()
        
    def _load_font(self, size: int = 20) -> None:
        """加载用于绘制文本的字体文件"""
        font_paths = [
            "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",  # 文泉驿微米黑
            "C:/Windows/Fonts/msyh.ttc",                      # 微软雅黑 (Windows)
            "/System/Library/Fonts/STHeitiLight.ttc",          # 黑体-简 (macOS)
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # DejaVu Sans
            "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",    # 文泉驿正黑
            "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"  # Droid Sans
        ]
        
        for font_path in font_paths:
            try:
                if os.path.exists(font_path):
                    self.font = ImageFont.truetype(font_path, size)
                    logging.debug(f"使用字体: {font_path}")
                    return
            except Exception as e:
                logging.warning(f"加载字体 {font_path} 失败: {e}")
                continue
        
        # 如果所有字体都加载失败，使用默认字体
        logging.warning("所有推荐字体加载失败，使用默认字体")
        self.font = ImageFont.load_default()

    def _load_haar_cascade(self) -> None:
        """加载 OpenCV Haar 级联分类器模型"""
        try:
            # cv2.data.haarcascades 提供了预训练模型文件的路径
            haar_xml_file = 'haarcascade_frontalface_default.xml'
            cascade_path = os.path.join(cv2.data.haarcascades, haar_xml_file)
            
            if not os.path.exists(cascade_path):
                logging.error(f"Haar cascade 文件不存在: {cascade_path}")
                logging.error("请确保已正确安装 OpenCV (opencv-python) 库。")
                self.haar_cascade = None
                return

            self.haar_cascade = cv2.CascadeClassifier(cascade_path)
            if self.haar_cascade.empty():
                logging.error(f"加载 Haar cascade 文件失败: {cascade_path}")
                self.haar_cascade = None
            else:
                logging.info(f"成功加载 Haar cascade 分类器。")
        except Exception as e:
            logging.error(f"加载 Haar cascade 时出错: {e}")
            self.haar_cascade = None

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
        if not FACE_RECOGNITION_AVAILABLE or self.haar_cascade is None:
            return []
            
        with self.detection_lock:
            # 使用 Haar Cascade 替换 HOG 进行快速检测
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            faces = self.haar_cascade.detectMultiScale(
                gray_frame,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(60, 60),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            # 转换坐标格式从 (x, y, w, h) 到 (top, right, bottom, left)
            face_locations = [(y, x + w, y + h, x) for (x, y, w, h) in faces]
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
        if not FACE_RECOGNITION_AVAILABLE or not self.need_detection() or self.haar_cascade is None:
            return []
            
        with self.detection_lock:
            # 1. 使用 Haar Cascade 替换 HOG 进行快速人脸位置检测
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            faces = self.haar_cascade.detectMultiScale(
                gray_frame,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(60, 60),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            # 转换坐标格式从 (x, y, w, h) 到 (top, right, bottom, left)
            face_locations = [(y, x + w, y + h, x) for (x, y, w, h) in faces]
            
            if not face_locations:
                return []
                
            # 2. 对检测到的人脸提取特征 (这一步仍然是计算密集型)
            face_encodings = face_recognition.face_encodings(frame, face_locations)
            
            # 3. 进行人脸比对
            results = []
            for i, face_encoding in enumerate(face_encodings):
                # 获取当前人脸的位置
                face_location = face_locations[i]
                
                # 与已知人脸比较
                matches = face_recognition.compare_faces(
                    self.known_face_encodings, face_encoding, self.tolerance
                )
                name = "Unknown"
                
                # 如果找到匹配
                if True in matches:
                    first_match_index = matches.index(True)
                    name = self.known_face_names[first_match_index]
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
            
            # 异步保存图片
            if self.file_writer:
                # 将图像从RGB转换为BGR以供cv2.imwrite使用
                face_image_bgr = cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR)
                self.file_writer.save(face_image_bgr, filepath)
                logging.debug(f"已提交未知人脸保存任务: {filepath}")
            else:
                # 如果没有提供写入器，则同步回退
                import cv2
                cv2.imwrite(filepath, cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR))
                logging.warning(f"文件写入器未提供，同步保存未知人脸: {filepath}")
        except Exception as e:
            logging.error(f"保存未知人脸任务提交失败: {e}")
            
    def draw_faces(
        self, 
        frame: np.ndarray, 
        face_results: List[Tuple[Tuple[int, int, int, int], str]]
    ) -> np.ndarray:
        """
        在图像上绘制人脸框和名字（优化版）
        该方法通过仅在需要绘制文本的局部区域使用PIL，避免了对整个帧进行格式转换，从而提高性能。
        
        Args:
            frame: 原始图像 (OpenCV BGR格式)
            face_results: 人脸识别结果列表
            
        Returns:
            绘制后的图像 (OpenCV BGR格式)
        """
        try:
            import cv2
            
            # 如果没有加载字体或没有识别结果，直接返回原图
            if self.font is None or not face_results:
                return frame

            # 绘制每个人脸
            for (top, right, bottom, left), name in face_results:
                # 1. 直接在OpenCV帧上绘制矩形框（非常快）
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                
                # 2. 准备文本和背景
                text = name if name != "未知" else "Unknown"
                
                # 3. 使用PIL在内存中创建文本图像（只处理小尺寸的文本区域）
                try:
                    # 获取文本尺寸
                    bbox = self.font.getbbox(text)
                    text_width = bbox[2] - bbox[0]
                    text_height = bbox[3] - bbox[1]
                    
                    # 创建文本背景PIL图像
                    text_bg_pil = Image.new("RGB", (text_width + 4, text_height + 4), (0, 255, 0))
                    draw = ImageDraw.Draw(text_bg_pil)
                    # 在PIL图像上绘制白色文本
                    draw.text((2, 2), text, font=self.font, fill=(255, 255, 255))
                    
                    # 4. 将PIL文本图像转换回OpenCV格式
                    text_bg_cv = cv2.cvtColor(np.array(text_bg_pil), cv2.COLOR_RGB2BGR)
                    
                    # 5. 计算文本在主图像上的粘贴位置
                    y1 = bottom
                    y2 = bottom + text_bg_cv.shape[0]
                    x1 = left
                    x2 = left + text_bg_cv.shape[1]
                    
                    # 确保粘贴区域不超出主图像边界
                    if y2 > frame.shape[0]:
                        y2 = frame.shape[0]
                        text_bg_cv = text_bg_cv[:y2-y1, :]
                    if x2 > frame.shape[1]:
                        x2 = frame.shape[1]
                        text_bg_cv = text_bg_cv[:, :x2-x1]

                    # 6. 将文本图像"粘贴"到主图像上
                    frame[y1:y2, x1:x2] = text_bg_cv

                except Exception as e:
                    logging.error(f"绘制文本 '{text}' 失败: {e}")
            
        except Exception as e:
            logging.error(f"绘制人脸框失败: {e}")
            
        return frame 