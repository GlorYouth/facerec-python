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
        """
        加载已知人脸数据库, 并在加载过程中自动清理无效的人脸图片和空的文件夹。
        """
        self.known_face_encodings = []
        self.known_face_names = []
        
        if not os.path.exists(self.known_faces_dir):
            logging.warning(f"已知人脸目录不存在: {self.known_faces_dir}")
            return
            
        logging.info(f"正在加载已知人脸数据库 (带清理功能): {self.known_faces_dir}")
        
        # 使用 list() 来复制列表，因为我们可能会在循环中修改目录内容
        for person_name in list(os.listdir(self.known_faces_dir)):
            person_dir = os.path.join(self.known_faces_dir, person_name)
            
            if not os.path.isdir(person_dir):
                continue

            image_files = os.listdir(person_dir)
            if not image_files:
                logging.info(f"发现空的人脸目录，正在清理: {person_dir}")
                try:
                    os.rmdir(person_dir)
                except OSError as e:
                    logging.error(f"清理空目录 {person_dir} 失败: {e}")
                continue

            # 遍历每个人的图片
            for image_file in list(image_files): # 同样使用副本迭代
                image_path = os.path.join(person_dir, image_file)
                
                if not image_path.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue
                    
                try:
                    face_image = face_recognition.load_image_file(image_path)
                    face_encodings = face_recognition.face_encodings(face_image)
                    
                    if face_encodings:
                        self.known_face_encodings.append(face_encodings[0])
                        self.known_face_names.append(person_name)
                        logging.debug(f"已加载人脸: {person_name} ({image_file})")
                    else:
                        logging.warning(f"图片中未检测到人脸，将自动清理: {image_path}")
                        os.remove(image_path)

                except Exception as e:
                    logging.error(f"加载人脸失败，将自动清理: {image_path} ({e})")
                    try:
                        os.remove(image_path)
                    except OSError as remove_error:
                        logging.error(f"清理失败的文件 {image_path} 失败: {remove_error}")

            # 在处理完一个人的所有图片后，再次检查目录是否变空
            try:
                if not os.listdir(person_dir):
                    logging.info(f"目录中所有图片均无效，清理空目录: {person_dir}")
                    os.rmdir(person_dir)
            except FileNotFoundError:
                # 目录可能已经被初始检查删除了，这是正常情况
                pass
            except OSError as e:
                logging.error(f"清理空目录 {person_dir} 失败: {e}")

        logging.info(f"清理完成。已加载 {len(self.known_face_encodings)} 个有效已知人脸")
        
    def add_known_face_from_frame(self, frame: np.ndarray, person_name: str) -> bool:
        """
        以异步方式，从一个完整的图像帧中检测、裁剪并添加新的人脸到数据库。
        """
        # 立即返回True，实际处理在后台进行
        thread = threading.Thread(target=self._add_known_face_worker, args=(frame, person_name))
        thread.daemon = True
        thread.start()
        return True

    def _add_known_face_worker(self, frame: np.ndarray, person_name: str) -> None:
        """
        在后台线程中执行添加人脸的实际工作。
        """
        if not FACE_RECOGNITION_AVAILABLE:
            logging.error("face_recognition 库不可用，无法添加人脸。")
            return

        try:
            # 使用更精确的模型来定位人脸以进行保存
            face_locations = face_recognition.face_locations(frame, model=self.model)
            
            if not face_locations:
                logging.warning("在当前帧中未检测到人脸，无法添加。")
                return

            # 如果有多张人脸，选择最大的一张
            largest_face_location = None
            max_area = 0
            for (top, right, bottom, left) in face_locations:
                area = (bottom - top) * (right - left)
                if area > max_area:
                    max_area = area
                    largest_face_location = (top, right, bottom, left)
            
            # 增加一个最小尺寸判断，避免保存过小的人脸
            if max_area < 40*40: # 假设最小人脸为40x40像素
                logging.warning(f"人脸尺寸过小 ({max_area}px)，无法添加。请离摄像头近一点。")
                return

            (top, right, bottom, left) = largest_face_location
            face_image_crop = frame[top:bottom, left:right]

            # 创建人名目录
            person_dir = os.path.join(self.known_faces_dir, person_name)
            os.makedirs(person_dir, exist_ok=True)
            
            # 生成唯一文件名
            image_filename = f"{person_name}_{uuid.uuid4().hex[:8]}.jpg"
            image_path = os.path.join(person_dir, image_filename)
            
            # 保存裁剪出的人脸图片
            Image.fromarray(face_image_crop).save(image_path)
            
            logging.info(f"成功保存新的人脸图像: {image_path}")
            
            # 立即重新加载人脸库，使新添加的人脸生效
            self.load_known_faces()
            
        except Exception as e:
            logging.error(f"添加人脸时发生错误: {e}", exc_info=True)

    def delete_known_face(self, person_name: str) -> bool:
        """
        从数据库中删除一个已知人脸（及其所有图片）。

        :param person_name: 要删除的人名
        :return: 是否成功删除
        """
        person_dir = os.path.join(self.known_faces_dir, person_name)
        
        if not os.path.exists(person_dir):
            logging.warning(f"要删除的人脸目录不存在: {person_dir}")
            return False
            
        try:
            import shutil
            shutil.rmtree(person_dir)
            logging.info(f"已成功删除人脸目录: {person_dir}")
            
            # 立即重新加载人脸库
            self.load_known_faces()
            return True
        except Exception as e:
            logging.error(f"删除人脸 '{person_name}' 时出错: {e}", exc_info=True)
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

        # 降采样以提高性能
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        gray_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        
        # 使用Haar Cascade检测人脸
        face_locations = self.haar_cascade.detectMultiScale(
            gray_frame,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        # 将坐标转换回原始尺寸
        face_locations_orig = []
        for (x, y, w, h) in face_locations:
            top, right, bottom, left = y * 2, (x + w) * 2, (y + h) * 2, x * 2
            face_locations_orig.append((top, right, bottom, left))
            
        # 目前只检测，不识别，所以名字都是 "Unknown"
        return [(loc, "Face") for loc in face_locations_orig]

    def recognize_faces(
        self, frame: np.ndarray
    ) -> List[Tuple[Tuple[int, int, int, int], str]]:
        """
        检测并识别图像中的人脸
        
        Args:
            frame: 输入图像
            
        Returns:
            人脸位置和姓名列表
        """
        if not FACE_RECOGNITION_AVAILABLE or not self.known_face_encodings:
            return []
            
        # 将图像从BGR转换为RGB（face_recognition使用RGB）
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 检测所有人脸的位置和编码
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
        face_results = []
        for face_location, face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding, self.tolerance)
            name = "Unknown"
            
            # 如果找到匹配项，使用第一个
            if True in matches:
                first_match_index = matches.index(True)
                name = self.known_face_names[first_match_index]
            elif self.save_unknown_faces:
                # 异步保存未知人脸
                self._save_unknown_face(frame, face_location)

            face_results.append((face_location, name))
            
        return face_results

    def _save_unknown_face(self, frame: np.ndarray, face_location: Tuple[int, int, int, int]) -> None:
        """异步保存未知人脸的图像"""
        if not self.file_writer:
            return
            
        try:
            top, right, bottom, left = face_location
            face_image = frame[top:bottom, left:right]
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"unknown_{timestamp}.jpg"
            filepath = os.path.join(self.unknown_faces_dir, filename)
            
            # 使用File Writer进行异步保存
            self.file_writer.save(face_image, filepath)
            
        except Exception as e:
            logging.error(f"保存未知人脸失败: {e}")

    def draw_faces(
        self, 
        frame: np.ndarray, 
        face_results: List[Tuple[Tuple[int, int, int, int], str]]
    ) -> np.ndarray:
        """
        在图像上绘制人脸框和姓名
        
        Args:
            frame: 输入图像
            face_results: 人脸检测结果
            
        Returns:
            绘制了标记的图像
        """
        # 使用Pillow进行绘制以支持中文
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        
        for (top, right, bottom, left), name in face_results:
            # 绘制人脸框
            draw.rectangle(((left, top), (right, bottom)), outline=(0, 255, 0), width=2)
            
            # 准备绘制文本
            try:
                text_bbox = draw.textbbox((left, bottom), name, font=self.font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
                
                # 创建文本背景
                draw.rectangle(
                    ((left, bottom - text_height - 5), (left + text_width + 5, bottom)),
                    fill=(0, 255, 0)
                )
                
                # 绘制文本
                draw.text((left + 2, bottom - text_height - 3), name, font=self.font, fill=(255, 255, 255))
            except Exception:
                # Pillow绘制失败时回退到OpenCV
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
                cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)
        
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)