"""
人脸跟踪类
提供人脸跟踪功能，用于在视频帧序列中跟踪同一个人脸
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import time
import logging


class FaceTracker:
    """人脸跟踪类，在连续视频帧中跟踪同一个人脸"""
    
    def __init__(
        self,
        max_disappeared: int = 50,
        min_distance: float = 0.6,
        min_iou: float = 0.3,
        overlap_threshold: float = 0.5,
        smoothing_factor: float = 0.3,
        min_detection_area: int = 1000,
        unknown_face_threshold: float = 0.5,  # 未知人脸特征相似度阈值
        max_retention_time: float = 1.0  # 跟踪框最大暂留时间(秒)，默认1秒
    ):
        """
        初始化人脸跟踪器
        
        Args:
            max_disappeared: 最大连续消失帧数，超过此值将删除跟踪对象
            min_distance: 特征向量最小距离阈值，小于此值认为是同一个人
            min_iou: 最小IOU阈值，小于此值不考虑合并检测框
            overlap_threshold: 重叠框过滤阈值，两个框的IOU大于此值时视为重叠
            smoothing_factor: 跟踪结果平滑因子，0表示不平滑，1表示完全使用历史位置
            min_detection_area: 最小检测框面积，小于此值的检测框将被忽略
            unknown_face_threshold: 未知人脸特征相似度阈值，小于此值认为是同一个人
            max_retention_time: 跟踪框最大暂留时间(秒)，超过此时间无更新将被删除
        """
        self.max_disappeared = max_disappeared
        self.min_distance = min_distance
        self.min_iou = min_iou
        self.overlap_threshold = overlap_threshold
        self.smoothing_factor = smoothing_factor
        self.min_detection_area = min_detection_area
        self.unknown_face_threshold = unknown_face_threshold
        self.max_retention_time = max_retention_time
        
        # 下一个可用的对象ID
        self.next_object_id = 0
        
        # 跟踪的人脸对象字典 {object_id: {bbox, name, encoding, disappeared_count, last_time, age}}
        self.tracked_faces = {}
        
        # 记录上一次的检测框，用于非匹配检测的处理
        self.previous_detections = []
        
        # 未知人脸特征库 {unknown_id: List[encoding]}
        self.unknown_face_features = {}
        
        # 未知人脸ID计数器
        self.next_unknown_id = 1
        
        logging.info("人脸跟踪器已初始化")
    
    def _calculate_iou(self, boxA: Tuple, boxB: Tuple) -> float:
        """
        计算两个框的IOU (Intersection over Union)
        
        Args:
            boxA: 第一个框的坐标 (top, right, bottom, left)
            boxB: 第二个框的坐标 (top, right, bottom, left)
            
        Returns:
            IOU值
        """
        # 转换为 (y1, x1, y2, x2) 格式
        boxA = (boxA[0], boxA[3], boxA[2], boxA[1])
        boxB = (boxB[0], boxB[3], boxB[2], boxB[1])
        
        # 确定相交矩形的坐标
        y1 = max(boxA[0], boxB[0])
        x1 = max(boxA[1], boxB[1])
        y2 = min(boxA[2], boxB[2])
        x2 = min(boxA[3], boxB[3])
        
        # 计算相交区域的面积
        interArea = max(0, x2 - x1) * max(0, y2 - y1)
        
        # 计算两个框的面积
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        
        # 计算IOU
        iou = interArea / float(boxAArea + boxBArea - interArea)
        
        return iou
    
    def _calculate_center_distance(self, boxA: Tuple, boxB: Tuple) -> float:
        """
        计算两个框中心点的欧氏距离
        
        Args:
            boxA: 第一个框的坐标 (top, right, bottom, left)
            boxB: 第二个框的坐标 (top, right, bottom, left)
            
        Returns:
            中心点距离
        """
        # 计算第一个框的中心点
        centerA = (
            (boxA[0] + boxA[2]) / 2,  # y
            (boxA[3] + boxA[1]) / 2   # x
        )
        
        # 计算第二个框的中心点
        centerB = (
            (boxB[0] + boxB[2]) / 2,  # y
            (boxB[3] + boxB[1]) / 2   # x
        )
        
        # 计算欧氏距离
        distance = np.sqrt((centerA[0] - centerB[0]) ** 2 + (centerA[1] - centerB[1]) ** 2)
        
        return distance
    
    def _calculate_feature_distance(self, encodingA: np.ndarray, encodingB: np.ndarray) -> float:
        """
        计算两个特征向量的欧氏距离
        
        Args:
            encodingA: 第一个特征向量
            encodingB: 第二个特征向量
            
        Returns:
            特征向量距离
        """
        return np.linalg.norm(encodingA - encodingB)
    
    def _calculate_bbox_area(self, bbox: Tuple) -> float:
        """
        计算边界框的面积
        
        Args:
            bbox: 边界框坐标 (top, right, bottom, left)
            
        Returns:
            面积
        """
        top, right, bottom, left = bbox
        height = bottom - top
        width = right - left
        return height * width
    
    def _detect_overlapping_boxes(self, detections: List[Tuple[Tuple[int, int, int, int], str, np.ndarray]]) -> List[int]:
        """
        检测重叠的检测框，并返回应该被忽略的检测框索引
        
        Args:
            detections: 检测结果列表，每个元素为 (bbox, name, encoding) 元组
            
        Returns:
            应该被忽略的检测索引列表
        """
        if len(detections) <= 1:
            return []
            
        ignored_indices = []
        for i in range(len(detections)):
            if i in ignored_indices:
                continue
                
            for j in range(i + 1, len(detections)):
                if j in ignored_indices:
                    continue
                    
                # 计算两个框的IOU
                bbox_i = detections[i][0]
                bbox_j = detections[j][0]
                
                iou = self._calculate_iou(bbox_i, bbox_j)
                
                # 如果IOU大于阈值，说明有重叠
                if iou > self.overlap_threshold:
                    # 保留面积较大的框，忽略较小的框
                    area_i = self._calculate_bbox_area(bbox_i)
                    area_j = self._calculate_bbox_area(bbox_j)
                    
                    if area_i >= area_j:
                        ignored_indices.append(j)
                    else:
                        ignored_indices.append(i)
                        break  # 如果i被忽略，就不再比较i和其他框
                        
        return ignored_indices
    
    def _smooth_bbox(self, track_id: int, new_bbox: Tuple) -> Tuple:
        """
        平滑边界框位置，减少抖动
        
        Args:
            track_id: 跟踪对象ID
            new_bbox: 新的边界框位置 (top, right, bottom, left)
            
        Returns:
            平滑后的边界框位置
        """
        if self.smoothing_factor <= 0:
            return new_bbox
            
        # 如果没有历史位置，直接返回新位置
        if 'bbox_history' not in self.tracked_faces[track_id]:
            self.tracked_faces[track_id]['bbox_history'] = new_bbox
            return new_bbox
            
        # 获取历史位置
        old_bbox = self.tracked_faces[track_id]['bbox_history']
        
        # 线性插值
        top = int(old_bbox[0] * self.smoothing_factor + new_bbox[0] * (1 - self.smoothing_factor))
        right = int(old_bbox[1] * self.smoothing_factor + new_bbox[1] * (1 - self.smoothing_factor))
        bottom = int(old_bbox[2] * self.smoothing_factor + new_bbox[2] * (1 - self.smoothing_factor))
        left = int(old_bbox[3] * self.smoothing_factor + new_bbox[3] * (1 - self.smoothing_factor))
        
        smoothed_bbox = (top, right, bottom, left)
        
        # 更新历史位置
        self.tracked_faces[track_id]['bbox_history'] = smoothed_bbox
        
        return smoothed_bbox
    
    def _associate_detections_to_tracks(
        self, 
        detections: List[Tuple[Tuple[int, int, int, int], str, np.ndarray]]
    ) -> Tuple[Dict[int, int], List[int]]:
        """
        将当前帧的检测结果与已跟踪的对象关联
        
        Args:
            detections: 检测结果列表，每个元素为 (bbox, name, encoding) 元组
            
        Returns:
            matched_tracks: 匹配的跟踪ID和检测索引的字典 {track_id: detection_idx}
            unmatched_detections: 未匹配的检测索引列表
        """
        # 如果没有已跟踪的对象，则所有检测都是未匹配的
        if len(self.tracked_faces) == 0:
            return {}, list(range(len(detections)))
        
        # 如果没有检测结果，则所有跟踪对象都是未匹配的
        if len(detections) == 0:
            return {}, []
        
        # 首先检测并移除重叠的检测框
        ignored_detections = self._detect_overlapping_boxes(detections)
        valid_detections = [i for i in range(len(detections)) if i not in ignored_detections]
        
        # 如果没有有效的检测，则所有跟踪对象都是未匹配的
        if not valid_detections:
            return {}, []
        
        matched_tracks = {}
        unmatched_detections = valid_detections.copy()
        
        # 计算所有检测框与跟踪框之间的距离矩阵
        iou_matrix = np.zeros((len(self.tracked_faces), len(valid_detections)))
        distance_matrix = np.zeros((len(self.tracked_faces), len(valid_detections)))
        feature_matrix = np.zeros((len(self.tracked_faces), len(valid_detections)))
        
        # 计算各种距离度量
        for i, (track_id, track_info) in enumerate(self.tracked_faces.items()):
            for j, det_idx in enumerate(valid_detections):
                det_bbox, _, det_encoding = detections[det_idx]
                
                # 计算IOU
                iou = self._calculate_iou(track_info['bbox'], det_bbox)
                iou_matrix[i, j] = iou
                
                # 计算中心点距离
                distance = self._calculate_center_distance(track_info['bbox'], det_bbox)
                distance_matrix[i, j] = distance
                
                # 计算特征向量距离
                if track_info.get('encoding') is not None and det_encoding is not None:
                    feature_distance = self._calculate_feature_distance(track_info['encoding'], det_encoding)
                    feature_matrix[i, j] = feature_distance
                else:
                    feature_matrix[i, j] = float('inf')
        
        # 匹配算法改进：两阶段匹配
        # 第一阶段：优先匹配有特征的老跟踪目标
        track_indices = list(range(len(self.tracked_faces)))
        track_ids = list(self.tracked_faces.keys())
        
        # 按照跟踪目标的年龄（存活时间）排序，优先匹配老的跟踪目标
        age_sorted_indices = sorted(
            track_indices,
            key=lambda i: self.tracked_faces[track_ids[i]].get('age', 0),
            reverse=True
        )
        
        # 第一阶段：匹配特征明显的对象
        for i in age_sorted_indices:
            if not unmatched_detections:
                break
                
            track_id = track_ids[i]
            track_info = self.tracked_faces[track_id]
            
            # 只处理有特征编码的跟踪目标
            if track_info.get('encoding') is None:
                continue
                
            best_match_idx = None
            best_score = float('-inf')
            
            for j, det_idx in enumerate(unmatched_detections):
                # 计算综合评分
                iou_score = iou_matrix[i, j]
                feature_score = 1.0 - min(1.0, feature_matrix[i, j] / self.min_distance)
                
                # 只有当特征和IOU都满足条件时才考虑匹配
                if feature_matrix[i, j] <= self.min_distance and iou_score >= self.min_iou:
                    score = 0.3 * iou_score + 0.7 * feature_score
                    
                    if score > best_score:
                        best_score = score
                        best_match_idx = j
            
            # 如果找到了最佳匹配
            if best_match_idx is not None:
                det_idx = unmatched_detections[best_match_idx]
                matched_tracks[track_id] = det_idx
                unmatched_detections.remove(det_idx)
        
        # 第二阶段：对剩余的跟踪目标，基于位置信息进行匹配
        remaining_track_indices = [i for i in track_indices if track_ids[i] not in matched_tracks]
        
        for i in remaining_track_indices:
            if not unmatched_detections:
                break
                
            track_id = track_ids[i]
            
            best_match_idx = None
            best_score = float('-inf')
            
            for j, det_idx in enumerate(unmatched_detections):
                # 计算位置评分
                iou_score = iou_matrix[i, j]
                distance_score = 1.0 / (1.0 + distance_matrix[i, j] / 100.0)
                
                # 如果IOU足够高，考虑匹配
                if iou_score >= self.min_iou:
                    score = 0.7 * iou_score + 0.3 * distance_score
                    
                    if score > best_score:
                        best_score = score
                        best_match_idx = j
            
            # 如果找到了最佳匹配
            if best_match_idx is not None:
                det_idx = unmatched_detections[best_match_idx]
                matched_tracks[track_id] = det_idx
                unmatched_detections.remove(det_idx)
        
        # 将ignored_detections添加到unmatched_detections中，确保它们被考虑为新的检测对象
        unmatched_detections.extend(ignored_detections)
        
        return matched_tracks, unmatched_detections
    
    def _find_similar_unknown_face(self, encoding: np.ndarray) -> Optional[str]:
        """
        在未知人脸特征库中查找相似的人脸
        
        Args:
            encoding: 人脸特征向量
            
        Returns:
            如果找到相似的未知人脸，返回其ID（格式为"Unknown_N"），否则返回None
        """
        if encoding is None:
            return None
            
        for unknown_id, feature_list in self.unknown_face_features.items():
            # 计算与该未知ID下所有特征向量的平均距离
            distances = [self._calculate_feature_distance(encoding, feat) for feat in feature_list]
            avg_distance = np.mean(distances)
            
            # 如果平均距离小于阈值，认为是同一个人
            if avg_distance < self.unknown_face_threshold:
                # 将新的特征向量添加到特征库中（最多保留10个特征）
                if len(feature_list) < 10:
                    feature_list.append(encoding)
                return unknown_id
                
        return None
        
    def _create_new_unknown_id(self, encoding: np.ndarray) -> str:
        """
        创建新的未知人脸ID并保存其特征
        
        Args:
            encoding: 人脸特征向量
            
        Returns:
            新的未知人脸ID（格式为"Unknown_N"）
        """
        unknown_id = f"Unknown_{self.next_unknown_id}"
        self.unknown_face_features[unknown_id] = [encoding] if encoding is not None else []
        self.next_unknown_id += 1
        return unknown_id
        
    def _should_create_new_track(
        self, 
        detection: Tuple[Tuple[int, int, int, int], str, Optional[np.ndarray]]
    ) -> Tuple[bool, Optional[str]]:
        """
        决定是否为未匹配的检测创建新的跟踪目标，并处理未知人脸的ID分配
        
        Args:
            detection: 检测结果 (bbox, name, encoding)
            
        Returns:
            (是否应该创建新的跟踪目标, 建议使用的名称)
        """
        bbox, name, encoding = detection
        
        # 检查与现有跟踪目标的IOU，防止在同一位置创建多个跟踪目标
        for track_info in self.tracked_faces.values():
            iou = self._calculate_iou(bbox, track_info['bbox'])
            if iou > 0.4:  # 使用较低的阈值
                return False, None
        
        # 检查与前一帧所有检测的IOU
        for prev_bbox, _, _ in self.previous_detections:
            iou = self._calculate_iou(bbox, prev_bbox)
            if iou > 0.4:  # 使用较低的阈值
                return True, name
        
        # 计算检测框大小
        area = self._calculate_bbox_area(bbox)
        
        # 太小的检测框可能是噪声
        if area < self.min_detection_area:
            return False, None
            
        # 处理未知人脸的ID分配
        if name == "Unknown" and encoding is not None:
            # 查找相似的未知人脸
            similar_unknown_id = self._find_similar_unknown_face(encoding)
            if similar_unknown_id is not None:
                # 使用已有的未知人脸ID
                return True, similar_unknown_id
            else:
                # 创建新的未知人脸ID
                new_unknown_id = self._create_new_unknown_id(encoding)
                return True, new_unknown_id
        
        # 对于已知人脸或无编码的未知人脸，保持原有名称
        return True, name
    
    def update(
        self, 
        detections: List[Tuple[Tuple[int, int, int, int], str, Optional[np.ndarray]]]
    ) -> Dict[int, Dict]:
        """
        更新跟踪状态
        
        Args:
            detections: 检测结果列表，每个元素为 (bbox, name, encoding) 元组
            
        Returns:
            跟踪结果字典 {object_id: {bbox, name, ...}}
        """
        # 记录当前时间
        current_time = time.time()
        
        # 关联检测结果和跟踪对象
        matched_tracks, unmatched_detections = self._associate_detections_to_tracks(detections)
        
        # 更新已匹配的跟踪对象
        for track_id, det_idx in matched_tracks.items():
            det_bbox, det_name, det_encoding = detections[det_idx]
            
            # 平滑边界框位置
            smoothed_bbox = self._smooth_bbox(track_id, det_bbox)
            
            # 如果是未知人脸，尝试更新ID
            if det_name == "Unknown" and det_encoding is not None:
                similar_unknown_id = self._find_similar_unknown_face(det_encoding)
                if similar_unknown_id is not None:
                    det_name = similar_unknown_id
            
            # 更新跟踪对象信息
            self.tracked_faces[track_id]['bbox'] = smoothed_bbox
            self.tracked_faces[track_id]['name'] = det_name
            if det_encoding is not None:
                self.tracked_faces[track_id]['encoding'] = det_encoding
            self.tracked_faces[track_id]['disappeared_count'] = 0
            self.tracked_faces[track_id]['last_time'] = current_time
            
            # 增加年龄（存活帧数）
            self.tracked_faces[track_id]['age'] = self.tracked_faces[track_id].get('age', 0) + 1
        
        # 所有未匹配的跟踪对象，增加消失计数
        for track_id in self.tracked_faces.keys():
            if track_id not in matched_tracks:
                self.tracked_faces[track_id]['disappeared_count'] += 1
        
        # 删除消失太久或超过最大暂留时间的跟踪对象
        self.tracked_faces = {
            track_id: track_info 
            for track_id, track_info in self.tracked_faces.items() 
            if (track_info['disappeared_count'] <= self.max_disappeared and
                (current_time - track_info['last_time']) <= self.max_retention_time)
        }
        
        # 添加新的检测对象，但有条件地
        for det_idx in unmatched_detections:
            detection = detections[det_idx]
            
            # 判断是否应该创建新的跟踪目标，并获取建议的名称
            should_create, suggested_name = self._should_create_new_track(detection)
            
            if should_create:
                det_bbox, _, det_encoding = detection  # 使用 suggested_name 替代原始名称
                
                # 创建新的跟踪对象
                self.tracked_faces[self.next_object_id] = {
                    'bbox': det_bbox,
                    'bbox_history': det_bbox,  # 初始化历史位置
                    'name': suggested_name,  # 使用建议的名称
                    'encoding': det_encoding,
                    'disappeared_count': 0,
                    'last_time': current_time,
                    'age': 0  # 初始年龄为0
                }
                
                # 更新下一个可用ID
                self.next_object_id += 1
        
        # 更新前一帧的检测结果，用于下一次更新
        self.previous_detections = detections
        
        # 返回当前跟踪的人脸
        return self.tracked_faces