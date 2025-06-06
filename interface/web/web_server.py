"""
Web服务器类
实现Web界面，提供对监控系统的控制和查看
"""

import os
import time
import io
import base64
import threading
import logging
from typing import Dict, Any, Optional
from datetime import datetime

from flask import Flask, render_template, Response, request, redirect, url_for, jsonify, send_from_directory
from flask_httpauth import HTTPBasicAuth
from werkzeug.security import generate_password_hash, check_password_hash

from config.config_manager import ConfigManager
from monitor.monitor import FaceMonitor


class WebServer:
    """Web服务器类，提供Web界面访问监控系统"""
    
    def __init__(self, config: ConfigManager, monitor: FaceMonitor):
        """
        初始化Web服务器
        
        Args:
            config: 配置管理器实例
            monitor: 监控系统实例
        """
        self.config = config
        self.monitor = monitor
        
        # Flask应用
        self.app = Flask(__name__, 
            template_folder=os.path.join(os.path.dirname(__file__), "templates"),
            static_folder=os.path.join(os.path.dirname(__file__), "static")
        )
        
        # 认证
        self.auth = HTTPBasicAuth()
        self.auth_required = self.config.get('web_interface.auth_required', True)
        
        # 配置认证
        if self.auth_required:
            username = self.config.get('web_interface.credentials.username', 'admin')
            password = self.config.get('web_interface.credentials.password', 'admin123')
            self.users = {
                username: generate_password_hash(password)
            }
            
            @self.auth.verify_password
            def verify_password(username, password):
                if username in self.users and check_password_hash(self.users.get(username), password):
                    return username
                return None
                
        # 创建路由
        self._setup_routes()
        
        # 服务器参数
        self.host = self.config.get('web_interface.host', '0.0.0.0')
        self.port = self.config.get('web_interface.port', 8080)
        self.server_thread = None
        
    def _setup_routes(self) -> None:
        """设置Flask路由"""
        # 主页
        @self.app.route('/')
        @self.auth.login_required if self.auth_required else lambda x: x
        def index():
            return render_template('index.html')
            
        # 视频流
        @self.app.route('/video_feed')
        @self.auth.login_required if self.auth_required else lambda x: x
        def video_feed():
            return Response(self._generate_frames(), 
                            mimetype='multipart/x-mixed-replace; boundary=frame')
                            
        # API - 获取状态
        @self.app.route('/api/status')
        @self.auth.login_required if self.auth_required else lambda x: x
        def get_status():
            return jsonify({
                'is_active': self.monitor.is_active(),
                'detection_count': self.monitor.detection_count,
                'timestamp': datetime.now().isoformat()
            })
            
        # API - 启动监控
        @self.app.route('/api/start', methods=['POST'])
        @self.auth.login_required if self.auth_required else lambda x: x
        def start_monitor():
            success = self.monitor.start()
            return jsonify({
                'success': success,
                'message': "监控已启动" if success else "启动失败"
            })
            
        # API - 停止监控
        @self.app.route('/api/stop', methods=['POST'])
        @self.auth.login_required if self.auth_required else lambda x: x
        def stop_monitor():
            self.monitor.stop()
            return jsonify({
                'success': True,
                'message': "监控已停止"
            })
            
        # API - 添加人脸
        @self.app.route('/api/add_face', methods=['POST'])
        @self.auth.login_required if self.auth_required else lambda x: x
        def add_face():
            data = request.json
            if not data or 'name' not in data:
                return jsonify({
                    'success': False,
                    'message': "缺少人名参数"
                }), 400
                
            name = data['name']
            success = self.monitor.add_face(name)
            
            return jsonify({
                'success': success,
                'message': f"已添加 {name}" if success else f"添加 {name} 失败"
            })
            
        # 静态文件（已知人脸）
        @self.app.route('/known_faces/<path:filename>')
        @self.auth.login_required if self.auth_required else lambda x: x
        def known_faces(filename):
            known_faces_dir = self.config.get('face_recognition.known_faces_dir', './data/known_faces')
            return send_from_directory(known_faces_dir, filename)
            
        # 静态文件（检测到的图像）
        @self.app.route('/detected_images/<path:filename>')
        @self.auth.login_required if self.auth_required else lambda x: x
        def detected_images(filename):
            detected_images_dir = self.config.get('monitoring.actions.images_dir', './data/detected_images')
            return send_from_directory(detected_images_dir, filename)
            
        # API - 获取已知人脸
        @self.app.route('/api/known_faces')
        @self.auth.login_required if self.auth_required else lambda x: x
        def get_known_faces():
            known_faces_dir = self.config.get('face_recognition.known_faces_dir', './data/known_faces')
            faces = []
            
            if os.path.exists(known_faces_dir):
                for person_name in os.listdir(known_faces_dir):
                    person_dir = os.path.join(known_faces_dir, person_name)
                    if os.path.isdir(person_dir):
                        images = [f for f in os.listdir(person_dir) 
                                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                        if images:
                            faces.append({
                                'name': person_name,
                                'image': f"/known_faces/{person_name}/{images[0]}"
                            })
                            
            return jsonify(faces)
            
        # API - 获取检测到的图像
        @self.app.route('/api/detected_images')
        @self.auth.login_required if self.auth_required else lambda x: x
        def get_detected_images():
            detected_images_dir = self.config.get('monitoring.actions.images_dir', './data/detected_images')
            images = []
            
            if os.path.exists(detected_images_dir):
                for filename in os.listdir(detected_images_dir):
                    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                        file_path = os.path.join(detected_images_dir, filename)
                        stat = os.stat(file_path)
                        images.append({
                            'url': f"/detected_images/{filename}",
                            'filename': filename,
                            'timestamp': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                            'size': stat.st_size
                        })
                        
                # 按时间降序排序
                images.sort(key=lambda x: x['timestamp'], reverse=True)
                
            return jsonify(images)
        
    def _generate_frames(self):
        """生成视频流的帧序列"""
        import cv2
        
        while True:
            # 获取最新帧
            frame = self.monitor.get_latest_frame()
            
            # 如果没有帧，等待一下再试
            if frame is None:
                time.sleep(0.1)
                continue
                
            # 检测人脸
            if self.monitor.is_active():
                # 获取最新的检测结果
                marked_frame, _ = self.monitor.get_latest_detection()
                if marked_frame is not None:
                    frame = marked_frame
                    
            # 编码为JPEG
            ret, buffer = cv2.imencode('.jpg', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            frame_bytes = buffer.tobytes()
            
            # 发送到浏览器
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                   
            # 控制帧率
            time.sleep(0.03)  # 约30 FPS
            
    def start(self, debug: bool = False) -> None:
        """
        启动Web服务器
        
        Args:
            debug: 是否使用调试模式
        """
        if debug:
            # 直接运行（调试模式）
            self.app.run(host=self.host, port=self.port, debug=True)
        else:
            # 在后台线程运行
            self.server_thread = threading.Thread(
                target=lambda: self.app.run(
                    host=self.host, 
                    port=self.port, 
                    debug=False,
                    use_reloader=False
                )
            )
            self.server_thread.daemon = True
            self.server_thread.start()
            logging.info(f"Web服务器已启动，地址: http://{self.host}:{self.port}")
            
    def stop(self) -> None:
        """停止Web服务器"""
        # Flask没有优雅的停止方法，我们在多线程模式下使用的是守护线程
        # 当主程序退出时，线程将自动终止
        logging.info("Web服务器已停止") 