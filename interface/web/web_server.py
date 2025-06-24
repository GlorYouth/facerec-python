"""
Web服务器类
实现Web界面，提供对监控系统的控制和查看
"""

import os
import time
import io
import threading
import logging
from typing import Dict, Any, Optional
from datetime import datetime

import cv2
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
        
        self.app = Flask(__name__, 
            template_folder=os.path.join(os.path.dirname(__file__), "templates"),
            static_folder=os.path.join(os.path.dirname(__file__), "static")
        )
        
        self.auth = HTTPBasicAuth()
        self.auth_required = self.config.get('web_interface.auth_required', True)
        
        if self.auth_required:
            self._setup_auth()
            
        self._setup_routes()
        
        self.host = self.config.get('web_interface.host', '0.0.0.0')
        self.port = self.config.get('web_interface.port', 8080)
        self.server_thread = None
        
        self.streaming_fps = self.config.get('web_interface.streaming_fps', 10)
        self.streaming_interval = 1.0 / self.streaming_fps if self.streaming_fps > 0 else 0.1
        self.streaming_thread = None
        self.is_streaming = False
        self.latest_jpeg_frame: Optional[bytes] = None
        self.jpeg_frame_lock = threading.Lock()

    def _setup_auth(self):
        """配置HTTP基础认证"""
        username = self.config.get('web_interface.credentials.username', 'admin')
        password = self.config.get('web_interface.credentials.password', 'admin123')
        self.users = {
            username: generate_password_hash(password)
        }
        
        @self.auth.verify_password
        def verify_password(username, password):
            if username in self.users and check_password_hash(self.users.get(username, ""), password):
                return username
            return None

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
                'detection_count': self.monitor.get_detection_count(),
                'timestamp': datetime.now().isoformat()
            })

        # API - 获取最近识别的人脸
        @self.app.route('/api/recent_faces')
        @self.auth.login_required if self.auth_required else lambda x: x
        def get_recent_faces():
            if hasattr(self.monitor, 'recorder'):
                status = self.monitor.recorder.get_status()
                recent_faces = [{'name': name, 'last_seen': ts} for name, ts in status.items()]
                recent_faces.sort(key=lambda x: x['last_seen'], reverse=True)
                return jsonify(recent_faces)
            return jsonify([])
            
        # API - 启动/停止监控
        @self.app.route('/api/start', methods=['POST'])
        @self.auth.login_required if self.auth_required else lambda x: x
        def start_monitor():
            success = self.monitor.start()
            return jsonify({'success': success, 'message': "监控已启动" if success else "启动失败"})
            
        @self.app.route('/api/stop', methods=['POST'])
        @self.auth.login_required if self.auth_required else lambda x: x
        def stop_monitor():
            self.monitor.stop()
            return jsonify({'success': True, 'message': "监控已停止"})
            
        # API - 添加人脸
        @self.app.route('/api/add_face', methods=['POST'])
        @self.auth.login_required if self.auth_required else lambda x: x
        def add_face():
            data = request.json
            if not data or 'name' not in data:
                return jsonify({'success': False, 'message': "缺少人名参数"}), 400
            name = data['name']
            success = self.monitor.add_face(name)
            return jsonify({'success': success, 'message': f"已添加 {name}" if success else f"添加 {name} 失败"})

        # API - 删除人脸
        @self.app.route('/api/delete_face', methods=['POST'])
        @self.auth.login_required if self.auth_required else lambda x: x
        def delete_face():
            data = request.json
            if not data or 'name' not in data:
                return jsonify({'success': False, 'message': "缺少人名参数"}), 400
            
            name = data.get('name')
            success = self.monitor.delete_face(name)
            
            if success:
                return jsonify({'success': True, 'message': f"人脸 '{name}' 已被删除"})
            else:
                return jsonify({'success': False, 'message': f"删除人脸 '{name}' 失败"}), 500

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
                        images = [f for f in os.listdir(person_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                        if images:
                            images.sort()
                            faces.append({
                                'name': person_name,
                                'image': f"/known_faces/{person_name}/{images[0]}"
                            })
            return jsonify(faces)

        # 静态文件 - 已知人脸图片
        @self.app.route('/known_faces/<path:filename>')
        @self.auth.login_required if self.auth_required else lambda x: x
        def known_faces(filename):
            known_faces_dir = self.config.get('face_recognition.known_faces_dir', './data/known_faces')
            return send_from_directory(known_faces_dir, filename, as_attachment=False)

    def _jpeg_encoder_thread(self):
        """后台线程，用于将视频帧编码为JPEG"""
        logging.info("视频帧编码器线程已启动")
        
        while self.is_streaming:
            frame = self.monitor.get_latest_frame()
            
            if frame is None:
                time.sleep(self.streaming_interval)
                continue
            
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
            ret, buffer = cv2.imencode('.jpg', frame, encode_param)
            
            if ret:
                with self.jpeg_frame_lock:
                    self.latest_jpeg_frame = buffer.tobytes()

            time.sleep(self.streaming_interval)
            
        logging.info("视频帧编码器线程已停止")

    def _generate_frames(self):
        """从缓存中读取已编码的帧，并作为视频流发送"""
        while True:
            time.sleep(self.streaming_interval)
            with self.jpeg_frame_lock:
                frame_bytes = self.latest_jpeg_frame
            
            if frame_bytes:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
    def start(self, debug: bool = False) -> None:
        """启动Web服务器"""
        if self.server_thread:
            logging.warning("Web服务器已在运行中")
            return

        self.is_streaming = True
        self.streaming_thread = threading.Thread(target=self._jpeg_encoder_thread)
        self.streaming_thread.daemon = True
        self.streaming_thread.start()
        
        self.server_thread = threading.Thread(
            target=self.app.run,
            kwargs={'host': self.host, 'port': self.port, 'debug': debug, 'use_reloader': False}
        )
        self.server_thread.daemon = True
        self.server_thread.start()
        logging.info(f"Web服务器已在 http://{self.host}:{self.port} 启动")

    def stop(self) -> None:
        """停止Web服务器"""
        self.is_streaming = False
        if self.streaming_thread:
            self.streaming_thread.join(1.0)
            self.streaming_thread = None
        
        # Flask没有内置的停止方法，因此我们不直接操作server_thread
        # 它会随主程序退出而关闭
        self.server_thread = None
        logging.info("Web服务器已停止") 