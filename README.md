# 树莓派人脸识别监控系统

基于Python的树莓派人脸识别监控系统，支持多种摄像头接口，提供Web界面进行实时监控和管理。

## 功能特点

- **多平台摄像头支持**：支持Picamera2、OpenCV和libcamera多种摄像头接口
- **实时人脸检测与识别**：基于`face_recognition`库实现高效的人脸检测和识别
- **Web界面**：提供友好的Web界面进行系统控制和监控
- **灵活配置**：通过YAML配置文件和命令行参数提供灵活的配置选项
- **历史记录**：保存识别到的人脸记录

## 安装

### 依赖项

- Python 3.7+
- OpenCV
- face_recognition
- Flask
- Picamera2 (用于树莓派摄像头)
- 详细依赖项请查看`requirements.txt`文件

### 安装步骤

1. 克隆仓库：
   ```bash
   git clone https://github.com/yourusername/facerec-python.git
   cd facerec-python
   ```

2. 创建并激活虚拟环境：
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # Linux/Mac
   # 或
   venv\Scripts\activate  # Windows
   ```

3. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```

4. `face_recognition` 库依赖安装 (可选, 仅针对部分系统):
   ```bash
   # 在树莓派 (Raspberry Pi OS) 或其他Debian系系统上，可能需要先安装dlib的依赖
   sudo apt-get update
   sudo apt-get install build-essential cmake
   sudo apt-get install libopenblas-dev liblapack-dev
   sudo apt-get install libx11-dev libgtk-3-dev
   
   # 然后通过pip安装dlib和face_recognition
   # pip install dlib
   # pip install face_recognition
   ```

## 使用方法

### 基本用法

1. 启动系统：
   ```bash
   python3 main.py
   ```

2. 使用配置文件启动：
   ```bash
   python3 main.py --config path/to/your_config.yaml
   ```

3. 通过Web界面访问系统：
   在浏览器中打开 `http://localhost:8080` 或 `http://<树莓派IP>:8080`

### 命令行参数

- `--config`: 指定配置文件路径 (例如: `my_config.yaml`)
- `--camera-type`: 覆盖配置文件中的摄像头类型 (`picamera2`, `opencv`, `libcamera`)
- `--resolution`: 覆盖分辨率 (格式: `宽x高`, 例如: `640x480`)
- `--detection-fps`: 覆盖人脸检测帧率
- `--web-port`: 覆盖Web界面端口
- `--log-file`: 覆盖日志文件路径
- `--verbose`: 显示更详细的日志输出 (DEBUG级别)

### 配置文件

系统启动时会加载 `config/default_config.yaml` 作为默认配置。你可以创建自己的配置文件（例如 `my_config.yaml`），在其中只填写需要覆盖的配置项，然后通过 `--config` 参数指定。

详细的可配置项请参考 `config/default_config.yaml`。

## 添加/删除人脸

1.  **通过Web界面添加**: 在监控画面中识别到人脸后，输入姓名并点击"添加当前人脸"。
2.  **手动添加**: 将人脸照片 (每张照片只包含一个清晰的人脸) 放置到 `data/known_faces/张三/` 这样的目录结构中，目录名即为人名。
3.  **通过Web界面删除**: 在Web界面的已知人脸列表中，点击姓名旁边的删除按钮。

## 目录结构

```
facerec-python/
│
├── assets/                 # 静态资源 (如字体)
│
├── camera/                 # 摄像头模块
│   ├── camera_factory.py   # 摄像头工厂
│   ├── camera_interface.py # 摄像头接口
│   └── ...                 # 各种摄像头实现
│
├── config/                 # 配置模块
│   ├── config_manager.py   # 配置管理器
│   └── default_config.yaml # 默认配置文件
│
├── camera/                 # 摄像头模块
│   ├── __init__.py
│   ├── camera_interface.py # 摄像头接口基类
│   ├── camera_factory.py   # 摄像头工厂类
│   ├── picamera2_camera.py # Picamera2实现
│   ├── opencv_camera.py    # OpenCV实现
│   └── libcamera_camera.py # libcamera实现
│
├── face/                   # 人脸识别模块
│   └── face_recognizer.py  # 人脸识别器
│
├── monitor/                # 监控模块
│   ├── __init__.py
│   └── monitor.py          # 人脸监控类
│
├── interface/              # 用户界面模块
│   └── web/                # Web界面
│       ├── __init__.py
│       ├── web_server.py   # Web服务器
│       ├── templates/      # HTML模板
│       └── static/         # 静态文件 (CSS, JS, images)
│
├── data/                   # 数据目录
│   ├── known_faces/        # 已知人脸
│   ├── unknown_faces/      # 未知人脸
│   └── detected_images/    # 检测到的图像
│
├── logs/                   # 日志目录
│
├── main.py                 # 主程序入口
├── requirements.txt        # Python依赖库
├── LICENSE                 # 项目许可证
└── README.md               # 本文档
```

## 注意事项

- **首次运行**: 程序首次运行时会自动创建 `data` 和 `logs` 目录。
- **中文字体**: 为了在监控画面上正确显示中文名，系统需要中文字体文件。默认配置指向 `assets/fonts/wqy-zenhei.ttc`。如果此文件不存在或路径不正确，中文将无法显示。你可以下载一个中文字体 (如"文泉驿正黑"、"思源黑体") 并更新 `config.yaml` 中的 `monitoring.font_path`。
- **性能**: 人脸识别是计算密集型任务。在树莓派等资源有限的设备上，建议通过配置文件将 `face_recognition.detection_fps`（人脸识别帧率）和 `camera.resolution`（摄像头分辨率）设置在较低的水平，以保证系统流畅运行。
- **摄像头**: 确保在运行前，你的摄像头已经正确安装并被操作系统识别。对于树莓派摄像头，可能需要在 `raspi-config` 中启用。

## 许可证

本项目采用 [MPL-2.0](LICENSE) 许可证。

## 联系方式

如有问题或建议，欢迎提交 Issue 或 Pull Request。 
