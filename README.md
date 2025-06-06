# 树莓派人脸识别监控系统

基于Python的树莓派人脸识别监控系统，支持多种摄像头接口，提供Web界面进行实时监控和管理。

## 功能特点

- **多平台摄像头支持**：支持Picamera2、OpenCV和libcamera多种摄像头接口
- **实时人脸检测与识别**：基于face_recognition库实现高效的人脸检测和识别
- **高效人脸跟踪**：利用前一帧的信息预测和关联当前帧中的人脸，减少计算资源消耗
- **Web界面**：提供友好的Web界面进行系统控制和监控
- **灵活配置**：通过YAML配置文件和命令行参数提供灵活的配置选项
- **运动检测预过滤**：可选的运动检测预过滤功能，减少CPU使用率
- **历史记录**：保存检测到的人脸图像和记录

## 安装

### 依赖项

- Python 3.7+
- OpenCV
- face_recognition
- Flask
- Picamera2 (用于树莓派摄像头)
- 其他依赖项请查看`requirements.txt`

### 安装步骤

1. 克隆仓库：
   ```bash
   git clone https://github.com/yourusername/facerec-python.git
   cd facerec-python
   ```

2. 创建并激活虚拟环境：
   ```bash
   python -m venv my_env
   source my_env/bin/activate  # Linux/Mac
   # 或
   my_env\Scripts\activate  # Windows
   ```

3. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```

4. 安装face_recognition库及其依赖：
   ```bash
   # 在树莓派上，首先安装dlib依赖
   sudo apt-get update
   sudo apt-get install build-essential cmake
   sudo apt-get install libopenblas-dev liblapack-dev
   sudo apt-get install libx11-dev libgtk-3-dev
   
   # 然后安装face_recognition
   pip install dlib
   pip install face_recognition
   ```

## 使用方法

### 基本用法

1. 启动系统：
   ```bash
   python main.py
   ```

2. 使用配置文件启动：
   ```bash
   python main.py --config path/to/config.yaml
   ```

3. 通过Web界面访问系统：
   在浏览器中打开 http://localhost:8080 或 http://树莓派IP:8080

### 命令行参数

- `--config`: 指定配置文件路径
- `--camera-type`: 设置摄像头类型 (picamera2, opencv, libcamera)
- `--resolution`: 设置分辨率，格式为 宽x高，如 640x480
- `--detection-fps`: 设置人脸检测帧率
- `--web-port`: 设置Web界面端口
- `--log-file`: 设置日志文件路径
- `--verbose`: 显示详细日志

### 配置文件结构

配置文件使用YAML格式，详细结构请参考`config/default_config.yaml`。你可以创建自己的配置文件并通过命令行参数`--config`指定。

## 添加人脸

1. 通过Web界面在监控画面中识别到人脸后，输入姓名并点击"添加当前人脸"
2. 或者将人脸照片直接放置到`data/known_faces/人名/`目录下

## 目录结构

```
facerec-python/
│
├── config/                 # 配置模块
│   ├── __init__.py
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
│   ├── __init__.py
│   ├── face_recognizer.py  # 人脸识别类
│   └── face_tracker.py     # 人脸跟踪类
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
│       └── static/         # 静态文件
│
├── data/                   # 数据目录
│   ├── known_faces/        # 已知人脸
│   ├── unknown_faces/      # 未知人脸
│   └── detected_images/    # 检测到的图像
│
├── logs/                   # 日志目录
│
├── main.py                 # 主程序入口
├── requirements.txt        # 依赖项
└── README.md               # 说明文档
```

## 人脸跟踪

系统实现了高效的人脸跟踪功能，具有以下特点：

- **对象关联**：使用IoU(Intersection over Union)和中心点距离来关联前后帧中的人脸
- **特征匹配**：比较人脸特征向量的相似度来增强关联准确性
- **优化计算**：只在必要时进行完整的人脸识别，其余时间仅进行跟踪，大幅降低CPU使用率

可通过配置文件中的`face_tracking`部分调整跟踪参数：

```yaml
face_tracking:
  # 是否启用人脸跟踪
  enable: true
  # 最大连续消失帧数，超过此值将删除跟踪对象
  max_disappeared: 30
  # 特征向量最小距离阈值，小于此值认为是同一个人
  min_distance: 0.6
  # 最小IOU阈值，小于此值不考虑合并检测框
  min_iou: 0.3
```

## 注意事项

- 在树莓派上使用时，确保已安装并配置好摄像头
- 人脸识别需要一定的计算资源，在资源有限的设备上可适当调低检测帧率
- 启用人脸跟踪功能可以显著减少CPU使用率，提高系统响应速度
- 首次使用时，请根据实际情况修改配置文件中的参数

## 许可证

MIT License

## 联系方式

如有问题或建议，请提交Issue或Pull Request。 