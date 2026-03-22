# Android 项目构建说明

## 项目结构

```
android/
├── app/
│   ├── src/main/
│   │   ├── java/com/yolo/driver/
│   │   │   ├── MainActivity.kt           # 主界面
│   │   │   ├── MainViewModel.kt          # ViewModel
│   │   │   ├── analyzer/
│   │   │   │   ├── KeypointDetector.kt   # 关键点检测 JNI 接口
│   │   │   │   ├── StateAnalyzer.kt      # 疲劳状态分析
│   │   │   │   └── CalibrationManager.kt # 校准管理
│   │   │   ├── data/
│   │   │   │   └── CalibrationData.kt    # 校准数据模型
│   │   │   └── ui/
│   │   │       └── CalibrationActivity.kt # 校准界面
│   │   ├── jni/
│   │   │   ├── CMakeLists.txt            # CMake 配置
│   │   │   ├── yolov8pose.h              # YOLOv8 接口头文件
│   │   │   ├── yolov8pose.cpp            # YOLOv8 推理实现
│   │   │   └── native-lib.cpp            # JNI 接口实现
│   │   ├── assets/                       # 模型文件目录
│   │   │   ├── yolov8n-pose.param
│   │   │   └── yolov8n-pose.bin
│   │   ├── res/
│   │   │   └── layout/
│   │   │       ├── activity_main.xml
│   │   │       └── activity_calibration.xml
│   │   └── AndroidManifest.xml
│   └── build.gradle.kts
├── build.gradle.kts
└── settings.gradle.kts
```

## 环境要求

- Android Studio: Hedgehog (2023.1.1) 或更高版本
- Android SDK: compileSdk 36
- NDK: 25.2.9519653 或更高版本
- CMake: 3.22.1
- Gradle: 8.9

## 构建步骤

### 1. 模型转换

运行模型转换脚本，将 PyTorch 模型转换为 NCNN 格式：

```powershell
cd D:\Python\Pycharm_Workplace\Pytorch_CUDA\Vehicle_face_pose_recognize_by_yolo_v8_shared
python convert_model.py
```

或者手动转换：

```python
from ultralytics import YOLO
model = YOLO('yolov8n-pose.pt')
model.export(format='onnx', imgsz=640, opset=12, simplify=True)
```

然后使用 onnx2ncnn 工具转换为 NCNN 格式。

### 2. 下载 NCNN SDK

从以下地址下载 NCNN Android SDK：

- GitHub Releases: https://github.com/Tencent/ncnn/releases
- 或直接构建: https://github.com/Tencent/ncnn/wiki/how-to-build#build-for-android

下载后解压，将以下文件复制到项目：

```
ncnn-android-vulkan-sdk/
├── arm64-v8a/
│   └── libncnn.so    → android/app/src/main/jni/ncnn/libs/arm64-v8a/
├── armeabi-v7a/
│   └── libncnn.so    → android/app/src/main/jni/ncnn/libs/armeabi-v7a/
└── include/          → android/app/src/main/jni/ncnn/include/
```

### 3. 添加 OpenCV (可选)

如果需要更好的图像处理支持，可以集成 OpenCV Android SDK：

1. 下载 OpenCV Android SDK: https://opencv.org/releases/
2. 解压后将 `sdk/native/jni/include` 复制到 `android/app/src/main/jni/opencv/include/`
3. 将 `sdk/native/libs/` 复制到 `android/app/src/main/jni/opencv/libs/`
4. 修改 CMakeLists.txt 添加 OpenCV 依赖

### 4. 用 Android Studio 打开项目

1. 打开 Android Studio
2. 选择 "Open an existing project"
3. 选择 `android` 目录
4. 等待 Gradle 同步完成
5. 连接 Android 设备或启动模拟器
6. 点击 Run 按钮

## 校准功能说明

### 校准流程

1. 用户选择采集时长 (2秒/3秒/5秒)
2. 点击"开始校准"
3. 按提示依次完成以下动作：
   - 保持正常驾驶姿势 (基准采集)
   - 抬头
   - 低头
   - 左看
   - 右看
   - 左右倾斜身体
4. 系统自动计算个性化阈值并保存

### 校准数据存储

校准数据以 JSON 格式保存在应用私有目录：

```
/data/data/com.yolo.driver/files/calibration.json
```

## 已知问题

1. JNI 代码需要 OpenCV 库支持，当前使用纯 NCNN 实现
2. 需要手动下载 NCNN SDK 和模型文件

## 下一步

1. 完善 JNI 接口（添加 OpenCV 依赖或使用纯 C++ 实现）
2. 添加校准动作示意图资源
3. 优化 UI 布局
4. 测试不同机型性能
