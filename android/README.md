# Android 项目构建说明

**[English](./README-en.md)** | **[返回主页](../README.md)**

## 项目结构

```
android/app/src/main/java/com/yolo/driver/
├── DriverApplication.kt           # 全局 Application (语言+设置持久化)
├── MainActivity.kt                # Compose 容器 (单 Activity 架构)
├── MainViewModel.kt               # ViewModel (StateFlow)
├── analyzer/
│   ├── KeypointDetector.kt        # JNI 接口 (引用计数单例)
│   ├── StateAnalyzer.kt           # 疲劳分析 + 姿态映射
│   ├── SlidingWindowAnalyzer.kt   # 时间窗口聚合分析
│   └── CalibrationManager.kt      # 校准管理
├── data/
│   └── CalibrationData.kt         # 校准数据模型
├── ui/
│   ├── compose/                   # Jetpack Compose UI
│   │   ├── AppNavigation.kt       # NavHost 导航控制器
│   │   ├── MainScreen.kt          # 主页面 (检测+设置)
│   │   ├── CalibrationScreen.kt   # 校准页面
│   │   ├── theme/                 # 主题配置
│   │   └── components/            # 可复用组件
│   │       ├── CameraPreview.kt   # CameraX PreviewView
│   │       ├── KeypointOverlay.kt # Canvas 关键点绘制
│   │       ├── StatePanel.kt      # 状态面板
│   │       ├── ControlBar.kt      # 底部控制栏
│   │       └── SettingsDialog.kt  # 设置对话框 (自动保存)
│   └── viewmodel/
│       └── CalibrationViewModel.kt
└── util/
    ├── AudioPlayer.kt             # 音频播放 (优先级+停止)
    ├── CameraController.kt        # 相机管理
    ├── CameraUtils.kt             # NV21 转换
    ├── KeypointDrawer.kt          # 关键点绘制 (旋转支持)
    └── VibrationController.kt     # 振动控制
```

## 架构说明

本项目采用 **纯 Jetpack Compose 架构**：

| 特征 | 说明 |
|------|------|
| UI 框架 | Jetpack Compose |
| Activity | 单 Activity 架构 |
| 导航 | Navigation Compose |
| 状态管理 | StateFlow + ViewModel |
| 布局文件 | 无 XML 布局，全部 Compose |

## 环境要求

- Android Studio: Hedgehog (2023.1.1) 或更高版本
- Android SDK: compileSdk 36
- NDK: 25.2.9519653 或更高版本
- CMake: 3.22.1
- Gradle: 8.9
- JDK: 17

## 功能特性

### 核心功能
- **实时姿态检测**: 使用 YOLOv8-Pose + NCNN 推理
- **疲劳状态分析**: 头部姿态 + 坐姿分析
- **三状态分级**: 正常、轻度疲劳、疲劳

### 高级功能
- **滑动窗口分析**: 时间聚合，1-30s 可配置窗口
- **姿态状态映射**: 用户自定义姿态到疲劳状态的映射
- **设置自动保存**: 所有设置变更即时持久化
- **多语言支持**: 中文/英文/自动检测
- **GPU/CPU 切换**: 设置中可切换推理设备
- **提醒播放模式**: 只播放一次/持续播放
- **关键点置信度阈值**: 绘制/分析阈值可独立配置

## 构建步骤

### 1. 模型转换

运行模型转换脚本，将 PyTorch 模型转换为 NCNN 格式：

```powershell
cd D:\Python\Pycharm_Workplace\Pytorch_CUDA\Vehicle_face_pose_recognize_by_yolo_v8_shared
python convert_model.py
```

转换后的模型文件 (`yolov8n_pose.ncnn.param` 和 `yolov8n_pose.ncnn.bin`) 会自动复制到 `app/src/main/assets/` 目录。

### 2. SDK 已内置

**无需手动下载！** 以下 SDK 已内置在项目中：

| SDK | 路径 | 说明 |
|-----|------|------|
| NCNN | `jni/ncnn/` | 高性能推理框架，支持 Vulkan GPU |
| OpenCV | `jni/opencv/` | 计算机视觉库（静态编译） |

### 3. 构建项目

**使用 Android Studio：**
1. 打开 Android Studio
2. 选择 "Open an existing project"
3. 选择 `android` 目录
4. 等待 Gradle 同步完成
5. 连接设备或启动模拟器
6. 点击 Run 按钮

**使用命令行：**
```powershell
$env:JAVA_HOME="D:\Java\jdk-17.0.12"
cd android

# Debug 版本
.\gradlew.bat assembleDebug

# Release 版本
.\gradlew.bat assembleRelease
```

### 4. 安装 APK

```powershell
adb install -r app\build\outputs\apk\debug\app-debug.apk
```

## 校准功能

### 校准流程

1. 用户选择采集时长 (快速2秒/正常3秒/精确5秒)
2. 点击"开始校准"
3. 按提示依次完成以下动作：
   - 保持正常驾驶姿势 (基准采集)
   - 抬头
   - 低头
   - 左看
   - 右看
   - 坐姿偏移
4. 系统自动计算个性化阈值并保存

### 校准数据存储

校准数据以 JSON 格式保存在应用私有目录：

```
/data/data/com.yolo.driver/files/calibration.json
```

## 性能指标

| 指标 | 数值 |
|------|------|
| 目标帧率 | 30+ FPS |
| 实测帧率 (骁龙 870) | 80-100 FPS |
| APK 大小 | ~37 MB |
| 内存占用 | ~150 MB |

## 常见问题

### 关键点覆盖层与相机预览不对齐
- **原因**: PreviewView 使用 FILL_CENTER，覆盖层使用 FIT_CENTER
- **解决**: 已在 `KeypointOverlay.kt` 中修复

### 平板设备检测只有第一帧有效
- **原因**: NCNN 优化选项导致内存布局问题
- **解决**: 已在 `yolov8pose.cpp` 中禁用相关优化

### 设备方向切换后关键点位置错误
- **原因**: 旋转角度未正确传递
- **解决**: 已添加 `rotationDegrees` 参数支持 0°/90°/180°/270° 旋转

## 致谢

本项目使用了以下开源项目，感谢他们的贡献：

- **[NCNN](https://github.com/Tencent/ncnn)** - 腾讯开源的高性能神经网络推理框架，支持移动端 GPU 加速
- **[OpenCV](https://opencv.org/)** - 开源计算机视觉库，提供丰富的图像处理功能
- **[CameraX](https://developer.android.com/training/camerax)** - Android 官方相机库，简化相机开发
- **[YOLOv8](https://github.com/ultralytics/ultralytics)** - Ultralytics 开发的先进目标检测和姿态估计模型
- **[Jetpack Compose](https://developer.android.com/jetpack/compose)** - Android 现代声明式 UI 工具包