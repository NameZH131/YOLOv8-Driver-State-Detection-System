# Android Project Build Guide

**[中文](./README.md)** | **[Back to Home](../README-en.md)**

## Project Structure

```
android/app/src/main/java/com/yolo/driver/
├── DriverApplication.kt           # Global Application (language + settings persistence)
├── MainActivity.kt                # Compose container (single Activity architecture)
├── MainViewModel.kt               # ViewModel (StateFlow)
├── analyzer/
│   ├── KeypointDetector.kt        # JNI interface (reference-counted singleton)
│   ├── StateAnalyzer.kt           # Fatigue analysis + pose mapping
│   ├── SlidingWindowAnalyzer.kt   # Time-window aggregation analysis
│   └── CalibrationManager.kt      # Calibration management
├── data/
│   └── CalibrationData.kt         # Calibration data models
├── ui/
│   ├── compose/                   # Jetpack Compose UI
│   │   ├── AppNavigation.kt       # NavHost navigation controller
│   │   ├── MainScreen.kt          # Main screen (detection + settings)
│   │   ├── CalibrationScreen.kt   # Calibration screen
│   │   ├── theme/                 # Theme configuration
│   │   └── components/            # Reusable components
│   │       ├── CameraPreview.kt   # CameraX PreviewView
│   │       ├── KeypointOverlay.kt # Canvas keypoint rendering
│   │       ├── StatePanel.kt      # State panel
│   │       ├── ControlBar.kt      # Bottom control bar
│   │       └── SettingsDialog.kt  # Settings dialog (auto-save)
│   └── viewmodel/
│       └── CalibrationViewModel.kt
└── util/
    ├── AudioPlayer.kt             # Audio playback (priority + stop)
    ├── CameraController.kt        # Camera management
    ├── CameraUtils.kt             # NV21 conversion
    ├── KeypointDrawer.kt          # Keypoint rendering (rotation support)
    └── VibrationController.kt     # Vibration control
```

## Architecture

This project uses **pure Jetpack Compose architecture**:

| Feature | Description |
|---------|-------------|
| UI Framework | Jetpack Compose |
| Activity | Single Activity architecture |
| Navigation | Navigation Compose |
| State Management | StateFlow + ViewModel |
| Layout Files | No XML layouts, pure Compose |

## Requirements

- Android Studio: Hedgehog (2023.1.1) or later
- Android SDK: compileSdk 36
- NDK: 25.2.9519653 or later
- CMake: 3.22.1
- Gradle: 8.9
- JDK: 17

## Features

### Core Features
- **Real-time Pose Detection**: YOLOv8-Pose + NCNN inference
- **Fatigue State Analysis**: Head pose + sitting posture analysis
- **Three-level Classification**: Normal, Slightly Tired, Tired

### Advanced Features
- **Sliding Window Analysis**: Time-based aggregation, 1-30s configurable window
- **Pose State Mapping**: User-customizable pose-to-fatigue state mapping
- **Auto-save Settings**: All settings persisted automatically on change
- **Multi-language Support**: Chinese/English/Auto-detect
- **GPU/CPU Toggle**: Switch inference device in settings
- **Alert Repeat Mode**: Play once / Continuous playback
- **Keypoint Confidence Thresholds**: Draw/analyze thresholds independently configurable

## Build Steps

### 1. Model Conversion

Run the model conversion script to convert PyTorch model to NCNN format:

```powershell
cd D:\Python\Pycharm_Workplace\Pytorch_CUDA\Vehicle_face_pose_recognize_by_yolo_v8_shared
python convert_model.py
```

The converted model files (`yolov8n_pose.ncnn.param` and `yolov8n_pose.ncnn.bin`) will be automatically copied to `app/src/main/assets/`.

### 2. SDKs Included

**No manual download required!** The following SDKs are already included:

| SDK | Path | Description |
|-----|------|-------------|
| NCNN | `jni/ncnn/` | High-performance inference framework with Vulkan GPU support |
| OpenCV | `jni/opencv/` | Computer vision library (static build) |

### 3. Build the Project

**Using Android Studio:**
1. Open Android Studio
2. Select "Open an existing project"
3. Choose the `android` directory
4. Wait for Gradle sync to complete
5. Connect device or start emulator
6. Click Run button

**Using Command Line:**
```powershell
$env:JAVA_HOME="D:\Java\jdk-17.0.12"
cd android

# Debug build
.\gradlew.bat assembleDebug

# Release build
.\gradlew.bat assembleRelease
```

### 4. Install APK

```powershell
adb install -r app\build\outputs\apk\debug\app-debug.apk
```

## Calibration Feature

### Calibration Process

1. User selects collection duration (Fast 2s / Normal 3s / Accurate 5s)
2. Click "Start Calibration"
3. Complete the following actions as prompted:
   - Maintain normal driving posture (baseline collection)
   - Head up
   - Head down
   - Look left
   - Look right
   - Posture deviation
4. System automatically calculates personalized thresholds and saves them

### Calibration Data Storage

Calibration data is saved in JSON format in the app's private directory:

```
/data/data/com.yolo.driver/files/calibration.json
```

## Performance

| Metric | Value |
|--------|-------|
| Target FPS | 30+ FPS |
| Tested FPS (Snapdragon 870) | 80-100 FPS |
| APK Size | ~37 MB |
| Memory Usage | ~150 MB |

## Troubleshooting

### Keypoint Overlay Misaligned with Camera Preview
- **Cause**: PreviewView uses FILL_CENTER, overlay used FIT_CENTER
- **Solution**: Fixed in `KeypointOverlay.kt`

### Detection Only Works on First Frame (Tablet)
- **Cause**: NCNN optimization options causing memory layout issues
- **Solution**: Disabled relevant optimizations in `yolov8pose.cpp`

### Keypoint Position Incorrect After Orientation Change
- **Cause**: Rotation angle not correctly passed
- **Solution**: Added `rotationDegrees` parameter for 0°/90°/180°/270° rotation support

## Acknowledgments

This project uses the following open-source projects:

- **[NCNN](https://github.com/Tencent/ncnn)** - High-performance neural network inference framework by Tencent, with mobile GPU acceleration support
- **[OpenCV](https://opencv.org/)** - Open-source computer vision library with rich image processing features
- **[CameraX](https://developer.android.com/training/camerax)** - Official Android camera library that simplifies camera development
- **[YOLOv8](https://github.com/ultralytics/ultralytics)** - Advanced object detection and pose estimation model by Ultralytics
- **[Jetpack Compose](https://developer.android.com/jetpack/compose)** - Modern declarative UI toolkit for Android