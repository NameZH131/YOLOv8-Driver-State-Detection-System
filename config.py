"""
@writer: zhangheng
Configuration module for YOLOv8 Driver State Detection System
Contains all configuration parameters and constants
"""
import os

# Environment Configuration
TORCH_HOME = './torch_cache'
os.environ['TORCH_HOME'] = TORCH_HOME

# YOLOv8-Pose Keypoint Indices (17 standard keypoints)
YOLO_KEYPOINTS = {
    "nose": 0,
    "left_eye": 1,
    "right_eye": 2,
    "left_ear": 3,
    "right_ear": 4,
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_elbow": 7,
    "right_elbow": 8,
    "left_wrist": 9,
    "right_wrist": 10,
    "left_hip": 11,
    "right_hip": 12,
    "left_knee": 13,
    "right_knee": 14,
    "left_ankle": 15,
    "right_ankle": 16
}

# Eye State Detection Configuration
EYE_CONFIDENCE_THRESHOLD = 0.5
BLINK_THRESHOLD = 0.25
BLINK_INTERVAL_THRESHOLD = 0.8
CONTINUOUS_TIRED_FRAMES = 1.2

# Model Configuration
YOLO_MODEL_PATH = './yolov8n-pose.pt'
CONF_THRESHOLD = 0.6
IOU_THRESHOLD = 0.45

# Head Pose Detection Thresholds
HEAD_UP_THRESHOLD = -250
HEAD_DOWN_THRESHOLD = -200
HEAD_OFFSET_THRESHOLD = 38
HEAD_TURNED_THRESHOLD = 20
POSTURE_DEVIATION_THRESHOLD = 33

# Server Configuration
HOST = "127.0.0.1"
PORT = 5002

# API Response Messages
MSG_SUCCESS = "success"
MSG_ANALYZER_INIT_FAILED = "Analyzer initialization failed"
MSG_MISSING_FRAME = "Missing frame_base64 parameter"
MSG_IMAGE_DECODE_FAILED = "Image decoding failed, please check base64 format"
MSG_PROCESSING_FAILED = "Processing failed"
MSG_CALIBRATION_SUCCESS = "Calibration successful! Adapted to your normal state"
MSG_CALIBRATION_FAILED = "Calibration failed, please ensure clear face and face forward"
MSG_SERVICE_RUNNING = "Service running normally"
MSG_RESET_SUCCESS = "Counter reset successful"