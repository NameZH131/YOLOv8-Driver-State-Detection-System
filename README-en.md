# YOLOv8 Driver State Detection System

A real-time driver fatigue detection system based on YOLOv8-Pose model, analyzing human pose keypoints to determine the driver's fatigue level.
if u think this is so good ,please start it.🤧🤧

![alt text](asset/1.png)

## Features

- **Real-time Pose Detection**: Uses YOLOv8-Pose model to detect 17 human keypoints
- **Fatigue State Analysis**: Analyzes head pose and sitting posture to determine driver fatigue level
- **State Classification**: Classifies driver state into three levels: normal, mild fatigue, and fatigue
- **System Calibration**: Supports personalized calibration to improve detection accuracy
- **RESTful API**: Provides standardized API interfaces for easy integration

## Tech Stack

- **Backend Framework**: FastAPI + Uvicorn
- **Deep Learning**: PyTorch + Ultralytics YOLOv8
- **Computer Vision**: OpenCV
- **Data Validation**: Pydantic

## Project Structure

```
.
├── main.py                 # Application entry point
├── config.py               # Configuration parameters
├── analyzer.py             # Core analyzer
├── models.py               # Data models
├── routes.py               # API routes
├── requirements.txt        # Dependencies list
├── .gitignore             # Git ignore file
├── yolov8n-pose.pt        # YOLOv8 model weights
└── index.html             # Frontend demo page
```

## Installation

### Requirements

- Python 3.12
- CUDA 12.0 (optional, for GPU acceleration)

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Start the Server

```bash
python main.py
```

Or using uvicorn:

```bash
uvicorn main:app --host 0.0.0.0 --port 5002
```

The service will start at `http://0.0.0.0:5002`.

### API Endpoints

#### 1. Analyze Driver State (with Annotated Frame)

```bash
POST /analyze_driver
Content-Type: application/json

{
  "frame_base64": "data:image/jpeg;base64,..."
}
```

#### 2. Analyze Driver State (without Annotated Frame)

```bash
POST /analyze_driver_silence
Content-Type: application/json

{
  "frame_base64": "data:image/jpeg;base64,..."
}
```

#### 3. System Calibration

```bash
POST /calibrate
Content-Type: application/json

{
  "frame_base64": "data:image/jpeg;base64,..."
}
```

#### 4. Health Check

```bash
GET /health
```

#### 5. Reset Counter

```bash
POST /reset
```

### API Documentation

After starting the service, visit `http://localhost:5002/docs` to view the interactive API documentation provided by FastAPI.

## Configuration

Main configuration parameters are in `config.py`:

- `YOLO_MODEL_PATH`: YOLOv8 model path
- `CONF_THRESHOLD`: Detection confidence threshold
- `IOU_THRESHOLD`: NMS threshold
- `EYE_CONFIDENCE_THRESHOLD`: Eye keypoint confidence threshold
- `CONTINUOUS_TIRED_FRAMES`: Fatigue determination threshold

## Core Algorithms

### Head Pose Detection

Determines by analyzing the relative position of the nose to the shoulder midpoint:
- **Head Up**: Nose position is above the shoulder midpoint
- **Head Down**: Nose position is below the shoulder midpoint
- **Head Tilted**: Nose deviates from the shoulder centerline
- **Head Turned**: Distance between eyes is too narrow

### Sitting Posture Detection

Determines whether the sitting posture is upright by analyzing the height difference between both shoulders.

### Fatigue Determination

Establishes a fatigue cumulative scoring model based on head pose and body posture, and determines fatigue state when the threshold is exceeded.

## Performance Optimization

- Supports GPU acceleration (CUDA)
- Uses lightweight YOLOv8n-pose model
- Real-time processing capability (30+ FPS)

## Notes

1. Ensure sufficient lighting, avoid backlight or too dark environments
2. System calibration is recommended for first-time use
3. Model weights file `yolov8n-pose.pt` will be downloaded automatically

## License

MIT License

## Contributing

Issues and Pull Requests are welcome!