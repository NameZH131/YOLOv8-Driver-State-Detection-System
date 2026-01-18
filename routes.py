"""
@writer: zhangheng
FastAPI routes for YOLOv8 Driver State Detection API
"""
import numpy as np
import cv2
import base64
from fastapi import APIRouter
from models import FrameRequest
from config import (
    MSG_ANALYZER_INIT_FAILED, MSG_MISSING_FRAME, MSG_IMAGE_DECODE_FAILED,
    MSG_PROCESSING_FAILED, MSG_CALIBRATION_SUCCESS, MSG_CALIBRATION_FAILED,
    MSG_SERVICE_RUNNING, MSG_RESET_SUCCESS
)

router = APIRouter()


def decode_base64_image(frame_base64: str):
    """Decode base64 image to numpy array"""
    frame_base64 = frame_base64.strip()
    if frame_base64.startswith('data:image'):
        frame_base64 = frame_base64.split(',')[1]

    frame_bytes = base64.b64decode(frame_base64)
    frame_np = np.frombuffer(frame_bytes, dtype=np.uint8)
    frame = cv2.imdecode(frame_np, cv2.IMREAD_COLOR)

    return frame


@router.post("/analyze_driver")
def analyze_driver(request: FrameRequest, analyzer):
    """Analyze driver state with marked frame"""
    if analyzer is None:
        return {
            "code": 500,
            "msg": MSG_ANALYZER_INIT_FAILED,
            "data": None
        }

    try:
        frame = decode_base64_image(request.frame_base64)

        if frame is None:
            return {
                "code": 400,
                "msg": MSG_IMAGE_DECODE_FAILED,
                "data": None
            }

        result, marked_frame, eye_keypoints = analyzer.process_frame(frame)
        marked_base64 = analyzer.frame_to_base64(marked_frame)

        result["marked_frame_base64"] = marked_base64
        result["model_type"] = "yolov8_pose"
        result["eye_keypoints"] = eye_keypoints

        return {
            "code": 200,
            "msg": "success",
            "data": result
        }

    except Exception as e:
        print(f"Frame processing error: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "code": 500,
            "msg": f"{MSG_PROCESSING_FAILED}: {str(e)}",
            "data": None
        }


@router.post("/analyze_driver_silence")
def analyze_driver_silence(request: FrameRequest, analyzer):
    """Analyze driver state without marked frame"""
    if analyzer is None:
        return {
            "code": 500,
            "msg": MSG_ANALYZER_INIT_FAILED,
            "data": None
        }

    try:
        frame = decode_base64_image(request.frame_base64)

        if frame is None:
            return {
                "code": 400,
                "msg": MSG_IMAGE_DECODE_FAILED,
                "data": None
            }

        result, marked_frame, eye_keypoints = analyzer.process_frame(frame)

        return {
            "code": 200,
            "msg": "success",
            "data": result
        }

    except Exception as e:
        print(f"Frame processing error: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "code": 500,
            "msg": f"{MSG_PROCESSING_FAILED}: {str(e)}",
            "data": None
        }


@router.post("/calibrate")
def calibrate(request: FrameRequest, analyzer):
    """Calibrate system with driver's normal state"""
    if analyzer is None:
        return {
            "code": 500,
            "msg": MSG_ANALYZER_INIT_FAILED,
            "data": None
        }

    try:
        frame = decode_base64_image(request.frame_base64)

        if frame is None:
            return {
                "code": 400,
                "msg": MSG_IMAGE_DECODE_FAILED,
                "data": None
            }

        keypoints, _ = analyzer.detect_keypoints(frame)
        success = analyzer.calibrate(keypoints)

        return {
            "code": 200 if success else 400,
            "msg": MSG_CALIBRATION_SUCCESS if success else MSG_CALIBRATION_FAILED,
            "data": {"calibrated": success}
        }

    except Exception as e:
        print(f"Calibration error: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "code": 500,
            "msg": f"{MSG_CALIBRATION_FAILED}: {str(e)}",
            "data": None
        }


@router.get("/health")
def health_check(analyzer):
    """Check service health status"""
    return {
        "code": 200,
        "msg": MSG_SERVICE_RUNNING,
        "data": {
            "analyzer_ready": analyzer is not None,
            "device": str(analyzer.device) if analyzer else "unknown",
            "model_loaded": analyzer.yolo_model is not None if analyzer else False,
            "calibrated": analyzer.calibrated if analyzer else False
        }
    }


@router.post("/reset")
def reset_counter(analyzer):
    """Reset all statistical counters"""
    if analyzer:
        analyzer.reset()

    return {
        "code": 200,
        "msg": MSG_RESET_SUCCESS,
        "data": None
    }