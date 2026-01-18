"""
@writer: zhangheng
YOLOv8 Driver State Analyzer Module
Contains the main analyzer class for detecting driver fatigue state
"""
import torch
import numpy as np
import cv2
import base64
from datetime import datetime
from ultralytics import YOLO
from config import (
    YOLO_KEYPOINTS, EYE_CONFIDENCE_THRESHOLD, BLINK_THRESHOLD,
    CONTINUOUS_TIRED_FRAMES, YOLO_MODEL_PATH, CONF_THRESHOLD,
    IOU_THRESHOLD, HEAD_UP_THRESHOLD, HEAD_DOWN_THRESHOLD,
    HEAD_OFFSET_THRESHOLD, HEAD_TURNED_THRESHOLD, POSTURE_DEVIATION_THRESHOLD
)


class YOLODriverStateAnalyzer:
    """Driver state analyzer using YOLOv8-Pose model"""

    def __init__(self, use_gpu=True):
        """Initialize the analyzer with YOLOv8-Pose model"""
        self.device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')
        self.use_gpu = use_gpu

        # Load YOLOv8-Pose model
        self.yolo_model = self._load_yolo_model()

        # State tracking
        self.prev_eye_state = "open"
        self.blink_count = 0
        self.last_blink_time = datetime.now()
        self.driver_state = "normal"
        self.tired_frame_count = 0
        self.frame_count = 0

        # Calibration parameters
        self.calibrated = False
        self.calibration_data = {
            "eye_ratio": 0.0,
            "head_angle": 0.0
        }

        print(f"Using device: {self.device}")
        print(f"YOLOv8-Pose model load status: {'Success' if self.yolo_model else 'Failed'}")

    def _load_yolo_model(self):
        """Load YOLOv8-Pose model"""
        try:
            model = YOLO(YOLO_MODEL_PATH)
            model.to(self.device)
            print(f"YOLOv8-Pose model loaded successfully ({YOLO_MODEL_PATH})")
            return model
        except Exception as e:
            print(f"Failed to load YOLOv8-Pose model: {e}")
            import traceback
            traceback.print_exc()
            return None

    def detect_keypoints(self, image_array):
        """YOLOv8-Pose unified detection of body + facial keypoints"""
        if self.yolo_model is None:
            return None, None

        try:
            results = self.yolo_model(
                image_array,
                conf=CONF_THRESHOLD,
                iou=IOU_THRESHOLD,
                device=self.device,
                verbose=False
            )

            if len(results) == 0 or results[0].keypoints is None:
                return None, None

            keypoints = results[0].keypoints.data[0].cpu().numpy()
            bbox = results[0].boxes.data[0].cpu().numpy()

            return keypoints, bbox
        except Exception as e:
            print(f"Keypoint detection error: {e}")
            return None, None

    def analyze_head_pose(self, keypoints):
        """Analyze head pose + sitting posture based on YOLO keypoints"""
        if keypoints is None:
            return "Unknown"

        nose = keypoints[YOLO_KEYPOINTS["nose"]]
        left_shoulder = keypoints[YOLO_KEYPOINTS["left_shoulder"]]
        right_shoulder = keypoints[YOLO_KEYPOINTS["right_shoulder"]]
        left_eye = keypoints[YOLO_KEYPOINTS["left_eye"]]
        right_eye = keypoints[YOLO_KEYPOINTS["right_eye"]]

        valid_points = [
            nose[2] > EYE_CONFIDENCE_THRESHOLD,
            left_shoulder[2] > EYE_CONFIDENCE_THRESHOLD,
            right_shoulder[2] > EYE_CONFIDENCE_THRESHOLD,
            left_eye[2] > EYE_CONFIDENCE_THRESHOLD,
            right_eye[2] > EYE_CONFIDENCE_THRESHOLD
        ]

        if not all(valid_points):
            return "Unknown"

        shoulder_mid_x = (left_shoulder[0] + right_shoulder[0]) / 2
        shoulder_mid_y = (left_shoulder[1] + right_shoulder[1]) / 2

        dy_nose_shoulder = nose[1] - shoulder_mid_y
        dx_nose_shoulder = abs(nose[0] - shoulder_mid_x)
        dy_shoulder = abs(left_shoulder[1] - right_shoulder[1])

        head_pose = []

        if dy_nose_shoulder < HEAD_UP_THRESHOLD:
            print('+'*5 + "Head Up:" + str(dy_nose_shoulder))
            head_pose.append("Head Up")
        elif dy_nose_shoulder > HEAD_DOWN_THRESHOLD:
            head_pose.append("Head Down")
        else:
            head_pose.append("Facing Forward")

        if dx_nose_shoulder > HEAD_OFFSET_THRESHOLD:
            print('+'*5 + "Head Offset:" + str(dx_nose_shoulder))
            head_pose.append("Head Offset")
        elif abs(left_eye[0] - right_eye[0]) < HEAD_TURNED_THRESHOLD:
            print('+'*5 + "Head Turned:" + str(abs(left_eye[0] - right_eye[0])))
            head_pose.append("Head Turned")

        if dy_shoulder > POSTURE_DEVIATION_THRESHOLD:
            print('+'*5 + "Posture Deviation:" + str(dy_shoulder))
            head_pose.append("Posture Deviation")

        return " + ".join(head_pose) if head_pose else "Unknown"

    def _calculate_eye_ratio(self, eye_keypoint, keypoints):
        """Calculate eye openness ratio"""
        nose = keypoints[YOLO_KEYPOINTS["nose"]]
        eye_x, eye_y, eye_conf = eye_keypoint

        eye_nose_dist = nose[1] - eye_y
        eye_nose_dist = max(10, eye_nose_dist)

        open_ratio = eye_nose_dist / 35
        return min(1.0, max(0.0, open_ratio))

    def analyze_eye_state(self, keypoints):
        """Analyze eye state (based on YOLO eye keypoints)"""
        if keypoints is None:
            return "unknown", 0.0

        left_eye = keypoints[YOLO_KEYPOINTS["left_eye"]]
        right_eye = keypoints[YOLO_KEYPOINTS["right_eye"]]

        if left_eye[2] < EYE_CONFIDENCE_THRESHOLD or right_eye[2] < EYE_CONFIDENCE_THRESHOLD:
            return "unknown", 0.0

        left_ratio = self._calculate_eye_ratio(left_eye, keypoints)
        right_ratio = self._calculate_eye_ratio(right_eye, keypoints)
        avg_ratio = (left_ratio + right_ratio) / 2

        if self.calibrated:
            calibrated_ratio = avg_ratio / self.calibration_data["eye_ratio"]
            calibrated_ratio = max(0.0, min(1.0, calibrated_ratio))
        else:
            calibrated_ratio = avg_ratio

        if calibrated_ratio < BLINK_THRESHOLD:
            return "closed", calibrated_ratio
        else:
            return "open", calibrated_ratio

    def calibrate(self, keypoints):
        """User calibration: Record eye ratio in normal state"""
        if keypoints is None:
            return False

        left_eye = keypoints[YOLO_KEYPOINTS["left_eye"]]
        right_eye = keypoints[YOLO_KEYPOINTS["right_eye"]]

        if left_eye[2] > EYE_CONFIDENCE_THRESHOLD and right_eye[2] > EYE_CONFIDENCE_THRESHOLD:
            left_ratio = self._calculate_eye_ratio(left_eye, keypoints)
            right_ratio = self._calculate_eye_ratio(right_eye, keypoints)
            self.calibration_data["eye_ratio"] = (left_ratio + right_ratio) / 2
            self.calibrated = True
            print(f"Calibration successful! Normal eye openness ratio: {self.calibration_data['eye_ratio']:.3f}")
            return True
        return False

    def analyze_driver_state(self, keypoints):
        """Comprehensive analysis of driving state"""
        self.tired_frame_count = 0

        current_eye_state, eye_ratio = self.analyze_eye_state(keypoints)
        head_pose = self.analyze_head_pose(keypoints)

        current_time = datetime.now()
        blink_interval = (current_time - self.last_blink_time).total_seconds()

        # Head pose/sitting posture fatigue detection
        if "Head Down" in head_pose or "Head Up" in head_pose:
            self.tired_frame_count += 1.2
        if "Posture Deviation" in head_pose:
            self.tired_frame_count += 2
        elif head_pose == "Unknown":
            self.tired_frame_count += 0.5

        # Fatigue state determination
        if self.tired_frame_count >= CONTINUOUS_TIRED_FRAMES * 1.2:
            self.driver_state = "tired"
        elif self.tired_frame_count > CONTINUOUS_TIRED_FRAMES * 0.8:
            self.driver_state = "slightly_tired"
        else:
            self.driver_state = "normal"
            self.tired_frame_count = 0

        # Update state
        self.prev_eye_state = current_eye_state
        self.frame_count += 1

        # Calculate blink rate
        elapsed_time = max(1.0, (current_time - self.last_blink_time).total_seconds())
        blink_rate = (self.blink_count / elapsed_time) * 60

        return {
            "driver_state": self.driver_state,
            "blink_count": int(self.blink_count),
            "blink_rate": float(round(blink_rate, 1)),
            "eye_state": current_eye_state,
            "head_pose": head_pose,
            "timestamp": current_time.isoformat(),
            "eye_ratio": float(round(eye_ratio * 100, 1)),
            "tired_frame_count": float(round(self.tired_frame_count, 1)),
            "frame_count": int(self.frame_count),
            "calibrated": self.calibrated
        }

    def draw_annotations(self, image, keypoints, bbox):
        """Draw annotations: keypoints + skeleton + state labels"""
        image_np = image.copy()

        if bbox is not None:
            x1, y1, x2, y2 = map(int, bbox[:4])
            cv2.rectangle(image_np, (x1, y1), (x2, y2), (0, 255, 0), 2)

        if keypoints is not None:
            for idx, (x, y, conf) in enumerate(keypoints):
                if conf > EYE_CONFIDENCE_THRESHOLD:
                    color = (0, 0, 255) if idx in [1, 2] else (0, 255, 0)
                    cv2.circle(image_np, (int(x), int(y)), 4, color, -1)

            skeleton = [
                (0, 1), (0, 2), (1, 3), (2, 4),
                (0, 5), (0, 6), (5, 6),
                (5, 7), (7, 9), (6, 8), (8, 10),
                (5, 11), (6, 12), (11, 12),
                (11, 13), (13, 15), (12, 14), (14, 16)
            ]
            for (i, j) in skeleton:
                x1, y1, c1 = keypoints[i]
                x2, y2, c2 = keypoints[j]
                if c1 > EYE_CONFIDENCE_THRESHOLD and c2 > EYE_CONFIDENCE_THRESHOLD:
                    cv2.line(image_np, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

        state_text = f"State: {self.driver_state}"
        state_color = (0, 255, 0) if self.driver_state == "normal" else (0, 0, 255)
        cv2.putText(image_np, state_text, (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, state_color, 3)

        if self.calibrated:
            cv2.putText(image_np, "Calibrated", (15, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        return image_np

    def frame_to_base64(self, frame):
        """Convert frame to base64"""
        _, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        return base64.b64encode(buffer).decode('utf-8')

    def process_frame(self, image_array):
        """Process single frame image"""
        keypoints, bbox = self.detect_keypoints(image_array)
        result = self.analyze_driver_state(keypoints)
        annotated_frame = self.draw_annotations(image_array, keypoints, bbox)

        eye_keypoints = []
        if keypoints is not None:
            eye_indices = [YOLO_KEYPOINTS["left_eye"], YOLO_KEYPOINTS["right_eye"]]
            eye_keypoints = [
                {
                    "x": float(round(keypoints[idx][0], 1)),
                    "y": float(round(keypoints[idx][1], 1)),
                    "confidence": float(round(keypoints[idx][2], 2))
                }
                for idx in eye_indices
            ]

        return result, annotated_frame, eye_keypoints

    def reset(self):
        """Reset all counters to initial state"""
        self.blink_count = 0
        self.tired_frame_count = 0
        self.last_blink_time = datetime.now()
        self.driver_state = "normal"
        self.prev_eye_state = "open"
        self.frame_count = 0