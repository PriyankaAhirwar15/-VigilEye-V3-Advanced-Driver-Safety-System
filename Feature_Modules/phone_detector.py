# -----------------------------------------
#   VigilEye-V3  |  phone_detector.py
#   Phone Detection using YOLOv8
# -----------------------------------------

import cv2
from ultralytics import YOLO
import time

# YOLOv8 detects 80 objects - phone is class 67
PHONE_CLASS_ID   = 67
PHONE_CLASS_NAME = "cell phone"
CONFIDENCE_MIN   = 0.45   # minimum confidence to count as detection

# Alert cooldown
last_phone_alert = 0
PHONE_COOLDOWN   = 4      # seconds between alerts

class PhoneDetector:
    """
    Uses YOLOv8 to detect if driver is holding a phone.
    Downloads model automatically on first run!
    """

    def __init__(self):
        print("[VigilEye-V3] Loading YOLOv8 model...")
        self.model       = YOLO("yolov8n.pt")  # nano = fastest
        self.phone_count = 0
        self.total_frames = 0
        print("[VigilEye-V3] YOLOv8 model loaded!")

    def detect(self, frame):
        """
        Runs YOLOv8 on frame and returns phone detection result.
        Returns detection info and annotated frame.
        """
        self.total_frames += 1

        results = self.model(
            frame,
            classes   = [PHONE_CLASS_ID],
            conf      = CONFIDENCE_MIN,
            verbose   = False
        )

        phone_detected   = False
        phone_confidence = 0.0
        boxes            = []

        for result in results:
            for box in result.boxes:
                cls  = int(box.cls[0])
                conf = float(box.conf[0])

                if cls == PHONE_CLASS_ID and conf >= CONFIDENCE_MIN:
                    phone_detected   = True
                    phone_confidence = conf
                    x1, y1, x2, y2  = map(int, box.xyxy[0])
                    boxes.append((x1, y1, x2, y2, conf))

        if phone_detected:
            self.phone_count += 1
            frame = self.draw_phone_alert(frame, boxes)

        return frame, {
            "phone_detected"   : phone_detected,
            "phone_confidence" : round(phone_confidence * 100, 1),
            "phone_boxes"      : boxes,
            "total_detections" : self.phone_count,
        }

    def draw_phone_alert(self, frame, boxes):
        """
        Draws red box around detected phone on frame
        """
        h, w = frame.shape[:2]

        for (x1, y1, x2, y2, conf) in boxes:
            # Red box around phone
            cv2.rectangle(frame, (x1, y1), (x2, y2),
                         (0, 0, 255), 2)

            # Label
            label = f"PHONE! {conf*100:.0f}%"
            cv2.rectangle(frame,
                         (x1, y1 - 25),
                         (x1 + len(label) * 10, y1),
                         (0, 0, 255), -1)
            cv2.putText(frame, label,
                       (x1, y1 - 6),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       0.6, (255, 255, 255), 2)

        # Big warning at top
        cv2.rectangle(frame, (0, 85), (w, 120),
                     (0, 0, 200), -1)
        cv2.putText(frame,
                   "WARNING: PHONE DETECTED WHILE DRIVING!",
                   (10, 108),
                   cv2.FONT_HERSHEY_SIMPLEX,
                   0.6, (255, 255, 255), 2)

        return frame

    def get_stats(self):
        """Returns phone detection statistics"""
        rate = (self.phone_count / max(self.total_frames, 1)) * 100
        return {
            "total_phone_detections": self.phone_count,
            "detection_rate"        : round(rate, 1),
        }