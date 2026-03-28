# -----------------------------------------
#   VigilEye-V3  |  face_recognition_module.py
#   Driver Face Recognition using OpenCV only
#   No dlib, No DeepFace, Zero conflicts!
# -----------------------------------------

import cv2
import os
import numpy as np

# Updated to point to your Database/Drivers folder for professional organization
DRIVERS_FOLDER = os.path.join("Database", "Drivers")
UNKNOWN_LABEL  = "Unknown Driver"

class DriverRecognizer:
    """
    Recognizes drivers using OpenCV LBPH algorithm.
    Works 100% offline, no extra libraries needed!
    """

    def __init__(self):
        self.current_driver   = UNKNOWN_LABEL
        self.recognition_conf = 0.0
        self.frame_skip       = 0
        self.known_names      = []
        self.face_cascade     = cv2.CascadeClassifier(
            cv2.data.haarcascades +
            "haarcascade_frontalface_default.xml"
        )
        # Professional Check: Ensure the recognizer is available
        try:
            self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        except AttributeError:
            print("[ERROR] opencv-contrib-python is required for Face Recognition!")
            print("Run: pip install opencv-contrib-python")
            
        self.trained    = False

        # Create drivers folder inside Database
        os.makedirs(DRIVERS_FOLDER, exist_ok=True)

        # Load existing drivers
        self._load_drivers()
        print("[VigilEye-V3] Face recognition ready!")
        print(f"[VigilEye-V3] Known drivers: {self.known_names}")

    def _load_drivers(self):
        """Loads registered drivers and trains recognizer"""
        if not os.path.exists(DRIVERS_FOLDER):
            return

        photos = [
            f for f in os.listdir(DRIVERS_FOLDER)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        if len(photos) == 0:
            return

        self.known_names = [f.rsplit('.', 1)[0] for f in photos]

        faces  = []
        labels = []

        for idx, photo in enumerate(photos):
            path  = os.path.join(DRIVERS_FOLDER, photo)
            img   = cv2.imread(path)
            if img is None:
                continue
            gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            detected = self.face_cascade.detectMultiScale(
                gray, 1.1, 5, minSize=(60, 60)
            )
            for (x, y, w, h) in detected:
                faces.append(gray[y:y+h, x:x+w])
                labels.append(idx)

        if len(faces) > 0:
            self.recognizer.train(faces, np.array(labels))
            self.trained = True
            print(f"[VigilEye-V3] Trained on {len(faces)} faces!")

    def register_driver(self, frame, driver_name):
        """
        Registers a new driver from current camera frame.
        """
        if not driver_name or driver_name.strip() == "":
            return "Please enter a driver name!"

        driver_name = driver_name.strip()

        gray     = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detected = self.face_cascade.detectMultiScale(
            gray, 1.1, 5, minSize=(60, 60)
        )

        if len(detected) == 0:
            return "No face detected! Please look at camera!"

        if len(detected) > 1:
            return "Multiple faces! Only one person please!"

        # Save driver photo into Database/Drivers
        photo_path = os.path.join(
            DRIVERS_FOLDER, f"{driver_name}.jpg"
        )
        cv2.imwrite(photo_path, frame)

        # Retrain recognizer
        self._load_drivers()

        return f"Driver '{driver_name}' registered successfully!"

    def recognize(self, frame):
        """
        Recognizes who is in the frame.
        Runs every 10 frames for performance.
        """
        if not self.trained:
            return frame, {
                "driver_name"   : UNKNOWN_LABEL,
                "confidence"    : 0.0,
                "face_detected" : False,
            }

        self.frame_skip += 1

        # Only run every 10 frames for speed
        if self.frame_skip % 10 != 0:
            return frame, {
                "driver_name"   : self.current_driver,
                "confidence"    : self.recognition_conf,
                "face_detected" : self.current_driver != UNKNOWN_LABEL,
            }

        try:
            gray     = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            detected = self.face_cascade.detectMultiScale(
                gray, 1.1, 5, minSize=(60, 60)
            )

            if len(detected) == 0:
                self.current_driver   = UNKNOWN_LABEL
                self.recognition_conf = 0.0
            else:
                for (x, y, w, h) in detected:
                    face        = gray[y:y+h, x:x+w]
                    label, conf = self.recognizer.predict(face)

                    # Lower confidence = better match in LBPH
                    if conf < 80:
                        self.current_driver   = self.known_names[label]
                        self.recognition_conf = round(
                            100 - conf, 1
                        )
                    else:
                        self.current_driver   = UNKNOWN_LABEL
                        self.recognition_conf = 0.0
                    break

        except Exception as e:
            print(f"[VigilEye-V3] Recognition error: {e}")

        # Draw name on frame
        frame = self.draw_driver_name(frame)

        return frame, {
            "driver_name"   : self.current_driver,
            "confidence"    : self.recognition_conf,
            "face_detected" : self.current_driver != UNKNOWN_LABEL,
        }

    def draw_driver_name(self, frame):
        """Draws driver name on frame"""
        h, w = frame.shape[:2]

        if self.current_driver == UNKNOWN_LABEL:
            color = (0, 165, 255) # Orange for unknown
            label = "Unknown Driver"
        else:
            color = (0, 255, 100) # Green for recognized
            label = (f"Driver: {self.current_driver} "
                    f"({self.recognition_conf:.0f}%)")

        cv2.putText(frame, label,
                    (w - 350, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, color, 2)
        return frame

    def get_all_drivers(self):
        """Returns list of all registered drivers"""
        self._load_drivers()
        if len(self.known_names) == 0:
            return "No drivers registered yet!"
        return "\n".join([
            f"Driver {i+1}: {name}"
            for i, name in enumerate(self.known_names)
        ])