# ---------------------------------------------------------
#   VigilEye-V3  |  predictor.py
#   MediaPipe Face Mesh (EAR/MAR/Gaze Analysis)
# ---------------------------------------------------------

import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance

# UPDATED IMPORT: Pointing to the new Config_Files directory
# This pulls the specific thresholds needed for the calculations
from Config_Files.config import (
    EAR_THRESHOLD, 
    MAR_THRESHOLD, 
    GAZE_X_THRESHOLD, 
    GAZE_Y_THRESHOLD
)

# Initialize MediaPipe Face Mesh with high-fidelity settings
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Landmark indices for calculation (Standard MediaPipe indices)
LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
MOUTH     = [61, 291, 39, 181, 0, 17, 269, 405]

# Gaze/Head orientation indices
NOSE_TIP  = 1
CHIN      = 152
LEFT_EAR  = 234
RIGHT_EAR = 454

def eye_aspect_ratio(landmarks, eye_indices, w, h):
    """
    Calculates Eye Aspect Ratio (EAR) to detect if eyes are closing.
    Uses Euclidean distance between vertical and horizontal eye points.
    """
    pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in eye_indices]
    # Vertical distances
    A = distance.euclidean(pts[1], pts[5])
    B = distance.euclidean(pts[2], pts[4])
    # Horizontal distance
    C = distance.euclidean(pts[0], pts[3])
    return (A + B) / (2.0 * C)

def mouth_aspect_ratio(landmarks, w, h):
    """
    Calculates Mouth Aspect Ratio (MAR) to detect yawning.
    """
    pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in MOUTH]
    # Vertical mouth points
    A = distance.euclidean(pts[2], pts[6])
    B = distance.euclidean(pts[3], pts[7])
    # Horizontal mouth points
    C = distance.euclidean(pts[0], pts[1])
    return (A + B) / (2.0 * C)

def get_gaze_direction(landmarks, w, h):
    """
    Calculates face offset (Gaze) to determine if the driver is looking away.
    Returns X and Y offset values.
    """
    nose  = np.array([landmarks[NOSE_TIP].x * w, landmarks[NOSE_TIP].y * h])
    chin  = np.array([landmarks[CHIN].x * w, landmarks[CHIN].y * h])
    left  = np.array([landmarks[LEFT_EAR].x * w, landmarks[LEFT_EAR].y * h])
    right = np.array([landmarks[RIGHT_EAR].x * w, landmarks[RIGHT_EAR].y * h])
    
    face_width = distance.euclidean(left, right)
    face_height = distance.euclidean(nose, chin)
    
    mid_x = (left[0] + right[0]) / 2
    offset_x = abs(nose[0] - mid_x) / (face_width + 1e-6)
    offset_y = abs(nose[1] - chin[1]) / (face_height + 1e-6)
    
    return offset_x, offset_y

def predict(frame, perclos_buffer):
    """
    Main function to process each video frame and extract AI insights.
    Updates the perclos_buffer and returns a structured data dictionary.
    """
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    # Initialize default data structure
    data = {
        "face_detected": False,
        "EAR": 0.0,
        "MAR": 0.0,
        "gaze_x": 0.0,
        "gaze_y": 0.0,
        "PERCLOS": 0.0,
        "drowsy": False,
        "yawning": False,
        "distracted": False,
    }

    if not result.multi_face_landmarks:
        return data

    # Extract landmarks for the first face detected
    lm = result.multi_face_landmarks[0].landmark
    data["face_detected"] = True

    # Calculate individual Eye Aspect Ratios
    ear_left  = eye_aspect_ratio(lm, LEFT_EYE, w, h)
    ear_right = eye_aspect_ratio(lm, RIGHT_EYE, w, h)
    
    # Average EAR and round for clean logs
    data["EAR"] = round((ear_left + ear_right) / 2.0, 3)
    data["MAR"] = round(mouth_aspect_ratio(lm, w, h), 3)

    # Gaze detection
    gx, gy = get_gaze_direction(lm, w, h)
    data["gaze_x"] = round(gx, 3)
    data["gaze_y"] = round(gy, 3)

    # PERCLOS logic (Percentage of Eye Closure time)
    perclos_buffer.append(1 if data["EAR"] < EAR_THRESHOLD else 0)
    if len(perclos_buffer) > 60:
        perclos_buffer.pop(0)
        
    data["PERCLOS"] = round(sum(perclos_buffer) / len(perclos_buffer) * 100, 1)

    # Comparison flags using imported constants from Config_Files
    data["drowsy"]     = data["EAR"] < EAR_THRESHOLD
    data["yawning"]    = data["MAR"] > MAR_THRESHOLD
    data["distracted"] = gx > GAZE_X_THRESHOLD or gy > GAZE_Y_THRESHOLD

    return data