# ---------------------------------------------------------
#   VigilEye-V3  |  fatigue_score.py
#   Weighted Fatigue Score Calculator (0-100)
# ---------------------------------------------------------

# Fixed Import Path to point to the new Config_Files folder
from Config_Files.config import (
    EAR_THRESHOLD, PERCLOS_THRESHOLD,
    MAR_THRESHOLD, GAZE_X_THRESHOLD,
    WEIGHT_EAR, WEIGHT_PERCLOS,
    WEIGHT_MAR, WEIGHT_GAZE
)

def normalize(value, min_val, max_val):
    """Converts any value into 0.0 to 1.0 range for standardized scoring"""
    if max_val == min_val:
        return 0.0
    result = (value - min_val) / (max_val - min_val)
    return max(0.0, min(1.0, result))

def calculate_ear_score(ear):
    """Lower EAR means eyes are more closed, which increases drowsiness score"""
    return normalize(EAR_THRESHOLD - ear, -0.35, EAR_THRESHOLD)

def calculate_perclos_score(perclos):
    """Higher PERCLOS means eyes were closed for a longer time"""
    return normalize(perclos, 0, 80)

def calculate_mar_score(mar):
    """Higher MAR indicates yawning, which increases fatigue score"""
    return normalize(mar - MAR_THRESHOLD, -0.15, 0.35)

def calculate_gaze_score(gaze_x, gaze_y):
    """Higher gaze offset means the driver is distracted and looking away"""
    combined = (gaze_x + gaze_y) / 2
    return normalize(combined, 0, 0.4)

def calculate_fatigue_score(ear, perclos, mar, gaze_x, gaze_y):
    """
    Master function: Combines EAR, PERCLOS, MAR, and Gaze into one final score.
    Returns a final score between 0 and 100.
    """
    s_ear     = calculate_ear_score(ear)
    s_perclos = calculate_perclos_score(perclos)
    s_mar     = calculate_mar_score(mar)
    s_gaze    = calculate_gaze_score(gaze_x, gaze_y)

    # Applying weights from the configuration file
    raw_score = (
        WEIGHT_EAR     * s_ear     +
        WEIGHT_PERCLOS * s_perclos +
        WEIGHT_MAR     * s_mar     +
        WEIGHT_GAZE    * s_gaze
    )

    final_score = round(raw_score * 100, 1)

    return final_score, {
        "ear_score"     : round(s_ear * 100, 1),
        "perclos_score" : round(s_perclos * 100, 1),
        "mar_score"     : round(s_mar * 100, 1),
        "gaze_score"    : round(s_gaze * 100, 1),
    }

def get_score_breakdown(ear, perclos, mar, gaze_x, gaze_y):
    """
    Returns a readable summary of all components to be displayed on the dashboard.
    """
    score, parts = calculate_fatigue_score(ear, perclos, mar, gaze_x, gaze_y)

    return {
        "fatigue_score"     : score,
        "eye_component"     : parts["ear_score"],
        "perclos_component" : parts["perclos_score"],
        "mouth_component"   : parts["mar_score"],
        "gaze_component"    : parts["gaze_score"],
    }