# ---------------------------------------------------------
#   VigilEye-V3  |  night_mode.py
#   Night Mode and Low Light Detection
#   Updated: Fixed 'Read-Only' buffer error for Gradio
# ---------------------------------------------------------

import cv2
import numpy as np

# Light level thresholds
VERY_DARK_THRESHOLD  = 40   # below this = very dark
DARK_THRESHOLD       = 80   # below this = dark
NORMAL_THRESHOLD     = 180  # above this = bright

def get_brightness(frame):
    """
    Measures average brightness of the frame.
    Returns a value from 0 (pitch black) to 255 (pure white)
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray))

def get_light_status(brightness):
    """
    Returns light condition label based on brightness value
    """
    if brightness < VERY_DARK_THRESHOLD:
        return "very_dark", "Very Dark - Night Mode ON"
    elif brightness < DARK_THRESHOLD:
        return "dark", "Low Light - Enhanced Mode ON"
    elif brightness > NORMAL_THRESHOLD:
        return "bright", "Good Lighting"
    else:
        return "normal", "Normal Lighting"

def enhance_frame(frame, brightness):
    """
    Automatically enhances frame based on light condition.
    Makes dark frames brighter so face mesh works better!
    """
    status, _ = get_light_status(brightness)

    # We use a copy to ensure we don't modify the original buffer incorrectly
    output = frame.copy()

    if status == "very_dark":
        # Very dark - apply strong enhancement
        alpha = 2.5    # contrast
        beta  = 60     # brightness boost
        output = cv2.convertScaleAbs(output, alpha=alpha, beta=beta)
        output = apply_clahe(output)

    elif status == "dark":
        # Dark - apply medium enhancement
        alpha = 1.8
        beta  = 40
        output = cv2.convertScaleAbs(output, alpha=alpha, beta=beta)
        output = apply_clahe(output)

    return output

def apply_clahe(frame):
    """
    CLAHE = Contrast Limited Adaptive Histogram Equalization
    Auto-adjusts image parts for better detail in dark areas.
    """
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

def draw_light_status(frame, brightness):
    """
    Draws light status indicator on the frame.
    """
    status, label = get_light_status(brightness)

    # Choose color based on status (BGR format)
    color_map = {
        "very_dark" : (0, 100, 255),  # Orange/Red
        "dark"      : (0, 200, 255),  # Yellow
        "normal"    : (0, 255, 100),  # Green
        "bright"    : (255, 255, 255),# White
    }
    color = color_map.get(status, (255, 255, 255))

    # Get dimensions
    h, w = frame.shape[:2]
    
    # Draw status text on frame (Ensuring it is a writeable copy)
    cv2.putText(frame, f"Light: {label}",
                (10, h - 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45, color, 1)
    cv2.putText(frame, f"Brightness: {brightness:.0f}/255",
                (10, h - 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45, color, 1)

    return frame

def process_night_mode(frame):
    """
    Main function - call this every frame BEFORE prediction.
    Returns enhanced frame + brightness info.
    """
    # CRITICAL FIX: Convert frame to a writeable NumPy array copy
    # This prevents the 'readonly' error in Gradio/OpenCV
    working_frame = np.array(frame).copy()

    brightness          = get_brightness(working_frame)
    status, label       = get_light_status(brightness)
    
    # Enhance the copy
    enhanced_frame      = enhance_frame(working_frame, brightness)
    # Draw text on the enhanced copy
    enhanced_frame      = draw_light_status(enhanced_frame, brightness)

    return enhanced_frame, {
        "brightness"   : round(brightness, 1),
        "light_status" : status,
        "light_label"  : label,
        "night_mode_on": status in ["very_dark", "dark"],
    }