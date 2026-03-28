# ---------------------------------------------------------
#   VigilEye-V3  |  logger.py
#   Session Data Logger - saves numerical data to CSV
# ---------------------------------------------------------

import csv
import os
import re
import time
from datetime import datetime

# UPDATED IMPORT: Pointing to the new Config_Files folder
from Config_Files.config import ENABLE_LOGGING, LOG_FILE

def _strip_emoji(text):
    """
    Removes emoji or unicode symbols so the CSV file writes 
    safely on Windows systems without encoding errors.
    """
    return re.sub(r'[^\x00-\x7F]+', '', str(text)).strip()

# Column headers for the CSV file
HEADERS = [
    "timestamp",
    "elapsed_seconds",
    "EAR",
    "MAR",
    "PERCLOS",
    "gaze_x",
    "gaze_y",
    "fatigue_score",
    "eye_component",
    "perclos_component",
    "mouth_component",
    "gaze_component",
    "drowsy",
    "yawning",
    "distracted",
    "severity_label",
    "alert_triggered",
]

class SessionLogger:
    """
    Logs data from every processed frame into a CSV file.
    Privacy focused: No images or video are saved, only mathematical values.
    """

    def __init__(self):
        # Configuration settings from Config_Files/config.py
        self.enabled     = ENABLE_LOGGING
        self.log_file    = LOG_FILE  
        self.start_time  = time.time()
        self.frame_count = 0
        self.alert_count = 0
        self.yawn_count  = 0
        self._setup_file()

    def _setup_file(self):
        """
        Creates the CSV file and writes the header if logging is enabled.
        Automatically creates the 'Session Data' directory if it doesn't exist.
        """
        if not self.enabled:
            return
            
        # Ensure the directory exists before trying to create the file
        log_dir = os.path.dirname(self.log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)

        file_exists = os.path.isfile(self.log_file)
        
        # Open in append mode so we don't delete previous data if same filename is used
        self.file   = open(self.log_file, "a", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.file, fieldnames=HEADERS)
        
        if not file_exists:
            self.writer.writeheader()
            self.file.flush()
        print(f"[VigilEye-V3] Logger initialized. Saving to: {self.log_file}")

    def log(self, predictor_data, score_data, severity_label, alert_triggered):
        """
        Records data for the current frame.
        Only executes if ENABLE_LOGGING is True in config.py.
        """
        self.frame_count += 1
        if alert_triggered:
            self.alert_count += 1
        if predictor_data.get("yawning"):
            self.yawn_count += 1

        if not self.enabled:
            return

        # Prepare the row data
        row = {
            "timestamp"         : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed_seconds"   : round(time.time() - self.start_time, 1),
            "EAR"               : predictor_data.get("EAR", 0),
            "MAR"               : predictor_data.get("MAR", 0),
            "PERCLOS"           : predictor_data.get("PERCLOS", 0),
            "gaze_x"            : predictor_data.get("gaze_x", 0),
            "gaze_y"            : predictor_data.get("gaze_y", 0),
            "fatigue_score"     : score_data.get("fatigue_score", 0),
            "eye_component"     : score_data.get("eye_component", 0),
            "perclos_component" : score_data.get("perclos_component", 0),
            "mouth_component"   : score_data.get("mouth_component", 0),
            "gaze_component"    : score_data.get("gaze_component", 0),
            "drowsy"            : predictor_data.get("drowsy", False),
            "yawning"           : predictor_data.get("yawning", False),
            "distracted"        : predictor_data.get("distracted", False),
            "severity_label"    : _strip_emoji(severity_label),
            "alert_triggered"   : alert_triggered,
        }
        
        self.writer.writerow(row)
        self.file.flush() # Forces writing to disk immediately

    def get_session_summary(self):
        """Returns a summarized dictionary of the current recording session."""
        elapsed = round(time.time() - self.start_time, 1)
        return {
            "total_frames"     : self.frame_count,
            "session_duration" : f"{int(elapsed // 60)}m {int(elapsed % 60)}s",
            "total_alerts"     : self.alert_count,
            "total_yawns"      : self.yawn_count,
            "log_file"         : self.log_file,
        }

    def close(self):
        """Safely closes the CSV file when the application stops."""
        if self.enabled and hasattr(self, "file"):
            self.file.close()
            print(f"[VigilEye-V3] Session log saved successfully.")