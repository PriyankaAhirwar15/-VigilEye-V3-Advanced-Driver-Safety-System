# -----------------------------------------
#   VigilEye-V3  |  alcohol_detector.py
#   Alcohol Impairment Detection
#   Camera based - no hardware needed!
# -----------------------------------------

import time
import numpy as np
from collections import deque

# Signal thresholds
HEAD_SWAY_THRESHOLD  = 0.08   # head wobble limit
BLINK_SLOW_THRESHOLD = 0.35   # very slow blink EAR
GAZE_DRIFT_THRESHOLD = 0.20   # random gaze jump
PERCLOS_DRUNK_THRESH = 40.0   # % eyes closed
YAWN_DRUNK_THRESH    = 4      # yawns per minute

# How many signals needed to trigger alcohol alert
SIGNALS_NEEDED       = 3

# Buffer sizes
BUFFER_SIZE          = 90     # 3 seconds at 30fps
HEAD_BUFFER_SIZE     = 60     # 2 seconds


class AlcoholDetector:
    """
    Detects alcohol impairment through 5 behavioural signals.
    No breathalyzer needed - pure camera analysis!
    """

    def __init__(self):
        self.ear_buffer       = deque(maxlen=BUFFER_SIZE)
        self.gaze_x_buffer    = deque(maxlen=BUFFER_SIZE)
        self.gaze_y_buffer    = deque(maxlen=BUFFER_SIZE)
        self.head_x_buffer    = deque(maxlen=HEAD_BUFFER_SIZE)
        self.head_y_buffer    = deque(maxlen=HEAD_BUFFER_SIZE)
        self.yawn_times       = []

        self.eye_sway_detected   = False
        self.head_sway_detected  = False
        self.microsleep_detected = False
        self.gaze_drift_detected = False
        self.yawn_spike_detected = False

        self.impairment_score = 0.0
        self.alert_active     = False
        self.last_alert_time  = 0
        self.alert_cooldown   = 8

        print("[VigilEye-V3] Alcohol detector ready!")

    def _update_buffers(self, data):
        self.ear_buffer.append(data.get("EAR", 0.3))
        self.gaze_x_buffer.append(data.get("gaze_x", 0.0))
        self.gaze_y_buffer.append(data.get("gaze_y", 0.0))
        if data.get("yawning"):
            self.yawn_times.append(time.time())
        now = time.time()
        self.yawn_times = [t for t in self.yawn_times if now - t <= 60]

    def _detect_eye_sway(self):
        if len(self.ear_buffer) < 30:
            return False
        ear_values = list(self.ear_buffer)
        ear_std    = np.std(ear_values)
        ear_mean   = np.mean(ear_values)
        return ear_mean < 0.26 and ear_std > 0.04

    def _detect_head_sway(self, gaze_x, gaze_y):
        self.head_x_buffer.append(gaze_x)
        self.head_y_buffer.append(gaze_y)
        if len(self.head_x_buffer) < 20:
            return False
        x_std = np.std(list(self.head_x_buffer))
        y_std = np.std(list(self.head_y_buffer))
        return x_std > HEAD_SWAY_THRESHOLD or y_std > HEAD_SWAY_THRESHOLD

    def _detect_microsleep(self, perclos):
        return perclos >= PERCLOS_DRUNK_THRESH

    def _detect_gaze_drift(self):
        if len(self.gaze_x_buffer) < 30:
            return False
        gx_std = np.std(list(self.gaze_x_buffer))
        gy_std = np.std(list(self.gaze_y_buffer))
        return gx_std > GAZE_DRIFT_THRESHOLD or gy_std > GAZE_DRIFT_THRESHOLD

    def _detect_yawn_spike(self):
        return len(self.yawn_times) >= YAWN_DRUNK_THRESH

    def _calculate_impairment_score(self, signals_active):
        base_score = (signals_active / 5) * 100
        if signals_active >= 4:
            base_score = min(100, base_score * 1.3)
        elif signals_active >= 3:
            base_score = min(100, base_score * 1.15)
        return round(base_score, 1)

    def detect(self, data):
        """
        Main detection function - call every frame.
        Accepts a dict with keys: EAR, MAR, PERCLOS, gaze_x, gaze_y, yawning.
        Returns impairment info dict.
        """
        self._update_buffers(data)

        self.eye_sway_detected   = self._detect_eye_sway()
        self.head_sway_detected  = self._detect_head_sway(
            data.get("gaze_x", 0), data.get("gaze_y", 0)
        )
        self.microsleep_detected = self._detect_microsleep(
            data.get("PERCLOS", 0)
        )
        self.gaze_drift_detected = self._detect_gaze_drift()
        self.yawn_spike_detected = self._detect_yawn_spike()

        signals = [
            self.eye_sway_detected,
            self.head_sway_detected,
            self.microsleep_detected,
            self.gaze_drift_detected,
            self.yawn_spike_detected,
        ]
        signals_active        = sum(signals)
        self.impairment_score = self._calculate_impairment_score(signals_active)
        alcohol_alert         = signals_active >= SIGNALS_NEEDED

        now          = time.time()
        should_alert = (
            alcohol_alert and
            now - self.last_alert_time > self.alert_cooldown
        )
        if should_alert:
            self.last_alert_time = now
            self.alert_active    = True
        elif not alcohol_alert:
            self.alert_active = False

        return {
            "impairment_score" : self.impairment_score,
            "signals_active"   : signals_active,
            "alcohol_alert"    : alcohol_alert,
            "should_alert"     : should_alert,
            "eye_sway"         : self.eye_sway_detected,
            "head_sway"        : self.head_sway_detected,
            "microsleep"       : self.microsleep_detected,
            "gaze_drift"       : self.gaze_drift_detected,
            "yawn_spike"       : self.yawn_spike_detected,
        }

    def get_status_label(self):
        if self.impairment_score >= 80:
            return "DRUNK - STOP DRIVING NOW!"
        elif self.impairment_score >= 60:
            return "Likely impaired - Pull over!"
        elif self.impairment_score >= 40:
            return "Possible impairment detected"
        else:
            return "No impairment detected"

    def get_active_signals(self):
        active = []
        if self.eye_sway_detected:   active.append("Eye sway")
        if self.head_sway_detected:  active.append("Head sway")
        if self.microsleep_detected: active.append("Micro-sleep")
        if self.gaze_drift_detected: active.append("Gaze drift")
        if self.yawn_spike_detected: active.append("Yawn spike")
        return active if active else ["None"]
