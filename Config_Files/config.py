# -----------------------------------------
#   VigilEye-V3  |  config.py
#   All thresholds and settings live here
# -----------------------------------------

PROJECT_NAME     = "VigilEye-V3"
VERSION          = "3.0"

# -- Eye settings --------------------------
EAR_THRESHOLD       = 0.22   # below this = eyes closing
EAR_CONSEC_FRAMES   = 20     # how many frames before drowsy alert

# -- Mouth settings ------------------------
MAR_THRESHOLD       = 0.45   # above this = yawning
MAR_CONSEC_FRAMES   = 15     # how many frames before yawn alert
MAX_YAWNS_PER_MIN   = 3      # more than this = fatigue warning

# -- Gaze / distraction settings -----------
GAZE_X_THRESHOLD    = 0.15   # horizontal look away limit
GAZE_Y_THRESHOLD    = 0.25   # vertical look away limit
GAZE_CONSEC_FRAMES  = 25     # frames before distraction alert

# -- PERCLOS settings ----------------------
PERCLOS_THRESHOLD   = 30.0   # % eye closure over 60 frames = danger

# -- Fatigue Score weights -----------------
# These 4 must add up to 1.0
WEIGHT_EAR          = 0.35
WEIGHT_PERCLOS      = 0.30
WEIGHT_MAR          = 0.20
WEIGHT_GAZE         = 0.15

# -- Alert severity levels -----------------
SEVERITY_MILD       = 40     # fatigue score 40–70  → mild warning
SEVERITY_MODERATE   = 70     # fatigue score 70–90  → moderate alert
SEVERITY_CRITICAL   = 90     # fatigue score 90+    → critical alarm

# -- Alert messages ------------------------
MSG_MILD       = "Stay alert! You seem a little tired."
MSG_MODERATE   = "Warning! Drowsiness detected. Take a break soon."
MSG_CRITICAL   = "DANGER! Pull over immediately. You are too drowsy!"
MSG_YAWN       = "Yawning detected. Fatigue level rising."
MSG_DISTRACTED = "Eyes on the road! Distraction detected."

# -- Session logging -----------------------
ENABLE_LOGGING      = True
# Updated path to match your "Session Data" folder structure
LOG_FILE            = "Session Data/vigileye_session_log.csv"

# -- Dashboard settings --------------------
DASHBOARD_TITLE     = "VigilEye-V3 | Live Fatigue Monitor"
CHART_HISTORY       = 100    # how many frames to show on live chart