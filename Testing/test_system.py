# =============================================================================
#   VigilEye-V3  |  test_system.py
#   Full System Test Suite — Tests all modules independently + integrated
#   Run from your project root: python test_system.py
# =============================================================================

import sys
import os
import time
import numpy as np

# ── CRITICAL: Make sure we run from the project root ──────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

# ── Terminal colors ────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

PASS_S = f"{GREEN}[PASS]{RESET}"
FAIL_S = f"{RED}[FAIL]{RESET}"
INFO_S = f"{CYAN}[INFO]{RESET}"

results = []

def section(title):
    print(f"\n{BOLD}{CYAN}{'='*60}{RESET}")
    print(f"{BOLD}{CYAN}  {title}{RESET}")
    print(f"{BOLD}{CYAN}{'='*60}{RESET}")

def test(name, fn):
    try:
        msg = fn()
        results.append(("PASS", name))
        print(f"  {PASS_S}  {name}")
        if msg:
            print(f"         {INFO_S} {msg}")
    except AssertionError as e:
        results.append(("FAIL", name))
        print(f"  {FAIL_S}  {name}")
        print(f"         {RED}-> {e}{RESET}")
    except Exception as e:
        results.append(("FAIL", name))
        print(f"  {FAIL_S}  {name}")
        print(f"         {RED}-> {type(e).__name__}: {e}{RESET}")

# ── Frame helpers ──────────────────────────────────────────────────────────────
def black_frame(h=480, w=640):
    return np.zeros((h, w, 3), dtype=np.uint8)

def gray_frame(brightness=128, h=480, w=640):
    return np.full((h, w, 3), brightness, dtype=np.uint8)

def face_frame():
    import cv2
    f = np.full((480, 640, 3), 60, dtype=np.uint8)
    cv2.ellipse(f, (320, 240), (100, 130), 0, 0, 360, (180, 140, 100), -1)
    return f


# =============================================================================
#   1. CONFIG MODULE
# =============================================================================
section("1. CONFIG MODULE")

def test_config_imports():
    import config
    for attr in [
        "EAR_THRESHOLD", "EAR_CONSEC_FRAMES",
        "MAR_THRESHOLD", "MAR_CONSEC_FRAMES", "MAX_YAWNS_PER_MIN",
        "GAZE_X_THRESHOLD", "GAZE_Y_THRESHOLD", "GAZE_CONSEC_FRAMES",
        "PERCLOS_THRESHOLD",
        "WEIGHT_EAR", "WEIGHT_PERCLOS", "WEIGHT_MAR", "WEIGHT_GAZE",
        "SEVERITY_MILD", "SEVERITY_MODERATE", "SEVERITY_CRITICAL",
        "MSG_MILD", "MSG_MODERATE", "MSG_CRITICAL", "MSG_YAWN", "MSG_DISTRACTED",
        "ENABLE_LOGGING", "LOG_FILE", "DASHBOARD_TITLE", "CHART_HISTORY",
    ]:
        assert hasattr(config, attr), f"Missing: {attr}"
    return "All config attributes present"

def test_config_weights_sum():
    from config import WEIGHT_EAR, WEIGHT_PERCLOS, WEIGHT_MAR, WEIGHT_GAZE
    total = WEIGHT_EAR + WEIGHT_PERCLOS + WEIGHT_MAR + WEIGHT_GAZE
    assert abs(total - 1.0) < 1e-6, f"Weights sum to {total}, not 1.0"
    return f"Weights sum = {total:.2f}"

def test_config_severity_order():
    from config import SEVERITY_MILD, SEVERITY_MODERATE, SEVERITY_CRITICAL
    assert SEVERITY_MILD < SEVERITY_MODERATE < SEVERITY_CRITICAL
    return f"MILD={SEVERITY_MILD} < MODERATE={SEVERITY_MODERATE} < CRITICAL={SEVERITY_CRITICAL}"

def test_config_thresholds():
    from config import EAR_THRESHOLD, MAR_THRESHOLD, PERCLOS_THRESHOLD
    assert 0 < EAR_THRESHOLD < 1
    assert 0 < MAR_THRESHOLD < 1
    assert 0 < PERCLOS_THRESHOLD < 100
    return "All thresholds in valid range"

def test_config_messages():
    from config import MSG_MILD, MSG_MODERATE, MSG_CRITICAL, MSG_YAWN, MSG_DISTRACTED
    for m in [MSG_MILD, MSG_MODERATE, MSG_CRITICAL, MSG_YAWN, MSG_DISTRACTED]:
        assert isinstance(m, str) and len(m) > 0
    return "All 5 alert messages are non-empty strings"

test("Config imports all constants",       test_config_imports)
test("Fatigue weights sum to 1.0",         test_config_weights_sum)
test("Severity levels in correct order",   test_config_severity_order)
test("Thresholds within valid ranges",     test_config_thresholds)
test("All alert messages non-empty",       test_config_messages)


# =============================================================================
#   2. FATIGUE SCORE MODULE
# =============================================================================
section("2. FATIGUE SCORE MODULE")

def test_fatigue_imports():
    from fatigue_score import (
        calculate_fatigue_score, get_score_breakdown, normalize,
        calculate_ear_score, calculate_perclos_score,
        calculate_mar_score, calculate_gaze_score
    )
    return "All 7 functions imported"

def test_normalize():
    from fatigue_score import normalize
    assert normalize(0,   0, 100) == 0.0
    assert normalize(100, 0, 100) == 1.0
    assert normalize(50,  0, 100) == 0.5
    assert normalize(-10, 0, 100) == 0.0
    assert normalize(110, 0, 100) == 1.0
    assert normalize(5,   5,   5) == 0.0
    return "Clamp + interpolation correct"

def test_score_range():
    from fatigue_score import calculate_fatigue_score
    score, _ = calculate_fatigue_score(0.30, 10.0, 0.30, 0.05, 0.10)
    assert 0 <= score <= 100
    return f"Normal case score = {score}"

def test_drowsy_score():
    from fatigue_score import calculate_fatigue_score
    score, _ = calculate_fatigue_score(0.10, 60.0, 0.70, 0.20, 0.30)
    assert score > 50, f"Drowsy inputs gave {score}, expected >50"
    return f"Drowsy score = {score} (>50)"

def test_safe_score():
    from fatigue_score import calculate_fatigue_score
    score, _ = calculate_fatigue_score(0.35, 2.0, 0.20, 0.02, 0.05)
    assert score < 50, f"Safe inputs gave {score}, expected <50"
    return f"Safe score = {score} (<50)"

def test_breakdown_keys():
    from fatigue_score import get_score_breakdown
    r = get_score_breakdown(0.25, 15.0, 0.40, 0.08, 0.12)
    for k in ["fatigue_score", "eye_component", "perclos_component",
              "mouth_component", "gaze_component"]:
        assert k in r, f"Missing: {k}"
    return "All 5 breakdown keys present"

def test_component_range():
    from fatigue_score import calculate_fatigue_score
    _, parts = calculate_fatigue_score(0.25, 15.0, 0.40, 0.08, 0.12)
    for k, v in parts.items():
        assert 0 <= v <= 100, f"{k}={v} out of range"
    return "All component scores in 0-100"

test("fatigue_score imports",              test_fatigue_imports)
test("normalize() clamps correctly",       test_normalize)
test("Score is within 0-100",             test_score_range)
test("Drowsy inputs give high score",      test_drowsy_score)
test("Safe inputs give low score",         test_safe_score)
test("get_score_breakdown() keys",         test_breakdown_keys)
test("Component scores in 0-100",          test_component_range)


# =============================================================================
#   3. ALERT MODULE
# =============================================================================
section("3. ALERT MODULE")

def test_alert_imports():
    from alert import trigger_alert, get_severity_label, get_severity_color, speak_async, beep
    return "All 5 functions imported"

def test_severity_labels():
    from alert import get_severity_label
    assert "SAFE"     in get_severity_label(10)
    assert "MILD"     in get_severity_label(50)
    assert "MODERATE" in get_severity_label(75)
    assert "CRITICAL" in get_severity_label(95)
    return "All 4 severity labels correct"

def test_severity_colors():
    from alert import get_severity_color
    for s in [10, 50, 75, 95]:
        c = get_severity_color(s)
        assert c.startswith("#") and len(c) == 7, f"Bad color: {c}"
    return "All severity colors are valid hex"

def test_trigger_no_crash():
    from alert import trigger_alert
    for s in [10, 50, 75, 95]:
        trigger_alert(s)
    trigger_alert(95, yawning=True, distracted=True)
    return "trigger_alert() ran without crash for all levels"

def test_cooldown():
    import alert as a
    a.last_alert_time = time.time()
    try:
        a.trigger_alert(95)
    except Exception as e:
        raise AssertionError(f"Cooldown raised: {e}")
    return "Cooldown skips alerts silently"

test("Alert module imports",               test_alert_imports)
test("Severity labels correct",            test_severity_labels)
test("Severity colors valid hex",          test_severity_colors)
test("trigger_alert() no crash",           test_trigger_no_crash)
test("Cooldown logic works",               test_cooldown)


# =============================================================================
#   4. LOGGER MODULE
# =============================================================================
section("4. LOGGER MODULE")

_P = {"EAR": 0.25, "MAR": 0.30, "PERCLOS": 10.0,
      "gaze_x": 0.05, "gaze_y": 0.08,
      "drowsy": False, "yawning": False, "distracted": False}
_S = {"fatigue_score": 30, "eye_component": 20,
      "perclos_component": 10, "mouth_component": 15, "gaze_component": 12}

def _lg():
    import config
    config.ENABLE_LOGGING = False
    from logger import SessionLogger
    return SessionLogger()

def test_logger_import():
    from logger import SessionLogger
    return "SessionLogger imported"

def test_logger_init():
    lg = _lg()
    assert lg.frame_count == 0 and lg.alert_count == 0 and lg.yawn_count == 0
    return "Counters start at zero"

def test_logger_frame_count():
    lg = _lg()
    lg.log(_P, _S, "SAFE", False)
    assert lg.frame_count == 1
    return "frame_count increments on log()"

def test_logger_alert_yawn():
    lg = _lg()
    p2 = {**_P, "yawning": True}
    s2 = {**_S, "fatigue_score": 85}
    lg.log(p2, s2, "MODERATE", True)
    lg.log(p2, s2, "MODERATE", True)
    assert lg.alert_count == 2 and lg.yawn_count == 2
    return "alert_count and yawn_count both = 2"

def test_logger_summary():
    lg = _lg()
    sm = lg.get_session_summary()
    for k in ["total_frames", "session_duration", "total_alerts",
              "total_yawns", "log_file"]:
        assert k in sm, f"Missing: {k}"
    return "All 5 summary keys present"

def test_logger_close():
    lg = _lg()
    lg.close()
    return "close() runs without crash"

test("SessionLogger imports",              test_logger_import)
test("Logger counters start at zero",      test_logger_init)
test("log() increments frame_count",       test_logger_frame_count)
test("Alert + yawn counts tracked",        test_logger_alert_yawn)
test("Session summary all keys",           test_logger_summary)
test("close() no crash",                   test_logger_close)


# =============================================================================
#   5. NIGHT MODE MODULE
# =============================================================================
section("5. NIGHT MODE MODULE")

def test_night_imports():
    from night_mode import (get_brightness, get_light_status,
                            enhance_frame, apply_clahe, process_night_mode)
    return "All 5 functions imported"

def test_brightness_black():
    from night_mode import get_brightness
    b = get_brightness(black_frame())
    assert b < 5
    return f"Black frame brightness = {b:.1f}"

def test_brightness_white():
    from night_mode import get_brightness
    b = get_brightness(np.full((480, 640, 3), 255, dtype=np.uint8))
    assert b > 250
    return f"White frame brightness = {b:.1f}"

def test_light_labels():
    from night_mode import get_light_status
    assert get_light_status(20)[0]  == "very_dark"
    assert get_light_status(60)[0]  == "dark"
    assert get_light_status(120)[0] == "normal"
    assert get_light_status(200)[0] == "bright"
    return "All 4 light levels correct"

def test_enhance_dark():
    from night_mode import enhance_frame
    d = gray_frame(30)
    e = enhance_frame(d, 30)
    assert e.mean() > d.mean()
    return f"Brightness boosted: {d.mean():.1f} -> {e.mean():.1f}"

def test_clahe():
    from night_mode import apply_clahe
    f = gray_frame(60)
    r = apply_clahe(f)
    assert r.shape == f.shape
    return f"apply_clahe() shape unchanged: {r.shape}"

def test_night_mode_keys():
    from night_mode import process_night_mode
    _, info = process_night_mode(gray_frame(100))
    for k in ["brightness", "light_status", "light_label", "night_mode_on"]:
        assert k in info
    assert isinstance(info["night_mode_on"], bool)
    return f"light_status={info['light_status']}, night_mode={info['night_mode_on']}"

test("night_mode imports",                 test_night_imports)
test("Brightness ~0 on black frame",       test_brightness_black)
test("Brightness ~255 on white frame",     test_brightness_white)
test("Light status labels correct",        test_light_labels)
test("Dark frame gets enhanced",           test_enhance_dark)
test("apply_clahe() shape unchanged",      test_clahe)
test("process_night_mode() output keys",   test_night_mode_keys)


# =============================================================================
#   6. CHARTS MODULE
# =============================================================================
section("6. CHARTS MODULE")

def test_charts_imports():
    from charts import (draw_fatigue_chart, draw_component_chart,
                        draw_gauge_chart, get_zone_color)
    return "All 4 functions imported"

def test_zone_colors():
    from charts import get_zone_color
    assert get_zone_color(10) == "#00CC44"
    assert get_zone_color(50) == "#FFD700"
    assert get_zone_color(80) == "#FF6600"
    assert get_zone_color(95) == "#FF0000"
    return "All 4 zone colors correct"

def test_fatigue_chart_empty():
    import matplotlib.pyplot as plt
    from charts import draw_fatigue_chart
    from collections import deque
    assert draw_fatigue_chart(deque()) is not None
    plt.close("all")
    return "Empty history renders without crash"

def test_fatigue_chart_data():
    import matplotlib.pyplot as plt
    from charts import draw_fatigue_chart
    from collections import deque
    h = deque([10, 25, 40, 55, 70, 85, 95, 80, 60, 35], maxlen=100)
    assert draw_fatigue_chart(h) is not None
    plt.close("all")
    return "Chart renders with 10-point history"

def test_component_chart():
    import matplotlib.pyplot as plt
    from charts import draw_component_chart
    fig = draw_component_chart({
        "eye_component": 45.0, "perclos_component": 30.0,
        "mouth_component": 60.0, "gaze_component": 20.0,
    })
    assert fig is not None
    plt.close("all")
    return "Component chart renders with data"

def test_component_chart_empty():
    import matplotlib.pyplot as plt
    from charts import draw_component_chart
    assert draw_component_chart({}) is not None
    plt.close("all")
    return "Component chart handles empty dict"

def test_gauge_all():
    import matplotlib.pyplot as plt
    from charts import draw_gauge_chart
    for s in [0, 25, 50, 75, 100]:
        assert draw_gauge_chart(s) is not None
    plt.close("all")
    return "Gauge renders for 0, 25, 50, 75, 100"

test("charts imports",                     test_charts_imports)
test("get_zone_color() all zones",         test_zone_colors)
test("Fatigue chart - empty history",      test_fatigue_chart_empty)
test("Fatigue chart - with data",          test_fatigue_chart_data)
test("Component chart - with data",        test_component_chart)
test("Component chart - empty dict",       test_component_chart_empty)
test("Gauge chart - all ranges",           test_gauge_all)


# =============================================================================
#   7. FACE RECOGNITION MODULE
# =============================================================================
section("7. FACE RECOGNITION MODULE")

def test_face_import():
    from face_recognition_module import DriverRecognizer
    return "DriverRecognizer imported"

def test_face_init():
    from face_recognition_module import DriverRecognizer
    dr = DriverRecognizer()
    assert hasattr(dr, "face_cascade") and hasattr(dr, "recognizer")
    assert hasattr(dr, "trained")
    assert os.path.exists("drivers")
    return f"Initialized - trained={dr.trained}"

def test_face_recognize_keys():
    from face_recognition_module import DriverRecognizer
    _, info = DriverRecognizer().recognize(face_frame())
    for k in ["driver_name", "confidence", "face_detected"]:
        assert k in info
    return "recognize() returns all 3 keys"

def test_face_get_drivers():
    from face_recognition_module import DriverRecognizer
    r = DriverRecognizer().get_all_drivers()
    assert isinstance(r, str) and len(r) > 0
    return f"get_all_drivers() -> '{r[:40]}'"

def test_face_empty_name():
    from face_recognition_module import DriverRecognizer
    msg = DriverRecognizer().register_driver(face_frame(), "")
    assert any(w in msg.lower() for w in ["name", "enter", "please"])
    return f"Empty name rejected: '{msg}'"

def test_face_whitespace_name():
    from face_recognition_module import DriverRecognizer
    msg = DriverRecognizer().register_driver(face_frame(), "   ")
    assert any(w in msg.lower() for w in ["name", "enter", "please"])
    return f"Whitespace name rejected: '{msg}'"

test("face_recognition imports",           test_face_import)
test("DriverRecognizer initializes",       test_face_init)
test("recognize() returns 3 keys",         test_face_recognize_keys)
test("get_all_drivers() returns string",   test_face_get_drivers)
test("Empty name rejected",                test_face_empty_name)
test("Whitespace name rejected",           test_face_whitespace_name)


# =============================================================================
#   8. PHONE DETECTOR MODULE
# =============================================================================
section("8. PHONE DETECTOR MODULE")

def test_phone_import():
    from phone_detector import PhoneDetector
    return "PhoneDetector imported"

def test_phone_init():
    from phone_detector import PhoneDetector
    pd = PhoneDetector()
    assert hasattr(pd, "model")
    assert pd.phone_count == 0 and pd.total_frames == 0
    return "PhoneDetector initialized"

def test_phone_blank():
    from phone_detector import PhoneDetector
    _, info = PhoneDetector().detect(gray_frame(120))
    for k in ["phone_detected", "phone_confidence", "phone_boxes", "total_detections"]:
        assert k in info
    assert info["phone_detected"] == False
    return "Blank frame -> phone_detected=False, all 4 keys present"

def test_phone_counter():
    from phone_detector import PhoneDetector
    pd = PhoneDetector()
    for _ in range(3):
        pd.detect(gray_frame(120))
    assert pd.total_frames == 3
    return f"total_frames = {pd.total_frames}"

def test_phone_stats():
    from phone_detector import PhoneDetector
    s = PhoneDetector().get_stats()
    assert "total_phone_detections" in s and "detection_rate" in s
    return f"Stats: {s}"

test("phone_detector imports",             test_phone_import)
test("PhoneDetector initializes",          test_phone_init)
test("Blank frame - no phone detected",    test_phone_blank)
test("total_frames counter increments",    test_phone_counter)
test("get_stats() returns correct keys",   test_phone_stats)


# =============================================================================
#   9. PREDICTOR MODULE
# =============================================================================
section("9. PREDICTOR MODULE")

def test_predictor_import():
    from predictor import predict
    return "predict() imported"

def test_predictor_keys():
    from predictor import predict
    data = predict(black_frame(), [])
    for k in ["face_detected", "EAR", "MAR", "PERCLOS",
              "gaze_x", "gaze_y", "drowsy", "yawning", "distracted"]:
        assert k in data, f"Missing: {k}"
    return "All 9 output keys present"

def test_predictor_black():
    from predictor import predict
    data = predict(black_frame(), [])
    assert data["face_detected"] == False
    assert data["EAR"] == 0.0 and data["MAR"] == 0.0 and data["PERCLOS"] == 0.0
    return "Black frame -> face_detected=False, metrics=0.0"

def test_predictor_buffer():
    from predictor import predict
    buf = []
    for _ in range(10):
        predict(black_frame(), buf)
    assert len(buf) <= 60
    return f"PERCLOS buffer length = {len(buf)} (<=60)"

def test_predictor_ear_range():
    from predictor import predict
    data = predict(gray_frame(120), [])
    assert 0.0 <= data["EAR"] <= 1.0
    return f"EAR = {data['EAR']}"

def test_predictor_bool_types():
    from predictor import predict
    data = predict(black_frame(), [])
    assert isinstance(data["drowsy"], bool)
    assert isinstance(data["yawning"], bool)
    assert isinstance(data["distracted"], bool)
    return "drowsy/yawning/distracted are all bool"

test("predictor imports",                  test_predictor_import)
test("All 9 output keys present",          test_predictor_keys)
test("Black frame -> face_detected=False", test_predictor_black)
test("PERCLOS buffer stays <=60",          test_predictor_buffer)
test("EAR value in 0-1 range",            test_predictor_ear_range)
test("Bool flags are bool type",           test_predictor_bool_types)


# =============================================================================
#   10. ALCOHOL DETECTOR MODULE
# =============================================================================
section("10. ALCOHOL DETECTOR MODULE")

_DATA = {"EAR": 0.28, "MAR": 0.30, "PERCLOS": 10.0,
         "gaze_x": 0.05, "gaze_y": 0.08, "yawning": False}

def test_alcohol_import():
    from alcohol_detector import AlcoholDetector
    return "AlcoholDetector imported"

def test_alcohol_init():
    from alcohol_detector import AlcoholDetector
    ad = AlcoholDetector()
    assert hasattr(ad, "impairment_score")
    assert hasattr(ad, "ear_buffer")
    assert ad.impairment_score == 0.0
    return "AlcoholDetector initialized correctly"

def test_alcohol_detect_keys():
    from alcohol_detector import AlcoholDetector
    ad   = AlcoholDetector()
    info = ad.detect(_DATA)
    for k in ["impairment_score", "signals_active", "alcohol_alert",
              "should_alert", "eye_sway", "head_sway",
              "microsleep", "gaze_drift", "yawn_spike"]:
        assert k in info, f"Missing key: {k}"
    return "All 9 output keys present"

def test_alcohol_score_range():
    from alcohol_detector import AlcoholDetector
    ad   = AlcoholDetector()
    info = ad.detect(_DATA)
    assert 0 <= info["impairment_score"] <= 100
    return f"impairment_score = {info['impairment_score']}"

def test_alcohol_safe_data():
    from alcohol_detector import AlcoholDetector
    ad = AlcoholDetector()
    for _ in range(30):
        info = ad.detect(_DATA)
    assert info["impairment_score"] < 80, \
        f"Normal data should give low score, got {info['impairment_score']}"
    return f"Normal data -> score={info['impairment_score']} (safe)"

def test_alcohol_status_label():
    from alcohol_detector import AlcoholDetector
    ad  = AlcoholDetector()
    lbl = ad.get_status_label()
    assert isinstance(lbl, str) and len(lbl) > 0
    return f"Status label: '{lbl}'"

def test_alcohol_active_signals():
    from alcohol_detector import AlcoholDetector
    ad      = AlcoholDetector()
    ad.detect(_DATA)
    signals = ad.get_active_signals()
    assert isinstance(signals, list) and len(signals) >= 1
    return f"Active signals: {signals}"

def test_alcohol_bool_flags():
    from alcohol_detector import AlcoholDetector
    ad   = AlcoholDetector()
    info = ad.detect(_DATA)
    for flag in ["alcohol_alert", "should_alert", "eye_sway",
                 "head_sway", "microsleep", "gaze_drift", "yawn_spike"]:
        assert isinstance(info[flag], bool), f"{flag} is not bool"
    return "All 7 flag fields are bool type"

test("AlcoholDetector imports",            test_alcohol_import)
test("AlcoholDetector initializes",        test_alcohol_init)
test("detect() returns 9 keys",            test_alcohol_detect_keys)
test("impairment_score in 0-100",          test_alcohol_score_range)
test("Normal data gives safe score",       test_alcohol_safe_data)
test("get_status_label() returns string",  test_alcohol_status_label)
test("get_active_signals() returns list",  test_alcohol_active_signals)
test("All flag fields are bool type",      test_alcohol_bool_flags)


# =============================================================================
#   11. REPORT GENERATOR MODULE
# =============================================================================
section("11. REPORT GENERATOR MODULE")

def test_report_import():
    from report_generator import ReportGenerator
    return "ReportGenerator imported"

def test_report_init():
    from report_generator import ReportGenerator
    ReportGenerator()
    assert os.path.exists("reports")
    return "reports/ folder created"

def test_report_recs_high():
    from report_generator import ReportGenerator
    recs = ReportGenerator()._get_recommendations(80, 10, 15, 20)
    assert len(recs) >= 1
    assert any("RISK" in r.upper() or "nap" in r.lower() or "tired" in r.lower()
               for r in recs)
    return f"{len(recs)} recommendation(s) for high fatigue"

def test_report_recs_safe():
    from report_generator import ReportGenerator
    recs = ReportGenerator()._get_recommendations(20, 1, 2, 0)
    assert len(recs) >= 1
    assert any("LOW" in r.upper() or "good" in r.lower()
               or "excel" in r.lower() or "hydrat" in r.lower()
               for r in recs)
    return f"{len(recs)} recommendation(s) for safe session"

def test_report_pdf():
    from report_generator import ReportGenerator
    from collections import deque
    rg           = ReportGenerator()
    session_data = {
        "driver_name": "TestDriver", "session_duration": "0m 10s",
        "total_frames": 100, "avg_fatigue_score": 45.0,
        "peak_fatigue": 80.0, "total_alerts": 3,
        "total_yawns": 2, "alcohol_score": 0.0,
    }
    history  = deque([20, 35, 45, 60, 75, 80, 65, 50, 40, 30], maxlen=100)
    avg_comp = {"eye": 45, "perclos": 30, "mouth": 50, "gaze": 25}
    path     = rg.generate(session_data, history, avg_comp, "TestDriver")
    assert os.path.exists(path) and os.path.getsize(path) > 1000
    return f"PDF: {os.path.basename(path)} ({os.path.getsize(path)//1024}KB)"

test("report_generator imports",           test_report_import)
test("ReportGenerator initializes",        test_report_init)
test("Recommendations for high fatigue",   test_report_recs_high)
test("Recommendations for safe session",   test_report_recs_safe)
test("PDF generated successfully",         test_report_pdf)


# =============================================================================
#   12. INTEGRATION TESTS
# =============================================================================
section("12. INTEGRATION TESTS")

def test_int_predict_score():
    from predictor import predict
    from fatigue_score import get_score_breakdown
    data  = predict(gray_frame(120), [])
    score = get_score_breakdown(
        data["EAR"], data["PERCLOS"], data["MAR"],
        data["gaze_x"], data["gaze_y"]
    )
    assert 0 <= score["fatigue_score"] <= 100
    return f"predict->score = {score['fatigue_score']}"

def test_int_score_alert():
    from fatigue_score import calculate_fatigue_score
    from alert import get_severity_label, get_severity_color
    score, _ = calculate_fatigue_score(0.15, 50.0, 0.60, 0.20, 0.30)
    label    = get_severity_label(score)
    color    = get_severity_color(score)
    assert label and color.startswith("#")
    return f"score={score} -> {label} {color}"

def test_int_predict_log():
    import config
    config.ENABLE_LOGGING = False
    from predictor import predict
    from fatigue_score import get_score_breakdown
    from alert import get_severity_label
    from logger import SessionLogger
    lg    = SessionLogger()
    data  = predict(gray_frame(120), [])
    score = get_score_breakdown(
        data["EAR"], data["PERCLOS"], data["MAR"],
        data["gaze_x"], data["gaze_y"]
    )
    lg.log(data, score, get_severity_label(score["fatigue_score"]), False)
    assert lg.frame_count == 1
    return "predict -> score -> log: frame_count=1"

def test_int_alcohol_pipeline():
    from predictor import predict
    from alcohol_detector import AlcoholDetector
    ad   = AlcoholDetector()
    data = predict(gray_frame(120), [])
    info = ad.detect(data)
    assert "impairment_score" in info
    assert 0 <= info["impairment_score"] <= 100
    return f"predict->alcohol: score={info['impairment_score']}"

def test_int_night_predict():
    from night_mode import process_night_mode
    from predictor import predict
    enh, info = process_night_mode(gray_frame(40))
    data      = predict(enh, [])
    assert "face_detected" in data
    assert info["night_mode_on"] == True
    return f"night_mode->predict OK, night_mode_on=True"

def test_int_score_charts():
    import matplotlib.pyplot as plt
    from fatigue_score import get_score_breakdown
    from charts import draw_fatigue_chart, draw_component_chart, draw_gauge_chart
    from collections import deque
    score = get_score_breakdown(0.20, 30.0, 0.50, 0.12, 0.20)
    h     = deque([score["fatigue_score"]] * 5, maxlen=100)
    f1    = draw_fatigue_chart(h)
    f2    = draw_component_chart(score)
    f3    = draw_gauge_chart(score["fatigue_score"])
    assert all(f is not None for f in [f1, f2, f3])
    plt.close("all")
    return f"All 3 charts rendered, score={score['fatigue_score']}"

def test_int_full_pipeline():
    import config
    config.ENABLE_LOGGING = False
    from night_mode import process_night_mode
    from predictor import predict
    from fatigue_score import get_score_breakdown
    from alert import get_severity_label, get_severity_color, trigger_alert
    from alcohol_detector import AlcoholDetector
    from logger import SessionLogger

    lg         = SessionLogger()
    ad         = AlcoholDetector()
    enh, _     = process_night_mode(gray_frame(100))
    data        = predict(enh, [])
    alc_info    = ad.detect(data)
    score       = get_score_breakdown(
        data["EAR"], data["PERCLOS"], data["MAR"],
        data["gaze_x"], data["gaze_y"]
    )
    label = get_severity_label(score["fatigue_score"])
    color = get_severity_color(score["fatigue_score"])
    trigger_alert(score["fatigue_score"],
                  yawning=data["yawning"], distracted=data["distracted"])
    lg.log(data, score, label, False)

    assert lg.frame_count == 1
    assert color.startswith("#")
    assert 0 <= alc_info["impairment_score"] <= 100
    return f"Full pipeline OK - score={score['fatigue_score']}, label={label}"

test("predict -> fatigue_score",           test_int_predict_score)
test("fatigue_score -> alert",             test_int_score_alert)
test("predict -> score -> log",            test_int_predict_log)
test("predict -> alcohol pipeline",        test_int_alcohol_pipeline)
test("night_mode -> predictor",            test_int_night_predict)
test("score -> all 3 charts",             test_int_score_charts)
test("Full end-to-end pipeline",           test_int_full_pipeline)


# =============================================================================
#   13. APP.PY STRUCTURE CHECKS
# =============================================================================
section("13. APP.PY STRUCTURE CHECKS")

def _src():
    app_path = os.path.join(SCRIPT_DIR, "app.py")
    assert os.path.exists(app_path), \
        f"app.py not found at {app_path}. Run test from project root!"
    with open(app_path, "r", encoding="utf-8") as f:
        return f.read()

def test_app_all_imports():
    src = _src()
    for mod in ["predictor", "fatigue_score", "alert", "logger",
                "charts", "night_mode", "phone_detector",
                "face_recognition_module", "alcohol_detector",
                "report_generator"]:
        assert mod in src, f"Missing import: {mod}"
    return "All 10 module imports found"

def test_app_predict():
    assert "predict(" in _src()
    return "predict() call found"

def test_app_stream():
    assert ".stream(" in _src()
    return "Gradio .stream() found"

def test_app_launch():
    src = _src()
    assert "app.launch(" in src and "server_port" in src
    return "app.launch() with server_port found"

def test_app_process_frame():
    assert "def process_frame(" in _src()
    return "process_frame() defined"

def test_app_get_summary():
    assert "def get_summary(" in _src()
    return "get_summary() defined"

def test_app_pdf_report():
    src = _src()
    assert ("generate_pdf_report" in src or
            "report_gen.generate(" in src or
            "report_generator.generate(" in src)
    return "PDF report generation code found"

def test_app_register_driver():
    src = _src()
    assert "register_driver" in src and "Driver Registration" in src
    return "Driver registration UI and logic found"

def test_app_alcohol():
    src = _src()
    assert "alcohol_detector" in src and "Alcohol" in src
    return "Alcohol detector used in app.py"

def test_app_charts():
    src = _src()
    assert "fatigue_chart"   in src
    assert "component_chart" in src
    assert "gauge_chart"     in src
    return "All 3 chart components found"

def test_app_draw_overlay():
    src = _src()
    assert "def draw_overlay(" in src and "cv2.putText" in src
    return "draw_overlay() with cv2.putText found"

def test_app_logger():
    src = _src()
    assert "SessionLogger" in src and "logger.log(" in src
    return "SessionLogger used and logger.log() called"

def test_app_correct_order():
    """Verify predict() comes BEFORE alcohol_detector.detect() in source."""
    src   = _src()
    lines = src.splitlines()
    p_line, a_line = None, None
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if p_line is None and stripped.startswith("data = predict("):
            p_line = i
        if a_line is None and "alcohol_detector.detect(" in stripped:
            a_line = i
    assert p_line is not None, "data = predict() not found"
    assert a_line is not None, "alcohol_detector.detect() not found"
    assert p_line < a_line, \
        f"BUG: alcohol_detector.detect() (line {a_line}) is before predict() (line {p_line})"
    return f"predict() at line {p_line} is before alcohol_detector.detect() at line {a_line}"

test("All 10 module imports present",      test_app_all_imports)
test("predict() called",                   test_app_predict)
test("Gradio .stream() present",           test_app_stream)
test("app.launch() configured",            test_app_launch)
test("process_frame() defined",            test_app_process_frame)
test("get_summary() defined",              test_app_get_summary)
test("PDF report generation present",      test_app_pdf_report)
test("Driver registration UI + logic",     test_app_register_driver)
test("Alcohol monitoring present",         test_app_alcohol)
test("All 3 chart components in UI",       test_app_charts)
test("draw_overlay() defined",             test_app_draw_overlay)
test("SessionLogger used correctly",       test_app_logger)
test("predict() before alcohol_detect()",  test_app_correct_order)


# =============================================================================
#   FINAL SUMMARY
# =============================================================================
section("FINAL SUMMARY")

total  = len(results)
passed = sum(1 for r in results if r[0] == "PASS")
failed = sum(1 for r in results if r[0] == "FAIL")

print(f"\n  Total Tests  : {total}")
print(f"  {GREEN}Passed       : {passed}{RESET}")
print(f"  {RED}Failed       : {failed}{RESET}")
print(f"  Pass Rate    : {passed / total * 100:.1f}%\n")

if failed > 0:
    print(f"{RED}{BOLD}  Failed Tests:{RESET}")
    for status, name in results:
        if status == "FAIL":
            print(f"    {RED}x  {name}{RESET}")

print(f"\n{BOLD}{'='*60}{RESET}")
if failed == 0:
    print(f"{GREEN}{BOLD}  ALL {total} TESTS PASSED! VigilEye-V3 is ready.{RESET}")
elif failed <= 2:
    print(f"{YELLOW}{BOLD}  Minor issues found. Review failures above.{RESET}")
else:
    print(f"{RED}{BOLD}  Multiple failures. Fix before running app.{RESET}")
print(f"{BOLD}{'='*60}{RESET}\n")

sys.exit(0 if failed == 0 else 1)
