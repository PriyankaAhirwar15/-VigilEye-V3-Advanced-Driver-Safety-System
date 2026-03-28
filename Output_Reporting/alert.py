# ---------------------------------------------------------
#   VigilEye-V3  |  alert.py
#   Voice + Sound Alert System (Multithreaded)
# ---------------------------------------------------------

import pyttsx3
import pygame
import threading
import time
import math

# UPDATED IMPORT: Pointing to the new Config_Files folder
from Config_Files.config import (
    MSG_MILD, MSG_MODERATE, MSG_CRITICAL,
    MSG_YAWN, MSG_DISTRACTED,
    SEVERITY_MILD, SEVERITY_MODERATE, SEVERITY_CRITICAL
)

# -- Setup voice engine --------------------
try:
    engine = pyttsx3.init()
    engine.setProperty("rate", 160)       # Speaking speed
    engine.setProperty("volume", 1.0)     # Full volume level
except Exception as e:
    print(f"[ERROR] Voice Init Failed: {e}")
    engine = None

# -- Setup pygame for beep sounds ---------
if not pygame.mixer.get_init():
    pygame.mixer.init()

# -- Global Variables for Alert Tracking ---
last_alert_time = 0
COOLDOWN_SECONDS = 5
voice_lock = threading.Lock()  # Prevents multiple threads from crashing the engine

def _speak(message):
    """Internal function to run voice alerts in the background"""
    # Use the lock to ensure only one thread uses the engine at a time
    if not voice_lock.acquire(blocking=False):
        return  # If the engine is already speaking, skip this new request

    try:
        if engine:
            engine.say(message)
            engine.runAndWait()
    except Exception as e:
        # Silently catch the "run loop" error to keep terminal clean
        pass
    finally:
        voice_lock.release()

def speak_async(message):
    """Triggers voice alert without blocking the main video stream"""
    t = threading.Thread(target=_speak, args=(message,), daemon=True)
    t.start()

def beep(frequency=1000, duration=500):
    """Generates and plays a beep sound using pygame buffer"""
    try:
        sample_rate = 44100
        n_samples = int(sample_rate * duration / 1000)
        
        # Generating a clean sine wave for the alert sound
        buf = bytes([int(127 + 127 * math.sin(2 * math.pi * frequency * i / sample_rate)) for i in range(n_samples)])
        
        sound = pygame.mixer.Sound(buffer=buf)
        sound.play()
    except Exception:
        pass  # Silently skip if audio device is busy or unavailable

def trigger_alert(fatigue_score, yawning=False, distracted=False):
    """
    Main Logic: Decides which alert to fire based on driver state.
    Priority order: Distraction > Yawning > Fatigue Levels.
    """
    global last_alert_time
    now = time.time()

    # Apply Cooldown to prevent repetitive alerts
    if now - last_alert_time < COOLDOWN_SECONDS:
        return False 

    alert_fired = False

    # Priority 1: Distraction Alert
    if distracted:
        beep(frequency=1200, duration=300)
        speak_async(MSG_DISTRACTED)
        alert_fired = True

    # Priority 2: Yawning Alert
    elif yawning:
        beep(frequency=900, duration=400)
        speak_async(MSG_YAWN)
        alert_fired = True

    # Priority 3: Fatigue Score Severity
    elif fatigue_score >= SEVERITY_CRITICAL:
        beep(frequency=1500, duration=800)
        speak_async(MSG_CRITICAL)
        alert_fired = True

    elif fatigue_score >= SEVERITY_MODERATE:
        beep(frequency=1100, duration=500)
        speak_async(MSG_MODERATE)
        alert_fired = True

    elif fatigue_score >= SEVERITY_MILD:
        beep(frequency=800, duration=300)
        speak_async(MSG_MILD)
        alert_fired = True

    if alert_fired:
        last_alert_time = now
    
    return alert_fired

def get_severity_label(fatigue_score):
    """Returns a visual label for the dashboard"""
    if fatigue_score >= SEVERITY_CRITICAL:
        return "🚨 CRITICAL"
    elif fatigue_score >= SEVERITY_MODERATE:
        return "🔴 MODERATE"
    elif fatigue_score >= SEVERITY_MILD:
        return "🟡 MILD"
    else:
        return "🟢 SAFE"

def get_severity_color(fatigue_score):
    """Returns a Hex color code for the dashboard UI elements"""
    if fatigue_score >= SEVERITY_CRITICAL:
        return "#FF0000"  # Red
    elif fatigue_score >= SEVERITY_MODERATE:
        return "#FF6600"  # Orange
    elif fatigue_score >= SEVERITY_MILD:
        return "#FFD700"  # Gold
    else:
        return "#00CC44"  # Green