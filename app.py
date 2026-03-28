# ---------------------------------------------------------
#   VigilEye-V3  |  app.py
#   Main Live Fatigue Monitoring Dashboard (Gradio)
# ---------------------------------------------------------

import cv2
import gradio as gr
import numpy as np
import time
from collections import deque

# Standardized Imports based on Project Structure
from Config_Files.config import DASHBOARD_TITLE, CHART_HISTORY
from Core_Detection.predictor import predict
from Core_Detection.fatigue_score import get_score_breakdown
from Output_Reporting.alert import trigger_alert, get_severity_label, get_severity_color
from Output_Reporting.logger import SessionLogger
from Output_Reporting.charts import draw_fatigue_chart, draw_component_chart, draw_gauge_chart
from Output_Reporting.report_generator import ReportGenerator

# Advanced Feature Modules
from Feature_Modules.night_mode import process_night_mode
from Feature_Modules.phone_detector import PhoneDetector
from Feature_Modules.face_recognition_module import DriverRecognizer
from Feature_Modules.alcohol_detector import AlcoholDetector

# -- Session State Initialization --
perclos_buffer    = []
score_history     = deque(maxlen=CHART_HISTORY)
logger            = SessionLogger()
report_gen        = ReportGenerator()
yawn_times        = []
phone_detector    = PhoneDetector()
driver_recognizer = DriverRecognizer()
alcohol_detector  = AlcoholDetector()

# Tracking average components for the final PDF report
component_tracker = {"eye": [], "perclos": [], "mouth": [], "gaze": []}

# -- Helper Functions --

def draw_overlay(frame, data, score_data, severity_label, color_hex):
    """Draws real-time safety metrics as an overlay on the camera frame."""
    h, w    = frame.shape[:2]
    hex_col = color_hex.lstrip("#")
    r, g, b = tuple(int(hex_col[i:i+2], 16) for i in (0, 2, 4))
    color   = (b, g, r)  # OpenCV uses BGR format

    # Top Header Bar (Dark Overlay)
    cv2.rectangle(frame, (0, 0), (w, 80), (20, 20, 20), -1)
    cv2.putText(frame, "VigilEye-V3 Live Feed", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Intensity: {score_data['fatigue_score']}%", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.putText(frame, severity_label, (w - 200, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # Bottom Statistics Bar
    cv2.rectangle(frame, (0, h - 40), (w, h), (20, 20, 20), -1)
    stats = f"EAR: {data['EAR']:.2f} | MAR: {data['MAR']:.2f} | PERCLOS: {data['PERCLOS']}%"
    cv2.putText(frame, stats, (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # Visual Border for Critical Fatigue
    if score_data["fatigue_score"] >= 90:
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), color, 6)

    return frame

def count_yawns_per_minute(yawning):
    """Tracks frequency of yawning within a sliding 60-second window."""
    now = time.time()
    if yawning:
        yawn_times.append(now)
    # Maintain only timestamps from the last 60 seconds
    yawn_times[:] = [t for t in yawn_times if now - t <= 60]
    return len(yawn_times)

def process_frame(frame):
    """Main processing pipeline triggered by Gradio's webcam stream."""
    if frame is None:
        # Return default values if no frame is received
        return (None, "Feed Lost", 0, "-", "-", "0/min", "-", None, None, None, 0, "Idle", "None")

    # 1. Image Enhancement (Night Mode processing)
    frame, _ = process_night_mode(frame)

    # 2. Facial Landmark & Biometric Extraction
    data = predict(frame, perclos_buffer)

    # 3. Impairment & Distraction Detection
    alcohol_info = alcohol_detector.detect(data)
    frame, phone_info = phone_detector.detect(frame)
    if phone_info["phone_detected"]:
        data["distracted"] = True

    # 4. Identity Recognition Check
    frame, _ = driver_recognizer.recognize(frame)

    # 5. Handle Face Detection Failure
    if not data["face_detected"]:
        empty_fig = draw_fatigue_chart(score_history)
        return (frame, "Searching for face...", 0, "-", "-", "0/min", "-", empty_fig, None, None, 0, "No User", "None")

    # 6. Core Fatigue Scoring
    score_data = get_score_breakdown(data["EAR"], data["PERCLOS"], data["MAR"], data["gaze_x"], data["gaze_y"])
    
    # Update trackers for session report
    for key in ["eye", "perclos", "mouth", "gaze"]:
        component_tracker[key].append(score_data.get(f"{key}_component", 0))

    # 7. Safety Status and Alerting
    severity_label = get_severity_label(score_data["fatigue_score"])
    color_code     = get_severity_color(score_data["fatigue_score"])
    yawns_count    = count_yawns_per_minute(data["yawning"])

    alert_active = (score_data["fatigue_score"] >= 40 or data["yawning"] or data["distracted"])
    
    # Audio Alerting (Note: Limited on Cloud environments)
    try:
        trigger_alert(score_data["fatigue_score"], yawning=data["yawning"], distracted=data["distracted"])
    except:
        pass # Silently fail if audio device is unavailable

    # 8. Logging and Data History
    logger.log(data, score_data, severity_label, alert_active)
    score_history.append(score_data["fatigue_score"])

    # 9. Visualization & UI Rendering
    frame         = draw_overlay(frame, data, score_data, severity_label, color_code)
    fatigue_fig   = draw_fatigue_chart(score_history)
    component_fig = draw_component_chart(score_data)
    gauge_fig     = draw_gauge_chart(score_data["fatigue_score"])

    # Convert BGR back to RGB for Gradio display
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    return (
        frame_rgb, severity_label, score_data["fatigue_score"],
        str(round(data['EAR'], 2)), f"{data['PERCLOS']}%", f"{yawns_count}/min",
        f"X:{data['gaze_x']} Y:{data['gaze_y']}",
        fatigue_fig, component_fig, gauge_fig,
        alcohol_info["impairment_score"], 
        alcohol_detector.get_status_label(),
        ", ".join(alcohol_detector.get_active_signals())
    )

def generate_pdf_report():
    """Compiles and generates the final PDF session summary."""
    if not score_history:
        return "Error: No data recorded. Please start monitoring first."
    
    try:
        summary = logger.get_session_summary()
        scores  = list(score_history)
        avg_comp = {k: round(sum(v)/len(v), 1) if v else 0 for k, v in component_tracker.items()}
        
        session_data = {
            "driver_name"       : "Active Driver",
            "session_duration"  : summary["session_duration"],
            "total_frames"      : summary["total_frames"],
            "avg_fatigue_score" : round(sum(scores)/len(scores), 1),
            "peak_fatigue"      : round(max(scores), 1),
            "total_alerts"      : summary["total_alerts"],
            "total_yawns"       : summary["total_yawns"],
            "alcohol_score"     : alcohol_detector.impairment_score,
        }
        
        path = report_gen.generate(session_data, score_history, avg_comp)
        return f"Report ready! File generated at: {path}"
    except Exception as e:
        return f"Report Error: {str(e)}"

# -- Gradio UI Construction --
with gr.Blocks(title=DASHBOARD_TITLE, theme=gr.themes.Default()) as app:
    gr.Markdown(f"# {DASHBOARD_TITLE}")
    
    with gr.Tabs():
        with gr.TabItem("Live Monitoring"):
            with gr.Row():
                with gr.Column(scale=2):
                    camera = gr.Image(sources=["webcam"], streaming=True, label="Webcam Input")
                    output_frame = gr.Image(label="AI Analysis Output")
                
                with gr.Column(scale=1):
                    severity_display = gr.Label(label="Safety Level")
                    fatigue_score = gr.Number(label="Fatigue Intensity (%)")
                    
                    with gr.Group():
                        gr.Markdown("### Real-Time Metrics")
                        with gr.Row():
                            ear_disp = gr.Textbox(label="EAR")
                            per_disp = gr.Textbox(label="PERCLOS")
                        with gr.Row():
                            yawn_disp = gr.Textbox(label="Yawns/Min")
                            gaze_disp = gr.Textbox(label="Gaze Coord")

            with gr.Row():
                pdf_btn = gr.Button("Generate PDF Session Report", variant="primary")
                pdf_status = gr.Textbox(label="Report Status", interactive=False)
                pdf_btn.click(generate_pdf_report, outputs=pdf_status)

        with gr.TabItem("Alcohol & Impairment"):
            with gr.Row():
                alc_score = gr.Number(label="Alcohol Confidence Score")
                alc_status = gr.Textbox(label="Detection State")
            alc_signals = gr.Textbox(label="Physical Signals Detected", lines=2)

        with gr.TabItem("Visual Analytics"):
            with gr.Row():
                f_chart = gr.Plot(label="Fatigue Trend")
                c_chart = gr.Plot(label="Metric Contribution")
            with gr.Row():
                g_chart = gr.Plot(label="Intensity Gauge")

    # Connect the streaming webcam feed to the processing function
    camera.stream(
        process_frame, 
        inputs=[camera],
        outputs=[
            output_frame, severity_display, fatigue_score, 
            ear_disp, per_disp, yawn_disp, gaze_disp, 
            f_chart, c_chart, g_chart,
            alc_score, alc_status, alc_signals
        ]
    )

if __name__ == "__main__":
    app.launch()