# ---------------------------------------------------------
#   VigilEye-V3  |  streamlit_app.py
#   Professional Live AI Monitoring (Industry Standard)
# ---------------------------------------------------------

import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import numpy as np

# Internal module imports
from Config_Files.config import CHART_HISTORY
from Core_Detection.predictor import predict 
from Core_Detection.fatigue_score import get_score_breakdown
from Output_Reporting.alert import get_severity_label, trigger_alert

# -- Page Configuration --
st.set_page_config(
    page_title="VigilEye-V3 AI Dashboard",
    page_icon="🛡️",
    layout="wide"
)

# -- Custom CSS for UI Enhancement --
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: white; }
    .stMetric { border: 1px solid #4B5563; padding: 10px; border-radius: 8px; }
    </style>
    """, unsafe_allow_html=True)

st.title("🛡️ VigilEye-V3 — AI Driver Safety Monitor")
st.caption("Live Computer Vision Pipeline for Drowsiness and Fatigue Detection")

# -- Video Processing Class --
class VigilEyeVideoProcessor(VideoTransformerBase):
    def __init__(self):
        # Initialize buffer for PERCLOS calculation
        self.perclos_buffer = []

    def transform(self, frame):
        """
        Main frame processing function. 
        Converts Web-RTC frame to OpenCV and runs AI logic.
        """
        # Convert incoming Web-RTC frame to numpy array (BGR format)
        img = frame.to_ndarray(format="bgr24")

        # 1. Run Core AI Predictor (Landmarks, EAR, MAR, Gaze)
        # The predict function updates landmarks directly on the img
        data = predict(img, self.perclos_buffer)

        if data["face_detected"]:
            # 2. Calculate Fatigue Breakdown Scores
            score_data = get_score_breakdown(
                data["EAR"], data["PERCLOS"], 
                data["MAR"], data["gaze_x"], data["gaze_y"]
            )
            
            # 3. Decision Logic: Trigger Voice/Sound Alerts
            # Alert thresholds are handled within the trigger_alert function
            trigger_alert(score_data["fatigue_score"])

            # 4. Visual Overlays (On-Screen Display)
            label = get_severity_label(score_data["fatigue_score"])
            
            # Render labels on the frame for the user
            cv2.putText(img, f"SAFETY STATUS: {label}", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(img, f"EAR: {data['EAR']:.2f}", (20, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        else:
            # Display warning if no driver is detected in the frame
            cv2.putText(img, "WARNING: NO FACE DETECTED", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Return the processed frame back to the Streamlit UI
        return img

# -- Layout Configuration --
col_video, col_info = st.columns([2, 1])

with col_video:
    st.subheader("📡 Real-Time AI Intelligence Feed")
    # Deploying the Web-RTC streamer for browser-based camera access
    webrtc_streamer(
        key="vigileye-v3-stream", 
        video_transformer_factory=VigilEyeVideoProcessor,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

with col_info:
    st.subheader("📌 System Instructions")
    st.markdown("""
    - **Step 1:** Click the **'Start'** button on the video feed.
    - **Step 2:** Grant camera permissions to your browser.
    - **Step 3:** Ensure your face is clearly visible and well-lit.
    - **Note:** The system uses **Web-RTC** for low-latency cloud compatibility.
    """)
    
    st.divider()
    st.info("System is monitoring Fatigue (EAR), Yawning (MAR), and PERCLOS.")

# -- End of Script --