---
license: mit
title: VigilEye V3 Advanced Driver Safety System
sdk: gradio
emoji: 👀
colorFrom: red
colorTo: yellow
pinned: true
---


<div align="center">

```
██╗   ██╗██╗ ██████╗ ██╗██╗     ███████╗██╗   ██╗███████╗   ██╗   ██╗██████╗
██║   ██║██║██╔════╝ ██║██║     ██╔════╝╚██╗ ██╔╝██╔════╝   ██║   ██║╚════██╗
██║   ██║██║██║  ███╗██║██║     █████╗   ╚████╔╝ █████╗     ██║   ██║ █████╔╝
╚██╗ ██╔╝██║██║   ██║██║██║     ██╔══╝    ╚██╔╝  ██╔══╝     ╚██╗ ██╔╝ ╚═══██╗
 ╚████╔╝ ██║╚██████╔╝██║███████╗███████╗   ██║   ███████╗    ╚████╔╝ ██████╔╝
  ╚═══╝  ╚═╝ ╚═════╝ ╚═╝╚══════╝╚══════╝   ╚═╝   ╚══════╝     ╚═══╝  ╚═════╝
```

# 🛡️ VigilEye-V3
### *Advanced AI Driver Safety & Fatigue Monitoring System*

<br/>

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10-0097A7?style=for-the-badge&logo=google&logoColor=white)](https://mediapipe.dev)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-FF6F00?style=for-the-badge&logo=yolo&logoColor=white)](https://ultralytics.com)
[![Gradio](https://img.shields.io/badge/Gradio-4.x-F97316?style=for-the-badge&logo=gradio&logoColor=white)](https://gradio.app)
[![License](https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge)](LICENSE)

<br/>

> **Real-time drowsiness detection, alcohol impairment analysis, phone detection,**
> **face recognition and PDF reporting — all running locally, zero cloud dependency.**

<br/>

[![Tests](https://img.shields.io/badge/Tests-87%2F87%20Passing-22C55E?style=flat-square&logo=pytest)](test_system.py)
[![Privacy](https://img.shields.io/badge/Privacy-100%25%20Local-6366F1?style=flat-square&logo=shield)](README.md)
[![No GPU](https://img.shields.io/badge/GPU-Not%20Required-F59E0B?style=flat-square)](requirements.txt)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=flat-square&logo=docker)](Dockerfile)

</div>

---

## 📋 Table of Contents

- [🎯 What is VigilEye-V3?](#-what-is-vigileye-v3)
- [✨ Features](#-features)
- [🏗️ System Architecture](#️-system-architecture)
- [📊 Detection Pipeline](#-detection-pipeline)
- [🧠 Fatigue Scoring Algorithm](#-fatigue-scoring-algorithm)
- [📁 Project Structure](#-project-structure)
- [⚙️ Tech Stack](#️-tech-stack)
- [🚀 Quick Start](#-quick-start)
- [🖥️ Dashboard Preview](#️-dashboard-preview)
- [📈 Performance Metrics](#-performance-metrics)
- [🔬 Module Deep Dive](#-module-deep-dive)
- [🧪 Testing](#-testing)
- [🐳 Docker Deployment](#-docker-deployment)
- [🛣️ Roadmap](#️-roadmap)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)

---

## 🎯 What is VigilEye-V3?

**VigilEye-V3** is a production-grade, privacy-first AI safety system that monitors driver alertness in real time using only a standard webcam. It combines computer vision, behavioural analysis, and machine learning to detect fatigue before it becomes fatal — all without sending a single byte of data to the cloud.

```
🚗  Camera  ──►  AI Analysis  ──►  Fatigue Score  ──►  Alert / Report
     📷           7 modules          0 – 100             🔊  📄  📊
```

**Why it matters:**
- Road accidents cause **1.35 million deaths** per year globally (WHO)
- **drowsy driving** accounts for **21%** of fatal crashes
- VigilEye-V3 runs on any laptop, **no specialised hardware required**

---

## ✨ Features

| Feature | Technology | Status |
|---------|-----------|--------|
| 👁️ Eye Aspect Ratio (EAR) Detection | MediaPipe Face Mesh | ✅ Live |
| 😮 Yawn Detection (MAR) | MediaPipe 468 Landmarks | ✅ Live |
| 👀 Gaze / Distraction Tracking | Head Pose Estimation | ✅ Live |
| 📊 PERCLOS Analysis | Rolling 60-frame buffer | ✅ Live |
| 🎯 Weighted Fatigue Score (0–100) | Custom Algorithm | ✅ Live |
| 🌙 Night Mode / Low-Light Enhancement | CLAHE + OpenCV | ✅ Live |
| 📱 Phone Detection While Driving | YOLOv8n (nano) | ✅ Live |
| 🧑‍💼 Driver Face Recognition | OpenCV LBPH | ✅ Live |
| 🍺 Alcohol Impairment Detection | 5-Signal Behavioural | ✅ Live |
| 🔊 Voice + Beep Alerts | pyttsx3 + pygame | ✅ Live |
| 📄 PDF Session Report | ReportLab + Charts | ✅ Live |
| 📋 CSV Session Logger | utf-8 safe | ✅ Live |
| 🌐 Live Dashboard | Gradio 4.x | ✅ Live |
| 🧪 Full Test Suite | 87 tests / 100% | ✅ Live |
| 🐳 Docker Support | Multi-stage build | ✅ Ready |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        VigilEye-V3  System                          │
│                                                                     │
│   📷 Webcam Input                                                   │
│        │                                                            │
│        ▼                                                            │
│   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐         │
│   │  night_mode  │───►│  predictor   │───►│fatigue_score │         │
│   │  (CLAHE)     │    │  (MediaPipe) │    │  (weighted)  │         │
│   └──────────────┘    └──────────────┘    └──────┬───────┘         │
│                              │                   │                  │
│                    ┌─────────┼──────────┐        │                  │
│                    ▼         ▼          ▼        ▼                  │
│             ┌──────────┐ ┌──────┐ ┌────────┐ ┌──────┐             │
│             │  phone_  │ │ face_│ │alcohol_│ │alert │             │
│             │ detector │ │ recog│ │detector│ │  .py │             │
│             │ (YOLOv8) │ │(LBPH)│ │(5-sig) │ └──┬───┘             │
│             └──────────┘ └──────┘ └────────┘    │                  │
│                    │         │          │        │                  │
│                    └─────────┴──────────┴────────┘                  │
│                                   │                                 │
│                    ┌──────────────┼──────────────┐                  │
│                    ▼              ▼              ▼                  │
│             ┌──────────┐  ┌──────────┐  ┌──────────┐              │
│             │  logger  │  │  charts  │  │  report_ │              │
│             │  (.csv)  │  │(matplot) │  │generator │              │
│             └──────────┘  └──────────┘  └──────────┘              │
│                                   │                                 │
│                    ┌──────────────▼──────────────┐                  │
│                    │       app.py  (Gradio)       │                  │
│                    │   Live Dashboard : 7860      │                  │
│                    └─────────────────────────────┘                  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 📊 Detection Pipeline

```
Every Frame (30fps)
        │
        ▼
   ┌─────────────────────────────────────────────────────┐
   │  Step 1 │  Night Mode Enhancement (CLAHE)           │
   │         │  brightness < 80  ──►  auto enhance       │
   └─────────────────────────────────────────────────────┘
        │
        ▼
   ┌─────────────────────────────────────────────────────┐
   │  Step 2 │  MediaPipe Face Mesh (468 landmarks)      │
   │         │  EAR · MAR · Gaze X/Y · PERCLOS           │
   └─────────────────────────────────────────────────────┘
        │
        ├──────────────────────────────────────────────►
        │                                              │
        ▼                                              ▼
   ┌──────────────┐                          ┌──────────────────┐
   │ Step 3       │                          │  Step 3b         │
   │ YOLOv8n      │                          │  Alcohol         │
   │ Phone detect │                          │  5-signal check  │
   └──────────────┘                          └──────────────────┘
        │
        ▼
   ┌─────────────────────────────────────────────────────┐
   │  Step 4 │  Weighted Fatigue Score Calculator        │
   │         │  EAR(35%) + PERCLOS(30%) + MAR(20%)       │
   │         │                        + GAZE(15%)        │
   └─────────────────────────────────────────────────────┘
        │
        ▼
   ┌─────────────────────────────────────────────────────┐
   │  Step 5 │  Severity Classification                  │
   │         │  0–39 🟢 SAFE  │  40–69 🟡 MILD          │
   │         │  70–89 🔴 MOD  │  90+   🚨 CRITICAL       │
   └─────────────────────────────────────────────────────┘
        │
        ▼
   ┌──────────┐    ┌──────────┐    ┌──────────┐
   │  Alert   │    │  Logger  │    │  Charts  │
   │  (voice) │    │  (CSV)   │    │  (live)  │
   └──────────┘    └──────────┘    └──────────┘
```

---

## 🧠 Fatigue Scoring Algorithm

The heart of VigilEye-V3 is a **weighted multi-signal fatigue score** (0–100):

```
┌─────────────────────────────────────────────────────────────┐
│                  FATIGUE SCORE FORMULA                      │
│                                                             │
│  Score = (EAR × 0.35) + (PERCLOS × 0.30)                  │
│        + (MAR × 0.20) + (GAZE × 0.15)                     │
│                                                             │
│  Each signal is normalized to [0.0 – 1.0] then ×100        │
│                                                             │
│  Signal          Weight   Measures                          │
│  ─────────────── ──────   ─────────────────────────────     │
│  EAR  (eye open)   35%   Eye closure speed + depth         │
│  PERCLOS           30%   % frames eyes closed / 60fr       │
│  MAR  (mouth)      20%   Yawning frequency + intensity     │
│  GAZE direction    15%   Head turn / look-away events      │
└─────────────────────────────────────────────────────────────┘
```

### EAR Calculation (Eye Aspect Ratio)

```
         P2 ─── P3
        /         \
  P1 ──             ── P4
        \         /
         P6 ─── P5

  EAR = (‖P2−P6‖ + ‖P3−P5‖) / (2 × ‖P1−P4‖)

  Normal open eye:  EAR ≈ 0.30 – 0.40
  Drowsy threshold: EAR < 0.22  ──► alert triggered
```

### PERCLOS (Percentage Eye Closure)

```
  60-frame rolling window (≈ 2 seconds at 30fps)

  PERCLOS = (frames with EAR < 0.22) / 60 × 100%

  0%  ──────────────────── 100%
  ███░░░░░░░░░░░░░░░░░░░░  normal   (~5%)
  ████████░░░░░░░░░░░░░░░  drowsy   (>30%)
  ███████████████░░░░░░░░  critical (>60%)
```

### Alert Severity Thresholds

```
  Score   Severity    Action
  ─────── ─────────── ───────────────────────────────────────
   0– 39  🟢 SAFE     Monitor only
  40– 69  🟡 MILD     Beep 800Hz + voice: "Stay alert!"
  70– 89  🔴 MODERATE Beep 1100Hz + voice: "Take a break soon"
  90–100  🚨 CRITICAL Beep 1500Hz + voice: "Pull over NOW!"
  any     📱 PHONE    Beep 1200Hz + voice: "Eyes on the road!"
  any     🍺 ALCOHOL  Beep 1800Hz + voice: "Stop driving!"
```

---

## 📁 Project Structure

```
VigilEye-V3/
│
├── 📱 app.py                    ← Gradio dashboard entry point
│
├── 🧠 Core Detection
│   ├── predictor.py             ← MediaPipe face mesh (EAR/MAR/Gaze)
│   ├── fatigue_score.py         ← Weighted score algorithm (0–100)
│   └── config.py                ← All thresholds & constants
│
├── 🔍 Feature Modules
│   ├── phone_detector.py        ← YOLOv8n phone detection
│   ├── face_recognition_module.py ← LBPH driver recognition
│   ├── alcohol_detector.py      ← 5-signal impairment analysis
│   └── night_mode.py            ← CLAHE low-light enhancement
│
├── 📊 Output & Reporting
│   ├── alert.py                 ← Voice + beep alert system
│   ├── logger.py                ← UTF-8 CSV session logger
│   ├── charts.py                ← Matplotlib live charts
│   └── report_generator.py     ← ReportLab PDF reports
│
├── 🧪 Testing
│   └── test_system.py           ← 87-test full system suite
│
├── 📂 Data Folders
│   ├── drivers/                 ← Registered driver face photos
│   └── reports/                 ← Generated PDF reports
│
├── 📄 Config Files
│   ├── requirements.txt
│   ├── Dockerfile
│   ├── .gitignore
│   └── README.md
│
└── 📊 Session Data
    └── vigileye_session_log.csv ← Live session CSV log
```

---

## ⚙️ Tech Stack

```
┌─────────────────────────────────────────────────────────────────┐
│                       TECH STACK                                │
├─────────────────┬──────────────────────────────────────────────┤
│  Layer          │  Technology                                   │
├─────────────────┼──────────────────────────────────────────────┤
│  Dashboard UI   │  Gradio 4.x  (web UI, webcam stream)         │
├─────────────────┼──────────────────────────────────────────────┤
│  Face Analysis  │  MediaPipe Face Mesh  (468 landmarks)        │
│                 │  OpenCV 4.x  (image processing)              │
│                 │  SciPy  (Euclidean distance)                  │
├─────────────────┼──────────────────────────────────────────────┤
│  Object Detect  │  Ultralytics YOLOv8n  (phone detection)      │
├─────────────────┼──────────────────────────────────────────────┤
│  Face Recog.    │  OpenCV LBPH  (100% offline, no dlib)        │
│                 │  Haar Cascade  (face detection)              │
├─────────────────┼──────────────────────────────────────────────┤
│  Alerts         │  pyttsx3  (text-to-speech)                   │
│                 │  pygame  (beep synthesis)                     │
├─────────────────┼──────────────────────────────────────────────┤
│  Visualisation  │  Matplotlib  (live charts, gauge, timeline)  │
│                 │  NumPy  (signal processing)                   │
├─────────────────┼──────────────────────────────────────────────┤
│  Reporting      │  ReportLab  (PDF generation)                  │
│                 │  CSV  (session logging)                       │
├─────────────────┼──────────────────────────────────────────────┤
│  Runtime        │  Python 3.10+  │  Docker                     │
└─────────────────┴──────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

```bash
Python 3.10+   │   Webcam   │   4GB RAM minimum   │   pip
```

### 1. Clone the Repository

```bash
git clone git clone [https://github.com/PriyankaAhirwar15/-VigilEye-V3-Advanced-Driver-Safety-System]
cd VigilEye-V3
```

### 2. Create Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the System Tests

```bash
python test_system.py
```

### 5. Run the app

```bash
python app.py
```

Expected output:
```
============================================================
  FINAL SUMMARY
============================================================

  Total Tests  : 87
  Passed       : 87
  Failed       : 0
  Pass Rate    : 100.0%

  ALL 87 TESTS PASSED! VigilEye-V3 is ready.
```

### 5. Launch the Dashboard

```bash
python app.py
```

Open your browser at: **http://localhost:7860**

---

## 🖥️ Dashboard Preview

```
┌─────────────────────────────────────────────────────────────────────┐
│  🛡️  VigilEye-V3  │  Live Fatigue Monitor                          │
├────────────────────────────┬────────────────────────────────────────┤
│                            │  Alert Status                          │
│                            │  ┌──────────────────────┐             │
│   📷 LIVE CAMERA FEED      │  │    🟢  SAFE           │             │
│                            │  └──────────────────────┘             │
│   ┌──────────────────────┐ │                                        │
│   │VigilEye-V3           │ │  Fatigue Score:  23.1                  │
│   │Fatigue: 23.1%        │ │                                        │
│   │                      │ │  EAR         │  PERCLOS               │
│   │   👤 detected        │ │  0.312       │  5.2%                   │
│   │                      │ │                                        │
│   │                      │ │  Yawns/min   │  Gaze Offset           │
│   │EAR:0.312 MAR:0.201   │ │  0/min       │  X:0.04 Y:0.08         │
│   └──────────────────────┘ │                                        │
│                            │  [ Get Session Summary ]               │
│                            │  [ Generate PDF Report  ]             │
├────────────────────────────┴────────────────────────────────────────┤
│  📊 Live Charts                                                     │
│                                                                     │
│  Fatigue Timeline    │  Component Breakdown  │  Gauge               │
│  ▁▂▃▄▅▃▂▁▂▃▄▅▄▃    │  ████ EAR    45.0     │    ╱‾╲              │
│  ──────────────────  │  ███░ PERC   30.0     │   ╱ 23 ╲            │
│                      │  █████ MAR  60.0      │  ╱______╲           │
│                      │  ██░ GAZE   20.0      │                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 📈 Performance Metrics

| Metric | Value |
|--------|-------|
| Inference Speed | ~30 FPS on CPU |
| MediaPipe Latency | < 20ms per frame |
| YOLOv8n Detection | < 40ms per frame |
| Face Recognition | Every 10th frame (optimised) |
| PERCLOS Window | 60 frames (≈ 2 seconds) |
| Alert Cooldown | 5 seconds (no spam) |
| CSV Write Speed | Async, non-blocking |
| PDF Generation | < 3 seconds |
| Test Suite | 87 tests, ~45 seconds |
| Memory Usage | ~350MB RAM typical |

### Signal Accuracy Reference

```
Signal        │  Threshold   │  Sensitivity  │  Typical Range
──────────────┼──────────────┼───────────────┼────────────────
EAR           │  < 0.22      │  High         │  0.15 – 0.40
MAR           │  > 0.45      │  Medium       │  0.20 – 0.80
PERCLOS       │  > 30%       │  High         │  2% – 80%
Gaze X        │  > 0.15      │  Medium       │  0.00 – 0.35
Gaze Y        │  > 0.25      │  Medium       │  0.00 – 0.45
Alcohol       │  3/5 signals │  Behavioural  │  0 – 100 score
```

---

## 🔬 Module Deep Dive

### `predictor.py` — Core Vision Engine

```python
# MediaPipe Face Mesh — 468 3D landmarks per frame
# Runs at 30fps on standard CPU hardware

EAR = (‖P2−P6‖ + ‖P3−P5‖) / (2 × ‖P1−P4‖)   # Eye Aspect Ratio
MAR = (‖P3−P7‖ + ‖P4−P8‖) / (2 × ‖P1−P5‖)    # Mouth Aspect Ratio

# PERCLOS: rolling 60-frame buffer
perclos_buffer.append(1 if EAR < 0.22 else 0)
PERCLOS = sum(buffer[-60:]) / 60 * 100
```

### `alcohol_detector.py` — 5-Signal Impairment Engine

```
Signal 1:  Eye Sway      — Low EAR mean + high EAR std deviation
Signal 2:  Head Sway     — High std deviation in gaze X/Y over 2 seconds
Signal 3:  Micro-sleep   — PERCLOS > 40% sustained
Signal 4:  Gaze Drift    — High variance in gaze direction buffer
Signal 5:  Yawn Spike    — > 4 yawns per minute

Trigger:   3 or more signals active simultaneously
Score:     (active_signals / 5) × 100, boosted if 3+ concurrent
```

### `night_mode.py` — CLAHE Enhancement

```
Input frame brightness:

  0  ──── 40       VERY DARK → contrast ×2.5, brightness +60, CLAHE
  40 ──── 80       DARK      → contrast ×1.8, brightness +40, CLAHE
  80 ──── 180      NORMAL    → no enhancement
  180 ─── 255      BRIGHT    → no enhancement

CLAHE: Contrast Limited Adaptive Histogram Equalization
  clipLimit=3.0, tileGridSize=(8,8)
  Applied in LAB colorspace for perceptual accuracy
```

### `face_recognition_module.py` — Offline Driver ID

```
Training:  OpenCV LBPH (Local Binary Pattern Histograms)
           100% offline — no cloud, no dlib, no DeepFace

Workflow:
  Register  → capture face → save JPEG → retrain LBPH model
  Recognize → run every 10th frame → LBPH predict → conf < 80 = match

Confidence:  LBPH confidence score (lower = better match)
             Display: 100 - raw_conf  (higher = more confident)
```

---

## 🧪 Testing

VigilEye-V3 ships with a complete automated test suite covering all modules.

```bash
python test_system.py
```

### Test Coverage

```
Section  1: Config Module           ─── 5 tests
Section  2: Fatigue Score Module    ─── 7 tests
Section  3: Alert Module            ─── 5 tests
Section  4: Logger Module           ─── 6 tests
Section  5: Night Mode Module       ─── 7 tests
Section  6: Charts Module           ─── 7 tests
Section  7: Face Recognition Module ─── 6 tests
Section  8: Phone Detector Module   ─── 5 tests
Section  9: Predictor Module        ─── 6 tests
Section 10: Alcohol Detector Module ─── 8 tests
Section 11: Report Generator Module ─── 5 tests
Section 12: Integration Tests       ─── 7 tests
Section 13: App.py Structure Checks ─── 13 tests
                                    ─────────────
                           TOTAL:   87 tests ✅
```

### What's Tested

- ✅ **Unit Tests** — every function in isolation with edge cases
- ✅ **Integration Tests** — full pipeline: night_mode → predict → score → alert → log
- ✅ **Static Analysis** — app.py structure, import order, bug detection
- ✅ **Edge Cases** — black frames, empty data, zero buffers, whitespace inputs
- ✅ **Type Safety** — bool flags, numeric ranges, key presence
- ✅ **File Generation** — PDF created, CSV written, folders exist

---

## 🐳 Docker Deployment

```bash
# Build the image
docker build -t vigileye-v3 .

# Run with webcam passthrough
docker run -p 7860:7860 \
  --device=/dev/video0:/dev/video0 \
  -v $(pwd)/reports:/app/reports \
  -v $(pwd)/drivers:/app/drivers \
  vigileye-v3

# Open dashboard
# http://localhost:7860
```

---

## 🛣️ Roadmap

```
Version   │  Feature
──────────┼────────────────────────────────────────────────
v3.0  ✅  │  Core fatigue detection, charts, PDF reports
          │  Phone detection, face recognition, alcohol
          │  Night mode, full test suite
──────────┼────────────────────────────────────────────────
v3.1  🔜  │  Mobile app (React Native + WebRTC)
          │  GPS integration + speed correlation
          │  SMS/WhatsApp emergency alert
──────────┼────────────────────────────────────────────────
v3.2  🔜  │  Fleet management dashboard
          │  Multi-camera support
          │  Historical fatigue trend analytics
──────────┼────────────────────────────────────────────────
v4.0  💡  │  Edge deployment (Raspberry Pi / Jetson Nano)
          │  Custom CNN model (replace MediaPipe)
          │  OBD-II integration (vehicle speed / brake)
```

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

```bash
# 1. Fork and clone
git clone https://github.com/yourusername/VigilEye-V3.git

# 2. Create a feature branch
git checkout -b feature/your-feature-name

# 3. Make changes, then run the full test suite
python test_system.py   # must be 87/87 before submitting

# 4. Commit with a clear message
git commit -m "feat: add X to Y module"

# 5. Push and open a Pull Request
git push origin feature/your-feature-name
```

### Code Standards

- All new modules must have corresponding tests in `test_system.py`
- Follow the existing docstring format
- No breaking changes to the `predict()` output dict keys
- CSV logger must remain UTF-8 safe (no raw emoji in file output)

---

## 📄 License

```
MIT License  ©  2024  Priyanka Ahirwar

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.
This system is for reference only — not a certified medical or legal device.
```

---

<div align="center">

**Built with ❤️ by Priyanka Ahirwar**

*VigilEye-V3 — Because every journey should have a safe ending.*

---

[![GitHub Stars](https://img.shields.io/github/stars/yourusername/VigilEye-V3?style=social)](https://github.com/yourusername/VigilEye-V3)
[![GitHub Forks](https://img.shields.io/github/forks/yourusername/VigilEye-V3?style=social)](https://github.com/yourusername/VigilEye-V3/fork)

</div>