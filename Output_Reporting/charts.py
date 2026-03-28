# ---------------------------------------------------------
#   VigilEye-V3  |  charts.py
#   Live Analytics & Visualization Engine
# ---------------------------------------------------------

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for Streamlit stability
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from collections import deque

# UPDATED IMPORT: Pointing to the new Config_Files directory
from Config_Files.config import CHART_HISTORY

# Standardized Color Palette for Driver Safety
SAFE_COLOR     = "#00CC44"  # Green
MILD_COLOR     = "#FFD700"  # Gold/Yellow
MODERATE_COLOR = "#FF6600"  # Orange
CRITICAL_COLOR = "#FF0000"  # Red

def get_zone_color(score):
    """Returns the corresponding color code based on the fatigue score severity."""
    if score >= 90:
        return CRITICAL_COLOR
    elif score >= 70:
        return MODERATE_COLOR
    elif score >= 40:
        return MILD_COLOR
    else:
        return SAFE_COLOR

def draw_fatigue_chart(score_history):
    """
    Generates a live fatigue score line chart.
    Features colored safety zones and dynamic line colors.
    """
    fig, ax = plt.subplots(figsize=(8, 3))
    fig.patch.set_facecolor("#1a1a2e")  # Dark UI Background
    ax.set_facecolor("#16213e")

    scores = list(score_history)

    if len(scores) == 0:
        ax.text(0.5, 0.5, "Awaiting AI Data...",
                ha="center", va="center",
                color="white", fontsize=12,
                transform=ax.transAxes)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 100)
    else:
        x = list(range(len(scores)))

        # Visual background bands for safety thresholds
        ax.axhspan(0,  40, alpha=0.08, color=SAFE_COLOR)
        ax.axhspan(40, 70, alpha=0.08, color=MILD_COLOR)
        ax.axhspan(70, 90, alpha=0.08, color=MODERATE_COLOR)
        ax.axhspan(90, 100, alpha=0.08, color=CRITICAL_COLOR)

        # Dashed threshold lines for better visibility
        ax.axhline(y=40, color=MILD_COLOR, linestyle="--", linewidth=0.8, alpha=0.5)
        ax.axhline(y=70, color=MODERATE_COLOR, linestyle="--", linewidth=0.8, alpha=0.5)
        ax.axhline(y=90, color=CRITICAL_COLOR, linestyle="--", linewidth=0.8, alpha=0.5)

        # Plot score segments with dynamic coloring
        for i in range(1, len(scores)):
            color = get_zone_color(scores[i])
            ax.plot([x[i-1], x[i]], [scores[i-1], scores[i]],
                    color=color, linewidth=2.0)

        # Visual fill below the line
        ax.fill_between(x, scores, alpha=0.15,
                        color=get_zone_color(scores[-1]))

        # Current state indicator dot
        ax.scatter([x[-1]], [scores[-1]],
                   color=get_zone_color(scores[-1]),
                   s=60, zorder=5)

        ax.set_xlim(0, max(CHART_HISTORY, len(scores)))
        ax.set_ylim(0, 100)

    # Chart aesthetics and labeling
    ax.set_title("Live Fatigue Score Progression", color="white", fontsize=11, pad=8)
    ax.set_xlabel("Time (Frames)", color="#aaaaaa", fontsize=9)
    ax.set_ylabel("Score %", color="#aaaaaa", fontsize=9)
    ax.tick_params(colors="#aaaaaa", labelsize=8)
    
    for spine in ax.spines.values():
        spine.set_edgecolor("#333355")

    # Dynamic Legend
    patches = [
        mpatches.Patch(color=SAFE_COLOR,     label="Safe (0-40)"),
        mpatches.Patch(color=MILD_COLOR,     label="Mild (40-70)"),
        mpatches.Patch(color=MODERATE_COLOR, label="Moderate (70-90)"),
        mpatches.Patch(color=CRITICAL_COLOR, label="Critical (90+)"),
    ]
    ax.legend(handles=patches, loc="upper left",
              fontsize=7, facecolor="#1a1a2e",
              labelcolor="white", framealpha=0.7)

    plt.tight_layout()
    return fig

def draw_component_chart(score_data):
    """
    Renders a bar chart breaking down the specific AI components:
    Eye state, PERCLOS, Mouth opening, and Gaze distraction.
    """
    fig, ax = plt.subplots(figsize=(8, 3))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")

    if not score_data:
        ax.text(0.5, 0.5, "Processing Signals...",
                ha="center", va="center",
                color="white", fontsize=12,
                transform=ax.transAxes)
    else:
        labels = ["Eye (EAR)", "PERCLOS", "Mouth (MAR)", "Gaze"]
        values = [
            score_data.get("eye_component", 0),
            score_data.get("perclos_component", 0),
            score_data.get("mouth_component", 0),
            score_data.get("gaze_component", 0),
        ]
        colors = [get_zone_color(v) for v in values]

        bars = ax.bar(labels, values, color=colors, alpha=0.85, width=0.5)

        # Annotate bars with numeric values
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1.5,
                    f"{val:.1f}", ha="center", va="bottom",
                    color="white", fontsize=9)

        ax.set_ylim(0, 110)

    ax.set_title("Fatigue Component Breakdown", color="white", fontsize=11, pad=8)
    ax.set_ylabel("Severity Score", color="#aaaaaa", fontsize=9)
    ax.tick_params(colors="#aaaaaa", labelsize=9)
    
    for spine in ax.spines.values():
        spine.set_edgecolor("#333355")

    plt.tight_layout()
    return fig

def draw_gauge_chart(fatigue_score):
    """
    Generates a polar projection gauge (speedometer style) for the current fatigue score.
    """
    fig, ax = plt.subplots(figsize=(4, 3), subplot_kw={"projection": "polar"})
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#1a1a2e")

    # Define polar arc segments
    theta_safe     = np.linspace(np.pi, np.pi * 1.40, 100)
    theta_mild     = np.linspace(np.pi * 1.40, np.pi * 1.70, 100)
    theta_moderate = np.linspace(np.pi * 1.70, np.pi * 1.90, 100)
    theta_critical = np.linspace(np.pi * 1.90, np.pi * 2.00, 100)

    # Plot the gauge background colors
    ax.plot(theta_safe,     [1]*100, color=SAFE_COLOR,     lw=12, alpha=0.7)
    ax.plot(theta_mild,     [1]*100, color=MILD_COLOR,     lw=12, alpha=0.7)
    ax.plot(theta_moderate, [1]*100, color=MODERATE_COLOR, lw=12, alpha=0.7)
    ax.plot(theta_critical, [1]*100, color=CRITICAL_COLOR, lw=12, alpha=0.7)

    # Needle calculation based on score percentage
    needle_angle = np.pi + (fatigue_score / 100) * np.pi
    ax.annotate("", xy=(needle_angle, 0.85), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color="white", lw=2.5))

    # Center score value
    ax.text(0, -0.3, f"{fatigue_score:.0f}",
            ha="center", va="center",
            fontsize=22, fontweight="bold",
            color=get_zone_color(fatigue_score),
            transform=ax.transData)

    ax.set_ylim(0, 1.2)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.spines["polar"].set_visible(False)
    ax.set_title("Real-time Fatigue Intensity", color="white", fontsize=10, pad=4)

    plt.tight_layout()
    return fig