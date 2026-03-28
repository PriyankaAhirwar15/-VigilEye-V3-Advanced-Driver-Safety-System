# ---------------------------------------------------------
#   VigilEye-V3  |  report_generator.py
#   Professional Driver Session PDF Report Generator
# ---------------------------------------------------------

import os
from datetime import datetime
import matplotlib
matplotlib.use("Agg")  # Required for background processing without a GUI
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# PDF Generation imports
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer,
    Table, TableStyle, Image, HRFlowable
)
from reportlab.lib.enums import TA_CENTER

# Constants for report layout
PAGE_W, PAGE_H = A4
REPORTS_FOLDER = "Reports" 

class ReportGenerator:
    """
    Generates a detailed PDF report summarizing the driving session.
    Includes fatigue timelines, component analysis, and AI-driven safety tips.
    """

    def __init__(self):
        # Ensure the output directory exists
        if not os.path.exists(REPORTS_FOLDER):
            os.makedirs(REPORTS_FOLDER)
        print("[VigilEye-V3] PDF Report Generator initialized successfully.")

    def _get_report_path(self, driver_name="Driver"):
        """Creates a unique filename for the PDF report."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = driver_name.replace(" ", "_")
        return os.path.join(REPORTS_FOLDER, f"VigilEye_Report_{safe_name}_{timestamp}.pdf")

    def _save_fatigue_chart(self, score_history, path):
        """Generates the fatigue timeline chart for the PDF."""
        fig, ax = plt.subplots(figsize=(10, 3))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("#f8f8f8")
        
        scores = list(score_history)
        if scores:
            x = list(range(len(scores)))
            # Visual zones for safety levels
            ax.axhspan(0,  40, alpha=0.1, color="#00CC44")  # Safe
            ax.axhspan(40, 70, alpha=0.1, color="#FFD700")  # Mild
            ax.axhspan(70, 90, alpha=0.1, color="#FF6600")  # Moderate
            ax.axhspan(90, 100, alpha=0.1, color="#FF0000") # Critical
            
            ax.plot(x, scores, color="#1a73e8", linewidth=2, label="Fatigue Score")
            ax.fill_between(x, scores, alpha=0.1, color="#1a73e8")
            ax.set_ylim(0, 100)
            ax.set_xlim(0, max(len(scores), 1))

        ax.set_title("Fatigue Score Timeline", fontsize=12, pad=8)
        ax.set_xlabel("Time (Frames)")
        ax.set_ylabel("Score (0-100)")
        
        # Legend configuration
        patches = [
            mpatches.Patch(color="#00CC44", label="Safe (0-40)"),
            mpatches.Patch(color="#FFD700", label="Mild (40-70)"),
            mpatches.Patch(color="#FF6600", label="Moderate (70-90)"),
            mpatches.Patch(color="#FF0000", label="Critical (90+)"),
        ]
        ax.legend(handles=patches, loc="upper right", fontsize=8, framealpha=0.8)
        
        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()

    def _save_component_chart(self, avg_components, path):
        """Generates a bar chart showing the average scores of AI components."""
        fig, ax = plt.subplots(figsize=(6, 3))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("#f8f8f8")
        
        labels = ["Eye (EAR)", "PERCLOS", "Mouth", "Gaze"]
        values = [
            avg_components.get("eye", 0),
            avg_components.get("perclos", 0),
            avg_components.get("mouth", 0),
            avg_components.get("gaze", 0),
        ]
        
        # Color coding bars based on severity
        bar_colors = []
        for v in values:
            if v >= 90:   bar_colors.append("#FF0000")
            elif v >= 70: bar_colors.append("#FF6600")
            elif v >= 40: bar_colors.append("#FFD700")
            else:         bar_colors.append("#00CC44")
            
        bars = ax.bar(labels, values, color=bar_colors, alpha=0.85, width=0.5)
        
        # Adding labels on top of bars
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1.5,
                f"{val:.1f}", ha="center", va="bottom", fontsize=9
            )
            
        ax.set_ylim(0, 110)
        ax.set_title("Average Component Scores", fontsize=11, pad=8)
        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()

    def _get_recommendations(self, avg_score, total_alerts, yawn_count, alcohol_score):
        """Generates safety recommendations based on calculated session metrics."""
        recs = []
        if avg_score >= 70:
            recs.append("CRITICAL: High fatigue levels detected. Do not operate a vehicle while in this state.")
        elif avg_score >= 40:
            recs.append("WARNING: Moderate fatigue detected. Ensure you take breaks every 2 hours.")
        else:
            recs.append("STATUS: Good alertness levels maintained. Keep staying hydrated.")
        
        if yawn_count > 10:
            recs.append("Frequent yawning detected. Suggests a lack of deep sleep; prioritize rest tonight.")
        
        if total_alerts > 5:
            recs.append("Frequent safety alerts triggered. Consider reviewing seat position or lighting.")
        
        if alcohol_score >= 60:
            recs.append("DANGER: Alcohol impairment signals detected. Seek alternative transportation.")
        
        if not recs:
            recs.append("Excellent driving session. Continue following all road safety protocols.")
        return recs

    def generate(self, session_data, score_history, avg_components, driver_name="Driver"):
        """Main method to compile data into a professional PDF document."""
        report_path    = self._get_report_path(driver_name)
        chart_path     = os.path.join(REPORTS_FOLDER, "temp_fatigue_chart.png")
        component_path = os.path.join(REPORTS_FOLDER, "temp_component_chart.png")

        # Create temporary visual assets
        self._save_fatigue_chart(score_history, chart_path)
        self._save_component_chart(avg_components, component_path)

        doc = SimpleDocTemplate(
            report_path, pagesize=A4,
            topMargin=1.5*cm, bottomMargin=1.5*cm,
            leftMargin=2*cm,  rightMargin=2*cm
        )
        styles = getSampleStyleSheet()
        story  = []

        # Define custom styles for branding
        title_style = ParagraphStyle(
            "title", parent=styles["Title"],
            fontSize=22, textColor=colors.HexColor("#1a1a2e"),
            spaceAfter=6, alignment=TA_CENTER
        )
        subtitle_style = ParagraphStyle(
            "subtitle", parent=styles["Normal"],
            fontSize=11, textColor=colors.HexColor("#666666"),
            alignment=TA_CENTER, spaceAfter=4
        )
        heading_style = ParagraphStyle(
            "heading", parent=styles["Heading2"],
            fontSize=13, textColor=colors.HexColor("#1a73e8"),
            spaceBefore=12, spaceAfter=6
        )

        # -- Header Section -----------------------------------------
        story.append(Paragraph("VigilEye-V3", title_style))
        story.append(Paragraph("Driver Safety Analysis Report", subtitle_style))
        story.append(Paragraph(f"Generated on: {datetime.now().strftime('%d %B %Y | %I:%M %p')}", subtitle_style))
        story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#1a73e8"), spaceAfter=12))

        # -- Session Statistics Table --------------------------------
        story.append(Paragraph("Executive Summary", heading_style))
        avg_score = session_data.get("avg_fatigue_score", 0)
        
        # Determine overall risk status
        if avg_score >= 70:
            risk_level, risk_color = "HIGH RISK", colors.red
        elif avg_score >= 40:
            risk_level, risk_color = "MODERATE RISK", colors.orange
        else:
            risk_level, risk_color = "LOW RISK", colors.green

        table_data = [
            ["Metric", "Result"],
            ["Driver Name",           session_data.get("driver_name", driver_name)],
            ["Duration",              session_data.get("session_duration", "N/A")],
            ["Frames Analyzed",       str(session_data.get("total_frames", 0))],
            ["Avg Fatigue Score",     f"{avg_score:.1f} / 100"],
            ["Total Safety Alerts",   str(session_data.get("total_alerts", 0))],
            ["Alcohol Impairment",    f"{session_data.get('alcohol_score', 0):.1f} / 100"],
            ["Final Safety Status",   risk_level],
        ]
        
        summary_table = Table(table_data, colWidths=[7*cm, 10*cm])
        summary_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a73e8")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#dddddd")),
            ("BACKGROUND", (0, -1), (-1, -1), colors.HexColor("#f8f9fa")),
            ("TEXTCOLOR", (1, -1), (1, -1), risk_color),
            ("FONTNAME", (1, -1), (1, -1), "Helvetica-Bold"),
            ("PADDING", (0, 0), (-1, -1), 8),
        ]))
        story.append(summary_table)

        # -- Charts Section -----------------------------------------
        story.append(Paragraph("Fatigue Progression Timeline", heading_style))
        if os.path.exists(chart_path):
            story.append(Image(chart_path, width=16*cm, height=5*cm))

        story.append(Paragraph("AI Component Analysis", heading_style))
        if os.path.exists(component_path):
            story.append(Image(component_path, width=12*cm, height=5*cm))

        # -- AI Recommendations Section ------------------------------
        story.append(Paragraph("Personalized Safety Advice", heading_style))
        recommendations = self._get_recommendations(
            avg_score,
            session_data.get("total_alerts", 0),
            session_data.get("total_yawns", 0),
            session_data.get("alcohol_score", 0)
        )
        for i, rec in enumerate(recommendations, 1):
            story.append(Paragraph(f"• {rec}", styles["Normal"]))

        # -- Footer --------------------------------------------------
        story.append(Spacer(1, 1*cm))
        story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#dddddd")))
        story.append(Paragraph("VigilEye-V3 | Automated Safety Analytics | Internal Use Only", subtitle_style))

        # Build PDF
        doc.build(story)

        # Cleanup temporary files
        for p in [chart_path, component_path]:
            if os.path.exists(p): os.remove(p)

        print(f"[VigilEye-V3] PDF Report saved to: {report_path}")
        return report_path