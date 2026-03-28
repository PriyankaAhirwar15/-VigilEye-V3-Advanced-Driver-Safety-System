# ---------------------------------------------------------
#   VigilEye-V3  |  main.py
#   Application Entry Point
# ---------------------------------------------------------

import os
import sys

# Ensure the root directory is in the path so imports work correctly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def initialize_project():
    """Checks and creates necessary folders before starting."""
    folders = ["Session Data", "Reports", "Known_Drivers"]
    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder)
            print(f"[System] Created folder: {folder}")

if __name__ == "__main__":
    print("--- VigilEye-V3: AI Driver Safety System ---")
    initialize_project()
    
    print("[System] Loading AI Models and Dashboard...")
    try:
        from app import app
        # Launching the Gradio app
        app.launch(server_name="0.0.0.0", server_port=7860, share=False)
    except Exception as e:
        print(f"[CRITICAL ERROR] Failed to start application: {e}")