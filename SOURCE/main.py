# main.py
"""
Main entry point for the Sacred 2 Tools application.
"""

import sys
import gui_manager

if gui_manager.IMGUI_BACKEND == "imgui_bundle":
    gui_manager.run_gui_imgui_bundle()
elif gui_manager.IMGUI_BACKEND == "pyimgui":
    gui_manager.run_gui_pyimgui()
else:
    sys.exit(1)
