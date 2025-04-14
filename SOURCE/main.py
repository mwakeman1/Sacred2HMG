# main.py
"""
Main entry point for the Sacred 2 Heightmap Tool application.
Detects the available ImGui backend and launches the appropriate GUI.
"""

import sys
import gui_manager # Import the gui module

# Check which backend was successfully imported in gui_manager
if gui_manager.IMGUI_BACKEND == "imgui_bundle":
    print("Using imgui-bundle backend.")
    gui_manager.run_gui_imgui_bundle()
elif gui_manager.IMGUI_BACKEND == "pyimgui":
    print("Using pyimgui (glfw) backend.")
    gui_manager.run_gui_pyimgui()
else:
    print("Error: No suitable ImGui backend found (imgui-bundle or pyimgui).")
    print("Please install either 'imgui-bundle' or 'pyimgui' with its dependencies (glfw, PyOpenGL).")
    sys.exit(1)