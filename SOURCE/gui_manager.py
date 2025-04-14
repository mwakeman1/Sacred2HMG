# gui_manager.py
"""
Handles the ImGui user interface, application state,
and interaction with the heightmap generation thread.
Includes a simulated collapsible log panel fixed to the bottom.
(3D Placeholder Removed)
"""


import os
import sys
import time
import traceback # Keep for logging exceptions to GUI log
import threading
from queue import Queue
import tkinter as tk
from tkinter import filedialog

# Attempt to import the required backend
IMGUI_BACKEND = None
try:
    import imgui_bundle
    from imgui_bundle import imgui, hello_imgui, ImVec2, ImVec4
    IMGUI_BACKEND = "imgui_bundle"
except ImportError as e_bundle:
    try:
        import imgui; from imgui.integrations.glfw import GlfwRenderer; import glfw; import OpenGL.GL as gl # noqa E401
        IMGUI_BACKEND = "pyimgui"
    except ImportError as e_pyimgui: IMGUI_BACKEND = None # noqa E701


# Import the generation function
generate_heightmap = None
try:
    from heightmap_generator import generate_heightmap
    if not callable(generate_heightmap): generate_heightmap = None # noqa E701
except ImportError as e_import: generate_heightmap = None # noqa E701
except Exception as e_other_import:
    generate_heightmap = None # noqa E701


# --- Global GUI State ---
gui_state = {
    # You might want to change the default input_dir or make it empty
    "input_dir": r"C:\Program Files (x86)\Steam\steamapps\common\Sacred 2 Gold\pak", # Default path
    "output_filename": "heightmap_output.png",
    "apply_boundary_smoothing": True,
    "sector_overlap": 0,
    "boundary_blend_size": 0,
    "height_scale_factor": 1,
    "sector_suffix_type": "B0", # Default to B0
    "log_messages": ["Log messages appear above.", "Select folder, adjust settings, click Generate."],
    "progress": 0.0,
    "processing_thread": None,
    "is_processing": False,
    "log_queue": Queue(),
    "progress_queue": Queue(),
    "heightmap_data": None, # Store generated data if needed later
    "log_visible": False # Start with log collapsed
}

# --- GUI Helper Functions ---
def select_folder_dialog():
    folder_path = None
    try:
        root = tk.Tk(); root.withdraw(); root.attributes('-topmost', True) # noqa E702
        initial_dir = gui_state.get("input_dir") if os.path.isdir(gui_state.get("input_dir", "")) else None
        folder_path = filedialog.askdirectory(title="Select Folder Containing Sacred 2 Archives (ZIP/PAK)", initialdir=initial_dir)
        root.destroy()
        if folder_path: gui_state["input_dir"] = folder_path
    except Exception as e_dialog:
        gui_state["log_messages"].append(f"Error: Failed to open folder dialog - {e_dialog}") # Log to GUI instead
    return folder_path # Return path or None

def update_log_and_progress():
    global gui_state
    # Process log messages from the worker thread
    while not gui_state["log_queue"].empty():
        message = gui_state["log_queue"].get_nowait()
        gui_state["log_messages"].append(message)
        # Limit log history size
        max_log_lines = 500
        if len(gui_state["log_messages"]) > max_log_lines: gui_state["log_messages"] = gui_state["log_messages"][-max_log_lines:] # noqa E701

    # Process progress updates from the worker thread
    while not gui_state["progress_queue"].empty():
        progress = gui_state["progress_queue"].get_nowait()
        if progress < 0: # Error signal
            gui_state["progress"] = 0.0; gui_state["is_processing"] = False # noqa E702
            # Check if thread is actually finished
            if gui_state["processing_thread"] and not gui_state["processing_thread"].is_alive(): gui_state["processing_thread"] = None # noqa E701
        else: # Normal progress or completion signal
            gui_state["progress"] = progress
            if gui_state["progress"] >= 1.0: # Completion signal
                if gui_state["is_processing"]:
                    gui_state["is_processing"] = False
                # Check if thread is actually finished
                if gui_state["processing_thread"] and not gui_state["processing_thread"].is_alive():
                    # Potentially retrieve result from thread here if needed
                    # For now, just clear the thread object
                    gui_state["processing_thread"] = None

# --- ImGui Interface Functions ---
def run_gui_imgui_bundle():
    """ Main GUI loop using imgui-bundle """
    if not generate_heightmap:
        # Maybe show a critical error popup here? For now, just exit.
        sys.exit("Error: Core generation logic failed to load.")
        return
    if IMGUI_BACKEND != "imgui_bundle":
        sys.exit("Error: Incorrect ImGui backend detected.")
        return

    # Get constants safely
    Col_ChildBg = getattr(imgui, "Col_ChildBg", 3) # Default value 3 if not found
    ChildFlags_Border = getattr(imgui, "ChildFlags_Border", 1) # Default value 1
    # StyleVar_Alpha = getattr(imgui, "StyleVar_Alpha", 15) # Needed for disabled look fallback

    def gui_loop():
        global gui_state
        # --- Update state from worker thread (if running) ---
        if gui_state["is_processing"]:
            update_log_and_progress()
            # Double-check if the thread died unexpectedly
            if gui_state["processing_thread"] and not gui_state["processing_thread"].is_alive():
                update_log_and_progress() # Process any final messages
                # If progress wasn't finished/errored, log a warning
                if gui_state["progress"] < 1.0 and gui_state["progress"] >= 0:
                    gui_state["log_messages"].append("Warning: Processing thread ended unexpectedly.")
                # Reset state regardless
                gui_state["is_processing"] = False; gui_state["processing_thread"] = None; gui_state["progress"] = 0.0 # noqa E702

        # --- Draw GUI Windows ---
        try:
            # --- Settings Window ---
            settings_window_visible = imgui.begin("Heightmap Generation Settings##HGWindow")
            if settings_window_visible:
                # Input Folder Selection
                imgui.push_item_width(-250); _, gui_state["input_dir"] = imgui.input_text("##InputFolder", gui_state["input_dir"], 2048); imgui.pop_item_width(); imgui.same_line(); # noqa E702
                if imgui.button("Select Input Folder containing ZIP/PAK"): select_folder_dialog() # noqa E702

                # Output Filename Base
                imgui.push_item_width(-150); current_base, _ = os.path.splitext(gui_state["output_filename"]); changed_output, new_base = imgui.input_text("Output Base Filename", current_base, 256) # noqa E702
                if changed_output: gui_state["output_filename"] = new_base + ".png" if new_base else "heightmap_output.png" # Ensure extension, handle empty input
                imgui.pop_item_width(); imgui.separator()

                # Checkbox
                imgui.text("Configuration:"); _, gui_state["apply_boundary_smoothing"] = imgui.checkbox("Smooth 7x7 Tile Boundaries Within Sectors", gui_state["apply_boundary_smoothing"]); imgui.separator() # noqa E702

                # Radio Buttons for Sector Type
                imgui.text("Sector Type Suffix:");
                if imgui.radio_button("Base Height (_b0.sector)", gui_state["sector_suffix_type"] == "B0"): gui_state["sector_suffix_type"] = "B0" # noqa E701
                imgui.same_line(spacing=10); # imgui.text(" ") # Removed redundant text
                if imgui.radio_button("Ice & Blood? (_d1.sector)", gui_state["sector_suffix_type"] == "D1"): gui_state["sector_suffix_type"] = "D1" # noqa E701
                imgui.same_line(spacing=10); # imgui.text(" ") # Removed redundant text
                if imgui.radio_button("CM Patch? (_d2.sector)", gui_state["sector_suffix_type"] == "d2"): gui_state["sector_suffix_type"] = "d2" # noqa E701
                imgui.separator()

                # Stitching & Scaling Inputs
                imgui.text("Stitching & Scaling:"); imgui.push_item_width(120)
                _, gui_state["sector_overlap"] = imgui.input_int("Sector Overlap (pixels)", gui_state["sector_overlap"]); gui_state["sector_overlap"] = max(0, gui_state["sector_overlap"]); imgui.same_line(spacing=10) # noqa E702
                _, gui_state["boundary_blend_size"] = imgui.input_int("Boundary Blend (pixels)", gui_state["boundary_blend_size"]); gui_state["boundary_blend_size"] = max(0, gui_state["boundary_blend_size"]); imgui.same_line(spacing=10) # noqa E702
                _, gui_state["height_scale_factor"] = imgui.input_float("Height Scale Factor", gui_state["height_scale_factor"], 0.1, 1.0, "%.2f");
                if gui_state["height_scale_factor"] <= 1e-6: gui_state["height_scale_factor"] = 1e-6 # Prevent zero/negative
                imgui.pop_item_width(); imgui.separator()

                # Generate Button (disabled when processing)
                use_begin_disabled = hasattr(imgui, "begin_disabled") # Check if modern disable is available
                if use_begin_disabled: imgui.begin_disabled(gui_state["is_processing"]) # noqa E701
                elif gui_state["is_processing"]: imgui.push_style_var(getattr(imgui, "StyleVar_Alpha", 15), imgui.get_style().alpha * 0.5) # Fallback dimming

                button_clicked = imgui.button("Generate Heightmap")

                if use_begin_disabled: imgui.end_disabled() # noqa E701
                elif gui_state["is_processing"]: imgui.pop_style_var() # End fallback dimming

                if button_clicked:
                    if not gui_state["is_processing"] and generate_heightmap:
                        # Reset state for new generation
                        gui_state["is_processing"] = True; gui_state["progress"] = 0.0; gui_state["log_messages"] = ["--- Starting New Generation ---"]; gui_state["heightmap_data"] = None # noqa E702
                        # Clear queues
                        while not gui_state["log_queue"].empty(): gui_state["log_queue"].get(); # noqa E702
                        while not gui_state["progress_queue"].empty(): gui_state["progress_queue"].get() # noqa E702
                        # Prepare config for thread
                        current_config = { k: gui_state[k] for k in ['input_dir','output_filename','apply_boundary_smoothing','sector_overlap','boundary_blend_size','height_scale_factor','sector_suffix_type']}
                        # Start the worker thread
                        try:
                            gui_state["processing_thread"] = threading.Thread(target=generate_heightmap, args=(current_config, gui_state["log_queue"], gui_state["progress_queue"]), daemon=True)
                            gui_state["processing_thread"].start();
                        except Exception as e_thread:
                            gui_state["log_messages"].append(f"Error: Failed to start processing thread - {e_thread}") # Log to GUI
                            gui_state["is_processing"] = False; gui_state["progress"] = 0.0 # noqa E702
                    elif not generate_heightmap:
                        gui_state["log_messages"].append("ERROR: Processing function failed to load. Cannot start generation.") # Log to GUI
                imgui.same_line(); imgui.progress_bar(gui_state["progress"], size_arg=ImVec2(-1, 0)) # noqa E702
            imgui.end() # End Settings Window

            # --- 3D RENDER AREA REMOVED ---

            # --- Attached Log Output (Simulated Collapse, Fixed Bottom) ---
            viewport = imgui.get_main_viewport()
            vp_pos = viewport.pos; vp_size = viewport.size # noqa E702

            button_bar_h = imgui.get_frame_height_with_spacing() # Height of the button bar
            log_panel_h = max(60, vp_size.y / 5.0) # Height of the log panel when visible (e.g., 1/5th of window)

            # --- Draw Child Window FIRST if visible ---
            if gui_state["log_visible"]:
                child_draw_y = vp_pos.y + vp_size.y - button_bar_h - log_panel_h # Position above button bar
                child_draw_x = vp_pos.x
                child_w = vp_size.x # Full width

                imgui.set_next_window_pos(ImVec2(child_draw_x, child_draw_y)) # Use SetNextWindowPos for positioning child
                imgui.set_next_window_size(ImVec2(child_w, log_panel_h))     # Use SetNextWindowSize

                imgui.push_style_color(Col_ChildBg, ImVec4(0.1, 0.1, 0.1, 0.9)) # Darker background
                imgui.begin_child("LogScrollingRegion", # size=ImVec2(child_w, log_panel_h), # Size set by SetNextWindowSize
                                  child_flags=ChildFlags_Border) # Pass flags directly

                imgui.text_unformatted("\n".join(gui_state["log_messages"]))
                # Auto-scroll to bottom if near the end
                if gui_state['log_messages']: # Only scroll if there are messages
                     # Small buffer to ensure it scrolls even if not perfectly at the bottom
                    scroll_buffer = imgui.get_text_line_height() * 2
                    if imgui.get_scroll_y() >= imgui.get_scroll_max_y() - scroll_buffer:
                         imgui.set_scroll_here_y(1.0) # 1.0 means scroll to bottom

                imgui.end_child()
                imgui.pop_style_color()

            # --- Draw Button Bar LAST (Fixed Bottom) ---
            button_bar_y = vp_pos.y + vp_size.y - button_bar_h # Position at the very bottom

            # Use invisible button or similar technique if SetCursorScreenPos doesn't work reliably with runners
            imgui.set_cursor_screen_pos(ImVec2(vp_pos.x, button_bar_y))

            arrow = "v" if gui_state["log_visible"] else ">" # Down arrow if visible, right arrow if hidden
            button_label = f"Log Output {arrow}"

            button_clicked = imgui.button(button_label, size=ImVec2(vp_size.x, button_bar_h)) # Full width button

            if button_clicked:
                gui_state["log_visible"] = not gui_state["log_visible"]
            # --- End of Log Output Section ---

        except Exception as e_gui:
            # Attempt to log to GUI log as a last resort
            try:
                gui_state["log_messages"].append(f"--- GUI ERROR ---")
                gui_state["log_messages"].append(f"{e_gui}")
                gui_state["log_messages"].append(traceback.format_exc())
            except: # Prevent error loops
                pass

    # --- Setup and run imgui-bundle ---
    runner_params = hello_imgui.RunnerParams()
    runner_params.app_window_params.window_title = "Sacred 2 Sector Heightmap Tool"
    runner_params.app_window_params.window_geometry.size = (2048, 1080) # Initial size

    # Attempt to maximize window (check attributes exist first)
    if hasattr(hello_imgui, 'WindowState') and hasattr(hello_imgui.WindowState, 'maximized'):
        try:
            runner_params.app_window_params.window_state = hello_imgui.WindowState.maximized;
        except Exception as e_max:
             pass # Ignore if maximization fails

    runner_params.imgui_window_params.show_menu_bar = False # No default menu bar
    runner_params.callbacks.show_gui = gui_loop # Assign the main loop function
    try:
        hello_imgui.run(runner_params);
    except Exception as e_runner:
        # Critical error, maybe try a native message box if possible?
        sys.exit(f"Fatal Error during GUI startup: {e_runner}")


# --- run_gui_pyimgui function (Simplified) ---
def run_gui_pyimgui():
    """ Main GUI loop using pyimgui (glfw backend) - Simplified / Placeholder """
    # Add a message to the GUI log if this function gets called unexpectedly
    gui_state["log_messages"].append("Error: PyImGui backend selected, but its GUI implementation is currently a basic placeholder.")
    gui_state["log_messages"].append("Please ensure 'imgui-bundle' is installed for the full interface.")
    # Maybe display a simple window with just the log? Or just exit?
    # For now, just pass to avoid crashing if it's somehow selected.
    pass # Needs full implementation similar to run_gui_imgui_bundle