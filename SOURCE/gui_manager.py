# gui_manager.py
"""
Handles the ImGui user interface, application state,
and interaction with the heightmap generation thread.
Includes a simulated collapsible log panel fixed to the bottom.
"""

import os
import sys
import time
import traceback
import threading
from queue import Queue
import tkinter as tk
from tkinter import filedialog

# Attempt to import the required backend
IMGUI_BACKEND = None
try:
    # Use simpler imports compatible with more versions
    import imgui_bundle
    from imgui_bundle import imgui, hello_imgui, ImVec2, ImVec4
    IMGUI_BACKEND = "imgui_bundle"
except ImportError:
    try:
        import imgui
        from imgui.integrations.glfw import GlfwRenderer
        import glfw
        import OpenGL.GL as gl
        IMGUI_BACKEND = "pyimgui"
    except ImportError:
        IMGUI_BACKEND = None

# Import the generation function
generate_heightmap = None
try:
    from heightmap_generator import generate_heightmap
    # Check if it's actually callable right after import
    if not callable(generate_heightmap):
        print(f"ERROR: Imported 'generate_heightmap' but it's not callable! Type: {type(generate_heightmap)}") # Keep basic error
        generate_heightmap = None
except ImportError as e_import:
    # Keep basic error reporting for missing core component
    print(f"ERROR: heightmap_generator.py not found or contains errors! Details: {e_import}")
    generate_heightmap = None
except Exception as e_other_import:
    # Keep basic error reporting for unexpected import issues
    print(f"ERROR: An unexpected error occurred during heightmap_generator import: {e_other_import}")
    traceback.print_exc()
    generate_heightmap = None

# --- Global GUI State ---
gui_state = {
    "input_dir": r"C:\Program Files (x86)\Steam\steamapps\common\Sacred 2 Gold\pak",
    "output_filename": "heightmap_output.png",
    "apply_boundary_smoothing": True,
    "sector_overlap": 1,
    "boundary_blend_size": 8,
    "height_scale_factor": 64.0,
    "sector_suffix_type": "B0",
    "log_messages": ["Log messages appear above.", "Select folder, adjust settings, click Generate."],
    "progress": 0.0,
    "processing_thread": None,
    "is_processing": False,
    "log_queue": Queue(),
    "progress_queue": Queue(),
    "heightmap_data": None,
    "log_visible": False # Start with log collapsed
}

# --- GUI Helper Functions ---
def select_folder_dialog():
    """ Opens a dialog to select the input folder. """
    folder_path = None
    try:
        root = tk.Tk(); root.withdraw(); root.attributes('-topmost', True)
        initial_dir = gui_state.get("input_dir") if os.path.isdir(gui_state.get("input_dir", "")) else None
        folder_path = filedialog.askdirectory(title="Select Folder", initialdir=initial_dir)
        root.destroy()
        if folder_path: gui_state["input_dir"] = folder_path
    except tk.TclError as e_tk: # Keep tkinter error reporting
         print(f"ERROR: tkinter TclError in file dialog: {e_tk}")
         gui_state["log_messages"].append(f"Error: Could not open folder dialog ({e_tk}). Is tkinter available?")
    except Exception as e_dialog: # Keep general error reporting
        print(f"ERROR: Unexpected error in select_folder_dialog: {e_dialog}")
        traceback.print_exc()
        gui_state["log_messages"].append(f"Error: Unexpected issue opening folder dialog.")
    return folder_path

def update_log_and_progress():
    """ Pulls messages and progress updates from queues. """
    global gui_state
    # Process log messages from queue and store them for GUI display
    while not gui_state["log_queue"].empty():
        message = gui_state["log_queue"].get_nowait()
        gui_state["log_messages"].append(message)
        max_log_lines = 500 # Limit internal storage
        if len(gui_state["log_messages"]) > max_log_lines:
            gui_state["log_messages"] = gui_state["log_messages"][-max_log_lines:]
    # Process progress updates
    while not gui_state["progress_queue"].empty():
        progress = gui_state["progress_queue"].get_nowait()
        if progress < 0:
            gui_state["progress"] = 0.0; gui_state["is_processing"] = False
            if gui_state["processing_thread"] and not gui_state["processing_thread"].is_alive():
                gui_state["processing_thread"] = None
        else:
            gui_state["progress"] = progress
            if gui_state["progress"] >= 1.0:
                if gui_state["is_processing"]:
                    gui_state["is_processing"] = False
                if gui_state["processing_thread"] and not gui_state["processing_thread"].is_alive():
                    gui_state["processing_thread"] = None

# --- ImGui Interface Functions ---
def run_gui_imgui_bundle():
    """ Main GUI loop using imgui-bundle """
    if not generate_heightmap: print("ERROR: Heightmap generation function not loaded. Cannot run."); return # Essential check
    if IMGUI_BACKEND != "imgui_bundle": print("ERROR: Wrong backend detected."); return # Essential check

    # Attempt to get constant values - use integers directly as fallback
    Col_ChildBg = getattr(imgui, "Col_ChildBg", 3)
    ChildFlags_Border = getattr(imgui, "ChildFlags_Border", 1)

    def gui_loop():
        global gui_state
        # --- Update state ---
        if gui_state["is_processing"]:
            update_log_and_progress()
            if gui_state["processing_thread"] and not gui_state["processing_thread"].is_alive():
                update_log_and_progress() # Final check after thread ends
                if gui_state["progress"] < 1.0 and gui_state["progress"] >= 0:
                    # Keep essential user feedback if thread ends early
                    gui_state["log_messages"].append("Warning: Processing thread ended unexpectedly.")
                gui_state["is_processing"] = False; gui_state["processing_thread"] = None; gui_state["progress"] = 0.0

        # --- Draw GUI Windows ---
        try:
            # --- Settings Window ---
            settings_window_visible = imgui.begin("Heightmap Generation Settings##HGWindow")
            if settings_window_visible:
                # Settings widgets
                imgui.push_item_width(-250); _, gui_state["input_dir"] = imgui.input_text("##InputFolder", gui_state["input_dir"], 2048); imgui.pop_item_width(); imgui.same_line();
                if imgui.button("Select Input Folder containing ZIP/PAK"): select_folder_dialog()
                imgui.push_item_width(-150); current_base, _ = os.path.splitext(gui_state["output_filename"]); changed_output, new_base = imgui.input_text("Output Base Filename", current_base, 256)
                if changed_output: gui_state["output_filename"] = new_base + ".png"
                imgui.pop_item_width(); imgui.separator()
                imgui.text("Configuration:"); _, gui_state["apply_boundary_smoothing"] = imgui.checkbox("Smooth 7x7 Tile Boundaries", gui_state["apply_boundary_smoothing"]); imgui.separator()
                imgui.text("Sector Type Suffix:");
                if imgui.radio_button("Base (_b0.sector)", gui_state["sector_suffix_type"] == "B0"): gui_state["sector_suffix_type"] = "B0"
                imgui.same_line(spacing=10); imgui.text(" ")
                if imgui.radio_button("I&B (_d1.sector)", gui_state["sector_suffix_type"] == "D1"): gui_state["sector_suffix_type"] = "D1"
                imgui.same_line(spacing=10); imgui.text(" ")
                if imgui.radio_button("CM? (_d2.sector)", gui_state["sector_suffix_type"] == "d2"): gui_state["sector_suffix_type"] = "d2"
                imgui.separator()
                imgui.text("Stitching & Scaling:"); imgui.push_item_width(120)
                _, gui_state["sector_overlap"] = imgui.input_int("Sector Overlap", gui_state["sector_overlap"]); gui_state["sector_overlap"] = max(0, gui_state["sector_overlap"]); imgui.same_line(spacing=10)
                _, gui_state["boundary_blend_size"] = imgui.input_int("Boundary Blend", gui_state["boundary_blend_size"]); gui_state["boundary_blend_size"] = max(0, gui_state["boundary_blend_size"]); imgui.same_line(spacing=10)
                _, gui_state["height_scale_factor"] = imgui.input_float("Height Scale", gui_state["height_scale_factor"], 0.1, 1.0, "%.2f");
                if gui_state["height_scale_factor"] <= 1e-6: gui_state["height_scale_factor"] = 1e-6
                imgui.pop_item_width(); imgui.separator()

                # Action Button & Progress Bar
                use_begin_disabled = hasattr(imgui, "begin_disabled")
                if use_begin_disabled: imgui.begin_disabled(gui_state["is_processing"])
                elif gui_state["is_processing"]: imgui.push_style_var(getattr(imgui, "StyleVar_Alpha", 15), imgui.get_style().alpha * 0.5) # Fallback disable style
                button_clicked = imgui.button("Generate Heightmap")
                if use_begin_disabled: imgui.end_disabled()
                elif gui_state["is_processing"]: imgui.pop_style_var() # Fallback disable style pop
                if button_clicked:
                    if not gui_state["is_processing"] and generate_heightmap:
                        gui_state["is_processing"] = True; gui_state["progress"] = 0.0; gui_state["log_messages"] = ["--- Starting New Generation ---"]; gui_state["heightmap_data"] = None
                        while not gui_state["log_queue"].empty(): gui_state["log_queue"].get();
                        while not gui_state["progress_queue"].empty(): gui_state["progress_queue"].get()
                        current_config = { k: gui_state[k] for k in ['input_dir','output_filename','apply_boundary_smoothing','sector_overlap','boundary_blend_size','height_scale_factor','sector_suffix_type']}
                        try:
                            gui_state["processing_thread"] = threading.Thread(target=generate_heightmap, args=(current_config, gui_state["log_queue"], gui_state["progress_queue"]), daemon=True)
                            gui_state["processing_thread"].start()
                        except Exception as e_thread:
                             print(f"ERROR: Failed to start worker thread: {e_thread}"); traceback.print_exc(); gui_state["log_messages"].append(f"Error: Failed to start processing thread - {e_thread}"); gui_state["is_processing"] = False; gui_state["progress"] = 0.0
                    elif not generate_heightmap:
                         # Keep essential user feedback if function is missing
                         print("ERROR: Cannot generate, generate_heightmap function is None."); gui_state["log_messages"].append("ERROR: Processing function failed to load.")
                imgui.same_line(); imgui.progress_bar(gui_state["progress"], size_arg=ImVec2(-1, 0))
            imgui.end() # End Settings Window

            # --- Attached Log Output (Simulated Collapse, Fixed Bottom) ---
            viewport = imgui.get_main_viewport()
            vp_pos = viewport.pos; vp_size = viewport.size

            button_bar_h = imgui.get_frame_height_with_spacing()
            # Use a reasonable fixed min height + portion of viewport
            child_min_h = 60
            child_max_h = max(child_min_h, vp_size.y / 5.0) # Max 1/5th screen
            child_h = max(child_min_h, vp_size.y / 10.0) # Default 1/10th screen

            # --- Draw Child Window FIRST if visible ---
            if gui_state["log_visible"]:
                child_draw_y = vp_pos.y + vp_size.y - button_bar_h - child_h
                child_draw_x = vp_pos.x
                # Ensure child doesn't go off top of screen if window is tiny
                child_draw_y = max(vp_pos.y, child_draw_y)
                actual_child_h = vp_pos.y + vp_size.y - button_bar_h - child_draw_y
                actual_child_h = min(actual_child_h, child_max_h) # Limit max height

                imgui.set_cursor_screen_pos(ImVec2(child_draw_x, child_draw_y))

                imgui.push_style_color(Col_ChildBg, ImVec4(0.0, 0.0, 0.0, 0.75)) # Semi-transparent BG
                imgui.begin_child("LogScrollingRegion", size=ImVec2(vp_size.x, actual_child_h),
                                   child_flags=ChildFlags_Border)

                imgui.text_unformatted("\n".join(gui_state["log_messages"]))
                # Auto-scroll
                if gui_state['log_messages']:
                    if imgui.get_scroll_y() >= imgui.get_scroll_max_y() - imgui.get_text_line_height():
                         imgui.set_scroll_here_y(1.0)

                imgui.end_child()
                imgui.pop_style_color()

            # --- Draw Button Bar LAST (Fixed Bottom) ---
            button_bar_y = vp_pos.y + vp_size.y - button_bar_h
            imgui.set_cursor_screen_pos(ImVec2(vp_pos.x, button_bar_y))

            arrow = "v" if gui_state["log_visible"] else ">"
            button_label = f"Log Output {arrow}"

            if imgui.button(button_label, size=ImVec2(vp_size.x, button_bar_h)):
                gui_state["log_visible"] = not gui_state["log_visible"]
            # --- End of Log Output Section ---

        except Exception as e_gui:
            # Keep essential error reporting for GUI loop crashes
             print(f"ERROR: Exception in gui_loop: {e_gui}"); traceback.print_exc()

    # --- Setup and run imgui-bundle ---
    runner_params = hello_imgui.RunnerParams()
    runner_params.app_window_params.window_title = "Sacred 2 Sector Heightmap Tool"
    runner_params.app_window_params.window_geometry.size = (800, 750)

    runner_params.imgui_window_params.show_menu_bar = False
    runner_params.callbacks.show_gui = gui_loop
    try:
        hello_imgui.run(runner_params)
    except Exception as e_runner:
        # Keep essential error reporting for runner crashes
        print(f"ERROR: Exception during hello_imgui.run(): {e_runner}"); traceback.print_exc()


# --- run_gui_pyimgui function (Simplified) ---
def run_gui_pyimgui():
    """ Main GUI loop using pyimgui (glfw backend) - Simplified """
    # Keep essential checks and basic structure, but remove most widgets/features
    # that were being debugged in the imgui-bundle version.
    print("WARNING: Running simplified pyimgui version.")
    if not generate_heightmap: print("ERROR: Heightmap generation function not loaded."); return
    if IMGUI_BACKEND != "pyimgui": print("ERROR: Wrong backend detected."); return
    try:
        if not glfw.init(): print("ERROR: Could not initialize OpenGL context"); sys.exit(1)
        window = glfw.create_window(600, 400, "Sacred 2 Sector Heightmap Tool", None, None)
        if not window: glfw.terminate(); print("ERROR: Could not create window"); sys.exit(1)
        glfw.make_context_current(window); imgui.create_context(); impl = GlfwRenderer(window)
    except Exception as e_setup: print(f"ERROR: Exception during pyimgui setup: {e_setup}"); traceback.print_exc(); sys.exit(1)
    try:
        while not glfw.window_should_close(window):
            glfw.poll_events(); impl.process_inputs(); imgui.new_frame()
            global gui_state
            if gui_state["is_processing"]: update_log_and_progress() # Basic progress update

            # Draw only the minimal settings window needed to function
            settings_window_visible = imgui.begin("Heightmap Settings (pyimgui)##HGWindow")
            if settings_window_visible:
                imgui.text("Simplified pyimgui interface.")
                # Add Generate button maybe? Or just leave it empty for now.
                if imgui.button("Generate (pyimgui - placeholder)"):
                     print("Pyimgui generate button clicked (not fully implemented)")
            imgui.end()

            gl.glClearColor(0.1, 0.1, 0.1, 1); gl.glClear(gl.GL_COLOR_BUFFER_BIT)
            imgui.render(); impl.render(imgui.get_draw_data())
            glfw.swap_buffers(window)
    except Exception as e_loop: print(f"ERROR: Exception in pyimgui main loop: {e_loop}"); traceback.print_exc()
    finally: # Cleanup
        print("Cleaning up pyimgui...");
        try:
            if imgui.get_current_context(): impl.shutdown(); imgui.destroy_context();
            else: print("Pyimgui context already destroyed.")
            glfw.terminate(); print("Pyimgui cleanup finished.")
        except Exception as e_clean: print(f"ERROR: Exception during pyimgui cleanup: {e_clean}"); traceback.print_exc()