# gui_manager.py
"""
Handles the ImGui user interface, application state,
and interaction with the heightmap generation thread.
Includes a simulated collapsible log panel fixed to the bottom.
(3D Placeholder Removed)
"""

print("DEBUG: gui_manager.py - Starting script execution")

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
    print("DEBUG: Attempting to import imgui_bundle...")
    import imgui_bundle
    from imgui_bundle import imgui, hello_imgui, ImVec2, ImVec4
    print("DEBUG: imgui_bundle imported successfully.")
    IMGUI_BACKEND = "imgui_bundle"
except ImportError as e_bundle:
    print(f"DEBUG: imgui_bundle import failed: {e_bundle}")
    try:
        print("DEBUG: Attempting to import pyimgui...")
        import imgui; from imgui.integrations.glfw import GlfwRenderer; import glfw; import OpenGL.GL as gl # noqa E401
        print("DEBUG: pyimgui and dependencies imported successfully.")
        IMGUI_BACKEND = "pyimgui"
    except ImportError as e_pyimgui: print(f"DEBUG: pyimgui import failed: {e_pyimgui}"); IMGUI_BACKEND = None # noqa E701

print(f"DEBUG: Determined IMGUI_BACKEND = {IMGUI_BACKEND}")

# Import the generation function
generate_heightmap = None
try:
    print("DEBUG: Attempting to import generate_heightmap from heightmap_generator...")
    from heightmap_generator import generate_heightmap
    print("DEBUG: generate_heightmap imported successfully.")
    if not callable(generate_heightmap): print(f"ERROR: Imported 'generate_heightmap' but it's not callable! Type: {type(generate_heightmap)}"); generate_heightmap = None # noqa E701
except ImportError as e_import: print(f"ERROR: heightmap_generator.py not found or contains errors! Details: {e_import}"); generate_heightmap = None # noqa E701
except Exception as e_other_import: print(f"ERROR: An unexpected error occurred during heightmap_generator import: {e_other_import}"); traceback.print_exc(); generate_heightmap = None # noqa E701

print(f"DEBUG: generate_heightmap function object after import block: {generate_heightmap}")

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
    print("DEBUG: select_folder_dialog called")
    folder_path = None
    try:
        root = tk.Tk(); root.withdraw(); root.attributes('-topmost', True)
        initial_dir = gui_state.get("input_dir") if os.path.isdir(gui_state.get("input_dir", "")) else None
        folder_path = filedialog.askdirectory(title="Select Folder", initialdir=initial_dir)
        root.destroy()
        if folder_path: gui_state["input_dir"] = folder_path
    except Exception as e_dialog: print(f"ERROR: Unexpected error in select_folder_dialog: {e_dialog}"); traceback.print_exc(); print("ERROR: Unexpected issue opening folder dialog.") # noqa E701
    return folder_path

def update_log_and_progress():
    global gui_state
    while not gui_state["log_queue"].empty():
        message = gui_state["log_queue"].get_nowait()
        print(f"LOG: {message}") # Also print to console
        gui_state["log_messages"].append(message)
        max_log_lines = 500
        if len(gui_state["log_messages"]) > max_log_lines: gui_state["log_messages"] = gui_state["log_messages"][-max_log_lines:] # noqa E701
    while not gui_state["progress_queue"].empty():
        progress = gui_state["progress_queue"].get_nowait()
        if progress < 0:
            print("DEBUG: Received error signal (progress < 0) from worker thread.")
            gui_state["progress"] = 0.0; gui_state["is_processing"] = False
            if gui_state["processing_thread"] and not gui_state["processing_thread"].is_alive(): gui_state["processing_thread"] = None # noqa E701
        else:
            gui_state["progress"] = progress
            if gui_state["progress"] >= 1.0:
                if gui_state["is_processing"]: print("DEBUG: Received completion signal (progress >= 1.0)."); gui_state["is_processing"] = False # noqa E701
                if gui_state["processing_thread"] and not gui_state["processing_thread"].is_alive(): print("DEBUG: Worker thread confirmed finished after completion signal."); gui_state["processing_thread"] = None # noqa E701

# --- ImGui Interface Functions ---
def run_gui_imgui_bundle():
    """ Main GUI loop using imgui-bundle """
    print("DEBUG: run_gui_imgui_bundle called")
    if not generate_heightmap: print("ERROR: Heightmap generation function not loaded (was None). Cannot run GUI."); return # noqa E701
    if IMGUI_BACKEND != "imgui_bundle": print("ERROR: Wrong backend detected."); return # noqa E701

    Col_ChildBg = getattr(imgui, "Col_ChildBg", 3)
    ChildFlags_Border = getattr(imgui, "ChildFlags_Border", 1)
    # StyleVar_Alpha = getattr(imgui, "StyleVar_Alpha", 15)

    def gui_loop():
        global gui_state
        # --- Update state ---
        if gui_state["is_processing"]:
            update_log_and_progress()
            if gui_state["processing_thread"] and not gui_state["processing_thread"].is_alive():
                print("DEBUG: Worker thread found dead while still marked as processing.")
                update_log_and_progress()
                if gui_state["progress"] < 1.0 and gui_state["progress"] >= 0: gui_state["log_messages"].append("Warning: Processing thread ended unexpectedly.") # noqa E701
                gui_state["is_processing"] = False; gui_state["processing_thread"] = None; gui_state["progress"] = 0.0 # noqa E701

        # --- Draw GUI Windows ---
        try:
            # --- Settings Window ---
            settings_window_visible = imgui.begin("Heightmap Generation Settings##HGWindow")
            if settings_window_visible:
                # ... (Settings widgets - code omitted for brevity) ...
                imgui.push_item_width(-250); _, gui_state["input_dir"] = imgui.input_text("##InputFolder", gui_state["input_dir"], 2048); imgui.pop_item_width(); imgui.same_line(); # noqa E702
                if imgui.button("Select Input Folder containing ZIP/PAK"): select_folder_dialog() # noqa E702
                imgui.push_item_width(-150); current_base, _ = os.path.splitext(gui_state["output_filename"]); changed_output, new_base = imgui.input_text("Output Base Filename", current_base, 256) # noqa E702
                if changed_output: gui_state["output_filename"] = new_base + ".png" # noqa E701
                imgui.pop_item_width(); imgui.separator()
                imgui.text("Configuration:"); _, gui_state["apply_boundary_smoothing"] = imgui.checkbox("Smooth 7x7 Tile Boundaries", gui_state["apply_boundary_smoothing"]); imgui.separator() # noqa E702
                imgui.text("Sector Type Suffix:");
                if imgui.radio_button("Base (_b0.sector)", gui_state["sector_suffix_type"] == "B0"): gui_state["sector_suffix_type"] = "B0" # noqa E701
                imgui.same_line(spacing=10); imgui.text(" ")
                if imgui.radio_button("I&B (_d1.sector)", gui_state["sector_suffix_type"] == "D1"): gui_state["sector_suffix_type"] = "D1" # noqa E701
                imgui.same_line(spacing=10); imgui.text(" ")
                if imgui.radio_button("CM? (_d2.sector)", gui_state["sector_suffix_type"] == "d2"): gui_state["sector_suffix_type"] = "d2" # noqa E701
                imgui.separator()
                imgui.text("Stitching & Scaling:"); imgui.push_item_width(120)
                _, gui_state["sector_overlap"] = imgui.input_int("Sector Overlap", gui_state["sector_overlap"]); gui_state["sector_overlap"] = max(0, gui_state["sector_overlap"]); imgui.same_line(spacing=10) # noqa E702
                _, gui_state["boundary_blend_size"] = imgui.input_int("Boundary Blend", gui_state["boundary_blend_size"]); gui_state["boundary_blend_size"] = max(0, gui_state["boundary_blend_size"]); imgui.same_line(spacing=10) # noqa E702
                _, gui_state["height_scale_factor"] = imgui.input_float("Height Scale", gui_state["height_scale_factor"], 0.1, 1.0, "%.2f");
                if gui_state["height_scale_factor"] <= 1e-6: gui_state["height_scale_factor"] = 1e-6 # noqa E701
                imgui.pop_item_width(); imgui.separator()
                use_begin_disabled = hasattr(imgui, "begin_disabled")
                if use_begin_disabled: imgui.begin_disabled(gui_state["is_processing"]) # noqa E701
                elif gui_state["is_processing"]: imgui.push_style_var(getattr(imgui, "StyleVar_Alpha", 15), imgui.get_style().alpha * 0.5) # noqa E701
                button_clicked = imgui.button("Generate Heightmap")
                if use_begin_disabled: imgui.end_disabled() # noqa E701
                elif gui_state["is_processing"]: imgui.pop_style_var() # noqa E701
                if button_clicked:
                    print("DEBUG: 'Generate Heightmap' button clicked.")
                    if not gui_state["is_processing"] and generate_heightmap:
                        print("DEBUG: Starting processing sequence...")
                        gui_state["is_processing"] = True; gui_state["progress"] = 0.0; gui_state["log_messages"] = ["--- Starting New Generation ---"]; gui_state["heightmap_data"] = None # noqa E702
                        while not gui_state["log_queue"].empty(): gui_state["log_queue"].get(); # noqa E702
                        while not gui_state["progress_queue"].empty(): gui_state["progress_queue"].get() # noqa E702
                        current_config = { k: gui_state[k] for k in ['input_dir','output_filename','apply_boundary_smoothing','sector_overlap','boundary_blend_size','height_scale_factor','sector_suffix_type']}
                        print(f"DEBUG: Config for worker thread: {current_config}")
                        try:
                            gui_state["processing_thread"] = threading.Thread(target=generate_heightmap, args=(current_config, gui_state["log_queue"], gui_state["progress_queue"]), daemon=True)
                            print("DEBUG: Starting worker thread..."); gui_state["processing_thread"].start(); print("DEBUG: Worker thread started.") # noqa E702
                        except Exception as e_thread: print(f"ERROR: Failed to start worker thread: {e_thread}"); traceback.print_exc(); gui_state["log_messages"].append(f"Error: Failed to start processing thread - {e_thread}"); gui_state["is_processing"] = False; gui_state["progress"] = 0.0 # noqa E702
                    elif not generate_heightmap: print("ERROR: Cannot generate, generate_heightmap function is None."); gui_state["log_messages"].append("ERROR: Processing function failed to load.") # noqa E701
                imgui.same_line(); imgui.progress_bar(gui_state["progress"], size_arg=ImVec2(-1, 0)) # noqa E702
            imgui.end() # End Settings Window


            # --- Placeholder for future 3D Render Area REMOVED ---


            # --- Attached Log Output (Simulated Collapse, Fixed Bottom) ---
            # print("DEBUG: Drawing Log Section...") # <--- DEBUG: Start of section
            viewport = imgui.get_main_viewport()
            vp_pos = viewport.pos; vp_size = viewport.size

            button_bar_h = imgui.get_frame_height_with_spacing()
            child_h = max(60, vp_size.y / 10.0)
            # print(f"DEBUG: vp_size.y={vp_size.y:.1f}, button_bar_h={button_bar_h:.1f}, child_h={child_h:.1f}") # <--- DEBUG: Heights

            # --- Draw Child Window FIRST if visible ---
            if gui_state["log_visible"]:
                child_draw_y = vp_pos.y + vp_size.y - button_bar_h - child_h
                child_draw_x = vp_pos.x
                # print(f"DEBUG: Drawing Log Child at x={child_draw_x:.1f}, y={child_draw_y:.1f}, w={vp_size.x:.1f}, h={child_h:.1f}") # <--- DEBUG: Child Pos/Size

                imgui.set_cursor_screen_pos(ImVec2(child_draw_x, child_draw_y))

                imgui.push_style_color(Col_ChildBg, ImVec4(0.0, 0.0, 0.0, 0.75))
                imgui.begin_child("LogScrollingRegion", size=ImVec2(vp_size.x, child_h), # Use desired width/height
                                   child_flags=ChildFlags_Border)

                imgui.text_unformatted("\n".join(gui_state["log_messages"]))
                if gui_state['log_messages']:
                    if imgui.get_scroll_y() >= imgui.get_scroll_max_y() - imgui.get_text_line_height():
                         imgui.set_scroll_here_y(1.0)

                imgui.end_child()
                imgui.pop_style_color()
            # else: # <--- DEBUG: Added else block
            #     print("DEBUG: Log Child is NOT visible, skipping draw.")


            # --- Draw Button Bar LAST (Fixed Bottom) ---
            button_bar_y = vp_pos.y + vp_size.y - button_bar_h
            # print(f"DEBUG: Drawing Button Bar at y={button_bar_y:.1f}") # <--- DEBUG: Button Pos

            imgui.set_cursor_screen_pos(ImVec2(vp_pos.x, button_bar_y))

            arrow = "v" if gui_state["log_visible"] else ">"
            button_label = f"Log Output {arrow}"

            button_clicked = imgui.button(button_label, size=ImVec2(vp_size.x, button_bar_h))
            # print(f"DEBUG: Button Click Result: {button_clicked}") # <--- DEBUG: Button Result

            if button_clicked:
                gui_state["log_visible"] = not gui_state["log_visible"]
                print(f"DEBUG: Toggled log visibility to: {gui_state['log_visible']}") # <--- DEBUG: Toggle Action
            # --- End of Log Output Section ---


        except Exception as e_gui: print(f"ERROR: Exception in gui_loop: {e_gui}"); traceback.print_exc() # noqa E701

    # --- Setup and run imgui-bundle ---
    runner_params = hello_imgui.RunnerParams()
    runner_params.app_window_params.window_title = "Sacred 2 Sector Heightmap Tool"
    runner_params.app_window_params.window_geometry.size = (800, 750)
    print(f"DEBUG: Checking for hello_imgui.WindowState... Exists: {hasattr(hello_imgui, 'WindowState')}")
    if hasattr(hello_imgui, 'WindowState') and hasattr(hello_imgui.WindowState, 'maximized'):
        try: runner_params.app_window_params.window_state = hello_imgui.WindowState.maximized; print("DEBUG: Set window_state to maximized.") # noqa E701
        except Exception as e_max: print(f"WARNING: Failed to set window_state to maximized: {e_max}") # noqa E701
    else: print("WARNING: hello_imgui.WindowState.maximized not found. Cannot set maximized state.") # noqa E701
    runner_params.imgui_window_params.show_menu_bar = False
    runner_params.callbacks.show_gui = gui_loop
    try:
        print("DEBUG: Calling hello_imgui.run()..."); hello_imgui.run(runner_params); print("DEBUG: hello_imgui.run() finished.") # noqa E702
    except Exception as e_runner: print(f"ERROR: Exception during hello_imgui.run(): {e_runner}"); traceback.print_exc() # noqa E701


# --- run_gui_pyimgui function (Simplified) ---
def run_gui_pyimgui():
    """ Main GUI loop using pyimgui (glfw backend) - Simplified """
    print("WARNING: run_gui_pyimgui is simplified and does not contain latest layout.")
    # ... (Existing simplified pyimgui code) ...
    pass