# --- gui manager ---
# Written for use with the Sacred 2 tools.
# Manages the UI using pyimgui

import os
import sys
import time
import traceback
import threading
from queue import Queue
import tkinter as tk
from tkinter import filedialog

IMGUI_BACKEND = None
try:
    import imgui_bundle
    from imgui_bundle import imgui, hello_imgui, ImVec2, ImVec4
    IMGUI_BACKEND = "imgui_bundle"
except ImportError as e_bundle:
    try:
        import imgui; from imgui.integrations.glfw import GlfwRenderer; import glfw; import OpenGL.GL as gl
        IMGUI_BACKEND = "pyimgui"
    except ImportError as e_pyimgui: IMGUI_BACKEND = None

generate_heightmap = None
try:
    from heightmap_generator import generate_heightmap
    if not callable(generate_heightmap): generate_heightmap = None
except ImportError as e_import: generate_heightmap = None
except Exception as e_other_import:
    generate_heightmap = None

gui_state = {
    "input_dir": r"C:\Program Files (x86)\Steam\steamapps\common\Sacred 2 Gold\pak",
    "output_filename": "heightmap_output.png",
    "apply_boundary_smoothing": True,
    "sector_overlap": 0,
    "boundary_blend_size": 0,
    "height_scale_factor": 1,
    "sector_suffix_type": "B0",
    "log_messages": ["Log messages appear above.", "Select folder, adjust settings, click Generate."],
    "progress": 0.0,
    "processing_thread": None,
    "is_processing": False,
    "log_queue": Queue(),
    "progress_queue": Queue(),
    "heightmap_data": None,
    "log_visible": False,
    "cancel_event": None
}

def select_folder_dialog():
    folder_path = None
    try:
        root = tk.Tk(); root.withdraw(); root.attributes('-topmost', True)
        initial_dir = gui_state.get("input_dir") if os.path.isdir(gui_state.get("input_dir", "")) else None
        folder_path = filedialog.askdirectory(title="Select Folder Containing Sacred 2 Archives (ZIP/PAK)", initialdir=initial_dir)
        root.destroy()
        if folder_path: gui_state["input_dir"] = folder_path
    except Exception as e_dialog:
        gui_state["log_messages"].append(f"Error: Failed to open folder dialog - {e_dialog}")
    return folder_path

def update_log_and_progress():
    global gui_state
    while not gui_state["log_queue"].empty():
        message = gui_state["log_queue"].get_nowait()
        gui_state["log_messages"].append(message)
        max_log_lines = 500
        if len(gui_state["log_messages"]) > max_log_lines: gui_state["log_messages"] = gui_state["log_messages"][-max_log_lines:]

    while not gui_state["progress_queue"].empty():
        progress = gui_state["progress_queue"].get_nowait()
        if progress < 0:
            gui_state["progress"] = 0.0; gui_state["is_processing"] = False
            if gui_state["processing_thread"] and not gui_state["processing_thread"].is_alive(): gui_state["processing_thread"] = None
            if gui_state["cancel_event"]: gui_state["cancel_event"].set()
        else:
            gui_state["progress"] = progress
            if gui_state["progress"] >= 1.0:
                if gui_state["is_processing"]:
                    gui_state["is_processing"] = False
                if gui_state["processing_thread"] and not gui_state["processing_thread"].is_alive():
                    gui_state["processing_thread"] = None
                if gui_state["cancel_event"]: gui_state["cancel_event"].set()

def run_gui_imgui_bundle():
    if not generate_heightmap:
        sys.exit("Error: Core generation logic failed to load.")
        return
    if IMGUI_BACKEND != "imgui_bundle":
        sys.exit("Error: Incorrect ImGui backend detected.")
        return

    Col_ChildBg = getattr(imgui, "Col_ChildBg", 3)
    ChildFlags_Border = getattr(imgui, "ChildFlags_Border", 1)

    def gui_loop():
        global gui_state
        if gui_state["is_processing"]:
            update_log_and_progress()
            if gui_state["processing_thread"] and not gui_state["processing_thread"].is_alive():
                update_log_and_progress()
                if gui_state["progress"] < 1.0 and not (gui_state["cancel_event"] and gui_state["cancel_event"].is_set()):

                    gui_state["log_messages"].append("Warning: Processing thread ended unexpectedly.")

                gui_state["is_processing"] = False; gui_state["processing_thread"] = None; gui_state["progress"] = 0.0

                if gui_state["cancel_event"]: gui_state["cancel_event"].set()

        try:
            settings_window_visible = imgui.begin("Heightmap Generation Settings##HGWindow")
            if settings_window_visible:
                imgui.push_item_width(-250); _, gui_state["input_dir"] = imgui.input_text("##InputFolder", gui_state["input_dir"], 2048); imgui.pop_item_width(); imgui.same_line();
                if imgui.button("Select Input Folder containing ZIP/PAK"): select_folder_dialog()

                imgui.push_item_width(-150); current_base, _ = os.path.splitext(gui_state["output_filename"]); changed_output, new_base = imgui.input_text("Output Base Filename", current_base, 256)
                if changed_output: gui_state["output_filename"] = new_base + ".png" if new_base else "heightmap_output.png"
                imgui.pop_item_width(); imgui.separator()

                imgui.text("Configuration:"); _, gui_state["apply_boundary_smoothing"] = imgui.checkbox("Smooth 7x7 Tile Boundaries Within Sectors", gui_state["apply_boundary_smoothing"]); imgui.separator()

                imgui.text("Sector Type Suffix:");
                if imgui.radio_button("Base Height (_b0.sector)", gui_state["sector_suffix_type"] == "B0"): gui_state["sector_suffix_type"] = "B0"
                imgui.same_line(spacing=10);
                if imgui.radio_button("Ice & Blood? (_d1.sector)", gui_state["sector_suffix_type"] == "D1"): gui_state["sector_suffix_type"] = "D1"
                imgui.same_line(spacing=10);
                if imgui.radio_button("CM Patch? (_d2.sector)", gui_state["sector_suffix_type"] == "d2"): gui_state["sector_suffix_type"] = "d2"
                imgui.separator()

                imgui.text("Stitching & Scaling:"); imgui.push_item_width(120)
                _, gui_state["sector_overlap"] = imgui.input_int("Sector Overlap (pixels)", gui_state["sector_overlap"]); gui_state["sector_overlap"] = max(0, gui_state["sector_overlap"]); imgui.same_line(spacing=10)
                _, gui_state["boundary_blend_size"] = imgui.input_int("Boundary Blend (pixels)", gui_state["boundary_blend_size"]); gui_state["boundary_blend_size"] = max(0, gui_state["boundary_blend_size"]); imgui.same_line(spacing=10)
                _, gui_state["height_scale_factor"] = imgui.input_float("Height Scale Factor", gui_state["height_scale_factor"], 0.1, 1.0, "%.2f");
                if gui_state["height_scale_factor"] <= 1e-6: gui_state["height_scale_factor"] = 1e-6
                imgui.pop_item_width(); imgui.separator()

                use_begin_disabled = hasattr(imgui, "begin_disabled")

                if use_begin_disabled: imgui.begin_disabled(gui_state["is_processing"])
                elif gui_state["is_processing"]: imgui.push_style_var(getattr(imgui, "StyleVar_Alpha", 15), imgui.get_style().alpha * 0.5)

                generate_clicked = imgui.button("Generate Heightmap")

                if use_begin_disabled: imgui.end_disabled()
                elif gui_state["is_processing"]: imgui.pop_style_var()

                imgui.same_line(spacing=10)
                if use_begin_disabled: imgui.begin_disabled(not gui_state["is_processing"])
                elif not gui_state["is_processing"]: imgui.push_style_var(getattr(imgui, "StyleVar_Alpha", 15), imgui.get_style().alpha * 0.5)

                cancel_clicked = imgui.button("Cancel")

                if use_begin_disabled: imgui.end_disabled()
                elif not gui_state["is_processing"]: imgui.pop_style_var()

                if generate_clicked:
                    if not gui_state["is_processing"] and generate_heightmap:
                        gui_state["is_processing"] = True; gui_state["progress"] = 0.0; gui_state["log_messages"] = ["--- Starting New Generation ---"]; gui_state["heightmap_data"] = None
                        while not gui_state["log_queue"].empty(): gui_state["log_queue"].get();
                        while not gui_state["progress_queue"].empty(): gui_state["progress_queue"].get()

                        gui_state["cancel_event"] = threading.Event()
                        gui_state["cancel_event"].clear()

                        current_config = { k: gui_state[k] for k in ['input_dir','output_filename','apply_boundary_smoothing','sector_overlap','boundary_blend_size','height_scale_factor','sector_suffix_type']}
                        try:

                            gui_state["processing_thread"] = threading.Thread(
                                target=generate_heightmap,
                                args=(current_config, gui_state["log_queue"], gui_state["progress_queue"], gui_state["cancel_event"]),
                                daemon=True
                            )
                            gui_state["processing_thread"].start();
                        except Exception as e_thread:
                            gui_state["log_messages"].append(f"Error: Failed to start processing thread - {e_thread}")
                            gui_state["is_processing"] = False; gui_state["progress"] = 0.0
                            gui_state["cancel_event"] = None
                    elif not generate_heightmap:
                        gui_state["log_messages"].append("ERROR: Processing function failed to load. Cannot start generation.")

                if cancel_clicked:
                    if gui_state["is_processing"] and gui_state["cancel_event"]:
                        gui_state["log_messages"].append("--- Cancel Requested by User ---")
                        gui_state["cancel_event"].set()
                        
                imgui.same_line(spacing=10); imgui.progress_bar(gui_state["progress"], size_arg=ImVec2(-1, 0))

            imgui.end()

            viewport = imgui.get_main_viewport()
            vp_pos = viewport.pos; vp_size = viewport.size

            button_bar_h = imgui.get_frame_height_with_spacing()
            log_panel_h = max(60, vp_size.y / 5.0)

            if gui_state["log_visible"]:
                child_draw_y = vp_pos.y + vp_size.y - button_bar_h - log_panel_h
                child_draw_x = vp_pos.x
                child_w = vp_size.x

                imgui.set_next_window_pos(ImVec2(child_draw_x, child_draw_y))
                imgui.set_next_window_size(ImVec2(child_w, log_panel_h))

                imgui.push_style_color(Col_ChildBg, ImVec4(0.1, 0.1, 0.1, 0.9))
                imgui.begin_child("LogScrollingRegion",
                                  child_flags=ChildFlags_Border)

                imgui.text_unformatted("\n".join(gui_state["log_messages"]))
                if gui_state['log_messages']:
                    scroll_buffer = imgui.get_text_line_height() * 2
                    if imgui.get_scroll_y() >= imgui.get_scroll_max_y() - scroll_buffer:
                         imgui.set_scroll_here_y(1.0)

                imgui.end_child()
                imgui.pop_style_color()

            button_bar_y = vp_pos.y + vp_size.y - button_bar_h

            imgui.set_cursor_screen_pos(ImVec2(vp_pos.x, button_bar_y))

            arrow = "v" if gui_state["log_visible"] else ">"
            button_label = f"Log Output {arrow}"

            log_button_clicked = imgui.button(button_label, size=ImVec2(vp_size.x, button_bar_h))

            if log_button_clicked:
                gui_state["log_visible"] = not gui_state["log_visible"]

        except Exception as e_gui:
            try:
                gui_state["log_messages"].append(f"--- GUI ERROR ---")
                gui_state["log_messages"].append(f"{e_gui}")
                gui_state["log_messages"].append(traceback.format_exc())
            except:
                pass

    runner_params = hello_imgui.RunnerParams()
    runner_params.app_window_params.window_title = "Sacred 2 Sector Heightmap Tool"
    runner_params.app_window_params.window_geometry.size = (2048, 1080)

    if hasattr(hello_imgui, 'WindowState') and hasattr(hello_imgui.WindowState, 'maximized'):
        try:
            runner_params.app_window_params.window_state = hello_imgui.WindowState.maximized;
        except Exception as e_max:
             pass

    runner_params.imgui_window_params.show_menu_bar = False
    runner_params.callbacks.show_gui = gui_loop
    try:
        hello_imgui.run(runner_params);
    except Exception as e_runner:
        if gui_state.get("cancel_event"): gui_state["cancel_event"].set()
        sys.exit(f"Fatal Error during GUI startup: {e_runner}")

def run_gui_pyimgui():
    gui_state["log_messages"].append("Error: PyImGui backend selected, but its GUI implementation is currently a basic placeholder.")
    gui_state["log_messages"].append("Please ensure 'imgui-bundle' is installed for the full interface.")
    pass
