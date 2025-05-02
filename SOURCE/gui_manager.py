# gui_manager.py
import os
import sys
import time
import traceback
import threading
from queue import Queue, Empty
import tkinter as tk
from tkinter import filedialog

IMGUI_BACKEND = None
try:
    from imgui_bundle import imgui, hello_imgui, ImVec2, ImVec4
    IMGUI_BACKEND = "imgui_bundle"
except ImportError as e_bundle:
    IMGUI_BACKEND = None

generate_heightmap = None
try:
    from heightmap_generator import generate_heightmap
    if not callable(generate_heightmap):
        generate_heightmap = None
except ImportError as e_import:
    generate_heightmap = None
except Exception as e_other_import:
    traceback.print_exc()
    generate_heightmap = None

default_input_dir = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
sacred_gold_pak_path = r"C:\Program Files (x86)\Steam\steamapps\common\Sacred 2 Gold\pak"
if os.path.exists(sacred_gold_pak_path):
     default_input_dir = sacred_gold_pak_path

gui_state = {
    "input_dir": default_input_dir,
    "output_dir": os.path.join(default_input_dir, "Heightmap_Output"),
    "sector_suffix_type": "B0",
    "log_messages": ["Log messages appear here.", "Select input/output folders, choose sector type, click Generate."],
    "progress": 0.0,
    "processing_thread": None,
    "is_processing": False,
    "log_queue": Queue(),
    "progress_queue": Queue(),
    "cancel_event": None,
    "log_visible": True,
    "max_log_lines": 1000,
    "heightmap_extractor_visible": True, # State for main tool window
    "toolbar_visible": True # State now controls if toolbar attempts to draw AT ALL
                           # Collapse state is handled internally by ImGui for the window
}

def select_folder_dialog(title, initial_dir_key):
    folder_path = None
    try:
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        initial_dir = gui_state.get(initial_dir_key)
        if not initial_dir or not os.path.isdir(initial_dir):
             initial_dir = os.path.expanduser("~")

        folder_path = filedialog.askdirectory(title=title, initialdir=initial_dir)
        root.destroy()

        if folder_path:
            gui_state[initial_dir_key] = folder_path
            gui_state["log_messages"].append(f"Set {initial_dir_key.replace('_',' ')} to: {folder_path}")
    except tk.TclError as e_tcl:
         gui_state["log_messages"].append(f"Warning: Folder dialog failed (Tkinter TclError): {e_tcl}")
         gui_state["log_messages"].append("  (This might happen on some systems/environments)")
    except Exception as e_dialog:
        gui_state["log_messages"].append(f"Error: Failed to open folder dialog - {e_dialog}")
        gui_state["log_messages"].append(traceback.format_exc())
    return folder_path

def update_log_and_progress():
    global gui_state

    while True:
        try:
            message = gui_state["log_queue"].get_nowait()
            gui_state["log_messages"].append(message)
            if len(gui_state["log_messages"]) > gui_state["max_log_lines"]:
                gui_state["log_messages"] = gui_state["log_messages"][-gui_state["max_log_lines"]:]
        except Empty:
            break
        except Exception as e:
            # Error reading log queue - do nothing in GUI, maybe log elsewhere if needed
            break

    final_progress_value = None
    while True:
        try:
            progress = gui_state["progress_queue"].get_nowait()
            final_progress_value = progress
        except Empty:
            break
        except Exception as e:
             # Error reading progress queue - do nothing in GUI
            break

    if final_progress_value is not None:
        if final_progress_value < 0:
            gui_state["progress"] = 0.0
            if gui_state["is_processing"]:
                gui_state["log_messages"].append("Processing stopped due to error or cancellation.")
                gui_state["is_processing"] = False
            if gui_state["cancel_event"] and not gui_state["cancel_event"].is_set():
                 gui_state["cancel_event"].set()
        elif final_progress_value >= 1.0:
            gui_state["progress"] = 1.0
            if gui_state["is_processing"]:
                 gui_state["log_messages"].append("Processing completed successfully.")
                 gui_state["is_processing"] = False
        else:
            gui_state["progress"] = final_progress_value

    if not gui_state["is_processing"] and gui_state["processing_thread"] is not None:
         gui_state["processing_thread"].join(timeout=0.1)
         if not gui_state["processing_thread"].is_alive():
             gui_state["processing_thread"] = None
             gui_state["cancel_event"] = None

    if gui_state["is_processing"] and gui_state["processing_thread"] and not gui_state["processing_thread"].is_alive():
        gui_state["log_messages"].append("Error: Processing thread terminated unexpectedly.")
        gui_state["is_processing"] = False
        gui_state["progress"] = 0.0
        gui_state["processing_thread"] = None
        gui_state["cancel_event"] = None

def gui_loop():
    global gui_state
    update_log_and_progress()

    try:
        viewport = imgui.get_main_viewport()

        # === Calculate height occupied by bottom elements (log panel + log toggle button) ===
        button_bar_height = imgui.get_frame_height_with_spacing()
        log_content_height = 0
        bottom_elements_total_height = button_bar_height # Button bar is always there
        if gui_state["log_visible"]:
            log_content_height = max(80, viewport.size.y / 4) # Calculate log content height
            bottom_elements_total_height += log_content_height # Add log content height if visible

        # === REMOVED Left Toolbar Toggle Button ===

        # === Left Toolbar Window (Collapsible via title bar, no move, no close) ===
        # We still use toolbar_visible to potentially hide it entirely if needed later
        # but collapsing is now handled internally by the window itself.
        if gui_state["toolbar_visible"]:
            toolbar_width = 200
            toolbar_height = viewport.size.y - bottom_elements_total_height

            imgui.set_next_window_pos(ImVec2(viewport.pos.x, viewport.pos.y), imgui.Cond_.always)
            imgui.set_next_window_size(ImVec2(toolbar_width, toolbar_height), imgui.Cond_.always)

            # Flags: No resize/move. Allow collapse via title bar triangle. No explicit close button.
            toolbar_flags = (
                  imgui.WindowFlags_.no_resize
                | imgui.WindowFlags_.no_move
                # | imgui.WindowFlags_.no_collapse # REMOVED
                | imgui.WindowFlags_.no_saved_settings
                # | imgui.WindowFlags_.no_title_bar # REMOVED
            )

            # Call begin WITHOUT the visibility bool to disable the 'X' button
            # The return value tells us if the window is open *and not collapsed*
            is_toolbar_content_visible = imgui.begin(
                "Tools##LeftToolbar", # Keep Title "Tools"
                flags=toolbar_flags
            )

            # Only draw content if the window is not collapsed
            if is_toolbar_content_visible:
                # Center button: Calculate offset based on fixed button size and toolbar width
                button_content_width = toolbar_width - 15 # Width used for the button
                cursor_x_centered = (toolbar_width - button_content_width) * 0.5
                # Set horizontal position for the button
                imgui.set_cursor_pos_x(cursor_x_centered)

                if imgui.button("Heightmap Extractor", size=ImVec2(button_content_width, 0)):
                    gui_state["heightmap_extractor_visible"] = True

                imgui.separator() # Keep separator below button

            # End the toolbar window unconditionally
            imgui.end()


        # === Main Settings Window ===
        if gui_state["heightmap_extractor_visible"]:
            window_flags = imgui.WindowFlags_.no_saved_settings

            imgui.set_next_window_size(ImVec2(650, 235), imgui.Cond_.first_use_ever)

            was_visible, gui_state["heightmap_extractor_visible"] = imgui.begin(
                "Heightmap Extractor##MainWindow",
                gui_state["heightmap_extractor_visible"], # Pass visibility bool to allow 'X' button
                flags=window_flags
            )

            if was_visible:
                imgui.text("Input Directory (containing Sacred 2 .zip/.pak files):")
                imgui.push_item_width(-120)
                changed_input, gui_state["input_dir"] = imgui.input_text("##InputFolder", gui_state["input_dir"], 2048)
                imgui.pop_item_width()
                imgui.same_line()
                if imgui.button("Select##Input", ImVec2(110, 0)):
                    select_folder_dialog("Select Input Folder (ZIP/PAK)", "input_dir")

                imgui.text("Output Directory (where PNG heightmaps will be saved):")
                imgui.push_item_width(-120)
                changed_output, gui_state["output_dir"] = imgui.input_text("##OutputFolder", gui_state["output_dir"], 2048)
                imgui.pop_item_width()
                imgui.same_line()
                if imgui.button("Select##Output", ImVec2(110, 0)):
                    select_folder_dialog("Select Output Folder", "output_dir")

                imgui.separator()

                imgui.text("Sector Type Suffix to Extract:")
                if imgui.radio_button("_b0.sector (Overworld)", gui_state["sector_suffix_type"] == "B0"):
                    gui_state["sector_suffix_type"] = "B0"
                imgui.same_line(spacing=20)
                if imgui.radio_button("_d1.sector (Dungeon 1)", gui_state["sector_suffix_type"] == "D1"):
                    gui_state["sector_suffix_type"] = "D1"
                imgui.same_line(spacing=20)
                if imgui.radio_button("_d2.sector (Dungeon 2)", gui_state["sector_suffix_type"] == "D2"):
                    gui_state["sector_suffix_type"] = "D2"

                imgui.separator()

                button_width = 120

                imgui.begin_disabled(gui_state["is_processing"])
                generate_clicked = imgui.button("Generate", ImVec2(button_width, 0))
                imgui.end_disabled()

                if generate_clicked and not gui_state["is_processing"]:
                    if generate_heightmap:
                        gui_state["is_processing"] = True
                        gui_state["progress"] = 0.0
                        gui_state["log_messages"] = ["--- Starting New Generation ---"]

                        while not gui_state["log_queue"].empty(): gui_state["log_queue"].get()
                        while not gui_state["progress_queue"].empty(): gui_state["progress_queue"].get()

                        gui_state["cancel_event"] = threading.Event()
                        gui_state["cancel_event"].clear()

                        current_config = {
                            'input_dir': gui_state['input_dir'],
                            'output_dir': gui_state['output_dir'],
                            'sector_suffix_type': gui_state['sector_suffix_type']
                        }
                        gui_state["log_messages"].append(f"Input: {current_config['input_dir']}")
                        gui_state["log_messages"].append(f"Output: {current_config['output_dir']}")
                        gui_state["log_messages"].append(f"Sector Type: {current_config['sector_suffix_type']}")

                        try:
                            gui_state["processing_thread"] = threading.Thread(
                                target=generate_heightmap,
                                args=(current_config, gui_state["log_queue"], gui_state["progress_queue"], gui_state["cancel_event"]),
                                daemon=True
                            )
                            gui_state["processing_thread"].start()
                            gui_state["log_messages"].append("Worker thread started...")
                        except Exception as e_thread:
                            gui_state["log_messages"].append(f"Error: Failed to start processing thread - {e_thread}")
                            gui_state["log_messages"].append(traceback.format_exc())
                            gui_state["is_processing"] = False
                            gui_state["progress"] = 0.0
                            gui_state["cancel_event"] = None
                    else:
                        gui_state["log_messages"].append("ERROR: Processing function (generate_heightmap) not loaded. Cannot start generation.")

                imgui.same_line(spacing=10)
                imgui.begin_disabled(not gui_state["is_processing"])
                cancel_clicked = imgui.button("Cancel", ImVec2(button_width, 0))
                imgui.end_disabled()

                if cancel_clicked and gui_state["is_processing"]:
                    if gui_state["cancel_event"]:
                        gui_state["log_messages"].append("--- User Requested Cancellation ---")
                        gui_state["cancel_event"].set()

                imgui.same_line(spacing=10)
                progress_bar_width = imgui.get_content_region_avail().x
                imgui.progress_bar(gui_state["progress"], size_arg=ImVec2(progress_bar_width, 0))

            imgui.end()


        # === Log Output Section (Collapsible) ===
        # Note: button_bar_height and log_content_height calculated earlier

        if gui_state["log_visible"]:
            # Position log panel content window *above* the button bar space
            log_window_y = viewport.pos.y + viewport.size.y - log_content_height - button_bar_height
            imgui.set_next_window_pos(ImVec2(viewport.pos.x, log_window_y))
            # Make log panel span full width regardless of toolbar
            imgui.set_next_window_size(ImVec2(viewport.size.x, log_content_height))

            imgui.push_style_color(imgui.Col_.child_bg, ImVec4(0.12, 0.12, 0.13, 1.0))
            log_window_flags = (
                  imgui.WindowFlags_.no_collapse
                | imgui.WindowFlags_.no_saved_settings
                | imgui.WindowFlags_.horizontal_scrollbar
                | imgui.WindowFlags_.no_move
                | imgui.WindowFlags_.no_resize
                | imgui.WindowFlags_.no_title_bar
            )
            if imgui.begin("Log Output##LogWindow", flags=log_window_flags):
                child_window_flags = imgui.WindowFlags_.horizontal_scrollbar
                child_flags = imgui.ChildFlags_.border if hasattr(imgui, "ChildFlags_") and hasattr(imgui.ChildFlags_, "border") else 0

                if imgui.begin_child("ScrollingRegion##LogScroll", size=ImVec2(0,0), child_flags=child_flags, window_flags=child_window_flags):
                     imgui.text_unformatted("\n".join(gui_state["log_messages"]))
                     if imgui.get_scroll_y() >= imgui.get_scroll_max_y() - imgui.get_text_line_height():
                         imgui.set_scroll_here_y(1.0)
                imgui.end_child()
            imgui.end()
            imgui.pop_style_color()

        # === Draw Log Toggle Button Bar ===
        # Position button bar at the very bottom, span full width
        button_bar_y = viewport.pos.y + viewport.size.y - button_bar_height
        imgui.set_cursor_screen_pos(ImVec2(viewport.pos.x, button_bar_y))
        arrow = "v" if gui_state["log_visible"] else ">"
        if imgui.button(f"Log Output {arrow}", size=ImVec2(viewport.size.x, button_bar_height)):
            gui_state["log_visible"] = not gui_state["log_visible"]

    except Exception as e_gui:
        try:
            gui_state["log_messages"].append("--- GUI ERROR ---")
            gui_state["log_messages"].append(f"{e_gui}")
            gui_state["log_messages"].append(traceback.format_exc())
        except:
            pass

def run_gui():
    if IMGUI_BACKEND != "imgui_bundle":
        return
    if not generate_heightmap:
         return

    runner_params = hello_imgui.RunnerParams()
    runner_params.app_window_params.window_title = "Sacred 2 Sector Heightmap Extractor"
    runner_params.app_window_params.window_geometry.size = (1280, 720)

    runner_params.imgui_window_params.show_menu_bar = False
    runner_params.imgui_window_params.show_status_bar = False

    if hasattr(runner_params, "show_demo_window"):
         runner_params.show_demo_window = False
    elif hasattr(runner_params.imgui_window_params, "show_imgui_demo_window"):
         runner_params.imgui_window_params.show_imgui_demo_window = False
    else:
         pass

    runner_params.callbacks.show_gui = gui_loop

    def on_exit():
        if gui_state.get("is_processing") and gui_state.get("cancel_event"):
            gui_state["cancel_event"].set()
            if gui_state["processing_thread"]:
                 gui_state["processing_thread"].join(timeout=1.0)

    runner_params.callbacks.before_exit = on_exit

    try:
        hello_imgui.run(runner_params)
    except Exception as e_runner:
        on_exit()
        raise

# if __name__ == "__main__":
#     pass