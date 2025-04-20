# gui_manager.py
import os
import sys
import time
import traceback
import threading
from queue import Queue, Empty
import tkinter as tk
from tkinter import filedialog

# --- ImGui Backend Check and Import ---
IMGUI_BACKEND = None
try:
    # Use imgui-bundle as primary
    from imgui_bundle import imgui, hello_imgui, ImVec2, ImVec4
    # DO NOT import flags directly from imgui_bundle
    IMGUI_BACKEND = "imgui_bundle"
    print("Using imgui-bundle backend.")
except ImportError as e_bundle:
    IMGUI_BACKEND = None
    print("Error: imgui-bundle not found. This tool requires imgui-bundle.")
    print(f"Import Error: {e_bundle}")
    # No fallback, exit if primary backend is missing

# --- Import Heightmap Generator ---
generate_heightmap = None
try:
    from heightmap_generator import generate_heightmap
    if not callable(generate_heightmap):
        print("Error: 'generate_heightmap' function not found or not callable in heightmap_generator.py.")
        generate_heightmap = None
except ImportError as e_import:
    print(f"Error importing heightmap_generator: {e_import}")
    generate_heightmap = None
except Exception as e_other_import:
    print(f"An unexpected error occurred importing heightmap_generator: {e_other_import}")
    traceback.print_exc()
    generate_heightmap = None

# --- Global GUI State ---
# Use user context for default input if available, otherwise a sensible default
default_input_dir = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__))) # Default to script dir or temp dir if frozen
# Check if user context provided a default input dir
# Note: Accessing 'user_context' directly might cause NameError if not run via specific setup.
# Use a safer check or rely on the hardcoded path / script dir.
sacred_gold_pak_path = r"C:\Program Files (x86)\Steam\steamapps\common\Sacred 2 Gold\pak"
if os.path.exists(sacred_gold_pak_path):
     default_input_dir = sacred_gold_pak_path


gui_state = {
    "input_dir": default_input_dir,
    "output_dir": os.path.join(default_input_dir, "Heightmap_Output"), # Default output to subfolder
    "sector_suffix_type": "B0", # Default to B0
    "log_messages": ["Log messages appear here.", "Select input/output folders, choose sector type, click Generate."],
    "progress": 0.0,
    "processing_thread": None,
    "is_processing": False,
    "log_queue": Queue(),
    "progress_queue": Queue(),
    "cancel_event": None,
    "log_visible": True, # Start with log visible
    "max_log_lines": 1000 # Limit log history
}

# --- Helper Functions ---

def select_folder_dialog(title, initial_dir_key):
    """Opens a folder selection dialog."""
    folder_path = None
    try:
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True) # Try to bring dialog to front
        initial_dir = gui_state.get(initial_dir_key)
        if not initial_dir or not os.path.isdir(initial_dir):
             initial_dir = os.path.expanduser("~") # Fallback to home directory

        folder_path = filedialog.askdirectory(title=title, initialdir=initial_dir)
        root.destroy() # Destroy the hidden Tk window

        if folder_path: # If user selected a folder
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
    """Processes messages from the worker thread queues."""
    global gui_state

    # Process log messages
    while True:
        try:
            message = gui_state["log_queue"].get_nowait()
            gui_state["log_messages"].append(message)
            # Trim log if it gets too long
            if len(gui_state["log_messages"]) > gui_state["max_log_lines"]:
                gui_state["log_messages"] = gui_state["log_messages"][-gui_state["max_log_lines"]:]
        except Empty:
            break # No more messages in queue
        except Exception as e:
            print(f"Error reading log queue: {e}") # Print to console if log append fails
            break

    # Process progress updates
    final_progress_value = None
    while True:
        try:
            progress = gui_state["progress_queue"].get_nowait()
            final_progress_value = progress # Keep track of the last value received this frame
        except Empty:
            break
        except Exception as e:
            print(f"Error reading progress queue: {e}")
            break

    # Update state based on the *last* progress value received this frame
    if final_progress_value is not None:
        if final_progress_value < 0: # Error or cancellation signalled by worker
            gui_state["progress"] = 0.0
            if gui_state["is_processing"]:
                gui_state["log_messages"].append("Processing stopped due to error or cancellation.")
                gui_state["is_processing"] = False
            # Ensure cancel event is set if not already
            if gui_state["cancel_event"] and not gui_state["cancel_event"].is_set():
                 gui_state["cancel_event"].set() # Signal cancellation fully
            # Thread should terminate itself upon seeing the event or error
        elif final_progress_value >= 1.0: # Normal completion
            gui_state["progress"] = 1.0
            if gui_state["is_processing"]:
                 gui_state["log_messages"].append("Processing completed successfully.")
                 gui_state["is_processing"] = False
            # Thread should have finished
        else: # Regular progress update
            gui_state["progress"] = final_progress_value

    # Check thread status after processing queues
    if not gui_state["is_processing"] and gui_state["processing_thread"] is not None:
         # If processing flag is off, but thread object exists, join it cleanly
         # Give it a short timeout in case it's stuck
         gui_state["processing_thread"].join(timeout=0.1)
         if gui_state["processing_thread"].is_alive():
              # This shouldn't normally happen if progress signals worked
              gui_state["log_messages"].append("Warning: Worker thread still alive after processing stop signal.")
         else:
             # print("Debug: Worker thread joined successfully.") # Optional Debug
             gui_state["processing_thread"] = None # Clear the thread object once joined
             gui_state["cancel_event"] = None # Clear the event object

    # Handle unexpected thread termination
    if gui_state["is_processing"] and gui_state["processing_thread"] and not gui_state["processing_thread"].is_alive():
        gui_state["log_messages"].append("Error: Processing thread terminated unexpectedly.")
        gui_state["is_processing"] = False
        gui_state["progress"] = 0.0 # Reset progress
        gui_state["processing_thread"] = None
        gui_state["cancel_event"] = None


# --- Main GUI Function (using imgui-bundle) ---

def gui_loop():
    """The main ImGui drawing loop, called every frame."""
    global gui_state

    # Update log/progress at the start of the frame
    update_log_and_progress()

    try:
        # === Main Settings Window ===
        # Access flags via imgui object
        window_flags = imgui.WindowFlags_.no_collapse | imgui.WindowFlags_.no_saved_settings
        viewport = imgui.get_main_viewport() # Still need viewport for log panel positioning

        # *** REMOVED the following lines to allow moving/resizing the main window ***
        # log_panel_height = 0
        # bottom_bar_height = imgui.get_frame_height_with_spacing() * 1.1
        # if gui_state["log_visible"]:
        #     log_panel_height = max(80, viewport.size.y / 4)
        # main_window_pos = ImVec2(viewport.pos.x, viewport.pos.y)
        # main_window_size = ImVec2(viewport.size.x, viewport.size.y - log_panel_height - bottom_bar_height)
        # imgui.set_next_window_pos(main_window_pos)
        # imgui.set_next_window_size(main_window_size)
        # *** END REMOVED LINES ***

        # Begin the main window without setting pos/size beforehand
        if imgui.begin("Sacred 2 Sector Heightmap Extractor Settings##MainWindow", flags=window_flags):

            # --- Input Directory ---
            imgui.text("Input Directory (containing Sacred 2 .zip/.pak files):")
            imgui.push_item_width(-120) # Adjust width to leave space for button
            changed_input, gui_state["input_dir"] = imgui.input_text("##InputFolder", gui_state["input_dir"], 2048)
            imgui.pop_item_width()
            imgui.same_line()
            if imgui.button("Select##Input", ImVec2(110, 0)):
                select_folder_dialog("Select Input Folder (ZIP/PAK)", "input_dir")

            # --- Output Directory ---
            imgui.text("Output Directory (where PNG heightmaps will be saved):")
            imgui.push_item_width(-120)
            changed_output, gui_state["output_dir"] = imgui.input_text("##OutputFolder", gui_state["output_dir"], 2048)
            imgui.pop_item_width()
            imgui.same_line()
            if imgui.button("Select##Output", ImVec2(110, 0)):
                select_folder_dialog("Select Output Folder", "output_dir")

            imgui.separator()

            # --- Sector Type Selection ---
            imgui.text("Sector Type Suffix to Extract:")
            if imgui.radio_button("_b0.sector (Base Height)", gui_state["sector_suffix_type"] == "B0"):
                 gui_state["sector_suffix_type"] = "B0"
            imgui.same_line(spacing=20)
            if imgui.radio_button("_d1.sector (Ice & Blood?)", gui_state["sector_suffix_type"] == "D1"):
                 gui_state["sector_suffix_type"] = "D1"
            imgui.same_line(spacing=20)
            if imgui.radio_button("_d2.sector (CM Patch?)", gui_state["sector_suffix_type"] == "D2"):
                  gui_state["sector_suffix_type"] = "D2"

            imgui.separator()

            # --- Action Buttons & Progress ---
            button_width = 120

            # Generate Button (disabled during processing)
            imgui.begin_disabled(gui_state["is_processing"])
            generate_clicked = imgui.button("Generate", ImVec2(button_width, 0))
            imgui.end_disabled()

            if generate_clicked and not gui_state["is_processing"]:
                if generate_heightmap: # Check if function is loaded
                    gui_state["is_processing"] = True
                    gui_state["progress"] = 0.0
                    gui_state["log_messages"] = ["--- Starting New Generation ---"] # Clear log
                    # gui_state["heightmap_data"] = None # Removed heightmap_data state

                    # Clear queues before starting
                    while not gui_state["log_queue"].empty(): gui_state["log_queue"].get()
                    while not gui_state["progress_queue"].empty(): gui_state["progress_queue"].get()

                    gui_state["cancel_event"] = threading.Event()
                    gui_state["cancel_event"].clear()

                    # Prepare config for the worker thread
                    current_config = {
                        'input_dir': gui_state['input_dir'],
                        'output_dir': gui_state['output_dir'],
                        'sector_suffix_type': gui_state['sector_suffix_type']
                    }
                    gui_state["log_messages"].append(f"Input: {current_config['input_dir']}")
                    gui_state["log_messages"].append(f"Output: {current_config['output_dir']}")
                    gui_state["log_messages"].append(f"Sector Type: {current_config['sector_suffix_type']}")

                    try:
                        # Create and start the worker thread
                        gui_state["processing_thread"] = threading.Thread(
                            target=generate_heightmap,
                            args=(current_config, gui_state["log_queue"], gui_state["progress_queue"], gui_state["cancel_event"]),
                            daemon=True # Allows app to exit even if thread is running (though cancellation is preferred)
                        )
                        gui_state["processing_thread"].start()
                        gui_state["log_messages"].append("Worker thread started...")
                    except Exception as e_thread:
                        gui_state["log_messages"].append(f"Error: Failed to start processing thread - {e_thread}")
                        gui_state["log_messages"].append(traceback.format_exc())
                        gui_state["is_processing"] = False
                        gui_state["progress"] = 0.0
                        gui_state["cancel_event"] = None # Reset cancel event
                else:
                    gui_state["log_messages"].append("ERROR: Processing function (generate_heightmap) not loaded. Cannot start generation.")

            # Cancel Button (enabled only during processing)
            imgui.same_line(spacing=10)
            imgui.begin_disabled(not gui_state["is_processing"])
            cancel_clicked = imgui.button("Cancel", ImVec2(button_width, 0))
            imgui.end_disabled()

            if cancel_clicked and gui_state["is_processing"]:
                if gui_state["cancel_event"]:
                    gui_state["log_messages"].append("--- User Requested Cancellation ---")
                    gui_state["cancel_event"].set()
                    # Don't immediately set is_processing=False here, wait for thread confirmation via progress queue (-1) or thread join

            # Progress Bar
            imgui.same_line(spacing=10)
            # Calculate remaining width for progress bar
            progress_bar_width = imgui.get_content_region_avail().x
            imgui.progress_bar(gui_state["progress"], size_arg=ImVec2(progress_bar_width, 0))

        # IMPORTANT: End the main window unconditionally after its begin block
        imgui.end()

        # === Log Output Panel (Docked at Bottom) ===
        # Log panel positioning calculations now need viewport info again
        log_panel_height = 0
        bottom_bar_height = imgui.get_frame_height_with_spacing() * 1.1
        if gui_state["log_visible"]:
            log_panel_height = max(80, viewport.size.y / 4)

        # Position the log toggle button bar first
        button_bar_y = viewport.pos.y + viewport.size.y - bottom_bar_height
        imgui.set_next_window_pos(ImVec2(viewport.pos.x, button_bar_y))
        imgui.set_next_window_size(ImVec2(viewport.size.x, bottom_bar_height))
        # Access flags via imgui object
        if imgui.begin("##LogToggleButton", flags=imgui.WindowFlags_.no_decoration | imgui.WindowFlags_.no_saved_settings | imgui.WindowFlags_.no_move | imgui.WindowFlags_.no_bring_to_front_on_focus ):
            arrow = "v" if gui_state["log_visible"] else ">"
            if imgui.button(f"Log Output {arrow}", size=ImVec2(-1, -1)): # Button fills the small window
                gui_state["log_visible"] = not gui_state["log_visible"]
        # IMPORTANT: End the toggle button window unconditionally
        imgui.end()


        # Draw the actual log panel if visible
        if gui_state["log_visible"]:
            log_window_y = viewport.pos.y + viewport.size.y - log_panel_height - bottom_bar_height
            imgui.set_next_window_pos(ImVec2(viewport.pos.x, log_window_y))
            imgui.set_next_window_size(ImVec2(viewport.size.x, log_panel_height))

            # Push style color for background
            # Access flags via imgui object
            imgui.push_style_color(imgui.Col_.child_bg, ImVec4(0.12, 0.12, 0.13, 1.0))
            # Begin the main log window
            # Access flags via imgui object
            if imgui.begin("Log Output##LogWindow", flags=imgui.WindowFlags_.no_collapse | imgui.WindowFlags_.no_saved_settings | imgui.WindowFlags_.horizontal_scrollbar | imgui.WindowFlags_.no_move | imgui.WindowFlags_.no_resize):

                # Create a scrolling child region inside the log window
                # Access flags via imgui object
                child_window_flags = imgui.WindowFlags_.horizontal_scrollbar
                child_flags = imgui.ChildFlags_.border if hasattr(imgui, "ChildFlags_") and hasattr(imgui.ChildFlags_, "border") else 0 # Safer access

                if imgui.begin_child("ScrollingRegion##LogScroll", size=ImVec2(0,0), child_flags=child_flags, window_flags=child_window_flags):

                     # Display log messages
                     imgui.text_unformatted("\n".join(gui_state["log_messages"]))

                     # Auto-scroll logic (if near the bottom, scroll down)
                     # Needs to happen *before* EndChild
                     if imgui.get_scroll_y() >= imgui.get_scroll_max_y() - imgui.get_text_line_height():
                         imgui.set_scroll_here_y(1.0)

                # IMPORTANT: End the child window unconditionally after its begin block
                imgui.end_child()

            # IMPORTANT: End the log window unconditionally after its begin block
            imgui.end()
            imgui.pop_style_color() # Pop style color for child background

    except Exception as e_gui:
        # Log GUI errors to prevent crash, print to console as fallback
        print(f"--- GUI ERROR ---")
        print(f"{e_gui}")
        print(traceback.format_exc())
        try:
            # Attempt to add to GUI log as well
            gui_state["log_messages"].append("--- GUI ERROR ---")
            gui_state["log_messages"].append(f"{e_gui}")
            gui_state["log_messages"].append(traceback.format_exc())
        except:
            pass # Avoid errors within the error handler
        # It's safer *not* to call end() or end_child() here, as the error
        # might have happened *during* a begin call, leaving the stack unbalanced.
        # Let the main loop handler catch the exception.


def run_gui():
    """Sets up and runs the hello_imgui application."""
    if IMGUI_BACKEND != "imgui_bundle":
        print("Error: Cannot run GUI. Required backend 'imgui-bundle' is not available.")
        return
    if not generate_heightmap:
         print("Error: Cannot run GUI. Core 'generate_heightmap' function failed to load.")
         return

    runner_params = hello_imgui.RunnerParams()
    runner_params.app_window_params.window_title = "Sacred 2 Sector Heightmap Extractor"
    runner_params.app_window_params.window_geometry.size = (800, 600) # Reasonable default size
    # runner_params.app_window_params.window_state = hello_imgui.WindowState.maximized # Optional: start maximized

    # Disable the default HelloImGui windows/menus
    runner_params.imgui_window_params.show_menu_bar = False
    runner_params.imgui_window_params.show_status_bar = False

    # CORRECTED LOCATION FOR show_demo_window CHECK
    if hasattr(runner_params, "show_demo_window"):
         runner_params.show_demo_window = False
    elif hasattr(runner_params.imgui_window_params, "show_imgui_demo_window"): # Another possible name
         runner_params.imgui_window_params.show_imgui_demo_window = False
    else:
         print("Warning: Could not find attribute to disable ImGui Demo Window. It might be hidden by default.")


    # Set the main GUI loop function
    runner_params.callbacks.show_gui = gui_loop

    # Add custom shutdown behavior
    def on_exit():
        print("Shutting down GUI...")
        if gui_state.get("is_processing") and gui_state.get("cancel_event"):
            print("Attempting to cancel background process...")
            gui_state["cancel_event"].set()
            if gui_state["processing_thread"]:
                 gui_state["processing_thread"].join(timeout=1.0) # Wait briefly for thread to exit
                 if gui_state["processing_thread"].is_alive():
                     print("Warning: Background thread did not exit cleanly on shutdown.")
        print("GUI Exited.")

    runner_params.callbacks.before_exit = on_exit

    try:
        print("Starting HelloImGui runner...")
        hello_imgui.run(runner_params)
    except Exception as e_runner:
        print(f"\n--- FATAL ERROR during GUI execution ---")
        print(f"Error: {e_runner}")
        print(traceback.format_exc())
        # Attempt cleanup even on fatal error
        on_exit()
        # Don't call sys.exit here if run from main.py, let main handle it
        raise # Reraise the exception so main.py can catch it if needed


# This check prevents running the GUI if the script is imported elsewhere
# if __name__ == "__main__":
#     # main.py should handle calling run_gui()
#     print("gui_manager.py should be run via main.py")
#     pass