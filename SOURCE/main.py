# main.py
import sys
import os

# --- Check for imgui-bundle ---
try:
    # Try importing necessary parts to confirm installation
    from imgui_bundle import hello_imgui, imgui
except ImportError:
    print("Error: imgui-bundle is required but not found.")
    print("Please install it, for example using: pip install imgui-bundle")
    input("Press Enter to exit...") # Keep console open
    sys.exit(1)

# --- Import and run the GUI Manager ---
try:
    # Ensure the script directory is in the path for local imports
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)

    import gui_manager
    # Check if the core generator function loaded correctly within the gui_manager
    if gui_manager.generate_heightmap is None:
         # Specific error message already printed by gui_manager
         input("Press Enter to exit...") # Keep console open
         sys.exit(1)

    # Run the GUI using the appropriate backend function determined in gui_manager
    gui_manager.run_gui()

except ImportError as e:
    print(f"Error importing gui_manager or its dependencies: {e}")
    print("Ensure gui_manager.py and heightmap_generator.py are in the same directory.")
    input("Press Enter to exit...")
    sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred during startup: {e}")
    import traceback
    traceback.print_exc()
    input("Press Enter to exit...")
    sys.exit(1)