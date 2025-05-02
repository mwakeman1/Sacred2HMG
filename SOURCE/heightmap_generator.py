# heightmap_generator.py
import struct
import zipfile
import io
import os
import glob
import math
from PIL import Image
import numpy as np
import traceback

# Queues and threading.Event are imported and handled by gui_manager.py

# --- Constants ---
SECTOR_MAGIC = b'SEC\0' # Not used in current logic, but kept for context
EXPECTED_VERSION = 0x1C # Not used in current logic, but kept for context

# Constants relevant to the data structure and interpretation (from original script)
VERTEX_STRUCT_FORMAT = '<HBBBxxx' # Little-endian: WORD height_int, 3x BYTE normals, 3 pad bytes
VERTEX_STRUCT_SIZE = struct.calcsize(VERTEX_STRUCT_FORMAT) # Should be 8
NUM_VERTICES = 1024 # 32 * 32 grid
GRID_SIZE = 32
CHUNK2_HEADER_FORMAT = '<ff' # Little-endian: float scale, float offset
CHUNK2_HEADER_SIZE = struct.calcsize(CHUNK2_HEADER_FORMAT) # Should be 8

# --- Constants for the marker search logic (adapted from second script) ---
START_MARKER = b'\x00\x00\x01\x00'
VARIABLE_DATA_AFTER_MARKER_SIZE = 4
INTERMEDIATE_HEADER_SIZE = 8 # This size seems specific to the *second* script's interpretation, but we need it to find the potential *start* of vertex data block based on that script's logic
BYTES_TO_SKIP_AFTER_MARKER = VARIABLE_DATA_AFTER_MARKER_SIZE + INTERMEDIATE_HEADER_SIZE # Total bytes between end of marker and start of data block in *second* script logic
N_CONTEXT_CHECK = 5 # How many initial vertex entries to check for validity
# Indices of expected zero padding bytes within the 8-byte VERTEX_STRUCT_FORMAT
EXPECTED_ZERO_BYTES_OFFSETS = [5, 6, 7] # Indices within the 8-byte vertex struct that should be padding (likely zero)

# --- HELPER Function for finding data offset ---
def find_data_block_offset(sector_data, sector_filename, log_queue):
    """
    Searches for the start of the vertex data block using marker and context checks.
    This logic is adapted from the second script.

    Args:
        sector_data (bytes): The binary content of the .sector file.
        sector_filename (str): Original filename for logging.
        log_queue (queue.Queue): Queue for sending log messages back to GUI.

    Returns:
        int: The offset to the start of the vertex data array, or -1 if not found.
    """
    log_prefix = f"  [{os.path.basename(sector_filename)}]"
    file_size = len(sector_data)
    if not sector_data:
        log_queue.put(f"{log_prefix} W: Skipping empty sector content.")
        return -1

    search_start_offset = 0
    found_valid_data_offset = -1

    while search_start_offset < file_size:
        current_marker_offset = sector_data.find(START_MARKER, search_start_offset)
        if current_marker_offset == -1:
            # log_queue.put(f"{log_prefix} DBG: Marker {START_MARKER.hex()} not found after offset {search_start_offset}.")
            break # Marker not found in the rest of the file

        # Calculate where the data block *should* start according to the marker logic
        potential_data_start = current_marker_offset + len(START_MARKER) + BYTES_TO_SKIP_AFTER_MARKER

        # Check if there's enough space for the context check entries
        if potential_data_start + (N_CONTEXT_CHECK * VERTEX_STRUCT_SIZE) <= file_size:
            context_valid = True
            # log_queue.put(f"{log_prefix} DBG: Marker found at {current_marker_offset}. Potential data start {potential_data_start}. Checking context...")
            for i in range(N_CONTEXT_CHECK):
                entry_offset = potential_data_start + (i * VERTEX_STRUCT_SIZE)
                if entry_offset + VERTEX_STRUCT_SIZE > file_size:
                    # log_queue.put(f"{log_prefix} DBG: Context check failed: Not enough data for entry {i} at {entry_offset}.")
                    context_valid = False
                    break # Not enough data left for this entry

                entry_bytes = sector_data[entry_offset : entry_offset + VERTEX_STRUCT_SIZE]

                # Check if padding bytes are zero (basic heuristic)
                if not all(entry_bytes[z] == 0x00 for z in EXPECTED_ZERO_BYTES_OFFSETS):
                    # log_queue.put(f"{log_prefix} DBG: Context check failed: Entry {i} at {entry_offset} failed zero byte check (Bytes: {entry_bytes.hex()}).")
                    context_valid = False
                    break # Found an entry where padding bytes aren't zero

            if context_valid:
                # log_queue.put(f"{log_prefix} DBG: Context check PASSED for {N_CONTEXT_CHECK} entries starting at {potential_data_start}.")
                # Check if this potential start allows enough space for the *full* vertex data AND the preceding header
                required_total_size_from_here = CHUNK2_HEADER_SIZE + (NUM_VERTICES * VERTEX_STRUCT_SIZE)
                potential_header_start = potential_data_start - CHUNK2_HEADER_SIZE
                if potential_header_start >= 0 and potential_header_start + required_total_size_from_here <= file_size:
                    log_queue.put(f"{log_prefix} Valid data block found via marker search. Vertex array starts at: {potential_data_start}")
                    found_valid_data_offset = potential_data_start
                    break # Found a valid offset, stop searching
                else:
                    log_queue.put(f"{log_prefix} W: Marker context valid at {potential_data_start}, but not enough space for header+data (Need {required_total_size_from_here} from {potential_header_start}, File size {file_size}). Continuing search...")
                    # This context was okay, but placement is bad, continue search
            #else: # Context check failed for this marker instance, continue search
                # log_queue.put(f"{log_prefix} DBG: Context check failed for marker at {current_marker_offset}. Continuing search...")

        # Move search start past the current marker to find the next one
        search_start_offset = current_marker_offset + 1

    if found_valid_data_offset == -1:
        log_queue.put(f"{log_prefix} Error: Could not find a valid data block offset using marker search.")

    return found_valid_data_offset


# --- MODIFIED Function ---
def extract_heightmap_from_sector(sector_data, sector_filename, log_queue):
    """
    Processes the raw binary data of a .sector file by dynamically finding the
    data block offset using marker search, then reads the scale/offset header
    preceding it and extracts 32x32 vertex height data.

    Args:
        sector_data (bytes): The binary content of the .sector file.
        sector_filename (str): Original filename for logging and lookup.
        log_queue (queue.Queue): Queue for sending log messages back to GUI.

    Returns:
        tuple: (numpy_array_float32, scale, offset) or (None, None, None) on error.
               The numpy array is 32x32 float32 height data (original scale).
    """
    log_prefix = f"  [{os.path.basename(sector_filename)}]"

    # --- Find the offset dynamically ---
    vertex_data_array_offset = find_data_block_offset(sector_data, sector_filename, log_queue)

    if vertex_data_array_offset == -1:
        # Error already logged by find_data_block_offset
        return None, None, None
    # --- End dynamic offset finding ---

    # Calculate the presumed header offset based on the found vertex data offset
    chunk2_header_offset = vertex_data_array_offset - CHUNK2_HEADER_SIZE

    # --- Basic sanity checks (remain the same) ---
    if chunk2_header_offset < 0:
        log_queue.put(f"{log_prefix} Error: Calculated header offset ({chunk2_header_offset}) is invalid for dynamically found vertex offset {vertex_data_array_offset}.")
        return None, None, None

    required_total_size = CHUNK2_HEADER_SIZE + (NUM_VERTICES * VERTEX_STRUCT_SIZE)
    if chunk2_header_offset + required_total_size > len(sector_data):
          log_queue.put(f"{log_prefix} Error: Header offset {chunk2_header_offset} + required size {required_total_size} exceeds file data length ({len(sector_data)}) for dynamic offset {vertex_data_array_offset}.")
          return None, None, None
    # --- End Sanity Checks ---

    log_queue.put(f"{log_prefix} Reading scale/offset header from calculated offset: {chunk2_header_offset}.")

    try:
        # --- Read Scale/Offset Header (same as before) ---
        scale, offset = struct.unpack_from(CHUNK2_HEADER_FORMAT, sector_data, chunk2_header_offset)
        log_queue.put(f"{log_prefix} Read Scale={scale:.6f}, Offset={offset:.6f}")

        # Handle potential zero scale (same as before)
        if abs(scale) < 1e-9: # Use a small threshold for float comparison
              log_queue.put(f"{log_prefix} Warning: Scale is near zero ({scale}). Heights will likely be uniform {offset}.")
              # Avoid division by zero later if we were using inv_scale, but fine for current formula
              # scale = 1e-9 # Prevent actual zero if needed elsewhere

        height_values_float = []
        # --- Read Vertex Data (using the dynamically found offset) ---
        for i in range(NUM_VERTICES): # 1024 times for 32x32 grid
            current_vertex_offset = vertex_data_array_offset + (i * VERTEX_STRUCT_SIZE)
            # Unpack just the height_int (first 2 bytes, '<H')
            height_int, = struct.unpack_from('<H', sector_data, current_vertex_offset) # Only need the height part

            # --- Apply the loading formula (same as before) ---
            y_float = (float(height_int) * scale) + offset
            height_values_float.append(y_float)

        # Create NumPy array with float32 dtype and reshape (same as before)
        height_map_array = np.array(height_values_float, dtype=np.float32).reshape((GRID_SIZE, GRID_SIZE)) # 32x32

        # Return the float array and the scale/offset read
        return height_map_array, scale, offset

    except struct.error as e:
        log_queue.put(f"{log_prefix} Error unpacking data (Header Offset: {chunk2_header_offset}, Vert Offset: {vertex_data_array_offset}): {e}")
        return None, None, None
    except Exception as e:
        log_queue.put(f"{log_prefix} An unexpected error occurred during processing: {e}")
        # log_queue.put(traceback.format_exc()) # Optional: Add full traceback for debugging
        return None, None, None


# --- Main Worker Function ---
def generate_heightmap(config, log_queue, progress_queue, cancel_event):
    """
    Main worker function. Scans archives, extracts sectors, generates PNGs.
    Handles float heightmaps and directly converts to uint16 for visualization.
    Communicates via queues and checks cancel_event.
    """
    try:
        input_dir_path = config.get('input_dir', '.')
        output_dir_path = config.get('output_dir', '.')
        sector_suffix_type = config.get('sector_suffix_type', 'B0')
        # --- Normalization Setting IS NO LONGER USED ---
        # normalization_mode = config.get('normalization', 'global')
        # log_queue.put(f"Using normalization mode: {normalization_mode}")
        # ---

        # Ensure output directory exists (code unchanged)
        if not os.path.exists(output_dir_path):
             try:
                 os.makedirs(output_dir_path, exist_ok=True)
                 log_queue.put(f"Created output directory: {output_dir_path}")
             except OSError as e:
                 log_queue.put(f"Error: Cannot create output directory '{output_dir_path}': {e}")
                 progress_queue.put(-1.0)
                 return
        elif not os.path.isdir(output_dir_path):
              log_queue.put(f"Error: Output path '{output_dir_path}' exists but is not a directory.")
              progress_queue.put(-1.0)
              return

        # Determine required suffix (code unchanged)
        if sector_suffix_type == "B0": required_suffix = "_b0.sector"
        elif sector_suffix_type == "D1": required_suffix = "_d1.sector"
        elif sector_suffix_type == "D2": required_suffix = "_d2.sector"
        else:
             log_queue.put(f"Warning: Invalid sector_suffix_type '{sector_suffix_type}'. Defaulting to _b0.sector")
             required_suffix = "_b0.sector"
        log_queue.put(f"Searching for sector files ending with: '{required_suffix}' (case-insensitive)")

        # Find archives (code unchanged)
        all_archives = []
        try:
             log_queue.put(f"Scanning input directory: {input_dir_path}")
             abs_input_dir = os.path.abspath(input_dir_path)
             all_archives.extend(glob.glob(os.path.join(abs_input_dir, '*.zip')))
             all_archives.extend(glob.glob(os.path.join(abs_input_dir, '*.pak')))
             all_archives = sorted(list(set(all_archives)))
             log_queue.put(f"Found {len(all_archives)} ZIP/PAK archives.")
        except Exception as e:
             log_queue.put(f"Error searching for archives in '{input_dir_path}': {e}")
             progress_queue.put(-1.0)
             return

        if not all_archives:
             log_queue.put(f"Error: No .zip or .pak archives found in '{input_dir_path}'.")
             progress_queue.put(-1.0)
             return

        total_processed_count = 0
        total_failed_count = 0
        total_archives = len(all_archives)
        progress_scale = 0.95 # Leave some room for final saving

        # --- Data storage ---
        all_height_data = {} # Store {output_png_path: float_array}
        # --- global min/max no longer needed for normalization ---
        # global_min_height = float('inf')
        # global_max_height = float('-inf')
        # ---

        # --- Pass 1: Extract all float data ---
        log_queue.put("\n--- Pass 1: Extracting height data ---")
        for archive_idx, archive_filepath in enumerate(all_archives):
             if cancel_event.is_set():
                 log_queue.put(f"Cancellation requested during Pass 1.")
                 progress_queue.put(-1.0)
                 return

             archive_filename = os.path.basename(archive_filepath)
             log_queue.put(f"\nProcessing archive: {archive_filename} ({archive_idx + 1}/{total_archives})")
             processed_in_archive = 0
             failed_in_archive = 0
             try:
                 with zipfile.ZipFile(archive_filepath, 'r') as zf:
                     file_list = zf.infolist()
                     matching_files = [fi for fi in file_list if fi.filename.lower().endswith(required_suffix.lower()) and not fi.is_dir()]
                     sectors_found_in_archive = len(matching_files)

                     if sectors_found_in_archive == 0:
                         log_queue.put(f"  No sectors matching '{required_suffix}' found.")
                         if total_archives > 0:
                             progress_queue.put(progress_scale * (archive_idx + 1) / total_archives * 0.5) # Halfway through pass 1
                         continue

                     log_queue.put(f"  Found {sectors_found_in_archive} matching sectors. Extracting...")
                     for file_idx, file_info in enumerate(matching_files):
                         if cancel_event.is_set():
                             log_queue.put(f"Cancellation requested while processing {archive_filename}.")
                             progress_queue.put(-1.0)
                             return

                         try:
                             sector_data = zf.read(file_info.filename)
                             # --- Call the extraction function ---
                             height_map_array, scale, offset = extract_heightmap_from_sector(
                                 sector_data, file_info.filename, log_queue
                             )
                             # ---

                             if height_map_array is not None:
                                 # --- Store for Pass 2 ---
                                 base_name = os.path.basename(file_info.filename)
                                 # Simplify suffix removal
                                 if base_name.lower().endswith(required_suffix.lower()):
                                    base_name = base_name[:-len(required_suffix)]
                                 elif base_name.lower().endswith(".sector"): # Fallback just in case
                                    base_name = base_name[:-len(".sector")]

                                 output_png_filename = f"{base_name}_height.png"
                                 output_png_path = os.path.join(output_dir_path, output_png_filename)

                                 all_height_data[output_png_path] = height_map_array

                                 # --- No longer updating global min/max here ---
                                 # if normalization_mode == 'global':
                                 #     min_h = np.min(height_map_array)
                                 #     max_h = np.max(height_map_array)
                                 #     if np.isfinite(min_h) and min_h < global_min_height: global_min_height = min_h
                                 #     if np.isfinite(max_h) and max_h > global_max_height: global_max_height = max_h
                                 # ---
                                 processed_in_archive += 1
                             else:
                                 failed_in_archive += 1

                         except zipfile.BadZipFile as bz:
                             log_queue.put(f"  -> Error reading file {file_info.filename} from archive (corrupted?): {bz}")
                             failed_in_archive +=1
                         except OSError as ose:
                             log_queue.put(f"  -> OS Error processing/saving file {file_info.filename}: {ose}")
                             failed_in_archive += 1
                         except Exception as e:
                             log_queue.put(f"  -> Unexpected error processing file {file_info.filename}: {e}")
                             log_queue.put(traceback.format_exc())
                             failed_in_archive += 1

                         # Update progress within Pass 1
                         if sectors_found_in_archive > 0:
                             current_archive_progress = (file_idx + 1) / sectors_found_in_archive
                             overall_progress = progress_scale * (archive_idx + current_archive_progress) / total_archives * 0.5 # Max 50% in pass 1
                             progress_queue.put(overall_progress)
                         elif total_archives > 0: # Update even if no sectors found in this archive
                             overall_progress = progress_scale * (archive_idx + 1) / total_archives * 0.5
                             progress_queue.put(overall_progress)

             except zipfile.BadZipFile:
                 log_queue.put(f"  Error: Invalid or corrupted archive: {archive_filename}. Skipping.")
                 total_failed_count += 1
             except FileNotFoundError:
                  log_queue.put(f"  Error: Archive file not found during processing loop (was it moved/deleted?): {archive_filename}")
                  total_failed_count += 1
             except PermissionError:
                  log_queue.put(f"  Error: Permission denied reading archive: {archive_filename}. Skipping.")
                  total_failed_count += 1
             except Exception as e:
                 log_queue.put(f"  An unexpected error occurred opening/reading archive {archive_filename}: {e}")
                 log_queue.put(traceback.format_exc())
                 total_failed_count += 1
             finally:
                 log_queue.put(f"  Finished archive extraction. Processed: {processed_in_archive}, Failed/Skipped: {failed_in_archive}")
                 total_processed_count += processed_in_archive
                 total_failed_count += failed_in_archive # Add file failures to total

        if cancel_event.is_set():
              log_queue.put("\n--- Processing Cancelled After Pass 1 ---")
              progress_queue.put(-1.0)
              return

        if not all_height_data:
              log_queue.put("\n--- No height data successfully extracted. Finished ---")
              progress_queue.put(1.0 if total_failed_count == 0 else -1.0) # Success only if no errors at all
              return

        # --- Pass 2: Convert and Save PNGs (NO NORMALIZATION) ---
        log_queue.put(f"\n--- Pass 2: Converting and saving {len(all_height_data)} heightmaps (Direct Conversion) ---")
        log_queue.put("NOTE: Normalization skipped. Float heights converted directly to uint16.")

        num_to_save = len(all_height_data)
        saved_count = 0
        save_failed_count = 0

        for idx, (output_png_path, float_array) in enumerate(all_height_data.items()):
              if cancel_event.is_set():
                  log_queue.put(f"Cancellation requested during Pass 2.")
                  progress_queue.put(-1.0)
                  return

              output_png_filename = os.path.basename(output_png_path)
              try:
                  # --- REMOVED NORMALIZATION BLOCK ---

                  # Handle potential NaNs before direct conversion
                  # Replace NaNs with 0 before casting to uint16
                  float_array_filled = np.nan_to_num(float_array, nan=0.0)

                  # Directly convert float array to uint16
                  # WARNING: This may result in data loss or unexpected visuals
                  # if float values are outside the 0-65535 range or negative.
                  img_array_uint16 = float_array_filled.astype(np.uint16)
                  # --- End Conversion ---

                  img = Image.fromarray(img_array_uint16, mode='I;16')
                  img.save(output_png_path)
                  log_queue.put(f"  -> Saved: {output_png_filename} (Direct Conversion)")
                  saved_count += 1
              except Exception as e:
                  log_queue.put(f"  -> Error saving {output_png_filename}: {e}")
                  log_queue.put(traceback.format_exc())
                  save_failed_count += 1

              # Update progress during Pass 2
              overall_progress = progress_scale * (0.5 + ( (idx + 1) / num_to_save * 0.5) ) # Progress from 50% to 95%
              progress_queue.put(overall_progress)

        total_failed_count += save_failed_count

        # --- Processing Finished ---
        if cancel_event.is_set():
             log_queue.put("\n--- Processing Cancelled During Pass 2 ---")
        else:
             log_queue.put("\n--- Finished processing all archives ---")
             log_queue.put(f"Total Sectors Matching '{required_suffix}' Successfully Processed & Saved: {saved_count}")
             log_queue.put(f"Total Extraction Failures (Marker Search/Read/File): {total_failed_count - save_failed_count}")
             log_queue.put(f"Total Save Failures: {save_failed_count}")
             progress_queue.put(1.0) # Signal normal completion

    except Exception as e:
         log_queue.put("\n--- FATAL ERROR in generate_heightmap ---")
         log_queue.put(f"Error: {e}")
         log_queue.put(traceback.format_exc())
         progress_queue.put(-1.0) # Signal error


# --- Example execution block (if needed for testing) ---
if __name__ == '__main__':
    import queue
    import threading

    # --- Configuration ---
    config = {
        'input_dir': r'C:\Program Files (x86)\Steam\steamapps\common\Sacred 2 Gold\pak', # Example Path
        'output_dir': r'.\heightmap_output_no_norm',  # Example Output Subdirectory
        'sector_suffix_type': 'B0', # Or D1, D2
        # 'normalization': 'global' # REMOVED - No longer used
    }

    # --- Setup Queues and Event ---
    log_queue = queue.Queue()
    progress_queue = queue.Queue()
    cancel_event = threading.Event()

    # --- Log Printing Function ---
    def log_printer(stop_event):
        print("Log Printer Started")
        while not stop_event.is_set():
            try:
                message = log_queue.get(timeout=0.2)
                print(message)
                log_queue.task_done()
            except queue.Empty:
                # If the generator is done and the queue is empty, stop
                if not generator_thread.is_alive() and log_queue.empty():
                    break
            except Exception as e:
                print(f"Log printer error: {e}")
                break
        # Drain any remaining messages quickly after stop signal
        while not log_queue.empty():
             try:
                 print(log_queue.get_nowait())
                 log_queue.task_done()
             except queue.Empty:
                 break
        print("Log Printer Stopped")


    # --- Start Generator ---
    print("Starting generator thread...")
    generator_thread = threading.Thread(target=generate_heightmap, args=(config, log_queue, progress_queue, cancel_event))
    generator_thread.start()

    # --- Start Log Printer ---
    log_stop_event = threading.Event()
    printer_thread = threading.Thread(target=log_printer, args=(log_stop_event,))
    printer_thread.start()

    # --- Monitor Progress ---
    final_progress = 0.0
    while generator_thread.is_alive():
        try:
            # You could add a cancel condition here, e.g., input("Press Enter to cancel...\n")
            # if user_input:
            #    cancel_event.set()
            #    print("Cancellation requested...")

            progress = progress_queue.get(timeout=0.5)
            final_progress = progress # Store last known progress
            if progress == 1.0:
                print("\nProgress: 100% (Complete signal)")
            elif progress < 0:
                 print("\nProgress: Error/Cancelled signal")
                 if not cancel_event.is_set(): # If error signal but not cancelled by user
                     cancel_event.set() # Ensure threads know to stop cleanly
            else:
                 print(f"Progress: {progress*100:.1f}%", end='\r') # Overwrite line
        except queue.Empty:
            pass
        except Exception as e:
             print(f"\nProgress queue error: {e}")
             cancel_event.set() # Signal stop on error
             break
    print() # Newline after progress updates

    # --- Wait for Threads ---
    print("Waiting for generator thread to finish...")
    generator_thread.join(timeout=10) # Add timeout
    if generator_thread.is_alive():
        print("Generator thread timed out.")
        cancel_event.set() # Force stop if stuck

    print("Signaling log printer to stop...")
    log_stop_event.set()
    print("Waiting for log printer thread to finish...")
    printer_thread.join(timeout=5) # Add timeout
    if printer_thread.is_alive():
        print("Log printer thread timed out.")

    print("\n--- Main script finished ---")
    if final_progress == 1.0:
        print("Processing completed successfully.")
    elif cancel_event.is_set() and final_progress >= 0:
        print("Processing was cancelled.")
    else:
        print("Processing finished with errors or was cancelled due to errors.")