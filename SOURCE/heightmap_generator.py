import struct
import zipfile
import io
import os
import glob
import math
from PIL import Image
import numpy as np
import traceback

SECTOR_MAGIC = b'SEC\0'
EXPECTED_VERSION = 0x1C

VERTEX_STRUCT_FORMAT = '<HBBBxxx'
VERTEX_STRUCT_SIZE = struct.calcsize(VERTEX_STRUCT_FORMAT)
NUM_VERTICES = 1024
GRID_SIZE = 32
CHUNK2_HEADER_FORMAT = '<ff'
CHUNK2_HEADER_SIZE = struct.calcsize(CHUNK2_HEADER_FORMAT)

START_MARKER = b'\x00\x00\x01\x00'
VARIABLE_DATA_AFTER_MARKER_SIZE = 4
INTERMEDIATE_HEADER_SIZE = 8
BYTES_TO_SKIP_AFTER_MARKER = VARIABLE_DATA_AFTER_MARKER_SIZE + INTERMEDIATE_HEADER_SIZE
N_CONTEXT_CHECK = 5
EXPECTED_ZERO_BYTES_OFFSETS = [5, 6, 7]

def find_data_block_offset(sector_data, sector_filename, log_queue):
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
            break

        potential_data_start = current_marker_offset + len(START_MARKER) + BYTES_TO_SKIP_AFTER_MARKER

        if potential_data_start + (N_CONTEXT_CHECK * VERTEX_STRUCT_SIZE) <= file_size:
            context_valid = True
            for i in range(N_CONTEXT_CHECK):
                entry_offset = potential_data_start + (i * VERTEX_STRUCT_SIZE)
                if entry_offset + VERTEX_STRUCT_SIZE > file_size:
                    context_valid = False
                    break

                entry_bytes = sector_data[entry_offset : entry_offset + VERTEX_STRUCT_SIZE]

                if not all(entry_bytes[z] == 0x00 for z in EXPECTED_ZERO_BYTES_OFFSETS):
                    context_valid = False
                    break

            if context_valid:
                required_total_size_from_here = CHUNK2_HEADER_SIZE + (NUM_VERTICES * VERTEX_STRUCT_SIZE)
                potential_header_start = potential_data_start - CHUNK2_HEADER_SIZE
                if potential_header_start >= 0 and potential_header_start + required_total_size_from_here <= file_size:
                    log_queue.put(f"{log_prefix} Valid data block found via marker search. Vertex array starts at: {potential_data_start}")
                    found_valid_data_offset = potential_data_start
                    break
                else:
                    log_queue.put(f"{log_prefix} W: Marker context valid at {potential_data_start}, but not enough space for header+data (Need {required_total_size_from_here} from {potential_header_start}, File size {file_size}). Continuing search...")

        search_start_offset = current_marker_offset + 1

    if found_valid_data_offset == -1:
        log_queue.put(f"{log_prefix} Error: Could not find a valid data block offset using marker search.")

    return found_valid_data_offset

def extract_heightmap_from_sector(sector_data, sector_filename, log_queue):
    log_prefix = f"  [{os.path.basename(sector_filename)}]"

    vertex_data_array_offset = find_data_block_offset(sector_data, sector_filename, log_queue)

    if vertex_data_array_offset == -1:
        return None, None, None

    chunk2_header_offset = vertex_data_array_offset - CHUNK2_HEADER_SIZE

    if chunk2_header_offset < 0:
        log_queue.put(f"{log_prefix} Error: Calculated header offset ({chunk2_header_offset}) is invalid for dynamically found vertex offset {vertex_data_array_offset}.")
        return None, None, None

    required_total_size = CHUNK2_HEADER_SIZE + (NUM_VERTICES * VERTEX_STRUCT_SIZE)
    if chunk2_header_offset + required_total_size > len(sector_data):
          log_queue.put(f"{log_prefix} Error: Header offset {chunk2_header_offset} + required size {required_total_size} exceeds file data length ({len(sector_data)}) for dynamic offset {vertex_data_array_offset}.")
          return None, None, None

    log_queue.put(f"{log_prefix} Reading scale/offset header from calculated offset: {chunk2_header_offset}.")

    try:
        scale, offset = struct.unpack_from(CHUNK2_HEADER_FORMAT, sector_data, chunk2_header_offset)
        log_queue.put(f"{log_prefix} Read Scale={scale:.6f}, Offset={offset:.6f}")

        if abs(scale) < 1e-9:
              log_queue.put(f"{log_prefix} Warning: Scale is near zero ({scale}). Heights will likely be uniform {offset}.")

        height_values_float = []
        for i in range(NUM_VERTICES):
            current_vertex_offset = vertex_data_array_offset + (i * VERTEX_STRUCT_SIZE)
            height_int, = struct.unpack_from('<H', sector_data, current_vertex_offset)

            y_float = (float(height_int) * scale) + offset
            height_values_float.append(y_float)

        height_map_array = np.array(height_values_float, dtype=np.float32).reshape((GRID_SIZE, GRID_SIZE))
        height_map_array = np.fliplr(height_map_array)
        log_queue.put(f"{log_prefix} Horizontally flipped the heightmap.")

        return height_map_array, scale, offset

    except struct.error as e:
        log_queue.put(f"{log_prefix} Error unpacking data (Header Offset: {chunk2_header_offset}, Vert Offset: {vertex_data_array_offset}): {e}")
        return None, None, None
    except Exception as e:
        log_queue.put(f"{log_prefix} An unexpected error occurred during processing: {e}")
        return None, None, None

def generate_heightmap(config, log_queue, progress_queue, cancel_event):
    try:
        input_dir_path = config.get('input_dir', '.')
        output_dir_path = config.get('output_dir', '.')
        sector_suffix_type = config.get('sector_suffix_type', 'B0')

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

        if sector_suffix_type == "B0": required_suffix = "_b0.sector"
        elif sector_suffix_type == "D1": required_suffix = "_d1.sector"
        elif sector_suffix_type == "D2": required_suffix = "_d2.sector"
        else:
             log_queue.put(f"Warning: Invalid sector_suffix_type '{sector_suffix_type}'. Defaulting to _b0.sector")
             required_suffix = "_b0.sector"
        log_queue.put(f"Searching for sector files ending with: '{required_suffix}' (case-insensitive)")

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
        progress_scale = 0.95

        all_height_data = {}

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
                             progress_queue.put(progress_scale * (archive_idx + 1) / total_archives * 0.5)
                         continue

                     log_queue.put(f"  Found {sectors_found_in_archive} matching sectors. Extracting...")
                     for file_idx, file_info in enumerate(matching_files):
                         if cancel_event.is_set():
                             log_queue.put(f"Cancellation requested while processing {archive_filename}.")
                             progress_queue.put(-1.0)
                             return

                         try:
                             sector_data = zf.read(file_info.filename)
                             height_map_array, scale, offset = extract_heightmap_from_sector(
                                 sector_data, file_info.filename, log_queue
                             )

                             if height_map_array is not None:
                                 base_name = os.path.basename(file_info.filename)
                                 if base_name.lower().endswith(required_suffix.lower()):
                                    base_name = base_name[:-len(required_suffix)]
                                 elif base_name.lower().endswith(".sector"):
                                    base_name = base_name[:-len(".sector")]

                                 output_png_filename = f"{base_name}_height.png"
                                 output_png_path = os.path.join(output_dir_path, output_png_filename)

                                 all_height_data[output_png_path] = height_map_array
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

                         if sectors_found_in_archive > 0:
                             current_archive_progress = (file_idx + 1) / sectors_found_in_archive
                             overall_progress = progress_scale * (archive_idx + current_archive_progress) / total_archives * 0.5
                             progress_queue.put(overall_progress)
                         elif total_archives > 0:
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
                 total_failed_count += failed_in_archive

        if cancel_event.is_set():
              log_queue.put("\n--- Processing Cancelled After Pass 1 ---")
              progress_queue.put(-1.0)
              return

        if not all_height_data:
              log_queue.put("\n--- No height data successfully extracted. Finished ---")
              progress_queue.put(1.0 if total_failed_count == 0 else -1.0)
              return

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
                  float_array_filled = np.nan_to_num(float_array, nan=0.0)
                  img_array_uint16 = float_array_filled.astype(np.uint16)
                  img = Image.fromarray(img_array_uint16, mode='I;16')
                  img.save(output_png_path)
                  log_queue.put(f"  -> Saved: {output_png_filename} (Direct Conversion)")
                  saved_count += 1
              except Exception as e:
                  log_queue.put(f"  -> Error saving {output_png_filename}: {e}")
                  log_queue.put(traceback.format_exc())
                  save_failed_count += 1

              overall_progress = progress_scale * (0.5 + ( (idx + 1) / num_to_save * 0.5) )
              progress_queue.put(overall_progress)

        total_failed_count += save_failed_count

        if cancel_event.is_set():
             log_queue.put("\n--- Processing Cancelled During Pass 2 ---")
        else:
             log_queue.put("\n--- Finished processing all archives ---")
             log_queue.put(f"Total Sectors Matching '{required_suffix}' Successfully Processed & Saved: {saved_count}")
             log_queue.put(f"Total Extraction Failures (Marker Search/Read/File): {total_failed_count - save_failed_count}")
             log_queue.put(f"Total Save Failures: {save_failed_count}")
             progress_queue.put(1.0)

    except Exception as e:
         log_queue.put("\n--- FATAL ERROR in generate_heightmap ---")
         log_queue.put(f"Error: {e}")
         log_queue.put(traceback.format_exc())
         progress_queue.put(-1.0)

if __name__ == '__main__':
    import queue
    import threading

    config = {
        'input_dir': r'C:\Program Files (x86)\Steam\steamapps\common\Sacred 2 Gold\pak',
        'output_dir': r'.\heightmap_output_no_norm_flipped',
        'sector_suffix_type': 'B0',
    }

    log_queue = queue.Queue()
    progress_queue = queue.Queue()
    cancel_event = threading.Event()

    def log_printer(stop_event):
        print("Log Printer Started")
        while not stop_event.is_set():
            try:
                message = log_queue.get(timeout=0.2)
                print(message)
                log_queue.task_done()
            except queue.Empty:
                if not generator_thread.is_alive() and log_queue.empty():
                    break
            except Exception as e:
                print(f"Log printer error: {e}")
                break
        while not log_queue.empty():
             try:
                 print(log_queue.get_nowait())
                 log_queue.task_done()
             except queue.Empty:
                 break
        print("Log Printer Stopped")

    print("Starting generator thread...")
    generator_thread = threading.Thread(target=generate_heightmap, args=(config, log_queue, progress_queue, cancel_event))
    generator_thread.start()

    log_stop_event = threading.Event()
    printer_thread = threading.Thread(target=log_printer, args=(log_stop_event,))
    printer_thread.start()

    final_progress = 0.0
    while generator_thread.is_alive():
        try:
            progress = progress_queue.get(timeout=0.5)
            final_progress = progress
            if progress == 1.0:
                print("\nProgress: 100% (Complete signal)")
            elif progress < 0:
                 print("\nProgress: Error/Cancelled signal")
                 if not cancel_event.is_set():
                     cancel_event.set()
            else:
                 print(f"Progress: {progress*100:.1f}%", end='\r')
        except queue.Empty:
            pass
        except Exception as e:
             print(f"\nProgress queue error: {e}")
             cancel_event.set()
             break
    print()

    print("Waiting for generator thread to finish...")
    generator_thread.join(timeout=10)
    if generator_thread.is_alive():
        print("Generator thread timed out.")
        cancel_event.set()

    print("Signaling log printer to stop...")
    log_stop_event.set()
    print("Waiting for log printer thread to finish...")
    printer_thread.join(timeout=5)
    if printer_thread.is_alive():
        print("Log printer thread timed out.")

    print("\n--- Main script finished ---")
    if final_progress == 1.0:
        print("Processing completed successfully.")
    elif cancel_event.is_set() and final_progress >= 0:
        print("Processing was cancelled.")
    else:
        print("Processing finished with errors or was cancelled due to errors.")