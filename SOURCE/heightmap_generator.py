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
SECTOR_MAGIC = b'SEC\0'
EXPECTED_VERSION = 0x1C # Decimal 28
EXPECTED_NUM_CHUNKS = 8
TERRAIN_VERTEX_CHUNK_ID = 2 # Target chunk ID for terrain data
EXPECTED_CHUNK_2_SIZE = 8200 # 8 byte header + 1024 * 8 byte vertices
VERTEX_STRUCT_FORMAT = '<HBBBxxx' # Little-endian: WORD height, 3x BYTE normals, 3 pad bytes
VERTEX_STRUCT_SIZE = struct.calcsize(VERTEX_STRUCT_FORMAT) # Should be 8
HEADER_STRUCT_FORMAT = '<4sII' # Magic (4s), Version (I), NumChunks (I)
HEADER_SIZE = struct.calcsize(HEADER_STRUCT_FORMAT) # Should be 12
CHUNK_DESC_FORMAT = '<IIII' # ID (I), Flags? (I), Offset (I), Size (I)
CHUNK_DESC_SIZE = struct.calcsize(CHUNK_DESC_FORMAT)
CHUNK_TABLE_SIZE = EXPECTED_NUM_CHUNKS * CHUNK_DESC_SIZE # Should be 128
CHUNK_2_HEADER_FORMAT = '<ff' # Presumed: float scale, float offset
CHUNK_2_HEADER_SIZE = struct.calcsize(CHUNK_2_HEADER_FORMAT) # Should be 8
NUM_VERTICES = 1024
GRID_SIZE = 32 # 32x32 vertices

def extract_heightmap_from_sector(sector_data, sector_filename, log_queue):
    """
    Processes the raw binary data of a .sector file based on RE'd structure.
    Uses log_queue for output.

    Args:
        sector_data (bytes): The binary content of the .sector file.
        sector_filename (str): Original filename for logging.
        log_queue (queue.Queue): Queue for sending log messages back to GUI.

    Returns:
        tuple: (numpy_array_uint16, scale, offset) or (None, None, None) on error.
               The numpy array is 32x32 uint16 height data.
               Scale and offset are floats read from the chunk header.
    """
    log_prefix = f"  [{os.path.basename(sector_filename)}]" # Use for specific file logs
    try:
        if len(sector_data) < HEADER_SIZE + CHUNK_TABLE_SIZE:
            log_queue.put(f"{log_prefix} Error: File data too small ({len(sector_data)} bytes) for header/table.")
            return None, None, None

        # 1. Read Header
        magic, version, num_chunks = struct.unpack_from(HEADER_STRUCT_FORMAT, sector_data, 0)

        if magic != SECTOR_MAGIC:
            log_queue.put(f"{log_prefix} Error: Invalid magic ID: {magic}")
            return None, None, None
        if version != EXPECTED_VERSION:
            log_queue.put(f"{log_prefix} Warning: Unexpected version {version} (expected {EXPECTED_VERSION})")
            # Continue processing if version differs, might still work
        if num_chunks != EXPECTED_NUM_CHUNKS:
            log_queue.put(f"{log_prefix} Warning: Expected {EXPECTED_NUM_CHUNKS} chunks, found {num_chunks}")
            # Continue but only read up to expected number of descriptors

        # 2. Read Chunk Table
        chunk_descriptors = []
        table_offset = HEADER_SIZE
        # Read up to num_chunks reported, but cap at EXPECTED_NUM_CHUNKS for safety
        effective_num_chunks_to_read = min(num_chunks, EXPECTED_NUM_CHUNKS)
        # Optional Debug log: log_queue.put(f"{log_prefix} --- Chunk Table (Reading {effective_num_chunks_to_read}/{num_chunks} Chunks) ---")

        for i in range(effective_num_chunks_to_read):
            desc_offset = table_offset + (i * CHUNK_DESC_SIZE)
            if desc_offset + CHUNK_DESC_SIZE > len(sector_data):
                log_queue.put(f"{log_prefix} Error: File data too small for chunk descriptor {i}")
                # Don't return yet, maybe the target chunk was already found
                break # Stop reading descriptors
            try:
                chunk_id, flags, chunk_offset, chunk_size = struct.unpack_from(
                    CHUNK_DESC_FORMAT, sector_data, desc_offset
                )
                # Optional Debug Log:
                # log_queue.put(f"{log_prefix}    Chunk[{i}]: ID={chunk_id:<3} Offset={chunk_offset:<8} Size={chunk_size:<8} Flags={flags:#010x}")
                chunk_descriptors.append({'id': chunk_id, 'offset': chunk_offset, 'size': chunk_size, 'flags': flags})
            except struct.error as e:
                 log_queue.put(f"{log_prefix} Error unpacking chunk descriptor {i}: {e}")
                 # Don't return yet

        # Optional Debug log: log_queue.put(f"{log_prefix} --- End Chunk Table ---")

        # 3. Find Terrain Vertex Chunk (ID 2)
        terrain_chunk_desc = None
        for desc in chunk_descriptors:
            # Check ID and also ensure Offset and Size seem plausible
            # Offset must be after header and chunk table
            # Size must accommodate at least the chunk header
            if desc['id'] == TERRAIN_VERTEX_CHUNK_ID and \
               desc['offset'] >= HEADER_SIZE + CHUNK_TABLE_SIZE and \
               desc['size'] >= CHUNK_2_HEADER_SIZE:
                terrain_chunk_desc = desc
                break

        if terrain_chunk_desc is None:
            log_queue.put(f"{log_prefix} Error: Terrain Vertex Chunk (ID {TERRAIN_VERTEX_CHUNK_ID}) not found or invalid in table.")
            return None, None, None

        chunk_offset = terrain_chunk_desc['offset']
        chunk_size = terrain_chunk_desc['size']

        if chunk_size != EXPECTED_CHUNK_2_SIZE:
            log_queue.put(f"{log_prefix} Warning: Terrain Chunk size is {chunk_size}, expected {EXPECTED_CHUNK_2_SIZE}.")
            # Continue if size differs, but might lead to read errors later

        if chunk_offset + chunk_size > len(sector_data):
             log_queue.put(f"{log_prefix} Error: Chunk offset/size ({chunk_offset}/{chunk_size}) exceeds file data length ({len(sector_data)}).")
             return None, None, None

        # 4. Read Chunk 2 Header (Scale/Offset)
        try:
            scale, offset = struct.unpack_from(
                CHUNK_2_HEADER_FORMAT, sector_data, chunk_offset
            )
            if not (math.isfinite(scale) and math.isfinite(offset)):
                 log_queue.put(f"{log_prefix} Warning: Scale/Offset values are non-finite ({scale}, {offset}). Check CHUNK_2_HEADER_FORMAT?")
                 # Continue, but scale/offset values are likely meaningless
            # log_queue.put(f"{log_prefix} Read Scale: {scale}, Offset: {offset}") # Optional Debug
        except struct.error as e:
            log_queue.put(f"{log_prefix} Error unpacking scale/offset header: {e}")
            return None, None, None # Can't proceed without reading header correctly

        # 5. Read Vertex Data Array
        vertex_data_offset = chunk_offset + CHUNK_2_HEADER_SIZE
        # Calculate available size for vertices based on reported chunk size
        vertex_data_size = chunk_size - CHUNK_2_HEADER_SIZE
        # Calculate max possible vertices based on available data and struct size
        max_vertices_possible = vertex_data_size // VERTEX_STRUCT_SIZE if VERTEX_STRUCT_SIZE > 0 else 0

        # Determine how many vertices to actually read: minimum of expected, possible, and prevent reading past chunk end
        num_vertices_to_read = min(NUM_VERTICES, max_vertices_possible)

        if num_vertices_to_read < NUM_VERTICES:
             log_queue.put(f"{log_prefix} Warning: Available vertex data size ({vertex_data_size} bytes) allows for only {num_vertices_to_read} vertices (expected {NUM_VERTICES}).")
             if num_vertices_to_read <= 0:
                 log_queue.put(f"{log_prefix} Error: No vertex data available after header.")
                 return None, None, None # Cannot proceed
        elif chunk_size > EXPECTED_CHUNK_2_SIZE and max_vertices_possible > NUM_VERTICES:
             log_queue.put(f"{log_prefix} Warning: Chunk data size ({chunk_size}) is larger than expected. Reading only the first {NUM_VERTICES} vertices.")
             num_vertices_to_read = NUM_VERTICES # Clamp reading to expected amount

        height_values = []
        for i in range(num_vertices_to_read):
            current_offset = vertex_data_offset + (i * VERTEX_STRUCT_SIZE)
            # Ensure we don't read past the *calculated* vertex data end
            if current_offset + VERTEX_STRUCT_SIZE > vertex_data_offset + vertex_data_size:
                log_queue.put(f"{log_prefix} Error: Reading vertex {i} would exceed calculated chunk data boundary ({vertex_data_offset + vertex_data_size}). Stopping read.")
                break

            try:
                # Unpack just the height (first 2 bytes, '<H')
                height_int, = struct.unpack_from('<H', sector_data, current_offset)
                height_values.append(height_int)
            except struct.error as e:
                log_queue.put(f"{log_prefix} Error unpacking height for vertex {i}: {e}")
                height_values.append(0) # Pad with default on error for this vertex

        # If we read fewer vertices than expected (due to errors or small chunk) pad the rest
        if len(height_values) < NUM_VERTICES:
             log_queue.put(f"{log_prefix} Warning: Read {len(height_values)} vertices, expected {NUM_VERTICES}. Padding with zeros.")
             height_values.extend([0] * (NUM_VERTICES - len(height_values)))

        # 6. Create NumPy array and reshape
        height_map_array = np.array(height_values, dtype=np.uint16).reshape((GRID_SIZE, GRID_SIZE))

        return height_map_array, scale, offset

    except struct.error as e:
        log_queue.put(f"{log_prefix} Error unpacking binary data: {e}")
        return None, None, None
    except Exception as e:
        log_queue.put(f"{log_prefix} An unexpected error occurred: {e}")
        log_queue.put(traceback.format_exc())
        return None, None, None

# --- Main worker function called by the GUI ---
def generate_heightmap(config, log_queue, progress_queue, cancel_event):
    """
    Main worker function. Scans archives based on config, extracts sectors,
    generates PNGs. Communicates via queues and checks cancel_event.
    """
    try:
        input_dir_path = config.get('input_dir', '.')
        output_dir_path = config.get('output_dir', '.') # Get output directory from config
        sector_suffix_type = config.get('sector_suffix_type', 'B0') # Default to B0

        # Ensure output directory exists
        if not os.path.exists(output_dir_path):
            try:
                os.makedirs(output_dir_path, exist_ok=True) # exist_ok=True prevents error if dir already exists
                log_queue.put(f"Created output directory: {output_dir_path}")
            except OSError as e:
                log_queue.put(f"Error: Cannot create output directory '{output_dir_path}': {e}")
                progress_queue.put(-1.0) # Signal error
                return
        elif not os.path.isdir(output_dir_path):
             log_queue.put(f"Error: Output path '{output_dir_path}' exists but is not a directory.")
             progress_queue.put(-1.0) # Signal error
             return


        # Determine the required suffix based on config
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
            # Ensure paths are absolute for glob
            abs_input_dir = os.path.abspath(input_dir_path)
            all_archives.extend(glob.glob(os.path.join(abs_input_dir, '*.zip')))
            all_archives.extend(glob.glob(os.path.join(abs_input_dir, '*.pak')))
            all_archives = sorted(list(set(all_archives))) # Remove duplicates and sort
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

        # Progress: 0% at start, up to 95% during file processing, last 5% for wrap-up/save
        progress_scale = 0.95

        for archive_idx, archive_filepath in enumerate(all_archives):
            if cancel_event.is_set():
                log_queue.put(f"Cancellation requested before processing {os.path.basename(archive_filepath)}.")
                progress_queue.put(-1.0) # Signal cancellation/error state
                return

            archive_filename = os.path.basename(archive_filepath)
            log_queue.put(f"\nProcessing archive: {archive_filename} ({archive_idx + 1}/{total_archives})")
            processed_in_archive = 0
            failed_in_archive = 0
            sectors_found_in_archive = 0
            try:
                with zipfile.ZipFile(archive_filepath, 'r') as zf:
                    file_list = zf.infolist()
                    # Filter for files ending with suffix, ignore directories, case-insensitive
                    matching_files = [fi for fi in file_list if fi.filename.lower().endswith(required_suffix.lower()) and not fi.is_dir()]
                    sectors_found_in_archive = len(matching_files)

                    if sectors_found_in_archive == 0:
                         log_queue.put(f"  No sectors matching '{required_suffix}' found.")
                         # Update progress based on completing this archive scan
                         overall_progress = progress_scale * (archive_idx + 1) / total_archives
                         progress_queue.put(overall_progress)
                         continue # Move to the next archive

                    log_queue.put(f"  Found {sectors_found_in_archive} matching sectors. Processing...")
                    for file_idx, file_info in enumerate(matching_files):
                        if cancel_event.is_set():
                            log_queue.put(f"Cancellation requested while processing {archive_filename}.")
                            progress_queue.put(-1.0)
                            return # Exit immediately

                        try:
                            sector_data = zf.read(file_info.filename)
                            height_map_array, scale, offset = extract_heightmap_from_sector(sector_data, file_info.filename, log_queue)

                            if height_map_array is not None:
                                # Construct output filename using the sector's base name
                                base_name = os.path.basename(file_info.filename)
                                # Robustly remove known suffixes
                                if base_name.lower().endswith('_b0.sector'): base_name = base_name[:-len('_b0.sector')]
                                elif base_name.lower().endswith('_d1.sector'): base_name = base_name[:-len('_d1.sector')]
                                elif base_name.lower().endswith('_d2.sector'): base_name = base_name[:-len('_d2.sector')]
                                elif base_name.lower().endswith('.sector'): base_name = base_name[:-len('.sector')]

                                output_png_filename = f"{base_name}_height.png"
                                output_png_path = os.path.join(output_dir_path, output_png_filename)

                                img = Image.fromarray(height_map_array, mode='I;16') # 'I;16' for 16-bit integer grayscale
                                img.save(output_png_path)
                                log_queue.put(f"    -> Saved: {output_png_filename} (Scale={scale:.2f}, Offset={offset:.2f})")
                                processed_in_archive += 1
                            else:
                                # extract_heightmap_from_sector already logged the specific error
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

                        # Update progress based on files within this archive relative to total archives
                        # Calculate progress within the current archive
                        current_archive_progress = (file_idx + 1) / sectors_found_in_archive
                        # Calculate overall progress considering completed archives and progress within current one
                        overall_progress = progress_scale * (archive_idx + current_archive_progress) / total_archives
                        progress_queue.put(overall_progress)

            except zipfile.BadZipFile:
                log_queue.put(f"  Error: Invalid or corrupted archive: {archive_filename}. Skipping.")
                total_failed_count += 1 # Count the archive itself as a failure
            except FileNotFoundError:
                 log_queue.put(f"  Error: Archive file not found during processing loop (was it moved/deleted?): {archive_filename}")
                 total_failed_count += 1
            except PermissionError:
                log_queue.put(f"  Error: Permission denied reading archive: {archive_filename}. Skipping.")
                total_failed_count += 1
            except Exception as e:
                log_queue.put(f"  An unexpected error occurred opening/reading archive {archive_filename}: {e}")
                log_queue.put(traceback.format_exc())
                total_failed_count += 1 # Count the archive itself as a failure
            finally:
                log_queue.put(f"  Finished archive. Processed: {processed_in_archive}, Failed/Skipped: {failed_in_archive}")
                total_processed_count += processed_in_archive
                total_failed_count += failed_in_archive

        # --- Processing Finished ---
        if cancel_event.is_set():
            log_queue.put(f"\n--- Processing Cancelled ---")
        else:
            log_queue.put(f"\n--- Finished processing all archives ---")
            log_queue.put(f"Total Sectors Matching '{required_suffix}' Successfully Processed: {total_processed_count}")
            log_queue.put(f"Total Failures (Files/Archives): {total_failed_count}")
            progress_queue.put(1.0) # Signal normal completion

    except Exception as e:
         log_queue.put(f"\n--- FATAL ERROR in generate_heightmap ---")
         log_queue.put(f"Error: {e}")
         log_queue.put(traceback.format_exc())
         progress_queue.put(-1.0) # Signal error
    # No finally block needed to send progress, handled by specific return paths