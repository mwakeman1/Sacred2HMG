import struct
import math
import os
import glob
import re
import numpy as np
from PIL import Image
import sys
import traceback
import threading
import time
from queue import Queue  # For thread-safe communication
import zipfile         # Keep this import

# --- ImGui / GUI specific imports ---
# Use tkinter for the folder dialog as it's often built-in
import tkinter as tk
from tkinter import filedialog

# Use imgui-bundle for easier setup
try:
    # Use newer API if available
    import imgui_bundle
    from imgui_bundle import imgui, implot, hello_imgui, ImVec2 # noqa F401
    import OpenGL.GL as gl
    IMGUI_BACKEND = "imgui_bundle"
except ImportError:
    # Fallback to older separate libraries if necessary
    import imgui
    from imgui.integrations.glfw import GlfwRenderer
    import glfw
    import OpenGL.GL as gl
    IMGUI_BACKEND = "pyimgui"


# --- Core Processing Logic (slightly modified from original) ---

# --- Configuration Defaults (will be controlled by UI) ---
# These remain constants as they describe the file format
ENTRY_SIZE = 8
HEIGHT_OFFSET_IN_ENTRY = 6  # Offset within the entry (uint16)
HEIGHT_FORMAT = '<H'        # Unsigned short for offset
EXPECTED_ZERO_BYTES_OFFSETS = [2, 3, 4, 5]

START_MARKER = b'\x00\x00\x01\x00'
VARIABLE_DATA_AFTER_MARKER_SIZE = 4
INTERMEDIATE_HEADER_SIZE = 8
BYTES_TO_SKIP_AFTER_MARKER = VARIABLE_DATA_AFTER_MARKER_SIZE + INTERMEDIATE_HEADER_SIZE  # 12 bytes
N_CONTEXT_CHECK = 5

BASE_H_OFFSET_IN_HEADER = 2
BASE_H_FORMAT = '<h'        # Signed short base from header

TILE_WIDTH = 7
TILE_HEIGHT = 7
POINTS_PER_TILE = TILE_WIDTH * TILE_HEIGHT

LAYOUT_SECTOR_WIDTH = 224
LAYOUT_SECTOR_HEIGHT = 224

# --- Helper Functions (mostly unchanged) ---
def parse_filename(filename):
    # Regex should still work fine as it captures digits before the suffix
    match = re.match(r'(\d{3})_(\d{3})_.*\.sector', os.path.basename(filename))
    if match:
        return int(match.group(1)), int(match.group(2))
    else:
        return None, None

def create_weight_map(sector_h, sector_w, blend_size):
    # Create a weight map with exponential ramp on the boundaries
    weight_map = np.ones((sector_h, sector_w), dtype=np.float32)
    if blend_size <= 0:
        return weight_map
    blend_pixels = min(blend_size, sector_w // 2, sector_h // 2)
    if blend_pixels <= 0:
        return weight_map
    # Smoother curve: cubic instead of exponential for potentially better visual blending
    x_blend = np.linspace(0., 1., blend_pixels + 1)[1:]
    ramp = 2.0 * x_blend**3 - 3.0 * x_blend**2 + 1.0 # Smoothstep function (flipped)

    center_start = blend_pixels
    center_end_y = sector_h - blend_pixels
    center_end_x = sector_w - blend_pixels

    # Apply ramp to edges
    for i in range(blend_pixels):
        weight = ramp[i]
        if center_start < center_end_x: # Top and Bottom edges
            weight_map[i, center_start:center_end_x] = weight
            weight_map[sector_h - 1 - i, center_start:center_end_x] = weight
        if center_start < center_end_y: # Left and Right edges
            weight_map[center_start:center_end_y, i] = weight
            weight_map[center_start:center_end_y, sector_w - 1 - i] = weight

    # Apply ramp to corners (use minimum of intersecting edge weights)
    for r in range(blend_pixels):
        for c in range(blend_pixels):
            weight_corner = min(ramp[r], ramp[c])
            weight_map[r, c] = min(weight_map[r, c], weight_corner) # Top-left
            weight_map[r, sector_w - 1 - c] = min(weight_map[r, sector_w - 1 - c], weight_corner) # Top-right
            weight_map[sector_h - 1 - r, c] = min(weight_map[sector_h - 1 - r, c], weight_corner) # Bottom-left
            weight_map[sector_h - 1 - r, sector_w - 1 - c] = min(weight_map[sector_h - 1 - r, sector_w - 1 - c], weight_corner) # Bottom-right


    if center_start < center_end_y and center_start < center_end_x:
        weight_map[center_start:center_end_y, center_start:center_end_x] = 1.0
    return weight_map

# --- MODIFIED Function Signature and Internal Logic ---
def process_sector_base_offset(sector_content_bytes, sector_filename_in_zip, expected_width, expected_height, log_queue):
    """Reads int16 base (Hdr@2) and uint16 offsets (Entry@6) from sector file content bytes."""
    height_offsets = []
    base_h = 0
    header_bytes = None
    data_start_offset = -1
    expected_vertices = expected_width * expected_height
    width = None
    height = None
    # Use the filename from within the zip for logging
    filename_short = os.path.basename(sector_filename_in_zip)

    try:
        # --- Operate directly on the passed bytes ---
        content_to_scan = sector_content_bytes
        file_size = len(content_to_scan)
        if not content_to_scan:
            log_queue.put(f" W: Skipping empty sector content for {filename_short}")
            return None, None, None, None, None, None

        search_start_offset = 0
        found_valid_context = False
        while search_start_offset < file_size:
            current_marker_offset = content_to_scan.find(START_MARKER, search_start_offset)
            if current_marker_offset == -1:
                break # No more markers

            potential_data_start = current_marker_offset + len(START_MARKER) + BYTES_TO_SKIP_AFTER_MARKER
            potential_header_start = current_marker_offset + len(START_MARKER)

            # Check if enough space for header and context check
            if potential_data_start + (N_CONTEXT_CHECK * ENTRY_SIZE) <= file_size:
                context_valid = True
                for i in range(N_CONTEXT_CHECK):
                    entry_offset = potential_data_start + (i * ENTRY_SIZE)
                    # Check bounds before slicing
                    if entry_offset + ENTRY_SIZE > file_size:
                         context_valid = False
                         break
                    entry_bytes = content_to_scan[entry_offset : entry_offset + ENTRY_SIZE]
                    # Basic check: length and expected zero bytes
                    # Ensure entry_bytes has expected length before indexing
                    if len(entry_bytes) < ENTRY_SIZE or not all(entry_bytes[z] == 0x00 for z in EXPECTED_ZERO_BYTES_OFFSETS):
                        context_valid = False
                        break # This context is not valid

                if context_valid:
                    # Context seems valid, try reading the base height from the header
                    try:
                        # Check bounds before slicing
                        if potential_header_start + BYTES_TO_SKIP_AFTER_MARKER > file_size:
                            log_queue.put(f" W:{filename_short} Not enough data for header after marker.")
                            context_valid = False # Treat as invalid context
                        else:
                            header_bytes = content_to_scan[potential_header_start : potential_header_start + BYTES_TO_SKIP_AFTER_MARKER]
                            if len(header_bytes) >= BASE_H_OFFSET_IN_HEADER + struct.calcsize(BASE_H_FORMAT):
                                base_h, = struct.unpack_from(BASE_H_FORMAT, header_bytes, BASE_H_OFFSET_IN_HEADER)
                                # Found valid context and read base_h
                                data_start_offset = potential_data_start
                                found_valid_context = True
                                break # Found what we need, exit the while loop
                            else:
                                # Header too short after marker - unusual
                                log_queue.put(f" W:{filename_short} Header size insufficient after marker.")
                                base_h = 0
                                header_bytes = None
                                # Continue searching for another marker (handled by loop)
                    except Exception as e:
                        log_queue.put(f" W:{filename_short} Error reading header base H: {e}")
                        base_h = 0
                        header_bytes = None
                        # Continue searching (handled by loop)

            # Move search start past the current marker to find the next one
            search_start_offset = current_marker_offset + 1
        # --- End of search loop ---

        if not found_valid_context:
            log_queue.put(f" W: No valid data block found in {filename_short}.")
            return None, None, None, None, None, None

        # --- Read entries directly from the byte content ---
        bytes_needed = expected_vertices * ENTRY_SIZE
        data_block_bytes = content_to_scan[data_start_offset:] # Slice the data part

        if len(data_block_bytes) < bytes_needed:
            log_queue.put(f" W:{filename_short} Not enough data for {expected_vertices} vertices after marker (found {len(data_block_bytes)} bytes, need {bytes_needed}). Skipping.")
            return None, None, None, None, None, None

        for i in range(expected_vertices):
            entry_offset_in_data = i * ENTRY_SIZE
            # Slice the specific entry from the data block
            entry_bytes = data_block_bytes[entry_offset_in_data : entry_offset_in_data + ENTRY_SIZE]

            # len check should not be needed due to overall check above, but belt-and-suspenders:
            if len(entry_bytes) < ENTRY_SIZE:
                 log_queue.put(f" E:{filename_short} Unexpected short read at vertex {i+1}/{expected_vertices} (internal error).")
                 return None, None, None, None, None, None
            try:
                # Check zero bytes again for this specific entry for robustness
                if not all(entry_bytes[z] == 0x00 for z in EXPECTED_ZERO_BYTES_OFFSETS):
                     log_queue.put(f" W:{filename_short} Entry {i} failed zero byte check. Using offset anyway.") # Or skip?
                h_val, = struct.unpack_from(HEIGHT_FORMAT, entry_bytes, HEIGHT_OFFSET_IN_ENTRY)
                height_offsets.append(h_val)
            except struct.error as e:
                log_queue.put(f" W:{filename_short} Struct error unpacking uint16 offset at vertex {i}: {e}")
                # Decide handling: return error, or append a default (like 0 or NaN)? Let's error out.
                return None, None, None, None, None, None

        if len(height_offsets) != expected_vertices:
            # This case should theoretically not be reached due to earlier checks
            log_queue.put(f" E:{filename_short} Read count mismatch ({len(height_offsets)} vs {expected_vertices}).")
            return None, None, None, None, None, None

        width = expected_width
        height = expected_height
        return height_offsets, width, height, base_h, header_bytes, data_start_offset # data_start_offset is relative to original bytes

    except Exception as e:
        # Catch errors during byte processing
        log_queue.put(f" Error processing sector content for {filename_short}: {e}")
        log_queue.put(traceback.format_exc())
        return None, None, None, None, None, None


def correct_detile_sector(sector_offset_values, sector_width, sector_height, log_queue):
    """Detiling function using row-major indexing."""
    detiled_heights = np.zeros((sector_height, sector_width), dtype=np.uint16)

    # Calculate the number of tiles per row and per column in the sector
    tiles_per_row = sector_width // TILE_WIDTH
    tiles_per_col = sector_height // TILE_HEIGHT

    if TILE_WIDTH <= 0 or TILE_HEIGHT <= 0:
         log_queue.put(" E: Invalid TILE_WIDTH or TILE_HEIGHT (must be > 0).")
         return None
    if sector_width % TILE_WIDTH != 0 or sector_height % TILE_HEIGHT != 0:
         log_queue.put(f" W: Sector dimensions ({sector_width}x{sector_height}) not perfectly divisible by tile size ({TILE_WIDTH}x{TILE_HEIGHT}).")
         # Adjust tiles_per_row/col maybe? Or just proceed? Let's proceed but it might be wrong.
         tiles_per_row = math.ceil(sector_width / TILE_WIDTH)
         tiles_per_col = math.ceil(sector_height / TILE_HEIGHT)


    expected_vertices = sector_width * sector_height
    points_per_tile = TILE_WIDTH * TILE_HEIGHT
    expected_input_length = tiles_per_row * tiles_per_col * points_per_tile

    # Adjust check to handle potentially non-exact fits if warning above is allowed
    if len(sector_offset_values) < expected_vertices:
         log_queue.put(f"  W: Detile input size mismatch. Expected >= {expected_vertices}, got {len(sector_offset_values)}")
         # Allow processing if it's *larger*, maybe extra data? But smaller is definitely an issue.
         return None
    # Optional: Warn if larger than expected
    # if len(sector_offset_values) > expected_input_length:
    #     log_queue.put(f"  W: Detile input size larger than calculated tile layout ({len(sector_offset_values)} > {expected_input_length}).")


    vertex_index = 0
    for tile_row in range(tiles_per_col):
        for tile_col in range(tiles_per_row):
            tile_base_y = tile_row * TILE_HEIGHT
            tile_base_x = tile_col * TILE_WIDTH
            for local_y in range(TILE_HEIGHT):
                for local_x in range(TILE_WIDTH):
                    output_y = tile_base_y + local_y
                    output_x = tile_base_x + local_x

                    # Ensure we don't write outside the actual sector bounds if not perfectly divisible
                    if output_y < sector_height and output_x < sector_width:
                        if vertex_index < len(sector_offset_values):
                            detiled_heights[output_y, output_x] = sector_offset_values[vertex_index]
                        else:
                            log_queue.put(f" E: Detile ran out of input data at index {vertex_index} for output ({output_x},{output_y}).")
                            # Fill with 0 or raise error? Let's fill with 0 and continue.
                            detiled_heights[output_y, output_x] = 0
                        vertex_index += 1
                    # else: skip this vertex as it's outside bounds (padding tile)

    if vertex_index < expected_vertices:
         log_queue.put(f" W: Detile finished but processed only {vertex_index}/{expected_vertices} expected vertices.")


    return detiled_heights


def smooth_tile_boundaries(heightmap, tile_width, tile_height, log_queue):
    """Apply simple averaging smoothing at tile boundaries."""
    if tile_width <= 1 or tile_height <= 1:
         log_queue.put(" W: Tile smoothing skipped, tile dimensions too small.")
         return heightmap.astype(np.float32) # Return copy

    height, width = heightmap.shape
    # Work on a float copy
    smoothed = heightmap.copy().astype(np.float32)

    # Create masks for boundary pixels
    h_boundary_mask = np.zeros_like(heightmap, dtype=bool)
    v_boundary_mask = np.zeros_like(heightmap, dtype=bool)

    # Mark horizontal boundaries (excluding image edges)
    for x in range(tile_width - 1, width - 1, tile_width):
        h_boundary_mask[:, x] = True

    # Mark vertical boundaries (excluding image edges)
    for y in range(tile_height - 1, height - 1, tile_height):
        v_boundary_mask[y, :] = True

    # --- Smooth Horizontal Boundaries ---
    # Pixels just BEFORE the boundary
    x_indices_before = np.arange(tile_width - 1, width - 1, tile_width)
    if len(x_indices_before) > 0:
        left_pixels = smoothed[:, x_indices_before - 1]
        right_pixels = smoothed[:, x_indices_before + 1]
        smoothed[:, x_indices_before] = (left_pixels + right_pixels) / 2.0

    # --- Smooth Vertical Boundaries ---
    # Pixels just BEFORE the boundary
    y_indices_before = np.arange(tile_height - 1, height - 1, tile_height)
    if len(y_indices_before) > 0:
        above_pixels = smoothed[y_indices_before - 1, :]
        below_pixels = smoothed[y_indices_before + 1, :]
        smoothed[y_indices_before, :] = (above_pixels + below_pixels) / 2.0

    # --- Smooth Corners (Intersection of Horizontal and Vertical Boundaries) ---
    corner_mask = h_boundary_mask & v_boundary_mask
    corner_y_indices, corner_x_indices = np.where(corner_mask)

    # Ensure indices are valid for neighbor access (not on image edge)
    valid_corners = (corner_y_indices > 0) & (corner_y_indices < height - 1) & \
                    (corner_x_indices > 0) & (corner_x_indices < width - 1)

    corner_y = corner_y_indices[valid_corners]
    corner_x = corner_x_indices[valid_corners]

    if len(corner_y) > 0:
         # Average the 4 diagonal neighbors
         smoothed[corner_y, corner_x] = (smoothed[corner_y - 1, corner_x - 1] +
                                         smoothed[corner_y - 1, corner_x + 1] +
                                         smoothed[corner_y + 1, corner_x - 1] +
                                         smoothed[corner_y + 1, corner_x + 1]) / 4.0

    return smoothed

# --- MODIFIED Main Processing Function ---
def generate_heightmap(config, log_queue, progress_queue):
    """
    Processes sector files found within ZIP archives based on config and updates GUI via queues.
    """
    try:
        input_dir = config['input_dir']
        output_filename = config['output_filename']
        apply_boundary_smoothing = config['apply_boundary_smoothing']
        sector_overlap = config['sector_overlap']
        boundary_blend_size = config['boundary_blend_size']
        height_scale_factor = config['height_scale_factor']
        sector_suffix_type = config['sector_suffix_type'] # <--- Get suffix type

        if not input_dir or not os.path.isdir(input_dir):
            log_queue.put("Error: Input directory is not valid.")
            progress_queue.put(-1.0) # Indicate error
            return

        # --- Parameter validation ---
        if height_scale_factor == 0:
             log_queue.put("Error: Height Scale Factor cannot be zero.")
             progress_queue.put(-1.0)
             return
        height_scale_factor = float(height_scale_factor)
        # ---

        output_file_path = os.path.join(input_dir, output_filename)

        log_queue.put("--- Starting Heightmap Generation (from ZIPs) ---")
        log_queue.put(f"Input Folder: {input_dir}")
        log_queue.put(f"Output File: {output_filename}")
        log_queue.put(f"Apply Boundary Smoothing: {apply_boundary_smoothing}")
        log_queue.put(f"Sector Overlap: {sector_overlap}")
        log_queue.put(f"Boundary Blend Size: {boundary_blend_size}")
        log_queue.put(f"Height Scale Factor: {height_scale_factor}")
        log_queue.put(f"Height = int16@Hdr[2] + (uint16@Entry[6] / {height_scale_factor})")
        log_queue.put("Normalization: Min-Max")
        progress_queue.put(0.0)

        # Determine the required suffix (case-insensitive)
        if sector_suffix_type == "B0":
            required_suffix = "_b0.sector"
        elif sector_suffix_type == "D1":
            required_suffix = "_d1.sector"
        else: # Assume d2
            required_suffix = "_d2.sector"
        log_queue.put(f"Searching for sectors ending with: '{required_suffix}' (case-insensitive)")
        # ---

        # --- Find ZIP files ---
        all_zip_files = glob.glob(os.path.join(input_dir, '*.zip'))
        log_queue.put(f"Found {len(all_zip_files)} ZIP files to scan.")
        if not all_zip_files:
            log_queue.put("Error: No .zip files found in the specified directory.")
            progress_queue.put(-1.0) # Indicate error
            return
        # ---

        log_queue.put(f"Using fixed layout size: {LAYOUT_SECTOR_WIDTH} x {LAYOUT_SECTOR_HEIGHT}")

        # --- Pass 1: Read Header Base and Entry Offsets from sectors within Zips ---
        log_queue.put("--- Pass 1: Reading sector data from ZIPs ---")
        sectors_data = {}
        all_coords = []
        processed_count = 0
        total_zips = len(all_zip_files)

        # --- Iterate through ZIP files ---
        for zip_idx, zip_filepath in enumerate(all_zip_files):
            zip_filename = os.path.basename(zip_filepath)
            log_queue.put(f" -> Processing ZIP: {zip_filename} ({zip_idx+1}/{total_zips})")
            try:
                with zipfile.ZipFile(zip_filepath, 'r') as zf:
                    # Get names of all files within the zip
                    member_names = zf.namelist()
                    # --- MODIFIED FILTER ---
                    sector_members = [
                        name for name in member_names
                        if name.lower().endswith(required_suffix) # Use the required suffix
                    ]
                    # --- END MODIFIED FILTER ---

                    if not sector_members:
                        log_queue.put(f"    No sectors matching '{required_suffix}' found in {zip_filename}")
                        continue

                    # --- Iterate through .sector files found WITHIN the current zip ---
                    for sector_name_in_zip in sector_members:
                        filename_short = os.path.basename(sector_name_in_zip) # Get short name for parsing/logging
                        sx, sy = parse_filename(filename_short)
                        if sx is None or sy is None:
                            log_queue.put(f"    W: Skipping '{filename_short}' in {zip_filename}, could not parse coordinates.")
                            continue

                        # Prevent processing duplicates if coordinates already exist
                        if (sx, sy) in sectors_data:
                            log_queue.put(f"    W: Skipping '{filename_short}' in {zip_filename}, coordinates ({sx},{sy}) already processed from another file.")
                            continue

                        try:
                            # Read the content of the sector file from the zip
                            sector_bytes = zf.read(sector_name_in_zip)

                            # Call the MODIFIED processing function with BYTES
                            result = process_sector_base_offset(
                                sector_bytes,
                                filename_short, # Pass the name for logging
                                LAYOUT_SECTOR_WIDTH,
                                LAYOUT_SECTOR_HEIGHT,
                                log_queue
                            )

                            if result[0] is not None:
                                height_offsets, w, h, base_h, _, _ = result
                                if w != LAYOUT_SECTOR_WIDTH or h != LAYOUT_SECTOR_HEIGHT:
                                     log_queue.put(f"    W: {filename_short} in {zip_filename} - dimensions mismatch ({w}x{h}). Skipping.")
                                     continue

                                detiled_offsets_uint16 = correct_detile_sector(height_offsets, w, h, log_queue)

                                if detiled_offsets_uint16 is not None:
                                    log_queue.put(f"    OK: Processed {filename_short} from {zip_filename}")
                                    sectors_data[(sx, sy)] = {
                                        'detiled_offsets': detiled_offsets_uint16,
                                        'base_h': base_h,
                                        'width': w,
                                        'height': h
                                    }
                                    all_coords.append((sx, sy))
                                    processed_count += 1
                                else:
                                    log_queue.put(f"    W: Failed to detile {filename_short} from {zip_filename}. Skipping.")
                            # else: process_sector_base_offset already logged the error for this sector

                        except KeyError:
                             log_queue.put(f"    E: Sector '{sector_name_in_zip}' not found in {zip_filename} (unexpected).")
                        except Exception as sector_e:
                             log_queue.put(f"    E: Error reading/processing sector '{sector_name_in_zip}' from {zip_filename}: {sector_e}")
                             log_queue.put(traceback.format_exc())
                    # --- End loop over sectors in zip ---

            except zipfile.BadZipFile:
                log_queue.put(f" W: Skipping invalid or corrupted ZIP file: {zip_filename}")
            except Exception as zip_e:
                log_queue.put(f" E: Error opening or reading ZIP file {zip_filename}: {zip_e}")
                log_queue.put(traceback.format_exc())
            # --- End processing one zip file ---

            # Update progress based on ZIP files processed
            progress_queue.put(0.3 * (zip_idx + 1) / total_zips) # Pass 1 is ~30%

        # --- End loop over all zip files ---


        log_queue.put(f"Finished Pass 1. Processed {processed_count} valid sectors from {total_zips} ZIP files.")
        if processed_count == 0:
            log_queue.put("Error: No valid sector data could be processed from any ZIP file.")
            progress_queue.put(-1.0) # Error
            return

        # --- Determine Map Bounds (remains the same) ---
        min_sx = min(c[0] for c in all_coords)
        max_sx = max(c[0] for c in all_coords)
        min_sy = min(c[1] for c in all_coords)
        max_sy = max(c[1] for c in all_coords)
        log_queue.put(f"Determined Sector Coordinate Range: X=[{min_sx}-{max_sx}], Y=[{min_sy}-{max_sy}]")

        # --- Calculate Final Map Size (remains the same) ---
        current_overlap = sector_overlap
        max_possible_overlap = min(LAYOUT_SECTOR_WIDTH, LAYOUT_SECTOR_HEIGHT) -1
        if not (0 <= current_overlap <= max_possible_overlap) :
             log_queue.put(f"Warning: Invalid SECTOR_OVERLAP ({current_overlap}). Clamping to range [0, {max_possible_overlap}].")
             current_overlap = max(0, min(current_overlap, max_possible_overlap))
        effective_sector_width = LAYOUT_SECTOR_WIDTH - current_overlap
        effective_sector_height = LAYOUT_SECTOR_HEIGHT - current_overlap
        if effective_sector_width <= 0 or effective_sector_height <= 0:
             log_queue.put(f"Error: Overlap ({current_overlap}) is too large for sector dimensions. Effective size is non-positive.")
             progress_queue.put(-1.0)
             return
        num_sectors_x = max_sx - min_sx + 1
        num_sectors_y = max_sy - min_sy + 1
        final_width = (num_sectors_x - 1) * effective_sector_width + LAYOUT_SECTOR_WIDTH if num_sectors_x > 0 else 0
        final_height = (num_sectors_y - 1) * effective_sector_height + LAYOUT_SECTOR_HEIGHT if num_sectors_y > 0 else 0
        final_width = max(0, final_width)
        final_height = max(0, final_height)
        if num_sectors_x <= 1: final_width = LAYOUT_SECTOR_WIDTH
        if num_sectors_y <= 1: final_height = LAYOUT_SECTOR_HEIGHT
        log_queue.put(f"Using Overlap={current_overlap}, Blend Size={boundary_blend_size}")
        log_queue.put(f"Calculated final map dimensions: {final_width} x {final_height}")
        if final_width <= 0 or final_height <= 0:
            log_queue.put("Error: Calculated final map dimensions are zero or negative.")
            progress_queue.put(-1.0) # Error
            return
        # ---

        # --- Allocate Memory (remains the same) ---
        try:
            log_queue.put(f"Allocating final map arrays ({final_height} x {final_width})...")
            heightmap_sum_array = np.zeros((final_height, final_width), dtype=np.float64)
            weight_sum_array = np.zeros((final_height, final_width), dtype=np.float64)
            log_queue.put("Allocation successful.")
        except MemoryError:
             log_queue.put(f"Error: Not enough memory to allocate map arrays of size {final_height} x {final_width}.")
             progress_queue.put(-1.0); return
        except Exception as e:
             log_queue.put(f"Error creating numpy arrays: {e}"); progress_queue.put(-1.0); return
        # ---

        # --- Global Base Correction (remains the same) ---
        base_sum = 0.0
        base_count = 0
        for data in sectors_data.values():
             if 'base_h' in data and data['base_h'] is not None:
                  base_sum += data['base_h']; base_count += 1
        if base_count > 0: global_base = base_sum / base_count
        else: global_base = 0.0; log_queue.put("W: Could not compute global average base...")
        log_queue.put(f"Global average base computed: {global_base:.2f} from {base_count} sectors.")
        # ---

        # --- Pass 2: Calculate Height, Smooth (Optional), and Blend (remains the same) ---
        log_queue.put("--- Pass 2: Calculating heights, smoothing, and blending ---")
        placed_count = 0
        overall_min_h = float('inf'); overall_max_h = float('-inf')
        first_sector_processed = False
        total_sectors_to_place = len(sectors_data) # Use count of successfully read sectors

        for i, ((sx, sy), data) in enumerate(sectors_data.items()):
            # --- This whole inner loop logic remains IDENTICAL ---
            sector_w = data['width']; sector_h = data['height']
            base_adjustment = data['base_h'] - global_base
            absolute_height_sector_float = (data['detiled_offsets'].astype(np.float64) / height_scale_factor) + base_adjustment
            if apply_boundary_smoothing:
                absolute_height_sector_float = smooth_tile_boundaries(absolute_height_sector_float, TILE_WIDTH, TILE_HEIGHT, log_queue)
            sector_min = np.nanmin(absolute_height_sector_float)
            sector_max = np.nanmax(absolute_height_sector_float)
            if np.isfinite(sector_min) and np.isfinite(sector_max):
                 overall_min_h = min(overall_min_h, sector_min); overall_max_h = max(overall_max_h, sector_max)
                 first_sector_processed = True
            else: log_queue.put(f" W: Sector ({sx},{sy}) resulted in non-finite heights...")
            weight_map = create_weight_map(sector_h, sector_w, boundary_blend_size)
            grid_x = sx - min_sx; grid_y = sy - min_sy
            paste_x_start = grid_x * effective_sector_width; paste_y_start = grid_y * effective_sector_height
            paste_x_end = paste_x_start + sector_w; paste_y_end = paste_y_start + sector_h
            target_y_start_clipped=max(0,paste_y_start); target_y_end_clipped=min(final_height,paste_y_end)
            target_x_start_clipped=max(0,paste_x_start); target_x_end_clipped=min(final_width,paste_x_end)
            if target_y_start_clipped >= target_y_end_clipped or target_x_start_clipped >= target_x_end_clipped: continue
            clip_top=target_y_start_clipped-paste_y_start; clip_left=target_x_start_clipped-paste_x_start
            clipped_height=target_y_end_clipped-target_y_start_clipped; clipped_width=target_x_end_clipped-target_x_start_clipped
            source_y_start=clip_top; source_y_end=clip_top+clipped_height
            source_x_start=clip_left; source_x_end=clip_left+clipped_width
            target_slice=np.s_[target_y_start_clipped:target_y_end_clipped, target_x_start_clipped:target_x_end_clipped]
            source_slice=np.s_[source_y_start:source_y_end, source_x_start:source_x_end]
            if absolute_height_sector_float[source_slice].shape != (clipped_height,clipped_width) or weight_map[source_slice].shape != (clipped_height,clipped_width):
                 log_queue.put(f" E: Slice mismatch for sector ({sx},{sy})! Skipping paste."); continue
            try:
                 source_heights = absolute_height_sector_float[source_slice]
                 source_weights = weight_map[source_slice]
                 valid_mask = np.isfinite(source_heights)
                 heightmap_sum_array[target_slice][valid_mask] += source_heights[valid_mask] * source_weights[valid_mask]
                 weight_sum_array[target_slice][valid_mask] += source_weights[valid_mask]
                 placed_count += 1
            except Exception as e: log_queue.put(f" E: Error during weighted add for ({sx},{sy}): {e}"); log_queue.put(traceback.format_exc()); continue
            # --- End of identical inner loop ---

            # Update progress based on SECTORS processed in this pass
            progress_queue.put(0.3 + 0.6 * (i + 1) / total_sectors_to_place) # Pass 2 is ~60%

        log_queue.put(f"Finished Pass 2. Blended data from {placed_count} sector placements.")
        if placed_count == 0:
            log_queue.put("Error: No sectors were successfully placed onto the map.")
            progress_queue.put(-1.0) # Error
            return

        # --- Pass 3: Finalize Map (Divide by weights and Normalize) (remains the same) ---
        log_queue.put("--- Pass 3: Finalizing map and normalizing ---")
        if not first_sector_processed or not np.isfinite(overall_min_h) or not np.isfinite(overall_max_h):
            log_queue.put(" W: No valid finite height range found...") # Handle as before
            temp_min = np.nanmin(heightmap_sum_array[weight_sum_array > 1e-9])
            temp_max = np.nanmax(heightmap_sum_array[weight_sum_array > 1e-9])
            if np.isfinite(temp_min) and np.isfinite(temp_max): overall_min_h=temp_min; overall_max_h=temp_max; log_queue.put(f" W: Using range from sums: [{overall_min_h:.2f}, {overall_max_h:.2f}]")
            else: overall_min_h=0.0; overall_max_h=1.0; log_queue.put(" W: Could not determine range...")
        else: log_queue.put(f"Global Height Range: [{overall_min_h:.2f}, {overall_max_h:.2f}]")
        log_queue.put("Calculating final heights...")
        final_heightmap_float = np.full((final_height, final_width), np.nan, dtype=np.float32)
        valid_weights_mask = weight_sum_array > 1e-9
        np.divide(heightmap_sum_array, weight_sum_array, out=final_heightmap_float, where=valid_weights_mask)
        log_queue.put("Normalizing heightmap...")
        norm_min = overall_min_h; norm_max = overall_max_h
        log_queue.put(f"Normalization Range Used: [{norm_min:.2f}, {norm_max:.2f}] -> [0, 255]")
        DEFAULT_BG_COLOR_FLOAT = norm_min
        final_heightmap_float[~valid_weights_mask] = DEFAULT_BG_COLOR_FLOAT
        final_heightmap_float[np.isnan(final_heightmap_float)] = DEFAULT_BG_COLOR_FLOAT
        norm_range = norm_max - norm_min
        if norm_range <= 1e-9:
            log_queue.put(" W: Normalization range zero...")
            heightmap_8bit = np.full((final_height, final_width), 0, dtype=np.uint8)
        else:
            normalized_map = (final_heightmap_float - norm_min) / norm_range
            np.clip(normalized_map, 0.0, 1.0, out=normalized_map)
            heightmap_8bit = (normalized_map * 255.0).astype(np.uint8)
        # ---

        progress_queue.put(0.95) # Almost done

        # --- Save the Final Image (remains the same) ---
        try:
            log_queue.put(f"Saving final heightmap to {output_file_path}...")
            img = Image.fromarray(heightmap_8bit, mode='L')
            img.save(output_file_path)
            log_queue.put("\n--- Processing Complete ---")
            log_queue.put(f"Output saved as: {output_file_path}")
            log_queue.put(f"Final map size: {final_width} x {final_height}")
            log_queue.put(f"Calculated Height Range: [{overall_min_h:.2f}, {overall_max_h:.2f}]")
            log_queue.put(f"Normalization Applied: [{norm_min:.2f}, {norm_max:.2f}] -> [0, 255]")
            progress_queue.put(1.0) # Done
        except Exception as e:
            log_queue.put(f"\nError saving final image: {e}")
            log_queue.put(traceback.format_exc())
            progress_queue.put(-1.0) # Error

    except Exception as e:
        # Catch-all for unexpected errors in the main processing thread
        log_queue.put(f"\n--- UNEXPECTED ERROR DURING PROCESSING ---")
        log_queue.put(f"Error: {e}")
        log_queue.put(traceback.format_exc())
        progress_queue.put(-1.0) # Error


# --- Tkinter Folder Dialog ---
def select_folder_dialog():
    """Opens a folder selection dialog and returns the selected path."""
    root = tk.Tk()
    root.withdraw()  # Hide the main tkinter window
    # Make the dialog appear on top
    root.attributes('-topmost', True)
    folder_path = filedialog.askdirectory(title="Select Folder Containing .sector ZIP Archives") # Modified title
    root.destroy() # Close the hidden tkinter window
    return folder_path

# --- ImGui Application ---

# Global state for the GUI
gui_state = {
    "input_dir": "C:\Program Files (x86)\Steam\steamapps\common\Sacred 2 Gold\pak",
    "output_filename": "heightmap_output.png",
    "apply_boundary_smoothing": True,
    "sector_overlap": 1,
    "boundary_blend_size": 10,
    "height_scale_factor": 1.0,
    "sector_suffix_type": "B0", # <--- Added suffix state, default "B0"
    "log_messages": ["Welcome! Select a folder containing ZIP archives and click Generate."], # Modified welcome
    "progress": 0.0,
    "processing_thread": None,
    "is_processing": False,
    "log_queue": Queue(),
    "progress_queue": Queue(),
}

def update_log_and_progress():
    """Check queues for updates from the processing thread."""
    while not gui_state["log_queue"].empty():
        message = gui_state["log_queue"].get_nowait()
        gui_state["log_messages"].append(message)
        # Optional: Limit log size
        max_log_lines = 200
        if len(gui_state["log_messages"]) > max_log_lines:
             gui_state["log_messages"] = gui_state["log_messages"][-max_log_lines:]

    while not gui_state["progress_queue"].empty():
        progress = gui_state["progress_queue"].get_nowait()
        if progress < 0: # Error signal
            gui_state["progress"] = 0.0
            gui_state["is_processing"] = False # Stop on error
        else:
            gui_state["progress"] = progress
        # If progress reaches 1.0, processing is done
        if gui_state["progress"] >= 1.0:
             # Keep is_processing True until thread fully joins? Optional.
             # For now, just set to False based on progress signal.
             gui_state["is_processing"] = False


def run_gui_imgui_bundle():
    """Main function to run the GUI using imgui-bundle."""
    def gui_loop():
        """This function is called every frame"""
        global gui_state

        # Check for updates from processing thread
        was_processing_last_frame = gui_state["is_processing"]
        if was_processing_last_frame:
            update_log_and_progress()
            if gui_state["processing_thread"] and not gui_state["processing_thread"].is_alive():
                update_log_and_progress()
                gui_state["is_processing"] = False
                gui_state["processing_thread"] = None
                if gui_state["progress"] < 1.0 and gui_state["progress"] >= 0:
                     gui_state["log_messages"].append("Warning: Processing ended prematurely.")
                     gui_state["progress"] = 0.0

        # NO imgui.set_next_window_size or imgui.begin here

        # --- Input Folder ---
        imgui.push_item_width(-200)
        changed, gui_state["input_dir"] = imgui.input_text("##InputFolder", gui_state["input_dir"], 2048)
        imgui.pop_item_width()
        imgui.same_line()
        if imgui.button("Select Input Folder"):
            selected = select_folder_dialog()
            if selected:
                gui_state["input_dir"] = selected

        # --- Output Filename ---
        imgui.push_item_width(-150)
        changed, gui_state["output_filename"] = imgui.input_text("Output Filename", gui_state["output_filename"], 256)
        imgui.pop_item_width()
        if imgui.is_item_hovered():
             imgui.set_tooltip("Filename for the output PNG image (saved in input folder).")


        imgui.separator()
        # --- Configuration Options ---
        imgui.text("Configuration:")

        changed, gui_state["apply_boundary_smoothing"] = imgui.checkbox("Apply Tile Boundary Smoothing", gui_state["apply_boundary_smoothing"])
        if imgui.is_item_hovered():
             imgui.set_tooltip("Averages pixels at the boundaries of the 7x7 tiles.")

        # --- ADDED Radio Buttons for Sector Suffix ---
        imgui.separator()
        imgui.text("Sector Type Suffix:")
        clicked_b0 = imgui.radio_button("B0 Sectors (_B0)", gui_state["sector_suffix_type"] == "B0")
        if imgui.is_item_hovered(): imgui.set_tooltip("Search for sector files like 001_001_B0.sector")
        imgui.same_line()
        clicked_d1 = imgui.radio_button("D1 Sectors (_D1)", gui_state["sector_suffix_type"] == "D1")
        if imgui.is_item_hovered(): imgui.set_tooltip("Search for sector files like 001_001_D1.sector")
        imgui.same_line()
        clicked_d2 = imgui.radio_button("d2 Sectors (_d2)", gui_state["sector_suffix_type"] == "d2")
        if imgui.is_item_hovered(): imgui.set_tooltip("Search for sector files like 001_001_d2.sector")

        if clicked_b0: gui_state["sector_suffix_type"] = "B0"
        if clicked_d1: gui_state["sector_suffix_type"] = "D1"
        if clicked_d2: gui_state["sector_suffix_type"] = "d2"
        imgui.separator()
        # --- End Radio Buttons ---


        # Use InputInt for integer values, InputFloat for float
        imgui.push_item_width(100) # Set width for next items
        changed, gui_state["sector_overlap"] = imgui.input_int("Sector Overlap", gui_state["sector_overlap"])
        gui_state["sector_overlap"] = max(0, gui_state["sector_overlap"])
        if imgui.is_item_hovered():
            imgui.set_tooltip("Number of pixels sectors overlap when stitching (0 to SectorDim-1).")


        changed, gui_state["boundary_blend_size"] = imgui.input_int("Boundary Blend Size", gui_state["boundary_blend_size"])
        gui_state["boundary_blend_size"] = max(0, gui_state["boundary_blend_size"])
        if imgui.is_item_hovered():
            imgui.set_tooltip("Size of the feathered edge (in pixels) for blending between overlapping sectors.")

        changed, gui_state["height_scale_factor"] = imgui.input_float("Height Scale Factor", gui_state["height_scale_factor"], 0.1, 1.0, "%.2f")
        if gui_state["height_scale_factor"] <= 0:
             gui_state["height_scale_factor"] = 0.01
        if imgui.is_item_hovered():
            imgui.set_tooltip("Divisor for the raw uint16 height offset value. Height = Base + Offset / Factor.")

        imgui.pop_item_width()


        imgui.separator()

        # --- Action Button & Progress ---

        # Evaluate the condition ONCE per frame before the button
        button_needs_disabling = gui_state["is_processing"]

        # Use the public API imgui.begin_disabled() / imgui.end_disabled()
        if button_needs_disabling:
            imgui.begin_disabled(True)
            imgui.push_style_var(imgui.StyleVar_.alpha, imgui.get_style().alpha * 0.5)

        # Render the button widget.
        button_clicked = imgui.button("Generate Heightmap")

        # End the disabled state block *if it was started*
        if button_needs_disabling:
            imgui.pop_style_var()
            imgui.end_disabled()

        # Now, handle the button click *after* the stack/state is potentially restored
        if button_clicked:
            if not gui_state["is_processing"]:
                gui_state["is_processing"] = True
                gui_state["progress"] = 0.0
                gui_state["log_messages"] = ["Starting processing..."]

                # --- ADDED suffix type to config ---
                current_config = {
                    'input_dir': gui_state["input_dir"],
                    'output_filename': gui_state["output_filename"],
                    'apply_boundary_smoothing': gui_state["apply_boundary_smoothing"],
                    'sector_overlap': gui_state["sector_overlap"],
                    'boundary_blend_size': gui_state["boundary_blend_size"],
                    'height_scale_factor': gui_state["height_scale_factor"],
                    'sector_suffix_type': gui_state["sector_suffix_type"], # Pass selected type
                   }
                # ---

                while not gui_state["log_queue"].empty(): gui_state["log_queue"].get()
                while not gui_state["progress_queue"].empty(): gui_state["progress_queue"].get()

                gui_state["processing_thread"] = threading.Thread(
                    target=generate_heightmap,
                    args=(current_config, gui_state["log_queue"], gui_state["progress_queue"]),
                    daemon=True
                )
                gui_state["processing_thread"].start()

        # Render the progress bar (always visible)
        imgui.same_line()
        imgui.progress_bar(gui_state["progress"], size_arg=ImVec2(-1, 0))

        # --- End of Action Button & Progress Section ---

        imgui.separator()

        # --- Log Area ---
        imgui.text("Log:")
        log_height = imgui.get_content_region_avail().y - 5
        # Make sure flag is singular 'border'
        imgui.begin_child("Log", size=ImVec2(-1, log_height), child_flags=imgui.ChildFlags_.borders)
        imgui.text_unformatted("\n".join(gui_state["log_messages"]))
        if imgui.get_scroll_y() >= imgui.get_scroll_max_y():
             imgui.set_scroll_here_y(1.0)
        imgui.end_child()

        # NO imgui.end() here

    # Use hello_imgui runner
    runner_params = hello_imgui.RunnerParams()
    runner_params.app_window_params.window_title = "Sector Heightmap Generator"
    # Set desired initial window size using window_geometry
    runner_params.app_window_params.window_geometry.size = (650, 600) # Adjusted size
    runner_params.imgui_window_params.show_menu_bar = False
    runner_params.callbacks.show_gui = gui_loop

    hello_imgui.run(runner_params)


def run_gui_pyimgui():
    """Main function to run the GUI using older pyimgui and glfw."""
    # --- GLFW Window Setup ---
    if not glfw.init():
        print("Could not initialize OpenGL context")
        sys.exit(1)

    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, gl.GL_TRUE)

    window = glfw.create_window(650, 600, "Sector Heightmap Generator", None, None)
    if not window:
        glfw.terminate()
        print("Could not initialize Window")
        sys.exit(1)

    glfw.make_context_current(window)

    # --- ImGui Context and Renderer ---
    imgui.create_context()
    impl = GlfwRenderer(window)

    # --- Main Loop ---
    while not glfw.window_should_close(window):
        glfw.poll_events()
        impl.process_inputs()

        imgui.new_frame()

        # --- GUI Definition ---
        global gui_state

        # Check for updates from processing thread
        was_processing_last_frame_pyi = gui_state["is_processing"]
        if was_processing_last_frame_pyi:
            update_log_and_progress()
            if gui_state["processing_thread"] and not gui_state["processing_thread"].is_alive():
                update_log_and_progress()
                gui_state["is_processing"] = False
                gui_state["processing_thread"] = None
                if gui_state["progress"] < 1.0 and gui_state["progress"] >= 0:
                     gui_state["log_messages"].append("Warning: Processing ended prematurely.")
                     gui_state["progress"] = 0.0

        # NO imgui.set_next_window_size or imgui.begin here

        # --- Input Folder ---
        imgui.push_item_width(-200)
        changed, gui_state["input_dir"] = imgui.input_text("##InputFolder", gui_state["input_dir"], 2048)
        imgui.pop_item_width()
        imgui.same_line()
        if imgui.button("Select Input Folder"):
            selected = select_folder_dialog()
            if selected:
                gui_state["input_dir"] = selected

        # --- Output Filename ---
        imgui.push_item_width(-150)
        changed, gui_state["output_filename"] = imgui.input_text("Output Filename", gui_state["output_filename"], 256)
        imgui.pop_item_width()
        if imgui.is_item_hovered():
             imgui.set_tooltip("Filename for the output PNG image (saved in input folder).")


        imgui.separator()
        # --- Configuration Options ---
        imgui.text("Configuration:")

        changed, gui_state["apply_boundary_smoothing"] = imgui.checkbox("Apply Tile Boundary Smoothing", gui_state["apply_boundary_smoothing"])
        if imgui.is_item_hovered():
             imgui.set_tooltip("Averages pixels at the boundaries of the 7x7 tiles.")

        # --- ADDED Radio Buttons for Sector Suffix ---
        imgui.separator()
        imgui.text("Sector Type Suffix:")
        if imgui.radio_button("B0 Sectors (_B0)", gui_state["sector_suffix_type"] == "B0"):
             gui_state["sector_suffix_type"] = "B0"
        if imgui.is_item_hovered(): imgui.set_tooltip("Search for sector files like 001_001_B0.sector")
        imgui.same_line()
        if imgui.radio_button("D1 Sectors (_D1)", gui_state["sector_suffix_type"] == "D1"):
             gui_state["sector_suffix_type"] = "D1"
        if imgui.is_item_hovered(): imgui.set_tooltip("Search for sector files like 001_001_D1.sector")
        imgui.same_line()
        if imgui.radio_button("d2 Sectors (_d2)", gui_state["sector_suffix_type"] == "d2"):
             gui_state["sector_suffix_type"] = "d2"
        if imgui.is_item_hovered(): imgui.set_tooltip("Search for sector files like 001_001_d2.sector")
        imgui.separator()
        # --- End Radio Buttons ---


        imgui.push_item_width(100)
        # Using lists for input_int/float in pyimgui
        overlap_val = [gui_state["sector_overlap"]]
        changed_overlap = imgui.input_int("Sector Overlap", overlap_val)
        if changed_overlap: gui_state["sector_overlap"] = max(0, overlap_val[0])
        if imgui.is_item_hovered(): imgui.set_tooltip("Number of pixels sectors overlap...")


        blend_val = [gui_state["boundary_blend_size"]]
        changed_blend = imgui.input_int("Boundary Blend Size", blend_val)
        if changed_blend: gui_state["boundary_blend_size"] = max(0, blend_val[0])
        if imgui.is_item_hovered(): imgui.set_tooltip("Size of the feathered edge...")


        scale_val = [gui_state["height_scale_factor"]]
        changed_scale = imgui.input_float("Height Scale Factor", scale_val, 0.1, 1.0, "%.2f")
        if changed_scale:
            gui_state["height_scale_factor"] = scale_val[0]
            if gui_state["height_scale_factor"] <= 0: gui_state["height_scale_factor"] = 0.01
        if imgui.is_item_hovered(): imgui.set_tooltip("Divisor for the raw uint16 height offset value...")

        imgui.pop_item_width()


        imgui.separator()

        # --- Action Button & Progress (Applying safer structure to pyimgui too) ---

        # Evaluate the condition ONCE per frame before the button
        button_needs_disabling_pyi = gui_state["is_processing"]

        # Conditionally push style/flags *just before* the button
        use_begin_disabled_pyi = hasattr(imgui, "begin_disabled")
        if button_needs_disabling_pyi:
            if use_begin_disabled_pyi:
                imgui.begin_disabled(True)
            else:
                try: # Fallback to internal flags
                    imgui.internal.push_item_flag(imgui.ITEM_DISABLED, True)
                except AttributeError: pass
            imgui.push_style_var(imgui.STYLE_ALPHA, imgui.get_style().alpha * 0.5)

        # Render the button widget AND store its clicked state
        button_clicked_pyi = imgui.button("Generate Heightmap")

        # Conditionally pop style/flags *immediately after* the button
        if button_needs_disabling_pyi:
             imgui.pop_style_var()
             if use_begin_disabled_pyi:
                 imgui.end_disabled()
             else:
                 try: # Pop flag if internal push was used
                     imgui.internal.pop_item_flag()
                 except AttributeError: pass

        # Now, handle the button click *after* the stack is potentially restored
        if button_clicked_pyi:
             if not gui_state["is_processing"]:
                 gui_state["is_processing"] = True
                 gui_state["progress"] = 0.0
                 gui_state["log_messages"] = ["Starting processing..."]
                 # --- ADDED suffix type to config ---
                 current_config = {
                     'input_dir': gui_state["input_dir"],
                     'output_filename': gui_state["output_filename"],
                     'apply_boundary_smoothing': gui_state["apply_boundary_smoothing"],
                     'sector_overlap': gui_state["sector_overlap"],
                     'boundary_blend_size': gui_state["boundary_blend_size"],
                     'height_scale_factor': gui_state["height_scale_factor"],
                     'sector_suffix_type': gui_state["sector_suffix_type"], # Pass selected type
                    }
                 # ---
                 while not gui_state["log_queue"].empty(): gui_state["log_queue"].get()
                 while not gui_state["progress_queue"].empty(): gui_state["progress_queue"].get()
                 gui_state["processing_thread"] = threading.Thread(
                     target=generate_heightmap,
                     args=(current_config, gui_state["log_queue"], gui_state["progress_queue"]),
                     daemon=True
                 )
                 gui_state["processing_thread"].start()

        # Render the progress bar (always visible)
        imgui.same_line()
        imgui.progress_bar(gui_state["progress"], size=(-1, 0)) # pyimgui often uses tuple

        # --- End of Action Button & Progress Section ---

        imgui.separator()

        # --- Log Area ---
        imgui.text("Log:")
        log_height = imgui.get_content_region_available()[1] - 5
        imgui.begin_child("Log", -1, log_height, border=True)
        imgui.text_unformatted("\n".join(gui_state["log_messages"]))
        if imgui.get_scroll_y() >= imgui.get_scroll_max_y():
             try:
                 imgui.set_scroll_here_y(1.0)
             except AttributeError:
                 imgui.set_scroll_here(1.0) # Fallback
        imgui.end_child()

        # NO imgui.end() here

        # --- Rendering ---
        gl.glClearColor(0.1, 0.1, 0.1, 1)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)
        imgui.render()
        impl.render(imgui.get_draw_data())
        glfw.swap_buffers(window)

    # --- Cleanup ---
    impl.shutdown()
    imgui.destroy_context()
    glfw.terminate()


if __name__ == "__main__":
    if IMGUI_BACKEND == "imgui_bundle":
        print("Using imgui-bundle backend.")
        run_gui_imgui_bundle()
    elif IMGUI_BACKEND == "pyimgui":
        print("Using pyimgui backend.")
        run_gui_pyimgui()
    else:
        print("Error: No suitable ImGui backend found.")