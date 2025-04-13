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


def process_sector_base_offset(filepath, expected_width, expected_height, log_queue):
    """Reads int16 base (Hdr@2) and uint16 offsets (Entry@6) from the .sector file."""
    height_offsets = []
    base_h = 0
    header_bytes = None
    data_start_offset = -1
    expected_vertices = expected_width * expected_height
    width = None
    height = None
    filename_short = os.path.basename(filepath)

    try:
        with open(filepath, 'rb') as f:
            content_to_scan = f.read()
            file_size = len(content_to_scan)
            if not content_to_scan:
                log_queue.put(f" W: Skipping empty file {filename_short}")
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
                        entry_bytes = content_to_scan[entry_offset : entry_offset + ENTRY_SIZE]
                        # Basic check: length and expected zero bytes
                        if len(entry_bytes) < ENTRY_SIZE or not all(entry_bytes[z] == 0x00 for z in EXPECTED_ZERO_BYTES_OFFSETS):
                            context_valid = False
                            break # This context is not valid

                    if context_valid:
                        # Context seems valid, try reading the base height from the header
                        try:
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
                                # Continue searching for another marker
                        except Exception as e:
                            log_queue.put(f" W:{filename_short} Error reading header base H: {e}")
                            base_h = 0
                            header_bytes = None
                            # Continue searching

                # Move search start past the current marker to find the next one
                search_start_offset = current_marker_offset + 1

            if not found_valid_context:
                log_queue.put(f" W: No valid data block found in {filename_short}.")
                return None, None, None, None, None, None

            # Seek to the start of the data block and read entries
            f.seek(data_start_offset)
            bytes_needed = expected_vertices * ENTRY_SIZE
            if f.tell() + bytes_needed > file_size:
                log_queue.put(f" W:{filename_short} Not enough data for {expected_vertices} vertices after marker. Skipping.")
                return None, None, None, None, None, None

            for i in range(expected_vertices):
                entry_bytes = f.read(ENTRY_SIZE)
                if len(entry_bytes) < ENTRY_SIZE:
                    log_queue.put(f" E:{filename_short} Unexpected EOF at vertex {i+1}/{expected_vertices}.")
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
            return height_offsets, width, height, base_h, header_bytes, data_start_offset

    except FileNotFoundError:
        log_queue.put(f" Error: File not found {filepath}")
        return None, None, None, None, None, None
    except Exception as e:
        log_queue.put(f" Error processing {filepath}: {e}")
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

# --- Main Processing Function (called by GUI) ---
def generate_heightmap(config, log_queue, progress_queue):
    """
    Processes sector files based on config and updates GUI via queues.

    Args:
        config (dict): Dictionary containing processing parameters.
        log_queue (Queue): Queue to send log messages (strings) to the GUI.
        progress_queue (Queue): Queue to send progress updates (floats 0.0-1.0) to the GUI.
    """
    try:
        input_dir = config['input_dir']
        output_filename = config['output_filename']
        apply_boundary_smoothing = config['apply_boundary_smoothing']
        sector_overlap = config['sector_overlap']
        boundary_blend_size = config['boundary_blend_size']
        height_scale_factor = config['height_scale_factor'] # Make sure this is float

        if not input_dir or not os.path.isdir(input_dir):
            log_queue.put("Error: Input directory is not valid.")
            progress_queue.put(-1.0) # Indicate error
            return

        # Ensure scale factor is not zero
        if height_scale_factor == 0:
             log_queue.put("Error: Height Scale Factor cannot be zero.")
             progress_queue.put(-1.0)
             return
        # Ensure scale factor is float for division
        height_scale_factor = float(height_scale_factor)


        output_file_path = os.path.join(input_dir, output_filename) # Save in the input dir for simplicity

        log_queue.put("--- Starting Heightmap Generation ---")
        log_queue.put(f"Input Folder: {input_dir}")
        log_queue.put(f"Output File: {output_filename}")
        log_queue.put(f"Apply Boundary Smoothing: {apply_boundary_smoothing}")
        log_queue.put(f"Sector Overlap: {sector_overlap}")
        log_queue.put(f"Boundary Blend Size: {boundary_blend_size}")
        log_queue.put(f"Height Scale Factor: {height_scale_factor}")
        log_queue.put(f"Height = int16@Hdr[2] + (uint16@Entry[6] / {height_scale_factor})")
        log_queue.put("Normalization: Min-Max")
        progress_queue.put(0.0)

        all_sector_files = glob.glob(os.path.join(input_dir, '*.sector'))
        # Filter out the potential output file itself if it exists and ends with .sector
        sector_files = [f for f in all_sector_files if os.path.abspath(f) != os.path.abspath(output_file_path)]

        log_queue.put(f"Found {len(sector_files)} sector files to process.")
        if not sector_files:
            log_queue.put("Error: No .sector files found in the specified directory.")
            progress_queue.put(-1.0) # Indicate error
            return

        # --- Set Fixed Layout Size ---
        log_queue.put(f"Using fixed layout size: {LAYOUT_SECTOR_WIDTH} x {LAYOUT_SECTOR_HEIGHT}")

        # --- Pass 1: Read Header Base and Entry Offsets ---
        log_queue.put("--- Pass 1: Reading sector data ---")
        sectors_data = {}
        all_coords = []
        processed_count = 0
        total_files = len(sector_files)
        for i, filepath in enumerate(sector_files):
            filename = os.path.basename(filepath)
            sx, sy = parse_filename(filename)
            if sx is None or sy is None:
                log_queue.put(f"  W: Skipping '{filename}', could not parse coordinates.")
                continue

            # Use the fixed layout size
            result = process_sector_base_offset(filepath, LAYOUT_SECTOR_WIDTH, LAYOUT_SECTOR_HEIGHT, log_queue)

            if result[0] is not None:
                height_offsets, w, h, base_h, _, _ = result
                if w != LAYOUT_SECTOR_WIDTH or h != LAYOUT_SECTOR_HEIGHT:
                     log_queue.put(f" W: {filename} - dimensions mismatch ({w}x{h} vs expected {LAYOUT_SECTOR_WIDTH}x{LAYOUT_SECTOR_HEIGHT}). Skipping.")
                     continue

                # Detile the raw offsets
                detiled_offsets_uint16 = correct_detile_sector(height_offsets, w, h, log_queue)

                if detiled_offsets_uint16 is not None:
                    sectors_data[(sx, sy)] = {
                        'detiled_offsets': detiled_offsets_uint16,
                        'base_h': base_h,
                        'width': w,
                        'height': h
                    }
                    all_coords.append((sx, sy))
                    processed_count += 1
                else:
                    log_queue.put(f"  W: Failed to detile {filename}. Skipping.")
            # else: process_sector_base_offset already logged the error

            progress_queue.put(0.3 * (i + 1) / total_files) # Pass 1 is ~30%

        log_queue.put(f"Finished Pass 1. Processed {processed_count} valid sectors.")
        if processed_count == 0:
            log_queue.put("Error: No valid sector data could be processed.")
            progress_queue.put(-1.0) # Error
            return

        # --- Determine Map Bounds ---
        min_sx = min(c[0] for c in all_coords)
        max_sx = max(c[0] for c in all_coords)
        min_sy = min(c[1] for c in all_coords)
        max_sy = max(c[1] for c in all_coords)
        log_queue.put(f"Determined Sector Coordinate Range: X=[{min_sx}-{max_sx}], Y=[{min_sy}-{max_sy}]")

        # --- Calculate Final Map Size ---
        current_overlap = sector_overlap
        # Clamp overlap to valid range
        max_possible_overlap = min(LAYOUT_SECTOR_WIDTH, LAYOUT_SECTOR_HEIGHT) -1 # Need at least 1 non-overlapping pixel
        if not (0 <= current_overlap <= max_possible_overlap) :
             log_queue.put(f"Warning: Invalid SECTOR_OVERLAP ({current_overlap}). Clamping to range [0, {max_possible_overlap}].")
             current_overlap = max(0, min(current_overlap, max_possible_overlap))

        effective_sector_width = LAYOUT_SECTOR_WIDTH - current_overlap
        effective_sector_height = LAYOUT_SECTOR_HEIGHT - current_overlap

        # Handle cases where effective size becomes zero or negative due to large overlap
        if effective_sector_width <= 0 or effective_sector_height <= 0:
             log_queue.put(f"Error: Overlap ({current_overlap}) is too large for sector dimensions ({LAYOUT_SECTOR_WIDTH}x{LAYOUT_SECTOR_HEIGHT}). Effective size is non-positive.")
             progress_queue.put(-1.0)
             return


        num_sectors_x = max_sx - min_sx + 1
        num_sectors_y = max_sy - min_sy + 1

        # Calculate dimensions based on number of sectors and effective size + the last full sector dim
        final_width = (num_sectors_x - 1) * effective_sector_width + LAYOUT_SECTOR_WIDTH if num_sectors_x > 0 else 0
        final_height = (num_sectors_y - 1) * effective_sector_height + LAYOUT_SECTOR_HEIGHT if num_sectors_y > 0 else 0

        # Ensure non-negative dimensions
        final_width = max(0, final_width)
        final_height = max(0, final_height)

        # Handle single sector case
        if num_sectors_x <= 1: final_width = LAYOUT_SECTOR_WIDTH
        if num_sectors_y <= 1: final_height = LAYOUT_SECTOR_HEIGHT


        log_queue.put(f"Using Overlap={current_overlap}, Blend Size={boundary_blend_size}")
        log_queue.put(f"Calculated final map dimensions: {final_width} x {final_height}")

        if final_width <= 0 or final_height <= 0:
            log_queue.put("Error: Calculated final map dimensions are zero or negative.")
            progress_queue.put(-1.0) # Error
            return

        # --- Allocate Memory ---
        try:
            log_queue.put(f"Allocating final map arrays ({final_height} x {final_width})...")
            # Use float64 for sums to minimize precision loss during accumulation
            heightmap_sum_array = np.zeros((final_height, final_width), dtype=np.float64)
            weight_sum_array = np.zeros((final_height, final_width), dtype=np.float64)
            log_queue.put("Allocation successful.")
        except MemoryError:
            log_queue.put(f"Error: Not enough memory to allocate map arrays of size {final_height} x {final_width}.")
            progress_queue.put(-1.0) # Error
            return
        except Exception as e:
            log_queue.put(f"Error creating numpy arrays: {e}")
            progress_queue.put(-1.0) # Error
            return

        # --- Global Base Correction (Optional but good for consistency) ---
        # Compute a global average base from all sectors
        base_sum = 0.0
        base_count = 0
        for data in sectors_data.values():
            # Check if base_h is valid (might be None or skipped)
            if 'base_h' in data and data['base_h'] is not None:
                 base_sum += data['base_h']
                 base_count += 1

        if base_count > 0:
            global_base = base_sum / base_count
            log_queue.put(f"Global average base computed: {global_base:.2f} from {base_count} sectors.")
        else:
            global_base = 0.0
            log_queue.put("W: Could not compute global average base (no valid base_h found). Using 0.0.")

        # --- Pass 2: Calculate Height, Smooth (Optional), and Blend ---
        log_queue.put("--- Pass 2: Calculating heights, smoothing, and blending ---")
        placed_count = 0
        overall_min_h = float('inf')
        overall_max_h = float('-inf')
        first_sector_processed = False
        total_sectors_to_place = len(sectors_data)

        for i, ((sx, sy), data) in enumerate(sectors_data.items()):
            sector_w = data['width']
            sector_h = data['height']
            if 'detiled_offsets' not in data or data['detiled_offsets'] is None or \
               'base_h' not in data or data['base_h'] is None:
                log_queue.put(f" W: Skipping sector ({sx},{sy}) due to missing data.")
                continue

            # Calculate absolute height: Apply base correction and scale factor
            # Corrected: Apply base adjustment relative to the global average
            base_adjustment = data['base_h'] - global_base
            absolute_height_sector_float = (data['detiled_offsets'].astype(np.float64) / height_scale_factor) + base_adjustment

            # Apply tile boundary smoothing if enabled
            if apply_boundary_smoothing:
                absolute_height_sector_float = smooth_tile_boundaries(absolute_height_sector_float, TILE_WIDTH, TILE_HEIGHT, log_queue)

            # Update overall min/max *after* smoothing and scaling
            sector_min = np.nanmin(absolute_height_sector_float) # Use nanmin/nanmax
            sector_max = np.nanmax(absolute_height_sector_float)
            if np.isfinite(sector_min) and np.isfinite(sector_max):
                 overall_min_h = min(overall_min_h, sector_min)
                 overall_max_h = max(overall_max_h, sector_max)
                 first_sector_processed = True
            else:
                 log_queue.put(f" W: Sector ({sx},{sy}) resulted in non-finite min/max heights after processing.")


            # Create weight map for blending
            weight_map = create_weight_map(sector_h, sector_w, boundary_blend_size)

            # Calculate paste position
            grid_x = sx - min_sx
            grid_y = sy - min_sy
            paste_x_start = grid_x * effective_sector_width
            paste_y_start = grid_y * effective_sector_height
            paste_x_end = paste_x_start + sector_w
            paste_y_end = paste_y_start + sector_h

            # Clip paste coordinates and calculate source/target slices
            target_y_start_clipped = max(0, paste_y_start)
            target_y_end_clipped = min(final_height, paste_y_end)
            target_x_start_clipped = max(0, paste_x_start)
            target_x_end_clipped = min(final_width, paste_x_end)

            # Skip if clipped region is empty
            if target_y_start_clipped >= target_y_end_clipped or target_x_start_clipped >= target_x_end_clipped:
                log_queue.put(f" W: Sector ({sx},{sy}) is entirely outside the final map bounds after clipping. Skipping paste.")
                continue

            # Calculate source region based on clipping
            clip_top = target_y_start_clipped - paste_y_start
            clip_left = target_x_start_clipped - paste_x_start
            clipped_height = target_y_end_clipped - target_y_start_clipped
            clipped_width = target_x_end_clipped - target_x_start_clipped

            source_y_start = clip_top
            source_y_end = clip_top + clipped_height
            source_x_start = clip_left
            source_x_end = clip_left + clipped_width

            # Define slices
            target_slice = np.s_[target_y_start_clipped:target_y_end_clipped, target_x_start_clipped:target_x_end_clipped]
            source_slice = np.s_[source_y_start:source_y_end, source_x_start:source_x_end]

            # Safety check slice dimensions before applying
            if absolute_height_sector_float[source_slice].shape != (clipped_height, clipped_width) or \
               weight_map[source_slice].shape != (clipped_height, clipped_width):
                 log_queue.put(f"  E: Slice dimension mismatch for sector ({sx},{sy})! Target({clipped_height},{clipped_width}), SourceH{absolute_height_sector_float[source_slice].shape}, SourceW{weight_map[source_slice].shape}. Skipping paste.")
                 continue


            # Perform weighted addition using the calculated slices
            try:
                source_heights = absolute_height_sector_float[source_slice]
                source_weights = weight_map[source_slice]

                # Ensure we only add valid (non-NaN) heights
                valid_mask = np.isfinite(source_heights)

                heightmap_sum_array[target_slice][valid_mask] += source_heights[valid_mask] * source_weights[valid_mask]
                weight_sum_array[target_slice][valid_mask] += source_weights[valid_mask]
                placed_count += 1
            except IndexError as e:
                 log_queue.put(f"  E: IndexError during weighted addition for sector ({sx},{sy}): {e}. Slice mismatch likely.")
                 log_queue.put(f"     Target slice: {target_slice}")
                 log_queue.put(f"     Source slice: {source_slice}")
                 continue # Skip this sector placement
            except Exception as e:
                 log_queue.put(f"  E: Unexpected error during weighted addition for sector ({sx},{sy}): {e}")
                 log_queue.put(traceback.format_exc())
                 continue # Skip this sector placement

            progress_queue.put(0.3 + 0.6 * (i + 1) / total_sectors_to_place) # Pass 2 is ~60%

        log_queue.put(f"Finished Pass 2. Blended data from {placed_count} sector placements.")
        if placed_count == 0:
            log_queue.put("Error: No sectors were successfully placed onto the map.")
            progress_queue.put(-1.0) # Error
            return

        # --- Pass 3: Finalize Map (Divide by weights and Normalize) ---
        log_queue.put("--- Pass 3: Finalizing map and normalizing ---")

        # Check if any valid heights were found
        if not first_sector_processed or not np.isfinite(overall_min_h) or not np.isfinite(overall_max_h):
            log_queue.put(" W: No valid finite height range found across all sectors. Cannot determine normalization range accurately.")
             # Attempt to find min/max from the blended result *before* division
            temp_min = np.nanmin(heightmap_sum_array[weight_sum_array > 1e-9])
            temp_max = np.nanmax(heightmap_sum_array[weight_sum_array > 1e-9])
            if np.isfinite(temp_min) and np.isfinite(temp_max):
                 overall_min_h = temp_min
                 overall_max_h = temp_max
                 log_queue.put(f" W: Using range from blended sums: [{overall_min_h:.2f}, {overall_max_h:.2f}]")
            else:
                 log_queue.put(" W: Could not determine range even from sums. Defaulting to [0, 1] for normalization, output may be blank.")
                 overall_min_h = 0.0
                 overall_max_h = 1.0
        else:
             log_queue.put(f"Global Calculated Height Range (Pre-Normalization): [{overall_min_h:.2f}, {overall_max_h:.2f}]")

        # Divide sum by weights to get average height, handle division by zero
        log_queue.put("Calculating final heights (division by weights)...")
        final_heightmap_float = np.full((final_height, final_width), np.nan, dtype=np.float32) # Use float32 for final map
        valid_weights_mask = weight_sum_array > 1e-9 # Avoid division by zero or near-zero

        # Perform division only where weights are valid
        np.divide(heightmap_sum_array, weight_sum_array, out=final_heightmap_float, where=valid_weights_mask)

        # Normalization (Min-Max to 0-255)
        log_queue.put("Normalizing heightmap to 8-bit grayscale using Min-Max...")
        norm_min = overall_min_h
        norm_max = overall_max_h
        log_queue.put(f"Normalization Range Used: [{norm_min:.2f}, {norm_max:.2f}] -> [0, 255]")

        # Determine background color (use the minimum value of the range)
        # Pixels with no data (or NaN results) will get this value
        DEFAULT_BG_COLOR_FLOAT = norm_min
        final_heightmap_float[~valid_weights_mask] = DEFAULT_BG_COLOR_FLOAT
        final_heightmap_float[np.isnan(final_heightmap_float)] = DEFAULT_BG_COLOR_FLOAT # Handle NaNs from calculation

        # Calculate normalization range, handle case where min == max
        norm_range = norm_max - norm_min
        if norm_range <= 1e-9: # Use a small epsilon for float comparison
            log_queue.put(" W: Normalization range is zero or negative. Output will be flat.")
            # Output a map of all zeros (or mid-gray 128?) Let's use 0.
            heightmap_8bit = np.full((final_height, final_width), 0, dtype=np.uint8)
        else:
            # Apply normalization: (value - min) / range
            normalized_map = (final_heightmap_float - norm_min) / norm_range
            # Clip values strictly between 0.0 and 1.0
            np.clip(normalized_map, 0.0, 1.0, out=normalized_map)
            # Scale to 0-255 and convert to uint8
            heightmap_8bit = (normalized_map * 255.0).astype(np.uint8)

        progress_queue.put(0.95) # Almost done

        # --- Save the Final Image ---
        try:
            log_queue.put(f"Saving final heightmap to {output_file_path}...")
            img = Image.fromarray(heightmap_8bit, mode='L') # 'L' mode for 8-bit grayscale
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
    folder_path = filedialog.askdirectory(title="Select Folder Containing .sector Files")
    root.destroy() # Close the hidden tkinter window
    return folder_path

# --- ImGui Application ---

# Global state for the GUI
gui_state = {
    "input_dir": "",
    "output_filename": "heightmap_output.png",
    "apply_boundary_smoothing": True,
    "sector_overlap": 1,
    "boundary_blend_size": 10,
    "height_scale_factor": 1.0,
    "log_messages": ["Welcome! Select a folder and click Generate."],
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
             gui_state["is_processing"] = False


def run_gui_imgui_bundle():
    """Main function to run the GUI using imgui-bundle."""
    def gui_loop():
        """This function is called every frame"""
        global gui_state

        # Check for updates from processing thread
        if gui_state["is_processing"]:
            update_log_and_progress()
            # Check if thread finished
            if gui_state["processing_thread"] and not gui_state["processing_thread"].is_alive():
                gui_state["is_processing"] = False
                gui_state["processing_thread"] = None
                # Final update in case messages arrived after last check
                update_log_and_progress()
                if gui_state["progress"] < 1.0 and gui_state["progress"] >= 0: # Didn't finish cleanly?
                     gui_state["log_messages"].append("Warning: Processing ended prematurely.")


        # Build the ImGui window
        imgui.set_next_window_size(ImVec2(600, 550), cond=imgui.Cond_.first_use_ever) #imgui_bundle API
        imgui.begin("Sector Heightmap Generator")

        # --- Input Folder ---
        imgui.push_item_width(-200) # Make text input fill width minus button
        changed, gui_state["input_dir"] = imgui.input_text("##InputFolder", gui_state["input_dir"], 2048)
        imgui.pop_item_width()
        imgui.same_line()
        if imgui.button("Select Input Folder"):
            selected = select_folder_dialog()
            if selected:
                gui_state["input_dir"] = selected

        # --- Output Filename ---
        imgui.push_item_width(-150) # Adjust width as needed
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

        # Use InputInt for integer values, InputFloat for float
        imgui.push_item_width(100) # Set width for next items
        changed, gui_state["sector_overlap"] = imgui.input_int("Sector Overlap", gui_state["sector_overlap"])
        # Basic validation / clamping
        gui_state["sector_overlap"] = max(0, gui_state["sector_overlap"]) # Ensure non-negative
        if imgui.is_item_hovered():
            imgui.set_tooltip("Number of pixels sectors overlap when stitching (0 to SectorDim-1).")


        changed, gui_state["boundary_blend_size"] = imgui.input_int("Boundary Blend Size", gui_state["boundary_blend_size"])
        gui_state["boundary_blend_size"] = max(0, gui_state["boundary_blend_size"]) # Ensure non-negative
        if imgui.is_item_hovered():
            imgui.set_tooltip("Size of the feathered edge (in pixels) for blending between overlapping sectors.")

        changed, gui_state["height_scale_factor"] = imgui.input_float("Height Scale Factor", gui_state["height_scale_factor"], 0.1, 1.0, "%.2f")
        # Prevent division by zero - enforce a minimum positive value
        if gui_state["height_scale_factor"] <= 0:
             gui_state["height_scale_factor"] = 0.01 # Set a small positive default
        if imgui.is_item_hovered():
            imgui.set_tooltip("Divisor for the raw uint16 height offset value. Height = Base + Offset / Factor.")

        imgui.pop_item_width()


        imgui.separator()

# --- Action Button & Progress ---

        # Evaluate the condition ONCE per frame before the button
        button_needs_disabling = gui_state["is_processing"]

        # Use the public API imgui.begin_disabled() / imgui.end_disabled()
        if button_needs_disabling:
            # Begin the disabled state block
            imgui.begin_disabled(True) # Pass True to disable
            # Push the style modification *inside* the disabled block
            imgui.push_style_var(imgui.StyleVar_.alpha, imgui.get_style().alpha * 0.5)

        # Render the button widget.
        # It will be visually disabled and non-interactive if begin_disabled(True) was called.
        button_clicked = imgui.button("Generate Heightmap")

        # End the disabled state block *if it was started*
        if button_needs_disabling:
            # Pop the style *before* ending the disabled block (LIFO)
            imgui.pop_style_var()
            # End the disabled state block
            imgui.end_disabled()

        # Now, handle the button click *after* the stack/state is potentially restored
        # Note: button_clicked should be False if the button was disabled.
        if button_clicked:
            # Only start if not already processing
            if not gui_state["is_processing"]:
                # Start processing in a separate thread
                gui_state["is_processing"] = True # State change happens here
                gui_state["progress"] = 0.0
                gui_state["log_messages"] = ["Starting processing..."] # Clear log

                # Prepare config dict
                current_config = {
                    'input_dir': gui_state["input_dir"],
                    'output_filename': gui_state["output_filename"],
                    'apply_boundary_smoothing': gui_state["apply_boundary_smoothing"],
                    'sector_overlap': gui_state["sector_overlap"],
                    'boundary_blend_size': gui_state["boundary_blend_size"],
                    'height_scale_factor': gui_state["height_scale_factor"],
                   }

                # Clear queues before starting
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
        imgui.progress_bar(gui_state["progress"], size_arg=ImVec2(-1, 0)) # Auto width

        # --- End of Action Button & Progress Section ---

        imgui.separator()

        # --- Log Area ---
        imgui.text("Log:")
        log_height = imgui.get_content_region_avail().y - 5 # Fill remaining vertical space
        # Use ImVec2 for size and child_flags for border with imgui-bundle
        # Make sure flag is singular 'border'
        imgui.begin_child("Log", size=ImVec2(-1, log_height), child_flags=imgui.ChildFlags_.borders) # Auto width
        imgui.text_unformatted("\n".join(gui_state["log_messages"]))
        # Auto-scroll
        if imgui.get_scroll_y() >= imgui.get_scroll_max_y():
             imgui.set_scroll_here_y(1.0)
        imgui.end_child()

        imgui.end() # End main window

    # Use hello_imgui runner
    runner_params = hello_imgui.RunnerParams()
    runner_params.app_window_params.window_title = "Sector Heightmap Generator"
    runner_params.imgui_window_params.show_menu_bar = False
    # Set the GUI function
    runner_params.callbacks.show_gui = gui_loop

    # Customize other params if needed
    # runner_params.app_window_params.window_size = (650, 600)

    hello_imgui.run(runner_params)


def run_gui_pyimgui():
    """Main function to run the GUI using older pyimgui and glfw."""
    # --- GLFW Window Setup ---
    if not glfw.init():
        print("Could not initialize OpenGL context")
        sys.exit(1)

    # OS X supports only forward-compatible core profiles from 3.2
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, gl.GL_TRUE)

    window = glfw.create_window(600, 550, "Sector Heightmap Generator", None, None)
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

        # --- GUI Definition (Adopt similar safe structure for pyimgui if needed) ---
        global gui_state

        # Check for updates from processing thread
        if gui_state["is_processing"]:
            update_log_and_progress()
            # Check if thread finished
            if gui_state["processing_thread"] and not gui_state["processing_thread"].is_alive():
                gui_state["is_processing"] = False
                gui_state["processing_thread"] = None
                update_log_and_progress() # Final update
                if gui_state["progress"] < 1.0 and gui_state["progress"] >= 0:
                     gui_state["log_messages"].append("Warning: Processing ended prematurely.")


        # Build the ImGui window
        imgui.set_next_window_size(600, 550, condition=imgui.FIRST_USE_EVER)
        imgui.begin("Sector Heightmap Generator")

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

        imgui.push_item_width(100)
        # Using lists for input_int/float in pyimgui
        overlap_val = [gui_state["sector_overlap"]]
        changed = imgui.input_int("Sector Overlap", overlap_val)
        if changed: gui_state["sector_overlap"] = max(0, overlap_val[0])

        blend_val = [gui_state["boundary_blend_size"]]
        changed = imgui.input_int("Boundary Blend Size", blend_val)
        if changed: gui_state["boundary_blend_size"] = max(0, blend_val[0])

        scale_val = [gui_state["height_scale_factor"]]
        changed = imgui.input_float("Height Scale Factor", scale_val, 0.1, 1.0, "%.2f")
        if changed:
            gui_state["height_scale_factor"] = scale_val[0]
            if gui_state["height_scale_factor"] <= 0: gui_state["height_scale_factor"] = 0.01
        imgui.pop_item_width()

        # Tooltips might need adjustment for pyimgui
        # Example: Check hover after the InputFloat call
        # if imgui.is_item_hovered(): imgui.set_tooltip("Divisor...")


        imgui.separator()

# --- Action Button & Progress ---

        # Evaluate the condition ONCE per frame before the button
        button_needs_disabling = gui_state["is_processing"]

        # Use the public API imgui.begin_disabled() / imgui.end_disabled()
        if button_needs_disabling:
            # Begin the disabled state block
            imgui.begin_disabled(True) # Pass True to disable
            # Push the style modification *inside* the disabled block
            imgui.push_style_var(imgui.StyleVar_.alpha, imgui.get_style().alpha * 0.5)

        # Render the button widget.
        # It will be visually disabled and non-interactive if begin_disabled(True) was called.
        button_clicked = imgui.button("Generate Heightmap")

        # End the disabled state block *if it was started*
        if button_needs_disabling:
            # Pop the style *before* ending the disabled block (LIFO)
            imgui.pop_style_var()
            # End the disabled state block
            imgui.end_disabled()

        # Now, handle the button click *after* the stack/state is potentially restored
        # Note: button_clicked should be False if the button was disabled.
        if button_clicked:
            # Only start if not already processing
            if not gui_state["is_processing"]:
                # Start processing in a separate thread
                gui_state["is_processing"] = True # State change happens here
                gui_state["progress"] = 0.0
                gui_state["log_messages"] = ["Starting processing..."] # Clear log

                # Prepare config dict
                current_config = {
                    'input_dir': gui_state["input_dir"],
                    'output_filename': gui_state["output_filename"],
                    'apply_boundary_smoothing': gui_state["apply_boundary_smoothing"],
                    'sector_overlap': gui_state["sector_overlap"],
                    'boundary_blend_size': gui_state["boundary_blend_size"],
                    'height_scale_factor': gui_state["height_scale_factor"],
                   }

                # Clear queues before starting
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
        imgui.progress_bar(gui_state["progress"], size_arg=ImVec2(-1, 0)) # Auto width

        # --- End of Action Button & Progress Section ---

        imgui.separator()

        # --- Log Area ---
        imgui.text("Log:")
        log_height = imgui.get_content_region_available()[1] - 5 # Get Y avail
        # pyimgui often uses width, height, border args
        imgui.begin_child("Log", -1, log_height, border=True) # w=-1 means auto
        imgui.text_unformatted("\n".join(gui_state["log_messages"]))
        if imgui.get_scroll_y() >= imgui.get_scroll_max_y():
             imgui.set_scroll_here(1.0) # pyimgui might use set_scroll_here() or set_scroll_here_y()
        imgui.end_child()

        imgui.end() # End main window

        # --- Rendering ---
        gl.glClearColor(0.1, 0.1, 0.1, 1) # Background color
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