import struct
import math
import os
import glob
import re
import numpy as np
from PIL import Image
import sys
import traceback

# --- Configuration: File Structure & Data Layout ---
ENTRY_SIZE = 8                  # Size of each data entry in bytes
HEIGHT_OFFSET_IN_ENTRY = 6      # Byte offset within an entry where the height offset (uint16) is stored
HEIGHT_FORMAT = '<H'            # Struct format for height offset: little-endian unsigned short
EXPECTED_ZERO_BYTES_OFFSETS = [2, 3, 4, 5] # Offsets within an entry expected to be zero, used for validation

START_MARKER = b'\x00\x00\x01\x00' # Byte sequence indicating the start of a relevant data block
VARIABLE_DATA_AFTER_MARKER_SIZE = 4 # Size of variable data immediately following the marker
INTERMEDIATE_HEADER_SIZE = 8    # Size of the header section between the variable data and the main data entries
BYTES_TO_SKIP_AFTER_MARKER = VARIABLE_DATA_AFTER_MARKER_SIZE + INTERMEDIATE_HEADER_SIZE # Total bytes to skip after marker to reach data entries
N_CONTEXT_CHECK = 5             # Number of initial entries to check for the expected zero byte pattern

BASE_H_OFFSET_IN_HEADER = 2     # Byte offset within the intermediate header for the base height value
BASE_H_FORMAT = '<h'            # Struct format for base height: little-endian signed short
# Note: While base_h is read from the header for potential debugging, it is *ignored* in the final height calculation.
HEIGHT_SCALE_FACTOR = 1.0       # Scaling factor for height values (kept at 1.0, meaning no scaling)

# --- Configuration: Tiling within Sectors ---
TILE_WIDTH = 7                  # Width of the tiles used for storing data within a sector
TILE_HEIGHT = 7                 # Height of the tiles used for storing data within a sector
POINTS_PER_TILE = TILE_WIDTH * TILE_HEIGHT # Total data points per tile

# --- Configuration: Sector Dimensions & Layout ---
LAYOUT_SECTOR_WIDTH = 224       # Expected width of each sector's data grid
LAYOUT_SECTOR_HEIGHT = 224      # Expected height of each sector's data grid

# --- Configuration: Stitching & Blending ---
SECTOR_OVERLAP = 1              # Number of pixels sectors overlap when stitched (used to calculate placement)
BOUNDARY_BLEND_SIZE = 10        # Width of the feathered edge blend region between adjacent sectors
APPLY_BOUNDARY_SMOOTHING = True # Flag to enable/disable smoothing filter at internal tile boundaries within each sector

# --- Configuration: Output ---
output_suffix = "HEIGHTMAP_IgnoreBase_GlobalMinMax" # Suffix for the output filename
if APPLY_BOUNDARY_SMOOTHING:
    output_suffix += "_smooth"  # Append indicator if smoothing is enabled
OUTPUT_FILENAME = f"heightmap_{output_suffix}.png" # Final output image filename

ENABLE_DIAGNOSTICS = False      # Flag for enabling diagnostic outputs (not used in this version)
DIAGNOSTICS_DIR = "diagnostics" # Directory for diagnostic outputs (not used in this version)

# --- Configuration: Normalization ---
# These percentile values are effectively ignored as a single global min-max normalization is performed at the end.
CONTRAST_PERCENTILE_LOW = 0.0
CONTRAST_PERCENTILE_HIGH = 100.0

def parse_filename(filename):
    """
    Extracts sector coordinates (X, Y) from a filename like 'XXX_YYY_*.sector'.
    Returns (int(X), int(Y)) or (None, None) if the pattern doesn't match.
    """
    match = re.match(r'(\d{3})_(\d{3})_.*\.sector', os.path.basename(filename))
    if match:
        return int(match.group(1)), int(match.group(2))
    else:
        return None, None

def create_weight_map(sector_h, sector_w, blend_size):
    """
    Generates a 2D weight map for blending sectors.
    Weights are 1.0 in the center and fall off towards the edges using an exponential ramp
    within the specified blend_size boundary region. This allows smooth transitions when
    averaging overlapping sectors.
    Returns a numpy array (float32) of shape (sector_h, sector_w).
    """
    weight_map = np.ones((sector_h, sector_w), dtype=np.float32)
    if blend_size <= 0:
        return weight_map # No blending if size is non-positive

    blend_pixels = min(blend_size, sector_w // 2, sector_h // 2)
    if blend_pixels <= 0:
        return weight_map # No blending if calculated pixels is non-positive

    # Create the falloff curve (exponential)
    ramp = np.exp(-5.0 * (1.0 - np.linspace(0.0, 1.0, blend_pixels + 1)[1:]))

    center_start = blend_pixels
    center_end_y = sector_h - blend_pixels
    center_end_x = sector_w - blend_pixels

    # Apply ramp to top/bottom edges (excluding corners)
    for i in range(blend_pixels):
        weight = ramp[i]
        if center_start < center_end_x: # Ensure valid slice
            weight_map[i, center_start:center_end_x] = weight
            weight_map[sector_h - 1 - i, center_start:center_end_x] = weight
    # Apply ramp to left/right edges (excluding corners)
    for i in range(blend_pixels):
        weight = ramp[i]
        if center_start < center_end_y: # Ensure valid slice
            weight_map[center_start:center_end_y, i] = weight
            weight_map[center_start:center_end_y, sector_w - 1 - i] = weight

    # Apply ramp to corners, taking the minimum weight (strongest falloff)
    for r in range(blend_pixels):
        for c in range(blend_pixels):
            weight_corner = ramp[min(r, c)]
            weight_map[r, c] = min(weight_map[r, c], weight_corner) # Top-left
            weight_map[r, sector_w - 1 - c] = min(weight_map[r, sector_w - 1 - c], weight_corner) # Top-right
            weight_map[sector_h - 1 - r, c] = min(weight_map[sector_h - 1 - r, c], weight_corner) # Bottom-left
            weight_map[sector_h - 1 - r, sector_w - 1 - c] = min(weight_map[sector_h - 1 - r, sector_w - 1 - c], weight_corner) # Bottom-right

    # Ensure the central area remains fully weighted at 1.0
    if center_start < center_end_y and center_start < center_end_x:
        weight_map[center_start:center_end_y, center_start:center_end_x] = 1.0

    return weight_map

def process_sector_base_offset(filepath, expected_width, expected_height):
    """
    Reads a .sector binary file to extract height offset data.
    It searches for a specific START_MARKER, validates the subsequent data structure briefly,
    reads (but ignores for final calculation) a base height from an intermediate header,
    and then reads the sequence of uint16 height offsets from the main data entries.

    Args:
        filepath (str): Path to the .sector file.
        expected_width (int): Expected width of the sector data grid.
        expected_height (int): Expected height of the sector data grid.

    Returns:
        tuple: Contains:
            - list[int] or None: List of raw uint16 height offsets if successful, else None.
            - int or None: Expected width, or None on failure.
            - int or None: Expected height, or None on failure.
            - int or None: Base height value read from header (signed short), or None on failure.
            - bytes or None: Raw bytes of the intermediate header, or None on failure.
            - int or None: Starting file offset of the height data entries, or None on failure.
    """
    height_offsets = []
    base_h = 0
    header_bytes = None
    data_start_offset = -1
    expected_vertices = expected_width * expected_height

    try:
        with open(filepath, 'rb') as f:
            content_to_scan = f.read()
            file_size = len(content_to_scan)
            if not content_to_scan:
                print(f" W:{os.path.basename(filepath)} File is empty.")
                return None, None, None, None, None, None

            search_start_offset = 0
            found_valid_context = False
            # --- Find the start marker and validate context ---
            while search_start_offset < file_size:
                current_marker_offset = content_to_scan.find(START_MARKER, search_start_offset)
                if current_marker_offset == -1:
                    break # Marker not found

                potential_data_start = current_marker_offset + len(START_MARKER) + BYTES_TO_SKIP_AFTER_MARKER
                potential_header_start = current_marker_offset + len(START_MARKER)

                # Check if there's enough data for context check
                if potential_data_start + (N_CONTEXT_CHECK * ENTRY_SIZE) <= file_size:
                    context_valid = True
                    # Check the first few entries for the expected zero-byte pattern
                    for i in range(N_CONTEXT_CHECK):
                        entry_offset = potential_data_start + (i * ENTRY_SIZE)
                        entry_bytes = content_to_scan[entry_offset : entry_offset + ENTRY_SIZE]
                        if (len(entry_bytes) < ENTRY_SIZE or
                            not all(entry_bytes[z] == 0x00 for z in EXPECTED_ZERO_BYTES_OFFSETS)):
                            context_valid = False
                            break # Pattern mismatch

                    if context_valid:
                        # Context seems valid, attempt to read the base height from the header
                        try:
                            header_bytes = content_to_scan[
                                potential_header_start : potential_header_start + BYTES_TO_SKIP_AFTER_MARKER
                            ]
                            if len(header_bytes) >= BASE_H_OFFSET_IN_HEADER + struct.calcsize(BASE_H_FORMAT):
                                base_h, = struct.unpack_from(BASE_H_FORMAT, header_bytes, BASE_H_OFFSET_IN_HEADER)
                            else: # Header too short
                                base_h = 0
                                header_bytes = None # Indicate header read failure
                        except Exception as e:
                            base_h = 0 # Reset on error
                            header_bytes = None # Indicate header read failure
                            print(f" W:{os.path.basename(filepath)} Error reading header base H: {e}")

                        if header_bytes is not None: # Successfully read header (even if base_h might be zero)
                            data_start_offset = potential_data_start
                            found_valid_context = True
                            break # Found a valid block, exit search loop

                # Move search position forward if context check failed or not enough data
                search_start_offset = current_marker_offset + 1

            if not found_valid_context:
                print(f" W: No valid data block found in {os.path.basename(filepath)} after scanning.")
                return None, None, None, None, None, None

            # --- Read the height offset entries ---
            f.seek(data_start_offset)
            bytes_needed = expected_vertices * ENTRY_SIZE
            if f.tell() + bytes_needed > file_size:
                print(f" W:{os.path.basename(filepath)} Not enough data remaining for expected vertices. Skipping.")
                return None, None, None, None, None, None

            for _ in range(expected_vertices):
                entry_bytes = f.read(ENTRY_SIZE)
                if len(entry_bytes) < ENTRY_SIZE:
                    print(f" E:{os.path.basename(filepath)} Unexpected EOF while reading data entries.")
                    return None, None, None, None, None, None
                try:
                    # Unpack the uint16 height offset from the correct position within the entry
                    h_val, = struct.unpack_from(HEIGHT_FORMAT, entry_bytes, HEIGHT_OFFSET_IN_ENTRY)
                    height_offsets.append(h_val)
                except struct.error:
                    print(f" W:{os.path.basename(filepath)} Struct error unpacking uint16 offset at entry index {len(height_offsets)}.")
                    return None, None, None, None, None, None

            if len(height_offsets) != expected_vertices:
                print(f" E:{os.path.basename(filepath)} Read count mismatch. Expected {expected_vertices}, got {len(height_offsets)}.")
                return None, None, None, None, None, None

            return (height_offsets,
                    expected_width,
                    expected_height,
                    base_h,           # Return base_h even though it's ignored later
                    header_bytes,
                    data_start_offset)

    except FileNotFoundError:
        print(f" Error: File not found {filepath}")
        return None, None, None, None, None, None
    except Exception as e:
        print(f" Error processing {filepath}: {e}")
        traceback.print_exc()
        return None, None, None, None, None, None

def correct_detile_sector(sector_offset_values, sector_width, sector_height):
    """
    Rearranges the height offset values from their tiled storage order
    into a standard 2D row-major grid representation. The input `sector_offset_values`
    is assumed to be a flat list where data for tile (0,0) comes first, then (0,1), etc.,
    and within each tile, data is stored row by row.

    Args:
        sector_offset_values (list[int]): Flat list of uint16 height offsets in tiled order.
        sector_width (int): Width of the sector grid.
        sector_height (int): Height of the sector grid.

    Returns:
        numpy.ndarray (uint16) or None: A 2D numpy array of shape (sector_height, sector_width)
                                         with detiled height offsets, or None if input size is wrong.
    """
    detiled_heights = np.zeros((sector_height, sector_width), dtype=np.uint16)
    tile_width = TILE_WIDTH
    tile_height = TILE_HEIGHT

    # Calculate how many tiles fit horizontally and vertically
    tiles_per_row = sector_width // tile_width
    tiles_per_col = sector_height // tile_height # Not directly used in index calc but good for understanding

    expected_vertices = sector_width * sector_height
    if len(sector_offset_values) != expected_vertices:
        print(f"  W: Detile input size mismatch. Expected {expected_vertices}, got {len(sector_offset_values)}.")
        return None

    # Iterate through each pixel of the output grid
    for output_y in range(sector_height):
        for output_x in range(sector_width):
            # Determine which tile this output pixel belongs to
            tile_row = output_y // tile_height
            tile_col = output_x // tile_width

            # Determine the pixel's coordinates *within* its tile
            local_y = output_y % tile_height
            local_x = output_x % tile_width

            # Calculate the linear index of the tile
            tile_index = tile_row * tiles_per_row + tile_col
            # Calculate the linear index of the pixel *within* its tile (row-major)
            position_in_tile = local_y * tile_width + local_x

            # Calculate the final flat index into the original tiled data list
            flat_index = tile_index * POINTS_PER_TILE + position_in_tile

            if 0 <= flat_index < len(sector_offset_values):
                detiled_heights[output_y, output_x] = sector_offset_values[flat_index]
            else:
                # This should ideally not happen if input size is correct and logic is sound
                print(f"  W: Calculated flat index {flat_index} is out of bounds for input length {len(sector_offset_values)} at ({output_x},{output_y})")
                # Assign a default value (e.g., 0) or handle as an error
                detiled_heights[output_y, output_x] = 0 # Or some other indicator

    return detiled_heights

def smooth_tile_boundaries(heightmap, tile_width=7, tile_height=7):
    """
    Applies a simple averaging filter specifically at the boundaries *between* tiles
    within a single sector's heightmap. This helps reduce visual seams caused by the
    tiling process itself. It averages horizontal, vertical, and corner boundary pixels
    with their direct neighbors perpendicular to the boundary.

    Args:
        heightmap (numpy.ndarray): The 2D sector heightmap (after detiling).
        tile_width (int): The width of the internal tiles.
        tile_height (int): The height of the internal tiles.

    Returns:
        numpy.ndarray (float32): The smoothed heightmap.
    """
    h, w = heightmap.shape
    smoothed = heightmap.copy().astype(np.float32) # Work with floats for averaging

    # Smooth Vertical tile boundaries (average with horizontal neighbors)
    # Iterate columns that are right edges of tiles (excluding the last image edge)
    for x in range(tile_width - 1, w - 1, tile_width):
        for y in range(h): # Apply to all rows for this column
              # Average the pixel to the left and the pixel to the right of the boundary
              smoothed[y, x] = (float(heightmap[y, x]) + float(heightmap[y, x + 1])) / 2.0 # Average boundary pixel and pixel to its right
              # Also average the pixel *after* the boundary using its neighbours
              smoothed[y, x+1] = (float(heightmap[y, x]) + float(heightmap[y, x + 1])) / 2.0

    # Smooth Horizontal tile boundaries (average with vertical neighbors)
    # Iterate rows that are bottom edges of tiles (excluding the last image edge)
    for y in range(tile_height - 1, h - 1, tile_height):
         for x in range(w): # Apply to all columns for this row
              # Average the pixel above and the pixel below the boundary
              smoothed[y, x] = (float(heightmap[y, x]) + float(heightmap[y + 1, x])) / 2.0 # Average boundary pixel and pixel below it
              # Also average the pixel *after* the boundary
              smoothed[y+1, x] = (float(heightmap[y, x]) + float(heightmap[y + 1, x])) / 2.0

    # Special handling for corner points where boundaries intersect (average 4 diagonal neighbors)
    # Iterate through the corner pixels formed by tile intersections
    for y in range(tile_height - 1, h - 1, tile_height):
        for x in range(tile_width - 1, w - 1, tile_width):
             # Average the four diagonal neighbors around the intersection corner
             smoothed[y, x] = (float(heightmap[y - 1, x - 1]) + # Top-left diag
                               float(heightmap[y - 1, x + 1]) + # Top-right diag
                               float(heightmap[y + 1, x - 1]) + # Bottom-left diag
                               float(heightmap[y + 1, x + 1])) / 4.0 # Bottom-right diag

    return smoothed

def main():
    """
    Main execution function:
    1. Finds all .sector files in the script's directory.
    2. Processes each sector file: reads height offsets, detiles them. (Pass 1)
    3. Calculates the total dimensions needed for the stitched map based on sector coordinates and overlap.
    4. Allocates large numpy arrays for the final summed heights and weights.
    5. Stitches sectors together: Places each sector's (optionally smoothed) height data onto the
       large arrays, weighted by the blending map. Tracks the global min/max height offset values. (Pass 2)
    6. Normalizes the final heightmap: Divides summed heights by summed weights, fills gaps,
       and normalizes the result using the overall min/max range found in Pass 2. (Pass 3)
    7. Saves the final normalized heightmap as a grayscale PNG image.
    """
    try:
        # Determine the directory where the script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        # Fallback for environments where __file__ is not defined (e.g., interactive)
        script_dir = os.path.abspath('.')
        print(f"Warning: Could not determine script directory via __file__, using cwd: {script_dir}")

    print("--- Sector Heightmap Stitcher ---")
    print("Mode: Reading uint16 offsets (@6), Ignoring base_h (@2)")
    print(f"Normalization: Single Global Min/Max across entire map.")
    print(f"Apply Tile Boundary Smoothing: {APPLY_BOUNDARY_SMOOTHING}")
    print(f"Sector Overlap for Stitching: {SECTOR_OVERLAP}")
    print(f"Boundary Blend Size: {BOUNDARY_BLEND_SIZE}")

    # --- Find and Filter Sector Files ---
    all_sector_files = glob.glob(os.path.join(script_dir, '*.sector'))
    output_file_path = os.path.join(script_dir, OUTPUT_FILENAME)
    diag_path_prefix = os.path.join(script_dir, DIAGNOSTICS_DIR) + os.path.sep # Ensure trailing slash

    # Filter out the potential output file and files in diagnostics dir
    sector_files = [
        f for f in all_sector_files
        if os.path.abspath(f) != os.path.abspath(output_file_path)
        and not os.path.abspath(f).startswith(os.path.abspath(diag_path_prefix))
    ]

    print(f"\nFound {len(sector_files)} potential sector files to process.")
    if not sector_files:
        print("Error: No sector files found in the script directory. Exiting.")
        return

    # Define fixed sector layout dimensions (assuming all sectors conform)
    print(f"\nAssuming sector layout: {LAYOUT_SECTOR_WIDTH} x {LAYOUT_SECTOR_HEIGHT} pixels")

    # --- Pass 1: Read sector data (offsets only) and detile ---
    print("\n--- Pass 1: Reading and Detiling Sector Data ---")
    sectors_data = {} # Dictionary to hold processed data: {(sx, sy): {data}}
    all_coords = []   # List to track coordinates of successfully processed sectors
    processed_count = 0
    for filepath in sector_files:
        filename = os.path.basename(filepath)
        sx, sy = parse_filename(filename)
        if sx is None or sy is None:
            print(f"  W: Skipping '{filename}': Cannot parse coordinates from filename.")
            continue

        # Process the file to get raw height offsets and other info
        result = process_sector_base_offset(filepath, LAYOUT_SECTOR_WIDTH, LAYOUT_SECTOR_HEIGHT)

        if result is not None and result[0] is not None:
            height_offsets, w, h, base_h_ignored, header_bytes_ignored, offset_ignored = result

            # Rearrange the flat list of offsets into a 2D grid
            detiled_offsets_uint16 = correct_detile_sector(height_offsets, w, h)

            if detiled_offsets_uint16 is not None:
                # Store the successfully detiled data
                sectors_data[(sx, sy)] = {
                    'detiled_offsets': detiled_offsets_uint16, # This is the uint16 2D numpy array
                    'width': w,
                    'height': h
                    # Base height is intentionally not stored or used further
                }
                all_coords.append((sx, sy))
                processed_count += 1
                # print(f"  OK: Processed and detiled {filename}") # Optional: verbose success
            else:
                print(f"  W: Failed to detile {filename}. Skipping.")
        else:
            # process_sector_base_offset would have printed a warning/error
            print(f"  W: Failed to process raw data from {filename}. Skipping.")

    print(f"\nFinished Pass 1. Successfully processed {processed_count} valid sectors.")
    if processed_count == 0:
        print("Error: No valid sector data could be processed. Exiting.")
        return

    # --- Calculate Final Map Dimensions ---
    min_sx = min(c[0] for c in all_coords)
    max_sx = max(c[0] for c in all_coords)
    min_sy = min(c[1] for c in all_coords)
    max_sy = max(c[1] for c in all_coords)

    print(f"Sector coordinate grid range: X=[{min_sx}...{max_sx}], Y=[{min_sy}...{max_sy}]")

    # Validate and clamp overlap value if necessary
    current_overlap = SECTOR_OVERLAP
    max_possible_overlap = min(LAYOUT_SECTOR_WIDTH, LAYOUT_SECTOR_HEIGHT) - 1 # Need at least 1 pixel non-overlap
    if not (0 <= current_overlap <= max_possible_overlap):
         print(f"Warning: SECTOR_OVERLAP ({current_overlap}) is invalid. Clamping to range [0, {max_possible_overlap}].")
         current_overlap = max(0, min(current_overlap, max_possible_overlap))

    # Calculate effective size (how much each sector contributes horizontally/vertically in the grid)
    effective_sector_width = LAYOUT_SECTOR_WIDTH - current_overlap
    effective_sector_height = LAYOUT_SECTOR_HEIGHT - current_overlap

    # Calculate dimensions of the final stitched map
    num_sectors_x = max_sx - min_sx + 1
    num_sectors_y = max_sy - min_sy + 1

    # Width = (N-1)*effective_width + full_width_of_last_sector
    final_width = (num_sectors_x - 1) * effective_sector_width + LAYOUT_SECTOR_WIDTH if num_sectors_x > 0 else 0
    # Height = (N-1)*effective_height + full_height_of_last_sector
    final_height = (num_sectors_y - 1) * effective_sector_height + LAYOUT_SECTOR_HEIGHT if num_sectors_y > 0 else 0

    # Handle cases with only one row or column of sectors
    if num_sectors_x <= 1: final_width = LAYOUT_SECTOR_WIDTH
    if num_sectors_y <= 1: final_height = LAYOUT_SECTOR_HEIGHT

    print(f"\nCalculated final map dimensions: {final_width} x {final_height}")
    if final_width <= 0 or final_height <= 0:
        print("Error: Calculated final dimensions are invalid (<= 0). Exiting.")
        return

    # --- Allocate Memory for Final Map ---
    try:
        print(f"Allocating final map arrays ({final_height} x {final_width}). Using float64 for accumulation...")
        # Use float64 for sums to avoid potential precision issues with many additions
        heightmap_sum_array = np.zeros((final_height, final_width), dtype=np.float64)
        weight_sum_array = np.zeros((final_height, final_width), dtype=np.float64)
        print("Allocation successful.")
    except MemoryError:
        print(f"Error: Not enough memory to allocate arrays of size {final_height}x{final_width}. Try reducing sectors or system load.")
        return
    except Exception as e:
        print(f"Error creating final numpy arrays: {e}")
        return

    # --- Pass 2: Stitch Sectors with Blending ---
    print("\n--- Pass 2: Stitching Sectors ---")
    placed_count = 0
    overall_min_h = float('inf') # Initialize overall min/max using the raw offset values
    overall_max_h = float('-inf')
    first_sector_processed = False # Flag to ensure min/max are updated at least once

    for (sx, sy), data in sectors_data.items():
        sector_w = data['width']
        sector_h = data['height']
        detiled_offsets = data['detiled_offsets'] # This is the uint16 2D array

        # Convert raw uint16 offsets directly to float32 for processing. Base height is ignored.
        absolute_height_sector = detiled_offsets.astype(np.float32) * HEIGHT_SCALE_FACTOR # Apply scaling (currently 1.0)

        # Optionally smooth internal tile boundaries *within* this sector
        if APPLY_BOUNDARY_SMOOTHING:
            absolute_height_sector = smooth_tile_boundaries(absolute_height_sector, TILE_WIDTH, TILE_HEIGHT)

        # Update the global min/max range based on this sector's valid data
        sector_min = np.min(absolute_height_sector) # np.min handles potential NaNs if any
        sector_max = np.max(absolute_height_sector)
        if np.isfinite(sector_min) and np.isfinite(sector_max): # Check if min/max are valid numbers
            overall_min_h = min(overall_min_h, sector_min)
            overall_max_h = max(overall_max_h, sector_max)
            first_sector_processed = True
        else:
            print(f"  W: Sector ({sx},{sy}) contains non-finite height values after processing. Min/max may be affected.")


        # Create the weight map for blending this sector's edges
        weight_map = create_weight_map(sector_h, sector_w, BOUNDARY_BLEND_SIZE)

        # Calculate where to place this sector in the final map grid
        grid_x = sx - min_sx # 0-based index in the sector grid
        grid_y = sy - min_sy
        paste_x_start = grid_x * effective_sector_width
        paste_y_start = grid_y * effective_sector_height
        paste_x_end = paste_x_start + sector_w
        paste_y_end = paste_y_start + sector_h

        # Determine the slice of the final arrays this sector overlaps with
        target_y_start_clipped = max(0, paste_y_start)
        target_y_end_clipped = min(final_height, paste_y_end)
        target_x_start_clipped = max(0, paste_x_start)
        target_x_end_clipped = min(final_width, paste_x_end)

        # Check if the clipped area is valid
        if target_y_start_clipped >= target_y_end_clipped or target_x_start_clipped >= target_x_end_clipped:
            print(f"  W: Sector ({sx},{sy}) falls completely outside the calculated map boundary. Skipping paste.")
            continue # Nothing to paste

        # Calculate the corresponding slice within the *source* sector data
        clip_top = target_y_start_clipped - paste_y_start
        clip_left = target_x_start_clipped - paste_x_start
        clipped_height = target_y_end_clipped - target_y_start_clipped
        clipped_width = target_x_end_clipped - target_x_start_clipped

        source_y_start = clip_top
        source_y_end = clip_top + clipped_height
        source_x_start = clip_left
        source_x_end = clip_left + clipped_width

        # Define the slice objects for numpy array indexing
        target_slice = (slice(target_y_start_clipped, target_y_end_clipped),
                        slice(target_x_start_clipped, target_x_end_clipped))
        source_slice = (slice(source_y_start, source_y_end),
                        slice(source_x_start, source_x_end))

        # --- Perform the weighted addition ---
        try:
            # Extract the relevant parts of the source height and weight maps
            source_heights = absolute_height_sector[source_slice]
            source_weights = weight_map[source_slice]

            # Ensure dimensions match exactly before adding (sanity check)
            if (source_heights.shape != (clipped_height, clipped_width) or
                source_weights.shape != (clipped_height, clipped_width) or
                heightmap_sum_array[target_slice].shape != (clipped_height, clipped_width)):
                print(f"  E: Slice dimension mismatch detected for sector ({sx},{sy})! "
                      f"SrcH:{source_heights.shape}, SrcW:{source_weights.shape}, TgtSum:{heightmap_sum_array[target_slice].shape} "
                      f"vs Expected:({clipped_height},{clipped_width}). Skipping paste.")
                continue

            # Create a mask for valid (non-NaN) height values in the source
            valid_mask = ~np.isnan(source_heights)

            # Add weighted heights to the sum array, only where source is valid
            heightmap_sum_array[target_slice][valid_mask] += source_heights[valid_mask] * source_weights[valid_mask]
            # Add weights to the weight sum array, only where source is valid
            weight_sum_array[target_slice][valid_mask] += source_weights[valid_mask]

            placed_count += 1
        except IndexError as e:
             print(f"  E: IndexError during array slicing/addition for sector ({sx},{sy}). Check calculations.")
             print(f"     Target slice: {target_slice}, Source slice: {source_slice}")
             print(f"     Sector dims: {sector_w}x{sector_h}, Final dims: {final_width}x{final_height}")
             traceback.print_exc()
             continue # Skip this sector on error
        except Exception as e:
            print(f"  E: Unexpected error during weighted addition for sector ({sx},{sy}): {e}")
            traceback.print_exc()
            continue # Skip this sector on error

    print(f"\nFinished Pass 2. Attempted to blend {placed_count} sectors.")
    if placed_count == 0:
        print("Error: No sectors were successfully placed onto the final map. Cannot proceed.")
        return

    # --- Pass 3: Finalization and Normalization ---
    print("\n--- Pass 3: Final Normalization and Saving ---")

    # Check if any valid heights were found across all sectors
    if not first_sector_processed or overall_min_h == float('inf'):
        print(" W: No valid finite height values were found in any processed sector. Defaulting range to [0, 1].")
        overall_min_h = 0.0
        overall_max_h = 1.0 # Avoid division by zero later
    elif not np.isfinite(overall_min_h) or not np.isfinite(overall_max_h):
         print(f" W: Overall height range contains non-finite values ([{overall_min_h}, {overall_max_h}]). Clamping range.")
         # Attempt to recover if only one bound is bad, else default
         if np.isfinite(overall_min_h): overall_max_h = overall_min_h + 1.0 # Default range
         elif np.isfinite(overall_max_h): overall_min_h = overall_max_h - 1.0 # Default range
         else: overall_min_h, overall_max_h = 0.0, 1.0

    print(f"Global raw height offset range found: [{overall_min_h}, {overall_max_h}]")

    # Calculate the final averaged heightmap
    # Initialize with NaN to easily identify areas with zero weight
    final_heightmap_float = np.full((final_height, final_width), np.nan, dtype=np.float32)
    # Create a mask where weights are non-negligible (avoid division by zero/tiny numbers)
    valid_weights_mask = weight_sum_array > 1e-9 # Use a small epsilon

    # Perform the division: height_sum / weight_sum only where weight is valid
    np.divide(
        heightmap_sum_array,
        weight_sum_array,
        out=final_heightmap_float, # Write result to the pre-allocated array
        where=valid_weights_mask   # Only compute where the mask is True
    )

    # Fill areas with no data (zero weight) or where result was NaN/Inf
    # Use the calculated overall minimum height as the background filler
    DEFAULT_BG_COLOR_FLOAT = overall_min_h
    # Replace initial NaNs (where weight was zero) with the default background
    final_heightmap_float[~valid_weights_mask] = DEFAULT_BG_COLOR_FLOAT
    # Replace any NaNs/Infs that might have resulted from division (e.g., 0/0)
    final_heightmap_float[~np.isfinite(final_heightmap_float)] = DEFAULT_BG_COLOR_FLOAT

    # --- Normalize the float map to 0-255 range for saving as image ---
    norm_range = overall_max_h - overall_min_h
    if norm_range <= 0:
        # Handle case where min and max are the same (flat terrain) or invalid range
        print(" W: Height range is zero or negative. Outputting a flat image (all zero).")
        heightmap_8bit = np.zeros((final_height, final_width), dtype=np.uint8)
    else:
        # Perform standard min-max normalization: (value - min) / range
        normalized_map = (final_heightmap_float - overall_min_h) / norm_range
        # Clip values to ensure they are strictly within [0, 1] after potential float inaccuracies
        np.clip(normalized_map, 0.0, 1.0, out=normalized_map)
        # Scale to 0-255 and convert to 8-bit unsigned integer
        heightmap_8bit = (normalized_map * 255).astype(np.uint8)

    # --- Save the Result ---
    try:
        img = Image.fromarray(heightmap_8bit, mode='L') # 'L' mode for grayscale
        out_path = os.path.join(script_dir, OUTPUT_FILENAME)
        img.save(out_path)
        print("\n--- Processing Complete ---")
        print(f"Saved final stitched and normalized heightmap to: {out_path}")
        print(f"Final map dimensions: {final_width} x {final_height} pixels")
        print(f"Raw height range used for normalization: [{overall_min_h}, {overall_max_h}] mapped to [0, 255]")
    except Exception as e:
        print(f"\nError saving the final image: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()