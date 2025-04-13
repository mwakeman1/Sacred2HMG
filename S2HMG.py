import struct
import math
import os
import glob
import re
import numpy as np
from PIL import Image
import sys
import traceback

# --- Configuration ---
ENTRY_SIZE = 8
HEIGHT_OFFSET_IN_ENTRY = 6  # Offset within the entry (uint16)
HEIGHT_FORMAT = '<H'        # Unsigned short for offset
EXPECTED_ZERO_BYTES_OFFSETS = [2, 3, 4, 5]

START_MARKER = b'\x00\x00\x01\x00'
VARIABLE_DATA_AFTER_MARKER_SIZE = 4
INTERMEDIATE_HEADER_SIZE = 8
BYTES_TO_SKIP_AFTER_MARKER = VARIABLE_DATA_AFTER_MARKER_SIZE + INTERMEDIATE_HEADER_SIZE  # 12 bytes
N_CONTEXT_CHECK = 5

BASE_H_OFFSET_IN_HEADER = 2    # We will READ this but ignore it.
BASE_H_FORMAT = '<h'           # Signed short base from header
HEIGHT_SCALE_FACTOR = 1.0      # We keep this at 1.0

# --- Tiling ---
TILE_WIDTH = 7
TILE_HEIGHT = 7
POINTS_PER_TILE = TILE_WIDTH * TILE_HEIGHT

# --- Sector Dimensions ---
LAYOUT_SECTOR_WIDTH = 224
LAYOUT_SECTOR_HEIGHT = 224

# --- Boundary Treatment ---
SECTOR_OVERLAP = 1
BOUNDARY_BLEND_SIZE = 10

# --- Smoothing Parameters ---
APPLY_BOUNDARY_SMOOTHING = True  # Keep boundary smoothing

# --- Output ---
output_suffix = "HEIGHTMAP_IgnoreBase_GlobalMinMax"
if APPLY_BOUNDARY_SMOOTHING:
    output_suffix += "_smooth"
OUTPUT_FILENAME = f"heightmap_{output_suffix}.png"

ENABLE_DIAGNOSTICS = False
DIAGNOSTICS_DIR = "diagnostics"

# --- Normalization ---
# We do a SINGLE min-max at the end, so these are just placeholders
CONTRAST_PERCENTILE_LOW = 0.0
CONTRAST_PERCENTILE_HIGH = 100.0

def parse_filename(filename):
    match = re.match(r'(\d{3})_(\d{3})_.*\.sector', os.path.basename(filename))
    if match:
        return int(match.group(1)), int(match.group(2))
    else:
        return None, None

def create_weight_map(sector_h, sector_w, blend_size):
    """ Create a weight map with an exponential ramp at edges for blending. """
    weight_map = np.ones((sector_h, sector_w), dtype=np.float32)
    if blend_size <= 0:
        return weight_map
    blend_pixels = min(blend_size, sector_w // 2, sector_h // 2)
    if blend_pixels <= 0:
        return weight_map
    ramp = np.exp(-5.0 * (1.0 - np.linspace(0.0, 1.0, blend_pixels + 1)[1:]))
    center_start = blend_pixels
    center_end_y = sector_h - blend_pixels
    center_end_x = sector_w - blend_pixels
    # top/bottom
    for i in range(blend_pixels):
        weight = ramp[i]
        if center_start < center_end_x:
            weight_map[i, center_start:center_end_x] = weight
            weight_map[sector_h - 1 - i, center_start:center_end_x] = weight
        if center_start < center_end_y:
            weight_map[center_start:center_end_y, i] = weight
            weight_map[center_start:center_end_y, sector_w - 1 - i] = weight
    # corners
    for r in range(blend_pixels):
        for c in range(blend_pixels):
            weight_corner = ramp[min(r, c)]
            weight_map[r, c] = min(weight_map[r, c], weight_corner)
            weight_map[r, sector_w - 1 - c] = min(weight_map[r, sector_w - 1 - c], weight_corner)
            weight_map[sector_h - 1 - r, c] = min(weight_map[sector_h - 1 - r, c], weight_corner)
            weight_map[sector_h - 1 - r, sector_w - 1 - c] = min(weight_map[sector_h - 1 - r, sector_w - 1 - c], weight_corner)
    if center_start < center_end_y and center_start < center_end_x:
        weight_map[center_start:center_end_y, center_start:center_end_x] = 1.0
    return weight_map

def process_sector_base_offset(filepath, expected_width, expected_height):
    """
    Reads int16 base (Hdr@2) and uint16 offsets (Entry@6) from the .sector file.
    We'll STILL read base_h for debugging, but won't actually use it in the final height.
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
                return None, None, None, None, None, None
            search_start_offset = 0
            found_valid_context = False
            while search_start_offset < file_size:
                current_marker_offset = content_to_scan.find(START_MARKER, search_start_offset)
                if current_marker_offset == -1:
                    break
                potential_data_start = current_marker_offset + len(START_MARKER) + BYTES_TO_SKIP_AFTER_MARKER
                potential_header_start = current_marker_offset + len(START_MARKER)
                if potential_data_start + (N_CONTEXT_CHECK * ENTRY_SIZE) <= file_size:
                    context_valid = True
                    for i in range(N_CONTEXT_CHECK):
                        entry_offset = potential_data_start + (i * ENTRY_SIZE)
                        entry_bytes = content_to_scan[entry_offset: entry_offset + ENTRY_SIZE]
                        if (len(entry_bytes) < ENTRY_SIZE or
                            not all(entry_bytes[z] == 0x00 for z in EXPECTED_ZERO_BYTES_OFFSETS)):
                            context_valid = False
                            break
                    if context_valid:
                        # Attempt to read the base from the sector header
                        try:
                            header_bytes = content_to_scan[
                                potential_header_start : potential_header_start + BYTES_TO_SKIP_AFTER_MARKER
                            ]
                            if len(header_bytes) >= BASE_H_OFFSET_IN_HEADER + struct.calcsize(BASE_H_FORMAT):
                                base_h, = struct.unpack_from(BASE_H_FORMAT, header_bytes, BASE_H_OFFSET_IN_HEADER)
                            else:
                                base_h = 0
                                header_bytes = None
                        except Exception as e:
                            base_h = 0
                            header_bytes = None
                            print(f" W:{os.path.basename(filepath)} Error reading header base H: {e}")
                        if header_bytes is not None:
                            data_start_offset = potential_data_start
                            found_valid_context = True
                            break
                    search_start_offset = current_marker_offset + 1
                else:
                    break
            if not found_valid_context:
                print(f" W: No valid data block found in {os.path.basename(filepath)}.")
                return None, None, None, None, None, None
            
            # Now read the offset entries
            f.seek(data_start_offset)
            bytes_needed = expected_vertices * ENTRY_SIZE
            if f.tell() + bytes_needed > file_size:
                print(f" W:{os.path.basename(filepath)} Not enough data. Skipping.")
                return None, None, None, None, None, None
            
            for _ in range(expected_vertices):
                entry_bytes = f.read(ENTRY_SIZE)
                if len(entry_bytes) < ENTRY_SIZE:
                    print(f" E:{os.path.basename(filepath)} Unexpected EOF.")
                    return None, None, None, None, None, None
                try:
                    h_val, = struct.unpack_from(HEIGHT_FORMAT, entry_bytes, HEIGHT_OFFSET_IN_ENTRY)
                    height_offsets.append(h_val)
                except struct.error:
                    print(f" W:{os.path.basename(filepath)} Struct error unpacking uint16 offset.")
                    return None, None, None, None, None, None
            
            if len(height_offsets) != expected_vertices:
                print(f" E:{os.path.basename(filepath)} Read count mismatch.")
                return None, None, None, None, None, None
            
            return (height_offsets,
                    expected_width,
                    expected_height,
                    base_h,            # read but ignoring for the final map
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
    """ Detiling function using row-major indexing. """
    detiled_heights = np.zeros((sector_height, sector_width), dtype=np.uint16)
    tile_width = TILE_WIDTH
    tile_height = TILE_HEIGHT

    tiles_per_row = sector_width // tile_width
    tiles_per_col = sector_height // tile_height

    expected_vertices = sector_width * sector_height
    if len(sector_offset_values) != expected_vertices:
        print("  W: Detile input size mismatch.")
        return None

    for output_y in range(sector_height):
        for output_x in range(sector_width):
            tile_row = output_y // tile_height
            tile_col = output_x // tile_width

            local_y = output_y % tile_height
            local_x = output_x % tile_width

            tile_index = tile_row * tiles_per_row + tile_col
            position_in_tile = local_y * tile_width + local_x

            flat_index = tile_index * (tile_width * tile_height) + position_in_tile
            if 0 <= flat_index < len(sector_offset_values):
                detiled_heights[output_y, output_x] = sector_offset_values[flat_index]
            else:
                print(f"  W: Calculated flat index {flat_index} is out of bounds for {len(sector_offset_values)}")

    return detiled_heights

def smooth_tile_boundaries(heightmap, tile_width=7, tile_height=7):
    """Apply simple averaging smoothing at tile boundaries."""
    h, w = heightmap.shape
    smoothed = heightmap.copy().astype(np.float32)

    # Horizontal boundaries
    for y in range(h):
        for x in range(tile_width - 1, w - 1, tile_width):
            smoothed[y, x] = (float(heightmap[y, x - 1]) + float(heightmap[y, x + 1])) / 2.0

    # Vertical boundaries
    for x in range(w):
        for y in range(tile_height - 1, h - 1, tile_height):
            smoothed[y, x] = (float(heightmap[y - 1, x]) + float(heightmap[y + 1, x])) / 2.0

    # Corners
    for y in range(tile_height - 1, h - 1, tile_height):
        for x in range(tile_width - 1, w - 1, tile_width):
            smoothed[y, x] = (float(heightmap[y - 1, x - 1]) +
                              float(heightmap[y - 1, x + 1]) +
                              float(heightmap[y + 1, x - 1]) +
                              float(heightmap[y + 1, x + 1])) / 4.0
    return smoothed

def main():
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        script_dir = os.path.abspath('.')
        print(f"Warning: Using cwd: {script_dir}")

    print("--- Processing ALL Sectors (Ignoring base_h) ---")
    print("--- Normalization: Single global Min-Max across entire map. ---")
    if APPLY_BOUNDARY_SMOOTHING:
        print("--- Boundary Smoothing: Enabled ---")

    all_sector_files = glob.glob(os.path.join(script_dir, '*.sector'))
    output_file_path = os.path.join(script_dir, OUTPUT_FILENAME)
    diag_path_prefix = os.path.join(script_dir, DIAGNOSTICS_DIR) + os.path.sep
    initial_filtered_files = [
        f for f in all_sector_files
        if os.path.abspath(f) != os.path.abspath(output_file_path)
        and not os.path.abspath(f).startswith(os.path.abspath(diag_path_prefix))
    ]

    sector_files = initial_filtered_files
    print(f"Processing {len(sector_files)} sector files found.")
    if not sector_files:
        print("Error: No sector files found.")
        return

    # We keep a fixed layout for each sector
    print("\n--- Using Layout ---")
    LAYOUT_SECTOR_WIDTH, LAYOUT_SECTOR_HEIGHT = 224, 224
    print(f"{LAYOUT_SECTOR_WIDTH} x {LAYOUT_SECTOR_HEIGHT}")

    # --- Pass 1: Read offset data (and base, though we ignore it) ---
    sectors_data = {}
    all_coords = []
    processed_count = 0
    for filepath in sector_files:
        filename = os.path.basename(filepath)
        sx, sy = parse_filename(filename)
        if sx is None or sy is None:
            print(f"  W: Skipping '{filename}', bad coords.")
            continue

        result = process_sector_base_offset(filepath, LAYOUT_SECTOR_WIDTH, LAYOUT_SECTOR_HEIGHT)
        if result[0] is not None:
            height_offsets, w, h, base_h, header_bytes, data_start_offset = result
            detiled_offsets_uint16 = correct_detile_sector(height_offsets, w, h)
            if detiled_offsets_uint16 is not None:
                sectors_data[(sx, sy)] = {
                    'detiled_offsets': detiled_offsets_uint16,
                    'width': w,
                    'height': h
                    # base_h is intentionally ignored
                }
                all_coords.append((sx, sy))
                processed_count += 1
            else:
                print(f"  W: Failed to detile {filename}.")
        else:
            print(f"  W: Failed to process {filename}.")

    print(f"\nFinished Pass 1. Processed {processed_count} valid sectors.")
    if processed_count == 0:
        print("Error: No valid sectors processed.")
        return

    min_sx = min(c[0] for c in all_coords)
    max_sx = max(c[0] for c in all_coords)
    min_sy = min(c[1] for c in all_coords)
    max_sy = max(c[1] for c in all_coords)

    print(f"Sector coordinate range: X=[{min_sx}-{max_sx}], Y=[{min_sy}-{max_sy}]")

    current_overlap = SECTOR_OVERLAP
    if current_overlap < 0 or current_overlap >= min(LAYOUT_SECTOR_WIDTH, LAYOUT_SECTOR_HEIGHT):
        print(f"Warning: Invalid SECTOR_OVERLAP ({current_overlap}). Clamping.")
        current_overlap = max(0, min(current_overlap, min(LAYOUT_SECTOR_WIDTH, LAYOUT_SECTOR_HEIGHT) - 1))
    effective_sector_width = LAYOUT_SECTOR_WIDTH - current_overlap
    effective_sector_height = LAYOUT_SECTOR_HEIGHT - current_overlap
    num_sectors_x = max_sx - min_sx + 1
    num_sectors_y = max_sy - min_sy + 1
    final_width = (num_sectors_x - 1) * effective_sector_width + LAYOUT_SECTOR_WIDTH if num_sectors_x > 0 else 0
    final_height = (num_sectors_y - 1) * effective_sector_height + LAYOUT_SECTOR_HEIGHT if num_sectors_y > 0 else 0

    if num_sectors_x <= 1:
        final_width = LAYOUT_SECTOR_WIDTH
    if num_sectors_y <= 1:
        final_height = LAYOUT_SECTOR_HEIGHT

    print(f"\nOverlap={current_overlap}, Blend Size={BOUNDARY_BLEND_SIZE}")
    print(f"Final map dimensions: {final_width} x {final_height}")
    if final_width <= 0 or final_height <= 0:
        print("Error: Invalid final dimensions.")
        return

    try:
        print(f"Allocating final map arrays ({final_height} x {final_width})...")
        heightmap_sum_array = np.zeros((final_height, final_width), dtype=np.float64)
        weight_sum_array = np.zeros((final_height, final_width), dtype=np.float64)
        print("Allocation successful.")
    except MemoryError:
        print("Error: Not enough memory for map arrays.")
        return
    except Exception as e:
        print(f"Error creating numpy arrays: {e}")
        return

    placed_count = 0
    overall_min_h = float('inf')
    overall_max_h = float('-inf')
    first_sector_processed = False

    # --- Pass 2: Stitch all sectors (ignoring base_h entirely) ---
    for (sx, sy), data in sectors_data.items():
        sector_w = data['width']
        sector_h = data['height']
        detiled_offsets = data['detiled_offsets']  # uint16

        # Convert to float; no base offset. 
        absolute_height_sector = detiled_offsets.astype(np.float32)

        # Optionally, apply smoothing to tile boundaries
        if APPLY_BOUNDARY_SMOOTHING:
            absolute_height_sector = smooth_tile_boundaries(absolute_height_sector, TILE_WIDTH, TILE_HEIGHT)

        # Update overall min/max
        sector_min = float(np.min(absolute_height_sector))
        sector_max = float(np.max(absolute_height_sector))
        if np.isfinite(sector_min) and np.isfinite(sector_max):
            overall_min_h = min(overall_min_h, sector_min)
            overall_max_h = max(overall_max_h, sector_max)
            first_sector_processed = True

        # Blend into final
        weight_map = create_weight_map(sector_h, sector_w, BOUNDARY_BLEND_SIZE)
        grid_x = sx - min_sx
        grid_y = sy - min_sy
        paste_x_start = grid_x * effective_sector_width
        paste_y_start = grid_y * effective_sector_height
        paste_x_end = paste_x_start + sector_w
        paste_y_end = paste_y_start + sector_h

        target_y_start_clipped = max(0, paste_y_start)
        target_y_end_clipped = min(final_height, paste_y_end)
        target_x_start_clipped = max(0, paste_x_start)
        target_x_end_clipped = min(final_width, paste_x_end)

        if target_y_start_clipped >= target_y_end_clipped or target_x_start_clipped >= target_x_end_clipped:
            continue

        clip_top = target_y_start_clipped - paste_y_start
        clip_left = target_x_start_clipped - paste_x_start
        clipped_height = target_y_end_clipped - target_y_start_clipped
        clipped_width = target_x_end_clipped - target_x_start_clipped

        source_y_start = clip_top
        source_y_end = clip_top + clipped_height
        source_x_start = clip_left
        source_x_end = clip_left + clipped_width

        target_slice = (slice(target_y_start_clipped, target_y_end_clipped),
                        slice(target_x_start_clipped, target_x_end_clipped))
        source_slice = (slice(source_y_start, source_y_end),
                        slice(source_x_start, source_x_end))

        if (absolute_height_sector[source_slice].shape != (clipped_height, clipped_width) or
            weight_map[source_slice].shape != (clipped_height, clipped_width) or
            heightmap_sum_array[target_slice].shape != (clipped_height, clipped_width)):
            print(f"  W: Slice dimension mismatch for sector ({sx},{sy})! Skipping paste.")
            continue

        try:
            source_heights = absolute_height_sector[source_slice]
            source_weights = weight_map[source_slice]
            valid_mask = ~np.isnan(source_heights)

            heightmap_sum_array[target_slice][valid_mask] += source_heights[valid_mask] * source_weights[valid_mask]
            weight_sum_array[target_slice][valid_mask] += source_weights[valid_mask]
            placed_count += 1
        except Exception as e:
            print(f"  E: Error during weighted addition for sector ({sx},{sy}): {e}")
            traceback.print_exc()
            continue

    print(f"\nFinished Pass 2. Blended {placed_count} sectors.")
    if placed_count == 0:
        print("Error: No sectors placed.")
        return

    # --- Pass 3: Single Global Min-Max Normalization ---
    if not first_sector_processed or overall_min_h == float('inf'):
        print(" W: No valid heights found. Defaulting range [0..1].")
        overall_min_h = 0.0
        overall_max_h = 1.0

    # Make sure we handle Inf/NaN
    overall_min_h = np.nanmin([overall_min_h, np.inf])
    overall_max_h = np.nanmax([overall_max_h, -np.inf])
    if not np.isfinite(overall_min_h) or not np.isfinite(overall_max_h):
        print(" W: Invalid height range. Defaulting to [0..1].")
        overall_min_h = 0.0
        overall_max_h = 1.0

    print(f"Global raw height range: [{overall_min_h}, {overall_max_h}]")

    # Divide sum by weights
    final_heightmap_float = np.full((final_height, final_width), np.nan, dtype=np.float32)
    valid_weights_mask = weight_sum_array > 1e-9
    np.divide(
        heightmap_sum_array,
        weight_sum_array,
        out=final_heightmap_float,
        where=valid_weights_mask
    )

    # Fill "no-data" with the overall min
    DEFAULT_BG_COLOR_FLOAT = overall_min_h
    final_heightmap_float[~valid_weights_mask] = DEFAULT_BG_COLOR_FLOAT
    final_heightmap_float[np.isnan(final_heightmap_float)] = DEFAULT_BG_COLOR_FLOAT

    # Normalize
    norm_range = overall_max_h - overall_min_h
    if norm_range <= 0:
        print(" W: Zero or negative range. Using blank image.")
        heightmap_8bit = np.zeros((final_height, final_width), dtype=np.uint8)
    else:
        normalized_map = (final_heightmap_float - overall_min_h) / norm_range
        np.clip(normalized_map, 0.0, 1.0, out=normalized_map)
        heightmap_8bit = (normalized_map * 255).astype(np.uint8)

    # Save result
    try:
        img = Image.fromarray(heightmap_8bit, mode='L')
        out_path = os.path.join(script_dir, OUTPUT_FILENAME)
        img.save(out_path)
        print("\n--- Complete ---")
        print(f"Saved final heightmap: {out_path}")
        print(f"Map size: {final_width}x{final_height}")
        print(f"Min-Max used for normalization: [{overall_min_h}, {overall_max_h}] → [0..255]")
    except Exception as e:
        print(f"Error saving final image: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
