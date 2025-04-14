import struct
import math
import os
# import cv2 # Removed import
import glob
import re
import numpy as np
from PIL import Image
import zipfile
import traceback

ENTRY_SIZE = 8
HEIGHT_OFFSET_IN_ENTRY = 6
HEIGHT_FORMAT = '<H'
EXPECTED_ZERO_BYTES_OFFSETS = [2, 3, 4, 5]
START_MARKER = b'\x00\x00\x01\x00'
VARIABLE_DATA_AFTER_MARKER_SIZE = 4
INTERMEDIATE_HEADER_SIZE = 8
BYTES_TO_SKIP_AFTER_MARKER = VARIABLE_DATA_AFTER_MARKER_SIZE + INTERMEDIATE_HEADER_SIZE
N_CONTEXT_CHECK = 5
BASE_H_OFFSET_IN_HEADER = 2
BASE_H_FORMAT = '<h'
TILE_WIDTH = 7
TILE_HEIGHT = 7
POINTS_PER_TILE = TILE_WIDTH * TILE_HEIGHT
LAYOUT_SECTOR_WIDTH = 224
LAYOUT_SECTOR_HEIGHT = 224

def parse_filename(filename):
    match = re.match(r'(\d{3})_(\d{3})_.*\.sector', os.path.basename(filename))
    if match:
        return int(match.group(1)), int(match.group(2))
    else:
        return None, None

def create_weight_map(sector_h, sector_w, blend_size):
    weight_map = np.ones((sector_h, sector_w), dtype=np.float32)
    if blend_size <= 0: return weight_map
    blend_pixels = min(blend_size, sector_w // 2, sector_h // 2)
    if blend_pixels <= 0: return weight_map
    x_blend = np.linspace(0., np.pi / 2., blend_pixels + 1)[1:]
    ramp = (np.cos(x_blend) ** 2)
    ramp = ramp[::-1]
    center_start = blend_pixels
    center_end_y = sector_h - blend_pixels
    center_end_x = sector_w - blend_pixels
    for i in range(blend_pixels):
        weight = ramp[i]
        if center_start < center_end_x:
             weight_map[i, center_start:center_end_x] = weight
             weight_map[sector_h - 1 - i, center_start:center_end_x] = weight
        if center_start < center_end_y:
             weight_map[center_start:center_end_y, i] = weight
             weight_map[center_start:center_end_y, sector_w - 1 - i] = weight
    for r in range(blend_pixels):
        for c in range(blend_pixels):
            weight_corner = min(ramp[r], ramp[c])
            weight_map[r, c] = min(weight_map[r, c], weight_corner)
            weight_map[r, sector_w - 1 - c] = min(weight_map[r, sector_w - 1 - c], weight_corner)
            weight_map[sector_h - 1 - r, c] = min(weight_map[sector_h - 1 - r, c], weight_corner)
            weight_map[sector_h - 1 - r, sector_w - 1 - c] = min(weight_map[sector_h - 1 - r, sector_w - 1 - c], weight_corner)
    if center_start < center_end_y and center_start < center_end_x:
        weight_map[center_start:center_end_y, center_start:center_end_x] = 1.0
    return weight_map

def process_sector_base_offset(sector_content_bytes, sector_filename_in_zip, expected_width, expected_height, log_queue):
    # Note: Cancellation check is not added here as it's very low-level file parsing.
    # Checks are added in the higher-level loops of generate_heightmap.
    height_offsets = []
    base_h = 0
    header_bytes = None
    data_start_offset = -1
    expected_vertices = expected_width * expected_height
    width = None
    height = None
    filename_short = os.path.basename(sector_filename_in_zip)
    try:
        content_to_scan = sector_content_bytes
        file_size = len(content_to_scan)
        if not content_to_scan:
            log_queue.put(f" W: Skipping empty sector content for {filename_short}")
            return None, None, None, None, None, None
        search_start_offset = 0
        found_valid_context = False
        while search_start_offset < file_size:
            current_marker_offset = content_to_scan.find(START_MARKER, search_start_offset)
            if current_marker_offset == -1: break
            potential_data_start = current_marker_offset + len(START_MARKER) + BYTES_TO_SKIP_AFTER_MARKER
            potential_header_start = current_marker_offset + len(START_MARKER)
            if potential_data_start + (N_CONTEXT_CHECK * ENTRY_SIZE) <= file_size:
                context_valid = True
                for i in range(N_CONTEXT_CHECK):
                    entry_offset = potential_data_start + (i * ENTRY_SIZE)
                    if entry_offset + ENTRY_SIZE > file_size: context_valid = False; break
                    entry_bytes = content_to_scan[entry_offset : entry_offset + ENTRY_SIZE]
                    if len(entry_bytes) < ENTRY_SIZE or not all(entry_bytes[z] == 0x00 for z in EXPECTED_ZERO_BYTES_OFFSETS):
                        context_valid = False; break
                if context_valid:
                    try:
                        if potential_header_start + BYTES_TO_SKIP_AFTER_MARKER > file_size:
                            log_queue.put(f" W:{filename_short} Not enough data for header after marker.")
                            context_valid = False
                        else:
                            header_bytes = content_to_scan[potential_header_start : potential_header_start + BYTES_TO_SKIP_AFTER_MARKER]
                            if len(header_bytes) >= BASE_H_OFFSET_IN_HEADER + struct.calcsize(BASE_H_FORMAT):
                                base_h, = struct.unpack_from(BASE_H_FORMAT, header_bytes, BASE_H_OFFSET_IN_HEADER)
                                data_start_offset = potential_data_start
                                found_valid_context = True; break
                            else:
                                log_queue.put(f" W:{filename_short} Header size insufficient after marker.")
                                base_h = 0; header_bytes = None
                    except Exception as e:
                        log_queue.put(f" W:{filename_short} Error reading header base H: {e}")
                        base_h = 0; header_bytes = None
            search_start_offset = current_marker_offset + 1
        if not found_valid_context:
            log_queue.put(f" W: No valid data block found in {filename_short}.")
            return None, None, None, None, None, None
        bytes_needed = expected_vertices * ENTRY_SIZE
        data_block_bytes = content_to_scan[data_start_offset:]
        if len(data_block_bytes) < bytes_needed:
            log_queue.put(f" W:{filename_short} Not enough data for {expected_vertices} vertices after marker (found {len(data_block_bytes)} bytes, need {bytes_needed}). Skipping.")
            return None, None, None, None, None, None
        for i in range(expected_vertices):
            entry_offset_in_data = i * ENTRY_SIZE
            entry_bytes = data_block_bytes[entry_offset_in_data : entry_offset_in_data + ENTRY_SIZE]
            if len(entry_bytes) < ENTRY_SIZE:
                   log_queue.put(f" E:{filename_short} Unexpected short read at vertex {i+1}/{expected_vertices} (internal error).")
                   return None, None, None, None, None, None
            try:
                if not all(entry_bytes[z] == 0x00 for z in EXPECTED_ZERO_BYTES_OFFSETS):
                      log_queue.put(f" W:{filename_short} Entry {i} failed zero byte check {entry_bytes[2:6].hex()}. Using offset anyway.")
                h_val, = struct.unpack_from(HEIGHT_FORMAT, entry_bytes, HEIGHT_OFFSET_IN_ENTRY)
                height_offsets.append(h_val)
            except struct.error as e:
                log_queue.put(f" W:{filename_short} Struct error unpacking uint16 offset at vertex {i}: {e}")
                return None, None, None, None, None, None
        if len(height_offsets) != expected_vertices:
            log_queue.put(f" E:{filename_short} Read count mismatch ({len(height_offsets)} vs {expected_vertices}).")
            return None, None, None, None, None, None
        width = expected_width
        height = expected_height
        return height_offsets, width, height, base_h, header_bytes, data_start_offset
    except Exception as e:
        log_queue.put(f" Error processing sector content for {filename_short}: {e}")
        log_queue.put(traceback.format_exc())
        return None, None, None, None, None, None


def correct_detile_sector(sector_offset_values, sector_width, sector_height, log_queue):
    # Note: Cancellation check is not added here as it's very low-level.
    detiled_heights = np.zeros((sector_height, sector_width), dtype=np.uint16)
    tiles_per_row = sector_width // TILE_WIDTH
    tiles_per_col = sector_height // TILE_HEIGHT
    if TILE_WIDTH <= 0 or TILE_HEIGHT <= 0: log_queue.put(" E: Invalid TILE_WIDTH/HEIGHT"); return None
    if sector_width % TILE_WIDTH != 0 or sector_height % TILE_HEIGHT != 0:
         log_queue.put(f" W: Sector dimensions ({sector_width}x{sector_height}) not perfectly divisible by tile size ({TILE_WIDTH}x{TILE_HEIGHT}).")
    expected_vertices = sector_width * sector_height
    if len(sector_offset_values) < expected_vertices:
         log_queue.put(f"  W: Detile input size mismatch. Expected >= {expected_vertices}, got {len(sector_offset_values)}")
         return None
    vertex_index = 0
    for tile_row in range(tiles_per_col):
        for tile_col in range(tiles_per_row):
            tile_base_y = tile_row * TILE_HEIGHT
            tile_base_x = tile_col * TILE_WIDTH
            for local_y in range(TILE_HEIGHT):
                for local_x in range(TILE_WIDTH):
                    output_y = tile_base_y + local_y
                    output_x = tile_base_x + local_x
                    if output_y < sector_height and output_x < sector_width:
                        if vertex_index < len(sector_offset_values):
                            detiled_heights[output_y, output_x] = sector_offset_values[vertex_index]
                        else:
                            log_queue.put(f" E: Detile ran out of input data at index {vertex_index} for output ({output_x},{output_y}).")
                            detiled_heights[output_y, output_x] = 0
                        vertex_index += 1
    if vertex_index < expected_vertices:
         log_queue.put(f" W: Detile finished but processed only {vertex_index}/{expected_vertices} expected vertices.")
    elif vertex_index > expected_vertices:
         log_queue.put(f" W: Detile finished and processed MORE than expected vertices ({vertex_index}/{expected_vertices}). Input longer than needed?")
    return detiled_heights

def smooth_tile_boundaries(heightmap, tile_width, tile_height, log_queue):
    # Note: Cancellation check is not added here as it's generally fast.
    if tile_width <= 1 or tile_height <= 1: return heightmap.astype(np.float32)
    height, width = heightmap.shape
    smoothed = heightmap.copy().astype(np.float32)
    for x in range(tile_width - 1, width - 1, tile_width):
        if x > 0 and x < width - 1:
             smoothed[:, x] = (smoothed[:, x - 1] + smoothed[:, x + 1]) / 2.0
    for y in range(tile_height - 1, height - 1, tile_height):
        if y > 0 and y < height - 1:
             smoothed[y, :] = (smoothed[y - 1, :] + smoothed[y + 1, :]) / 2.0
    for y in range(tile_height - 1, height - 1, tile_height):
        for x in range(tile_width - 1, width - 1, tile_width):
            if y > 0 and y < height - 1 and x > 0 and x < width - 1:
                smoothed[y, x] = (smoothed[y - 1, x - 1] + smoothed[y - 1, x + 1] +
                                  smoothed[y + 1, x - 1] + smoothed[y + 1, x + 1]) / 4.0
    return smoothed


# Modified function signature to accept cancel_event
def generate_heightmap(config, log_queue, progress_queue, cancel_event):
    final_heightmap_float = None
    try:
        input_dir = config['input_dir']
        output_filename = config['output_filename']
        apply_boundary_smoothing = config['apply_boundary_smoothing']
        sector_overlap = config['sector_overlap']
        boundary_blend_size = config['boundary_blend_size']
        height_scale_factor = config['height_scale_factor']
        sector_suffix_type = config['sector_suffix_type']

        if not input_dir or not os.path.isdir(input_dir):
            log_queue.put("Error: Input directory is not valid."); progress_queue.put(-1.0); return None
        try:
            height_scale_factor = float(height_scale_factor)
            if height_scale_factor <= 1e-6:
                raise ValueError("Height Scale Factor must be significantly > 0.")
        except ValueError as e:
             log_queue.put(f"Error: Invalid Height Scale Factor ({e})."); progress_queue.put(-1.0); return None

        base_output_name, _ = os.path.splitext(output_filename)
        if not base_output_name: base_output_name = "heightmap_output"
        output_dir = input_dir
        output_file_path_png = os.path.join(output_dir, base_output_name + ".png")

        log_queue.put("--- Starting Heightmap Generation ---")
        log_queue.put(f"Input Folder: {input_dir}"); log_queue.put(f"Output Base Name: {base_output_name}")
        log_queue.put(f"Smooth Tiles: {apply_boundary_smoothing}"); log_queue.put(f"Overlap: {sector_overlap}")
        log_queue.put(f"Blend Size: {boundary_blend_size}"); log_queue.put(f"Height Scale: {height_scale_factor}")
        log_queue.put(f"Formula: H = base + (offset / {height_scale_factor})"); log_queue.put("Norm: Global Min/Max -> 0-255 PNG")

        progress_queue.put(0.0)

        # Check for cancellation before starting expensive operations
        if cancel_event.is_set():
            log_queue.put("--- Processing Cancelled by User (Before Start) ---")
            progress_queue.put(-1.0)
            return None

        if sector_suffix_type == "B0": required_suffix = "_b0.sector"
        elif sector_suffix_type == "D1": required_suffix = "_d1.sector"
        else: required_suffix = "_d2.sector"
        log_queue.put(f"Searching for: *{required_suffix}' (case-insensitive)")

        all_archives = []
        try:
            all_archives.extend(glob.glob(os.path.join(input_dir, '*.zip')))
            all_archives.extend(glob.glob(os.path.join(input_dir, '*.pak')))
            all_archives = sorted(list(set(all_archives)))
        except Exception as e:
            log_queue.put(f"Error searching for archives: {e}"); progress_queue.put(-1.0); return None

        log_queue.put(f"Found {len(all_archives)} ZIP/PAK archives.")
        if not all_archives:
            log_queue.put("Error: No .zip or .pak archives found."); progress_queue.put(-1.0); return None

        log_queue.put(f"Sector Grid Size (Expected): {LAYOUT_SECTOR_WIDTH} x {LAYOUT_SECTOR_HEIGHT}")
        log_queue.put("--- Pass 1: Reading sector data from Archives ---")
        sectors_data = {}; all_coords = []; processed_count = 0; total_archives = len(all_archives)

        for archive_idx, archive_filepath in enumerate(all_archives):
            # Check for cancellation at the start of each archive processing
            if cancel_event.is_set():
                log_queue.put("--- Processing Cancelled by User (During Pass 1) ---")
                progress_queue.put(-1.0)
                return None

            archive_filename = os.path.basename(archive_filepath)
            log_queue.put(f" -> Archive: {archive_filename} ({archive_idx+1}/{total_archives})")
            processed_in_archive = 0
            try:
                with zipfile.ZipFile(archive_filepath, 'r') as zf:
                    member_names = zf.namelist()
                    sector_members = [ name for name in member_names if name.lower().endswith(required_suffix.lower()) ]
                    if not sector_members: log_queue.put(f"    No sectors matching '{required_suffix}' found."); continue

                    for sector_idx, sector_name_in_zip in enumerate(sector_members):
                        # Optional: Check more frequently within large archives
                        if sector_idx % 20 == 0 and cancel_event.is_set(): # Check every 20 sectors
                             log_queue.put("--- Processing Cancelled by User (During Pass 1 - Sector Read) ---")
                             progress_queue.put(-1.0)
                             return None

                        filename_short = os.path.basename(sector_name_in_zip)
                        sx, sy = parse_filename(filename_short)
                        if sx is None or sy is None: log_queue.put(f"    W: Skip '{filename_short}', parse fail."); continue
                        if (sx, sy) in sectors_data: log_queue.put(f"    W: Skip '{filename_short}', duplicate coords ({sx},{sy})."); continue

                        try:
                            sector_bytes = zf.read(sector_name_in_zip)
                            result = process_sector_base_offset(sector_bytes, filename_short, LAYOUT_SECTOR_WIDTH, LAYOUT_SECTOR_HEIGHT, log_queue)
                            if result[0] is not None:
                                height_offsets, w, h, base_h, _, _ = result
                                if w is None or h is None or w != LAYOUT_SECTOR_WIDTH or h != LAYOUT_SECTOR_HEIGHT:
                                    log_queue.put(f"    W: Skip '{filename_short}', dim mismatch ({w}x{h})."); continue
                                detiled_offsets_uint16 = correct_detile_sector(height_offsets, w, h, log_queue)
                                if detiled_offsets_uint16 is not None:
                                    # No cancel check needed here, very fast operation
                                    log_queue.put(f"    OK: Processed {filename_short} ({sx},{sy}) BaseH={base_h}")
                                    sectors_data[(sx, sy)] = {'detiled_offsets': detiled_offsets_uint16, 'base_h': base_h, 'width': w, 'height': h}
                                    all_coords.append((sx, sy)); processed_count += 1; processed_in_archive += 1
                                else: log_queue.put(f"    W: Failed detile '{filename_short}'.")
                        except KeyError: log_queue.put(f"    E: Sector '{sector_name_in_zip}' not found (unexpected).")
                        except Exception as sector_e: log_queue.put(f"    E: Error reading sector '{sector_name_in_zip}': {sector_e}")
            except zipfile.BadZipFile: log_queue.put(f" W: Skipping bad archive: {archive_filename}")
            except Exception as zip_e: log_queue.put(f" E: Error with archive {archive_filename}: {zip_e}")
            if processed_in_archive > 0: log_queue.put(f"    -> Found {processed_in_archive} valid sectors.")
            progress_queue.put(0.3 * (archive_idx + 1) / total_archives)

        log_queue.put(f"Pass 1 Done. Processed {processed_count} unique sectors.")
        if processed_count == 0:
            log_queue.put(f"Error: No valid sector data found matching suffix '{required_suffix}'.");
            progress_queue.put(-1.0); return None

        # Check for cancellation before allocating large memory
        if cancel_event.is_set():
             log_queue.put("--- Processing Cancelled by User (Before Pass 2) ---")
             progress_queue.put(-1.0)
             return None

        min_sx=min(c[0] for c in all_coords); max_sx=max(c[0] for c in all_coords)
        min_sy=min(c[1] for c in all_coords); max_sy=max(c[1] for c in all_coords)
        log_queue.put(f"Coord Range: X=[{min_sx}-{max_sx}], Y=[{min_sy}-{max_sy}]")

        current_overlap = max(0, min(sector_overlap, min(LAYOUT_SECTOR_WIDTH, LAYOUT_SECTOR_HEIGHT) - 1))
        current_blend_size = max(0, min(boundary_blend_size, min(LAYOUT_SECTOR_WIDTH // 2, LAYOUT_SECTOR_HEIGHT // 2)))
        if current_overlap != sector_overlap: log_queue.put(f"W: Clamped Overlap to {current_overlap}")
        if current_blend_size != boundary_blend_size: log_queue.put(f"W: Clamped Blend Size to {current_blend_size}")

        effective_sector_width = LAYOUT_SECTOR_WIDTH - current_overlap
        effective_sector_height = LAYOUT_SECTOR_HEIGHT - current_overlap
        if effective_sector_width <= 0 or effective_sector_height <= 0:
            log_queue.put(f"E: Overlap ({current_overlap}) too large."); progress_queue.put(-1.0); return None

        num_sectors_x = max_sx - min_sx + 1; num_sectors_y = max_sy - min_sy + 1
        final_width = (num_sectors_x - 1) * effective_sector_width + LAYOUT_SECTOR_WIDTH if num_sectors_x > 0 else 0
        final_height = (num_sectors_y - 1) * effective_sector_height + LAYOUT_SECTOR_HEIGHT if num_sectors_y > 0 else 0
        if num_sectors_x <= 1: final_width = LAYOUT_SECTOR_WIDTH
        if num_sectors_y <= 1: final_height = LAYOUT_SECTOR_HEIGHT
        final_width = max(0, final_width); final_height = max(0, final_height)

        log_queue.put(f"Final Map: {final_width} x {final_height} (Overlap={current_overlap}, Blend={current_blend_size})")
        if final_width <= 0 or final_height <= 0: log_queue.put("E: Map dim invalid."); progress_queue.put(-1.0); return None

        try:
            estimated_gb = (final_width * final_height * 8 * 2) / (1024**3)
            log_queue.put(f"Allocating map arrays (~{estimated_gb:.2f} GB)...")
            heightmap_sum_array = np.zeros((final_height, final_width), dtype=np.float64)
            weight_sum_array = np.zeros((final_height, final_width), dtype=np.float64)
            log_queue.put("Allocation OK.")
        except MemoryError:
             log_queue.put(f"E: Memory allocation failed."); progress_queue.put(-1.0); return None
        except Exception as e:
             log_queue.put(f"E: Numpy array error: {e}"); progress_queue.put(-1.0); return None

        valid_base_heights = [data['base_h'] for data in sectors_data.values() if 'base_h' in data and data['base_h'] is not None]
        global_base = np.mean(valid_base_heights) if valid_base_heights else 0.0
        log_queue.put(f"Global Avg Base Height: {global_base:.2f}")

        log_queue.put("--- Pass 2: Calculating heights, smoothing, blending ---")
        placed_count = 0; overall_min_h = float('inf'); overall_max_h = float('-inf'); first_sector_processed = False
        total_sectors_to_place = len(sectors_data)

        static_weight_map = create_weight_map(LAYOUT_SECTOR_HEIGHT, LAYOUT_SECTOR_WIDTH, current_blend_size)
        if static_weight_map is None: log_queue.put("E: Failed to create weight map."); progress_queue.put(-1.0); return None

        for i, ((sx, sy), data) in enumerate(sectors_data.items()):
             # Check for cancellation periodically during Pass 2
            if i % 10 == 0 and cancel_event.is_set(): # Check every 10 sectors blended
                log_queue.put("--- Processing Cancelled by User (During Pass 2) ---")
                progress_queue.put(-1.0)
                # Clean up potentially large arrays before returning
                del heightmap_sum_array, weight_sum_array, static_weight_map
                return None

            sector_w = data['width']; sector_h = data['height']
            local_base_h = data.get('base_h', global_base)
            absolute_height_sector_float64 = (data['detiled_offsets'].astype(np.float64) / height_scale_factor) + local_base_h
            if apply_boundary_smoothing:
                absolute_height_sector_float = smooth_tile_boundaries(absolute_height_sector_float64, TILE_WIDTH, TILE_HEIGHT, log_queue).astype(np.float32)
            else:
                absolute_height_sector_float = absolute_height_sector_float64.astype(np.float32)

            sector_min = np.nanmin(absolute_height_sector_float); sector_max = np.nanmax(absolute_height_sector_float)
            if np.isfinite(sector_min) and np.isfinite(sector_max):
                 overall_min_h = min(overall_min_h, sector_min); overall_max_h = max(overall_max_h, sector_max); first_sector_processed = True
            else: log_queue.put(f" W: ({sx},{sy}) non-finite heights.")

            weight_map_to_use = static_weight_map
            if weight_map_to_use.shape != (sector_h, sector_w): log_queue.put(f" E: Weight map shape mismatch ({sx},{sy}). Skip."); continue

            grid_x = sx - min_sx; grid_y = sy - min_sy
            paste_x_start=grid_x*effective_sector_width; paste_y_start=grid_y*effective_sector_height
            target_y_start_clipped=max(0,paste_y_start); target_y_end_clipped=min(final_height,paste_y_start+sector_h)
            target_x_start_clipped=max(0,paste_x_start); target_x_end_clipped=min(final_width,paste_x_start+sector_w)
            if target_y_start_clipped >= target_y_end_clipped or target_x_start_clipped >= target_x_end_clipped: continue
            clip_top=target_y_start_clipped-paste_y_start; clip_left=target_x_start_clipped-paste_x_start
            clipped_height=target_y_end_clipped-target_y_start_clipped; clipped_width=target_x_end_clipped-target_x_start_clipped
            source_y_start=clip_top; source_y_end=clip_top+clipped_height
            source_x_start=clip_left; source_x_end=clip_left+clipped_width
            target_slice=np.s_[target_y_start_clipped:target_y_end_clipped, target_x_start_clipped:target_x_end_clipped]
            source_slice=np.s_[source_y_start:source_y_end, source_x_start:source_x_end]

            if absolute_height_sector_float[source_slice].shape != (clipped_height,clipped_width) or weight_map_to_use[source_slice].shape != (clipped_height,clipped_width):
                log_queue.put(f" E: Slice mismatch ({sx},{sy})!"); continue

            try:
                source_heights = absolute_height_sector_float[source_slice]
                source_weights = weight_map_to_use[source_slice].astype(np.float64)
                valid_mask = np.isfinite(source_heights)
                heightmap_sum_array[target_slice][valid_mask] += source_heights[valid_mask].astype(np.float64) * source_weights[valid_mask]
                weight_sum_array[target_slice][valid_mask] += source_weights[valid_mask]
                placed_count += 1
            except Exception as e: log_queue.put(f" E: Weighted add err ({sx},{sy}): {e}"); continue

            if i % 10 == 0 or i == total_sectors_to_place - 1:
                 progress_queue.put(0.3 + 0.6 * (i + 1) / total_sectors_to_place)

        log_queue.put(f"Pass 2 Done. Blended {placed_count} sectors.")
        if placed_count == 0: log_queue.put("E: No sectors placed."); progress_queue.put(-1.0); return None

        # Check for cancellation before final processing/saving
        if cancel_event.is_set():
             log_queue.put("--- Processing Cancelled by User (Before Pass 3) ---")
             progress_queue.put(-1.0)
             # Clean up potentially large arrays before returning
             del heightmap_sum_array, weight_sum_array, static_weight_map
             return None

        log_queue.put("--- Pass 3: Finalizing map and saving outputs ---")
        if not first_sector_processed or not np.isfinite(overall_min_h) or not np.isfinite(overall_max_h):
             log_queue.put("W: No valid overall height range found across sectors. Calculating from sum...")
             valid_mask_tmp = weight_sum_array > 1e-9
             if np.any(valid_mask_tmp):
                  safe_divisor = np.where(valid_mask_tmp, weight_sum_array, 1.0)
                  safe_numerator = np.where(valid_mask_tmp, heightmap_sum_array, 0.0)
                  calculated_heights = safe_numerator[valid_mask_tmp] / safe_divisor[valid_mask_tmp]
                  temp_min = np.nanmin(calculated_heights); temp_max = np.nanmax(calculated_heights)
                  if np.isfinite(temp_min) and np.isfinite(temp_max): overall_min_h=temp_min; overall_max_h=temp_max; log_queue.put(f" W: Using sum range: [{overall_min_h:.2f}, {overall_max_h:.2f}]")
                  else: overall_min_h=0.0; overall_max_h=1.0; log_queue.put(" W: Could not determine range...")
             else: overall_min_h=0.0; overall_max_h=1.0; log_queue.put(" W: No valid weights...")
        else: log_queue.put(f"Overall Height Range: [{overall_min_h:.2f}, {overall_max_h:.2f}]")

        log_queue.put("Calculating final heights...")
        final_heightmap_float = np.full((final_height, final_width), np.nan, dtype=np.float32)
        valid_weights_mask = weight_sum_array > 1e-9
        np.divide(heightmap_sum_array, weight_sum_array, out=final_heightmap_float, where=valid_weights_mask)
        del heightmap_sum_array, weight_sum_array

        log_queue.put("Normalizing heightmap for PNG output...")
        norm_min = overall_min_h; norm_max = overall_max_h
        log_queue.put(f"Norm Range Used (PNG): [{norm_min:.2f}, {norm_max:.2f}] -> [0, 255]")

        DEFAULT_BG_COLOR_FLOAT = norm_min
        final_heightmap_filled = np.nan_to_num(final_heightmap_float, nan=DEFAULT_BG_COLOR_FLOAT)

        norm_range = norm_max - norm_min
        if norm_range <= 1e-9:
            log_queue.put(" W: Normalization range is zero. PNG will be flat.")
            heightmap_8bit = np.full((final_height, final_width), 0, dtype=np.uint8)
        else:
            normalized_map = (final_heightmap_filled - norm_min) / norm_range
            np.clip(normalized_map, 0.0, 1.0, out=normalized_map)
            heightmap_8bit = (normalized_map * 255.0).astype(np.uint8)

        progress_queue.put(0.95)

        try:
             # Check for cancellation one last time before saving
            if cancel_event.is_set():
                 log_queue.put("--- Processing Cancelled by User (Before Save) ---")
                 progress_queue.put(-1.0)
                 del final_heightmap_float, heightmap_8bit # Clean up
                 return None

            log_queue.put(f"Saving normalized heightmap PNG to {os.path.basename(output_file_path_png)}...")
            img = Image.fromarray(heightmap_8bit, mode='L')
            img.save(output_file_path_png)

            log_queue.put("\n--- Processing Complete ---")
            log_queue.put(f"Outputs saved with base name: {base_output_name}")
            log_queue.put(f"Final map size: {final_width} x {final_height}")
            log_queue.put(f"PNG Normalization Applied: [{norm_min:.2f}, {norm_max:.2f}] -> [0, 255]")
            progress_queue.put(1.0)

        except Exception as e:
            log_queue.put(f"\nE: Saving final image: {e}"); traceback.print_exc(); progress_queue.put(-1.0)
            # Check if cancel was requested during save attempt (less likely but possible)
            if cancel_event.is_set():
                log_queue.put("--- NOTE: Cancellation was also requested around save time ---")
            return None

        return final_heightmap_float

    except Exception as e:
        log_queue.put(f"\n--- UNEXPECTED ERROR in generate_heightmap ---")
        log_queue.put(f"Error: {e}")
        log_queue.put(traceback.format_exc())
        # Check if cancel was requested during the error
        if cancel_event and cancel_event.is_set():
             log_queue.put("--- NOTE: Cancellation was also requested around error time ---")
        progress_queue.put(-1.0)
        return None