import sys
import math
import numpy as np
import pyrr  # For 3D math (vectors, matrices)
from OpenGL.GL import * # PyOpenGL for GL calls
import imgui_bundle  # For window, context, and UI widgets
from imgui_bundle import imgui, hello_imgui, ImVec2, ImVec4
import traceback
import ctypes # For glVertexAttribPointer

# --- NEW: Imports for File Handling and Parsing ---
import os
import zipfile
import struct
import time # For basic timing/logging
import re # <-- Import regex for filename parsing

# --- Configuration Constants ---
# Use the user-provided default input folder
INPUT_DIR = r"C:\Program Files (x86)\Steam\steamapps\common\Sacred 2 Gold\pak"
SECTOR_SUFFIX = "_b0.sector"
SECTOR_MARKER = b'\x00\x00\x01\x00'
# Bytes to skip AFTER the marker to reach vertex data start
# (This might need adjustment based on exact file format)
HEADER_SKIP_AFTER_MARKER = 12
VERTEX_ENTRY_SIZE = 8  # 8 bytes per vertex entry
HEIGHT_OFFSET_IN_ENTRY = 6 # Start byte index for the height offset
HEIGHT_STRUCT_FORMAT = '<H' # Little-endian unsigned 16-bit integer

SECTOR_DIM = 224 # Vertices per side
TILE_DIM = 7     # Vertices per tile side
NUM_TILES_PER_SIDE = SECTOR_DIM // TILE_DIM # Should be 32
VERTICES_PER_TILE = TILE_DIM * TILE_DIM # Should be 49
TOTAL_VERTICES_PER_SECTOR = SECTOR_DIM * SECTOR_DIM # Should be 50176

# --- Mesh Generation Parameters ---
BASE_HEIGHT = 0.0 # Assume base height is 0 for now, adjust if needed
# Adjust Y_SCALE to control vertical exaggeration of terrain
Y_SCALE = 1 / 32.0
# Adjust XZ_SCALE to control horizontal spacing of vertices
XZ_SCALE = 1.0
DEFAULT_TERRAIN_COLOR = [0.6, 0.7, 0.6] # Slightly lighter base green ## MODIFIED BASE COLOR ##

# --- Render Distance & Spawn Configuration ---
INITIAL_RENDER_DISTANCE = 500.0 # Initial view distance in world units
MIN_RENDER_DISTANCE = 100.0
MAX_RENDER_DISTANCE = 5000.0 # Adjust max as needed
SPAWN_SECTOR_X = 21
SPAWN_SECTOR_Z = 27
SPAWN_Y_OFFSET = 50.0 # How high above the estimated sector center to spawn

# --- Lighting Configuration --- ## NEW ##
LIGHT_DIRECTION = pyrr.Vector3([-0.7, -0.8, -0.6]) # Direction *towards* light source
LIGHT_COLOR = pyrr.Vector3([1.0, 1.0, 0.95])     # Slightly warm white light
AMBIENT_STRENGTH = 0.25
DIFFUSE_STRENGTH = 0.8
SPECULAR_STRENGTH = 0.2  # Lower specular for terrain typically
SHININESS = 16.0        # Lower shininess for diffuse surfaces

# --- Global State ---
class AppState:
    def __init__(self):
        self.shader_program = None
        self.vao_axes = None
        self.vbo_axes = None

        # --- Store lighting uniform locations --- ## NEW ##
        self.loc_model = -1
        self.loc_view = -1
        self.loc_projection = -1
        self.loc_light_dir = -1
        self.loc_light_color = -1
        self.loc_view_pos = -1
        self.loc_ambient = -1
        self.loc_diffuse = -1
        self.loc_specular = -1
        self.loc_shininess = -1

        # --- Terrain Data Storage ---
        # Dictionary: sector_name -> {'vao': id, 'vbo': id, 'ebo': id, 'index_count': int,
        #                              'sector_offset': pyrr.Vector3, 'world_center': pyrr.Vector3}
        self.terrain_meshes = {}
        self.loading_status = "Idle"
        self.sectors_rendered_last_frame = 0

        # --- Render distance state ---
        self.render_distance = INITIAL_RENDER_DISTANCE
        self.render_distance_squared = INITIAL_RENDER_DISTANCE * INITIAL_RENDER_DISTANCE

        # --- Camera/View Variables ---
        spawn_world_x = (SPAWN_SECTOR_X + 0.5) * SECTOR_DIM * XZ_SCALE
        spawn_world_z = (SPAWN_SECTOR_Z + 0.5) * SECTOR_DIM * XZ_SCALE
        self.camera_position = pyrr.Vector3([spawn_world_x, SPAWN_Y_OFFSET, spawn_world_z]) # Set spawn position
        self.camera_front = pyrr.Vector3([0.0, -0.3, -1.0]) # Look slightly down
        self.camera_up = pyrr.Vector3([0.0, 1.0, 0.0])
        # camera_right derived later
        self.camera_speed = 100.0 # Increased speed
        self.mouse_sensitivity = 0.002
        self.yaw = math.radians(-90.0) # Facing -Z
        self.pitch = math.radians(-15.0) # Look slightly down
        self.fov_y = 60.0 # Wider FOV
        self.near_plane = 0.1
        self.far_plane = self.render_distance * 1.5 # Adjust far plane

        self.last_mouse_pos = None
        self.is_looking = False
        self.update_camera_vectors()

    def update_camera_vectors(self):
        # (Keep this function as is)
        front = np.array([0.0, 0.0, 0.0]); front[0] = math.cos(self.yaw) * math.cos(self.pitch); front[1] = math.sin(self.pitch); front[2] = math.sin(self.yaw) * math.cos(self.pitch)
        self.camera_front = pyrr.vector.normalize(front); world_up = pyrr.Vector3([0.0, 1.0, 0.0]); self.camera_right = pyrr.vector.normalize(pyrr.vector3.cross(self.camera_front, world_up))
        self.camera_up = pyrr.vector.normalize(pyrr.vector3.cross(self.camera_right, self.camera_front))

    # --- Cleanup Function --- ## SYNTAX FIXED ##
    def cleanup_gl_resources(self):
        print("Cleaning up GL resources...")
        # Check if function exists before calling (safer during shutdown/errors)
        _glDeleteProgram = getattr(OpenGL.GL, "glDeleteProgram", None)
        _glDeleteVertexArrays = getattr(OpenGL.GL, "glDeleteVertexArrays", None)
        _glDeleteBuffers = getattr(OpenGL.GL, "glDeleteBuffers", None)

        if self.shader_program and _glDeleteProgram:
            try: # Moved try/except to separate lines
                _glDeleteProgram(self.shader_program)
            except Exception as e:
                print(f"Ignoring error during glDeleteProgram: {e}")
            self.shader_program = None # Clear ref

        if self.vao_axes and _glDeleteVertexArrays:
            try: # Moved try/except to separate lines
                _glDeleteVertexArrays(1, [self.vao_axes])
            except Exception as e:
                print(f"Ignoring error during glDeleteVertexArrays(axes): {e}")
            self.vao_axes = None

        if self.vbo_axes and _glDeleteBuffers:
            try: # Moved try/except to separate lines
                _glDeleteBuffers(1, [self.vbo_axes])
            except Exception as e:
                print(f"Ignoring error during glDeleteBuffers(axes): {e}")
            self.vbo_axes = None

        # Cleanup terrain meshes
        if _glDeleteVertexArrays and _glDeleteBuffers:
            for mesh_data in self.terrain_meshes.values():
                try: # Moved try/except to separate lines
                    if 'vao' in mesh_data and mesh_data['vao']:
                         _glDeleteVertexArrays(1, [mesh_data['vao']])
                    if 'vbo' in mesh_data and mesh_data['vbo']:
                        _glDeleteBuffers(1, [mesh_data['vbo']])
                    if 'ebo' in mesh_data and mesh_data['ebo']:
                         _glDeleteBuffers(1, [mesh_data['ebo']])
                except Exception as e:
                    print(f"Ignoring error during terrain mesh cleanup: {e}")
        self.terrain_meshes = {} # Clear python refs regardless
        print("Cleanup attempt finished.")


app_state = AppState()

# --- Shader Sources --- ## MODIFIED for Lighting ##
VERTEX_SHADER_SOURCE_LIGHTING = """
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aColor;     // Base color (set to location 1)
layout (location = 2) in vec3 aNormal;    // Normal vector (set to location 2)

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

// Outputs to fragment shader
out vec3 v_FragPos_World;  // Vertex position in world space
out vec3 v_Normal_World;   // Normal vector in world space
out vec3 v_Color;          // Base color passed through

void main()
{
    // Calculate world position of the vertex
    v_FragPos_World = vec3(model * vec4(aPos, 1.0));

    // Calculate world normal: needs inverse transpose of model matrix's 3x3 part
    // to handle non-uniform scaling correctly. Normalizing here might be redundant
    // as it will be normalized per-fragment anyway, but can help interpolation.
    v_Normal_World = normalize(mat3(transpose(inverse(model))) * aNormal);

    // Pass base color through
    v_Color = aColor;

    // Calculate final clip space position
    gl_Position = projection * view * vec4(v_FragPos_World, 1.0);
}
"""

FRAGMENT_SHADER_SOURCE_LIGHTING = """
#version 330 core
out vec4 FragColorOut;

// Inputs from vertex shader (interpolated)
in vec3 v_FragPos_World;
in vec3 v_Normal_World;
in vec3 v_Color;         // Base color from vertex data

// Lighting Uniforms
uniform vec3 u_ViewPos;         // Camera position in world space
uniform vec3 u_LightDirection;  // Direction *towards* the light source (in world space)
uniform vec3 u_LightColor;
uniform float u_AmbientStrength;
uniform float u_DiffuseStrength;
uniform float u_SpecularStrength;
uniform float u_Shininess;

void main()
{
    // Ensure normal is normalized after interpolation
    vec3 norm = normalize(v_Normal_World);

    // Ambient component
    vec3 ambient = u_AmbientStrength * u_LightColor;

    // Diffuse component
    vec3 lightDir = normalize(-u_LightDirection); // Direction from surface point TO light source
    float diff = max(dot(norm, lightDir), 0.0);   // Lambertian factor
    vec3 diffuse = u_DiffuseStrength * diff * u_LightColor;

    // Specular component (Phong)
    vec3 viewDir = normalize(u_ViewPos - v_FragPos_World); // Direction from surface point TO camera
    vec3 reflectDir = reflect(-lightDir, norm);            // Direction of reflected light
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), u_Shininess); // Specular intensity
    vec3 specular = u_SpecularStrength * spec * u_LightColor;

    // Combine lighting components and modulate with base vertex color
    vec3 result = (ambient + diffuse + specular) * v_Color;
    FragColorOut = vec4(result, 1.0);
}
"""

# --- Shader Compilation & Linking (Keep As Is) ---
def compile_shader(source, shader_type):
    shader = glCreateShader(shader_type); glShaderSource(shader, source); glCompileShader(shader)
    if not glGetShaderiv(shader, GL_COMPILE_STATUS): raise RuntimeError(f"Shader {shader_type} compile error: {glGetShaderInfoLog(shader).decode()}")
    return shader
def create_shader_program(vertex_source, fragment_source):
    v = compile_shader(vertex_source, GL_VERTEX_SHADER); f = compile_shader(fragment_source, GL_FRAGMENT_SHADER)
    p = glCreateProgram(); glAttachShader(p, v); glAttachShader(p, f); glLinkProgram(p)
    if not glGetProgramiv(p, GL_LINK_STATUS): raise RuntimeError(f"Shader link error: {glGetProgramInfoLog(p).decode()}")
    glDeleteShader(v); glDeleteShader(f); return p

# --- Sector Parsing and Detiling (Keep As Is) ---
def parse_sector_and_detile(data: bytes) -> np.ndarray | None:
    # (Implementation is the same as before)
    try:
        marker_pos=data.find(SECTOR_MARKER);
        if marker_pos==-1:return None
        vertex_data_start=marker_pos+len(SECTOR_MARKER)+HEADER_SKIP_AFTER_MARKER;expected_data_size=TOTAL_VERTICES_PER_SECTOR*VERTEX_ENTRY_SIZE;available_data_size=len(data)-vertex_data_start;
        if available_data_size<expected_data_size:return None
        heightmap=np.zeros((SECTOR_DIM,SECTOR_DIM),dtype=np.float32);vertex_stream_offset=vertex_data_start;num_tiles_total=NUM_TILES_PER_SIDE*NUM_TILES_PER_SIDE;
        for tile_idx in range(num_tiles_total):
            tile_z=tile_idx//NUM_TILES_PER_SIDE;tile_x=tile_idx%NUM_TILES_PER_SIDE
            for v_in_tile_idx in range(VERTICES_PER_TILE):
                entry_start=vertex_stream_offset+v_in_tile_idx*VERTEX_ENTRY_SIZE;entry_bytes=data[entry_start:entry_start+VERTEX_ENTRY_SIZE]
                if len(entry_bytes)<VERTEX_ENTRY_SIZE:continue
                height_offset=struct.unpack(HEIGHT_STRUCT_FORMAT,entry_bytes[HEIGHT_OFFSET_IN_ENTRY:HEIGHT_OFFSET_IN_ENTRY+2])[0];final_y=BASE_HEIGHT+(float(height_offset)*Y_SCALE);v_in_tile_z=v_in_tile_idx//TILE_DIM;v_in_tile_x=v_in_tile_idx%TILE_DIM;dest_x=tile_x*TILE_DIM+v_in_tile_x;dest_z=tile_z*TILE_DIM+v_in_tile_z
                if 0<=dest_x<SECTOR_DIM and 0<=dest_z<SECTOR_DIM:heightmap[dest_z,dest_x]=final_y
            vertex_stream_offset+=VERTICES_PER_TILE*VERTEX_ENTRY_SIZE
        return heightmap
    except Exception as e: return None

# --- Mesh Generation --- ## MODIFIED to Calculate Normals ##
def create_mesh_from_heightmap(heightmap: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Generates vertices (pos+normal+color) and indices for a 3D mesh from a heightmap."""
    try:
        sector_z_dim, sector_x_dim = heightmap.shape
        if sector_x_dim != SECTOR_DIM or sector_z_dim != SECTOR_DIM: return None

        positions = np.zeros((sector_z_dim, sector_x_dim, 3), dtype=np.float32)
        colors = np.zeros((sector_z_dim, sector_x_dim, 3), dtype=np.float32)
        normals = np.zeros((sector_z_dim, sector_x_dim, 3), dtype=np.float32)

        # Generate positions and default colors
        for z in range(sector_z_dim):
            for x in range(sector_x_dim):
                positions[z, x] = [x * XZ_SCALE, heightmap[z, x], z * XZ_SCALE]
                colors[z, x] = DEFAULT_TERRAIN_COLOR

        # Calculate normals using finite differences (simple method)
        for z in range(sector_z_dim):
            for x in range(sector_x_dim):
                y_l = heightmap[z, max(0, x-1)]                         # Left
                y_r = heightmap[z, min(sector_x_dim-1, x+1)]           # Right
                y_d = heightmap[max(0, z-1), x]                         # Down (-Z direction)
                y_u = heightmap[min(sector_z_dim-1, z+1), x]           # Up (+Z direction)
                norm_x = -(y_r - y_l) * (2.0 * XZ_SCALE)
                norm_y = (2.0 * XZ_SCALE) * (2.0 * XZ_SCALE)
                norm_z = -(y_u - y_d) * (2.0 * XZ_SCALE)
                norm = np.array([norm_x, norm_y, norm_z])
                length = np.linalg.norm(norm)
                if length > 1e-6: normals[z, x] = norm / length
                else: normals[z, x] = [0.0, 1.0, 0.0] # Default up

        # --- Interleave data: [x, y, z, nx, ny, nz, r, g, b] ---
        num_vertices = sector_z_dim * sector_x_dim
        pos_flat = positions.reshape(num_vertices, 3)
        norm_flat = normals.reshape(num_vertices, 3)
        col_flat = colors.reshape(num_vertices, 3)
        vertices_interleaved = np.hstack((pos_flat, norm_flat, col_flat)).flatten()
        vertices_np = np.array(vertices_interleaved, dtype=np.float32)

        # Generate indices (same as before)
        indices = []
        for z in range(sector_z_dim - 1):
            for x in range(sector_x_dim - 1):
                topLeft = z * sector_x_dim + x; topRight = topLeft + 1
                bottomLeft = (z + 1) * sector_x_dim + x; bottomRight = bottomLeft + 1
                indices.extend([topLeft, bottomLeft, topRight, topRight, bottomLeft, bottomRight])
        indices_np = np.array(indices, dtype=np.uint32)

        return vertices_np, indices_np

    except Exception as e: print(f"Error creating mesh with normals: {e}"); traceback.print_exc(); return None

# --- Terrain Loading Function --- ## MODIFIED for New Vertex Attributes ##
# --- Terrain Loading Function --- ## MODIFIED for New Vertex Attributes and Sector X = 10-30 filter ##
def load_terrain_sectors():
    global app_state
    # Updated status message for the new filter range
    print(f"--- Starting Terrain Load ---"); app_state.loading_status="Scanning (Filter: 10 <= sector_x <= 30)..."; start_time=time.time(); sectors_processed=0; sectors_added=0; archives_found=0; files_in_archives=0
    if not os.path.isdir(INPUT_DIR): print(f"Error: Dir not found: {INPUT_DIR}"); app_state.loading_status=f"Error: Dir not found"; return
    sector_coord_regex = re.compile(r"(\d+)_(\d+)_b0\.sector$", re.IGNORECASE) # Corrected Regex

    for root, _, files in os.walk(INPUT_DIR):
        for file in files:
            if file.lower().endswith(('.zip', '.pak')):
                archives_found += 1; archive_path = os.path.join(root, file)
                try:
                    with zipfile.ZipFile(archive_path, 'r') as zf:
                        for sector_name in zf.namelist():
                            files_in_archives += 1
                            match = sector_coord_regex.search(sector_name)
                            if match:
                                sectors_processed += 1 # Count all potential sectors found by regex
                                if sector_name in app_state.terrain_meshes: continue
                                try:
                                    sector_x_str, sector_z_str = match.groups()
                                    sector_x = int(sector_x_str)

                                    # --- !!! MODIFIED FILTER CONDITION HERE !!! ---
                                    if 10 <= sector_x <= 30: # Check if sector_x is between 10 and 30 (inclusive)
                                        # Only proceed if the condition is met
                                        sector_z = int(sector_z_str)
                                        sector_offset = pyrr.Vector3([sector_x*SECTOR_DIM*XZ_SCALE, 0.0, sector_z*SECTOR_DIM*XZ_SCALE])
                                        world_center = sector_offset + pyrr.Vector3([SECTOR_DIM*XZ_SCALE/2.0, 0.0, SECTOR_DIM*XZ_SCALE/2.0])
                                        raw_data = zf.read(sector_name); heightmap = parse_sector_and_detile(raw_data)
                                        if heightmap is not None:
                                            mesh_data = create_mesh_from_heightmap(heightmap) # Now includes normals
                                            if mesh_data is not None:
                                                vertices, indices = mesh_data
                                                vao=glGenVertexArrays(1); vbo=glGenBuffers(1); ebo=glGenBuffers(1)
                                                glBindVertexArray(vao)
                                                glBindBuffer(GL_ARRAY_BUFFER, vbo); glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
                                                glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo); glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)

                                                # --- UPDATED Vertex Attribute Layout ---
                                                stride = 9 * vertices.itemsize # Stride is 9 floats
                                                glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
                                                glEnableVertexAttribArray(0)
                                                glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(6 * vertices.itemsize))
                                                glEnableVertexAttribArray(1)
                                                glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(3 * vertices.itemsize))
                                                glEnableVertexAttribArray(2)
                                                # --- END Updated Vertex Attribute Layout ---

                                                glBindVertexArray(0)
                                                print(f"Successfully loaded mesh for sector: {sector_name}")
                                                app_state.terrain_meshes[sector_name] = {'vao':vao,'vbo':vbo,'ebo':ebo,'index_count':len(indices),'sector_offset':sector_offset,'world_center':world_center}
                                                sectors_added += 1 # Increment added count only if x is in range and successful
                                    # --- !!! END OF MODIFIED FILTER BLOCK !!! ---

                                except ValueError: pass # Catch potential int conversion errors
                                except Exception as parse_err: print(f"Error processing {sector_name}: {parse_err}") # Report errors for sectors that matched regex
                except zipfile.BadZipFile: pass
                except Exception as archive_err: print(f" Error opening archive {archive_path}: {archive_err}")
    end_time = time.time(); load_duration = end_time - start_time
    # Updated summary message slightly to reflect filtering
    summary = f"Finished scan ({load_duration:.2f}s). Found {archives_found} archives, checked {files_in_archives} files. Processed {sectors_processed} potential sectors (filtered for 10<=x<=30), added {sectors_added} meshes."
    print(f"\n--- Terrain Load Summary ---"); print(summary)
    # Updated status message for the new filter range
    app_state.loading_status = f"Finished: {sectors_added} meshes loaded (10<=x<=30 filter, {load_duration:.2f}s)"


# --- OpenGL Resource Initialization --- ## MODIFIED for Lighting Shaders/Uniforms ##
def initialize_opengl_resources():
    global app_state
    try:
        # Create shader program using the NEW lighting shaders
        app_state.shader_program = create_shader_program(VERTEX_SHADER_SOURCE_LIGHTING, FRAGMENT_SHADER_SOURCE_LIGHTING) # Use new shaders
        print("Lighting shader program created.")

        # Get uniform locations (do this once after linking)
        app_state.loc_model = glGetUniformLocation(app_state.shader_program, "model")
        app_state.loc_view = glGetUniformLocation(app_state.shader_program, "view")
        app_state.loc_projection = glGetUniformLocation(app_state.shader_program, "projection")
        app_state.loc_light_dir = glGetUniformLocation(app_state.shader_program, "u_LightDirection")
        app_state.loc_light_color = glGetUniformLocation(app_state.shader_program, "u_LightColor")
        app_state.loc_view_pos = glGetUniformLocation(app_state.shader_program, "u_ViewPos")
        app_state.loc_ambient = glGetUniformLocation(app_state.shader_program, "u_AmbientStrength")
        app_state.loc_diffuse = glGetUniformLocation(app_state.shader_program, "u_DiffuseStrength")
        app_state.loc_specular = glGetUniformLocation(app_state.shader_program, "u_SpecularStrength")
        app_state.loc_shininess = glGetUniformLocation(app_state.shader_program, "u_Shininess")
        print("Uniform locations retrieved.")

        # --- Axes VAO needs update for new vertex format (Pos, Normal, Color) ---
        axes_vertices = np.array([
            # Pos          # Normal (dummy) # Color
            0.,0.,0.,      0.,1.,0.,      1.,0.,0., # O Red
            1.,0.,0.,      0.,1.,0.,      1.,0.,0., # X Red
            0.,0.,0.,      0.,1.,0.,      0.,1.,0., # O Green
            0.,1.,0.,      0.,1.,0.,      0.,1.,0., # Y Green
            0.,0.,0.,      0.,1.,0.,      0.,0.,1., # O Blue
            0.,0.,1.,      0.,1.,0.,      0.,0.,1., # Z Blue
        ], dtype=np.float32) # Now 9 floats per vertex
        app_state.vao_axes = glGenVertexArrays(1); app_state.vbo_axes = glGenBuffers(1)
        glBindVertexArray(app_state.vao_axes); glBindBuffer(GL_ARRAY_BUFFER, app_state.vbo_axes)
        glBufferData(GL_ARRAY_BUFFER, axes_vertices.nbytes, axes_vertices, GL_STATIC_DRAW)
        stride = 9 * axes_vertices.itemsize # Stride updated to 9
        # Position attribute (location = 0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0)); glEnableVertexAttribArray(0)
        # Color attribute (location = 1) - offset updated
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(6 * axes_vertices.itemsize)); glEnableVertexAttribArray(1)
        # Normal attribute (location = 2) - added
        glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(3 * axes_vertices.itemsize)); glEnableVertexAttribArray(2)
        glBindBuffer(GL_ARRAY_BUFFER, 0); glBindVertexArray(0)

        app_state.loading_status = "Ready to load terrain"
        print("OpenGL resources initialized (terrain not loaded automatically).")
    except Exception as e:
        print(f"Error initializing OpenGL: {e}"); traceback.print_exc()
        app_state.shader_program = None; app_state.loading_status = "OpenGL Init Failed!"


# --- Camera Input Handling --- ## BLOCK MOVED & Syntax Corrected ##
# Try importing GLFW and getting the window function FIRST
# --- Start Moved Block ---
try:
    import glfw
    # Define get_glfw_window function (attempt various sources)
    try:
        from imgui_bundle import glfw_utils
        get_glfw_window = glfw_utils.glfw_window_hello_imgui
        # print("DEBUG: Using imgui_bundle.glfw_utils.glfw_window_hello_imgui")
    except (ImportError, AttributeError):
        if hasattr(hello_imgui, 'get_glfw_window'):
            get_glfw_window = hello_imgui.get_glfw_window
            # print("DEBUG: Using hello_imgui.get_glfw_window")
        else:
            get_glfw_window = lambda: None
            # print("DEBUG: No GLFW window getter function found.")

    # Check if the getter function itself exists. Calling it is deferred.
    # We assume import success if the glfw module itself imported.
    GLFW_IMPORT_SUCCESS = True
    # A check for the window function can be done later if needed, but
    # setting the flag based on import is sufficient for the dummy class logic.

except ImportError:
    # --- Corrected Dummy Definitions ---
    print("WARNING: GLFW not found (pip install glfw). Mouselook disabled.")
    GLFW_IMPORT_SUCCESS = False
    # Define dummy class and functions on separate lines
    class glfw:
        CURSOR=0
        CURSOR_DISABLED=0
        CURSOR_NORMAL=0
        @staticmethod
        def set_input_mode(w,m,v): pass # Dummy method
    # Assign dummy function after class definition
    get_glfw_window = lambda: None
    # --- End Corrected Block ---
# --- End Moved Block ---

def handle_camera_input(io, window_width, window_height):
    # --- Start function definition ---
    global app_state
    is_rmb_down = imgui.is_mouse_down(1); shift_down = imgui.is_key_down(imgui.Key.left_shift) or imgui.is_key_down(imgui.Key.right_shift)
    current_mouse_pos = ImVec2(io.mouse_pos.x, io.mouse_pos.y)

    # Determine desired state FIRST
    should_be_looking = is_rmb_down and not shift_down

    # State transition: Start Looking
    if should_be_looking and not app_state.is_looking:
        app_state.is_looking = True
        app_state.last_mouse_pos = current_mouse_pos # Store pos before potential capture
        if GLFW_IMPORT_SUCCESS:
            window = get_glfw_window() # Get handle only when needed
            if window: # Check if handle is valid
                # --- Corrected Try/Except Block 1 --- ## SYNTAX FIXED ##
                try:
                    glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_DISABLED)
                except Exception as e:
                    print(f"DEBUG: GLFW Error disabling cursor: {e}")
                # --- End Corrected Block 1 ---
        imgui.set_mouse_cursor(-1) # Suggest hide

    # State transition: Stop Looking
    elif not should_be_looking and app_state.is_looking:
        app_state.is_looking = False
        if GLFW_IMPORT_SUCCESS:
            window = get_glfw_window() # Get handle only when needed
            if window: # Check if handle is valid
                # --- Corrected Try/Except Block 2 --- ## SYNTAX FIXED ##
                try:
                    glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_NORMAL)
                except Exception as e:
                    print(f"DEBUG: GLFW Error enabling cursor: {e}")
                # --- End Corrected Block 2 ---
        imgui.set_mouse_cursor(0) # Suggest arrow

    # Process input based on current state
    if app_state.is_looking:
         if app_state.last_mouse_pos: # Check if we have a valid last position
             mdx = current_mouse_pos.x - app_state.last_mouse_pos.x
             mdy = current_mouse_pos.y - app_state.last_mouse_pos.y # ImGui Y is often inverted screen coords
             app_state.yaw += mdx * app_state.mouse_sensitivity
             app_state.pitch -= mdy * app_state.mouse_sensitivity # Invert pitch control
             app_state.pitch = max(-math.pi/2 + 0.01, min(math.pi/2 - 0.01, app_state.pitch)) # Clamp pitch
             app_state.update_camera_vectors()
         # Update last_mouse_pos for next frame, even if GLFW locks the visible cursor
         app_state.last_mouse_pos = current_mouse_pos

    elif is_rmb_down and shift_down: # Panning logic
         if app_state.last_mouse_pos:
            mdx = current_mouse_pos.x - app_state.last_mouse_pos.x; mdy = current_mouse_pos.y - app_state.last_mouse_pos.y
            pan_speed = 0.01 * app_state.camera_speed # Adjust pan speed maybe
            pan = (-app_state.camera_right * mdx + app_state.camera_up * mdy) * pan_speed
            app_state.camera_position += pan
         app_state.last_mouse_pos = current_mouse_pos # Update for panning delta
    else: # Neither looking nor panning
         app_state.last_mouse_pos = None # Reset last pos when idle

    # Keyboard movement (Keep as is)
    velocity = app_state.camera_speed * io.delta_time; move_direction = pyrr.Vector3([0.,0.,0.]); moved = False # Use floats
    if imgui.is_key_down(imgui.Key.w): move_direction += app_state.camera_front; moved = True
    if imgui.is_key_down(imgui.Key.s): move_direction -= app_state.camera_front; moved = True
    if imgui.is_key_down(imgui.Key.a): move_direction -= app_state.camera_right; moved = True
    if imgui.is_key_down(imgui.Key.d): move_direction += app_state.camera_right; moved = True
    if imgui.is_key_down(imgui.Key.e): move_direction += pyrr.Vector3([0.,1.,0.]); moved = True # World Up
    if imgui.is_key_down(imgui.Key.q): move_direction -= pyrr.Vector3([0.,1.,0.]); moved = True # World Down
    if moved and pyrr.vector.length(move_direction) > 1e-6: app_state.camera_position += pyrr.vector.normalize(move_direction) * velocity


# --- GUI Loop --- ## MODIFIED for Lighting Uniforms ##
def gui_loop():
    global app_state
    io = imgui.get_io(); window_width = int(io.display_size.x); window_height = int(io.display_size.y)

    if window_width <= 0 or window_height <= 0: imgui.begin("Info"); imgui.text("Window minimized."); imgui.end(); return

# Inside the gui_loop function...

    if app_state.shader_program is None and "Failed" not in app_state.loading_status :
        try:  # <<< The 'try' statement
            initialize_opengl_resources() # <<< THIS LINE must be indented under the 'try'
        except Exception as e: # <<< This 'except' correctly follows the try block
            # ... (the rest of your except block, including the nested try/except)
            app_state.loading_status = "OpenGL Init Failed!"
            print(f"Fatal Error during OpenGL Init: {e}\n{traceback.format_exc()}")
            try:
                imgui.begin("Critical Error")
                imgui.text_colored(ImVec4(1,0,0,1), "OpenGL Initialization Failed!")
                imgui.text_wrapped(f"Error: {e}\n{traceback.format_exc()}")
                imgui.end()
            except Exception:
                pass
    handle_camera_input(io, window_width, window_height)

    # --- Rendering ---
    if app_state.shader_program:
        try:
            glViewport(0, 0, window_width, window_height)
            glClearColor(0.15, 0.15, 0.2, 1.0)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            glEnable(GL_DEPTH_TEST)
            # glEnable(GL_CULL_FACE)

            view_matrix = pyrr.matrix44.create_look_at(app_state.camera_position, app_state.camera_position + app_state.camera_front, app_state.camera_up)
            aspect_ratio = float(window_width) / window_height if window_height > 0 else 1.0
            projection_matrix = pyrr.matrix44.create_perspective_projection(app_state.fov_y, aspect_ratio, app_state.near_plane, app_state.far_plane)

            glUseProgram(app_state.shader_program)

            # --- Set Lighting Uniforms (Once per frame, if locations are valid) --- ## NEW ##
            if app_state.loc_light_dir != -1: # Check if uniforms were found
                glUniform3fv(app_state.loc_light_dir, 1, LIGHT_DIRECTION)
                glUniform3fv(app_state.loc_light_color, 1, LIGHT_COLOR)
                glUniform3fv(app_state.loc_view_pos, 1, app_state.camera_position)
                glUniform1f(app_state.loc_ambient, AMBIENT_STRENGTH)
                glUniform1f(app_state.loc_diffuse, DIFFUSE_STRENGTH)
                glUniform1f(app_state.loc_specular, SPECULAR_STRENGTH)
                glUniform1f(app_state.loc_shininess, SHININESS)

            # Set View and Projection uniforms
            glUniformMatrix4fv(app_state.loc_view, 1, GL_FALSE, view_matrix)
            glUniformMatrix4fv(app_state.loc_projection, 1, GL_FALSE, projection_matrix)

            # Draw Axes
            if app_state.vao_axes:
                model_matrix_identity = pyrr.matrix44.create_identity(dtype=np.float32)
                glUniformMatrix4fv(app_state.loc_model, 1, GL_FALSE, model_matrix_identity)
                glBindVertexArray(app_state.vao_axes)
                glDrawArrays(GL_LINES, 0, 6) # Draw 6 vertices for 3 lines

            # Draw Terrain Meshes with Culling
            rendered_count = 0
            if app_state.terrain_meshes:
                 cam_pos = app_state.camera_position
                 render_dist_sq = app_state.render_distance_squared
                 # glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)

                 for sector_name, mesh_data in app_state.terrain_meshes.items():
                    vec_to_center = mesh_data['world_center'] - cam_pos
                    dist_sq = pyrr.vector3.dot(vec_to_center, vec_to_center)
                    if dist_sq <= render_dist_sq:
                        model_matrix = pyrr.matrix44.create_from_translation(mesh_data['sector_offset'])
                        glUniformMatrix4fv(app_state.loc_model, 1, GL_FALSE, model_matrix) # Set model matrix per sector
                        glBindVertexArray(mesh_data['vao'])
                        glDrawElements(GL_TRIANGLES, mesh_data['index_count'], GL_UNSIGNED_INT, None)
                        rendered_count += 1

                 # glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
            app_state.sectors_rendered_last_frame = rendered_count

            glBindVertexArray(0); glUseProgram(0)
            # glDisable(GL_CULL_FACE)

        except Exception as e:
            imgui.begin("Render Error"); imgui.text_colored(ImVec4(1, 0, 0, 1), f"OpenGL Rendering Error:"); imgui.text_wrapped(f"{e}\n{traceback.format_exc()}"); imgui.end()
            print(f"OpenGL Rendering Error: {e}"); traceback.print_exc()


    # --- ImGui Window (Keep As Is) ---
    # (UI code unchanged, including Load/Reload buttons and sliders)
    imgui.begin("Controls & Info"); imgui.text("3D Navigation: Move: WASDQE, Look: RMB+Mouse, Pan: Shift+RMB"); imgui.separator(); imgui.text(f"Window: {window_width}x{window_height}"); imgui.text(f"Cam Pos: ({app_state.camera_position[0]:.1f}, {app_state.camera_position[1]:.1f}, {app_state.camera_position[2]:.1f})"); imgui.text(f"Yaw: {math.degrees(app_state.yaw):.1f} Pitch: {math.degrees(app_state.pitch):.1f}"); changed_speed,app_state.camera_speed=imgui.drag_float("Move Speed",app_state.camera_speed,1.0,1.0,1000.0); imgui.drag_float("Look Sensitivity",app_state.mouse_sensitivity,0.0001,0.0001,0.01,"%.4f"); changed_fov,app_state.fov_y=imgui.slider_float("FOV Y",app_state.fov_y,10.0,120.0); imgui.separator(); imgui.text("Performance:"); changed_dist,new_dist=imgui.slider_float("Render Dist",app_state.render_distance,MIN_RENDER_DISTANCE,MAX_RENDER_DISTANCE,"%.0f")
    if changed_dist: app_state.render_distance=new_dist; app_state.render_distance_squared=new_dist*new_dist; app_state.far_plane=new_dist*1.5
    imgui.separator(); imgui.text("Terrain:"); imgui.text(f"Status: {app_state.loading_status}"); imgui.text(f"Meshes Loaded: {len(app_state.terrain_meshes)}"); imgui.text(f"Meshes Rendered: {app_state.sectors_rendered_last_frame}")
    if imgui.button("Load Terrain Now"):
        if "Failed" in app_state.loading_status and not app_state.shader_program: print("Cannot load terrain: OpenGL Initialization failed.")
        else:
            print("Load button clicked - Cleaning up and loading terrain..."); app_state.cleanup_gl_resources()
            if not app_state.shader_program: # Try to recreate shader if missing
                try:
                    # Use new lighting shaders when recreating
                    app_state.shader_program=create_shader_program(VERTEX_SHADER_SOURCE_LIGHTING,FRAGMENT_SHADER_SOURCE_LIGHTING)
                    print("Shader program recreated for loading.")
                    # IMPORTANT: Re-get uniform locations after recreating shader
                    app_state.loc_model = glGetUniformLocation(app_state.shader_program, "model")
                    app_state.loc_view = glGetUniformLocation(app_state.shader_program, "view")
                    app_state.loc_projection = glGetUniformLocation(app_state.shader_program, "projection")
                    app_state.loc_light_dir = glGetUniformLocation(app_state.shader_program, "u_LightDirection")
                    app_state.loc_light_color = glGetUniformLocation(app_state.shader_program, "u_LightColor")
                    app_state.loc_view_pos = glGetUniformLocation(app_state.shader_program, "u_ViewPos")
                    app_state.loc_ambient = glGetUniformLocation(app_state.shader_program, "u_AmbientStrength")
                    app_state.loc_diffuse = glGetUniformLocation(app_state.shader_program, "u_DiffuseStrength")
                    app_state.loc_specular = glGetUniformLocation(app_state.shader_program, "u_SpecularStrength")
                    app_state.loc_shininess = glGetUniformLocation(app_state.shader_program, "u_Shininess")
                    print("Uniform locations re-acquired.")
                    # Recreate axes VAO to match new shader vertex layout
                    axes_vertices = np.array([0.,0.,0.,0.,1.,0.,1.,0.,0., 1.,0.,0.,0.,1.,0.,1.,0.,0., 0.,0.,0.,0.,1.,0.,0.,1.,0., 0.,1.,0.,0.,1.,0.,0.,1.,0., 0.,0.,0.,0.,1.,0.,0.,0.,1., 0.,0.,1.,0.,1.,0.,0.,0.,1.], dtype=np.float32)
                    app_state.vao_axes = glGenVertexArrays(1); app_state.vbo_axes = glGenBuffers(1)
                    glBindVertexArray(app_state.vao_axes); glBindBuffer(GL_ARRAY_BUFFER, app_state.vbo_axes)
                    glBufferData(GL_ARRAY_BUFFER, axes_vertices.nbytes, axes_vertices, GL_STATIC_DRAW)
                    stride = 9 * axes_vertices.itemsize
                    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0)); glEnableVertexAttribArray(0)
                    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(6 * axes_vertices.itemsize)); glEnableVertexAttribArray(1)
                    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(3 * axes_vertices.itemsize)); glEnableVertexAttribArray(2)
                    glBindBuffer(GL_ARRAY_BUFFER, 0); glBindVertexArray(0); print("Axes VAO recreated.")

                except Exception as shader_err:
                    print(f"Error recreating shader for loading: {shader_err}"); app_state.loading_status="Shader Creation Failed!"
                    app_state.shader_program = None # Ensure it's None if creation failed
            if app_state.shader_program: load_terrain_sectors()
    imgui.same_line()
    if imgui.button("Reload Terrain"):
        print("Reload button clicked - Cleaning up and reloading terrain..."); app_state.cleanup_gl_resources()
        if not app_state.shader_program: # Try to recreate shader if missing
            try:
                app_state.shader_program=create_shader_program(VERTEX_SHADER_SOURCE_LIGHTING,FRAGMENT_SHADER_SOURCE_LIGHTING)
                print("Shader program recreated for reloading.")
                # Re-get uniform locations
                app_state.loc_model = glGetUniformLocation(app_state.shader_program, "model")
                app_state.loc_view = glGetUniformLocation(app_state.shader_program, "view")
                app_state.loc_projection = glGetUniformLocation(app_state.shader_program, "projection")
                app_state.loc_light_dir = glGetUniformLocation(app_state.shader_program, "u_LightDirection")
                app_state.loc_light_color = glGetUniformLocation(app_state.shader_program, "u_LightColor")
                app_state.loc_view_pos = glGetUniformLocation(app_state.shader_program, "u_ViewPos")
                app_state.loc_ambient = glGetUniformLocation(app_state.shader_program, "u_AmbientStrength")
                app_state.loc_diffuse = glGetUniformLocation(app_state.shader_program, "u_DiffuseStrength")
                app_state.loc_specular = glGetUniformLocation(app_state.shader_program, "u_SpecularStrength")
                app_state.loc_shininess = glGetUniformLocation(app_state.shader_program, "u_Shininess")
                print("Uniform locations re-acquired.")
                 # Recreate axes VAO
                axes_vertices = np.array([0.,0.,0.,0.,1.,0.,1.,0.,0., 1.,0.,0.,0.,1.,0.,1.,0.,0., 0.,0.,0.,0.,1.,0.,0.,1.,0., 0.,1.,0.,0.,1.,0.,0.,1.,0., 0.,0.,0.,0.,1.,0.,0.,0.,1., 0.,0.,1.,0.,1.,0.,0.,0.,1.], dtype=np.float32)
                app_state.vao_axes = glGenVertexArrays(1); app_state.vbo_axes = glGenBuffers(1)
                glBindVertexArray(app_state.vao_axes); glBindBuffer(GL_ARRAY_BUFFER, app_state.vbo_axes); glBufferData(GL_ARRAY_BUFFER, axes_vertices.nbytes, axes_vertices, GL_STATIC_DRAW); stride = 9 * axes_vertices.itemsize
                glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0)); glEnableVertexAttribArray(0)
                glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(6 * axes_vertices.itemsize)); glEnableVertexAttribArray(1)
                glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(3 * axes_vertices.itemsize)); glEnableVertexAttribArray(2)
                glBindBuffer(GL_ARRAY_BUFFER, 0); glBindVertexArray(0); print("Axes VAO recreated.")
            except Exception as shader_err:
                print(f"Error recreating shader for reloading: {shader_err}"); app_state.loading_status="Shader Creation Failed!"
                app_state.shader_program = None # Ensure it's None if creation failed
        if app_state.shader_program: load_terrain_sectors()
    imgui.end()

def main():
    global app_state
    runner_params = hello_imgui.RunnerParams()
    runner_params.app_window_params.window_title = "Sacred 2 Terrain Viewer"
    runner_params.app_window_params.window_geometry.size = (1280, 720)
    try:
        print("VSync setting attempt skipped/passed.")
    except AttributeError:
        print("Warning: Failed to set swap_interval (VSync).")

    runner_params.callbacks.show_gui = gui_loop

    # --- Corrected perform_cleanup function definition ---
    def perform_cleanup():
        print("Runner exiting, calling cleanup...")
        try:
            app_state.cleanup_gl_resources()
        except Exception as cleanup_err:
            print(f"Error during exit cleanup: {cleanup_err}")
            # Ensure terrain_meshes is cleared even if cleanup fails partially
            app_state.terrain_meshes = {}
    # --- End Corrected Definition ---

    runner_params.callbacks.before_exit = perform_cleanup # Assign the function ONCE

    # --- CORRECTED try/except/finally block ---
    try:
        print("Attempting to run hello_imgui...") # Optional: Add print for debugging
        hello_imgui.run(runner_params)
    except Exception as e:
        print(f"Fatal Error during GUI execution: {e}\n{traceback.format_exc()}")
        try:
            print("Attempting cleanup after GUI error...") # Optional debug print
            app_state.cleanup_gl_resources()
        except Exception as cleanup_err:
            print(f"Error during crash cleanup: {cleanup_err}")
        sys.exit(1) # Exit after handling the main error
    finally:
        # This finally block will execute whether the try block succeeded or failed
        print("Exiting main function (finally block).")
        # Clearing here might be redundant if cleanup_gl_resources already does it,
        # but doesn't hurt.
        app_state.terrain_meshes = {}
    # --- END CORRECTED block ---

if __name__ == "__main__":
    main()