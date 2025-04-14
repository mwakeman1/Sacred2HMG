Creates a heightmap from Sacred 2 .sector files. 

Usage: 

Extract the .sector files from the sector zips.
Set the directory to your .pak folder and click Generate Heightmap.

Options are availble for blending between the sectors, as well as what should be generated.

Bugs: 

No smoothing between the sectors, so it appears blocky. This seems to be an issue with the way the sectors are being normalzied, but I honestly have no idea how to fix it.

![{164BCDF7-ECCE-4279-8671-A744EAD3AA2C}](https://github.com/user-attachments/assets/17c76a41-d128-4d9f-8323-3d4478857415)



File Structure & Identification:

Files are named like XXX_YYY_*.sector, where XXX and YYY represent the sector's coordinates in a larger grid. I believe the _XX section is levels, where B0 is ground, D1 is dungeon 1, and D2 is dungeon 2. 

This script searches for a specific 4-byte marker (\x00\x00\x01\x00) to locate the terrain data.

Header Information (Immediately After Marker):

Following the terrain marker, there are 12 bytes that seem to constitute a header or intermediate data section before the main terrain entries. The next 8 bytes contain some header info.

Base Height (base_h): At byte offset 2 within this 8-byte section (so, 4+2 = 6 bytes after the terrain marker), a signed 16-bit integer (short, little-endian) is stored. This likely represents a base elevation level for the entire sector. I only presume this due to the contrasting gamma between sections in the heightmap itself!

Data Entry Structure (Per Vertex):

The height data is stored as a sequence of fixed-size entries; each entry is 8 bytes long. Height Offset (h_val) is located at byte offset 6 within each 8-byte entry.

Other Bytes:

Bytes 2, 3, 4, 5: Expected to be 0x00 (used for validation).

Bytes 0, 1, 7: These bytes are read as part of the 8-byte entry but are not used by this script for height information. They might contain other data like material IDs, texture coordinates, vertex normals, or flags.

Data Organization (Tiling):

The sequence of 8-byte entries is not stored in a simple row-by-row (row-major) order for the whole sector (e.g., 224x224). The sector is divided into smaller width x height (7x7) tiles. The data entries are stored tile by tile. All 49 entries for the top-left tile (0,0) are stored first, then all 49 entries for the tile (0,1) to its right, and so on, across the first row of tiles. Then it proceeds to the second row of tiles, etc. Inside the block of 49 entries corresponding to a single tile, the data seems to be stored in standard row-major order for that tile.
