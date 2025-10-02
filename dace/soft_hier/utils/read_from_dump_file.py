import dace
import struct
from typing import Dict, List
from dace.soft_hier.utils.interleave_handler import InterleaveHandler


def get_address_and_read_from_file(i: int,
                                   j: int,
                                   interleave_handler: InterleaveHandler,
                                   array_name: str,
                                   array: dace.data.Data,
                                   element_size_in_bytes: int,
                                   dtype: str,
                                   parsed_sections: Dict[str, Dict[int, List[str]]],
                                   debug_print: bool = False,
                                   debug_i: int = None,
                                   debug_j: int = None):

    # Only print debug info for specific elements
    if not debug_print:
        if debug_i is None or debug_j is None:
            debug_print = False
        else:
            debug_print = (i == debug_i and j == debug_j)

    if debug_print:
        print(f"\n=== STARTING ADDRESS COMPUTATION FOR ELEMENT ({i}, {j}) ===")

    # Extract configuration from interleave handler
    block_shape = interleave_handler.block_shape
    cluster_dims = interleave_handler.cluster_dims
    cluster_dims_dace = interleave_handler.cluster_dims_dace
    split_scheme = interleave_handler.split_scheme
    placement_scheme = interleave_handler.placement_scheme
    tiling_shape = interleave_handler.tiling_shape
    num_channels = interleave_handler.num_channels

    if debug_print:
        print(f"Configuration:")
        print(f"  - Block shape: {block_shape}")
        print(f"  - Tiling shape: {tiling_shape}")
        print(f"  - Split scheme: {split_scheme}")
        print(f"  - Number of channels: {num_channels}")
        print(f"  - Element size: {element_size_in_bytes} bytes")
        print(f"  - Data type: {dtype}")
        print(f"  - Array name: {array_name}")

    # Assume matrix A ∈ [M, K]
    # Get Tile ID:
    # Tiling shape is the shape of a tile [tileM, tileK]
    tileM, tileK = tiling_shape

    if debug_print:
        print(f"\n--- STEP 1: TILE COMPUTATION ---")
        print(f"Tile dimensions: {tileM} x {tileK}")

    tile_id_i = i // tileM
    tile_id_j = j // tileK
    num_tiles_j = split_scheme[1]
    tile_offset_i = i % tileM
    tile_offset_j = j % tileK
    linearized_tile_id = tile_id_i * num_tiles_j + tile_id_j
    tiles_before_me = placement_scheme[0:linearized_tile_id]

    if debug_print:
        print(f"Element position ({i}, {j}) maps to:")
        print(f"  - Tile ID: ({tile_id_i}, {tile_id_j})")
        print(f"  - Offset within tile: ({tile_offset_i}, {tile_offset_j})")
        print(f"  - Linearited tile id: ({linearized_tile_id})")

    # Get tile id (used to access the channel)
    numTilesM, numTilesK = split_scheme
    linearized_tile_id = tile_id_j + tile_id_i * numTilesK

    if debug_print:
        print(f"\n--- STEP 2: CHANNEL SELECTION ---")
        print(f"Grid has {numTilesM} x {numTilesK} = {numTilesM * numTilesK} total tiles")
        print(f"Linearized tile ID: {tile_id_j} + {tile_id_i} * {numTilesK} = {linearized_tile_id}")

    channel_id = placement_scheme[linearized_tile_id]
    tiles_before_me_on_the_same_channel = len([tid for tid in tiles_before_me if tid == channel_id])

    if debug_print:
        print(f"Placement scheme maps tile {linearized_tile_id} to channel {channel_id}")
        print(f"Tiles of the same array before me on the channel: {tiles_before_me_on_the_same_channel}")
        print(f"Tile placement scheme until this tile: {tiles_before_me}")

    # Get Block ID (for block size [blockM, blockK]):
    blockM, blockK = block_shape

    if debug_print:
        print(f"\n--- STEP 3: BLOCK COMPUTATION ---")
        print(f"Block dimensions: {blockM} x {blockK}")

    block_id_i = tile_offset_i // blockM
    block_id_j = tile_offset_j // blockK

    if debug_print:
        print(f"Tile offset ({tile_offset_i}, {tile_offset_j}) maps to:")
        print(f"  - Block ID within tile: ({block_id_i}, {block_id_j})")

    # Get Block Offset
    block_offset_i = tile_offset_i % blockM
    block_offset_j = tile_offset_j % blockK

    if debug_print:
        print(f"  - Offset within block: ({block_offset_i}, {block_offset_j})")

    # Linearize block id (offset within a tile) (always stored row major)
    # Get the number of blocks in each direction
    numBlocksM, numBlocksK = tileM // blockM, tileK // blockK
    linearized_block_id = block_id_j + block_id_i * numBlocksK

    if debug_print:
        print(f"\nBlock grid within tile: {numBlocksM} x {numBlocksK} blocks")
        print(f"Linearized block ID: {block_id_j} + {block_id_i} * {numBlocksK} = {linearized_block_id}")

    # Linearized element offset in a block
    linearized_block_offset = block_offset_j + block_offset_i * blockK

    if debug_print:
        print(
            f"Linearized element offset within block: {block_offset_j} + {block_offset_i} * {blockK} = {linearized_block_offset}"
        )

    # Address computation:
    # Get sizes
    tile_size_bytes = tileM * tileK * element_size_in_bytes
    block_size_bytes = blockM * blockK * element_size_in_bytes

    if debug_print:
        print(f"\n--- STEP 4: ADDRESS CALCULATION ---")
        print(f"Size calculations:")
        print(f"  - Tile size: {tileM} * {tileK} * {element_size_in_bytes} = {tile_size_bytes} bytes")
        print(f"  - Block size: {blockM} * {blockK} * {element_size_in_bytes} = {block_size_bytes} bytes")

    # Add tile offset, block offset, element offset:
    # Get base address in the channel
    base_address = 0  # Read from file

    if debug_print:
        print(f"Base address: {base_address}")

    #if filename in loaded_files:
    # file
    #tile_address = base_address + tiles_before_me * tile_size_bytes
    #block_address = tile_address + linearized_block_id * block_size_bytes
    block_address = linearized_block_id * block_size_bytes
    element_address = block_address + linearized_block_offset * element_size_in_bytes

    if debug_print:
        print(f"Address calculation:")
        print(f"  - Block address: {linearized_block_id} * {block_size_bytes} = {block_address}")
        print(
            f"  - Element address: {block_address} + {linearized_block_offset} * {element_size_in_bytes} = {element_address}"
        )

    # Alignment check
    assert element_address % 2 == 0, "Implement alignment <2 bytes addressing"

    if debug_print:
        print(f"Address alignment: {element_address} is aligned to 2-byte boundary ✓")

    # Convert to line addressing (16-bit lines)
    line_id = element_address // 2

    if debug_print:
        print(f"Line ID (16-bit addressing): {element_address} // 2 = {line_id}")

    assert element_size_in_bytes % 2 == 0, "Element size needs to be multiple of 16-bits"
    lines_needed = element_size_in_bytes // 2

    if debug_print:
        print(f"Lines needed for {dtype}: {element_size_in_bytes} // 2 = {lines_needed}")
        print(f"\n--- STEP 5: DATA RETRIEVAL ---")
        print(f"Reading from channel {channel_id}, lines {line_id} to {line_id + lines_needed - 1}")

    line_contents = parsed_sections[array_name][channel_id][tiles_before_me_on_the_same_channel][line_id:line_id + lines_needed]

    if debug_print:
        print(f"Raw line contents (hex): {line_contents}")

    raw_bytes = b''.join(int(line, 16).to_bytes(2, "big") for line in line_contents)

    if debug_print:
        print(f"Raw bytes: {raw_bytes.hex()}")

    # Data type conversion
    fmt_map = {
        "int16": ("h", 2),
        "uint16": ("H", 2),
        "int32": ("i", 4),
        "uint32": ("I", 4),
        "float32": ("f", 4),
        "float64": ("d", 8),
    }

    if dtype not in fmt_map:
        raise ValueError(f"Unsupported dtype {dtype}")

    fmt, nbytes = fmt_map[dtype]
    endian_prefix = ">"

    if debug_print:
        print(f"\n--- STEP 6: DATA INTERPRETATION ---")
        print(f"Data type mapping: {dtype} -> format '{fmt}', {nbytes} bytes")
        print(f"Using big-endian format: '{endian_prefix + fmt}'")

    if len(raw_bytes) != nbytes:
        raise ValueError(f"Need {nbytes} bytes for {dtype}, but got {len(raw_bytes)}")

    raw_bytes = raw_bytes[:nbytes]
    result = struct.unpack(endian_prefix + fmt, raw_bytes)[0]

    if debug_print:
        print(f"Final value: {result}")
        print(f"=== ADDRESS COMPUTATION COMPLETE ===\n")

    return result
