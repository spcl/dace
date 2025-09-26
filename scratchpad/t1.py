import dace
import os
import numpy as np
import subprocess
import re
import sys
from typing import Dict, Iterable, List
from dace.soft_hier.utils.interleave_handler import InterleaveHandler
from dace.soft_hier.utils.generate_arch_config import generate_arg_cfg
from dace.soft_hier.utils.preload import make_preload_elf_hbm_interleaved_new
import ctypes
from dace.soft_hier.utils.generate_sdfg import _my_gen_baseline_matmul_sdfg
import struct

# Configuration
GVSOC_PATH = os.environ.get("GVSOC_PATH", "/home/primrose/Work/SoftHier/gvsoc")
DACE_PATH = os.environ.get("DACE_PATH", "/home/primrose/Work/SoftHier/softhierdace")
THREAD_GROUP_DIMS = (2, 2)
HBM_ADDR_BASE = 0xc0000000
HBM_ADDR_SPACE = 0x04000000
HBM_ADDR_BASE_STR = "0xc0000000"
HBM_ADDR_SPACE_STR = "0x04000000"
TCDM_SIZE = 0x00100000
DTYPE_INPUT = np.uint16
DACE_INPUT_TYPE = dace.uint16
DTYPE_OUTPUT = np.uint16
DACE_OUTPUT_TYPE = dace.uint16
ARRAY_TO_DTYPE = dict()

def setup_environment():
    """Setup paths and environment"""
    os.environ["GVSOC_INSTALL_PATH"] = GVSOC_PATH
    os.environ["GVSOC_DIR"] = GVSOC_PATH
    os.environ["GVSOC_PATH"] = GVSOC_PATH
    os.environ["PATH"] = f"{GVSOC_PATH}/third_party/toolchain/install/bin:{os.environ['PATH']}"
    os.environ["SHCC"] = f"{GVSOC_PATH}/third_party/toolchain/install/bin/riscv32-unknown-elf-gcc"
    os.environ["CPLUS_INCLUDE_PATH"] = f"{DACE_PATH}/dace/runtime/include/dace/soft_hier/runtime/include:{os.environ.get('CPLUS_INCLUDE_PATH', '')}"
    os.environ["C_INCLUDE_PATH"] = f"{DACE_PATH}/dace/runtime/include/dace/soft_hier/runtime/include:{os.environ.get('C_INCLUDE_PATH','')}"
    os.environ["SOFTHIER_INSTALL_PATH"] = f"{GVSOC_PATH}/soft_hier/flex_cluster_sdk/runtime/"

def setup_architecture():
    """Generate and compile architecture"""
    generate_arg_cfg(
        cluster_tcdm_size=hex(TCDM_SIZE),
        num_cluster_x=THREAD_GROUP_DIMS[0], num_cluster_y=THREAD_GROUP_DIMS[1],
        redmule_ce_height=64, redmule_ce_width=64, redmule_ce_pipe=1,
        hbm_start_base=hex(HBM_ADDR_BASE),
        hbm_node_addr_space=hex(HBM_ADDR_SPACE),
        hbm_placement="2,2,2,2", num_node_per_ctrl=1, noc_link_width=4096
    )
    
    subprocess.run([
        "bash", "-c", 
        f"cd {GVSOC_PATH} && source sourceme.sh && export cfg={os.path.dirname(__file__)}/generated_arch.py && make hw"
    ], check=True)

def get_address_and_read_from_file(i: int,
                                  j: int,
                                  interleave_handler: InterleaveHandler,
                                  array_name: str,
                                  element_size_in_bytes: int,
                                  dtype: str,
                                  parsed_sections: Dict[str, Dict[int, List[str]]]):
    
    # Only print debug info for specific elements
    debug_print = (i == 1 and j == 0)
    
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
    tile_offset_i = i % tileM
    tile_offset_j = j % tileK  # Note: This should probably be j % tileK, not i % tileK
    
    if debug_print:
        print(f"Element position ({i}, {j}) maps to:")
        print(f"  - Tile ID: ({tile_id_i}, {tile_id_j})")
        print(f"  - Offset within tile: ({tile_offset_i}, {tile_offset_j})")
    
    # Get tile id (used to access the channel)
    numTilesM, numTilesK = split_scheme
    linearized_tile_id = tile_id_j + tile_id_i * numTilesK
    
    if debug_print:
        print(f"\n--- STEP 2: CHANNEL SELECTION ---")
        print(f"Grid has {numTilesM} x {numTilesK} = {numTilesM * numTilesK} total tiles")
        print(f"Linearized tile ID: {tile_id_j} + {tile_id_i} * {numTilesK} = {linearized_tile_id}")
    
    channel_id = placement_scheme[linearized_tile_id]
    
    if debug_print:
        print(f"Placement scheme maps tile {linearized_tile_id} to channel {channel_id}")
    
    # Get number of tiles before our tile
    # Tiles are placed round-robin to channels, this means we can compute
    tiles_before_me = linearized_tile_id // num_channels
    
    if debug_print:
        print(f"Tiles placed before this tile on channel {channel_id}: {linearized_tile_id} // {num_channels} = {tiles_before_me}")
    
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
        print(f"Linearized element offset within block: {block_offset_j} + {block_offset_i} * {blockK} = {linearized_block_offset}")
    
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
        print(f"  - Element address: {block_address} + {linearized_block_offset} * {element_size_in_bytes} = {element_address}")
    
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
    
    line_contents = parsed_sections[array_name][channel_id][line_id:line_id+lines_needed]
    
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


def hbm_to_np_impl(array_name: str,
                   interleave_handler: InterleaveHandler,
                   element_size_in_bytes: int,
                   dtype: str,
                   parsed_sections: Dict[str, Dict[int, List[str]]],
                   numpy_buffer: np.ndarray):
    assert len(numpy_buffer.shape) == 2
    for i in range(numpy_buffer.shape[0]):
        for j in range(numpy_buffer.shape[1]):
            value = get_address_and_read_from_file(
                i=i, j=j, interleave_handler=interleave_handler,
                array_name=array_name, element_size_in_bytes=element_size_in_bytes,
                dtype=dtype, parsed_sections=parsed_sections)
            numpy_buffer[i, j] = value

def create_test_data(M, N, K, hwM, hwN, hwK):
    """Create test matrices and handlers"""
    A = np.fromfunction(lambda i, j: i + j, (M, K), dtype=DTYPE_INPUT)
    
    # Setup handlers
    A_handler = InterleaveHandler(name="A", array=A, cluster_dims=THREAD_GROUP_DIMS, block_shape=(16,16))
    A_handler.split_vertical()
    A_handler.place_to_range((0, 7, 1))


    return A, A_handler

def np_to_hbm(A_handler, M, N, K):
    """Step 1: Move NumPy to HBM"""
    make_preload_elf_hbm_interleaved_new(
        "output.elf", [A_handler], 
        KMN=[K, M, N],
        hbm_node_addr_base=HBM_ADDR_BASE, hbm_node_addr_space=HBM_ADDR_SPACE,
        args_only=False
    )
    A_handler.print_info()

def step2_run_softhier(A, M, N, K, hwM, hwN, hwK, A_handler):
    pass



def step4_run_numpy(A, B):
    pass

def step5_compare(C_expected, C_softhier, tolerance=1e-3):
    pass

def run_test(M, N, K, hwM, hwN, hwK):
    """Run complete test pipeline"""
    print(f"Testing GEMM({M}, {N}, {K}) with HW({hwM}, {hwN}, {hwK})")
    
    # Create data
    A, A_handler = create_test_data(M, N, K, hwM, hwN, hwK)
    
    # Run pipeline
    #step1_np_to_hbm(A_handler, M, N, K)
    timing = step2_run_softhier(A, M, N, K, hwM, hwN, hwK, A_handler)
    #step3_hbm_to_np(A)
    #C_expected = step4_run_numpy(A,)
    #comparison = step5_compare(C_expected, C)
    
    #print(f"Result: {comparison['matches']}, Max diff: {comparison['max_diff']:.6f}")
    #if timing:
    #    print(f"Timing: {timing} ns")
    
    #return comparison["matches"]
    return False

def main():
    setup_environment()
    setup_architecture()
    
    # Run test
    success = run_test(512, 512, 512, THREAD_GROUP_DIMS[0], THREAD_GROUP_DIMS[1], 128)
    print(f"Test {'PASSED' if success else 'FAILED'}")


def run_sdfg_in_tempdir(combo, extra_arr, extra_interleaver):
    dace.config.Config.set("backend", "softhier", "HBM_ADDRESS_SPACE", value=HBM_ADDR_SPACE_STR)
    dace.config.Config.set("backend", "softhier", "HBM_ADDRESS_BASE", value=HBM_ADDR_BASE_STR)
    dace.config.Config.set("backend", "softhier", "HBM_NUM_CHANNELS", value=8)

    """
    Each call uses the SLOT environment variable set in init_worker.
    Returns a dict of the relevant parameters plus the measured execution_period_ns.
    """
    # Retrieve the SLOT assigned to this worker process
    slot_id = os.environ.get("SLOT", "UNKNOWN")

    (
        M_val,
        N_val,
        K_val,
        hwM,
        hwN,
        hwK,
        thread_group_dims,
        tcdm_size
    ) = combo

    (dim_x, dim_y) = thread_group_dims

    hardware_matmul_mnk = (hwM, hwN, hwK)
    combo_summary = (
        f"SLOT={slot_id}, "
        f"M={M_val}, N={N_val}, K={K_val}, "
        f"hwMNK={hardware_matmul_mnk}"
    )
    log_file = open("./log", "a")
    log_file.close()
    # Redirect stdout and stderr to a log file
    slot_dir = f"./slot_{slot_id}"
    # log_file_path = ""
    execution_period_ns = None

    tmp_dir = "."
    A_host = np.fromfunction(lambda i, j: i + j, (M_val, K_val), dtype=DTYPE_INPUT)
    B_host = np.fromfunction(lambda i, j: 2*i + 2*j, (K_val, N_val), dtype=DTYPE_INPUT)
    C_host = np.ones((M_val, N_val), dtype=DTYPE_OUTPUT)

    A_handler = InterleaveHandler(array=A_host, block_shape=(hwM, hwK), cluster_dims=thread_group_dims)
    A_handler.split_horizental()
    A_handler.place_to_range(place_range=(0, 7, 1))
    split_A = A_handler.split_scheme
    place_A = A_handler.placement_scheme

    B_handler = InterleaveHandler(array=B_host, block_shape=(hwK, hwN), cluster_dims=thread_group_dims)
    B_handler.split_vertical()
    B_handler.place_to_range(place_range=(0, 7, 1))
    split_B = B_handler.split_scheme
    place_B = B_handler.placement_scheme

    C_handler = InterleaveHandler(array=C_host, block_shape=(hwM, hwN), cluster_dims=thread_group_dims)
    C_handler.split_to_blocks()
    C_handler.place_to_range(place_range=(0, 7, 1))
    split_C = C_handler.split_scheme
    place_C = C_handler.placement_scheme



    make_preload_elf_hbm_interleaved_new(
        "output.elf",
        [A_handler, B_handler, C_handler],
        KMN=[K_val, M_val, N_val],
        hbm_node_addr_base=HBM_ADDR_BASE,
        hbm_node_addr_space=HBM_ADDR_SPACE,
        args_only=False
    )

    M = M_val
    N = N_val
    K = K_val

    sdfg = _my_gen_baseline_matmul_sdfg(
        hardware_matmul_mnk=hardware_matmul_mnk,
        global_storage=dace.dtypes.StorageType.SoftHier_HBM,
        local_storage=dace.dtypes.StorageType.SoftHier_TCDM,
        device_schedule=dace.dtypes.ScheduleType.SoftHier_Device,
        thread_group_schedule=dace.dtypes.ScheduleType.SoftHier_Cluster,
        thread_group_dims=thread_group_dims,
        hbm_split_scheme=[split_A, split_B, split_C],
        hbm_placement_scheme=[place_A, place_B, place_C],
        is_hbm_interleaved=True,
        input_float=DACE_INPUT_TYPE,
        output_float=DACE_OUTPUT_TYPE,
        coarsening_factor=1,
        mmad_tasklet_str="flex_redmule_trigger(_in_local_a, _in_local_b, _in_accumulator, REDMULE_NONE_16);",
        GEMM_shape=(M_val, N_val, K_val),
    )

    sdfg.validate()
    #_, dace_A = sdfg.add_array(
    #    name="extraArray",
    #    shape=[512,512],
    #    dtype=dace.uint16,
    #    storage=dace.dtypes.StorageType.SoftHier_HBM,
    #    transient=False,
    #)
    #dace_A.is_hbm_interleaved = True
    #dace_A.split_scheme = extra_interleaver.split_scheme
    #dace_A.placement_scheme = extra_interleaver.placement_scheme
    sdfg.save("matmul_base.sdfgz")

    compiled_sdfg = sdfg.compile()
    compiled_sdfg(A=A_host, B=B_host, C=C_host,
                    M=M_val, N=N_val, K=K_val)
    # flush the stdout/stderr
    sys.stdout.flush()
    sys.stderr.flush()
    # flush the log file
    #os.system(f"cp -rf {tmp_dir} {python_script_path}")

    # Parse the log file for the performance counter
    with open("./log", "r") as log_file:
        for line in log_file:
            match = re.search(r"\[Performance Counter\]: Execution period is (\d+) ns", line)
            if match:
                execution_period_ns = int(match.group(1))
                break
            else:
                execution_period_ns = None



    log_file = open("./log", "a")
    log_file.close()

    # Return all relevant info in a dictionary
    if not execution_period_ns:
        return {
            "sdfg": sdfg
        }
    else:
        return {
            "thread_group_dims": thread_group_dims,
            "M": M_val,
            "N": N_val,
            "K": K_val,
            "hwM": hwM,
            "hwN": hwN,
            "hwK": hwK,
            "execution_period_ns": execution_period_ns,
            "sdfg": sdfg
        }


def parse_dump_file(
    file_path: str,
    num_channels: int,
    arrays: Iterable[str]
) -> Dict[str, Dict[int, List[str]]]:
    parsed_dump_sections: Dict[str, Dict[int, List[str]]] = dict()
    sections: Dict[str, List[str]] = dict()

    array_list = list(arrays)
    array_list.sort()

    section_id = -1
    # Split file into sections. To dump occurs as:
    # Per Array (lexicographically sorted), each channel's content is dumped
    with open(file_path, "r") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue  # skip empty lines

            if line.startswith("HBM offset =="):
                # Start a new section
                section_id += 1
                current_offset = line.split("==")[-1].strip()
                sections[section_id] = []
            elif line.startswith("0x") and current_offset is not None:
                sections[section_id].append(line)
            else:
                # Unrecognized line; could log or ignore
                pass
    
    # Re-arrange the 1D integer based indexing to a dictionary
    for i in range(len(array_list)):
        for j in range(num_channels):
            offset = i * num_channels + j
            section = sections[offset]
            array_name = array_list[i]
            if array_name not in parsed_dump_sections:
                parsed_dump_sections[array_name] = dict()
            assert j not in parsed_dump_sections[array_name]
            if j not in parsed_dump_sections[array_name]:
                parsed_dump_sections[array_name][j] = section

    return parsed_dump_sections



if __name__ == "__main__":
    setup_environment()
    setup_architecture()
    M, N, K, hwM, hwN, hwK = 512, 512, 512, THREAD_GROUP_DIMS[0], THREAD_GROUP_DIMS[1], 128
    A, A_handler = create_test_data(M, N, K, hwM, hwN, hwK)
    assert isinstance(A, np.ndarray)
    # np_to_hbm(A_handler, M, N, K)

    preA = A.copy()

    combo = (512,512,512,64,64,128, (2,2), (4096))
    d = run_sdfg_in_tempdir(combo, A, A_handler)
    sdfg = d["sdfg"]

    postA = A.copy()

    parsed_sections: Dict[str, Dict[int, List[str]]] = parse_dump_file(file_path=GVSOC_PATH + "/dump_0",
                    num_channels=8,
                    arrays={arr_name for arr_name, arr in sdfg.arrays.items() if arr.transient is False and isinstance(arr, dace.data.Array)})

    #print(parsed_sections)

    hbm_to_np_impl(
        array_name="A",
        interleave_handler=A_handler,
        element_size_in_bytes=2,
        dtype="uint16",
        parsed_sections=parsed_sections,
        numpy_buffer=postA
    )

    print(preA)

    print(postA)

    print(postA - preA)