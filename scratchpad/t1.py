import dace
import os
import numpy as np
import subprocess
import re
import sys
from typing import Dict, List
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
                                   file_base_path: str,
                                   loaded_files: Dict[str, str]):
    block_shape = interleave_handler.block_shape
    cluster_dims = interleave_handler.cluster_dims
    cluster_dims_dace = interleave_handler.cluster_dims_dace
    split_scheme = interleave_handler.split_scheme
    placement_scheme = interleave_handler.placement_scheme
    tiling_shape = interleave_handler.tiling_shape
    num_channels = interleave_handler.num_channels

    # Assume matrix A \in [M, K]
    # Get Tile ID:
    # Tiling shape is the shape of a tile [tileM, tileK]
    tileM, tileK = tiling_shape
    tile_id_i = i // tileM
    tile_id_j  = j // tileK
    tile_offset_i = i % tileM
    tile_offset_j = i % tileK

    # Get tile id (used to access the channel)
    numTilesM, numTilesK = split_scheme
    linearized_tile_id = tile_id_j + tile_id_i * numTilesK
    channel_id = placement_scheme[linearized_tile_id]
    # Get number of tiles before our tile
    # Tiles are placed round-robin to tiles, this means we can compute
    tiles_before_me = linearized_tile_id // num_channels

    # Get Block ID (for block size [blockM, blockK]):
    blockM, blockK = block_shape
    block_id_i = tile_offset_i // blockM
    block_id_j = tile_offset_j // blockK

    # Get Block Offset
    block_offset_i = tile_offset_i % blockM
    block_offset_j = tile_offset_j % blockK

    # Linearize block id (offset within a tile) (always stored row major)
    # Get the number of blocks in each direction
    numBlocksM, numBlocksK = tileM // blockM, tileK // blockK
    linearized_block_id = block_offset_j + block_offset_i * numBlocksK

    # Linearized element offset in a block
    linearized_block_offset = block_offset_j + block_offset_i * blockK

    # Address computation:
    # Get sizes
    tile_size_bytes = tileM * tileK * element_size_in_bytes
    block_size_bytes = blockM * blockK * element_size_in_bytes

    # Add tile offset, block offset, element offset:
    # Get base address in the channel
    base_address = 0 # Read from file
    filename = f"{file_base_path}/dump_{array_name}_ch{channel_id}"
    if filename not in loaded_files:
        with open(filename, "r") as f:
            lines = f.readlines()
            cleaned_lines = []
            for line in lines:
                line = line.strip()
                if line.startswith("0x"):
                    cleaned_lines.append(line)
            loaded_files[filename] = cleaned_lines

    #lines = 
    #if filename in loaded_files:
    #    file 
    tile_address = base_address + tiles_before_me * tile_size_bytes
    block_address = tile_address + linearized_block_id * block_size_bytes
    element_address = block_address + linearized_block_offset * element_size_in_bytes
    assert element_address % 2 == 0, "Implement alignment <2 bytes addressing"
    line_id = element_address // 2
    assert element_size_in_bytes % 2 == 0, "Element size needs to be multiple of 16-bits"
    lines_needed = element_size_in_bytes // 2

    line_contents = loaded_files[filename][line_id:line_id+lines_needed]
    raw_bytes = b''.join(int(line, 16).to_bytes(2, "big") for line in line_contents)

    fmt_map = {
        "int16":   ("h", 2),
        "uint16":  ("H", 2),
        "int32":   ("i", 4),
        "uint32":  ("I", 4),
        "float32": ("f", 4),
        "float64": ("d", 8),
    }
    if dtype not in fmt_map:
        raise ValueError(f"Unsupported dtype {dtype}")

    fmt, nbytes = fmt_map[dtype]
    endian_prefix = ">"

    if len(raw_bytes) != nbytes:
        raise ValueError(f"Need {nbytes} bytes for {dtype}, but got {len(raw_bytes)}")
    raw_bytes = raw_bytes[:nbytes]

    return struct.unpack(endian_prefix + fmt, raw_bytes)[0]


def hbm_to_np_impl(array_name: str,
                   interleave_handler: InterleaveHandler,
                   element_size_in_bytes: int,
                   dtype: str,
                   file_directory: str,
                   numpy_buffer: np.ndarray):
    loaded_files : Dict[str, str] = dict()
    assert len(numpy_buffer.shape) == 2
    for i in range(numpy_buffer.shape[0]):
        for j in range(numpy_buffer.shape[1]):
            value = get_address_and_read_from_file(
                i=i, j=j, interleave_handler=interleave_handler,
                array_name=array_name, element_size_in_bytes=element_size_in_bytes,
                dtype=dtype, file_base_path=file_directory, loaded_files=loaded_files)
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
    dace.config.Config.set("backend", "softhier", "HBM_ADDRESS_BASE", value=HBM_ADDR_SPACE_STR)
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
    _, dace_A = sdfg.add_array(
        name="extraArray",
        shape=[512,512],
        dtype=dace.uint16,
        storage=dace.dtypes.StorageType.SoftHier_HBM,
        transient=False,
    )
    dace_A.is_hbm_interleaved = True
    dace_A.split_scheme = extra_interleaver.split_scheme
    dace_A.placement_scheme = extra_interleaver.placement_scheme
    sdfg.save("matmul_base.sdfgz")

    compiled_sdfg = sdfg.compile()
    compiled_sdfg(A=A_host, B=B_host, C=C_host,
                    M=M_val, N=N_val, K=K_val, extraArray=extra_arr)
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
    hbm_to_np_impl(
        array_name="A",
        interleave_handler=A_handler,
        element_size_in_bytes=2,
        dtype="uint16",
        file_directory=GVSOC_PATH,
        numpy_buffer=postA
    )