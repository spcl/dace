from pathlib import Path
import shutil
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
from dace.transformation.passes.detect_and_rename_softhier_tasklets import DetectAndRenameSoftHierTasklets
import ctypes
from dace.soft_hier.utils.generate_sdfg import _my_gen_baseline_matmul_sdfg, _my_gen_BSP_matmul_sdfg, _my_gen_split_K_BSP_matmul_sdfg
from dace.soft_hier.utils.BSP_generator import generate_summa_BSP, generate_split_K_summa_systolic_BSP
from dace.soft_hier.utils.read_from_dump_file import get_address_and_read_from_file
from dace.soft_hier.utils.run_e2e_verification import run_e2e_verification, HardwareConfig, setup_hw_env_dace
from dace.soft_hier.utils.reduce_cond_generator import reduce_cond_generator
from functools import partial

import copy
from typing import Tuple
from dace.libraries.standard.nodes.copy_node import CopyLibraryNode
from dace.libraries.standard.nodes.memset_node import MemsetLibraryNode

from dace.transformation.passes import ConstantPropagation, InlineSDFGs
from dace.transformation.passes.explicit_vectorization_gpu import ExplicitVectorizationPipelineGPU
from dace.transformation.passes.offset_loop_and_maps import OffsetLoopsAndMaps
from softhier_dace_artifacts.offload_to_softhier import offload_to_softhier, _gpu_to_softhier
# import cupy as cp
import numpy as np

from dace.transformation.passes.explicit_vectorization_softhier import ExplicitVectorizationPipelineSoftHier
from dace.transformation.passes.remove_assignment_tasklets import RemoveAssignmentTasklets
from dace.transformation.dataflow.move_alloc_up import move_access_node_up, move_exit_access_node_down

import dace.sdfg.construction_utils as cutil
from dace.sdfg.fp_utils.change_fp_types import change_fptype

# Blocked version for SoftHier (Note: GPU names are used by the names will be moved be SoftHier compatible)
Y = dace.symbol("X")
X = dace.symbol("Y")
CORES_X = dace.symbol("CORES_X")
CORES_Y = dace.symbol("CORES_Y")
BLOCK_X = dace.symbol("BLOCK_X")
BLOCK_Y = dace.symbol("BLOCK_Y")
NUM_BLOCKS_X = dace.symbol("NUM_BLOCKS_X")
NUM_BLOCKS_Y = dace.symbol("NUM_BLOCKS_Y")

# Blocked version for SoftHier (Note: GPU names are used by the names will be moved be SoftHier compatible)
@dace.program
def matrix_addition(A: dace.float64[Y, X] @ dace.dtypes.StorageType.GPU_Global,
                    B: dace.float64[Y, X] @ dace.dtypes.StorageType.GPU_Global,
                    C: dace.float64[Y, X] @ dace.dtypes.StorageType.GPU_Global):
    for i, j in dace.map[0:Y:(BLOCK_Y * CORES_Y * NUM_BLOCKS_Y),
                         0:X:(BLOCK_X * CORES_X * NUM_BLOCKS_X)] @ dace.dtypes.ScheduleType.GPU_Device:
        for c_i, c_j in dace.map[i:i + (BLOCK_Y * CORES_Y * NUM_BLOCKS_Y):(BLOCK_Y * NUM_BLOCKS_Y),
                                 j:j + (BLOCK_X * CORES_X * NUM_BLOCKS_X):(BLOCK_X * NUM_BLOCKS_X)] @ dace.dtypes.ScheduleType.GPU_ThreadBlock:
            for b_i, b_j in dace.map[c_i:c_i + (BLOCK_Y * NUM_BLOCKS_Y):BLOCK_Y,
                                     c_j:c_j + (BLOCK_X * NUM_BLOCKS_X):BLOCK_X] @ dace.dtypes.ScheduleType.Sequential:
                for k_i, k_j in dace.map[b_i:b_i + BLOCK_Y:1,
                                         b_j:b_j + BLOCK_X:1] @ dace.dtypes.ScheduleType.Sequential:
                    C[k_i, k_j] = A[k_i, k_j] + B[k_i, k_j]

def move_up(sdfg: dace.SDFG, prefix: str, offset_memlets: bool):
    for state in sdfg.all_states():
        for node in state.nodes():
            if isinstance(node, dace.nodes.AccessNode) and node.data.startswith(prefix):
                move_access_node_up(state, node, offset_memlets)
                return

def move_down(sdfg: dace.SDFG, prefix: str, offset_memlets: bool):
    for state in sdfg.all_states():
        for node in state.nodes():
            if isinstance(node, dace.nodes.AccessNode) and node.data.startswith(prefix):
                move_exit_access_node_down(state, node, offset_memlets)
                return

def try_dealias_map_connectors(sdfg: dace.SDFG, var_1: bool):
   for state in sdfg.all_states():
        top_level_maps = set()
        sdict = state.scope_dict()
        for node in state.nodes():
            if isinstance(node, dace.nodes.MapEntry) and sdict[node] is None:
                top_level_maps.add(node)

        for map_entry in top_level_maps:
            # Collect names
            in_data_names = {ie.data.data for ie in state.in_edges(map_entry) if ie.data.data is not None}
            out_data_names = {oe.data.data for oe in state.out_edges(state.exit_node(map_entry)) if oe.data.data is not None}
            data_names = in_data_names.union(out_data_names)

            nodes_to_check = {oe.dst for oe in state.out_edges(map_entry)}

            all_nodes_between = state.all_nodes_between(map_entry, state.exit_node(map_entry))
            all_edges = state.all_edges(*all_nodes_between)

            for e in all_edges:
                if e.data.data is not None:
                    original_data_name = e.data.data[6:] if e.data.data.startswith("local_") else e.data.data
                    if original_data_name in data_names:
                        dname = e.data.data if var_1 else original_data_name
                        if e.src_conn is not None and e.src_conn.startswith("IN_"):
                            assert False
                        elif e.src_conn is not None and e.src_conn.startswith("OUT_"):
                            e.src.remove_out_connector(e.src_conn)
                            e.src.add_out_connector(f"OUT_{dname}", force=True)
                            e.src_conn = f"OUT_{dname}"
                        if e.dst_conn is not None and e.dst_conn.startswith("IN_"):
                            e.dst.remove_in_connector(e.dst_conn)
                            e.dst.add_in_connector(f"IN_{dname}", force=True)
                            e.dst_conn = f"IN_{dname}"
                        elif e.dst_conn is not None and e.dst_conn.startswith("OUT_"):
                            assert False



def _get_gvsoc_path() -> str:
    """Get GVSOC path from environment or by locating gvsoc binary"""
    # set gvsoc path
    os.environ["GVSOC_PATH"] = "/scratch/dace4softhier/gvsoc"

    # First try environment variable
    if "GVSOC_PATH" in os.environ:
        return os.environ["GVSOC_PATH"]

    # Try to find gvsoc binary
    gvsoc_binary = shutil.which("gvsoc")
    if gvsoc_binary:
        # Get parent directory three times: bin -> install -> gvsoc_root
        return str(Path(gvsoc_binary).parent.parent.parent)


_get_gvsoc_path()

# Configuration
config = HardwareConfig(
    hardware_thread_group_dims=(4, 4),
    hbm_addr_base=0xc0000000,
    hbm_addr_space=0x01000000,
    tcdm_size=0x00100000,
    cluster_zomem_size=0x00100000,
    redmule_ce_height=64,
    redmule_ce_width=16,
    redmule_ce_pipe=3,
    hbm_placement="4,0,0,4",
    num_node_per_ctrl=1,
    noc_link_width=4096,
    num_hbm_channels=4*4,
    dtype_input=np.uint16,
    dtype_output=np.uint16,
    dace_input_type=dace.uint16,
    dace_output_type=dace.uint16,
)

PATTERN_DICT = {
    "baseline"              : {"sdfg": _my_gen_baseline_matmul_sdfg},
    "summa"                 : {"sdfg": _my_gen_BSP_matmul_sdfg,
                                "BSP": generate_summa_BSP},
    "multistage"           : {"sdfg": _my_gen_BSP_matmul_sdfg,
                                "BSP": generate_summa_BSP},
    "split_k_summa"         : {"sdfg": _my_gen_split_K_BSP_matmul_sdfg,
                                "BSP": generate_split_K_summa_systolic_BSP},
    "remap_split_k_summa"   : {"sdfg": _my_gen_split_K_BSP_matmul_sdfg,
                                "BSP": generate_split_K_summa_systolic_BSP}
}

def create_data_and_handlers(M_val, N_val, K_val, hw_config: HardwareConfig, hwM, hwN, hwK,  thread_group_dims_dace: tuple):
    DTYPE_INPUT = hw_config.dtype_input
    DTYPE_OUTPUT = hw_config.dtype_output
    hardware_thread_group_dims = hw_config.hardware_thread_group_dims
    dim_x_real, dim_y_real = hardware_thread_group_dims
    

    A_host = np.fromfunction(lambda i, j: i + j, (M_val, K_val), dtype=DTYPE_INPUT)
    B_host = np.fromfunction(lambda i, j: 2 * i + 2 * j, (K_val, N_val), dtype=DTYPE_INPUT)
    # A_host = np.ones((M_val, K_val), dtype=DTYPE_INPUT)
    # B_host = np.ones((K_val, N_val), dtype=DTYPE_INPUT)
    C_host = np.zeros((M_val, N_val), dtype=DTYPE_OUTPUT)

    A_handler = InterleaveHandler(array=A_host, block_shape=(hwM, hwK), cluster_dims=hardware_thread_group_dims)
    A_handler.split_horizental()
    A_handler.place_to_range(place_range=(dim_x_real+2*dim_y_real, 2*dim_x_real+2*dim_y_real-1, 1))

    B_handler = InterleaveHandler(array=B_host, block_shape=(hwK, hwN), cluster_dims=hardware_thread_group_dims)
    B_handler.split_vertical()
    B_handler.place_to_range(place_range=(0, dim_y_real - 1, 1))

    C_handler = InterleaveHandler(array=C_host, block_shape=(hwM, hwN), cluster_dims=hardware_thread_group_dims)
    C_handler.split_to_blocks()
    C_handler.summa_place_to_left_and_bottom()

    return {
        "numpy_data": {
            "A": A_host,
            "B": B_host,
            "C": C_host,
        },
        "interleavers": {
            "A": A_handler,
            "B": B_handler,
            "C": C_handler,
        }
    }


def run_sdfg_in_tempdir(combo, interleavers: Dict[str, InterleaveHandler], hw_config: HardwareConfig,
                        host_data: Dict[str, np.ndarray]):
    """
    Each call uses the SLOT environment variable set in init_worker.
    Returns a dict of the relevant parameters plus the measured execution_period_ns.
    """
    # Retrieve the SLOT assigned to this worker process
    slot_id = os.environ.get("SLOT", "UNKNOWN")

    (M_val, N_val, K_val, hwM, hwN, hwK, thread_group_dims_dace) = combo
    (dim_x_dace, dim_y_dace) = thread_group_dims_dace

    hardware_matmul_mnk = (hwM, hwN, hwK)
    combo_summary = (f"SLOT={slot_id}, "
                     f"M={M_val}, N={N_val}, K={K_val}, "
                     f"hwMNK={hardware_matmul_mnk}")
    log_file = open("./log", "a")
    log_file.close()
    # Redirect stdout and stderr to a log file
    slot_dir = f"./slot_{slot_id}"
    # log_file_path = ""
    execution_period_ns = None

    tmp_dir = "."

    interleaver_list = [interleavers["A"], interleavers["B"], interleavers["C"]]
    make_preload_elf_hbm_interleaved_new("output.elf",
                                         interleaver_list,
                                         KMN=[K_val, M_val, N_val],
                                         hbm_node_addr_base=hw_config.hbm_addr_base,
                                         hbm_node_addr_space=hw_config.hbm_addr_space,
                                         args_only=True if hw_config.test_mode == 'perf_only' else False)

    M = M_val
    N = N_val
    K = K_val

    # reduce_condition_func = reduce_cond_generator().reduce_to_first
    X_val = M_val
    Y_val = N_val
    CORES_X_val = dim_x_dace
    CORES_Y_val = dim_y_dace
    BLOCK_X_val = hwM
    BLOCK_Y_val = hwN
    NUM_BLOCKS_X_val = 2
    NUM_BLOCKS_Y_val = 2

    # Run DaCe SDFG
    sdfg = matrix_addition.to_sdfg()
    # Re-enable this with different parameters
    sdfg.replace_dict(
        repldict={
            "Y": Y_val,
            "X": X_val,
            "BLOCK_X": BLOCK_X_val,
            "BLOCK_Y": BLOCK_Y_val,
            "CORES_Y": CORES_Y_val,
            "CORES_X": CORES_X_val,
            "NUM_BLOCKS_Y": NUM_BLOCKS_Y_val,
            "NUM_BLOCKS_X": NUM_BLOCKS_X_val,
        }
    )
    # sdfg.save("s0.sdfg")
    sdfg.validate()

    copy_sdfg = copy.deepcopy(sdfg)
    change_fptype(sdfg=copy_sdfg,
                  src_fptype=dace.float64,
                  dst_fptype=dace.uint16,
                  cast_in_and_out_data=False)
    # copy_sdfg.save("s1.sdfg")

    #_gpu_to_softhier(copy_sdfg)

    copy_sdfg.validate()
    # copy_sdfg.save("s2.sdfg")
    #ExplicitVectorizationPipelineCPU(32).apply_pass(copy_sdfg, {})
    ExplicitVectorizationPipelineSoftHier(32).apply_pass(copy_sdfg, {})

    # You can play around moving up when data is moved from HBM to TCDM
    # First moveup/movedown require offsetting memelts
    move_up(copy_sdfg, "A_vec", True)
    move_up(copy_sdfg, "A_vec", False)
    move_up(copy_sdfg, "B_vec", True)
    move_up(copy_sdfg, "B_vec", False)

    _gpu_to_softhier(copy_sdfg)
    copy_sdfg.validate()
    # copy_sdfg.save("s3.sdfg")

    move_down(copy_sdfg, "C_vec", True)
    move_down(copy_sdfg, "C_vec", False)
    copy_sdfg.validate()
    # copy_sdfg.save("s4.sdfg")


    # Offset TBlock map (Make sure you run this before Remove assignment tasklets)
    # This pass does not handle other subset well
    OffsetLoopsAndMaps(
        offset_expr="-i",
        begin_expr="i",
        convert_leq_to_lt=False,
        normalize_loops=False,
        squeeze=True,
    ).apply_pass(copy_sdfg, {})
    OffsetLoopsAndMaps(
        offset_expr="-j",
        begin_expr="j",
        convert_leq_to_lt=False,
        normalize_loops=False,
        squeeze=True,
    ).apply_pass(copy_sdfg, {})
    # copy_sdfg.save("s5.sdfg")

    RemoveAssignmentTasklets().apply_pass(copy_sdfg, {})
    # copy_sdfg.save("s6.sdfg")

    # Match TCDM names to be local_<HBM_name>
    # Make sure everything is fp16
    cutil.connect_array_names(
        sdfg=copy_sdfg,
        local_storage=dace.dtypes.StorageType.SoftHier_TCDM,
        src_storage=dace.dtypes.StorageType.SoftHier_HBM,
        local_name_prefix="local_"
    )
    change_fptype(sdfg=copy_sdfg,
                  src_fptype=dace.float64,
                  dst_fptype=dace.float16,
                  cast_in_and_out_data=False)
    change_fptype(sdfg=copy_sdfg,
                  src_fptype=dace.float32,
                  dst_fptype=dace.float16,
                  cast_in_and_out_data=False)
    DetectAndRenameSoftHierTasklets().apply_pass(sdfg=copy_sdfg, pipeline_results={})
    # copy_sdfg.save("s7.sdfg")
    copy2_sdfg = copy.deepcopy(copy_sdfg)

    try_dealias_map_connectors(
        sdfg=copy_sdfg,
        var_1=True
    )
    copy_sdfg.save("s8.sdfg")
    copy_sdfg.compile()
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
        return {"sdfg": sdfg}
    else:
        return {
            "M": M_val,
            "N": N_val,
            "K": K_val,
            "hwM": hwM,
            "hwN": hwN,
            "hwK": hwK,
            "execution_period_ns": execution_period_ns,
            "sdfg": sdfg
        }


def gemm(A: np.ndarray, B: np.ndarray, C: np.ndarray):
    """Naive GEMM: C = A @ B"""
    C = A @ B
    return C


if __name__ == "__main__":

    M, N, K, hwM, hwN, hwK, thread_group_dims_dace = 256, 256, 256, 32, 32, 32, (4, 4)
    combo = (M, N, K, hwM, hwN, hwK, thread_group_dims_dace)
    config.dace_thread_group_dims = thread_group_dims_dace
    config.test_mode = 'perf_only'
    config.skip_build_hw = True 
    config.pattern_name = "remap_split_k_summa"
    data_and_interleavers = create_data_and_handlers(M, N, K, config, hwM, hwN, hwK, thread_group_dims_dace)

    data = data_and_interleavers["numpy_data"]
    interleavers = data_and_interleavers["interleavers"]

    run_numpy_fn = partial(gemm, data["A"], data["B"], data["C"])
    run_sdfg_fn = partial(run_sdfg_in_tempdir, combo, interleavers, config, data)

    run_e2e_verification(hw_config=config,
                         data=data,
                         interleave_handlers=interleavers,
                         numpy_fn=run_numpy_fn,
                         sdfg_fn=run_sdfg_fn)
