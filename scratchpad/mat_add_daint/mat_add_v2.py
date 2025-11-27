from pathlib import Path
import csv
import shutil
import dace
import os
import multiprocessing
import tempfile
import time
import numpy as np
import subprocess
import re
import sys
import argparse
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
from dace.transformation.passes.offset_loop_and_maps import OffsetLoopsAndMaps
from softhier_dace_artifacts.offload_to_softhier import offload_to_softhier, _gpu_to_softhier
# import cupy as cp
import numpy as np

from dace.transformation.passes.vectorization.vectorize_softhier import VectorizeSoftHier
#from dace.transformation.passes.remove_assignment_tasklets import RemoveAssignmentTasklets
from dace.transformation.dataflow.move_alloc_up import move_access_node_up, move_exit_access_node_down, offset_tblock_param

import dace.sdfg.construction_utils as cutil
from dace.sdfg.fp_utils.change_fp_types import change_fptype


########################################################################
# Initializer: Called ONCE in each worker process
########################################################################
def init_worker(slots, lock):
    """
    Assign a unique slot number to each worker process and set it
    in the environment variable SLOT.
    """
    with lock:
        slot_id = slots['counter']
        os.environ["SLOT"] = str(slot_id)
        slots['counter'] += 1

    # Create a directory specific to this slot
    os.mkdir(f"{temp_run_dir}/slot_{slot_id}")
    slot_dir = f"{temp_run_dir}/slot_{slot_id}"

    # redirect ccache path
    os.environ["CCACHE_DIR"] = ccache_path


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
        for c_i, c_j in dace.map[i:i + (BLOCK_Y * CORES_Y * NUM_BLOCKS_Y):(BLOCK_Y * NUM_BLOCKS_Y), j:j +
                                 (BLOCK_X * CORES_X * NUM_BLOCKS_X):
                                 (BLOCK_X * NUM_BLOCKS_X)] @ dace.dtypes.ScheduleType.GPU_ThreadBlock:
            for b_i, b_j in dace.map[c_i:c_i + (BLOCK_Y * NUM_BLOCKS_Y):BLOCK_Y, c_j:c_j +
                                     (BLOCK_X * NUM_BLOCKS_X):BLOCK_X] @ dace.dtypes.ScheduleType.Sequential:
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
            out_data_names = {
                oe.data.data
                for oe in state.out_edges(state.exit_node(map_entry)) if oe.data.data is not None
            }
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
    os.environ["GVSOC_PATH"] = "/home/primrose/Work/SoftHier/gvsoc"

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
    hardware_thread_group_dims=(2, 2),
    hbm_addr_base=0xc0000000,
    hbm_addr_space=0x04000000,
    tcdm_size=0x00100000,
    redmule_ce_height=64,
    redmule_ce_width=64,
    redmule_ce_pipe=1,
    hbm_placement="2,2,2,2",
    num_node_per_ctrl=1,
    noc_link_width=4096,
    num_hbm_channels=8,
    dtype_input=np.uint16,
    dtype_output=np.uint16,
    dace_input_type=dace.uint16,
    dace_output_type=dace.uint16,
)

def rm_assignment_tasklets(sdfg: dace.SDFG):
    for state in sdfg.all_states():
        for node in state.nodes():
            if isinstance(node, dace.nodes.Tasklet) and node.label.startswith("assign"):
                ies = state.in_edges(node)
                oes = state.out_edges(node)
                assert len(ies) == 1
                assert len(oes) == 1
                ie = ies[0]
                oe = oes[0]
                src_data = ie.data.data
                src_subset = ie.data.subset
                dst_data = oe.data.data
                dst_subset = oe.data.subset

                new_data = dst_data
                new_other_subset = src_subset
                new_subset = dst_subset
                state.remove_node(node)
                state.add_edge(
                    ie.src, ie.src_conn, oe.dst, oe.dst_conn,
                    dace.memlet.Memlet(data=new_data, other_subset=new_other_subset, subset=new_subset)
                )



def create_data_and_handlers(M_val, N_val, K_val, hw_config: HardwareConfig, hwM, hwN, hwK,
                             thread_group_dims_dace: tuple):
    DTYPE_INPUT = hw_config.dtype_input
    DTYPE_OUTPUT = hw_config.dtype_output
    hardware_thread_group_dims = hw_config.hardware_thread_group_dims
    dim_x_real, dim_y_real = hardware_thread_group_dims

    A_host = np.fromfunction(lambda i, j: i + j, (M_val, K_val), dtype=DTYPE_INPUT)
    B_host = np.fromfunction(lambda i, j: 2 * i + 2 * j, (K_val, N_val), dtype=DTYPE_INPUT)
    # A_host = np.zeros((M_val, K_val), dtype=DTYPE_INPUT)
    # B_host = np.zeros((K_val, N_val), dtype=DTYPE_INPUT)
    C_host = np.zeros((M_val, N_val), dtype=DTYPE_OUTPUT)

    A_handler = InterleaveHandler(array=A_host, block_shape=(hwM, hwK), cluster_dims=hardware_thread_group_dims)
    A_handler.split_horizental()
    A_handler.place_to_range(place_range=(0, dim_y_real - 1, 1))

    B_handler = InterleaveHandler(array=B_host, block_shape=(hwK, hwN), cluster_dims=hardware_thread_group_dims)
    B_handler.split_vertical()
    B_handler.place_to_range(place_range=(0, dim_y_real - 1, 1))

    C_handler = InterleaveHandler(array=C_host, block_shape=(hwM, hwN), cluster_dims=hardware_thread_group_dims)
    C_handler.split_to_blocks()
    C_handler.place_to_range(place_range=(0, dim_y_real - 1, 1))
    # C_handler.summa_place_to_left_and_bottom()

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

    # Redirect stdout and stderr to a log file
    slot_dir = f"./slot_{slot_id}"
    # log_file_path = ""
    execution_period_ns = None

    # tmp_dir = "."
    with tempfile.TemporaryDirectory() as tmp_dir:
        os.chdir(tmp_dir)
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
        NUM_BLOCKS_X_val = 1
        NUM_BLOCKS_Y_val = 1

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
            })
        # sdfg.save("s0.sdfg")
        sdfg.validate()

        copy_sdfg = copy.deepcopy(sdfg)
        change_fptype(sdfg=copy_sdfg, src_fptype=dace.float64, dst_fptype=dace.uint16, cast_in_and_out_data=False)
        # copy_sdfg.save("s1.sdfg")

        #_gpu_to_softhier(copy_sdfg)

        copy_sdfg.validate()
        # copy_sdfg.save("s2.sdfg")
        #ExplicitVectorizationPipelineCPU(32).apply_pass(copy_sdfg, {})

        VectorizeSoftHier(vector_width=32, insert_copies=True, eliminate_trivial_vector_map=True, dtype=dace.uint16).apply_pass(copy_sdfg, {})

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
        offset_tblock_param(copy_sdfg, {"c_i", "c_j"})
        OffsetLoopsAndMaps(
            offset_expr="-i",
            begin_expr="i",
            convert_leq_to_lt=False,
            normalize_loops=False,
            squeeze_maps=True,
        ).apply_pass(copy_sdfg, {})
        OffsetLoopsAndMaps(
            offset_expr="-j",
            begin_expr="j",
            convert_leq_to_lt=False,
            normalize_loops=False,
            squeeze_maps=True,
        ).apply_pass(copy_sdfg, {})
        # copy_sdfg.save("s5.sdfg")
        copy_sdfg.save("/home/primrose/Work/dace/scratchpad/mat_add_daint/x.sdfg")

        #RemoveAssignmentTasklets().apply_pass(copy_sdfg, {})
        # copy_sdfg.save("s6.sdfg")

        # Match TCDM names to be local_<HBM_name>
        # Make sure everything is fp16
        #cutil.connect_array_names(sdfg=copy_sdfg,
        #                          local_storage=dace.dtypes.StorageType.SoftHier_TCDM,
        #                          src_storage=dace.dtypes.StorageType.SoftHier_HBM,
        #                          local_name_prefix="local_")
        change_fptype(sdfg=copy_sdfg, src_fptype=dace.float64, dst_fptype=dace.uint16, cast_in_and_out_data=False)
        change_fptype(sdfg=copy_sdfg, src_fptype=dace.uint16, dst_fptype=dace.uint16, cast_in_and_out_data=False)
        #DetectAndRenameSoftHierTasklets().apply_pass(sdfg=copy_sdfg, pipeline_results={})
        # copy_sdfg.save("s7.sdfg")

        try_dealias_map_connectors(sdfg=copy_sdfg, var_1=True)
        rm_assignment_tasklets(copy_sdfg)


        # copy_sdfg = dace.SDFG.from_file("/scratch/dace4softhier/scratchpad/mat_add/s8.sdfg")
        for array_name in copy_sdfg.arrays:
            data_desc = copy_sdfg.arrays[array_name]
            if data_desc.storage == dace.dtypes.StorageType.SoftHier_HBM:
                print(f"Array {array_name} is in HBM, changing to SoftHier_HBM")
                data_desc.is_hbm_interleaved = True
                data_desc.hbm_placement_scheme = interleavers[array_name].placement_scheme
                data_desc.hbm_split_scheme = interleavers[array_name].split_scheme
                data_desc.dtype = dace.uint16
        copy_sdfg.save("s8.sdfg")
        compiled_sdfg = copy_sdfg.compile()
        compiled_sdfg(
            A=host_data["A"],
            B=host_data["B"],
            C=host_data["C"],
        )
        sdfg = copy_sdfg
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


def mat_add(A: np.ndarray, B: np.ndarray, C: np.ndarray):
    """Naive GEMM: C = A @ B"""
    C = A + B
    return C


def run_combo_in_worker(combo, config):
    """
    A wrapper function to be executed by each pool worker.

    It handles the creation of a temporary directory, setting up
    the environment, and running the e2e verification for a
    single combination.
    """
    try:
        # 1. Unpack the combination for this task
        (M, N, K, hwM, hwN, hwK, thread_group_dims_dace) = combo

        # 2. Create a temporary directory *for this worker*
        # This directory will only exist for the duration of this task.
        with tempfile.TemporaryDirectory() as tmp_dir:

            # 3. Change directory *within the worker*
            # This is safe as it only affects this worker process.
            os.chdir(tmp_dir)

            # 4. Prepare data and functions (this logic is moved from the main loop)
            data_and_interleavers = create_data_and_handlers(M, N, K, config, hwM, hwN, hwK, thread_group_dims_dace)

            data = data_and_interleavers["numpy_data"]
            interleavers = data_and_interleavers["interleavers"]

            run_numpy_fn = partial(mat_add, data["A"], data["B"], data["C"])
            run_sdfg_fn = partial(run_sdfg_in_tempdir, combo, interleavers, config, data)

            # 5. Run the actual verification function
            # This is the same call you had commented out
            ret_dict = run_e2e_verification(hw_config=config,
                                            data=data,
                                            interleave_handlers=interleavers,
                                            numpy_fn=run_numpy_fn,
                                            sdfg_fn=run_sdfg_fn)

            # 6. Return the result dictionary to the callback
            return ret_dict

    except Exception as e:
        # Log any exceptions that happen *inside* the worker
        # This helps in debugging parallel failures
        print(f"Error in worker (PID: {os.getpid()}) processing {combo}: {e}", flush=True)
        # You might want to log to a file here, but be careful of race conditions
        return None  # Return None so the callback can handle it


script_start_time = time.time()
SLURM_JOB_ID = str(os.environ.get("SLURM_JOB_ID"))
max_procs = int(os.environ.get("MAX_PROCS", 1))
gvsoc_path = "/home/primrose/Work/SoftHier/gvsoc"
ccache_path = "/usr/bin/ccache"
python_script_path = os.path.dirname(os.path.realpath(__file__))
temp_run_dir = str(os.environ.get("SOFTHIER_TEMP_RUN_PATH", python_script_path))
job_array_id = str(os.environ.get("SLURM_ARRAY_TASK_ID", "10"))
log_path = f"{python_script_path}/sweeplog_{SLURM_JOB_ID}.txt"
csv_filename = f"id_{job_array_id}.csv"
config.test_mode = 'perf_only'
config.skip_build_hw = True

if __name__ == "__main__":

    M, N, K, hwM, hwN, hwK, thread_group_dims_dace = 256, 256, 256, 32, 32, 32, (4, 4)
    config.dace_thread_group_dims = thread_group_dims_dace
    combo_1 = (M, N, K, hwM, hwN, hwK, thread_group_dims_dace)
    M, N, K, hwM, hwN, hwK, thread_group_dims_dace = 128, 128, 128, 32, 32, 32, (4, 4)
    combo_2 = (M, N, K, hwM, hwN, hwK, thread_group_dims_dace)
    all_combinations = [combo_1, combo_2]

    # How many processes (slots) you want
    MAX_PROCS = max_procs

    # We need a Manager to share a counter and lock across processes
    manager = multiprocessing.Manager()
    lock = manager.Lock()
    slots = manager.dict()
    slots['counter'] = 0

    # create a log file
    log_file = open(log_path, "w")
    print(f"Starting parallel execution of {len(all_combinations)} combinations...", flush=True, file=log_file)
    log_file.close()

    print(all_combinations)

    # exit(0)
    fieldnames = ["execution_period_ns"]

    # Use a Lock to ensure only one callback writes at a time
    csv_lock = multiprocessing.Lock()

    # We'll open the CSV once, write the header, and keep it open
    with open(csv_filename, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # This callback is invoked every time a job finishes
        def on_result_returned(ret_dict):
            if ret_dict is not None:
                with csv_lock:
                    execution_period_ns = ret_dict.get('execution_period_ns', None)
                    writer.writerow({"execution_period_ns": execution_period_ns})
                    # flush to ensure data is written immediately
                    csvfile.flush()
            else:
                # Handle the case where the worker returned None (due to an error)
                print("A worker task failed and returned None.", flush=True)

        # Create pool
        pool = multiprocessing.Pool(processes=MAX_PROCS, initializer=init_worker, initargs=(slots, lock))

        # --- THIS IS THE MODIFIED PART ---

        # Submit each combo to the pool via apply_async, using the new wrapper
        for combo in all_combinations:
            # We pass the wrapper function, and its arguments: (combo, config)
            # The callback 'on_result_returned' remains the same.
            pool.apply_async(
                run_combo_in_worker,  # The new wrapper function
                args=(combo, config),  # Arguments for the wrapper
                callback=on_result_returned)

        # --- END OF MODIFIED PART ---

        pool.close()
        pool.join()

    # Cleanup slot_* directories
    os.system(f"rm -rf {python_script_path}/slot_*")
    print("All combinations have finished. CSV results are in:", csv_filename)
