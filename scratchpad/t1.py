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
import ctypes
from dace.soft_hier.utils.generate_sdfg import _my_gen_baseline_matmul_sdfg
from dace.soft_hier.utils.read_from_dump_file import get_address_and_read_from_file
from dace.soft_hier.utils.run_e2e_verification import run_e2e_verification, HardwareConfig, setup_hw_env_dace
from functools import partial


def _get_gvsoc_path() -> str:
    """Get GVSOC path from environment or by locating gvsoc binary"""
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


def create_data_and_handlers(M_val, N_val, K_val, hw_config: HardwareConfig):
    DTYPE_INPUT = hw_config.dtype_input
    DTYPE_OUTPUT = hw_config.dtype_output
    hardware_thread_group_dims = hw_config.hardware_thread_group_dims

    A_host = np.fromfunction(lambda i, j: i + j, (M_val, K_val), dtype=DTYPE_INPUT)
    B_host = np.fromfunction(lambda i, j: 2 * i + 2 * j, (K_val, N_val), dtype=DTYPE_INPUT)
    C_host = np.ones((M_val, N_val), dtype=DTYPE_OUTPUT)

    A_handler = InterleaveHandler(array=A_host, block_shape=(hwM, hwK), cluster_dims=hardware_thread_group_dims)
    A_handler.split_horizental()
    A_handler.place_to_range(place_range=(0, 7, 1))

    B_handler = InterleaveHandler(array=B_host, block_shape=(hwK, hwN), cluster_dims=hardware_thread_group_dims)
    B_handler.split_vertical()
    B_handler.place_to_range(place_range=(0, 7, 1))

    C_handler = InterleaveHandler(array=C_host, block_shape=(hwM, hwN), cluster_dims=hardware_thread_group_dims)
    C_handler.split_to_blocks()
    C_handler.place_to_range(place_range=(0, 7, 1))

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

    (M_val, N_val, K_val, hwM, hwN, hwK) = combo

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
                                         args_only=False)

    M = M_val
    N = N_val
    K = K_val

    sdfg = _my_gen_baseline_matmul_sdfg(
        hardware_matmul_mnk=hardware_matmul_mnk,
        global_storage=dace.dtypes.StorageType.SoftHier_HBM,
        local_storage=dace.dtypes.StorageType.SoftHier_TCDM,
        device_schedule=dace.dtypes.ScheduleType.SoftHier_Device,
        thread_group_schedule=dace.dtypes.ScheduleType.SoftHier_Cluster,
        thread_group_dims=hw_config.hardware_thread_group_dims,
        hbm_split_scheme=[interleaver.split_scheme for interleaver in interleaver_list],
        hbm_placement_scheme=[interleaver.placement_scheme for interleaver in interleaver_list],
        is_hbm_interleaved=True,
        input_float=hw_config.dace_input_type,
        output_float=hw_config.dace_output_type,
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
    compiled_sdfg(A=host_data["A"], B=host_data["B"], C=host_data["C"], M=M_val, N=N_val, K=K_val)
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
    """Naive GEMM: C = A @ B + C"""
    C = A @ B + C


if __name__ == "__main__":
    setup_hw_env_dace(config)

    M, N, K, hwM, hwN, hwK = 512, 512, 512, 64, 64, 128
    combo = (512, 512, 512, 64, 64, 128)
    data_and_interleavers = create_data_and_handlers(M, N, K, config)
    data = data_and_interleavers["numpy_data"]
    interleavers = data_and_interleavers["interleavers"]
    d = run_sdfg_in_tempdir(combo, interleavers, config, data)

    run_numpy = partial(gemm, data["A"], data["B"], data["C"])
    run_sdfg = partial(run_sdfg_in_tempdir, combo, interleavers, config, data)

    run_e2e_verification(hw_config=config,
                         data=data,
                         interleave_handlers=interleavers,
                         numpy_fn=run_numpy,
                         sdfg_fn=run_sdfg)
