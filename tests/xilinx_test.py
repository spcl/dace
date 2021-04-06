#!/usr/bin/env python3
# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
from collections import OrderedDict
from dace import Config
from datetime import datetime
import multiprocessing as mp
import os
from pathlib import Path
import re
import shutil
import subprocess as sp
import sys
from typing import Any, Iterable, Union

DACE_DIR = Path(__file__).parent.parent.absolute()
TEST_DIR = Path(__file__).parent.absolute()

# (relative path, sdfg name(s), run synthesis, assert II=1, args to executable)
TESTS = [
    ("tests/fpga/remove_degenerate_loop.py", "remove_degenerate_loop_test",
     True, True, []),
    ("tests/fpga/pipeline_scope.py", "pipeline_test", True, True, []),
    ("tests/fpga/veclen_copy_conversion.py", "veclen_copy_conversion", True,
     True, []),
    ("samples/fpga/axpy_transformed.py", "axpy_fpga_24", True, True, [24]),
    ("samples/fpga/spmv_fpga_stream.py", "spmv_fpga_stream", True, False,
     [64, 64, 640]),
    ("samples/fpga/matrix_multiplication_systolic.py",
     "mm_fpga_systolic_4_64x64x64", True, True, [64, 64, 64, 4, "-specialize"]),
    ("samples/fpga/filter_fpga_vectorized.py", "filter_fpga_vectorized_4", True,
     True, [8192, 4, 0.25]),
    # ("jacobi_fpga_systolic.py", "jacobi_fpga_systolic_4_Hx128xT", True, True, [1, 128, 128, 8, 4]),
    ("samples/fpga/gemv_transposed_fpga.py", "gemv_transposed_1024xM", True,
     False, [1024, 1024]),
    ("tests/fpga/multiple_kernels.py", "multiple_kernels", True, False, []),
    ("tests/fpga/unique_nested_sdfg_fpga.py", "two_vecAdd", True, False, []),
    ("tests/fpga/nested_sdfg_as_kernel.py", "nested_sdfg_kernels", True, False,
     []),
    ("tests/fpga/streaming_memory.py", "streamingcomp_1", True, True, []),
    ("tests/fpga/conflict_resolution.py", "fpga_conflict_resolution", True,
     False, []),
    # BLAS
    ("tests/blas/nodes/axpy_test.py", "axpy_test_fpga_1_w4_1", True, True,
     ["--target", "fpga"]),
    ("tests/blas/nodes/dot_test.py", "dot_FPGA_PartialSums_float_w16_1", True,
     True, ["--target", "xilinx"]),
    ("tests/blas/nodes/gemv_test.py", "gemv_FPGA_TilesByColumn_float_True_w4_1",
     True, True,
     ["--target", "tiles_by_column", "--transpose", "--vectorize", "4"]),
    ("tests/blas/nodes/gemv_test.py", "gemv_FPGA_Accumulate_float_False_w4_1",
     True, True, ["--target", "accumulate", "--vectorize", "4"]),
    ("tests/blas/nodes/ger_test.py", "ger_test_1", True, True,
     ["--target", "fpga"]),
    # This test contains three SDFGs: full check only the first one for the sake of testing time
    ("tests/fpga/gemm_fpga.py", "gemm_vectorized", True, True, []),
    # STL library nodes
    ("tests/fpga/reduce_fpga.py", [
        "reduction_sum_one_axis", "reduction_sum_all_axis", "reduction_sum_4D",
        "reduction_max"
    ], True, True, []),
    # Multiple gearboxing
    ("tests/fpga/multiple_veclen_conversions.py", "multiple_veclen_conversions",
     True, False, []),
    # Views
    ("tests/fpga/reshape_view_fpga.py",
     ["view_fpga", "reshp_np_1", "reshapedst_1"], True, False, []),
    # RTL cores
    ("tests/rtl/hardware_test.py", "floating_point_vector_plus_scalar", True,
     False, [1]),
]


class Colors:
    SUCCESS = "\033[92m"
    STATUS = "\033[94m"
    ERROR = "\033[91m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    END = "\033[0m"


def print_status(message):
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"{Colors.STATUS}{Colors.BOLD}[{timestamp}]{Colors.END} {message}")


def print_success(message):
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"{Colors.SUCCESS}{Colors.BOLD}[{timestamp}]{Colors.END} {message}")


def print_error(message):
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"{Colors.ERROR}{Colors.BOLD}[{timestamp}]{Colors.END} {message}")


# Find Xilinx compiler
xilinx_compiler = Config.get("compiler", "xilinx", "path")
if not xilinx_compiler.strip():
    xilinx_compiler = shutil.which("v++")
if not xilinx_compiler:
    xilinx_compiler = shutil.which("xocc")
if not xilinx_compiler:
    raise RuntimeError("Cannot find Xilinx compiler executable v++/xocc.")

# Set environment variables
master_env = os.environ.copy()
master_env["DACE_compiler_fpga_vendor"] = "xilinx"
master_env["DACE_compiler_use_cache"] = "0"
master_env["DACE_testing_single_cache"] = "0"
master_env["DACE_compiler_xilinx_mode"] = "simulation"


def run_test(path: Path, sdfg_names: Union[str, Iterable[str]],
             run_synthesis: bool, assert_ii_1: bool, args: Iterable[Any]):
    path = DACE_DIR / path
    if not path.exists():
        print_error(f"Path {path} does not exist.")
        return False
    base_name = f"{Colors.UNDERLINE}{path.stem}{Colors.END}"
    print_status(f"{base_name}: Running simulation.")
    env = master_env.copy()
    if "rtl" in path.parts:
        env["DACE_compiler_xilinx_mode"] = "hardware_emulation"
        if "LIBRARY_PATH" not in env:
            env["LIBRARY_PATH"] = ""
        env["LIBRARY_PATH"] += ":/usr/lib/x86_64-linux-gnu"
    proc = sp.run(map(str, [sys.executable, path] + args),
                  env=env,
                  cwd=TEST_DIR,
                  capture_output=True,
                  check=False,
                  timeout=600)
    sim_out = proc.stdout.decode("utf-8").strip()
    sim_err = proc.stderr.decode("utf-8").strip()
    if proc.returncode != 0:
        if sim_out:
            print(sim_out)
        if sim_err:
            print(sim_err)
        print_error(f"{base_name}: Simulation failed.")
        return False
    print_success(f"{base_name}: Simulation successful.")
    if isinstance(sdfg_names, str):
        sdfg_names = [sdfg_names]
    for sdfg_name in sdfg_names:
        build_folder = TEST_DIR / ".dacecache" / sdfg_name / "build"
        if not build_folder.exists():
            print_error(f"Invalid SDFG name {sdfg_name} for {base_name}.")
            return False
        open(build_folder / "simulation.out", "w").write(sim_out)
        open(build_folder / "simulation.err", "w").write(sim_err)
        print_status(f"{base_name}: Running synthesis for {sdfg_name}.")
        if run_synthesis:
            proc = sp.run(["make", "xilinx_synthesis"],
                          env=env,
                          cwd=build_folder,
                          capture_output=True,
                          check=False,
                          timeout=600)
            syn_out = proc.stdout.decode("utf-8").strip()
            syn_err = proc.stderr.decode("utf-8").strip()
            if proc.returncode != 0:
                print_error(f"{base_name}: Synthesis failed.")
                return syn_out, syn_err
            print_success(f"{base_name}: Synthesis successful for {sdfg_name}.")
            open(build_folder / "synthesis.out", "w").write(syn_out)
            open(build_folder / "synthesis.err", "w").write(syn_err)
            if assert_ii_1:
                loops_found = False
                for m in re.finditer(r"Final II = ([0-9]+)", str(proc.stdout)):
                    loops_found = True
                    if int(m.group(1)) != 1:
                        if syn_out:
                            print(syn_out)
                        if syn_err:
                            print(syn_err)
                        print_error(f"{base_name}: Failed to achieve II=1.")
                        return False
                if not loops_found:
                    if syn_out:
                        print(syn_out)
                    if syn_err:
                        print(syn_err)
                    print_error("{base_name}: No pipelined loops found.")
                    return False
                print_success(f"{base_name}: II=1 achieved.")
    return True


# Run tests in parallel using default number of workers
with mp.Pool() as pool:
    results = pool.starmap(run_test, TESTS)
    if all(results):
        sys.exit(0)
    else:
        sys.exit(1)
