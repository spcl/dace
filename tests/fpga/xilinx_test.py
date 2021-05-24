#!/usr/bin/env python3
# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import click
import os
from pathlib import Path
import re
import subprocess as sp
import sys
from typing import Any, Iterable, Union

TEST_TIMEOUT = 600  # Seconds

from fpga_testing import (Colors, DACE_DIR, TEST_DIR, cli, dump_logs,
                          print_status, print_success, print_error)

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
    # This doesn't pipeline with Vitis 2020.1 for whatever reason (it pipelines
    # with both 2019.2 and 2020.2), so just switch this back on once CI starts
    # running 2020.2 or newer.
    ("tests/transformations/mapfusion_fpga.py",
     ["multiple_fusions_1", "fusion_with_transient_1"], True, False, []),
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
    # TODO: Reduce should achieve II=1, but currently does not.
    ("tests/fpga/reduce_fpga.py", [
        "reduction_sum_one_axis", "reduction_sum_all_axis", "reduction_sum_4D",
        "reduction_max"
    ], True, False, []),
    # Multiple gearboxing
    ("tests/fpga/multiple_veclen_conversions.py", "multiple_veclen_conversions",
     True, False, []),
    # Views
    ("tests/fpga/reshape_view_fpga.py",
     ["view_fpga", "reshp_np_1", "reshapedst_1"], True, False, []),
    # RTL cores
    ("tests/rtl/hardware_test.py", "floating_point_vector_plus_scalar", True,
     False, [1]),
    # Auto-opt for FPGA
    ("tests/fpga/auto_opt_fpga.py", ["global_to_local_1", "rr_interleave_1"], True, False, []),
    # Over approx Shapes
    ("tests/fpga/overapprox_transient_shapes.py", ["overapprox_1"], True, False, []),
]


def run(path: Path, sdfg_names: Union[str, Iterable[str]], run_synthesis: bool,
        assert_ii_1: bool, args: Iterable[Any]):

    # Set environment variables
    env = os.environ.copy()
    env["DACE_compiler_fpga_vendor"] = "xilinx"
    env["DACE_compiler_use_cache"] = "0"
    # We would like to use DACE_cache=hash, but we need to know which folder to
    # run synthesis in.
    env["DACE_cache"] = "name"
    env["DACE_compiler_xilinx_mode"] = "simulation"
    os.environ["DACE_optimizer_transform_on_call"] = "0"
    os.environ["DACE_optimizer_interface"] = ""
    os.environ["DACE_optimizer_autooptimize"] = "0"

    path = DACE_DIR / path
    if not path.exists():
        print_error(f"Path {path} does not exist.")
        return False
    base_name = f"{Colors.UNDERLINE}{path.stem}{Colors.END}"

    # Simulation in software
    print_status(f"{base_name}: Running simulation.")
    if "rtl" in path.parts:
        env["DACE_compiler_xilinx_mode"] = "hardware_emulation"
        if "LIBRARY_PATH" not in env:
            env["LIBRARY_PATH"] = ""
        env["LIBRARY_PATH"] += ":/usr/lib/x86_64-linux-gnu"
    try:
        proc = sp.Popen(map(str, [sys.executable, path] + args),
                        env=env,
                        cwd=TEST_DIR,
                        stdout=sp.PIPE,
                        stderr=sp.PIPE,
                        encoding="utf-8")
        sim_out, sim_err = proc.communicate(timeout=TEST_TIMEOUT)
    except sp.TimeoutExpired:
        dump_logs(proc)
        print_error(f"{base_name}: Simulation timed out "
                    f"after {TEST_TIMEOUT} seconds.")
        return False
    if proc.returncode != 0:
        dump_logs((sim_out, sim_err))
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

        # High-level synthesis
        if run_synthesis:
            print_status(
                f"{base_name}: Running high-level synthesis for {sdfg_name}.")
            try:
                proc = sp.Popen(["make", "xilinx_synthesis"],
                                env=env,
                                cwd=build_folder,
                                stdout=sp.PIPE,
                                stderr=sp.PIPE,
                                encoding="utf=8")
                syn_out, syn_err = proc.communicate(timeout=TEST_TIMEOUT)
            except sp.TimeoutExpired:
                dump_logs(proc)
                print_error(f"{base_name}: High-level synthesis timed out "
                            f"after {TEST_TIMEOUT} seconds.")
                return False
            if proc.returncode != 0:
                dump_logs(proc)
                print_error(f"{base_name}: High-level synthesis failed.")
                return False
            print_success(f"{base_name}: High-level synthesis "
                          f"successful for {sdfg_name}.")
            open(build_folder / "synthesis.out", "w").write(syn_out)
            open(build_folder / "synthesis.err", "w").write(syn_err)

            # Check if loops were pipelined with II=1
            if assert_ii_1:
                loops_found = False
                for f in build_folder.iterdir():
                    if "hls.log" in f.name:
                        hls_log = f
                        break
                else:
                    print_error(f"{base_name}: HLS log file not found.")
                    return False
                hls_log = open(hls_log, "r").read()
                for m in re.finditer(r"Final II = ([0-9]+)", hls_log):
                    loops_found = True
                    if int(m.group(1)) != 1:
                        dump_logs((syn_out, syn_err))
                        print_error(f"{base_name}: Failed to achieve II=1.")
                        return False
                if not loops_found:
                    dump_logs((syn_out, syn_err))
                    print_error("{base_name}: No pipelined loops found.")
                    return False
                print_success(f"{base_name}: II=1 achieved.")

    return True


@click.command()
@click.option("--parallel/--no-parallel", default=True)
@click.argument("tests", nargs=-1)
def xilinx_cli(parallel, tests):
    """
    If no arguments are specified, runs all tests. If any arguments are
    specified, runs only the tests specified (matching on file name or SDFG
    name).
    """
    cli(TESTS, run, tests, parallel)


if __name__ == "__main__":
    xilinx_cli()
