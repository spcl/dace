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
    ("tests/fpga/veclen_conversion.py", "veclen_conversion", []),
    ("tests/fpga/veclen_copy_conversion.py", "veclen_copy_conversion", []),
    # Test removing degenerate loops that only have a single iteration
    ("tests/fpga/remove_degenerate_loop.py", "remove_degenerate_loop_test", []),
    ("tests/fpga/pipeline_scope.py", "pipeline_test", []),
    # Test shift register abstraction with stencil code
    ("tests/fpga/fpga_stencil.py", "fpga_stcl_test", []),
    ### Map tiling and WCR ####
    # First tile then transform
    ("tests/intel_fpga/dot.py", "dot_1", ["--tile-first"]),
    # Other way around
    ("tests/intel_fpga/dot.py", "dot_1", ["--no-tile-first"]),
    # simple WCR (accumulates on scalar)
    ("tests/fpga/conflict_resolution.py", "fpga_conflict_resolution", []),
    # Simple reduce
    ("tests/intel_fpga/vector_reduce.py", "vector_reduce", []),
    # Matrix multiplication sample
    ("samples/simple/matmul.py", "matmul_1", ["--version", "fpga_naive"]),
    ### Type inference ###
    ("samples/simple/mandelbrot.py", "mandelbrot_1", ["--fpga"]),
    ("tests/intel_fpga/type_inference.py", "type_inference_1", []),
    ("tests/intel_fpga/constant_type_inference.py", "constant_type_inference",
     []),
    ### Systolic array ###
    ("tests/intel_fpga/simple_systolic_array.py", "simple_systolic_array_4",
     [128, 4]),
    ("samples/fpga/matrix_multiplication_systolic.py",
     "mm_fpga_systolic_4_NxKx256", [256, 256, 256, 4]),
    ("samples/fpga/jacobi_fpga_systolic.py", "jacobi_fpga_systolic_8_Hx8192xT",
     []),
    # Execute some of the compatible tests in samples/fpga (some of them have C++ code in tasklet)
    # They contain streams
    ("tests/intel_fpga/async.py", "async_test", []),
    ("samples/fpga/filter_fpga.py", "filter_fpga", [1000, 0.2]),
    ("samples/fpga/matrix_multiplication_stream.py", "mm_fpga_stream_NxKx128",
     [128, 128, 128]),
    ("samples/fpga/spmv_fpga_stream.py", "spmv_fpga_stream", [128, 128, 64]),
    ("samples/fpga/axpy_transformed.py",
     ["axpy_test_fpga_0_w1_1", "axpy_test_fpga_1_w4_1"], [24]),
    ("tests/fpga/multiple_kernels.py", "multiple_kernels", []),
    ("tests/fpga/unique_nested_sdfg_fpga.py", "two_vecAdd", []),
    ### BLAS ###
    ("tests/blas/nodes/axpy_test.py",
     ["axpy_test_fpga_0_w1_1", "axpy_test_fpga_1_w4_1"], ["--target", "fpga"]),
    ("tests/blas/nodes/dot_test.py", "dot_FPGA_Accumulate_float_w16_1",
     ["--target", "intel_fpga"]),
    ("tests/blas/nodes/gemv_test.py", "gemv_FPGA_TilesByColumn_float_True_w4_1",
     ["--target", "tiles_by_column", "--transpose", "--vectorize", 4]),
    ("tests/blas/nodes/gemv_test.py", "gemv_FPGA_Accumulate_float_False_w4_1",
     ["--target", "accumulate", "--vectorize", 4]),
    ("tests/blas/nodes/ger_test.py", "ger_test_1", ["--target", "fpga"]),
    ("tests/fpga/gemm_fpga.py",
     ["gemm_not_multiple_of", "gemm_vectorized", "matmul_np_1"], []),
    # STL
    ("tests/fpga/reduce_fpga.py", "reduction_sum_one_axis", []),
    # Nested SDFGs generated as FPGA kernels
    ("tests/fpga/nested_sdfg_as_kernel.py", "nested_sdfg_kernels", []),
    # Generating autorun kernels
    ("tests/intel_fpga/autorun.py", "autorun_test", []),
    # Multiple gearboxing
    ("tests/fpga/multiple_veclen_conversions.py", "multiple_veclen_conversions",
     []),
    # Channels mangling
    ("tests/intel_fpga/channels_mangling.py", "channels_mangling", []),
    # Views
    ("tests/fpga/reshape_view_fpga.py", "reshape_view_fpga", []),
    # Test map fusion resulting in Tasklet -> Tasklet memlets
    ("tests/transformations/mapfusion_fpga.py",
     ["multiple_fusions_1", "fusion_with_transient_1"], []),
]


def run(path: Path, sdfg_names: Union[str, Iterable[str]], args: Iterable[Any]):

    # Set environment variables
    os.environ["DACE_compiler_fpga_vendor"] = "intel_fpga"
    os.environ["DACE_compiler_use_cache"] = "0"
    os.environ["DACE_compiler_default_data_types"] = "C"
    os.environ["DACE_testing_single_cache"] = "0"
    os.environ["DACE_compiler_intel_fpga_mode"] = "emulator"

    path = DACE_DIR / path
    if not path.exists():
        print_error(f"Path {path} does not exist.")
        return False
    base_name = f"{Colors.UNDERLINE}{path.stem}{Colors.END}"

    # Simulation in software
    print_status(f"{base_name}: Running emulation.")
    try:
        proc = sp.Popen(map(str, [sys.executable, path] + args),
                        cwd=TEST_DIR,
                        stdout=sp.PIPE,
                        stderr=sp.PIPE,
                        encoding="utf-8")
        sim_out, sim_err = proc.communicate(timeout=TEST_TIMEOUT)
    except sp.TimeoutExpired:
        dump_logs(proc)
        print_error(f"{base_name}: Emulation timed out "
                    f"after {TEST_TIMEOUT} seconds.")
        return False
    if proc.returncode != 0:
        dump_logs((sim_out, sim_err))
        print_error(f"{base_name}: Emulation failed.")
        return False
    print_success(f"{base_name}: Emulation successful.")

    if isinstance(sdfg_names, str):
        sdfg_names = [sdfg_names]
    for sdfg_name in sdfg_names:
        build_folder = TEST_DIR / ".dacecache" / sdfg_name / "build"
        if not build_folder.exists():
            print_error(f"Invalid SDFG name {sdfg_name} for {base_name}.")
            return False
        open(build_folder / "simulation.out", "w").write(sim_out)
        open(build_folder / "simulation.err", "w").write(sim_err)

    return True


@click.command()
@click.option("--parallel/--no-parallel", default=True)
@click.argument("tests", nargs=-1)
def intel_fpga_cli(parallel, tests):
    """
    If no arguments are specified, runs all tests. If any arguments are
    specified, runs only the tests specified (matching on file name or SDFG
    name).
    """
    cli(TESTS, run, tests, parallel)


if __name__ == "__main__":
    intel_fpga_cli()
