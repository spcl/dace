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

# (relative path, sdfg name, run synthesis, assert II=1, args to executable)
TESTS = [
    ("tests/fpga/remove_degenerate_loop.py", "remove_degenerate_loop_test",
     True, False, []),
    ("tests/fpga/pipeline_scope.py", "pipeline_test", True, True, []),
]


class Colors:
    SUCCESS = "\033[92m"
    STATUS = "\033[94m"
    ERROR = "\033[91m"
    BOLD = "\033[1m"
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


def run_test(path: Path, sdfg_name: str, run_synthesis: bool, assert_ii_1: bool,
             args: Iterable[Any]):
    path = DACE_DIR / path
    if not path.exists():
        raise ValueError(f"Path {path} does not exist.")
    base_name = path.stem
    print_status(f"Running simulation for {base_name}.")
    env = master_env.copy()
    proc = sp.run(map(str, [sys.executable, path] + args),
           env=env,
           cwd=TEST_DIR,
           capture_output=True,
           check=True)
    print_success(f"Simulation of {base_name} successful.")
    build_folder = TEST_DIR / ".dacecache" / sdfg_name / "build"
    if not build_folder.exists():
        raise ValueError(f"Invalid SDFG name {sdfg_name} for {base_name}.")
    open(build_folder / "simulation.out", "wb").write(proc.stdout)
    open(build_folder / "simulation.err", "wb").write(proc.stderr)
    print_status(f"Running synthesis for {base_name}.")
    if run_synthesis:
        proc = sp.run(["make", "xilinx_synthesis"],
               env=env,
               cwd=build_folder,
               capture_output=True,
               check=True)
        print_success(f"Synthesis successful for {base_name}.")
        open(build_folder / "synthesis.out", "wb").write(proc.stdout)
        open(build_folder / "synthesis.err", "wb").write(proc.stderr)
        if assert_ii_1:
            for m in re.finditer(r"Final II = ([0-9]+)", str(proc.stdout)):
               if int(m.group(1)) != 1:
                  raise RuntimeError("{base_name} did not achieve II=1.")
            print_success(f"Verified II=1 for {base_name}.")

with mp.Pool() as pool:
    pool.starmap(run_test, TESTS)
