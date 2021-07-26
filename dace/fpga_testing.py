# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
from datetime import datetime
import inspect
import os
import multiprocessing as mp
from pathlib import Path
import pytest
import re
import subprocess as sp
from typing import Callable, Iterable, Optional, Tuple, Union

from dace import SDFG
from dace.config import Config, temporary_config

TEST_TIMEOUT = 600  # Seconds


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


def dump_logs(proc_or_logs: Union[sp.CompletedProcess, Tuple[str, str]]):
    if isinstance(proc_or_logs, tuple):
        log_out, log_err = proc_or_logs
    else:
        proc_or_logs.terminate()
        proc_or_logs.kill()
        try:
            log_out, log_err = proc_or_logs.communicate(timeout=10)
        except sp.TimeoutExpired:
            return None  # Failed to even kill the process
    if log_out:
        print(log_out)
    if log_err:
        print(log_err)
    return log_out, log_err


def _run_fpga_test(vendor: str,
                   test_function: Callable,
                   run_synthesis: bool = True,
                   assert_ii_1: bool = True):
    path = Path(inspect.getfile(test_function))
    base_name = f"{Colors.UNDERLINE}{path.stem}{Colors.END}"
    with temporary_config():
        Config.set("compiler", "use_cache", value=False)
        Config.set("cache", value="hash")
        Config.set("optimizer", "transform_on_call", value=False)
        Config.set("optimizer", "interface", value=None)
        Config.set("optimizer", "autooptimize", value=False)
        if vendor == "xilinx":
            Config.set("compiler", "fpga_vendor", value="xilinx")
            Config.set("compiler", "xilinx", "mode", value="simulation")

            # Simulation in software
            print_status(f"{base_name} [Xilinx]: Running simulation.")
            env = os.environ.copy()
            if "rtl" in path.parts:
                Config.set("compiler",
                           "xilinx",
                           "mode",
                           value="hardware_emulation")
                if "LIBRARY_PATH" not in env:
                    env["LIBRARY_PATH"] = ""
                env["LIBRARY_PATH"] += ":/usr/lib/x86_64-linux-gnu"
            sdfgs = test_function()
            if sdfgs is None:
                sdfgs = []
            elif isinstance(sdfgs, SDFG):
                sdfgs = [sdfgs]
            print_success(f"{base_name} [Xilinx]: " "Simulation successful.")

            for sdfg in sdfgs:
                build_folder = Path(sdfg.build_folder) / "build"
                if not build_folder.exists():
                    print_error(f"Build folder {build_folder} "
                                f"not found for {base_name}.")
                    return False

                # High-level synthesis
                if run_synthesis:
                    print_status(f"{base_name} [Xilinx]: Running high-level "
                                 f"synthesis for {sdfg.name}.")
                    try:
                        proc = sp.Popen(["make", "xilinx_synthesis"],
                                        env=env,
                                        cwd=build_folder,
                                        stdout=sp.PIPE,
                                        stderr=sp.PIPE,
                                        encoding="utf=8")
                        syn_out, syn_err = proc.communicate(
                            timeout=TEST_TIMEOUT)
                    except sp.TimeoutExpired:
                        dump_logs(proc)
                        print_error(f"{base_name} [Xilinx]: High-level "
                                    f"synthesis timed out after "
                                    f"{TEST_TIMEOUT} seconds.")
                        return False
                    if proc.returncode != 0:
                        dump_logs(proc)
                        print_error(f"{base_name} [Xilinx]: High-level "
                                    f"synthesis failed.")
                        return False
                    print_success(f"{base_name} [Xilinx]: High-level "
                                  f"synthesis successful for "
                                  f"{sdfg.name}.")
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
                            print_error(f"{base_name} [Xilinx]: HLS "
                                        f"log file not found.")
                            return False
                        hls_log = open(hls_log, "r").read()
                        for m in re.finditer(r"Final II = ([0-9]+)", hls_log):
                            loops_found = True
                            if int(m.group(1)) != 1:
                                dump_logs((syn_out, syn_err))
                                print_error(f"{base_name} [Xilinx]: Failed "
                                            f"to achieve II=1.")
                                return False
                        if not loops_found:
                            dump_logs((syn_out, syn_err))
                            print_error(f"{base_name} [Xilinx]: No "
                                        f"pipelined loops found.")
                            return False
                        print_success(f"{base_name} [Xilinx]: II=1 "
                                      f"achieved.")

        elif vendor == "intel_fpga":
            # Set environment variables
            Config.set("compiler", "fpga_vendor", value="intel_fpga")
            Config.set("compiler", "default_data_types", value="C")
            Config.set("compiler", "intel_fpga", "mode", value="emulator")

            # Simulation in software
            print_status(f"{base_name} [Intel FPGA]: Running " f"emulation.")
            test_function()
            print_success(f"{base_name} [Intel FPGA]: Emulation "
                          f"successful.")
        else:
            raise ValueError(f"Unrecognized vendor {vendor}.")


def fpga_test(run_synthesis: bool = True,
              assert_ii_1: bool = True,
              xilinx: bool = True,
              intel: bool = True):
    """
    Decorator to run an FPGA test with pytest, setting the appropriate
    variables and performing additional checks, such as running HLS and
    asserting II=1. The test function must return an SDFG or a list of SDFGs
    that will be used for this check.

    :param run_synthesis: Whether to run HLS for Xilinx tests (Intel tests will always run synthesis).
    :param assert_ii_1: Assert that all loops have been fully pipelined (currently only implemented for Xilinx).
    :param xilinx: Run as a Xilinx test.
    :param intel: Run as an Intel test.
    """

    # Check arguments
    if not xilinx and not intel:
        raise ValueError("FPGA test must be run for Xilinx, Intel, or both.")
    pytest_params = []
    if xilinx:
        pytest_params.append("xilinx")
    if intel:
        pytest_params.append("intel_fpga")

    def decorator(test_function: Callable):

        @pytest.mark.fpga
        @pytest.mark.parametrize("vendor", pytest_params)
        def wrapper(vendor: Optional[str]):
            if vendor == None:
                vendor = Config.get("compiler", "fpga_vendor")
            p = mp.Process(target=_run_fpga_test,
                           args=(vendor, test_function, run_synthesis,
                                 assert_ii_1))
            p.start()
            p.join()

        return wrapper

    return decorator
