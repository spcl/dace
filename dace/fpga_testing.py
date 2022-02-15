# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
from datetime import datetime
import importlib.util
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

TEST_TIMEOUT = 900  # Timeout tests after 15 minutes


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


# https://stackoverflow.com/a/33599967/2949968
class FPGATestProcess(mp.Process):
    def __init__(self, *args, **kwargs):
        mp.Process.__init__(self, *args, **kwargs)
        self._pconn, self._cconn = mp.Pipe()
        self._exception = None

    def run(self):
        try:
            ret = mp.Process.run(self)
            self._cconn.send(ret)
        except Exception as e:
            self._cconn.send(e)
            raise e

    @property
    def exception(self):
        if self._pconn.poll():
            self._exception = self._pconn.recv()
        return self._exception


class TestFailed(Exception):
    pass


def raise_error(message):
    print_error(message)
    raise TestFailed(message)


def _run_fpga_test(vendor: str, test_function: Callable, run_synthesis: bool = True, assert_ii_1: bool = True):
    path = Path(inspect.getfile(test_function))
    base_name = f"{path.stem}::{Colors.UNDERLINE}{test_function.__name__}{Colors.END}"
    with temporary_config():
        Config.set("compiler", "use_cache", value=False)
        Config.set("cache", value="unique")
        Config.set("optimizer", "transform_on_call", value=False)
        Config.set("optimizer", "interface", value=None)
        Config.set("optimizer", "autooptimize", value=False)
        if vendor == "xilinx":
            Config.set("compiler", "fpga", "vendor", value="xilinx")
            Config.set("compiler", "xilinx", "mode", value="simulation")

            # Simulation in software
            print_status(f"{base_name} [Xilinx]: Running simulation.")
            if "rtl" in path.parts:
                Config.set("compiler", "xilinx", "mode", value="hardware_emulation")
                if "LIBRARY_PATH" not in os.environ:
                    os.environ["LIBRARY_PATH"] = ""
                    library_path_backup = None
                else:
                    library_path_backup = os.environ["LIBRARY_PATH"]
                os.environ["LIBRARY_PATH"] += ":/usr/lib/x86_64-linux-gnu"
            sdfgs = test_function()
            if "rtl" in path.parts:
                if library_path_backup is None:
                    del os.environ["LIBRARY_PATH"]
                else:
                    os.environ["LIBRARY_PATH"] = library_path_backup
            if sdfgs is None:
                raise_error("No SDFG(s) returned by FPGA test.")
            elif isinstance(sdfgs, SDFG):
                sdfgs = [sdfgs]
            print_success(f"{base_name} [Xilinx]: " "Simulation successful.")

            for sdfg in sdfgs:
                build_folder = Path(sdfg.build_folder) / "build"
                if not build_folder.exists():
                    raise_error(f"Build folder {build_folder} " f"not found for {base_name}.")

                # High-level synthesis
                if run_synthesis:
                    print_status(f"{base_name} [Xilinx]: Running high-level " f"synthesis for {sdfg.name}.")
                    try:
                        proc = sp.Popen(["make", "synthesis"],
                                        cwd=build_folder,
                                        stdout=sp.PIPE,
                                        stderr=sp.PIPE,
                                        encoding="utf=8")
                        syn_out, syn_err = proc.communicate(timeout=TEST_TIMEOUT)
                    except sp.TimeoutExpired:
                        dump_logs(proc)
                        raise_error(f"{base_name} [Xilinx]: High-level "
                                    f"synthesis timed out after "
                                    f"{TEST_TIMEOUT} seconds.")
                    if proc.returncode != 0:
                        dump_logs(proc)
                        raise_error(f"{base_name} [Xilinx]: High-level " f"synthesis failed.")
                    print_success(f"{base_name} [Xilinx]: High-level " f"synthesis successful for " f"{sdfg.name}.")
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
                            raise_error(f"{base_name} [Xilinx]: HLS " f"log file not found.")
                        hls_log = open(hls_log, "r").read()
                        for m in re.finditer(r"Final II = ([0-9]+)", hls_log):
                            loops_found = True
                            if int(m.group(1)) != 1:
                                dump_logs((syn_out, syn_err))
                                raise_error(f"{base_name} [Xilinx]: " f"Failed to achieve II=1.")
                        if not loops_found:
                            dump_logs((syn_out, syn_err))
                            raise_error(f"{base_name} [Xilinx]: No " f"pipelined loops found.")
                        print_success(f"{base_name} [Xilinx]: II=1 " f"achieved.")

        elif vendor == "intel_fpga":
            # Set environment variables
            Config.set("compiler", "fpga", "vendor", value="intel_fpga")
            Config.set("compiler", "default_data_types", value="C")
            Config.set("compiler", "intel_fpga", "mode", value="emulator")

            # Simulation in software
            print_status(f"{base_name} [Intel FPGA]: Running " f"emulation.")
            test_function()
            print_success(f"{base_name} [Intel FPGA]: Emulation " f"successful.")
        else:
            raise ValueError(f"Unrecognized vendor {vendor}.")


def fpga_test(run_synthesis: bool = True, assert_ii_1: bool = True, xilinx: bool = True, intel: bool = True):
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
                vendor = Config.get("compiler", "fpga", "vendor")
            p = FPGATestProcess(target=_run_fpga_test, args=(vendor, test_function, run_synthesis, assert_ii_1))
            p.start()
            p.join(timeout=TEST_TIMEOUT)
            if p.is_alive():
                p.kill()
                raise_error(f"Test {Colors.UNDERLINE}{test_function.__name__}" f"{Colors.END} timed out.")
            if p.exception:
                raise p.exception

        return wrapper

    return decorator


def xilinx_test(*args, **kwargs):
    return fpga_test(*args, xilinx=True, intel=False, **kwargs)


def intel_fpga_test(*args, **kwargs):
    return fpga_test(*args, xilinx=False, intel=True, **kwargs)


def import_sample(path: Union[Path, str]):
    """
    Import a Python file from the samples directory as a module so it can be
    used in a test.

    :param path: Path relative to the DaCe samples directory.
    """
    path = Path(__file__).parent.parent / "samples" / Path(path)
    if not path.exists():
        raise ValueError(f"Sample {path} not found.")
    name = path.stem
    spec = importlib.util.spec_from_file_location(name, path)
    loaded_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded_module)
    return loaded_module
