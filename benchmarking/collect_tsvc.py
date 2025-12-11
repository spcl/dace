#!/usr/bin/env python3
import copy
import importlib.util
import inspect
from pathlib import Path
from types import ModuleType
from typing import List, Callable
import dace
from dace.frontend.python.parser import DaceProgram
import ctypes
import subprocess
import pathlib
import os
import numpy as np

from dace.transformation.interstate import LoopToMap

LEN_1D = dace.symbol("LEN_1D")
LEN_2D = dace.symbol("LEN_2D")
ITERATIONS = dace.symbol("ITERATIONS")

symbol_value_map = {
    LEN_1D: 32,
    LEN_2D: 32,
    ITERATIONS: 1
}

import csv
from pathlib import Path

def write_runtimes(csv_path, cpp_name, result):
    """
    Append runtimes for a function to a CSV.
    
    Writes columns:
        cpp_name, cpp_runtime_ns, dace_runtime_ns
    """
    csv_path = Path(csv_path)
    write_header = not csv_path.exists()

    with csv_path.open("a", newline="") as f:
        w = csv.writer(f)

        if write_header:
            w.writerow(["cpp_name", "cpp_runtime_ns", "dace_runtime_ns"])

        w.writerow([
            cpp_name,
            result["cpp_runtime_ns"],
            result["dace_runtime_ns"],
        ])

def load_module_from_path(path: str) -> ModuleType:
    """
    Dynamically load a Python module from a file path.
    """
    file_path = Path(path).resolve()
    spec = importlib.util.spec_from_file_location(file_path.stem, file_path)
    if spec is None:
        raise ImportError(f"Could not create module spec for {path}")

    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise ImportError(f"Spec loader missing for {path}")

    spec.loader.exec_module(module)
    return module


def collect_dace_programs(module) -> List[DaceProgram]:
    """
    Collect all DaceProgram objects from the module.
    Optionally filter by name prefix 'dace_'.
    """
    programs = []

    for name, obj in inspect.getmembers(module):
        # Dace programs are instances of DaceProgram
        if isinstance(obj, DaceProgram) and name.startswith("dace_"):
            programs.append(obj)

    return programs



def load_dace_functions_from_file(path: str) -> List[Callable]:
    """
    Convenience wrapper: load the module, extract dace_* functions.
    """
    module = load_module_from_path(path)
    return collect_dace_programs(module)


LIB_NAME = "libtsvcpp.so"
CPP_FILE = "/home/primrose/Work/dace/tests/passes/tsvcpp.cpp"

SAVE_SDFGS=False

def build_tsvcpp_lib():
    """Compile tsvcpp.cpp into a shared library located next to this Python file."""

    # Directory where THIS Python file is located
    base_dir = pathlib.Path(__file__).resolve().parent
    pid = os.getpid()
    cpp_path = base_dir / CPP_FILE
    lib_path = base_dir / f"{LIB_NAME}.{pid}.so"

    # Always rebuild for each worker
    if not lib_path.exists() or cpp_path.stat().st_mtime > lib_path.stat().st_mtime:
        cmd = [
            "g++",
            "-O3",
            "-std=c++17",
            "-fPIC",
            "-shared",
            str(cpp_path),
            "-o",
            str(lib_path),
        ]

        print(f"[PID {pid}] Compiling:", " ".join(cmd))
        subprocess.check_call(cmd, cwd=str(base_dir))

    return str(lib_path)

def load_tsvcpp():
    """Load shared library and set ctypes signatures."""
    libpath = build_tsvcpp_lib()
    lib = ctypes.CDLL(libpath)
    return lib


def get_cpp_function(lib, dace_func):
    """
    Map `dace_s317` → `s317_run_timed`.
    """
    name = dace_func.name.split("dace_", 1)[1]
    short = name     # e.g., "s317"
    cpp_name = f"{short}_run_timed"

    try:
        return getattr(lib, cpp_name), cpp_name
    except AttributeError:
        raise RuntimeError(f"C++ function `{cpp_name}` not found in library.")


def generate_arrays_from_sdfg(sdfg, symbol_map):
    """
    From an SDFG, generate NON-TRANSIENT arrays.
    
    Returns:
        arrays_dace  -- Python dict for calling DaCe SDFG
        arrays_cpp   -- Python dict for C++ call (copies)
        params       -- updated symbol map
        time_ns      -- np.int64 array of length 1
        args_cpp     -- ordered pointer list for C++ call
    """

    arrays_dace = {}
    arrays_cpp = {}

    # ---- 1. Collect non-transient arrays ----
    for name, desc in sdfg.arrays.items():
        if desc.transient:
            continue  # Skip transients entirely

        # ---- 2. Resolve shapes (symbolic expressions allowed) ----
        shape_list = list()
        for expr in desc.shape:
            cexpr = copy.deepcopy(expr)
            if hasattr(cexpr, "subs"):
                for k,v in symbol_value_map.items():
                    cexpr = cexpr.subs(k, v)
            shape_list.append(cexpr)
        resolved_shape = tuple(shape_list)
        print(desc.shape, "->", resolved_shape)
        # ---- 3. Allocate NumPy array in correct dtype ----
        dtype = desc.dtype.as_numpy_dtype()

        arr = np.random.rand(*resolved_shape).astype(dtype)
        arrays_dace[name] = np.copy(arr)
        arrays_cpp[name] = np.copy(arr)

    # ---- 4. Prepare parameter values ----
    # Convert symbol_map to lower-case for consistent ordering
    params = {str(k).lower(): int(v) for k, v in symbol_map.items() if str(k) in sdfg.symbols}

    # ---- 5. Build C++ argument list ----
    args_cpp = []

    # ARRAY ARGS (alphabetical)
    for name in sorted(arrays_cpp.keys()):
        arr = arrays_cpp[name]
        if arr.dtype == np.float64:
            ptr = arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        elif arr.dtype == np.int64:
            ptr = arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int64))
        elif arr.dtype == np.float32:
            ptr = arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        else:
            raise TypeError(f"Unsupported dtype for C++ array: {arr.dtype}")

        args_cpp.append(ptr)

    # SCALAR SYMBOL ARGS (alphabetical)
    for key in sorted(params.keys()):
        args_cpp.append(ctypes.c_int(int(params[key])))

    # ---- 6. Add time_ns buffer (last argument) ----
    time_ns = np.zeros(1, dtype=np.int64)
    args_cpp.append(time_ns.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)))

    return arrays_dace, arrays_cpp, params, time_ns, args_cpp


def run_sdfg_and_cpp(dace_func, symbol_map, lib):
    """
    Execute both the C++ reference kernel and the DaCe SDFG version
    for a given DaceProgram.

    Parameters
    ----------
    dace_func : DaceProgram
    symbol_map : dict
        Mapping of symbols to integers.
    lib : ctypes.CDLL

    Returns
    -------
    dict with fields:
        {
            'cpp_runtime_ns': int,
            'dace_runtime_ns': int,
            'correct': bool,
            'max_diffs': { array_name: float }
        }
    """

    # -------- 1. Extract SDFG --------
    sdfg = dace_func.to_sdfg()
    sdfg.apply_transformations_repeated(LoopToMap)
    sdfg.instrument = dace.dtypes.InstrumentationType.Timer

    # -------- 2. Auto-generate arrays + C++ args --------
    arrays_dace, arrays_cpp, params, time_ns, args_cpp = \
        generate_arrays_from_sdfg(sdfg, symbol_map)

    # -------- 3. Find matching C++ function name --------
    cpp_func, cpp_name = get_cpp_function(lib, dace_func)
    cpp_func.restype = None

    # -------- 4. Run C++ reference kernel --------
    print("Run C++")
    cpp_func(*args_cpp)
    cpp_runtime_ns = int(time_ns[0])

    # -------- 5. Run DaCe SDFG (measure time) --------
    print("Run SDFG")
    sdfg(**arrays_dace, **params, **{str(k): v for k, v in symbol_value_map.items()})
    report = sdfg.get_latest_report()
    total_time = report.events[0].duration  # useconds
    dace_runtime_ns = total_time * 1000

    # -------- 6. Compare correctness --------
    max_diffs = {}
    correct = True

    for arr_name in arrays_cpp.keys():
        diff = np.max(np.abs(arrays_cpp[arr_name] - arrays_dace[arr_name]))
        max_diffs[arr_name] = float(diff)
        if diff > 1e-12:
            correct = False

    # -------- 7. Return structured results --------
    return {
        "cpp_runtime_ns": cpp_runtime_ns,
        "dace_runtime_ns": dace_runtime_ns,
        "correct": correct,
        "max_diffs": max_diffs,
    }


def list_nontransient_arrays(dace_programs):
    """
    For each DaceProgram in `dace_programs`, materialize its SDFG and
    print all arrays where transient == False.
    """
    for dp in dace_programs:
        print(f"\n=== {dp.name} ===")

        # Materialize SDFG
        sdfg = dp.to_sdfg()

        # Collect all non-transient arrays
        nontransient = {
            name: desc
            for name, desc in sdfg.arrays.items()
            if not desc.transient
        }

        if not nontransient:
            print("  No non-transient arrays.")
            continue

        print("  Non-transient arrays:")
        for name, desc in nontransient.items():
            shape = desc.shape if hasattr(desc, "shape") else None
            dtype = getattr(desc, "dtype", None)
            print(f"    - {name}  (shape={shape}, dtype={dtype})")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Collect dace_* functions from a Python file.")
    parser.add_argument("file", help="Path to the Python file to inspect.")

    args = parser.parse_args()
    lib = load_tsvcpp()

    funcs = load_dace_functions_from_file(args.file)
    print(f"Found {len(funcs)} dace_* functions:")
    i = 0
    for f in funcs:
        print(" -", f.name)
        result = run_sdfg_and_cpp(f, symbol_value_map, lib)
        print(result)
        _, cpp_name = get_cpp_function(lib, f)
        write_runtimes("runtimes.csv", cpp_name, result)

        i += 1