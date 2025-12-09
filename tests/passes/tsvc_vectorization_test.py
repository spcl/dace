# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
import math
import os
from typing import Tuple
import dace
import copy
import pytest
import numpy as np
from dace import InterstateEdge
from dace import Union
from dace.memlet import Memlet
from dace.properties import CodeBlock
from dace.sdfg import ControlFlowRegion
from dace.sdfg.graph import Edge
from dace.sdfg.state import ConditionalBlock
from dace.transformation.interstate import LoopToMap, branch_elimination
from dace.transformation.passes import eliminate_branches
from dace.transformation.passes.vectorization.vectorize_cpu import VectorizeCPU
from dace.transformation.passes.vectorization.vectorize_gpu import VectorizeGPU
import ctypes
import subprocess
import pathlib
from math import sin, cos, log, exp, pow

LEN_1D = dace.symbol("LEN_1D")
LEN_2D = dace.symbol("LEN_2D")
ITERATIONS = dace.symbol("ITERATIONS")
S = dace.symbol('S')

LIB_NAME = "libtsvcpp.so"
CPP_FILE = "tsvcpp.cpp"

SAVE_SDFGS = False


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
    short = name  # e.g., "s317"
    cpp_name = f"{short}_run_timed"

    try:
        return getattr(lib, cpp_name), cpp_name
    except AttributeError:
        raise RuntimeError(f"C++ function `{cpp_name}` not found in library.")


def prepare_arguments(arrays, params):
    """
    Convert Python arrays + params into (args_cpp, args_dace)
    where:
      - args_dace is passed directly to DaCe
      - args_cpp is a list of ctypes ptrs + ints
    """

    # ---- Lowercase & sort keys ----
    arrays = {k.lower(): v for k, v in arrays.items()}
    params = {k.lower(): v for k, v in params.items()}

    arrays_cpp = {k: np.copy(v) for k, v in arrays.items()}
    arrays_dace = {k: np.copy(v) for k, v in arrays.items()}

    # ---- Prepare ordered C++ argument list ----
    args_cpp = []

    # Sort arrays alphabetically
    for name in sorted(arrays_cpp.keys()):
        arr = arrays_cpp[name]
        ptr = arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        args_cpp.append(ptr)
    print("ARRAY ARGS:", sorted(arrays_cpp.keys()))

    # Add params in sorted order
    for key in sorted(params.keys()):
        val = params[key]
        args_cpp.append(ctypes.c_int(val))
    print("SCALAR ARGS:", sorted(params.keys()))

    # Time buffer (last argument)
    time_ns = np.zeros(1, dtype=np.int64)
    args_cpp.append(time_ns.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)))

    print("CPP ARG ORDER:", args_cpp)

    # Not reached, but kept for compatibility
    return arrays_dace, arrays_cpp, args_cpp, time_ns


def compare_kernel(dace_func, arrays, params):
    """
    Runs DaCe and C++ reference for a TSVC kernel and compares *all* arrays in-place.
    Raises an AssertionError on mismatch.
    Returns:
        time_cpp (int)
    """
    global_lib = load_tsvcpp()

    lib = global_lib
    cpp_func, cpp_name = get_cpp_function(lib, dace_func)

    arrays_dace, arrays_cpp, args_cpp, time_ns = prepare_arguments(arrays, params)

    # ---- Run DaCe version ----
    dace_sdfg: dace.SDFG = dace_func.to_sdfg()
    dace_sdfg.apply_transformations_repeated(LoopToMap)
    dace_sdfg(**arrays_dace, **params)

    # ---- Configure C++ function signature dynamically ----
    cpp_func.restype = None
    cpp_func.argtypes = [arg.__class__ for arg in args_cpp]

    # ---- Run C++ version ----
    cpp_func(*args_cpp)

    # ---- Compare all arrays ----
    for name in arrays:
        if not np.allclose(arrays_dace[name], arrays_cpp[name], rtol=1e-12, atol=1e-12):
            diff = np.abs(arrays_dace[name] - arrays_cpp[name])
            max_err = np.max(diff)
            raise AssertionError(f"Kernel {dace_func.name}: mismatch in array '{name}'. "
                                 f"Max error = {max_err}")

    return int(time_ns[0])


def run_vectorization_test(dace_func: Union[dace.SDFG, callable],
                           arrays,
                           params,
                           vector_width=8,
                           simplify=True,
                           skip_simplify=None,
                           save_sdfgs=False,
                           sdfg_name=None,
                           fuse_overlapping_loads=False,
                           insert_copies=True,
                           filter_map=-1,
                           cleanup=False,
                           from_sdfg=False,
                           no_inline=False,
                           exact=None,
                           apply_loop_to_map=False):

    # Create copies for comparison
    arrays_orig = {k: copy.deepcopy(v) for k, v in arrays.items()}
    arrays_vec = {k: copy.deepcopy(v) for k, v in arrays.items()}

    # Original SDFG
    if not from_sdfg:
        sdfg: dace.SDFG = dace_func.to_sdfg(simplify=False)
        sdfg.name = sdfg_name
        if simplify:
            sdfg.simplify(validate=True, validate_all=True, skip=skip_simplify or set())
    else:
        sdfg: dace.SDFG = dace_func

    if apply_loop_to_map:
        sdfg.apply_transformations_repeated(LoopToMap())
        sdfg.simplify()

    if save_sdfgs and sdfg_name:
        sdfg.save(f"{sdfg_name}.sdfg")
    c_sdfg = sdfg.compile()

    # Vectorized SDFG
    copy_sdfg: dace.SDFG = copy.deepcopy(sdfg)
    copy_sdfg.name = copy_sdfg.name + "_vectorized"

    if cleanup:
        for e, g in copy_sdfg.all_edges_recursive():
            if isinstance(g, dace.SDFGState):
                if (isinstance(e.src, dace.nodes.AccessNode) and isinstance(e.dst, dace.nodes.AccessNode)
                        and isinstance(g.sdfg.arrays[e.dst.data], dace.data.Scalar)
                        and e.data.other_subset is not None):
                    # Add assignment taskelt
                    src_data = e.src.data
                    src_subset = e.data.subset if e.data.data == src_data else e.data.other_subset
                    dst_data = e.dst.data
                    dst_subset = e.data.subset if e.data.data == dst_data else e.data.other_subset
                    g.remove_edge(e)
                    t = g.add_tasklet(name=f"assign_dst_{dst_data}_from_{src_data}",
                                      code="_out = _in",
                                      inputs={"_in"},
                                      outputs={"_out"})
                    g.add_edge(e.src, e.src_conn, t, "_in",
                               dace.memlet.Memlet(data=src_data, subset=copy.deepcopy(src_subset)))
                    g.add_edge(t, "_out", e.dst, e.dst_conn,
                               dace.memlet.Memlet(data=dst_data, subset=copy.deepcopy(dst_subset)))
        copy_sdfg.validate()

    if filter_map != -1:
        map_labels = [n.map.label for (n, g) in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.MapEntry)]
        filter_map_labels = map_labels[0:filter_map]
        filter_map = filter_map_labels
    else:
        filter_map = None

    #VectorizeCPU(vector_width=vector_width,
    #             fuse_overlapping_loads=fuse_overlapping_loads,
    #             insert_copies=insert_copies,
    #             apply_on_maps=filter_map,
    #             no_inline=no_inline,
    #             fail_on_unvectorizable=True).apply_pass(copy_sdfg, {})
    copy_sdfg.validate()

    if save_sdfgs and sdfg_name:
        copy_sdfg.save(f"{sdfg_name}_vectorized.sdfg")
    c_copy_sdfg = copy_sdfg.compile()

    # Run both
    c_sdfg(**arrays_orig, **params)

    c_copy_sdfg(**arrays_vec, **params)

    # Compare results
    for name in arrays.keys():
        diff = arrays_vec[name] - arrays_orig[name]
        print(diff)
        assert np.allclose(arrays_orig[name], arrays_vec[name], rtol=1e-32), \
            f"{name} Diff: {arrays_orig[name] - arrays_vec[name]}"
        if exact is not None:
            diff = arrays_vec[name] - exact
            assert np.allclose(arrays_vec[name], exact, rtol=0, atol=1e-300), \
                f"{name} Diff: max abs diff = {np.max(np.abs(diff))}"
    return copy_sdfg


def initialise_arrays():
    # Create array handles equivalent to the globals in C
    # Adjust shapes to match your actual code.
    a = np.zeros(LEN_1D, dtype=np.float64)
    b = np.zeros(LEN_1D, dtype=np.float64)
    c = np.zeros(LEN_1D, dtype=np.float64)
    d = np.zeros(LEN_1D, dtype=np.float64)
    e = np.zeros(LEN_1D, dtype=np.float64)
    aa = np.zeros(LEN_1D, dtype=np.float64)
    bb = np.zeros(LEN_1D, dtype=np.float64)
    cc = np.zeros(LEN_1D, dtype=np.float64)
    return a, b, c, d, e, aa, bb, cc


def _run_template(func, arrays, params, sdfg_name: str):
    sdfg: dace.SDFG = func.to_sdfg()
    eliminate_branches.EliminateBranches().apply_pass(sdfg, {})
    run_vectorization_test(
        dace_func=func,
        arrays=arrays,
        params=params,
        save_sdfgs=SAVE_SDFGS,
        sdfg_name=sdfg_name,
        apply_loop_to_map=True,
    )


@dace.program
def dace_s000(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D]):
    for nl in range(2 * ITERATIONS):
        for i in range(LEN_1D):
            a[i] = b[i] + 1.0


@dace.program
def dace_s111(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D]):
    for nl in range(2 * ITERATIONS):
        for i in range(1, LEN_1D, 2):
            a[i] = a[i - 1] + b[i]


@dace.program
def dace_s1111(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D], c: dace.float64[LEN_1D], d: dace.float64[LEN_1D]):
    for nl in range(2 * ITERATIONS):
        for i in range(LEN_1D // 2):
            a[2 * i] = (c[i] * b[i] + d[i] * b[i] + c[i] * c[i] + d[i] * b[i] + d[i] * c[i])


@dace.program
def dace_s112(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D]):
    for nl in range(3 * ITERATIONS):
        for i in range(LEN_1D - 2, -1, -1):
            a[i + 1] = a[i] + b[i]


@dace.program
def dace_s1112(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D]):
    for nl in range(3 * ITERATIONS):
        for i in range(LEN_1D - 1, -1, -1):
            a[i] = b[i] + 1.0


# s113: a(i)=a(1) but no actual dependence cycle
@dace.program
def dace_s113(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D]):
    for nl in range(4 * ITERATIONS):
        for i in range(1, LEN_1D):
            a[i] = a[0] + b[i]


# s1113: one iteration dependency on a(LEN_1D/2) but still vectorizable
@dace.program
def dace_s1113(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D]):
    for nl in range(2 * ITERATIONS):
        for i in range(LEN_1D):
            a[i] = a[LEN_1D // 2] + b[i]


# s114: transpose vectorization - Jump in data access
@dace.program
def dace_s114(aa: dace.float64[LEN_2D, LEN_2D], bb: dace.float64[LEN_2D, LEN_2D]):
    for nl in range(200 * (ITERATIONS // LEN_2D)):
        for i in range(LEN_2D):
            for j in range(i):
                aa[i, j] = aa[j, i] + bb[i, j]


# s115: triangular saxpy loop
@dace.program
def dace_s115(a: dace.float64[LEN_2D], aa: dace.float64[LEN_2D, LEN_2D]):
    for nl in range(1000 * (ITERATIONS // LEN_2D)):
        for j in range(LEN_2D):
            for i in range(j + 1, LEN_2D):
                a[i] = a[i] - aa[j, i] * a[j]


# s1115: triangular saxpy loop variant
@dace.program
def dace_s1115(aa: dace.float64[LEN_2D, LEN_2D], bb: dace.float64[LEN_2D, LEN_2D], cc: dace.float64[LEN_2D, LEN_2D]):
    for nl in range(100 * (ITERATIONS // LEN_2D)):
        for i in range(LEN_2D):
            for j in range(LEN_2D):
                aa[i, j] = aa[i, j] * cc[j, i] + bb[i, j]


@dace.program
def dace_s116(a: dace.float64[LEN_1D]):
    for nl in range(ITERATIONS * 10):
        for i in range(0, LEN_1D - 5, 5):
            a[i] = a[i + 1] * a[i]
            a[i + 1] = a[i + 2] * a[i + 1]
            a[i + 2] = a[i + 3] * a[i + 2]
            a[i + 3] = a[i + 4] * a[i + 3]
            a[i + 4] = a[i + 5] * a[i + 4]


@dace.program
def dace_s118(a: dace.float64[LEN_2D], bb: dace.float64[LEN_2D, LEN_2D]):
    for nl in range(200 * (ITERATIONS // LEN_2D)):
        for i in range(1, LEN_2D):
            for j in range(0, i):
                a[i] = a[i] + bb[j, i] * a[i - j - 1]


@dace.program
def dace_s119(aa: dace.float64[LEN_2D, LEN_2D], bb: dace.float64[LEN_2D, LEN_2D]):
    for nl in range(200 * (ITERATIONS // LEN_2D)):
        for i in range(1, LEN_2D):
            for j in range(1, LEN_2D):
                aa[i, j] = aa[i - 1, j - 1] + bb[i, j]


@dace.program
def dace_s121(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D]):
    for nl in range(3 * ITERATIONS):
        for i in range(LEN_1D - 1):
            j = i + 1
            a[i] = a[j] + b[i]


@dace.program
def dace_s122(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D], n1: dace.int64, n3: dace.int64):
    for nl in range(ITERATIONS):
        j = 1
        k = 0
        for i in range(n1 - 1, LEN_1D, n3):
            k = k + j
            a[i] = a[i] + b[LEN_1D - k]


@dace.program
def dace_s123(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D], c: dace.float64[LEN_1D], d: dace.float64[LEN_1D],
              e: dace.float64[LEN_1D]):
    for nl in range(ITERATIONS):
        j = -1
        for i in range(LEN_1D // 2):
            j = j + 1
            a[j] = b[i] + d[i] * e[i]
            if c[i] > 0.0:
                j = j + 1
                a[j] = c[i] + d[i] * e[i]


@dace.program
def dace_s124(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D], c: dace.float64[LEN_1D], d: dace.float64[LEN_1D],
              e: dace.float64[LEN_1D]):
    for nl in range(ITERATIONS):
        j = -1
        for i in range(LEN_1D):
            if b[i] > 0.0:
                j = j + 1
                a[j] = b[i] + d[i] * e[i]
            else:
                j = j + 1
                a[j] = c[i] + d[i] * e[i]


@dace.program
def dace_s125(flat_2d_array: dace.float64[LEN_2D * LEN_2D], aa: dace.float64[LEN_2D, LEN_2D],
              bb: dace.float64[LEN_2D, LEN_2D], cc: dace.float64[LEN_2D, LEN_2D]):
    for nl in range(100 * (ITERATIONS // LEN_2D)):
        k = -1
        for i in range(LEN_2D):
            for j in range(LEN_2D):
                k = k + 1
                flat_2d_array[k] = aa[i, j] + bb[i, j] * cc[i, j]


@dace.program
def dace_s126(bb: dace.float64[LEN_2D, LEN_2D], flat_2d_array: dace.float64[LEN_2D * LEN_2D], cc: dace.float64[LEN_2D,
                                                                                                               LEN_2D]):
    for nl in range(10 * (ITERATIONS // LEN_2D)):
        k = 1
        for i in range(LEN_2D):
            for j in range(1, LEN_2D):
                bb[j, i] = bb[j - 1, i] + flat_2d_array[k - 1] * cc[j, i]
                k = k + 1
            k = k + 1


@dace.program
def dace_s127(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D], c: dace.float64[LEN_1D], d: dace.float64[LEN_1D],
              e: dace.float64[LEN_1D]):
    for nl in range(2 * ITERATIONS):
        j = -1
        for i in range(LEN_1D // 2):
            j = j + 1
            a[j] = b[i] + c[i] * d[i]
            j = j + 1
            a[j] = b[i] + d[i] * e[i]


@dace.program
def dace_s128(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D], c: dace.float64[LEN_1D], d: dace.float64[LEN_1D]):
    for nl in range(2 * ITERATIONS):
        j = -1
        for i in range(LEN_1D // 2):
            k = j + 1
            a[i] = b[k] - d[i]
            j = k + 1
            b[k] = a[i] + c[k]


@dace.program
def dace_s131(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D]):
    m = 1
    for nl in range(5 * ITERATIONS):
        for i in range(LEN_1D - 1):
            a[i] = a[i + m] + b[i]


@dace.program
def dace_s132(aa: dace.float64[LEN_2D, LEN_2D], b: dace.float64[LEN_2D], c: dace.float64[LEN_2D]):
    j = 0
    k = 1
    for nl in range(400 * ITERATIONS):
        for i in range(1, LEN_2D):
            aa[j, i] = aa[k, i - 1] + b[i] * c[1]


@dace.program
def dace_s151(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D]):
    for nl in range(5 * ITERATIONS):
        for i in range(LEN_1D - 1):
            a[i] = a[i + 1] + b[i]


@dace.program
def dace_s152(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D], c: dace.float64[LEN_1D], d: dace.float64[LEN_1D],
              e: dace.float64[LEN_1D]):
    for nl in range(ITERATIONS):
        for i in range(LEN_1D):
            b[i] = d[i] * e[i]
            a[i] = a[i] + b[i] * c[i]


@dace.program
def dace_s161(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D], c: dace.float64[LEN_1D], d: dace.float64[LEN_1D],
              e: dace.float64[LEN_1D]):
    for nl in range(ITERATIONS // 2):
        for i in range(LEN_1D - 1):
            if b[i] < 0.0:
                c[i + 1] = a[i] + d[i] * d[i]
            else:
                a[i] = c[i] + d[i] * e[i]


@dace.program
def dace_s1161(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D], c: dace.float64[LEN_1D], d: dace.float64[LEN_1D],
               e: dace.float64[LEN_1D]):
    for nl in range(ITERATIONS):
        for i in range(LEN_1D - 1):
            if c[i] < 0.0:
                b[i] = a[i] + d[i] * d[i]
            else:
                a[i] = c[i] + d[i] * e[i]


@dace.program
def dace_s162(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D], c: dace.float64[LEN_1D], k: dace.int64):
    for nl in range(ITERATIONS):
        if k > 0:
            for i in range(0, LEN_1D - k):
                a[i] = a[i + k] + b[i] * c[i]


@dace.program
def dace_s171(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D], inc: dace.int64):
    for nl in range(ITERATIONS):
        for i in range(LEN_1D):
            a[i * inc] = a[i * inc] + b[i]


@dace.program
def dace_s172(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D], n1: dace.int64, n3: dace.int64):
    for nl in range(ITERATIONS):
        for i in range(n1 - 1, LEN_1D, n3):
            a[i] = a[i] + b[i]


@dace.program
def dace_s173(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D]):
    k = LEN_1D // 2
    for nl in range(10 * ITERATIONS):
        for i in range(LEN_1D // 2):
            a[i + k] = a[i] + b[i]


@dace.program
def dace_s174(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D], M: dace.int64):
    for nl in range(10 * ITERATIONS):
        for i in range(M):
            a[i + M] = a[i] + b[i]


@dace.program
def dace_s175(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D], inc: dace.int64):
    for nl in range(ITERATIONS):
        for i in range(0, LEN_1D - inc, inc):
            a[i] = a[i + inc] + b[i]


# s176  (convolution)


@dace.program
def dace_s176(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D], c: dace.float64[LEN_1D]):

    m = LEN_1D // 2
    outer = 4 * (ITERATIONS // LEN_1D)

    for nl in range(outer):
        for j in range(LEN_1D // 2):
            for i in range(m):
                a[i] = a[i] + b[i + m - j - 1] * c[j]


# s211


@dace.program
def dace_s211(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D], c: dace.float64[LEN_1D], d: dace.float64[LEN_1D],
              e: dace.float64[LEN_1D]):

    for nl in range(ITERATIONS):
        for i in range(1, LEN_1D - 1):
            a[i] = b[i - 1] + c[i] * d[i]
            b[i] = b[i + 1] - e[i] * d[i]


# s212


@dace.program
def dace_s212(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D], c: dace.float64[LEN_1D], d: dace.float64[LEN_1D]):

    for nl in range(ITERATIONS):
        for i in range(LEN_1D - 1):
            a[i] = a[i] * c[i]
            b[i] = b[i] + (a[i + 1] * d[i])


# s1213


@dace.program
def dace_s1213(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D], c: dace.float64[LEN_1D], d: dace.float64[LEN_1D]):

    for nl in range(ITERATIONS):
        for i in range(1, LEN_1D - 1):
            a[i] = b[i - 1] + c[i]
            b[i] = a[i + 1] * d[i]


@dace.program
def dace_s221(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D], c: dace.float64[LEN_1D], d: dace.float64[LEN_1D]):

    outer = ITERATIONS // 2
    for nl in range(outer):
        for i in range(1, LEN_1D):
            a[i] = a[i] + c[i] * d[i]
            b[i] = b[i - 1] + a[i] + d[i]


@dace.program
def dace_s1221(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D]):

    for nl in range(ITERATIONS):
        for i in range(4, LEN_1D):
            b[i] = b[i - 4] + a[i]


@dace.program
def dace_s222(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D], c: dace.float64[LEN_1D], e: dace.float64[LEN_1D]):

    outer = ITERATIONS // 2

    for nl in range(outer):
        for i in range(1, LEN_1D):
            a[i] = a[i] + b[i] * c[i]
            e[i] = e[i - 1] * e[i - 1]
            a[i] = a[i] - b[i] * c[i]


# s231 (loop interchange)


@dace.program
def dace_s231(aa: dace.float64[LEN_2D, LEN_2D], bb: dace.float64[LEN_2D, LEN_2D]):

    outer = 100 * (ITERATIONS // LEN_2D)

    for nl in range(outer):
        for i in range(LEN_2D):
            for j in range(1, LEN_2D):
                aa[j, i] = aa[j - 1, i] + bb[j, i]


# s232  (triangular)


@dace.program
def dace_s232(aa: dace.float64[LEN_2D, LEN_2D], bb: dace.float64[LEN_2D, LEN_2D]):

    outer = 100 * (ITERATIONS // LEN_2D)

    for nl in range(outer):
        for j in range(1, LEN_2D):
            for i in range(1, j + 1):
                aa[j, i] = aa[j, i - 1] * aa[j, i - 1] + bb[j, i]


# s1232


@dace.program
def dace_s1232(aa: dace.float64[LEN_2D, LEN_2D], bb: dace.float64[LEN_2D, LEN_2D], cc: dace.float64[LEN_2D, LEN_2D]):

    outer = 100 * (ITERATIONS // LEN_2D)

    for nl in range(outer):
        for j in range(LEN_2D):
            for i in range(j, LEN_2D):
                aa[i, j] = bb[i, j] + cc[i, j]


# s233


@dace.program
def dace_s233(aa: dace.float64[LEN_2D, LEN_2D], bb: dace.float64[LEN_2D, LEN_2D], cc: dace.float64[LEN_2D, LEN_2D]):

    outer = 100 * (ITERATIONS // LEN_2D)

    for nl in range(outer):
        for i in range(1, LEN_2D):

            for j in range(1, LEN_2D):
                aa[j, i] = aa[j - 1, i] + cc[j, i]

            for j in range(1, LEN_2D):
                bb[j, i] = bb[j, i - 1] + cc[j, i]


# s2233


@dace.program
def dace_s2233(aa: dace.float64[LEN_2D, LEN_2D], bb: dace.float64[LEN_2D, LEN_2D], cc: dace.float64[LEN_2D, LEN_2D]):

    outer = 100 * (ITERATIONS // LEN_2D)

    for nl in range(outer):
        for i in range(1, LEN_2D):

            for j in range(1, LEN_2D):
                aa[j, i] = aa[j - 1, i] + cc[j, i]

            for j in range(1, LEN_2D):
                bb[i, j] = bb[i - 1, j] + cc[i, j]


@dace.program
def dace_s235(a: dace.float64[LEN_2D], b: dace.float64[LEN_2D], c: dace.float64[LEN_2D],
              aa: dace.float64[LEN_2D, LEN_2D], bb: dace.float64[LEN_2D, LEN_2D]):

    outer = 200 * (ITERATIONS // LEN_2D)

    for nl in range(outer):
        for i in range(LEN_2D):
            a[i] = a[i] + b[i] * c[i]
            for j in range(1, LEN_2D):
                aa[j, i] = aa[j - 1, i] + bb[j, i] * a[i]


@dace.program
def dace_s241(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D], c: dace.float64[LEN_1D], d: dace.float64[LEN_1D]):

    for nl in range(2 * ITERATIONS):
        for i in range(LEN_1D - 1):
            a[i] = b[i] * c[i] * d[i]
            b[i] = a[i] * a[i + 1] * d[i]


@dace.program
def dace_s242(
    a: dace.float64[LEN_1D],
    b: dace.float64[LEN_1D],
    c: dace.float64[LEN_1D],
    d: dace.float64[LEN_1D],
):

    outer = ITERATIONS // 5

    for nl in range(outer):
        for i in range(1, LEN_1D):
            a[i] = a[i - 1] + 0.5 + 1.0 + b[i] + c[i] + d[i]


@dace.program
def dace_s243(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D], c: dace.float64[LEN_1D], d: dace.float64[LEN_1D],
              e: dace.float64[LEN_1D]):

    for nl in range(ITERATIONS):
        for i in range(LEN_1D - 1):
            a[i] = b[i] + c[i] * d[i]
            b[i] = a[i] + d[i] * e[i]
            a[i] = b[i] + a[i + 1] * d[i]


@dace.program
def dace_s244(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D], c: dace.float64[LEN_1D], d: dace.float64[LEN_1D]):

    for nl in range(ITERATIONS):
        for i in range(LEN_1D - 1):
            a[i] = b[i] + c[i] * d[i]
            b[i] = c[i] + b[i]
            a[i + 1] = b[i] + a[i + 1] * d[i]


@dace.program
def dace_s1244(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D], c: dace.float64[LEN_1D], d: dace.float64[LEN_1D]):

    for nl in range(ITERATIONS):
        for i in range(LEN_1D - 1):
            a[i] = b[i] + c[i] * c[i] + b[i] * b[i] + c[i]
            d[i] = a[i] + a[i + 1]


@dace.program
def dace_s2244(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D], c: dace.float64[LEN_1D], e: dace.float64[LEN_1D]):

    for nl in range(ITERATIONS):
        for i in range(LEN_1D - 1):
            a[i + 1] = b[i] + e[i]
            a[i] = b[i] + c[i]


@dace.program
def dace_s251(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D], c: dace.float64[LEN_1D], d: dace.float64[LEN_1D]):

    for nl in range(4 * ITERATIONS):
        for i in range(LEN_1D):
            s = b[i] + c[i] * d[i]
            a[i] = s * s


@dace.program
def dace_s1251(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D], c: dace.float64[LEN_1D], d: dace.float64[LEN_1D],
               e: dace.float64[LEN_1D]):

    for nl in range(4 * ITERATIONS):
        for i in range(LEN_1D):
            s = b[i] + c[i]
            b[i] = a[i] + d[i]
            a[i] = s * e[i]


@dace.program
def dace_s2251(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D], c: dace.float64[LEN_1D], d: dace.float64[LEN_1D],
               e: dace.float64[LEN_1D]):

    for nl in range(ITERATIONS):
        s = 0.0
        for i in range(LEN_1D):
            a[i] = s * e[i]
            s = b[i] + c[i]
            b[i] = a[i] + d[i]


@dace.program
def dace_s3251(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D], c: dace.float64[LEN_1D], d: dace.float64[LEN_1D],
               e: dace.float64[LEN_1D]):

    for nl in range(ITERATIONS):
        for i in range(LEN_1D - 1):
            a[i + 1] = b[i] + c[i]
            b[i] = c[i] * e[i]
            d[i] = a[i] * e[i]


@dace.program
def dace_s252(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D], c: dace.float64[LEN_1D]):

    for nl in range(ITERATIONS):
        t = 0.0
        for i in range(LEN_1D):
            s = b[i] * c[i]
            a[i] = s + t
            t = s


@dace.program
def dace_s253(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D], c: dace.float64[LEN_1D], d: dace.float64[LEN_1D]):

    for nl in range(ITERATIONS):
        for i in range(LEN_1D):
            if a[i] > b[i]:
                s = a[i] - b[i] * d[i]
                c[i] = c[i] + s
                a[i] = s


@dace.program
def dace_s254(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D]):

    for nl in range(4 * ITERATIONS):
        x = b[LEN_1D - 1]
        for i in range(LEN_1D):
            a[i] = 0.5 * (b[i] + x)
            x = b[i]


@dace.program
def dace_s235(a: dace.float64[LEN_2D], b: dace.float64[LEN_2D], c: dace.float64[LEN_2D],
              aa: dace.float64[LEN_2D, LEN_2D], bb: dace.float64[LEN_2D, LEN_2D]):
    outer = 200 * (ITERATIONS // LEN_2D)
    for nl in range(outer):
        for i in range(LEN_2D):
            a[i] = a[i] + b[i] * c[i]
            for j in range(1, LEN_2D):
                aa[j, i] = aa[j - 1, i] + bb[j, i] * a[i]


@dace.program
def dace_s241(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D], c: dace.float64[LEN_1D], d: dace.float64[LEN_1D]):
    outer = 2 * ITERATIONS
    for nl in range(outer):
        for i in range(LEN_1D - 1):
            a[i] = b[i] * c[i] * d[i]
            b[i] = a[i] * a[i + 1] * d[i]


@dace.program
def dace_s243(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D], c: dace.float64[LEN_1D], d: dace.float64[LEN_1D],
              e: dace.float64[LEN_1D]):
    for nl in range(ITERATIONS):
        for i in range(LEN_1D - 1):
            a[i] = b[i] + c[i] * d[i]
            b[i] = a[i] + d[i] * e[i]
            a[i] = b[i] + a[i + 1] * d[i]


@dace.program
def dace_s244(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D], c: dace.float64[LEN_1D], d: dace.float64[LEN_1D]):
    for nl in range(ITERATIONS):
        for i in range(LEN_1D - 1):
            a[i] = b[i] + c[i] * d[i]
            b[i] = c[i] + b[i]
            a[i + 1] = b[i] + a[i + 1] * d[i]


@dace.program
def dace_s1244(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D], c: dace.float64[LEN_1D], d: dace.float64[LEN_1D]):
    for nl in range(ITERATIONS):
        for i in range(LEN_1D - 1):
            a[i] = b[i] + c[i] * c[i] + b[i] * b[i] + c[i]
            d[i] = a[i] + a[i + 1]


@dace.program
def dace_s2244(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D], c: dace.float64[LEN_1D], e: dace.float64[LEN_1D]):
    for nl in range(ITERATIONS):
        for i in range(LEN_1D - 1):
            a[i + 1] = b[i] + e[i]
            a[i] = b[i] + c[i]


@dace.program
def dace_s251(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D], c: dace.float64[LEN_1D], d: dace.float64[LEN_1D]):
    for nl in range(4 * ITERATIONS):
        for i in range(LEN_1D):
            s = b[i] + c[i] * d[i]
            a[i] = s * s


@dace.program
def dace_s1251(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D], c: dace.float64[LEN_1D], d: dace.float64[LEN_1D],
               e: dace.float64[LEN_1D]):
    for nl in range(4 * ITERATIONS):
        for i in range(LEN_1D):
            s = b[i] + c[i]
            b[i] = a[i] + d[i]
            a[i] = s * e[i]


@dace.program
def dace_s2251(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D], c: dace.float64[LEN_1D], d: dace.float64[LEN_1D],
               e: dace.float64[LEN_1D]):
    for nl in range(ITERATIONS):
        s = 0.0
        for i in range(LEN_1D):
            a[i] = s * e[i]
            s = b[i] + c[i]
            b[i] = a[i] + d[i]


@dace.program
def dace_s3251(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D], c: dace.float64[LEN_1D], d: dace.float64[LEN_1D],
               e: dace.float64[LEN_1D]):
    for nl in range(ITERATIONS):
        for i in range(LEN_1D - 1):
            a[i + 1] = b[i] + c[i]
            b[i] = c[i] * e[i]
            d[i] = a[i] * e[i]


@dace.program
def dace_s252(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D], c: dace.float64[LEN_1D]):
    for nl in range(ITERATIONS):
        t = 0.0
        for i in range(LEN_1D):
            s = b[i] * c[i]
            a[i] = s + t
            t = s


@dace.program
def dace_s253(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D], c: dace.float64[LEN_1D], d: dace.float64[LEN_1D]):
    for nl in range(ITERATIONS):
        for i in range(LEN_1D):
            if a[i] > b[i]:
                s = a[i] - b[i] * d[i]
                c[i] = c[i] + s
                a[i] = s


@dace.program
def dace_s254(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D]):
    for nl in range(4 * ITERATIONS):
        x = b[LEN_1D - 1]
        for i in range(LEN_1D):
            a[i] = (b[i] + x) * 0.5
            x = b[i]


@dace.program
def dace_s255(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D]):
    for nl in range(ITERATIONS):
        x = b[LEN_1D - 1]
        y = b[LEN_1D - 2]
        for i in range(LEN_1D):
            a[i] = (b[i] + x + y) * 0.333
            y = x
            x = b[i]


@dace.program
def dace_s256(a: dace.float64[LEN_2D], aa: dace.float64[LEN_2D, LEN_2D], bb: dace.float64[LEN_2D, LEN_2D],
              d: dace.float64[LEN_2D]):
    outer = 10 * (ITERATIONS // LEN_2D)
    for nl in range(outer):
        for i in range(LEN_2D):
            for j in range(1, LEN_2D):
                a[j] = 1.0 - a[j - 1]
                aa[j, i] = a[j] + bb[j, i] * d[j]


@dace.program
def dace_s257(a: dace.float64[LEN_2D], aa: dace.float64[LEN_2D, LEN_2D], bb: dace.float64[LEN_2D, LEN_2D]):
    outer = 10 * (ITERATIONS // LEN_2D)
    for nl in range(outer):
        for i in range(1, LEN_2D):
            for j in range(LEN_2D):
                a[i] = aa[j, i] - a[i - 1]
                aa[j, i] = a[i] + bb[j, i]


@dace.program
def dace_s258(a: dace.float64[LEN_2D], b: dace.float64[LEN_2D], c: dace.float64[LEN_2D], d: dace.float64[LEN_2D],
              e: dace.float64[LEN_2D], aa: dace.float64[1, LEN_2D]):
    for nl in range(ITERATIONS):
        s = 0.0
        for i in range(LEN_2D):
            if a[i] > 0.0:
                s = d[i] * d[i]
            b[i] = s * c[i] + d[i]
            e[i] = (s + 1.0) * aa[0, i]


@dace.program
def dace_s261(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D], c: dace.float64[LEN_1D], d: dace.float64[LEN_1D]):
    for nl in range(ITERATIONS):
        for i in range(1, LEN_1D):
            t = a[i] + b[i]
            a[i] = t + c[i - 1]
            c[i] = c[i] * d[i]


@dace.program
def dace_s271(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D], c: dace.float64[LEN_1D]):
    for nl in range(4 * ITERATIONS):
        for i in range(LEN_1D):
            if b[i] > 0.0:
                a[i] = a[i] + b[i] * c[i]


@dace.program
def dace_s272(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D], c: dace.float64[LEN_1D], d: dace.float64[LEN_1D],
              e: dace.float64[LEN_1D], threshold: dace.int64):
    for nl in range(ITERATIONS):
        for i in range(LEN_1D):
            if e[i] >= threshold:
                a[i] = a[i] + c[i] * d[i]
                b[i] = b[i] + c[i] * c[i]


@dace.program
def dace_s273(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D], c: dace.float64[LEN_1D], d: dace.float64[LEN_1D],
              e: dace.float64[LEN_1D]):
    for nl in range(ITERATIONS):
        for i in range(LEN_1D):
            a[i] = a[i] + d[i] * e[i]
            if a[i] < 0.0:
                b[i] = b[i] + d[i] * e[i]
            c[i] = c[i] + a[i] * d[i]


@dace.program
def dace_s274(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D], c: dace.float64[LEN_1D], d: dace.float64[LEN_1D],
              e: dace.float64[LEN_1D]):
    for nl in range(ITERATIONS):
        for i in range(LEN_1D):
            a[i] = c[i] + e[i] * d[i]
            if a[i] > 0.0:
                b[i] = a[i] + b[i]
            else:
                a[i] = d[i] * e[i]


@dace.program
def dace_s275(aa: dace.float64[LEN_2D, LEN_2D], bb: dace.float64[LEN_2D, LEN_2D], cc: dace.float64[LEN_2D, LEN_2D]):
    outer = 10 * (ITERATIONS // LEN_2D)
    for nl in range(outer):
        for i in range(LEN_2D):
            if aa[0, i] > 0.0:
                for j in range(1, LEN_2D):
                    aa[j, i] = aa[j - 1, i] + bb[j, i] * cc[j, i]


# ============================================================
# s281
# ============================================================
@dace.program
def dace_s281(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D], c: dace.float64[LEN_1D]):
    for nl in range(ITERATIONS):
        for i in range(LEN_1D):
            x = a[LEN_1D - i - 1] + b[i] * c[i]
            a[i] = x - 1.0
            b[i] = x


# ============================================================
# s1281
# ============================================================
@dace.program
def dace_s1281(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D], c: dace.float64[LEN_1D], d: dace.float64[LEN_1D],
               e: dace.float64[LEN_1D]):
    for nl in range(4 * ITERATIONS):
        for i in range(LEN_1D):
            x = b[i] * c[i] + a[i] * d[i] + e[i]
            a[i] = x - 1.0
            b[i] = x


# ============================================================
# s291
# ============================================================
@dace.program
def dace_s291(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D]):
    for nl in range(2 * ITERATIONS):
        im1 = LEN_1D - 1
        for i in range(LEN_1D):
            a[i] = (b[i] + b[im1]) * 0.5
            im1 = i


# ============================================================
# s292
# ============================================================
@dace.program
def dace_s292(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D]):
    for nl in range(ITERATIONS):
        im1 = LEN_1D - 1
        im2 = LEN_1D - 2
        for i in range(LEN_1D):
            a[i] = (b[i] + b[im1] + b[im2]) * 0.333
            im2 = im1
            im1 = i


# ============================================================
# s293
# ============================================================
@dace.program
def dace_s293(a: dace.float64[LEN_1D]):
    a0 = a[0]
    for nl in range(4 * ITERATIONS):
        for i in range(LEN_1D):
            a[i] = a0


# ============================================================
# s2101
# ============================================================
@dace.program
def dace_s2101(aa: dace.float64[LEN_2D, LEN_2D], bb: dace.float64[LEN_2D, LEN_2D], cc: dace.float64[LEN_2D, LEN_2D]):
    for nl in range(10 * ITERATIONS):
        for i in range(LEN_2D):
            aa[i, i] = aa[i, i] + bb[i, i] * cc[i, i]


# ============================================================
# s2102
# ============================================================
@dace.program
def dace_s2102(aa: dace.float64[LEN_2D, LEN_2D]):
    for nl in range(100 * (ITERATIONS // LEN_2D)):
        for i in range(LEN_2D):
            for j in range(LEN_2D):
                aa[j, i] = 0.0
            aa[i, i] = 1.0


# ============================================================
# s2111
# ============================================================
@dace.program
def dace_s2111(aa: dace.float64[LEN_2D, LEN_2D]):
    for nl in range(100 * (ITERATIONS // LEN_2D)):
        for j in range(1, LEN_2D):
            for i in range(1, LEN_2D):
                aa[j, i] = (aa[j, i - 1] + aa[j - 1, i]) / 1.9


# ============================================================
# s311
# ============================================================
@dace.program
def dace_s311(a: dace.float64[LEN_1D], sum_out: dace.float64[LEN_1D]):
    for nl in range(10 * ITERATIONS):
        sum_out[0] = 0.0
        for i in range(LEN_1D):
            sum_out[0] = sum_out[0] + a[i]


# ============================================================
# s31111
# ============================================================
@dace.program
def dace_s31111(a: dace.float64[LEN_1D]):
    for nl in range(2000 * ITERATIONS):
        sum_val = 0.0
        for base in range(0, LEN_1D, 4):
            partial = 0.0
            partial = partial + a[base + 0]
            partial = partial + a[base + 1]
            partial = partial + a[base + 2]
            partial = partial + a[base + 3]
            sum_val = partial + partial


@dace.program
def dace_s2275(
    a: dace.float64[LEN_2D],
    b: dace.float64[LEN_2D],
    c: dace.float64[LEN_2D],
    d: dace.float64[LEN_2D],
    aa: dace.float64[LEN_2D, LEN_2D],
    bb: dace.float64[LEN_2D, LEN_2D],
    cc: dace.float64[LEN_2D, LEN_2D],
):
    for nl in range(100 * (ITERATIONS // LEN_2D)):
        for i in range(LEN_2D):
            for j in range(LEN_2D):
                aa[j, i] = aa[j, i] + bb[j, i] * cc[j, i]
            a[i] = b[i] + c[i] * d[i]


@dace.program
def dace_s276(
    a: dace.float64[LEN_1D],
    b: dace.float64[LEN_1D],
    c: dace.float64[LEN_1D],
    d: dace.float64[LEN_1D],
):
    mid = LEN_1D // 2
    for nl in range(4 * ITERATIONS):
        for i in range(LEN_1D):
            if i + 1 < mid:
                a[i] = a[i] + b[i] * c[i]
            else:
                a[i] = a[i] + b[i] * d[i]


@dace.program
def dace_s277(
    a: dace.float64[LEN_1D],
    b: dace.float64[LEN_1D],
    c: dace.float64[LEN_1D],
    d: dace.float64[LEN_1D],
    e: dace.float64[LEN_1D],
):
    for nl in range(ITERATIONS):
        for i in range(LEN_1D - 1):
            if a[i] < 0.0:
                if b[i] < 0.0:
                    a[i] = a[i] + c[i] * d[i]
                b[i + 1] = c[i] + d[i] * e[i]


@dace.program
def dace_s278(
    a: dace.float64[LEN_1D],
    b: dace.float64[LEN_1D],
    c: dace.float64[LEN_1D],
    d: dace.float64[LEN_1D],
    e: dace.float64[LEN_1D],
):
    for nl in range(ITERATIONS):
        for i in range(LEN_1D):
            if a[i] > 0.0:
                c[i] = -c[i] + d[i] * e[i]
            else:
                b[i] = -b[i] + d[i] * e[i]
            a[i] = b[i] + c[i] * d[i]


@dace.program
def dace_s279(
    a: dace.float64[LEN_1D],
    b: dace.float64[LEN_1D],
    c: dace.float64[LEN_1D],
    d: dace.float64[LEN_1D],
    e: dace.float64[LEN_1D],
):
    for nl in range(ITERATIONS // 2):
        for i in range(LEN_1D):
            if a[i] > 0.0:
                c[i] = -c[i] + e[i] * e[i]
            else:
                b[i] = -b[i] + d[i] * d[i]
                if b[i] > a[i]:
                    c[i] = c[i] + d[i] * e[i]
            a[i] = b[i] + c[i] * d[i]


@dace.program
def dace_s1279(
    a: dace.float64[LEN_1D],
    b: dace.float64[LEN_1D],
    c: dace.float64[LEN_1D],
    d: dace.float64[LEN_1D],
    e: dace.float64[LEN_1D],
):
    for nl in range(ITERATIONS):
        for i in range(LEN_1D):
            if a[i] < 0.0:
                if b[i] > a[i]:
                    c[i] = c[i] + d[i] * e[i]


@dace.program
def dace_s2710(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D], c: dace.float64[LEN_1D], d: dace.float64[LEN_1D],
               e: dace.float64[LEN_1D], x: dace.float64[LEN_1D]):
    for nl in range(ITERATIONS // 2):
        for i in range(LEN_1D):
            if a[i] > b[i]:
                a[i] = a[i] + b[i] * d[i]
                if LEN_1D > 10:
                    c[i] = c[i] + d[i] * d[i]
                else:
                    c[i] = d[i] * e[i] + 1.0
            else:
                b[i] = a[i] + e[i] * e[i]
                if x[0] > 0.0:
                    c[i] = a[i] + d[i] * d[i]
                else:
                    c[i] = c[i] + e[i] * e[i]


@dace.program
def dace_s2711(
    a: dace.float64[LEN_1D],
    b: dace.float64[LEN_1D],
    c: dace.float64[LEN_1D],
):
    for nl in range(4 * ITERATIONS):
        for i in range(LEN_1D):
            if b[i] != 0.0:
                a[i] = a[i] + b[i] * c[i]


@dace.program
def dace_s2712(
    a: dace.float64[LEN_1D],
    b: dace.float64[LEN_1D],
    c: dace.float64[LEN_1D],
):
    for nl in range(4 * ITERATIONS):
        for i in range(LEN_1D):
            if a[i] > b[i]:
                a[i] = a[i] + b[i] * c[i]


# ============================================================
# s312: product reduction
# ============================================================
@dace.program
def dace_s312(a: dace.float64[LEN_1D]):
    for nl in range(10 * ITERATIONS):
        prod = 1.0
        for i in range(LEN_1D):
            prod = prod * a[i]


# ============================================================
# s313: dot product
# ============================================================
@dace.program
def dace_s313(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D], dot: dace.float64[LEN_1D]):
    for nl in range(5 * ITERATIONS):
        dot[0] = 0.0
        for i in range(LEN_1D):
            dot[0] = dot[0] + a[i] * b[i]


# ============================================================
# s314: max reduction
# ============================================================
@dace.program
def dace_s314(a: dace.float64[LEN_1D]):
    for nl in range(5 * ITERATIONS):
        x = a[0]
        for i in range(LEN_1D):
            if a[i] > x:
                x = a[i]


# ============================================================
# s315: max+index reduction in 1D
# ============================================================
@dace.program
def dace_s315(a: dace.float64[LEN_1D]):
    # permutation of a inside timed region
    for i in range(LEN_1D):
        a[i] = float((i * 7) % LEN_1D)

    for nl in range(ITERATIONS):
        x = a[0]
        index = 0
        for i in range(LEN_1D):
            if a[i] > x:
                x = a[i]
                index = i
        chksum = x + float(index)
        tmp = chksum  # keep use
        tmp = tmp  # no-op to silence unused


# ============================================================
# s316: min reduction
# ============================================================
@dace.program
def dace_s316(a: dace.float64[LEN_1D]):
    for nl in range(5 * ITERATIONS):
        x = a[0]
        for i in range(1, LEN_1D):
            if a[i] < x:
                x = a[i]


# ============================================================
# s317: scalar product reduction (q = q * 0.99)
# ============================================================
@dace.program
def dace_s317(q: dace.float64[LEN_1D]):
    for nl in range(5 * ITERATIONS):
        q[0] = 1.0
        for i in range(LEN_1D // 2):
            q[0] = q[0] * 0.99


# ============================================================
# s318: isamax-like with increment inc
# ============================================================
@dace.program
def dace_s318(a: dace.float64[LEN_1D], inc: dace.int32):
    for nl in range(ITERATIONS // 2):
        k = 0
        index = 0
        maxv = abs(a[0])
        k = k + inc
        for i in range(1, LEN_1D):
            v = abs(a[k])
            if v > maxv:
                index = i
                maxv = v
            k = k + inc
        chksum = maxv + float(index)
        tmp = chksum
        tmp = tmp


# ============================================================
# s319: coupled reductions
# ============================================================
@dace.program
def dace_s319(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D], c: dace.float64[LEN_1D], d: dace.float64[LEN_1D],
              e: dace.float64[LEN_1D]):
    for nl in range(2 * ITERATIONS):
        sum_val = 0.0
        for i in range(LEN_1D):
            a[i] = c[i] + d[i]
            sum_val = sum_val + a[i]
            b[i] = c[i] + e[i]
            sum_val = sum_val + b[i]


# ============================================================
# s3110: 2D max+index
# ============================================================
@dace.program
def dace_s3110(aa: dace.float64[LEN_2D, LEN_2D]):
    for nl in range(100 * (ITERATIONS // LEN_2D)):
        maxv = aa[0, 0]
        xindex = 0
        yindex = 0
        for i in range(LEN_2D):
            for j in range(LEN_2D):
                if aa[i, j] > maxv:
                    maxv = aa[i, j]
                    xindex = i
                    yindex = j
        chksum = maxv + float(xindex) + float(yindex)
        tmp = chksum
        tmp = tmp


# ============================================================
# s13110: same pattern as s3110 (variant
# ============================================================
@dace.program
def dace_s13110(aa: dace.float64[LEN_2D, LEN_2D]):
    for nl in range(100 * (ITERATIONS // LEN_2D)):
        maxv = aa[0, 0]
        xindex = 0
        yindex = 0
        for i in range(LEN_2D):
            for j in range(LEN_2D):
                if aa[i, j] > maxv:
                    maxv = aa[i, j]
                    xindex = i
                    yindex = j
        chksum = maxv + float(xindex) + float(yindex)
        tmp = chksum
        tmp = tmp


# ============================================================
# s3111: conditional sum reduction
# ============================================================
@dace.program
def dace_s3111(a: dace.float64[LEN_1D]):
    for nl in range(ITERATIONS // 2):
        sum_val = 0.0
        for i in range(LEN_1D):
            if a[i] > 0.0:
                sum_val = sum_val + a[i]


@dace.program
def dace_s3112(
    a: dace.float64[LEN_1D],
    b: dace.float64[LEN_1D],
):
    # running sum stored in b
    for nl in range(ITERATIONS):
        sum = 0.0
        for i in range(LEN_1D):
            sum = sum + a[i]
            b[i] = sum


@dace.program
def dace_s3113(a: dace.float64[LEN_1D], ):
    # maximum of absolute value
    maxv = dace.float64(0)
    for nl in range(ITERATIONS * 4):
        maxv = abs(a[0])
        for i in range(LEN_1D):
            av = abs(a[i])
            if av > maxv:
                maxv = av


# ======================
# %3.2 – Recurrences
# ======================


@dace.program
def dace_s321(
    a: dace.float64[LEN_1D],
    b: dace.float64[LEN_1D],
):
    for nl in range(ITERATIONS):
        for i in range(1, LEN_1D):
            a[i] = a[i] + a[i - 1] * b[i]


@dace.program
def dace_s322(
    a: dace.float64[LEN_1D],
    b: dace.float64[LEN_1D],
    c: dace.float64[LEN_1D],
):
    for nl in range(ITERATIONS // 2):
        for i in range(2, LEN_1D):
            a[i] = a[i] + a[i - 1] * b[i] + a[i - 2] * c[i]


@dace.program
def dace_s323(
    a: dace.float64[LEN_1D],
    b: dace.float64[LEN_1D],
    c: dace.float64[LEN_1D],
    d: dace.float64[LEN_1D],
    e: dace.float64[LEN_1D],
):
    for nl in range(ITERATIONS // 2):
        for i in range(1, LEN_1D):
            a[i] = b[i - 1] + c[i] * d[i]
            b[i] = a[i] + c[i] * e[i]


# ======================
# %3.3 – Search loops
# ======================


@dace.program
def dace_s331(a: dace.float64[LEN_1D], ):
    j = dace.int32(-1)
    for nl in range(ITERATIONS):
        j = -1
        for i in range(LEN_1D):
            if a[i] < 0.0:
                j = i
    # return value would be j+1 in C version


@dace.program
def dace_s332(a: dace.float64[LEN_1D], ):
    index = -2
    value = -1.0
    for nl in range(ITERATIONS):
        index = -2
        value = -1.0
        for i in range(LEN_1D):
            if a[i] > 0.5:
                index = i
                value = a[i]
                break
        # chksum = value + index


# ======================
# %3.4 – Packing
# ======================


@dace.program
def dace_s341(
    a: dace.float64[LEN_1D],
    b: dace.float64[LEN_1D],
):
    # pack positive values from b into a
    for nl in range(ITERATIONS):
        j = -1
        for i in range(LEN_1D):
            if b[i] > 0.0:
                j = j + 1
                a[j] = b[i]


@dace.program
def dace_s342(
    a: dace.float64[LEN_1D],
    b: dace.float64[LEN_1D],
):
    # unpack values from b into positions of positive a
    for nl in range(ITERATIONS):
        j = -1
        for i in range(LEN_1D):
            if a[i] > 0.0:
                j = j + 1
                a[i] = b[j]


@dace.program
def dace_s343(
    aa: dace.float64[LEN_2D, LEN_2D],
    bb: dace.float64[LEN_2D, LEN_2D],
    flat_2d_array: dace.float64[LEN_2D * LEN_2D],
):
    # pack aa(j,i) where bb(j,i) > 0 into flat_2d_array
    for nl in range(10 * (ITERATIONS // LEN_2D)):
        k = -1
        for i in range(LEN_2D):
            for j in range(LEN_2D):
                if bb[j, i] > 0.0:
                    k = k + 1
                    flat_2d_array[k] = aa[j, i]


# ======================
# %3.5 – Loop rerolling
# ======================


@dace.program
def dace_s351(
    a: dace.float64[LEN_1D],
    b: dace.float64[LEN_1D],
    c: dace.float64[LEN_1D],
):
    alpha = c[0]
    for nl in range(8 * ITERATIONS):
        for i in range(0, LEN_1D, 5):
            a[i] = a[i] + alpha * b[i]
            a[i + 1] = a[i + 1] + alpha * b[i + 1]
            a[i + 2] = a[i + 2] + alpha * b[i + 2]
            a[i + 3] = a[i + 3] + alpha * b[i + 3]
            a[i + 4] = a[i + 4] + alpha * b[i + 4]


@dace.program
def dace_s1351(
    a: dace.float64[LEN_1D],
    b: dace.float64[LEN_1D],
    c: dace.float64[LEN_1D],
):
    for nl in range(8 * ITERATIONS):
        for i in range(LEN_1D):
            a[i] = b[i] + c[i]


@dace.program
def dace_s352(
    a: dace.float64[LEN_1D],
    b: dace.float64[LEN_1D],
):
    dot = dace.float64(0)
    for nl in range(8 * ITERATIONS):
        dot = 0.0
        for i in range(0, LEN_1D, 5):
            dot = dot + (a[i] * b[i] + a[i + 1] * b[i + 1] + a[i + 2] * b[i + 2] + a[i + 3] * b[i + 3] +
                         a[i + 4] * b[i + 4])


@dace.program
def dace_s353(
    a: dace.float64[LEN_1D],
    b: dace.float64[LEN_1D],
    c: dace.float64[LEN_1D],
    ip: dace.int32[LEN_1D],
):
    alpha = c[0]
    for nl in range(ITERATIONS):
        for i in range(0, LEN_1D, 5):
            a[i] = a[i] + alpha * b[ip[i]]
            a[i + 1] = a[i + 1] + alpha * b[ip[i + 1]]
            a[i + 2] = a[i + 2] + alpha * b[ip[i + 2]]
            a[i + 3] = a[i + 3] + alpha * b[ip[i + 3]]
            a[i + 4] = a[i + 4] + alpha * b[ip[i + 4]]


# ===============================
# %4.1–4.2 – Storage / aliasing
# ===============================


@dace.program
def dace_s1421(
    b: dace.float64[LEN_1D],
    a: dace.float64[LEN_1D],
):
    # xx = &b[LEN_1D/2]; b[i] = xx[i] + a[i]
    half = LEN_1D // 2
    for nl in range(8 * ITERATIONS):
        for i in range(half):
            b[i] = b[half + i] + a[i]


@dace.program
def dace_s422(
    a: dace.float64[LEN_1D],
    flat_2d_array: dace.float64[LEN_1D * LEN_1D],
):
    # xx = flat_2d_array + 4;
    # xx[i] = flat_2d_array[i+8] + a[i]
    for nl in range(8 * ITERATIONS):
        for i in range(LEN_1D):
            flat_2d_array[4 + i] = flat_2d_array[8 + i] + a[i]


@dace.program
def dace_vpvtv(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D], c: dace.float64[LEN_1D]):
    for nl in range(4 * ITERATIONS):
        for i in range(LEN_1D):
            a[i] = a[i] + b[i] * c[i]


@dace.program
def dace_vpvts(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D]):
    for nl in range(ITERATIONS):
        for i in range(LEN_1D):
            a[i] = a[i] + b[i] * S


@dace.program
def dace_vpvpv(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D], c: dace.float64[LEN_1D]):
    for nl in range(4 * ITERATIONS):
        for i in range(LEN_1D):
            a[i] = a[i] + b[i] + c[i]


@dace.program
def dace_vtvtv(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D], c: dace.float64[LEN_1D]):
    for nl in range(4 * ITERATIONS):
        for i in range(LEN_1D):
            a[i] = a[i] * b[i] * c[i]


@dace.program
def dace_vsumr(a: dace.float64[LEN_1D]):
    s = 0.0
    for nl in range(ITERATIONS * 10):
        s = 0.0
        for i in range(LEN_1D):
            s = s + a[i]


@dace.program
def dace_vdotr(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D], dot_out: dace.float64[LEN_1D]):
    dot_out[0] = 0.0
    for nl in range(ITERATIONS * 10):
        dot_out[0] = 0.0
        for i in range(LEN_1D):
            dot_out[0] = dot_out[0] + a[i] * b[i]


@dace.program
def dace_vbor(a: dace.float64[LEN_2D], b: dace.float64[LEN_2D], c: dace.float64[LEN_2D], d: dace.float64[LEN_2D],
              e: dace.float64[LEN_2D], x: dace.float64[LEN_2D]):
    for nl in range(ITERATIONS * 10):
        for i in range(LEN_2D):
            a1 = a[i]
            b1 = b[i]
            c1 = c[i]
            d1 = d[i]
            e1 = e[i]
            f1 = a[i]

            a1 = (a1 * b1 * c1 + a1 * b1 * d1 + a1 * b1 * e1 + a1 * b1 * f1 + a1 * c1 * d1 + a1 * c1 * e1 +
                  a1 * c1 * f1 + a1 * d1 * e1 + a1 * d1 * f1 + a1 * e1 * f1)

            b1 = (b1 * c1 * d1 + b1 * c1 * e1 + b1 * c1 * f1 + b1 * d1 * e1 + b1 * d1 * f1 + b1 * e1 * f1)

            c1 = (c1 * d1 * e1 + c1 * d1 * f1 + c1 * e1 * f1)

            d1 = d1 * e1 * f1

            x[i] = a1 * b1 * c1 * d1


@dace.program
def dace_s424(a: dace.float64[LEN_1D], xx: dace.float64[LEN_1D], flat: dace.float64[LEN_1D]):
    for nl in range(4 * ITERATIONS):
        for i in range(LEN_1D - 1):
            xx[i + 1] = flat[i] + a[i]


@dace.program
def dace_s431(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D]):
    for nl in range(ITERATIONS * 10):
        for i in range(LEN_1D):
            a[i] = a[i] + b[i]


@dace.program
def dace_s441(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D], c: dace.float64[LEN_1D], d: dace.float64[LEN_1D]):
    for nl in range(ITERATIONS):
        for i in range(LEN_1D):
            if d[i] < 0.0:
                a[i] = a[i] + b[i] * c[i]
            elif d[i] == 0.0:
                a[i] = a[i] + b[i] * b[i]
            else:
                a[i] = a[i] + c[i] * c[i]


@dace.program
def dace_s442(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D], c: dace.float64[LEN_1D], d: dace.float64[LEN_1D],
              e: dace.float64[LEN_1D], indx: dace.int32[LEN_1D]):
    for nl in range(ITERATIONS // 2):
        for i in range(LEN_1D):
            if indx[i] == 1:
                a[i] = a[i] + b[i] * b[i]
            elif indx[i] == 2:
                a[i] = a[i] + c[i] * c[i]
            elif indx[i] == 3:
                a[i] = a[i] + d[i] * d[i]
            elif indx[i] == 4:
                a[i] = a[i] + e[i] * e[i]


@dace.program
def dace_s443(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D], c: dace.float64[LEN_1D], d: dace.float64[LEN_1D]):
    for nl in range(2 * ITERATIONS):
        for i in range(LEN_1D):
            if d[i] <= 0.0:
                a[i] = a[i] + b[i] * c[i]
            else:
                a[i] = a[i] + b[i] * b[i]


@dace.program
def dace_s451(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D], c: dace.float64[LEN_1D]):
    for nl in range(ITERATIONS // 4):
        for i in range(LEN_1D):
            a[i] = sin(b[i]) + cos(c[i])


@dace.program
def dace_s452(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D], c: dace.float64[LEN_1D]):
    for nl in range(4 * ITERATIONS):
        for i in range(LEN_1D):
            a[i] = b[i] + c[i] * (i + 1)


@dace.program
def dace_s453(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D]):
    for nl in range(ITERATIONS * 2):
        s = 0.0
        for i in range(LEN_1D):
            s = s + 2.0
            a[i] = s * b[i]


@dace.program
def dace_s471(x: dace.float64[LEN_1D], b: dace.float64[LEN_1D], c: dace.float64[LEN_1D], d: dace.float64[LEN_1D],
              e: dace.float64[LEN_1D]):
    for nl in range(ITERATIONS // 2):
        for i in range(LEN_1D):
            x[i] = b[i] + d[i] * d[i]
            # s471s() is a no-op in DaCe
            b[i] = c[i] + d[i] * e[i]


@dace.program
def dace_s481(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D], c: dace.float64[LEN_1D], d: dace.float64[LEN_1D]):
    for nl in range(ITERATIONS):
        for i in range(LEN_1D):
            if d[i] < 0.0:
                break
            a[i] = a[i] + b[i] * c[i]


@dace.program
def dace_s482(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D], c: dace.float64[LEN_1D]):
    for nl in range(ITERATIONS):
        for i in range(LEN_1D):
            a[i] = a[i] + b[i] * c[i]
            if c[i] > b[i]:
                break


@dace.program
def dace_s491(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D], c: dace.float64[LEN_1D], d: dace.float64[LEN_1D],
              ip: dace.int32[LEN_1D]):
    for nl in range(ITERATIONS):
        for i in range(LEN_1D):
            a[ip[i]] = b[i] + c[i] * d[i]


@dace.program
def dace_s4112(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D], ip: dace.int32[LEN_1D]):
    for nl in range(ITERATIONS):
        for i in range(LEN_1D):
            a[i] = a[i] + b[ip[i]] * 2.0


@dace.program
def dace_s4113(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D], c: dace.float64[LEN_1D], ip: dace.int32[LEN_1D]):
    for nl in range(ITERATIONS):
        for i in range(LEN_1D):
            a[ip[i]] = b[ip[i]] + c[i]


@dace.program
def dace_s4114(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D], c: dace.float64[LEN_1D], d_: dace.float64[LEN_1D],
               ip: dace.int32[LEN_1D], n1: dace.int32):
    for nl in range(ITERATIONS):
        for i in range(n1 - 1, LEN_1D):
            k = ip[i]
            a[i] = b[i] + c[LEN_1D - k - 1] * d_[i]


@dace.program
def dace_s4115(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D], ip: dace.int32[LEN_1D], sum_out: dace.float64[1]):
    sum_val = 0.0
    for nl in range(ITERATIONS):
        sum_val = 0.0
        for i in range(LEN_1D):
            sum_val = sum_val + a[i] * b[ip[i]]
    sum_out[0] = sum_val


@dace.program
def dace_s4116(a: dace.float64[LEN_1D], aa: dace.float64[LEN_2D, LEN_2D], ip: dace.int32[LEN_2D], j: dace.int32,
               inc: dace.int32, sum_out: dace.float64[1]):
    sum_val = 0.0
    for nl in range(100 * ITERATIONS):
        sum_val = 0.0
        for i in range(LEN_2D - 1):
            off = inc + i
            sum_val = sum_val + a[off] * aa[j - 1, ip[i]]
    sum_out[0] = sum_val


@dace.program
def dace_s4117(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D], c: dace.float64[LEN_1D], d: dace.float64[LEN_1D]):
    for nl in range(ITERATIONS):
        for i in range(LEN_1D):
            a[i] = b[i] + c[i // 2] * d[i]


@dace.program
def dace_s4121(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D], c: dace.float64[LEN_1D]):
    for nl in range(ITERATIONS):
        for i in range(LEN_1D):
            a[i] = a[i] + b[i] * c[i]


@dace.program
def dace_va(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D]):
    for nl in range(ITERATIONS * 10):
        for i in range(LEN_1D):
            a[i] = b[i]


@dace.program
def dace_vag(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D], ip: dace.int32[LEN_1D]):
    for nl in range(2 * ITERATIONS):
        for i in range(LEN_1D):
            a[i] = b[ip[i]]


@dace.program
def dace_vas(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D], ip: dace.int32[LEN_1D]):
    for nl in range(2 * ITERATIONS):
        for i in range(LEN_1D):
            a[ip[i]] = b[i]


@dace.program
def dace_vif(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D]):
    for nl in range(ITERATIONS):
        for i in range(LEN_1D):
            if b[i] > 0.0:
                a[i] = b[i]


@dace.program
def dace_vpv(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D]):
    for nl in range(ITERATIONS * 10):
        for i in range(LEN_1D):
            a[i] = a[i] + b[i]


@dace.program
def dace_vtv(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D]):
    for nl in range(ITERATIONS * 10):
        for i in range(LEN_1D):
            a[i] = a[i] * b[i]


#
#
#
# DaCe Programs end, below are the test runners
#
#
#


def test_s000():
    LEN_1D_val = 64
    ITERATIONS_val = 2

    a = np.random.rand(LEN_1D_val).astype(np.float64)
    b = np.random.rand(LEN_1D_val).astype(np.float64)

    compare_kernel(dace_s000, {"a": a, "b": b}, {"LEN_1D": LEN_1D_val, "ITERATIONS": ITERATIONS_val})

    run_vectorization_test(
        dace_func=dace_s000,
        arrays={
            "a": a,
            "b": b
        },
        params={
            "LEN_1D": LEN_1D_val,
            "ITERATIONS": ITERATIONS_val
        },
        save_sdfgs=SAVE_SDFGS,
        sdfg_name="dace_s000",
        apply_loop_to_map=True,
    )

    return a


def test_s111():
    LEN_1D_val = 64
    ITERATIONS_val = 2

    a = np.random.rand(LEN_1D_val).astype(np.float64)
    b = np.random.rand(LEN_1D_val).astype(np.float64)

    compare_kernel(dace_s111, {"a": a, "b": b}, {"LEN_1D": LEN_1D_val, "ITERATIONS": ITERATIONS_val})

    run_vectorization_test(
        dace_func=dace_s111,
        arrays={
            "a": a,
            "b": b
        },
        params={
            "LEN_1D": LEN_1D_val,
            "ITERATIONS": ITERATIONS_val
        },
        save_sdfgs=SAVE_SDFGS,
        sdfg_name="dace_s111",
        apply_loop_to_map=True,
    )

    return a


def test_s1111():
    LEN_1D_val = 64
    ITERATIONS_val = 2

    a = np.random.rand(LEN_1D_val).astype(np.float64)
    b = np.random.rand(LEN_1D_val).astype(np.float64)
    c = np.random.rand(LEN_1D_val).astype(np.float64)
    d = np.random.rand(LEN_1D_val).astype(np.float64)

    compare_kernel(dace_s1111, {"a": a, "b": b, "c": c, "d": d}, {"LEN_1D": LEN_1D_val, "ITERATIONS": ITERATIONS_val})

    run_vectorization_test(
        dace_func=dace_s1111,
        arrays={
            "a": a,
            "b": b,
            "c": c,
            "d": d
        },
        params={
            "LEN_1D": LEN_1D_val,
            "ITERATIONS": ITERATIONS_val
        },
        save_sdfgs=SAVE_SDFGS,
        sdfg_name="dace_s1111",
        apply_loop_to_map=True,
    )

    return a


def test_s112():
    LEN_1D_val = 64
    ITERATIONS_val = 2

    a = np.random.rand(LEN_1D_val).astype(np.float64)
    b = np.random.rand(LEN_1D_val).astype(np.float64)

    compare_kernel(dace_s112, {"a": a, "b": b}, {"LEN_1D": LEN_1D_val, "ITERATIONS": ITERATIONS_val})

    run_vectorization_test(
        dace_func=dace_s112,
        arrays={
            "a": a,
            "b": b
        },
        params={
            "LEN_1D": LEN_1D_val,
            "ITERATIONS": ITERATIONS_val
        },
        save_sdfgs=SAVE_SDFGS,
        sdfg_name="dace_s112",
        apply_loop_to_map=True,
    )

    return a


def test_s1112():
    LEN_1D_val = 64
    ITERATIONS_val = 2

    a = np.random.rand(LEN_1D_val).astype(np.float64)
    b = np.random.rand(LEN_1D_val).astype(np.float64)

    compare_kernel(dace_s1112, {"a": a, "b": b}, {"LEN_1D": LEN_1D_val, "ITERATIONS": ITERATIONS_val})

    run_vectorization_test(
        dace_func=dace_s1112,
        arrays={
            "a": a,
            "b": b
        },
        params={
            "LEN_1D": LEN_1D_val,
            "ITERATIONS": ITERATIONS_val
        },
        save_sdfgs=SAVE_SDFGS,
        sdfg_name="dace_s1112",
        apply_loop_to_map=True,
    )

    return a


def test_s113():
    LEN_1D_val = 64
    ITERATIONS_val = 2

    # Allocate random inputs
    a = np.random.rand(LEN_1D_val).astype(np.float64)
    b = np.random.rand(LEN_1D_val).astype(np.float64)

    compare_kernel(dace_s113, {"a": a, "b": b}, {"LEN_1D": LEN_1D_val, "ITERATIONS": ITERATIONS_val})

    run_vectorization_test(
        dace_func=dace_s113,
        arrays={
            "a": a,
            "b": b
        },
        params={
            "LEN_1D": LEN_1D_val,
            "ITERATIONS": ITERATIONS_val
        },
        save_sdfgs=SAVE_SDFGS,
        sdfg_name="dace_s113",
        apply_loop_to_map=True,
    )

    return a


def test_s1113():
    LEN_1D_val = 64
    ITERATIONS_val = 2

    # Allocate random inputs
    a = np.random.rand(LEN_1D_val).astype(np.float64)
    b = np.random.rand(LEN_1D_val).astype(np.float64)

    compare_kernel(dace_s1113, {"a": a, "b": b}, {"LEN_1D": LEN_1D_val, "ITERATIONS": ITERATIONS_val})

    run_vectorization_test(
        dace_func=dace_s1113,
        arrays={
            "a": a,
            "b": b
        },
        params={
            "LEN_1D": LEN_1D_val,
            "ITERATIONS": ITERATIONS_val
        },
        save_sdfgs=SAVE_SDFGS,
        sdfg_name="dace_s1113",
        apply_loop_to_map=True,
    )

    return a


def test_s114():
    LEN_2D_val = 32
    ITERATIONS_val = 2

    # Allocate random inputs
    aa = np.random.rand(LEN_2D_val, LEN_2D_val).astype(np.float64)
    bb = np.random.rand(LEN_2D_val, LEN_2D_val).astype(np.float64)

    compare_kernel(dace_s114, {"aa": aa, "bb": bb}, {"LEN_2D": LEN_2D_val, "ITERATIONS": ITERATIONS_val})

    run_vectorization_test(
        dace_func=dace_s114,
        arrays={
            "aa": aa,
            "bb": bb
        },
        params={
            "LEN_2D": LEN_2D_val,
            "ITERATIONS": ITERATIONS_val
        },
        save_sdfgs=SAVE_SDFGS,
        sdfg_name="dace_s114",
        apply_loop_to_map=True,
    )

    return aa


def test_s115():
    LEN_2D_val = 32
    ITERATIONS_val = 2

    # Allocate random inputs
    a = np.random.rand(LEN_2D_val).astype(np.float64)
    aa = np.random.rand(LEN_2D_val, LEN_2D_val).astype(np.float64)

    compare_kernel(dace_s115, {"a": a, "aa": aa}, {"LEN_2D": LEN_2D_val, "ITERATIONS": ITERATIONS_val})

    run_vectorization_test(
        dace_func=dace_s115,
        arrays={
            "a": a,
            "aa": aa
        },
        params={
            "LEN_2D": LEN_2D_val,
            "ITERATIONS": ITERATIONS_val
        },
        save_sdfgs=SAVE_SDFGS,
        sdfg_name="dace_s115",
        apply_loop_to_map=True,
    )

    return a


def test_s1115():
    LEN_2D_val = 32
    ITERATIONS_val = 2

    # Allocate random inputs
    aa = np.random.rand(LEN_2D_val, LEN_2D_val).astype(np.float64)
    bb = np.random.rand(LEN_2D_val, LEN_2D_val).astype(np.float64)
    cc = np.random.rand(LEN_2D_val, LEN_2D_val).astype(np.float64)

    compare_kernel(dace_s1115, {"aa": aa, "bb": bb, "cc": cc}, {"LEN_2D": LEN_2D_val, "ITERATIONS": ITERATIONS_val})

    run_vectorization_test(
        dace_func=dace_s1115,
        arrays={
            "aa": aa,
            "bb": bb,
            "cc": cc
        },
        params={
            "LEN_2D": LEN_2D_val,
            "ITERATIONS": ITERATIONS_val
        },
        save_sdfgs=SAVE_SDFGS,
        sdfg_name="dace_s1115",
        apply_loop_to_map=True,
    )

    return aa


def test_s116():
    LEN_1D_val = 64
    ITERATIONS_val = 2

    # a is both source and destination
    a = np.random.rand(LEN_1D_val).astype(np.float64)

    compare_kernel(dace_s116, {
        "a": a,
    }, {
        "LEN_1D": LEN_1D_val,
        "ITERATIONS": ITERATIONS_val
    })

    run_vectorization_test(
        dace_func=dace_s116,
        arrays={"a": a},
        params={
            "LEN_1D": LEN_1D_val,
            "ITERATIONS": ITERATIONS_val
        },
        save_sdfgs=SAVE_SDFGS,
        sdfg_name="dace_s116",
        apply_loop_to_map=True,
    )

    return a


def test_s118():
    LEN_2D_val = 32
    ITERATIONS_val = 320

    a = np.random.rand(LEN_2D_val).astype(np.float64)
    bb = np.random.rand(LEN_2D_val, LEN_2D_val).astype(np.float64)

    compare_kernel(dace_s118, {"a": a, "bb": bb}, {"LEN_2D": LEN_2D_val, "ITERATIONS": ITERATIONS_val})

    run_vectorization_test(
        dace_func=dace_s118,
        arrays={
            "a": a,
            "bb": bb
        },
        params={
            "LEN_2D": LEN_2D_val,
            "ITERATIONS": ITERATIONS_val
        },
        save_sdfgs=SAVE_SDFGS,
        sdfg_name="dace_s118",
        apply_loop_to_map=True,
    )

    return a


def test_s119():
    LEN_2D_val = 32
    ITERATIONS_val = 320

    aa = np.random.rand(LEN_2D_val, LEN_2D_val).astype(np.float64)
    bb = np.random.rand(LEN_2D_val, LEN_2D_val).astype(np.float64)

    compare_kernel(dace_s119, {"aa": aa, "bb": bb}, {"LEN_2D": LEN_2D_val, "ITERATIONS": ITERATIONS_val})

    run_vectorization_test(
        dace_func=dace_s119,
        arrays={
            "aa": aa,
            "bb": bb
        },
        params={
            "LEN_2D": LEN_2D_val,
            "ITERATIONS": ITERATIONS_val
        },
        save_sdfgs=SAVE_SDFGS,
        sdfg_name="dace_s119",
        apply_loop_to_map=True,
    )

    return aa


def test_s121():
    LEN_1D_val = 64
    ITERATIONS_val = 2

    a = np.random.rand(LEN_1D_val).astype(np.float64)
    b = np.random.rand(LEN_1D_val).astype(np.float64)

    compare_kernel(dace_s121, {"a": a, "b": b}, {"LEN_1D": LEN_1D_val, "ITERATIONS": ITERATIONS_val})

    run_vectorization_test(
        dace_func=dace_s121,
        arrays={
            "a": a,
            "b": b
        },
        params={
            "LEN_1D": LEN_1D_val,
            "ITERATIONS": ITERATIONS_val
        },
        save_sdfgs=SAVE_SDFGS,
        sdfg_name="dace_s121",
        apply_loop_to_map=True,
    )

    return a


def test_s122():
    LEN_1D_val = 64
    ITERATIONS_val = 2
    n1_val = 2
    n3_val = 3

    a = np.random.rand(LEN_1D_val).astype(np.float64)
    b = np.random.rand(LEN_1D_val).astype(np.float64)

    compare_kernel(dace_s122, {
        "a": a,
        "b": b
    }, {
        "LEN_1D": LEN_1D_val,
        "ITERATIONS": ITERATIONS_val,
        "n1": n1_val,
        "n3": n3_val
    })

    run_vectorization_test(
        dace_func=dace_s122,
        arrays={
            "a": a,
            "b": b
        },
        params={
            "LEN_1D": LEN_1D_val,
            "ITERATIONS": ITERATIONS_val,
            "n1": n1_val,
            "n3": n3_val,
        },
        save_sdfgs=SAVE_SDFGS,
        sdfg_name="dace_s122",
        apply_loop_to_map=True,
    )

    return a


def test_s123():
    LEN_1D_val = 64
    ITERATIONS_val = 2

    a = np.random.rand(LEN_1D_val).astype(np.float64)
    b = np.random.rand(LEN_1D_val).astype(np.float64)
    c = np.random.randn(LEN_1D_val).astype(np.float64)
    d = np.random.rand(LEN_1D_val).astype(np.float64)
    e = np.random.rand(LEN_1D_val).astype(np.float64)

    compare_kernel(dace_s123, {
        "a": a,
        "b": b,
        "c": c,
        "d": d,
        "e": e
    }, {
        "LEN_1D": LEN_1D_val,
        "ITERATIONS": ITERATIONS_val
    })

    run_vectorization_test(
        dace_func=dace_s123,
        arrays={
            "a": a,
            "b": b,
            "c": c,
            "d": d,
            "e": e
        },
        params={
            "LEN_1D": LEN_1D_val,
            "ITERATIONS": ITERATIONS_val
        },
        save_sdfgs=SAVE_SDFGS,
        sdfg_name="dace_s123",
        apply_loop_to_map=True,
    )

    return a


def test_s124():
    LEN_1D_val = 64
    ITERATIONS_val = 2

    a = np.random.rand(LEN_1D_val).astype(np.float64)
    b = np.random.randn(LEN_1D_val).astype(np.float64)
    c = np.random.rand(LEN_1D_val).astype(np.float64)
    d = np.random.rand(LEN_1D_val).astype(np.float64)
    e = np.random.rand(LEN_1D_val).astype(np.float64)

    compare_kernel(dace_s124, {
        "a": a,
        "b": b,
        "c": c,
        "d": d,
        "e": e
    }, {
        "LEN_1D": LEN_1D_val,
        "ITERATIONS": ITERATIONS_val
    })

    run_vectorization_test(
        dace_func=dace_s124,
        arrays={
            "a": a,
            "b": b,
            "c": c,
            "d": d,
            "e": e
        },
        params={
            "LEN_1D": LEN_1D_val,
            "ITERATIONS": ITERATIONS_val
        },
        save_sdfgs=SAVE_SDFGS,
        sdfg_name="dace_s124",
        apply_loop_to_map=True,
    )

    return a


def test_s125():
    LEN_2D_val = 32
    ITERATIONS_val = 2

    flat_2d_array = np.random.rand(LEN_2D_val * LEN_2D_val).astype(np.float64)
    aa = np.random.rand(LEN_2D_val, LEN_2D_val).astype(np.float64)
    bb = np.random.rand(LEN_2D_val, LEN_2D_val).astype(np.float64)
    cc = np.random.rand(LEN_2D_val, LEN_2D_val).astype(np.float64)

    compare_kernel(dace_s125, {
        "flat_2d_array": flat_2d_array,
        "aa": aa,
        "bb": bb,
        "cc": cc
    }, {
        "LEN_2D": LEN_2D_val,
        "ITERATIONS": ITERATIONS_val
    })

    run_vectorization_test(
        dace_func=dace_s125,
        arrays={
            "flat_2d_array": flat_2d_array,
            "aa": aa,
            "bb": bb,
            "cc": cc
        },
        params={
            "LEN_2D": LEN_2D_val,
            "ITERATIONS": ITERATIONS_val
        },
        save_sdfgs=SAVE_SDFGS,
        sdfg_name="dace_s125",
        apply_loop_to_map=True,
    )

    return flat_2d_array


def test_s126():
    LEN_2D_val = 32
    ITERATIONS_val = 2

    flat_2d_array = np.random.rand(LEN_2D_val * LEN_2D_val).astype(np.float64)
    bb = np.random.rand(LEN_2D_val, LEN_2D_val).astype(np.float64)
    cc = np.random.rand(LEN_2D_val, LEN_2D_val).astype(np.float64)

    compare_kernel(dace_s126, {
        "flat_2d_array": flat_2d_array,
        "bb": bb,
        "cc": cc
    }, {
        "LEN_2D": LEN_2D_val,
        "ITERATIONS": ITERATIONS_val
    })

    run_vectorization_test(
        dace_func=dace_s126,
        arrays={
            "bb": bb,
            "flat_2d_array": flat_2d_array,
            "cc": cc
        },
        params={
            "LEN_2D": LEN_2D_val,
            "ITERATIONS": ITERATIONS_val
        },
        save_sdfgs=SAVE_SDFGS,
        sdfg_name="dace_s126",
        apply_loop_to_map=True,
    )

    return flat_2d_array


def test_s127():
    LEN_1D_val = 1024
    ITERATIONS_val = 2

    a = np.random.rand(LEN_1D_val).astype(np.float64)
    b = np.random.rand(LEN_1D_val).astype(np.float64)
    c = np.random.rand(LEN_1D_val).astype(np.float64)
    d = np.random.rand(LEN_1D_val).astype(np.float64)
    e = np.random.rand(LEN_1D_val).astype(np.float64)

    compare_kernel(dace_s127, {
        "a": a,
        "b": b,
        "c": c,
        "d": d,
        "e": e
    }, {
        "LEN_1D": LEN_1D_val,
        "ITERATIONS": ITERATIONS_val
    })

    run_vectorization_test(
        dace_func=dace_s127,
        arrays={
            "a": a,
            "b": b,
            "c": c,
            "d": d,
            "e": e
        },
        params={
            "LEN_1D": LEN_1D_val,
            "ITERATIONS": ITERATIONS_val
        },
        save_sdfgs=SAVE_SDFGS,
        sdfg_name="dace_s127",
        apply_loop_to_map=True,
    )

    return a


def test_s128():
    LEN_1D_val = 1024
    ITERATIONS_val = 2

    a = np.random.rand(LEN_1D_val).astype(np.float64)
    b = np.random.rand(LEN_1D_val).astype(np.float64)
    c = np.random.rand(LEN_1D_val).astype(np.float64)
    d = np.random.rand(LEN_1D_val).astype(np.float64)

    compare_kernel(dace_s128, {"a": a, "b": b, "c": c, "d": d}, {"LEN_1D": LEN_1D_val, "ITERATIONS": ITERATIONS_val})

    run_vectorization_test(
        dace_func=dace_s128,
        arrays={
            "a": a,
            "b": b,
            "c": c,
            "d": d
        },
        params={
            "LEN_1D": LEN_1D_val,
            "ITERATIONS": ITERATIONS_val
        },
        save_sdfgs=SAVE_SDFGS,
        sdfg_name="dace_s128",
        apply_loop_to_map=True,
    )

    return a


def test_s131():
    LEN_1D_val = 1024
    ITERATIONS_val = 2

    a = np.random.rand(LEN_1D_val).astype(np.float64)
    b = np.random.rand(LEN_1D_val).astype(np.float64)

    compare_kernel(dace_s131, {"a": a, "b": b}, {"LEN_1D": LEN_1D_val, "ITERATIONS": ITERATIONS_val})

    run_vectorization_test(
        dace_func=dace_s131,
        arrays={
            "a": a,
            "b": b
        },
        params={
            "LEN_1D": LEN_1D_val,
            "ITERATIONS": ITERATIONS_val
        },
        save_sdfgs=SAVE_SDFGS,
        sdfg_name="dace_s131",
        apply_loop_to_map=True,
    )

    return a


def test_s132():
    LEN_2D_val = 32
    ITERATIONS_val = 2

    aa = np.random.rand(LEN_2D_val, LEN_2D_val)
    b = np.random.rand(LEN_2D_val)
    c = np.random.rand(LEN_2D_val)

    compare_kernel(dace_s132, {"aa": aa, "b": b, "c": c}, {"LEN_2D": LEN_2D_val, "ITERATIONS": ITERATIONS_val})

    run_vectorization_test(
        dace_func=dace_s132,
        arrays={
            "aa": aa,
            "b": b,
            "c": c
        },
        params={
            "LEN_2D": LEN_2D_val,
            "ITERATIONS": ITERATIONS_val
        },
        save_sdfgs=SAVE_SDFGS,
        sdfg_name="dace_s132",
        apply_loop_to_map=True,
    )

    return aa


def test_s151():
    LEN_1D_val = 64
    ITERATIONS_val = 2

    a = np.random.rand(LEN_1D_val)
    b = np.random.rand(LEN_1D_val)

    compare_kernel(dace_s151, {"a": a, "b": b}, {"LEN_1D": LEN_1D_val, "ITERATIONS": ITERATIONS_val})

    run_vectorization_test(
        dace_func=dace_s151,
        arrays={
            "a": a,
            "b": b
        },
        params={
            "LEN_1D": LEN_1D_val,
            "ITERATIONS": ITERATIONS_val
        },
        save_sdfgs=SAVE_SDFGS,
        sdfg_name="dace_s151",
        apply_loop_to_map=True,
    )

    return a


def test_s152():
    LEN_1D_val = 64
    ITERATIONS_val = 2

    a = np.random.rand(LEN_1D_val)
    b = np.random.rand(LEN_1D_val)
    c = np.random.rand(LEN_1D_val)
    d = np.random.rand(LEN_1D_val)
    e = np.random.rand(LEN_1D_val)

    compare_kernel(dace_s152, {
        "a": a,
        "b": b,
        "c": c,
        "d": d,
        "e": e
    }, {
        "LEN_1D": LEN_1D_val,
        "ITERATIONS": ITERATIONS_val
    })

    run_vectorization_test(
        dace_func=dace_s152,
        arrays={
            "a": a,
            "b": b,
            "c": c,
            "d": d,
            "e": e
        },
        params={
            "LEN_1D": LEN_1D_val,
            "ITERATIONS": ITERATIONS_val
        },
        save_sdfgs=SAVE_SDFGS,
        sdfg_name="dace_s152",
        apply_loop_to_map=True,
    )

    return a


def test_s161():
    LEN_1D_val = 64
    ITERATIONS_val = 2

    a = np.random.rand(LEN_1D_val)
    b = np.random.randn(LEN_1D_val)
    c = np.random.rand(LEN_1D_val)
    d = np.random.rand(LEN_1D_val)
    e = np.random.rand(LEN_1D_val)

    compare_kernel(dace_s161, {
        "a": a,
        "b": b,
        "c": c,
        "d": d,
        "e": e
    }, {
        "LEN_1D": LEN_1D_val,
        "ITERATIONS": ITERATIONS_val
    })

    run_vectorization_test(
        dace_func=dace_s161,
        arrays={
            "a": a,
            "b": b,
            "c": c,
            "d": d,
            "e": e
        },
        params={
            "LEN_1D": LEN_1D_val,
            "ITERATIONS": ITERATIONS_val
        },
        save_sdfgs=SAVE_SDFGS,
        sdfg_name="dace_s161",
        apply_loop_to_map=True,
    )

    return a


def test_s1161():
    LEN_1D_val = 64
    ITERATIONS_val = 2

    a = np.random.rand(LEN_1D_val)
    b = np.random.rand(LEN_1D_val)
    c = np.random.randn(LEN_1D_val)
    d = np.random.rand(LEN_1D_val)
    e = np.random.rand(LEN_1D_val)

    compare_kernel(dace_s1161, {
        "a": a,
        "b": b,
        "c": c,
        "d": d,
        "e": e
    }, {
        "LEN_1D": LEN_1D_val,
        "ITERATIONS": ITERATIONS_val
    })

    run_vectorization_test(
        dace_func=dace_s1161,
        arrays={
            "a": a,
            "b": b,
            "c": c,
            "d": d,
            "e": e
        },
        params={
            "LEN_1D": LEN_1D_val,
            "ITERATIONS": ITERATIONS_val
        },
        save_sdfgs=SAVE_SDFGS,
        sdfg_name="dace_s1161",
        apply_loop_to_map=True,
    )

    return a


def test_s162():
    LEN_1D_val = 64
    ITERATIONS_val = 2
    k_val = 3

    a = np.random.rand(LEN_1D_val)
    b = np.random.rand(LEN_1D_val)
    c = np.random.rand(LEN_1D_val)

    compare_kernel(dace_s162, {
        "a": a,
        "b": b,
        "c": c
    }, {
        "LEN_1D": LEN_1D_val,
        "ITERATIONS": ITERATIONS_val,
        "k": k_val
    })

    run_vectorization_test(
        dace_func=dace_s162,
        arrays={
            "a": a,
            "b": b,
            "c": c
        },
        params={
            "LEN_1D": LEN_1D_val,
            "ITERATIONS": ITERATIONS_val,
            "k": k_val
        },
        save_sdfgs=SAVE_SDFGS,
        sdfg_name="dace_s162",
        apply_loop_to_map=True,
    )

    return a


def test_s171():
    LEN_1D_val = 64
    ITERATIONS_val = 2
    inc_val = 1

    a = np.random.rand(LEN_1D_val)
    b = np.random.rand(LEN_1D_val)

    compare_kernel(dace_s171, {"a": a, "b": b}, {"LEN_1D": LEN_1D_val, "ITERATIONS": ITERATIONS_val, "inc": inc_val})

    run_vectorization_test(
        dace_func=dace_s171,
        arrays={
            "a": a,
            "b": b
        },
        params={
            "LEN_1D": LEN_1D_val,
            "ITERATIONS": ITERATIONS_val,
            "inc": inc_val
        },
        save_sdfgs=SAVE_SDFGS,
        sdfg_name="dace_s171",
        apply_loop_to_map=True,
    )

    return a


def test_s172():
    LEN_1D_val = 64
    ITERATIONS_val = 2
    n1_val = 1
    n3_val = 1

    a = np.random.rand(LEN_1D_val)
    b = np.random.rand(LEN_1D_val)

    compare_kernel(dace_s172, {
        "a": a,
        "b": b
    }, {
        "LEN_1D": LEN_1D_val,
        "ITERATIONS": ITERATIONS_val,
        "n1": n1_val,
        "n3": n3_val
    })

    run_vectorization_test(
        dace_func=dace_s172,
        arrays={
            "a": a,
            "b": b
        },
        params={
            "LEN_1D": LEN_1D_val,
            "ITERATIONS": ITERATIONS_val,
            "n1": n1_val,
            "n3": n3_val
        },
        save_sdfgs=SAVE_SDFGS,
        sdfg_name="dace_s172",
        apply_loop_to_map=True,
    )

    return a


def test_s173():
    LEN_1D_val = 64
    ITERATIONS_val = 2

    a = np.random.rand(LEN_1D_val)
    b = np.random.rand(LEN_1D_val)

    compare_kernel(dace_s173, {"a": a, "b": b}, {"LEN_1D": LEN_1D_val, "ITERATIONS": ITERATIONS_val})

    run_vectorization_test(
        dace_func=dace_s173,
        arrays={
            "a": a,
            "b": b
        },
        params={
            "LEN_1D": LEN_1D_val,
            "ITERATIONS": ITERATIONS_val
        },
        save_sdfgs=SAVE_SDFGS,
        sdfg_name="dace_s173",
        apply_loop_to_map=True,
    )

    return a


def test_s174():
    LEN_1D_val = 64
    ITERATIONS_val = 2
    M_val = 16

    a = np.random.rand(LEN_1D_val)
    b = np.random.rand(LEN_1D_val)

    compare_kernel(dace_s174, {"a": a, "b": b}, {"LEN_1D": LEN_1D_val, "ITERATIONS": ITERATIONS_val, "M": M_val})

    run_vectorization_test(
        dace_func=dace_s174,
        arrays={
            "a": a,
            "b": b
        },
        params={
            "LEN_1D": LEN_1D_val,
            "ITERATIONS": ITERATIONS_val,
            "M": M_val
        },
        save_sdfgs=SAVE_SDFGS,
        sdfg_name="dace_s174",
        apply_loop_to_map=True,
    )

    return a


def test_s175():
    LEN_1D_val = 64
    ITERATIONS_val = 2
    inc_val = 3

    a = np.random.rand(LEN_1D_val)
    b = np.random.rand(LEN_1D_val)

    compare_kernel(dace_s175, {"a": a, "b": b}, {"LEN_1D": LEN_1D_val, "ITERATIONS": ITERATIONS_val, "inc": inc_val})

    run_vectorization_test(
        dace_func=dace_s175,
        arrays={
            "a": a,
            "b": b
        },
        params={
            "LEN_1D": LEN_1D_val,
            "ITERATIONS": ITERATIONS_val,
            "inc": inc_val
        },
        save_sdfgs=SAVE_SDFGS,
        sdfg_name="dace_s175",
        apply_loop_to_map=True,
    )

    return a


def test_s176():
    LEN_1D_val = 64
    ITERATIONS_val = 2

    a = np.random.rand(LEN_1D_val)
    b = np.random.rand(LEN_1D_val)
    c = np.random.rand(LEN_1D_val)

    compare_kernel(dace_s176, {"a": a, "b": b, "c": c}, {"LEN_1D": LEN_1D_val, "ITERATIONS": ITERATIONS_val})

    run_vectorization_test(
        dace_func=dace_s176,
        arrays={
            "a": a,
            "b": b,
            "c": c
        },
        params={
            "LEN_1D": LEN_1D_val,
            "ITERATIONS": ITERATIONS_val
        },
        save_sdfgs=SAVE_SDFGS,
        sdfg_name="dace_s176",
        apply_loop_to_map=True,
    )

    return a


def test_s211():
    LEN_1D_val = 64
    ITERATIONS_val = 2

    a = np.random.rand(LEN_1D_val)
    b = np.random.rand(LEN_1D_val)
    c = np.random.rand(LEN_1D_val)
    d = np.random.rand(LEN_1D_val)
    e = np.random.rand(LEN_1D_val)

    compare_kernel(dace_s211, {
        "a": a,
        "b": b,
        "c": c,
        "d": d,
        "e": e
    }, {
        "LEN_1D": LEN_1D_val,
        "ITERATIONS": ITERATIONS_val
    })

    run_vectorization_test(
        dace_func=dace_s211,
        arrays={
            "a": a,
            "b": b,
            "c": c,
            "d": d,
            "e": e
        },
        params={
            "LEN_1D": LEN_1D_val,
            "ITERATIONS": ITERATIONS_val
        },
        save_sdfgs=SAVE_SDFGS,
        sdfg_name="dace_s211",
        apply_loop_to_map=True,
    )

    return a, b


def test_s212():
    LEN_1D_val = 64
    ITERATIONS_val = 2

    a = np.random.rand(LEN_1D_val)
    b = np.random.rand(LEN_1D_val)
    c = np.random.rand(LEN_1D_val)
    d = np.random.rand(LEN_1D_val)

    compare_kernel(dace_s212, {"a": a, "b": b, "c": c, "d": d}, {"LEN_1D": LEN_1D_val, "ITERATIONS": ITERATIONS_val})

    run_vectorization_test(
        dace_func=dace_s212,
        arrays={
            "a": a,
            "b": b,
            "c": c,
            "d": d
        },
        params={
            "LEN_1D": LEN_1D_val,
            "ITERATIONS": ITERATIONS_val
        },
        save_sdfgs=SAVE_SDFGS,
        sdfg_name="dace_s212",
        apply_loop_to_map=True,
    )
    return a, b


def test_s1213():
    LEN_1D_val = 64
    ITERATIONS_val = 2

    a = np.random.rand(LEN_1D_val)
    b = np.random.rand(LEN_1D_val)
    c = np.random.rand(LEN_1D_val)
    d = np.random.rand(LEN_1D_val)

    compare_kernel(dace_s1213, {"a": a, "b": b, "c": c, "d": d}, {"LEN_1D": LEN_1D_val, "ITERATIONS": ITERATIONS_val})

    run_vectorization_test(
        dace_func=dace_s1213,
        arrays={
            "a": a,
            "b": b,
            "c": c,
            "d": d
        },
        params={
            "LEN_1D": LEN_1D_val,
            "ITERATIONS": ITERATIONS_val
        },
        save_sdfgs=SAVE_SDFGS,
        sdfg_name="dace_s1213",
        apply_loop_to_map=True,
    )
    return a, b


def test_s221():
    LEN_1D_val = 64
    ITERATIONS_val = 2

    a = np.random.rand(LEN_1D_val)
    b = np.random.rand(LEN_1D_val)
    c = np.random.rand(LEN_1D_val)
    d = np.random.rand(LEN_1D_val)

    compare_kernel(dace_s221, {"a": a, "b": b, "c": c, "d": d}, {"LEN_1D": LEN_1D_val, "ITERATIONS": ITERATIONS_val})

    run_vectorization_test(
        dace_func=dace_s221,
        arrays={
            "a": a,
            "b": b,
            "c": c,
            "d": d
        },
        params={
            "LEN_1D": LEN_1D_val,
            "ITERATIONS": ITERATIONS_val
        },
        save_sdfgs=SAVE_SDFGS,
        sdfg_name="dace_s221",
        apply_loop_to_map=True,
    )
    return a, b


def test_s1221():
    LEN_1D_val = 64
    ITERATIONS_val = 2

    a = np.random.rand(LEN_1D_val)
    b = np.random.rand(LEN_1D_val)

    compare_kernel(dace_s1221, {"a": a, "b": b}, {"LEN_1D": LEN_1D_val, "ITERATIONS": ITERATIONS_val})

    run_vectorization_test(
        dace_func=dace_s1221,
        arrays={
            "a": a,
            "b": b
        },
        params={
            "LEN_1D": LEN_1D_val,
            "ITERATIONS": ITERATIONS_val
        },
        save_sdfgs=SAVE_SDFGS,
        sdfg_name="dace_s1221",
        apply_loop_to_map=True,
    )
    return b


def test_s222():
    LEN_1D_val = 64
    ITERATIONS_val = 2

    a = np.random.rand(LEN_1D_val)
    b = np.random.rand(LEN_1D_val)
    c = np.random.rand(LEN_1D_val)
    e = np.random.rand(LEN_1D_val)

    compare_kernel(dace_s222, {"a": a, "b": b, "c": c, "e": e}, {"LEN_1D": LEN_1D_val, "ITERATIONS": ITERATIONS_val})

    run_vectorization_test(
        dace_func=dace_s222,
        arrays={
            "a": a,
            "b": b,
            "c": c,
            "e": e
        },
        params={
            "LEN_1D": LEN_1D_val,
            "ITERATIONS": ITERATIONS_val
        },
        save_sdfgs=SAVE_SDFGS,
        sdfg_name="dace_s222",
        apply_loop_to_map=True,
    )
    return a, e


def test_s231():
    LEN_2D_val = 32
    ITERATIONS_val = 32

    aa = np.random.rand(LEN_2D_val, LEN_2D_val)
    bb = np.random.rand(LEN_2D_val, LEN_2D_val)

    compare_kernel(dace_s231, {"aa": aa, "bb": bb}, {"LEN_2D": LEN_2D_val, "ITERATIONS": ITERATIONS_val})

    run_vectorization_test(
        dace_func=dace_s231,
        arrays={
            "aa": aa,
            "bb": bb
        },
        params={
            "LEN_2D": LEN_2D_val,
            "ITERATIONS": ITERATIONS_val
        },
        save_sdfgs=SAVE_SDFGS,
        sdfg_name="dace_s231",
        apply_loop_to_map=True,
    )
    return aa


def test_s232():
    LEN_2D_val = 32
    ITERATIONS_val = 32

    aa = np.random.rand(LEN_2D_val, LEN_2D_val)
    bb = np.random.rand(LEN_2D_val, LEN_2D_val)

    compare_kernel(dace_s232, {"aa": aa, "bb": bb}, {"LEN_2D": LEN_2D_val, "ITERATIONS": ITERATIONS_val})

    run_vectorization_test(
        dace_func=dace_s232,
        arrays={
            "aa": aa,
            "bb": bb
        },
        params={
            "LEN_2D": LEN_2D_val,
            "ITERATIONS": ITERATIONS_val
        },
        save_sdfgs=SAVE_SDFGS,
        sdfg_name="dace_s232",
        apply_loop_to_map=True,
    )
    return aa


def test_s1232():
    LEN_2D_val = 32
    ITERATIONS_val = 32

    aa = np.random.rand(LEN_2D_val, LEN_2D_val)
    bb = np.random.rand(LEN_2D_val, LEN_2D_val)
    cc = np.random.rand(LEN_2D_val, LEN_2D_val)

    compare_kernel(dace_s1232, {"aa": aa, "bb": bb, "cc": cc}, {"LEN_2D": LEN_2D_val, "ITERATIONS": ITERATIONS_val})

    run_vectorization_test(
        dace_func=dace_s1232,
        arrays={
            "aa": aa,
            "bb": bb,
            "cc": cc
        },
        params={
            "LEN_2D": LEN_2D_val,
            "ITERATIONS": ITERATIONS_val
        },
        save_sdfgs=SAVE_SDFGS,
        sdfg_name="dace_s1232",
        apply_loop_to_map=True,
    )
    return aa


def test_s233():
    LEN_2D_val = 32
    ITERATIONS_val = 32

    aa = np.random.rand(LEN_2D_val, LEN_2D_val)
    bb = np.random.rand(LEN_2D_val, LEN_2D_val)
    cc = np.random.rand(LEN_2D_val, LEN_2D_val)

    compare_kernel(dace_s233, {"aa": aa, "bb": bb, "cc": cc}, {"LEN_2D": LEN_2D_val, "ITERATIONS": ITERATIONS_val})

    run_vectorization_test(
        dace_func=dace_s233,
        arrays={
            "aa": aa,
            "bb": bb,
            "cc": cc
        },
        params={
            "LEN_2D": LEN_2D_val,
            "ITERATIONS": ITERATIONS_val
        },
        save_sdfgs=SAVE_SDFGS,
        sdfg_name="dace_s233",
        apply_loop_to_map=True,
    )
    return aa, bb


def test_s2233():
    LEN_2D_val = 32
    ITERATIONS_val = 32

    aa = np.random.rand(LEN_2D_val, LEN_2D_val)
    bb = np.random.rand(LEN_2D_val, LEN_2D_val)
    cc = np.random.rand(LEN_2D_val, LEN_2D_val)

    compare_kernel(dace_s2233, {"aa": aa, "bb": bb, "cc": cc}, {"LEN_2D": LEN_2D_val, "ITERATIONS": ITERATIONS_val})

    run_vectorization_test(
        dace_func=dace_s2233,
        arrays={
            "aa": aa,
            "bb": bb,
            "cc": cc
        },
        params={
            "LEN_2D": LEN_2D_val,
            "ITERATIONS": ITERATIONS_val
        },
        save_sdfgs=SAVE_SDFGS,
        sdfg_name="dace_s2233",
        apply_loop_to_map=True,
    )
    return aa, bb


def test_s235():
    LEN_2D_val = 32
    ITERATIONS_val = 2

    a = np.random.rand(LEN_2D_val)
    b = np.random.rand(LEN_2D_val)
    c = np.random.rand(LEN_2D_val)
    aa = np.random.rand(LEN_2D_val, LEN_2D_val)
    bb = np.random.rand(LEN_2D_val, LEN_2D_val)

    compare_kernel(dace_s235, {
        "a": a,
        "b": b,
        "c": c,
        "aa": aa,
        "bb": bb
    }, {
        "LEN_2D": LEN_2D_val,
        "ITERATIONS": ITERATIONS_val
    })

    run_vectorization_test(
        dace_func=dace_s235,
        arrays={
            "a": a,
            "b": b,
            "c": c,
            "aa": aa,
            "bb": bb
        },
        params={
            "LEN_2D": LEN_2D_val,
            "ITERATIONS": ITERATIONS_val
        },
        sdfg_name="dace_s235",
        save_sdfgs=SAVE_SDFGS,
        apply_loop_to_map=True,
    )
    return a, aa


def test_s241():
    LEN_1D_val = 64
    ITERATIONS_val = 2

    a = np.random.rand(LEN_1D_val)
    b = np.random.rand(LEN_1D_val)
    c = np.random.rand(LEN_1D_val)
    d = np.random.rand(LEN_1D_val)

    compare_kernel(dace_s241, {"a": a, "b": b, "c": c, "d": d}, {"LEN_1D": LEN_1D_val, "ITERATIONS": ITERATIONS_val})

    run_vectorization_test(
        dace_func=dace_s241,
        arrays={
            "a": a,
            "b": b,
            "c": c,
            "d": d
        },
        params={
            "LEN_1D": LEN_1D_val,
            "ITERATIONS": ITERATIONS_val
        },
        sdfg_name="dace_s241",
        save_sdfgs=SAVE_SDFGS,
        apply_loop_to_map=True,
    )
    return a, b


def test_s243():
    LEN_1D_val = 64
    ITERATIONS_val = 2

    a = np.random.rand(LEN_1D_val)
    b = np.random.rand(LEN_1D_val)
    c = np.random.rand(LEN_1D_val)
    d = np.random.rand(LEN_1D_val)
    e = np.random.rand(LEN_1D_val)

    compare_kernel(dace_s243, {
        "a": a,
        "b": b,
        "c": c,
        "d": d,
        "e": e
    }, {
        "LEN_1D": LEN_1D_val,
        "ITERATIONS": ITERATIONS_val
    })

    run_vectorization_test(
        dace_func=dace_s243,
        arrays={
            "a": a,
            "b": b,
            "c": c,
            "d": d,
            "e": e
        },
        params={
            "LEN_1D": LEN_1D_val,
            "ITERATIONS": ITERATIONS_val
        },
        sdfg_name="dace_s243",
        save_sdfgs=SAVE_SDFGS,
        apply_loop_to_map=True,
    )
    return a, b


def test_s244():
    LEN = 64
    ITERS = 2

    a = np.random.rand(LEN)
    b = np.random.rand(LEN)
    c = np.random.rand(LEN)
    d = np.random.rand(LEN)

    # Automatically added compare_kernel call
    compare_kernel(dace_s244, {"a": a, "b": b, "c": c, "d": d}, {"LEN_1D": LEN, "ITERATIONS": ITERS})

    run_vectorization_test(
        dace_func=dace_s244,
        arrays={
            "a": a,
            "b": b,
            "c": c,
            "d": d
        },
        params={
            "LEN_1D": LEN,
            "ITERATIONS": ITERS
        },
        sdfg_name="dace_s244",
        save_sdfgs=SAVE_SDFGS,
        apply_loop_to_map=True,
    )

    return a


def test_s1244():
    LEN = 64
    ITERS = 2

    a = np.random.rand(LEN)
    b = np.random.rand(LEN)
    c = np.random.rand(LEN)
    d = np.random.rand(LEN)

    compare_kernel(
        dace_s1244,
        {
            "a": a,
            "b": b,
            "c": c,
            "d": d
        },
        {
            "LEN_1D": LEN,
            "ITERATIONS": ITERS
        },
    )

    run_vectorization_test(
        dace_func=dace_s1244,
        arrays={
            "a": a,
            "b": b,
            "c": c,
            "d": d
        },
        params={
            "LEN_1D": LEN,
            "ITERATIONS": ITERS
        },
        sdfg_name="dace_s1244",
        apply_loop_to_map=True,
    )
    return a, d


def test_s2244():
    LEN = 64
    ITERS = 2

    a = np.random.rand(LEN)
    b = np.random.rand(LEN)
    c = np.random.rand(LEN)
    e = np.random.rand(LEN)

    #sdfg = dace_s2244.to_sdfg()
    #sdfg.save("s2244_v2.sdfg")

    compare_kernel(
        dace_s2244,
        {
            "a": a,
            "b": b,
            "c": c,
            "e": e
        },
        {
            "LEN_1D": LEN,
            "ITERATIONS": ITERS
        },
    )

    run_vectorization_test(
        dace_func=dace_s2244,
        arrays={
            "a": a,
            "b": b,
            "c": c,
            "e": e
        },
        params={
            "LEN_1D": LEN,
            "ITERATIONS": ITERS
        },
        sdfg_name="dace_s2244",
        apply_loop_to_map=True,
    )
    return a


def test_s251():
    LEN = 64
    ITERS = 2

    a = np.random.rand(LEN)
    b = np.random.rand(LEN)
    c = np.random.rand(LEN)
    d = np.random.rand(LEN)

    compare_kernel(
        dace_s251,
        {
            "a": a,
            "b": b,
            "c": c,
            "d": d
        },
        {
            "LEN_1D": LEN,
            "ITERATIONS": ITERS
        },
    )

    run_vectorization_test(
        dace_func=dace_s251,
        arrays={
            "a": a,
            "b": b,
            "c": c,
            "d": d
        },
        params={
            "LEN_1D": LEN,
            "ITERATIONS": ITERS
        },
        sdfg_name="dace_s251",
        apply_loop_to_map=True,
    )
    return a


def test_s3251():
    LEN = 64
    ITERS = 2

    a = np.random.rand(LEN)
    b = np.random.rand(LEN)
    c = np.random.rand(LEN)
    d = np.random.rand(LEN)
    e = np.random.rand(LEN)

    compare_kernel(
        dace_s3251,
        {
            "a": a,
            "b": b,
            "c": c,
            "d": d,
            "e": e
        },
        {
            "LEN_1D": LEN,
            "ITERATIONS": ITERS
        },
    )

    run_vectorization_test(
        dace_func=dace_s3251,
        arrays={
            "a": a,
            "b": b,
            "c": c,
            "d": d,
            "e": e
        },
        params={
            "LEN_1D": LEN,
            "ITERATIONS": ITERS
        },
        sdfg_name="dace_s3251",
        apply_loop_to_map=True,
    )
    return a, b, d


def test_s253():
    LEN = 64
    ITERS = 2

    a = np.random.rand(LEN)
    b = np.random.rand(LEN)
    c = np.random.rand(LEN)
    d = np.random.rand(LEN)

    compare_kernel(
        dace_s253,
        {
            "a": a,
            "b": b,
            "c": c,
            "d": d
        },
        {
            "LEN_1D": LEN,
            "ITERATIONS": ITERS
        },
    )

    run_vectorization_test(
        dace_func=dace_s253,
        arrays={
            "a": a,
            "b": b,
            "c": c,
            "d": d
        },
        params={
            "LEN_1D": LEN,
            "ITERATIONS": ITERS
        },
        sdfg_name="dace_s253",
        apply_loop_to_map=True,
    )
    return a, c


def test_s254():
    LEN = 64
    ITERS = 2

    a = np.random.rand(LEN)
    b = np.random.rand(LEN)

    compare_kernel(
        dace_s254,
        {
            "a": a,
            "b": b
        },
        {
            "LEN_1D": LEN,
            "ITERATIONS": ITERS
        },
    )

    run_vectorization_test(
        dace_func=dace_s254,
        arrays={
            "a": a,
            "b": b
        },
        params={
            "LEN_1D": LEN,
            "ITERATIONS": ITERS
        },
        sdfg_name="dace_s254",
        apply_loop_to_map=True,
    )
    return a


def test_s242():
    LEN = 64
    ITERS = 2

    a = np.random.rand(LEN)
    b = np.random.rand(LEN)
    c = np.random.rand(LEN)
    d = np.random.rand(LEN)

    compare_kernel(
        dace_s242,
        {
            "a": a,
            "b": b,
            "c": c,
            "d": d,
        },
        {
            "LEN_1D": LEN,
            "ITERATIONS": ITERS,
        },
    )

    run_vectorization_test(
        dace_func=dace_s242,
        arrays={
            "a": a,
            "b": b,
            "c": c,
            "d": d,
        },
        params={
            "LEN_1D": LEN,
            "ITERATIONS": ITERS,
        },
        sdfg_name="dace_s242",
        apply_loop_to_map=True,
    )
    return a


def test_s1251():
    LEN_1D_val = 64
    ITERATIONS_val = 2

    a = np.random.rand(LEN_1D_val)
    b = np.random.rand(LEN_1D_val)
    c = np.random.rand(LEN_1D_val)
    d = np.random.rand(LEN_1D_val)
    e = np.random.rand(LEN_1D_val)

    compare_kernel(dace_s1251, {
        "a": a,
        "b": b,
        "c": c,
        "d": d,
        "e": e
    }, {
        "LEN_1D": LEN_1D_val,
        "ITERATIONS": ITERATIONS_val
    })

    run_vectorization_test(
        dace_func=dace_s1251,
        arrays={
            "a": a,
            "b": b,
            "c": c,
            "d": d,
            "e": e
        },
        params={
            "LEN_1D": LEN_1D_val,
            "ITERATIONS": ITERATIONS_val
        },
        sdfg_name="dace_s1251",
        save_sdfgs=SAVE_SDFGS,
        apply_loop_to_map=True,
    )
    return a, b


def test_s2251():
    LEN_1D_val = 64
    ITERATIONS_val = 2

    a = np.random.rand(LEN_1D_val)
    b = np.random.rand(LEN_1D_val)
    c = np.random.rand(LEN_1D_val)
    d = np.random.rand(LEN_1D_val)
    e = np.random.rand(LEN_1D_val)

    compare_kernel(dace_s2251, {
        "a": a,
        "b": b,
        "c": c,
        "d": d,
        "e": e
    }, {
        "LEN_1D": LEN_1D_val,
        "ITERATIONS": ITERATIONS_val
    })

    run_vectorization_test(
        dace_func=dace_s2251,
        arrays={
            "a": a,
            "b": b,
            "c": c,
            "d": d,
            "e": e
        },
        params={
            "LEN_1D": LEN_1D_val,
            "ITERATIONS": ITERATIONS_val
        },
        sdfg_name="dace_s2251",
        save_sdfgs=SAVE_SDFGS,
        apply_loop_to_map=True,
    )
    return a, b


def test_s252():
    LEN_1D_val = 64
    ITERATIONS_val = 2

    a = np.random.rand(LEN_1D_val)
    b = np.random.rand(LEN_1D_val)
    c = np.random.rand(LEN_1D_val)

    compare_kernel(dace_s252, {"a": a, "b": b, "c": c}, {"LEN_1D": LEN_1D_val, "ITERATIONS": ITERATIONS_val})

    run_vectorization_test(
        dace_func=dace_s252,
        arrays={
            "a": a,
            "b": b,
            "c": c
        },
        params={
            "LEN_1D": LEN_1D_val,
            "ITERATIONS": ITERATIONS_val
        },
        sdfg_name="dace_s252",
        save_sdfgs=SAVE_SDFGS,
        apply_loop_to_map=True,
    )
    return a


def test_s255():
    LEN_1D_val = 64
    ITERATIONS_val = 2

    a = np.random.rand(LEN_1D_val)
    b = np.random.rand(LEN_1D_val)

    compare_kernel(dace_s255, {"a": a, "b": b}, {"LEN_1D": LEN_1D_val, "ITERATIONS": ITERATIONS_val})

    run_vectorization_test(
        dace_func=dace_s255,
        arrays={
            "a": a,
            "b": b
        },
        params={
            "LEN_1D": LEN_1D_val,
            "ITERATIONS": ITERATIONS_val
        },
        sdfg_name="dace_s255",
        save_sdfgs=SAVE_SDFGS,
        apply_loop_to_map=True,
    )
    return a


def test_s256():
    LEN_2D_val = 32
    ITERATIONS_val = 32

    a = np.random.rand(LEN_2D_val)
    aa = np.random.rand(LEN_2D_val, LEN_2D_val)
    bb = np.random.rand(LEN_2D_val, LEN_2D_val)
    d = np.random.rand(LEN_2D_val)

    compare_kernel(dace_s256, {
        "a": a,
        "aa": aa,
        "bb": bb,
        "d": d
    }, {
        "LEN_2D": LEN_2D_val,
        "ITERATIONS": ITERATIONS_val
    })

    run_vectorization_test(
        dace_func=dace_s256,
        arrays={
            "a": a,
            "aa": aa,
            "bb": bb,
            "d": d
        },
        params={
            "LEN_2D": LEN_2D_val,
            "ITERATIONS": ITERATIONS_val
        },
        sdfg_name="dace_s256",
        save_sdfgs=SAVE_SDFGS,
        apply_loop_to_map=True,
    )
    return a, aa


def test_s257():
    LEN_2D_val = 32
    ITERATIONS_val = 32

    a = np.random.rand(LEN_2D_val)
    aa = np.random.rand(LEN_2D_val, LEN_2D_val)
    bb = np.random.rand(LEN_2D_val, LEN_2D_val)

    compare_kernel(dace_s257, {"a": a, "aa": aa, "bb": bb}, {"LEN_2D": LEN_2D_val, "ITERATIONS": ITERATIONS_val})

    run_vectorization_test(
        dace_func=dace_s257,
        arrays={
            "a": a,
            "aa": aa,
            "bb": bb
        },
        params={
            "LEN_2D": LEN_2D_val,
            "ITERATIONS": ITERATIONS_val
        },
        sdfg_name="dace_s257",
        save_sdfgs=SAVE_SDFGS,
        apply_loop_to_map=True,
    )
    return a, aa


def test_s258():
    LEN_2D_val = 32
    ITERATIONS_val = 32

    a = np.random.rand(LEN_2D_val)
    b = np.random.rand(LEN_2D_val)
    c = np.random.rand(LEN_2D_val)
    d = np.random.rand(LEN_2D_val)
    e = np.random.rand(LEN_2D_val)
    aa = np.random.rand(1, LEN_2D_val)

    compare_kernel(dace_s258, {
        "a": a,
        "b": b,
        "c": c,
        "d": d,
        "e": e,
        "aa": aa
    }, {
        "LEN_2D": LEN_2D_val,
        "ITERATIONS": ITERATIONS_val
    })

    run_vectorization_test(
        dace_func=dace_s258,
        arrays={
            "a": a,
            "b": b,
            "c": c,
            "d": d,
            "e": e,
            "aa": aa
        },
        params={
            "LEN_2D": LEN_2D_val,
            "ITERATIONS": ITERATIONS_val
        },
        sdfg_name="dace_s258",
        save_sdfgs=SAVE_SDFGS,
        apply_loop_to_map=True,
    )
    return a, b, e


def test_s261():
    LEN_1D_val = 64
    ITERATIONS_val = 2

    a = np.random.rand(LEN_1D_val)
    b = np.random.rand(LEN_1D_val)
    c = np.random.rand(LEN_1D_val)
    d = np.random.rand(LEN_1D_val)

    compare_kernel(dace_s261, {"a": a, "b": b, "c": c, "d": d}, {"LEN_1D": LEN_1D_val, "ITERATIONS": ITERATIONS_val})

    run_vectorization_test(
        dace_func=dace_s261,
        arrays={
            "a": a,
            "b": b,
            "c": c,
            "d": d
        },
        params={
            "LEN_1D": LEN_1D_val,
            "ITERATIONS": ITERATIONS_val
        },
        sdfg_name="dace_s261",
        save_sdfgs=SAVE_SDFGS,
        apply_loop_to_map=True,
    )
    return a, c


def test_s271():
    LEN_1D_val = 64
    ITERATIONS_val = 2

    a = np.random.rand(LEN_1D_val)
    b = np.random.rand(LEN_1D_val)
    c = np.random.rand(LEN_1D_val)

    compare_kernel(dace_s271, {"a": a, "b": b, "c": c}, {"LEN_1D": LEN_1D_val, "ITERATIONS": ITERATIONS_val})

    run_vectorization_test(
        dace_func=dace_s271,
        arrays={
            "a": a,
            "b": b,
            "c": c
        },
        params={
            "LEN_1D": LEN_1D_val,
            "ITERATIONS": ITERATIONS_val
        },
        sdfg_name="dace_s271",
        save_sdfgs=SAVE_SDFGS,
        apply_loop_to_map=True,
    )
    return a


def test_s272():
    LEN_1D_val = 64
    ITERATIONS_val = 2

    a = np.random.rand(LEN_1D_val)
    b = np.random.rand(LEN_1D_val)
    c = np.random.rand(LEN_1D_val)
    d = np.random.rand(LEN_1D_val)
    e = np.random.rand(LEN_1D_val)

    compare_kernel(dace_s272, {
        "a": a,
        "b": b,
        "c": c,
        "d": d,
        "e": e
    }, {
        "LEN_1D": LEN_1D_val,
        "ITERATIONS": ITERATIONS_val,
        "threshold": 2
    })

    run_vectorization_test(
        dace_func=dace_s272,
        arrays={
            "a": a,
            "b": b,
            "c": c,
            "d": d,
            "e": e
        },
        params={
            "LEN_1D": LEN_1D_val,
            "ITERATIONS": ITERATIONS_val,
            "threshold": 2
        },
        sdfg_name="dace_s272",
        save_sdfgs=SAVE_SDFGS,
        apply_loop_to_map=True,
    )
    return a, b


def test_s273():
    LEN_1D_val = 64
    ITERATIONS_val = 2

    a = np.random.rand(LEN_1D_val)
    b = np.random.rand(LEN_1D_val)
    c = np.random.rand(LEN_1D_val)
    d = np.random.rand(LEN_1D_val)
    e = np.random.rand(LEN_1D_val)

    compare_kernel(dace_s273, {
        "a": a,
        "b": b,
        "c": c,
        "d": d,
        "e": e
    }, {
        "LEN_1D": LEN_1D_val,
        "ITERATIONS": ITERATIONS_val
    })

    run_vectorization_test(
        dace_func=dace_s273,
        arrays={
            "a": a,
            "b": b,
            "c": c,
            "d": d,
            "e": e
        },
        params={
            "LEN_1D": LEN_1D_val,
            "ITERATIONS": ITERATIONS_val
        },
        sdfg_name="dace_s273",
        save_sdfgs=SAVE_SDFGS,
        apply_loop_to_map=True,
    )
    return a, b, c


def test_s274():
    LEN_1D_val = 64
    ITERATIONS_val = 2

    a = np.random.rand(LEN_1D_val)
    b = np.random.rand(LEN_1D_val)
    c = np.random.rand(LEN_1D_val)
    d = np.random.rand(LEN_1D_val)
    e = np.random.rand(LEN_1D_val)

    compare_kernel(dace_s274, {
        "a": a,
        "b": b,
        "c": c,
        "d": d,
        "e": e
    }, {
        "LEN_1D": LEN_1D_val,
        "ITERATIONS": ITERATIONS_val
    })

    run_vectorization_test(
        dace_func=dace_s274,
        arrays={
            "a": a,
            "b": b,
            "c": c,
            "d": d,
            "e": e
        },
        params={
            "LEN_1D": LEN_1D_val,
            "ITERATIONS": ITERATIONS_val
        },
        sdfg_name="dace_s274",
        save_sdfgs=SAVE_SDFGS,
        apply_loop_to_map=True,
    )
    return a, b


def test_s275():
    LEN = 32
    ITERS = 32
    aa = np.random.rand(LEN, LEN)
    bb = np.random.rand(LEN, LEN)
    cc = np.random.rand(LEN, LEN)

    compare_kernel(dace_s275, {"aa": aa, "bb": bb, "cc": cc}, {"LEN_2D": LEN, "ITERATIONS": ITERS})

    run_vectorization_test(
        dace_func=dace_s275,
        arrays={
            "aa": aa,
            "bb": bb,
            "cc": cc
        },
        params={
            "LEN_2D": LEN,
            "ITERATIONS": ITERS
        },
        sdfg_name="dace_s275",
        apply_loop_to_map=True,
    )
    return aa


def test_s2102():
    LEN_2D = 32
    aa = np.random.rand(LEN_2D, LEN_2D)

    compare_kernel(dace_s2102, {
        "aa": aa,
    }, {
        "LEN_2D": LEN_2D,
        "ITERATIONS": 10
    })

    run_vectorization_test(
        dace_func=dace_s2102,
        arrays={"aa": aa},
        params={
            "LEN_2D": LEN_2D,
            "ITERATIONS": 10
        },
        sdfg_name="dace_s2102",
        save_sdfgs=SAVE_SDFGS,
        apply_loop_to_map=True,
    )


def test_s2111():
    LEN_2D = 32
    aa = np.random.rand(LEN_2D, LEN_2D)

    compare_kernel(dace_s2111, {
        "aa": aa,
    }, {
        "LEN_2D": LEN_2D,
        "ITERATIONS": 10
    })

    run_vectorization_test(
        dace_func=dace_s2111,
        arrays={"aa": aa},
        params={
            "LEN_2D": LEN_2D,
            "ITERATIONS": 10
        },
        sdfg_name="dace_s2111",
        save_sdfgs=SAVE_SDFGS,
        apply_loop_to_map=True,
    )


def test_s2275():
    LEN = 64
    ITERS = 2

    a = np.random.rand(LEN)
    b = np.random.rand(LEN)
    c = np.random.rand(LEN)
    d = np.random.rand(LEN)
    aa = np.random.rand(LEN, LEN)
    bb = np.random.rand(LEN, LEN)
    cc = np.random.rand(LEN, LEN)

    compare_kernel(
        dace_s2275,
        {
            "a": a,
            "b": b,
            "c": c,
            "d": d,
            "aa": aa,
            "bb": bb,
            "cc": cc
        },
        {
            "LEN_2D": LEN,
            "ITERATIONS": ITERS
        },
    )

    run_vectorization_test(
        dace_func=dace_s2275,
        arrays={
            "a": a,
            "b": b,
            "c": c,
            "d": d,
            "aa": aa,
            "bb": bb,
            "cc": cc
        },
        params={
            "LEN_2D": LEN,
            "ITERATIONS": ITERS
        },
        sdfg_name="dace_s2275",
        save_sdfgs=SAVE_SDFGS,
        apply_loop_to_map=True,
    )
    return a, aa


def test_s276():
    LEN = 64
    ITERS = 2

    a = np.random.rand(LEN)
    b = np.random.rand(LEN)
    c = np.random.rand(LEN)
    d = np.random.rand(LEN)

    compare_kernel(
        dace_s276,
        {
            "a": a,
            "b": b,
            "c": c,
            "d": d
        },
        {
            "LEN_1D": LEN,
            "ITERATIONS": ITERS
        },
    )

    run_vectorization_test(
        dace_func=dace_s276,
        arrays={
            "a": a,
            "b": b,
            "c": c,
            "d": d
        },
        params={
            "LEN_1D": LEN,
            "ITERATIONS": ITERS
        },
        sdfg_name="dace_s276",
        save_sdfgs=SAVE_SDFGS,
        apply_loop_to_map=True,
    )
    return a


def test_s277():
    LEN = 64
    ITERS = 2

    a = np.random.rand(LEN)
    b = np.random.rand(LEN)
    c = np.random.rand(LEN)
    d = np.random.rand(LEN)
    e = np.random.rand(LEN)

    compare_kernel(
        dace_s277,
        {
            "a": a,
            "b": b,
            "c": c,
            "d": d,
            "e": e
        },
        {
            "LEN_1D": LEN,
            "ITERATIONS": ITERS
        },
    )

    run_vectorization_test(
        dace_func=dace_s277,
        arrays={
            "a": a,
            "b": b,
            "c": c,
            "d": d,
            "e": e
        },
        params={
            "LEN_1D": LEN,
            "ITERATIONS": ITERS
        },
        sdfg_name="dace_s277",
        save_sdfgs=SAVE_SDFGS,
        apply_loop_to_map=True,
    )
    return a, b


def test_s278():
    LEN = 64
    ITERS = 2

    a = np.random.rand(LEN)
    b = np.random.rand(LEN)
    c = np.random.rand(LEN)
    d = np.random.rand(LEN)
    e = np.random.rand(LEN)

    compare_kernel(
        dace_s278,
        {
            "a": a,
            "b": b,
            "c": c,
            "d": d,
            "e": e
        },
        {
            "LEN_1D": LEN,
            "ITERATIONS": ITERS
        },
    )

    run_vectorization_test(
        dace_func=dace_s278,
        arrays={
            "a": a,
            "b": b,
            "c": c,
            "d": d,
            "e": e
        },
        params={
            "LEN_1D": LEN,
            "ITERATIONS": ITERS
        },
        sdfg_name="dace_s278",
        save_sdfgs=SAVE_SDFGS,
        apply_loop_to_map=True,
    )
    return a, b, c


def test_s279():
    LEN = 64
    ITERS = 2

    a = np.random.rand(LEN)
    b = np.random.rand(LEN)
    c = np.random.rand(LEN)
    d = np.random.rand(LEN)
    e = np.random.rand(LEN)

    compare_kernel(
        dace_s279,
        {
            "a": a,
            "b": b,
            "c": c,
            "d": d,
            "e": e
        },
        {
            "LEN_1D": LEN,
            "ITERATIONS": ITERS
        },
    )

    run_vectorization_test(
        dace_func=dace_s279,
        arrays={
            "a": a,
            "b": b,
            "c": c,
            "d": d,
            "e": e
        },
        params={
            "LEN_1D": LEN,
            "ITERATIONS": ITERS
        },
        sdfg_name="dace_s279",
        save_sdfgs=SAVE_SDFGS,
        apply_loop_to_map=True,
    )
    return a, b, c


def test_s1279():
    LEN = 64
    ITERS = 2

    a = np.random.rand(LEN)
    b = np.random.rand(LEN)
    c = np.random.rand(LEN)
    d = np.random.rand(LEN)
    e = np.random.rand(LEN)

    compare_kernel(
        dace_s1279,
        {
            "a": a,
            "b": b,
            "c": c,
            "d": d,
            "e": e
        },
        {
            "LEN_1D": LEN,
            "ITERATIONS": ITERS
        },
    )

    run_vectorization_test(
        dace_func=dace_s1279,
        arrays={
            "a": a,
            "b": b,
            "c": c,
            "d": d,
            "e": e
        },
        params={
            "LEN_1D": LEN,
            "ITERATIONS": ITERS
        },
        sdfg_name="dace_s1279",
        save_sdfgs=SAVE_SDFGS,
        apply_loop_to_map=True,
    )
    return c


def test_s2710():
    LEN_1D_val = 64
    ITERATIONS_val = 2

    a = np.random.rand(LEN_1D_val)
    b = np.random.rand(LEN_1D_val)
    c = np.random.rand(LEN_1D_val)
    d = np.random.rand(LEN_1D_val)
    e = np.random.rand(LEN_1D_val)
    x = np.random.rand(LEN_1D_val)

    compare_kernel(
        dace_s2710,
        {
            "a": a,
            "b": b,
            "c": c,
            "d": d,
            "e": e,
            "x": x
        },
        {
            "LEN_1D": LEN_1D_val,
            "ITERATIONS": ITERATIONS_val
        },
    )

    run_vectorization_test(
        dace_func=dace_s2710,
        arrays={
            "a": a,
            "b": b,
            "c": c,
            "d": d,
            "e": e,
            "x": x
        },
        params={
            "LEN_1D": LEN_1D_val,
            "ITERATIONS": ITERATIONS_val
        },
        sdfg_name="dace_s2710",
        save_sdfgs=SAVE_SDFGS,
        apply_loop_to_map=True,
    )
    return a, b, c


def test_s2711():
    LEN_1D_val = 64
    ITERATIONS_val = 2

    a = np.random.rand(LEN_1D_val)
    b = np.random.rand(LEN_1D_val)
    c = np.random.rand(LEN_1D_val)

    compare_kernel(
        dace_s2711,
        {
            "a": a,
            "b": b,
            "c": c
        },
        {
            "LEN_1D": LEN_1D_val,
            "ITERATIONS": ITERATIONS_val
        },
    )

    run_vectorization_test(
        dace_func=dace_s2711,
        arrays={
            "a": a,
            "b": b,
            "c": c
        },
        params={
            "LEN_1D": LEN_1D_val,
            "ITERATIONS": ITERATIONS_val
        },
        sdfg_name="dace_s2711",
        save_sdfgs=SAVE_SDFGS,
        apply_loop_to_map=True,
    )
    return a


def test_s2712():
    LEN_1D_val = 64
    ITERATIONS_val = 2

    a = np.random.rand(LEN_1D_val)
    b = np.random.rand(LEN_1D_val)
    c = np.random.rand(LEN_1D_val)

    compare_kernel(
        dace_s2712,
        {
            "a": a,
            "b": b,
            "c": c
        },
        {
            "LEN_1D": LEN_1D_val,
            "ITERATIONS": ITERATIONS_val
        },
    )

    run_vectorization_test(
        dace_func=dace_s2712,
        arrays={
            "a": a,
            "b": b,
            "c": c
        },
        params={
            "LEN_1D": LEN_1D_val,
            "ITERATIONS": ITERATIONS_val
        },
        sdfg_name="dace_s2712",
        save_sdfgs=SAVE_SDFGS,
        apply_loop_to_map=True,
    )
    return a


# ============================================================
# === 3.x REDUCTIONS (replace _run_template fully) ============
# ============================================================


def test_s312():
    LEN_1D_val = 64
    a = np.random.rand(LEN_1D_val)

    compare_kernel(
        dace_s312,
        {"a": a},
        {
            "LEN_1D": LEN_1D_val,
            "ITERATIONS": 10
        },
    )

    run_vectorization_test(
        dace_func=dace_s312,
        arrays={"a": a},
        params={
            "LEN_1D": LEN_1D_val,
            "ITERATIONS": 10
        },
        sdfg_name="dace_s312",
        save_sdfgs=SAVE_SDFGS,
        apply_loop_to_map=True,
    )
    return a


def test_s313():
    LEN_1D_val = 64
    a = np.random.rand(LEN_1D_val)
    b = np.random.rand(LEN_1D_val)
    dot = np.random.rand(LEN_1D_val)

    compare_kernel(
        dace_s313,
        {
            "a": a,
            "b": b,
            "dot": dot
        },
        {
            "LEN_1D": LEN_1D_val,
            "ITERATIONS": 10
        },
    )

    run_vectorization_test(
        dace_func=dace_s313,
        arrays={
            "a": a,
            "b": b,
            "dot": dot
        },
        params={
            "LEN_1D": LEN_1D_val,
            "ITERATIONS": 10
        },
        sdfg_name="dace_s313",
        save_sdfgs=SAVE_SDFGS,
        apply_loop_to_map=True,
    )
    return a


def test_s314():
    LEN_1D_val = 64
    a = np.random.rand(LEN_1D_val)

    compare_kernel(
        dace_s314,
        {"a": a},
        {
            "LEN_1D": LEN_1D_val,
            "ITERATIONS": 10
        },
    )

    run_vectorization_test(
        dace_func=dace_s314,
        arrays={"a": a},
        params={
            "LEN_1D": LEN_1D_val,
            "ITERATIONS": 10
        },
        sdfg_name="dace_s314",
        save_sdfgs=SAVE_SDFGS,
        apply_loop_to_map=True,
    )
    return a


def test_s315():
    LEN_1D_val = 64
    a = np.zeros(LEN_1D_val, dtype=np.float64)

    compare_kernel(
        dace_s315,
        {"a": a},
        {
            "LEN_1D": LEN_1D_val,
            "ITERATIONS": 10
        },
    )

    run_vectorization_test(
        dace_func=dace_s315,
        arrays={"a": a},
        params={
            "LEN_1D": LEN_1D_val,
            "ITERATIONS": 10
        },
        sdfg_name="dace_s315",
        save_sdfgs=SAVE_SDFGS,
        apply_loop_to_map=True,
    )
    return a


def test_s316():
    LEN_1D_val = 64
    a = np.random.rand(LEN_1D_val)

    compare_kernel(
        dace_s316,
        {"a": a},
        {
            "LEN_1D": LEN_1D_val,
            "ITERATIONS": 10
        },
    )

    run_vectorization_test(
        dace_func=dace_s316,
        arrays={"a": a},
        params={
            "LEN_1D": LEN_1D_val,
            "ITERATIONS": 10
        },
        sdfg_name="dace_s316",
        save_sdfgs=SAVE_SDFGS,
        apply_loop_to_map=True,
    )
    return a


def test_s317():
    LEN_1D_val = 64
    q = np.random.rand(LEN_1D_val)

    compare_kernel(
        dace_s317,
        {"q": q},
        {
            "LEN_1D": LEN_1D_val,
            "ITERATIONS": 10
        },
    )

    run_vectorization_test(
        dace_func=dace_s317,
        arrays={"q": q},
        params={
            "LEN_1D": LEN_1D_val,
            "ITERATIONS": 10
        },
        sdfg_name="dace_s317",
        save_sdfgs=SAVE_SDFGS,
        apply_loop_to_map=True,
    )
    return q


def test_s318():
    LEN_1D_val = 64
    a = np.random.rand(LEN_1D_val)
    inc_val = 2

    compare_kernel(
        dace_s318,
        {"a": a},
        {
            "LEN_1D": LEN_1D_val,
            "ITERATIONS": 10,
            "inc": inc_val
        },
    )

    run_vectorization_test(
        dace_func=dace_s318,
        arrays={"a": a},
        params={
            "LEN_1D": LEN_1D_val,
            "ITERATIONS": 10,
            "inc": inc_val
        },
        sdfg_name="dace_s318",
        save_sdfgs=SAVE_SDFGS,
        apply_loop_to_map=True,
    )
    return a


def test_s319():
    LEN_1D_val = 64
    a = np.zeros(LEN_1D_val)
    b = np.zeros(LEN_1D_val)
    c = np.random.rand(LEN_1D_val)
    d = np.random.rand(LEN_1D_val)
    e = np.random.rand(LEN_1D_val)

    compare_kernel(
        dace_s319,
        {
            "a": a,
            "b": b,
            "c": c,
            "d": d,
            "e": e
        },
        {
            "LEN_1D": LEN_1D_val,
            "ITERATIONS": 10
        },
    )

    run_vectorization_test(
        dace_func=dace_s319,
        arrays={
            "a": a,
            "b": b,
            "c": c,
            "d": d,
            "e": e
        },
        params={
            "LEN_1D": LEN_1D_val,
            "ITERATIONS": 10
        },
        sdfg_name="dace_s319",
        save_sdfgs=SAVE_SDFGS,
        apply_loop_to_map=True,
    )
    return a, b


def test_s3110():
    LEN_2D_val = 32
    aa = np.random.rand(LEN_2D_val, LEN_2D_val)

    compare_kernel(
        dace_s3110,
        {"aa": aa},
        {
            "LEN_2D": LEN_2D_val,
            "ITERATIONS": 10
        },
    )

    run_vectorization_test(
        dace_func=dace_s3110,
        arrays={"aa": aa},
        params={
            "LEN_2D": LEN_2D_val,
            "ITERATIONS": 10
        },
        sdfg_name="dace_s3110",
        save_sdfgs=SAVE_SDFGS,
        apply_loop_to_map=True,
    )
    return aa


def test_s13110():
    LEN_2D_val = 32
    aa = np.random.rand(LEN_2D_val, LEN_2D_val)

    compare_kernel(
        dace_s13110,
        {"aa": aa},
        {
            "LEN_2D": LEN_2D_val,
            "ITERATIONS": 10
        },
    )

    run_vectorization_test(
        dace_func=dace_s13110,
        arrays={"aa": aa},
        params={
            "LEN_2D": LEN_2D_val,
            "ITERATIONS": 10
        },
        sdfg_name="dace_s13110",
        save_sdfgs=SAVE_SDFGS,
        apply_loop_to_map=True,
    )
    return aa


def test_s3111():
    LEN_1D_val = 64
    a = np.random.randn(LEN_1D_val)

    compare_kernel(
        dace_s3111,
        {"a": a},
        {
            "LEN_1D": LEN_1D_val,
            "ITERATIONS": 10
        },
    )

    run_vectorization_test(
        dace_func=dace_s3111,
        arrays={"a": a},
        params={
            "LEN_1D": LEN_1D_val,
            "ITERATIONS": 10
        },
        sdfg_name="dace_s3111",
        save_sdfgs=SAVE_SDFGS,
        apply_loop_to_map=True,
    )
    return a


def test_s3112():
    LEN_1D_val = 64
    ITERATIONS_val = 2

    a = np.random.rand(LEN_1D_val)
    b = np.zeros(LEN_1D_val)

    compare_kernel(
        dace_s3112,
        {
            "a": a,
            "b": b
        },
        {
            "LEN_1D": LEN_1D_val,
            "ITERATIONS": ITERATIONS_val
        },
    )

    run_vectorization_test(
        dace_func=dace_s3112,
        arrays={
            "a": a,
            "b": b
        },
        params={
            "LEN_1D": LEN_1D_val,
            "ITERATIONS": ITERATIONS_val
        },
        sdfg_name="dace_s3112",
        save_sdfgs=SAVE_SDFGS,
        apply_loop_to_map=True,
    )
    return b


def test_s3113():
    LEN_1D_val = 64
    ITERATIONS_val = 2

    a = np.random.rand(LEN_1D_val)

    compare_kernel(
        dace_s3113,
        {"a": a},
        {
            "LEN_1D": LEN_1D_val,
            "ITERATIONS": ITERATIONS_val
        },
    )

    run_vectorization_test(
        dace_func=dace_s3113,
        arrays={"a": a},
        params={
            "LEN_1D": LEN_1D_val,
            "ITERATIONS": ITERATIONS_val
        },
        sdfg_name="dace_s3113",
        save_sdfgs=SAVE_SDFGS,
        apply_loop_to_map=True,
    )
    return a


def test_s321():
    LEN = 64
    ITERS = 2

    a = np.random.rand(LEN)
    b = np.random.rand(LEN)

    compare_kernel(
        dace_s321,
        {
            "a": a,
            "b": b
        },
        {
            "LEN_1D": LEN,
            "ITERATIONS": ITERS
        },
    )

    run_vectorization_test(
        dace_func=dace_s321,
        arrays={
            "a": a,
            "b": b
        },
        params={
            "LEN_1D": LEN,
            "ITERATIONS": ITERS
        },
        sdfg_name="dace_s321",
        save_sdfgs=SAVE_SDFGS,
        apply_loop_to_map=True,
    )
    return a


def test_s322():
    LEN = 64
    ITERS = 2

    a = np.random.rand(LEN)
    b = np.random.rand(LEN)
    c = np.random.rand(LEN)

    compare_kernel(
        dace_s322,
        {
            "a": a,
            "b": b,
            "c": c
        },
        {
            "LEN_1D": LEN,
            "ITERATIONS": ITERS
        },
    )

    run_vectorization_test(
        dace_func=dace_s322,
        arrays={
            "a": a,
            "b": b,
            "c": c
        },
        params={
            "LEN_1D": LEN,
            "ITERATIONS": ITERS
        },
        sdfg_name="dace_s322",
        save_sdfgs=SAVE_SDFGS,
        apply_loop_to_map=True,
    )
    return a


def test_s323():
    LEN = 64
    ITERS = 2

    a = np.random.rand(LEN)
    b = np.random.rand(LEN)
    c = np.random.rand(LEN)
    d = np.random.rand(LEN)
    e = np.random.rand(LEN)

    compare_kernel(
        dace_s323,
        {
            "a": a,
            "b": b,
            "c": c,
            "d": d,
            "e": e
        },
        {
            "LEN_1D": LEN,
            "ITERATIONS": ITERS
        },
    )

    run_vectorization_test(
        dace_func=dace_s323,
        arrays={
            "a": a,
            "b": b,
            "c": c,
            "d": d,
            "e": e
        },
        params={
            "LEN_1D": LEN,
            "ITERATIONS": ITERS
        },
        sdfg_name="dace_s323",
        save_sdfgs=SAVE_SDFGS,
        apply_loop_to_map=True,
    )
    return a, b


def test_s331():
    LEN = 64
    ITERS = 2

    a = np.random.rand(LEN) - 0.5

    compare_kernel(
        dace_s331,
        {"a": a},
        {
            "LEN_1D": LEN,
            "ITERATIONS": ITERS
        },
    )

    run_vectorization_test(
        dace_func=dace_s331,
        arrays={"a": a},
        params={
            "LEN_1D": LEN,
            "ITERATIONS": ITERS
        },
        sdfg_name="dace_s331",
        save_sdfgs=SAVE_SDFGS,
        apply_loop_to_map=True,
    )
    return a


def test_s332():
    LEN = 64
    ITERS = 2

    a = np.random.rand(LEN)

    compare_kernel(
        dace_s332,
        {"a": a},
        {
            "LEN_1D": LEN,
            "ITERATIONS": ITERS
        },
    )

    run_vectorization_test(
        dace_func=dace_s332,
        arrays={"a": a},
        params={
            "LEN_1D": LEN,
            "ITERATIONS": ITERS
        },
        sdfg_name="dace_s332",
        save_sdfgs=SAVE_SDFGS,
        apply_loop_to_map=True,
    )
    return a


def test_s341():
    LEN = 64
    ITERS = 2

    a = np.zeros(LEN)
    b = np.random.rand(LEN) - 0.5

    compare_kernel(
        dace_s341,
        {
            "a": a,
            "b": b
        },
        {
            "LEN_1D": LEN,
            "ITERATIONS": ITERS
        },
    )

    run_vectorization_test(
        dace_func=dace_s341,
        arrays={
            "a": a,
            "b": b
        },
        params={
            "LEN_1D": LEN,
            "ITERATIONS": ITERS
        },
        sdfg_name="dace_s341",
        save_sdfgs=SAVE_SDFGS,
        apply_loop_to_map=True,
    )
    return a


def test_s342():
    LEN = 64
    ITERS = 2

    a = np.random.rand(LEN) - 0.5
    b = np.random.rand(LEN)

    compare_kernel(
        dace_s342,
        {
            "a": a,
            "b": b
        },
        {
            "LEN_1D": LEN,
            "ITERATIONS": ITERS
        },
    )

    run_vectorization_test(
        dace_func=dace_s342,
        arrays={
            "a": a,
            "b": b
        },
        params={
            "LEN_1D": LEN,
            "ITERATIONS": ITERS
        },
        sdfg_name="dace_s342",
        save_sdfgs=SAVE_SDFGS,
        apply_loop_to_map=True,
    )
    return a


def test_s343():
    LEN2 = 32
    ITERS = 2

    aa = np.random.rand(LEN2, LEN2)
    bb = np.random.rand(LEN2, LEN2) - 0.5
    flat = np.zeros(LEN2 * LEN2)

    compare_kernel(
        dace_s343,
        {
            "aa": aa,
            "bb": bb,
            "flat_2d_array": flat
        },
        {
            "LEN_2D": LEN2,
            "ITERATIONS": ITERS
        },
    )

    run_vectorization_test(
        dace_func=dace_s343,
        arrays={
            "aa": aa,
            "bb": bb,
            "flat_2d_array": flat
        },
        params={
            "LEN_2D": LEN2,
            "ITERATIONS": ITERS
        },
        sdfg_name="dace_s343",
        save_sdfgs=SAVE_SDFGS,
        apply_loop_to_map=True,
    )
    return flat


def test_s351():
    LEN = 64
    ITERS = 2

    a = np.random.rand(LEN)
    b = np.random.rand(LEN)
    c = np.random.rand(LEN)

    compare_kernel(
        dace_s351,
        {
            "a": a,
            "b": b,
            "c": c
        },
        {
            "LEN_1D": LEN,
            "ITERATIONS": ITERS
        },
    )

    run_vectorization_test(
        dace_func=dace_s351,
        arrays={
            "a": a,
            "b": b,
            "c": c
        },
        params={
            "LEN_1D": LEN,
            "ITERATIONS": ITERS
        },
        sdfg_name="dace_s351",
        save_sdfgs=SAVE_SDFGS,
        apply_loop_to_map=True,
    )
    return a


def test_s1351():
    LEN = 64
    ITERS = 2

    a = np.zeros(LEN)
    b = np.random.rand(LEN)
    c = np.random.rand(LEN)

    compare_kernel(
        dace_s1351,
        {
            "a": a,
            "b": b,
            "c": c
        },
        {
            "LEN_1D": LEN,
            "ITERATIONS": ITERS
        },
    )

    run_vectorization_test(
        dace_func=dace_s1351,
        arrays={
            "a": a,
            "b": b,
            "c": c
        },
        params={
            "LEN_1D": LEN,
            "ITERATIONS": ITERS
        },
        sdfg_name="dace_s1351",
        save_sdfgs=SAVE_SDFGS,
        apply_loop_to_map=True,
    )
    return a


def test_s352():
    LEN = 64
    ITERS = 2

    a = np.random.rand(LEN)
    b = np.random.rand(LEN)

    compare_kernel(
        dace_s352,
        {
            "a": a,
            "b": b
        },
        {
            "LEN_1D": LEN,
            "ITERATIONS": ITERS
        },
    )

    run_vectorization_test(
        dace_func=dace_s352,
        arrays={
            "a": a,
            "b": b
        },
        params={
            "LEN_1D": LEN,
            "ITERATIONS": ITERS
        },
        sdfg_name="dace_s352",
        save_sdfgs=SAVE_SDFGS,
        apply_loop_to_map=True,
    )
    return a, b


def test_s353():
    LEN = 64
    ITERS = 2

    a = np.random.rand(LEN)
    b = np.random.rand(LEN)
    c = np.random.rand(LEN)
    ip = np.random.randint(0, LEN, size=LEN, dtype=np.int32)

    compare_kernel(
        dace_s353,
        {
            "a": a,
            "b": b,
            "c": c,
            "ip": ip
        },
        {
            "LEN_1D": LEN,
            "ITERATIONS": ITERS
        },
    )

    run_vectorization_test(
        dace_func=dace_s353,
        arrays={
            "a": a,
            "b": b,
            "c": c,
            "ip": ip
        },
        params={
            "LEN_1D": LEN,
            "ITERATIONS": ITERS
        },
        sdfg_name="dace_s353",
        save_sdfgs=SAVE_SDFGS,
        apply_loop_to_map=True,
    )
    return a


def test_vdotr():
    LEN = 64
    ITERS = 16

    a = np.random.rand(LEN)
    b = np.random.rand(LEN)
    dot_out = np.random.rand(LEN)

    compare_kernel(
        dace_vdotr,
        {
            "a": a,
            "b": b,
            "dot_out": dot_out
        },
        {
            "LEN_1D": LEN,
            "ITERATIONS": ITERS
        },
    )

    run_vectorization_test(
        dace_func=dace_vdotr,
        arrays={
            "a": a,
            "b": b,
            "dot_out": dot_out
        },
        params={
            "LEN_1D": LEN,
            "ITERATIONS": ITERS
        },
        sdfg_name="dace_vdotr",
        save_sdfgs=SAVE_SDFGS,
        apply_loop_to_map=True,
    )
    return a


def test_vbor():
    LEN = 64
    ITERS = 16

    a = np.random.rand(LEN)
    b = np.random.rand(LEN)
    c = np.random.rand(LEN)
    d = np.random.rand(LEN)
    e = np.random.rand(LEN)
    x = np.zeros(LEN)

    compare_kernel(
        dace_vbor,
        {
            "a": a,
            "b": b,
            "c": c,
            "d": d,
            "e": e,
            "x": x
        },
        {
            "LEN_2D": LEN,
            "ITERATIONS": ITERS
        },
    )

    run_vectorization_test(
        dace_func=dace_vbor,
        arrays={
            "a": a,
            "b": b,
            "c": c,
            "d": d,
            "e": e,
            "x": x
        },
        params={
            "LEN_2D": LEN,
            "ITERATIONS": ITERS
        },
        sdfg_name="dace_vbor",
        save_sdfgs=SAVE_SDFGS,
        apply_loop_to_map=True,
    )
    return x


def test_s281():
    LEN = 64
    ITERS = 16

    a = np.random.rand(LEN)
    b = np.random.rand(LEN)
    c = np.random.rand(LEN)

    compare_kernel(
        dace_s281,
        {
            "a": a,
            "b": b,
            "c": c,
        },
        {
            "LEN_1D": LEN,
            "ITERATIONS": ITERS
        },
    )

    run_vectorization_test(
        dace_func=dace_s281,
        arrays={
            "a": a,
            "b": b,
            "c": c,
        },
        params={
            "LEN_1D": LEN,
            "ITERATIONS": ITERS
        },
        sdfg_name="s281",
        save_sdfgs=SAVE_SDFGS,
        apply_loop_to_map=True,
    )
    return a


def test_s1281():
    LEN = 64
    ITERS = 16

    a = np.random.rand(LEN)
    b = np.random.rand(LEN)
    c = np.random.rand(LEN)
    d = np.random.rand(LEN)
    e = np.random.rand(LEN)

    compare_kernel(
        dace_s1281,
        {
            "a": a,
            "b": b,
            "c": c,
            "d": d,
            "e": e,
        },
        {
            "LEN_1D": LEN,
            "ITERATIONS": ITERS
        },
    )

    run_vectorization_test(
        dace_func=dace_s1281,
        arrays={
            "a": a,
            "b": b,
            "c": c,
            "d": d,
            "e": e,
        },
        params={
            "LEN_1D": LEN,
            "ITERATIONS": ITERS
        },
        sdfg_name="s1281",
        save_sdfgs=SAVE_SDFGS,
        apply_loop_to_map=True,
    )
    return a


def test_s291():
    LEN = 64
    ITERS = 16

    a = np.random.rand(LEN)
    b = np.random.rand(LEN)

    #sdfg = dace_s291.to_sdfg()
    #sdfg.apply_transformations_repeated(LoopToMap)
    #sdfg.save("s291.sdfg")

    compare_kernel(
        dace_s291,
        {
            "a": a,
            "b": b,
        },
        {
            "LEN_1D": LEN,
            "ITERATIONS": ITERS
        },
    )

    run_vectorization_test(
        dace_func=dace_s291,
        arrays={
            "a": a,
            "b": b,
        },
        params={
            "LEN_1D": LEN,
            "ITERATIONS": ITERS
        },
        sdfg_name="s291",
        save_sdfgs=SAVE_SDFGS,
        apply_loop_to_map=True,
    )
    return a


def test_s292():
    LEN = 64
    ITERS = 16

    a = np.random.rand(LEN)
    b = np.random.rand(LEN)
    compare_kernel(
        dace_s292,
        {
            "a": a,
            "b": b,
        },
        {
            "LEN_1D": LEN,
            "ITERATIONS": ITERS
        },
    )

    run_vectorization_test(
        dace_func=dace_s292,
        arrays={
            "a": a,
            "b": b,
        },
        params={
            "LEN_1D": LEN,
            "ITERATIONS": ITERS
        },
        sdfg_name="s292",
        save_sdfgs=SAVE_SDFGS,
        apply_loop_to_map=True,
    )
    return a


def test_s293():
    LEN = 64
    ITERS = 16

    a = np.random.rand(LEN)

    compare_kernel(
        dace_s293,
        {
            "a": a,
        },
        {
            "LEN_1D": LEN,
            "ITERATIONS": ITERS
        },
    )

    run_vectorization_test(
        dace_func=dace_s293,
        arrays={
            "a": a,
        },
        params={
            "LEN_1D": LEN,
            "ITERATIONS": ITERS
        },
        sdfg_name="s293",
        save_sdfgs=SAVE_SDFGS,
        apply_loop_to_map=True,
    )
    return a


def test_s2101():
    LEN = 32
    ITERS = 8

    aa = np.random.rand(LEN, LEN)
    bb = np.random.rand(LEN, LEN)
    cc = np.random.rand(LEN, LEN)

    compare_kernel(
        dace_s2101,
        {
            "aa": aa,
            "bb": bb,
            "cc": cc
        },
        {
            "LEN_2D": LEN,
            "ITERATIONS": ITERS
        },
    )

    run_vectorization_test(
        dace_func=dace_s2101,
        arrays={
            "aa": aa,
            "bb": bb,
            "cc": cc
        },
        params={
            "LEN_2D": LEN,
            "ITERATIONS": ITERS
        },
        sdfg_name="s2101",
        save_sdfgs=SAVE_SDFGS,
        apply_loop_to_map=True,
    )
    return aa


def test_s311():
    LEN = 256
    ITERS = 2

    a = np.random.rand(LEN)
    sum_out = np.random.rand(LEN)

    compare_kernel(
        dace_s311,
        {
            "a": a,
            "sum_out": sum_out
        },
        {
            "LEN_1D": LEN,
            "ITERATIONS": ITERS
        },
    )

    run_vectorization_test(
        dace_func=dace_s311,
        arrays={
            "a": a,
            "sum_out": sum_out
        },
        params={
            "LEN_1D": LEN,
            "ITERATIONS": ITERS
        },
        sdfg_name="s311",
        save_sdfgs=SAVE_SDFGS,
        apply_loop_to_map=True,
    )
    return a


# Missing test functions for TSVC kernels


def test_s1421():
    LEN = 64
    ITERS = 2

    # b is updated based on a; order matches dace_s1421 signature (b, a)
    b = np.random.rand(LEN).astype(np.float64)
    a = np.random.rand(LEN).astype(np.float64)

    compare_kernel(
        dace_s1421,
        {
            "b": b,
            "a": a
        },
        {
            "LEN_1D": LEN,
            "ITERATIONS": ITERS
        },
    )

    run_vectorization_test(
        dace_func=dace_s1421,
        arrays={
            "b": b,
            "a": a
        },
        params={
            "LEN_1D": LEN,
            "ITERATIONS": ITERS
        },
        sdfg_name="dace_s1421",
        save_sdfgs=SAVE_SDFGS,
        apply_loop_to_map=True,
    )
    return b


def test_s4112():
    LEN = 64
    ITERS = 2

    a = np.random.rand(LEN).astype(np.float64)
    b = np.random.rand(LEN).astype(np.float64)
    ip = np.random.randint(0, LEN, size=LEN).astype(np.int32)

    compare_kernel(
        dace_s4112,
        {
            "a": a,
            "b": b,
            "ip": ip
        },
        {
            "LEN_1D": LEN,
            "ITERATIONS": ITERS
        },
    )

    run_vectorization_test(
        dace_func=dace_s4112,
        arrays={
            "a": a,
            "b": b,
            "ip": ip
        },
        params={
            "LEN_1D": LEN,
            "ITERATIONS": ITERS
        },
        sdfg_name="dace_s4112",
        save_sdfgs=SAVE_SDFGS,
        apply_loop_to_map=True,
    )
    return a


def test_s4113():
    LEN = 64
    ITERS = 2

    a = np.random.rand(LEN).astype(np.float64)
    b = np.random.rand(LEN).astype(np.float64)
    c = np.random.rand(LEN).astype(np.float64)
    ip = np.random.randint(0, LEN, size=LEN).astype(np.int32)

    compare_kernel(
        dace_s4113,
        {
            "a": a,
            "b": b,
            "c": c,
            "ip": ip
        },
        {
            "LEN_1D": LEN,
            "ITERATIONS": ITERS
        },
    )

    run_vectorization_test(
        dace_func=dace_s4113,
        arrays={
            "a": a,
            "b": b,
            "c": c,
            "ip": ip
        },
        params={
            "LEN_1D": LEN,
            "ITERATIONS": ITERS
        },
        sdfg_name="dace_s4113",
        save_sdfgs=SAVE_SDFGS,
        apply_loop_to_map=True,
    )
    return a


def test_s4114():
    LEN = 64
    ITERS = 2
    n1_val = LEN // 2

    a = np.random.rand(LEN).astype(np.float64)
    b = np.random.rand(LEN).astype(np.float64)
    c = np.random.rand(LEN).astype(np.float64)
    d_arr = np.random.rand(LEN).astype(np.float64)
    ip = np.random.randint(0, LEN, size=LEN).astype(np.int32)

    compare_kernel(
        dace_s4114,
        {
            "a": a,
            "b": b,
            "c": c,
            "d_": d_arr,
            "ip": ip
        },
        {
            "LEN_1D": LEN,
            "ITERATIONS": ITERS,
            "n1": n1_val
        },
    )

    run_vectorization_test(
        dace_func=dace_s4114,
        arrays={
            "a": a,
            "b": b,
            "c": c,
            "d_": d_arr,
            "ip": ip
        },
        params={
            "LEN_1D": LEN,
            "ITERATIONS": ITERS,
            "n1": n1_val
        },
        sdfg_name="dace_s4114",
        save_sdfgs=SAVE_SDFGS,
        apply_loop_to_map=True,
    )
    return a


def test_s4115():
    LEN = 64
    ITERS = 2

    a = np.random.rand(LEN).astype(np.float64)
    b = np.random.rand(LEN).astype(np.float64)
    ip = np.random.randint(0, LEN, size=LEN).astype(np.int32)
    sum_out = np.zeros(1, dtype=np.float64)

    compare_kernel(
        dace_s4115,
        {
            "a": a,
            "b": b,
            "ip": ip,
            "sum_out": sum_out
        },
        {
            "LEN_1D": LEN,
            "ITERATIONS": ITERS
        },
    )

    run_vectorization_test(
        dace_func=dace_s4115,
        arrays={
            "a": a,
            "b": b,
            "ip": ip,
            "sum_out": sum_out
        },
        params={
            "LEN_1D": LEN,
            "ITERATIONS": ITERS
        },
        sdfg_name="dace_s4115",
        save_sdfgs=SAVE_SDFGS,
        apply_loop_to_map=True,
    )
    return sum_out


def test_s4116():
    LEN1 = 64  # length of vector a
    LEN2 = 32  # dimensions of aa and ip
    ITERS = 2
    j_val = 1
    inc_val = 0

    a = np.random.rand(LEN1).astype(np.float64)
    aa = np.random.rand(LEN2, LEN2).astype(np.float64)
    ip = np.random.randint(0, LEN2, size=LEN2).astype(np.int32)
    sum_out = np.zeros(2, dtype=np.float64)

    compare_kernel(
        dace_s4116,
        {
            "a": a,
            "aa": aa,
            "ip": ip,
            "sum_out": sum_out
        },
        {
            "LEN_1D": LEN1,
            "LEN_2D": LEN2,
            "ITERATIONS": ITERS,
            "j": j_val,
            "inc": inc_val
        },
    )

    run_vectorization_test(
        dace_func=dace_s4116,
        arrays={
            "a": a,
            "aa": aa,
            "ip": ip,
            "sum_out": sum_out
        },
        params={
            "LEN_1D": LEN1,
            "LEN_2D": LEN2,
            "ITERATIONS": ITERS,
            "j": j_val,
            "inc": inc_val
        },
        sdfg_name="dace_s4116",
        save_sdfgs=SAVE_SDFGS,
        apply_loop_to_map=True,
    )
    return sum_out


def test_s4117():
    LEN = 64
    ITERS = 2

    a = np.random.rand(LEN).astype(np.float64)
    b = np.random.rand(LEN).astype(np.float64)
    c = np.random.rand(LEN).astype(np.float64)
    d = np.random.rand(LEN).astype(np.float64)

    compare_kernel(
        dace_s4117,
        {
            "a": a,
            "b": b,
            "c": c,
            "d": d
        },
        {
            "LEN_1D": LEN,
            "ITERATIONS": ITERS
        },
    )

    run_vectorization_test(
        dace_func=dace_s4117,
        arrays={
            "a": a,
            "b": b,
            "c": c,
            "d": d
        },
        params={
            "LEN_1D": LEN,
            "ITERATIONS": ITERS
        },
        sdfg_name="dace_s4117",
        save_sdfgs=SAVE_SDFGS,
        apply_loop_to_map=True,
    )
    return a


def test_s4121():
    LEN = 64
    ITERS = 2

    a = np.random.rand(LEN).astype(np.float64)
    b = np.random.rand(LEN).astype(np.float64)
    c = np.random.rand(LEN).astype(np.float64)

    compare_kernel(
        dace_s4121,
        {
            "a": a,
            "b": b,
            "c": c
        },
        {
            "LEN_1D": LEN,
            "ITERATIONS": ITERS
        },
    )

    run_vectorization_test(
        dace_func=dace_s4121,
        arrays={
            "a": a,
            "b": b,
            "c": c
        },
        params={
            "LEN_1D": LEN,
            "ITERATIONS": ITERS
        },
        sdfg_name="dace_s4121",
        save_sdfgs=SAVE_SDFGS,
        apply_loop_to_map=True,
    )
    return a


def test_s422():
    LEN1 = 64
    ITERS = 2

    a = np.random.rand(LEN1).astype(np.float64)
    flat = np.random.rand(LEN1 * LEN1).astype(np.float64)

    compare_kernel(
        dace_s422,
        {
            "a": a,
            "flat_2d_array": flat
        },
        {
            "LEN_1D": LEN1,
            "ITERATIONS": ITERS
        },
    )

    run_vectorization_test(
        dace_func=dace_s422,
        arrays={
            "a": a,
            "flat_2d_array": flat
        },
        params={
            "LEN_1D": LEN1,
            "ITERATIONS": ITERS
        },
        sdfg_name="dace_s422",
        save_sdfgs=SAVE_SDFGS,
        apply_loop_to_map=True,
    )
    return flat


def test_s424():
    LEN = 64
    ITERS = 2

    a = np.random.rand(LEN).astype(np.float64)
    xx = np.random.rand(LEN).astype(np.float64)
    flat = np.random.rand(LEN).astype(np.float64)

    compare_kernel(
        dace_s424,
        {
            "a": a,
            "xx": xx,
            "flat": flat
        },
        {
            "LEN_1D": LEN,
            "ITERATIONS": ITERS
        },
    )

    run_vectorization_test(
        dace_func=dace_s424,
        arrays={
            "a": a,
            "xx": xx,
            "flat": flat
        },
        params={
            "LEN_1D": LEN,
            "ITERATIONS": ITERS
        },
        sdfg_name="dace_s424",
        save_sdfgs=SAVE_SDFGS,
        apply_loop_to_map=True,
    )
    return xx


def test_s431():
    LEN = 64
    ITERS = 2

    a = np.random.rand(LEN).astype(np.float64)
    b = np.random.rand(LEN).astype(np.float64)

    compare_kernel(
        dace_s431,
        {
            "a": a,
            "b": b
        },
        {
            "LEN_1D": LEN,
            "ITERATIONS": ITERS
        },
    )

    run_vectorization_test(
        dace_func=dace_s431,
        arrays={
            "a": a,
            "b": b
        },
        params={
            "LEN_1D": LEN,
            "ITERATIONS": ITERS
        },
        sdfg_name="dace_s431",
        save_sdfgs=SAVE_SDFGS,
        apply_loop_to_map=True,
    )
    return a


def test_s441():
    LEN = 64
    ITERS = 2

    a = np.random.rand(LEN).astype(np.float64)
    b = np.random.rand(LEN).astype(np.float64)
    c = np.random.rand(LEN).astype(np.float64)
    d = np.random.rand(LEN).astype(np.float64)

    compare_kernel(
        dace_s441,
        {
            "a": a,
            "b": b,
            "c": c,
            "d": d
        },
        {
            "LEN_1D": LEN,
            "ITERATIONS": ITERS
        },
    )

    run_vectorization_test(
        dace_func=dace_s441,
        arrays={
            "a": a,
            "b": b,
            "c": c,
            "d": d
        },
        params={
            "LEN_1D": LEN,
            "ITERATIONS": ITERS
        },
        sdfg_name="dace_s441",
        save_sdfgs=SAVE_SDFGS,
        apply_loop_to_map=True,
    )
    return a


def test_s442():
    LEN = 64
    ITERS = 2

    a = np.random.rand(LEN).astype(np.float64)
    b = np.random.rand(LEN).astype(np.float64)
    c = np.random.rand(LEN).astype(np.float64)
    d = np.random.rand(LEN).astype(np.float64)
    e = np.random.rand(LEN).astype(np.float64)
    indx = np.random.randint(1, 5, size=LEN).astype(np.int32)  # values 1..4

    compare_kernel(
        dace_s442,
        {
            "a": a,
            "b": b,
            "c": c,
            "d": d,
            "e": e,
            "indx": indx
        },
        {
            "LEN_1D": LEN,
            "ITERATIONS": ITERS
        },
    )

    run_vectorization_test(
        dace_func=dace_s442,
        arrays={
            "a": a,
            "b": b,
            "c": c,
            "d": d,
            "e": e,
            "indx": indx
        },
        params={
            "LEN_1D": LEN,
            "ITERATIONS": ITERS
        },
        sdfg_name="dace_s442",
        save_sdfgs=SAVE_SDFGS,
        apply_loop_to_map=True,
    )
    return a


def test_s443():
    LEN = 64
    ITERS = 2

    a = np.random.rand(LEN).astype(np.float64)
    b = np.random.rand(LEN).astype(np.float64)
    c = np.random.rand(LEN).astype(np.float64)
    d = np.random.rand(LEN).astype(np.float64)

    compare_kernel(
        dace_s443,
        {
            "a": a,
            "b": b,
            "c": c,
            "d": d
        },
        {
            "LEN_1D": LEN,
            "ITERATIONS": ITERS
        },
    )

    run_vectorization_test(
        dace_func=dace_s443,
        arrays={
            "a": a,
            "b": b,
            "c": c,
            "d": d
        },
        params={
            "LEN_1D": LEN,
            "ITERATIONS": ITERS
        },
        sdfg_name="dace_s443",
        save_sdfgs=SAVE_SDFGS,
        apply_loop_to_map=True,
    )
    return a


def test_s451():
    LEN = 64
    ITERS = 10  # ensure ITERATIONS//5 > 0

    a = np.random.rand(LEN).astype(np.float64)
    b = np.random.rand(LEN).astype(np.float64)
    c = np.random.rand(LEN).astype(np.float64)

    compare_kernel(
        dace_s451,
        {
            "a": a,
            "b": b,
            "c": c
        },
        {
            "LEN_1D": LEN,
            "ITERATIONS": ITERS
        },
    )

    run_vectorization_test(
        dace_func=dace_s451,
        arrays={
            "a": a,
            "b": b,
            "c": c
        },
        params={
            "LEN_1D": LEN,
            "ITERATIONS": ITERS
        },
        sdfg_name="dace_s451",
        save_sdfgs=SAVE_SDFGS,
        apply_loop_to_map=True,
    )
    return a


def test_s452():
    LEN = 64
    ITERS = 2

    a = np.random.rand(LEN).astype(np.float64)
    b = np.random.rand(LEN).astype(np.float64)
    c = np.random.rand(LEN).astype(np.float64)

    compare_kernel(
        dace_s452,
        {
            "a": a,
            "b": b,
            "c": c
        },
        {
            "LEN_1D": LEN,
            "ITERATIONS": ITERS
        },
    )

    run_vectorization_test(
        dace_func=dace_s452,
        arrays={
            "a": a,
            "b": b,
            "c": c
        },
        params={
            "LEN_1D": LEN,
            "ITERATIONS": ITERS
        },
        sdfg_name="dace_s452",
        save_sdfgs=SAVE_SDFGS,
        apply_loop_to_map=True,
    )
    return a


def test_s453():
    LEN = 64
    ITERS = 2

    a = np.random.rand(LEN).astype(np.float64)
    b = np.random.rand(LEN).astype(np.float64)

    compare_kernel(
        dace_s453,
        {
            "a": a,
            "b": b
        },
        {
            "LEN_1D": LEN,
            "ITERATIONS": ITERS
        },
    )

    run_vectorization_test(
        dace_func=dace_s453,
        arrays={
            "a": a,
            "b": b
        },
        params={
            "LEN_1D": LEN,
            "ITERATIONS": ITERS
        },
        sdfg_name="dace_s453",
        save_sdfgs=SAVE_SDFGS,
        apply_loop_to_map=True,
    )
    return a


def test_s471():
    LEN = 64
    ITERS = 2

    x = np.random.rand(LEN).astype(np.float64)
    b = np.random.rand(LEN).astype(np.float64)
    c = np.random.rand(LEN).astype(np.float64)
    d = np.random.rand(LEN).astype(np.float64)
    e = np.random.rand(LEN).astype(np.float64)

    compare_kernel(
        dace_s471,
        {
            "x": x,
            "b": b,
            "c": c,
            "d": d,
            "e": e
        },
        {
            "LEN_1D": LEN,
            "ITERATIONS": ITERS
        },
    )

    run_vectorization_test(
        dace_func=dace_s471,
        arrays={
            "x": x,
            "b": b,
            "c": c,
            "d": d,
            "e": e
        },
        params={
            "LEN_1D": LEN,
            "ITERATIONS": ITERS
        },
        sdfg_name="dace_s471",
        save_sdfgs=SAVE_SDFGS,
        apply_loop_to_map=True,
    )
    return x


def test_s481():
    LEN = 64
    ITERS = 2

    a = np.random.rand(LEN).astype(np.float64)
    b = np.random.rand(LEN).astype(np.float64)
    c = np.random.rand(LEN).astype(np.float64)
    d = np.random.rand(LEN).astype(np.float64)

    compare_kernel(
        dace_s481,
        {
            "a": a,
            "b": b,
            "c": c,
            "d": d
        },
        {
            "LEN_1D": LEN,
            "ITERATIONS": ITERS
        },
    )

    run_vectorization_test(
        dace_func=dace_s481,
        arrays={
            "a": a,
            "b": b,
            "c": c,
            "d": d
        },
        params={
            "LEN_1D": LEN,
            "ITERATIONS": ITERS
        },
        sdfg_name="dace_s481",
        save_sdfgs=SAVE_SDFGS,
        apply_loop_to_map=True,
    )
    return a


def test_s482():
    LEN = 64
    ITERS = 2

    a = np.random.rand(LEN).astype(np.float64)
    b = np.random.rand(LEN).astype(np.float64)
    c = np.random.rand(LEN).astype(np.float64)

    compare_kernel(
        dace_s482,
        {
            "a": a,
            "b": b,
            "c": c
        },
        {
            "LEN_1D": LEN,
            "ITERATIONS": ITERS
        },
    )

    run_vectorization_test(
        dace_func=dace_s482,
        arrays={
            "a": a,
            "b": b,
            "c": c
        },
        params={
            "LEN_1D": LEN,
            "ITERATIONS": ITERS
        },
        sdfg_name="dace_s482",
        save_sdfgs=SAVE_SDFGS,
        apply_loop_to_map=True,
    )
    return a


def test_s491():
    LEN = 64
    ITERS = 2

    a = np.random.rand(LEN).astype(np.float64)
    b = np.random.rand(LEN).astype(np.float64)
    c = np.random.rand(LEN).astype(np.float64)
    d = np.random.rand(LEN).astype(np.float64)
    ip = np.random.randint(0, LEN, size=LEN).astype(np.int32)

    compare_kernel(
        dace_s491,
        {
            "a": a,
            "b": b,
            "c": c,
            "d": d,
            "ip": ip
        },
        {
            "LEN_1D": LEN,
            "ITERATIONS": ITERS
        },
    )

    run_vectorization_test(
        dace_func=dace_s491,
        arrays={
            "a": a,
            "b": b,
            "c": c,
            "d": d,
            "ip": ip
        },
        params={
            "LEN_1D": LEN,
            "ITERATIONS": ITERS
        },
        sdfg_name="dace_s491",
        save_sdfgs=SAVE_SDFGS,
        apply_loop_to_map=True,
    )
    return a


def test_s31111():
    # Length and iteration parameters
    LEN_1D_val = 32
    ITERATIONS_val = 10

    # Initialize input array
    a = np.random.rand(LEN_1D_val).astype(np.float64)

    # Check against the reference C++ kernel
    compare_kernel(
        dace_s31111,
        {"a": a},
        {
            "LEN_1D": LEN_1D_val,
            "ITERATIONS": ITERATIONS_val
        },
    )

    # Run vectorization test
    run_vectorization_test(
        dace_func=dace_s31111,
        arrays={"a": a},
        params={
            "LEN_1D": LEN_1D_val,
            "ITERATIONS": ITERATIONS_val
        },
        sdfg_name="dace_s31111",
        save_sdfgs=SAVE_SDFGS,
        apply_loop_to_map=True,
    )

    # Return the (unchanged) array for consistency
    return a
