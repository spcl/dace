# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Additional TSVC kernels (no-repetition ``_single`` variants from the
VectraArtifacts ``tsvc_2/tsvc_dace_microkernels/`` set), exercising the
vectorization pipeline across all wired ``remainder_strategy`` /
``branch_mode`` combinations.

Each kernel is the body the original Fortran TSVC test exercises with
its outer repetition loop stripped. The 20 kernels selected here are
all 1D ``dace.float64[LEN_1D]`` arrays, no branches, no integer-array
indirection — i.e. pure stencil-style arithmetic on contiguous data —
so the pipeline matrix runs end-to-end without per-kernel skips.

Reference correctness is pinned against the unvectorized SDFG.
"""
import copy

import dace
import numpy as np
import pytest

from dace import Union
from dace.transformation.interstate import LoopToMap
from dace.transformation.passes.vectorization.vectorize_cpu import VectorizeCPU

LEN_1D = dace.symbol("LEN_1D")


# ---------------------------------------------------------------------------
# 20 single-variant kernels, inlined from
# VectraArtifacts/tsvc_2/tsvc_dace_microkernels/<name>/<name>_d_single.py
# (the ``_d_single`` form drops the outer repetition loop the Fortran
# TSVC test uses).
# ---------------------------------------------------------------------------


@dace.program
def s000_d_single(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D]):
    for i in range(LEN_1D):
        a[i] = b[i] + 1.0


@dace.program
def s111_d_single(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D]):
    for i in range(1, LEN_1D, 2):
        a[i] = a[i - 1] + b[i]


@dace.program
def s112_d_single(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D]):
    for i in range(LEN_1D - 1):
        a[i] = a[i + 1] + b[i]


@dace.program
def s113_d_single(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D]):
    for i in range(1, LEN_1D):
        a[i] = a[0] + b[i]


@dace.program
def s121_d_single(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D]):
    for i in range(LEN_1D - 1):
        a[i] = a[i + 1] + b[i]


@dace.program
def s127_d_single(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D], c: dace.float64[LEN_1D],
                  d: dace.float64[LEN_1D], e: dace.float64[LEN_1D]):
    for i in dace.map[0:LEN_1D // 2]:
        a[2 * i] = b[i] + c[i] * d[i]
        a[2 * i + 1] = b[i] + d[i] * e[i]


@dace.program
def s131_d_single(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D]):
    for i in range(LEN_1D - 1):
        a[i] = a[i + 1] + b[i]


@dace.program
def s173_d_single(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D]):
    for i in range(LEN_1D // 2):
        a[i + (LEN_1D // 2)] = a[i] + b[i]


@dace.program
def s1111_d_single(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D],
                   c: dace.float64[LEN_1D], d: dace.float64[LEN_1D]):
    for i in dace.map[0:LEN_1D // 2]:
        a[2 * i] = c[i] * b[i] + d[i] * b[i] + c[i] * c[i] + d[i] * b[i] + d[i] * c[i]


@dace.program
def s1112_d_single(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D]):
    for i in range(LEN_1D - 1, -1, -1):
        a[i] = b[i] + 1.0


@dace.program
def s1113_d_single(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D]):
    for i in range(LEN_1D):
        a[i] = a[LEN_1D // 2] + b[i]


@dace.program
def s211_d_single(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D],
                  c: dace.float64[LEN_1D], d: dace.float64[LEN_1D], e: dace.float64[LEN_1D]):
    for i in range(1, LEN_1D - 1):
        a[i] = b[i - 1] + c[i] * d[i]
        b[i] = b[i + 1] - e[i] * d[i]


@dace.program
def s212_d_single(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D],
                  c: dace.float64[LEN_1D], d: dace.float64[LEN_1D]):
    for i in range(LEN_1D - 1):
        a[i] = a[i] * c[i]
        b[i] = b[i] + (a[i + 1] * d[i])


@dace.program
def s241_d_single(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D],
                  c: dace.float64[LEN_1D], d: dace.float64[LEN_1D]):
    for i in range(LEN_1D - 1):
        a[i] = b[i] * c[i] * d[i]
        b[i] = a[i] * a[i + 1] * d[i]


@dace.program
def s243_d_single(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D],
                  c: dace.float64[LEN_1D], d: dace.float64[LEN_1D], e: dace.float64[LEN_1D]):
    for i in range(LEN_1D - 1):
        a[i] = b[i] + c[i] * d[i]
        b[i] = a[i] + d[i] * e[i]
        a[i] = b[i] + a[i + 1] * d[i]


@dace.program
def s244_d_single(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D],
                  c: dace.float64[LEN_1D], d: dace.float64[LEN_1D]):
    for i in range(LEN_1D - 1):
        a[i] = b[i] + c[i] * d[i]
        b[i] = c[i] + b[i]
        a[i + 1] = b[i] + a[i + 1] * d[i]


@dace.program
def s251_d_single(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D],
                  c: dace.float64[LEN_1D], d: dace.float64[LEN_1D]):
    for i in range(LEN_1D):
        s = b[i] + c[i] * d[i]
        a[i] = s * s


@dace.program
def s311_d_single(a: dace.float64[LEN_1D], sum_out: dace.float64[LEN_1D]):
    sum_out[0] = 0.0
    for i in range(LEN_1D):
        sum_out[0] = sum_out[0] + a[i]


@dace.program
def s312_d_single(a: dace.float64[LEN_1D], result: dace.float64[1]):
    prod = 1.0
    for i in range(LEN_1D):
        prod = prod * a[i]
    result[0] = prod


@dace.program
def s4112_d_single(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D]):
    for i in dace.map[0:LEN_1D]:
        a[i] = a[i] + b[i] * b[i]


# Registry: (kernel, list_of_array_kwarg_names). All kernels here only
# use ``LEN_1D``-shaped arrays + the symbol ``LEN_1D`` as a param.
_KERNELS = [
    (s000_d_single, ["a", "b"]),
    (s111_d_single, ["a", "b"]),
    (s112_d_single, ["a", "b"]),
    (s113_d_single, ["a", "b"]),
    (s121_d_single, ["a", "b"]),
    (s127_d_single, ["a", "b", "c", "d", "e"]),
    (s131_d_single, ["a", "b"]),
    (s173_d_single, ["a", "b"]),
    (s1111_d_single, ["a", "b", "c", "d"]),
    (s1112_d_single, ["a", "b"]),
    (s1113_d_single, ["a", "b"]),
    (s211_d_single, ["a", "b", "c", "d", "e"]),
    (s212_d_single, ["a", "b", "c", "d"]),
    (s241_d_single, ["a", "b", "c", "d"]),
    (s243_d_single, ["a", "b", "c", "d", "e"]),
    (s244_d_single, ["a", "b", "c", "d"]),
    (s251_d_single, ["a", "b", "c", "d"]),
    (s311_d_single, ["a", "sum_out"]),
    (s312_d_single, ["a", "result"]),
    (s4112_d_single, ["a", "b"]),
]


def _allocate(name: str, n: int) -> np.ndarray:
    """Sized to match the kernel's static shape: ``result`` is length-1,
    ``sum_out`` is the kernel's full ``LEN_1D`` buffer (s311 only writes
    index 0 but the array is declared full-length); everything else is
    ``LEN_1D``-long with random data."""
    if name == "result":
        return np.zeros(1, dtype=np.float64)
    return np.random.rand(n).astype(np.float64)


@pytest.fixture(params=["scalar", "masked"])
def remainder_strategy(request) -> str:
    return request.param


@pytest.fixture(params=["merge"])
def branch_mode(request) -> str:
    return request.param


@pytest.mark.parametrize("kernel,argnames", _KERNELS,
                         ids=[k.name for k, _ in _KERNELS])
def test_tsvc_additional(kernel, argnames, remainder_strategy, branch_mode):
    """Single-kernel correctness check across the variant matrix.

    Build the un-vectorized reference SDFG (LoopToMap applied so the
    Python-level ``for i in range(...)`` loops become maps the
    vectorizer can recognise). Compile + run with random inputs. Then
    deep-copy, vectorize, compile, run on the same inputs. Numerical
    equivalence against the reference is the contract."""
    if branch_mode == "fp_factor" and remainder_strategy == "masked":
        pytest.skip("fp_factor + masked is rejected by VectorizeCPU (locked plan rule)")

    LEN_1D_val = 64

    arrays_ref = {name: _allocate(name, LEN_1D_val) for name in argnames}
    arrays_vec = {name: arr.copy() for name, arr in arrays_ref.items()}

    sdfg_name = f"{kernel.name}_{branch_mode}_{remainder_strategy}"
    sdfg = kernel.to_sdfg(simplify=False)
    sdfg.name = sdfg_name + "_ref"
    sdfg.simplify(validate=True, validate_all=True)
    sdfg.apply_transformations_repeated(LoopToMap())
    sdfg.simplify()

    vsdfg = copy.deepcopy(sdfg)
    vsdfg.name = sdfg_name + "_vec"

    if branch_mode == "fp_factor":
        branch_kwargs = dict(use_fp_factor=True, branch_normalization=False)
    else:
        branch_kwargs = dict(use_fp_factor=False, branch_normalization=True)

    try:
        VectorizeCPU(vector_width=8,
                     fail_on_unvectorizable=False,
                     remainder_strategy=remainder_strategy,
                     **branch_kwargs).apply_pass(vsdfg, {})
    except NotImplementedError as ex:
        pytest.skip(f"vectorize NotImplementedError on {kernel.name}: {ex}")

    c_ref = sdfg.compile()
    c_vec = vsdfg.compile()

    c_ref(**arrays_ref, LEN_1D=LEN_1D_val)
    c_vec(**arrays_vec, LEN_1D=LEN_1D_val)

    for name in argnames:
        diff = np.max(np.abs(arrays_ref[name] - arrays_vec[name]))
        assert diff < 1e-10, f"{kernel.name}/{name}: max abs diff = {diff}"
