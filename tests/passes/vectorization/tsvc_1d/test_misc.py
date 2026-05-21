# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""TSVC 1D indirect/gather + miscellaneous kernels (former block4, 1D subset). Curated duplicates (s4115, s491) live in test_selected."""
import copy

import dace
import numpy as np
import pytest

from dace.transformation.interstate import LoopToMap
from dace.transformation.passes.vectorization.vectorize_cpu import VectorizeCPU
from tests.passes.vectorization.helpers.tsvc_matrix import build_tsvc_matrix

LEN_1D = dace.symbol("LEN_1D")
LEN_2D = dace.symbol("LEN_2D")

VLEN = 8
_LEN_2D_VAL = 16


@dace.program
def s122_d_single(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D], n1: dace.int64, n3: dace.int64):
    j = 1
    k = 0
    for i in range(n1 - 1, LEN_1D, n3):
        k = k + j
        a[i] = a[i] + b[LEN_1D - k]

@dace.program
def s162_d_single(
    a: dace.float64[LEN_1D],
    b: dace.float64[LEN_1D],
    c: dace.float64[LEN_1D],
    k: dace.int64,
):
    if k > 0:
        for i in range(0, LEN_1D - k):
            a[i] = a[i + k] + b[i] * c[i]

@dace.program
def s171_d_single(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D], inc: dace.int64):
    for i in range(LEN_1D):
        a[i * inc] = a[i * inc] + b[i]

@dace.program
def s172_d_single(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D], n1: dace.int64, n3: dace.int64):
    for i in range(n1 - 1, LEN_1D, n3):
        a[i] = a[i] + b[i]

@dace.program
def s174_d_single(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D], M: dace.int64):
    for i in range(M):
        a[i + M] = a[i] + b[i]

@dace.program
def s175_d_single(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D], inc: dace.int64):
    for i in range(0, LEN_1D - inc, inc):
        a[i] = a[i + inc] + b[i]

@dace.program
def s272_d_single(
    a: dace.float64[LEN_1D],
    b: dace.float64[LEN_1D],
    c: dace.float64[LEN_1D],
    d: dace.float64[LEN_1D],
    e: dace.float64[LEN_1D],
    threshold: dace.int64,
):
    for i in range(LEN_1D):
        if e[i] >= threshold:
            a[i] = a[i] + c[i] * d[i]
            b[i] = b[i] + c[i] * c[i]

@dace.program
def s318_d_single(a: dace.float64[LEN_1D], result: dace.float64[1], inc: dace.int32):
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
    result[0] = maxv + float(index)

@dace.program
def s332_d_single(a: dace.float64[LEN_1D], result: dace.float64[1], threshold: dace.int64):
    index = -2
    value = -1.0
    for i in range(LEN_1D):
        if a[i] > threshold:
            index = i
            value = a[i]
            break
    result[0] = value + float(index)

@dace.program
def s353_d_single(
    a: dace.float64[LEN_1D],
    b: dace.float64[LEN_1D],
    c: dace.float64[LEN_1D],
    ip: dace.int32[LEN_1D],
):
    alpha = c[0]
    for i in range(0, LEN_1D, 4):
        a[i] = a[i] + alpha * b[ip[i]]
        a[i + 1] = a[i + 1] + alpha * b[ip[i + 1]]
        a[i + 2] = a[i + 2] + alpha * b[ip[i + 2]]
        a[i + 3] = a[i + 3] + alpha * b[ip[i + 3]]

@dace.program
def s4113_d_single(
    a: dace.float64[LEN_1D],
    b: dace.float64[LEN_1D],
    c: dace.float64[LEN_1D],
    ip: dace.int32[LEN_1D],
):
    for i in range(LEN_1D):
        a[ip[i]] = b[ip[i]] + c[i]

@dace.program
def s4114_d_single(
    a: dace.float64[LEN_1D],
    b: dace.float64[LEN_1D],
    c: dace.float64[LEN_1D],
    d_: dace.float64[LEN_1D],
    ip: dace.int32[LEN_1D],
    n1: dace.int32,
):
    for i in range(n1 - 1, LEN_1D):
        k = ip[i]
        a[i] = b[i] + c[LEN_1D - k - 1] * d_[i]

@dace.program
def s442_d_single(
    a: dace.float64[LEN_1D],
    b: dace.float64[LEN_1D],
    c: dace.float64[LEN_1D],
    d: dace.float64[LEN_1D],
    e: dace.float64[LEN_1D],
    indx: dace.int32[LEN_1D],
):
    for i in range(LEN_1D):
        if indx[i] == 1:
            a[i] = a[i] + (b[i] * b[i])
        elif indx[i] == 2:
            a[i] = a[i] + (c[i] * c[i])
        elif indx[i] == 3:
            a[i] = a[i] + (d[i] * d[i])
        elif indx[i] == 4:
            a[i] = a[i] + (e[i] * e[i])

@dace.program
def s481_d_single(
    a: dace.float64[LEN_1D],
    b: dace.float64[LEN_1D],
    c: dace.float64[LEN_1D],
    d: dace.float64[LEN_1D],
):
    for i in range(LEN_1D):
        if d[i] < 0.0:
            break
        a[i] = a[i] + b[i] * c[i]

@dace.program
def s482_d_single(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D], c: dace.float64[LEN_1D]):
    for i in range(LEN_1D):
        a[i] = a[i] + b[i] * c[i]
        if c[i] > b[i]:
            break

@dace.program
def vag_d_single(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D], ip: dace.int32[LEN_1D]):
    for i in range(LEN_1D):
        a[i] = b[ip[i]]

@dace.program
def vas_d_single(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D], ip: dace.int32[LEN_1D]):
    for i in range(LEN_1D):
        a[ip[i]] = b[i]


def _needs_2d(argspec) -> bool:
    return any(v in ("F2", "F2v", "F1L2", "F22", "I2") for v in argspec.values())


def _alloc(kind, L1, L2, rng):
    """Allocate one argument by kind code with safe in-bounds data."""
    if kind == "F1":
        return rng.random(L1).astype(np.float64)
    if kind == "F2":
        return rng.random((L2, L2)).astype(np.float64)
    if kind == "F2v":
        return rng.random(L2).astype(np.float64)
    if kind == "F1L2":
        return rng.random((1, L2)).astype(np.float64)
    if kind == "F22":
        return rng.random((2, 2)).astype(np.float64)
    if kind == "R1":
        return np.zeros(1, dtype=np.float64)
    if kind == "I1":
        return np.arange(L1, dtype=np.int32)
    if kind == "I2":
        return np.arange(L2, dtype=np.int32)
    raise AssertionError(f"unknown arg kind {kind!r}")


_KERNELS = [
    (s122_d_single, {'a': 'F1', 'b': 'F1'}, {'n1': 1, 'n3': 2}),
    (s162_d_single, {'a': 'F1', 'b': 'F1', 'c': 'F1'}, {'k': 3}),
    (s171_d_single, {'a': 'F1', 'b': 'F1'}, {'inc': 1}),
    (s172_d_single, {'a': 'F1', 'b': 'F1'}, {'n1': 1, 'n3': 2}),
    (s174_d_single, {'a': 'F1', 'b': 'F1'}, {'M': 'N4'}),
    (s175_d_single, {'a': 'F1', 'b': 'F1'}, {'inc': 2}),
    (s272_d_single, {'a': 'F1', 'b': 'F1', 'c': 'F1', 'd': 'F1', 'e': 'F1'}, {'threshold': 0}),
    (s318_d_single, {'a': 'F1', 'result': 'R1'}, {'inc': 1}),
    (s332_d_single, {'a': 'F1', 'result': 'R1'}, {'threshold': 0}),
    (s353_d_single, {'a': 'F1', 'b': 'F1', 'c': 'F1', 'ip': 'I1'}, {}),
    (s4113_d_single, {'a': 'F1', 'b': 'F1', 'c': 'F1', 'ip': 'I1'}, {}),
    (s4114_d_single, {'a': 'F1', 'b': 'F1', 'c': 'F1', 'd_': 'F1', 'ip': 'I1'}, {'n1': 1}),
    (s442_d_single, {'a': 'F1', 'b': 'F1', 'c': 'F1', 'd': 'F1', 'e': 'F1', 'indx': 'I1'}, {}),
    (s481_d_single, {'a': 'F1', 'b': 'F1', 'c': 'F1', 'd': 'F1'}, {}),
    (s482_d_single, {'a': 'F1', 'b': 'F1', 'c': 'F1'}, {}),
    (vag_d_single, {'a': 'F1', 'b': 'F1', 'ip': 'I1'}, {}),
    (vas_d_single, {'a': 'F1', 'b': 'F1', 'ip': 'I1'}, {}),
]


_MATRIX, _IDS = build_tsvc_matrix(_KERNELS, (64, 65))


@pytest.mark.parametrize("kernel,argspec,params,remainder_strategy,branch_mode,len_1d_val", _MATRIX, ids=_IDS)
def test_tsvc_1d_misc(kernel, argspec, params, remainder_strategy, branch_mode, len_1d_val):
    L1 = len_1d_val
    L2 = _LEN_2D_VAL
    rng = np.random.default_rng(seed=L1)
    arrays_ref = {name: _alloc(kind, L1, L2, rng) for name, kind in argspec.items()}
    arrays_vec = {name: arr.copy() for name, arr in arrays_ref.items()}

    scalar_params = {pn: (L1 // 4 if pv == "N4" else pv) for pn, pv in params.items()}
    symbols = {}
    if any(k in ("F1", "I1") for k in argspec.values()):
        symbols["LEN_1D"] = L1
    if _needs_2d(argspec):
        symbols["LEN_2D"] = L2

    sdfg_name = f"{kernel.name}_1dm_{branch_mode}_{remainder_strategy}_{L1}"
    sdfg = copy.deepcopy(kernel.to_sdfg(simplify=False))
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

    c_ref(**arrays_ref, **scalar_params, **symbols)
    c_vec(**arrays_vec, **scalar_params, **symbols)

    for name, kind in argspec.items():
        if kind in ("I1", "I2"):
            continue
        diff = np.max(np.abs(arrays_ref[name] - arrays_vec[name]))
        assert diff < 1e-10, f"{kernel.name}/{name}: max abs diff = {diff}"
