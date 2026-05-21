# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""TSVC 2D kernels with the kind-code arg encoding (former block4, 2D subset): genuine 2D-array kernels (F2/F1L2/F22)."""
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
def s114_d_single(aa: dace.float64[LEN_2D, LEN_2D], bb: dace.float64[LEN_2D, LEN_2D]):
    for i in range(LEN_2D // VLEN):
        for j in range(i * VLEN):
            aa[i, j] = aa[j, i] + bb[i, j]

@dace.program
def s1232_d_single(
    aa: dace.float64[LEN_2D, LEN_2D],
    bb: dace.float64[LEN_2D, LEN_2D],
    cc: dace.float64[LEN_2D, LEN_2D],
):
    for j in range(LEN_2D):
        for i in range(j * VLEN, LEN_2D):
            aa[i, j] = bb[i, j] + cc[i, j]

@dace.program
def s258_d_single(
    a: dace.float64[LEN_2D],
    b: dace.float64[LEN_2D],
    c: dace.float64[LEN_2D],
    d: dace.float64[LEN_2D],
    e: dace.float64[LEN_2D],
    aa: dace.float64[1, LEN_2D],
):
    s = 0.0
    for i in range(LEN_2D):
        if a[i] > 0.0:
            s = d[i] * d[i]
        b[i] = s * c[i] + d[i]
        e[i] = (s + 1.0) * aa[0, i]

@dace.program
def s3110_d_single(aa: dace.float64[LEN_2D, LEN_2D], bb: dace.float64[2, 2]):
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
    bb[0, 0] = chksum

@dace.program
def s4116_d_single(
    a: dace.float64[LEN_1D],
    aa: dace.float64[LEN_2D, LEN_2D],
    ip: dace.int32[LEN_2D],
    j: dace.int32,
    inc: dace.int32,
    sum_out: dace.float64[1],
):
    sum_val = 0.0
    sum_val = 0.0
    for i in range(LEN_2D - 1):
        off = inc + i
        sum_val = sum_val + a[off] * aa[j - 1, ip[i]]
    sum_out[0] = sum_val


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
    (s114_d_single, {'aa': 'F2', 'bb': 'F2'}, {}),
    (s1232_d_single, {'aa': 'F2', 'bb': 'F2', 'cc': 'F2'}, {}),
    (s258_d_single, {'a': 'F2v', 'b': 'F2v', 'c': 'F2v', 'd': 'F2v', 'e': 'F2v', 'aa': 'F1L2'}, {}),
    (s3110_d_single, {'aa': 'F2', 'bb': 'F22'}, {}),
    (s4116_d_single, {'a': 'F1', 'aa': 'F2', 'ip': 'I2', 'sum_out': 'R1'}, {'j': 1, 'inc': 1}),
]


_MATRIX, _IDS = build_tsvc_matrix(_KERNELS, (64, 65))


@pytest.mark.parametrize("kernel,argspec,params,remainder_strategy,branch_mode,len_1d_val", _MATRIX, ids=_IDS)
def test_tsvc_2d_misc(kernel, argspec, params, remainder_strategy, branch_mode, len_1d_val):
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

    sdfg_name = f"{kernel.name}_2dm_{branch_mode}_{remainder_strategy}_{L1}"
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
