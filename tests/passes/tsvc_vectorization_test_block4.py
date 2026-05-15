# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
TSVC ``_d_single`` kernels imported in bulk from VectraArtifacts, block 4.

The remaining 25 ``_d_single`` (no-repetition-loop) kernels that the
simple block1-3 harness could not drive: they take int scalar params
(``n1``/``n3``/``inc``/``M``/``k``/``threshold``/``j``), ``int32``
index/gather arrays (``ip``/``indx``), 2D arrays (``LEN_2D``), or
reduction result scalars (``result``/``sum_out``).

This file uses a spec-driven harness: every arg is allocated by kind
with safe in-bounds data (index arrays are the identity ``arange`` so
the gather/scatter stays in range and the scalar reference is
deterministic). Each kernel is parametrised over the same
``remainder_strategy`` x ``branch_mode`` matrix as block1; LEN_1D in
``{64, 65}`` (1D kernels) and LEN_2D fixed at 16 (2D kernels, also
satisfies the ``LEN_2D // VLEN`` loops with VLEN=8).

Kernels that the vectorizer cannot yet handle raise
``NotImplementedError`` and are ``pytest.skip``-ped (same contract as
block1-3) — importing them documents the coverage gap rather than
hiding it. Reference correctness is pinned against the unvectorized
SDFG.
"""
import copy

import dace
import numpy as np
import pytest

from dace.transformation.interstate import LoopToMap
from dace.transformation.passes.vectorization.vectorize_cpu import VectorizeCPU
from tests.passes._tsvc_harness_helper import build_tsvc_matrix

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
def s122_d_single(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D], n1: dace.int64, n3: dace.int64):
    j = 1
    k = 0
    for i in range(n1 - 1, LEN_1D, n3):
        k = k + j
        a[i] = a[i] + b[LEN_1D - k]


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
def s4115_d_single(
    a: dace.float64[LEN_1D],
    b: dace.float64[LEN_1D],
    ip: dace.int32[LEN_1D],
    sum_out: dace.float64[1],
):
    sum_val = 0.0
    sum_val = 0.0
    for i in range(LEN_1D):
        sum_val = sum_val + a[i] * b[ip[i]]
    sum_out[0] = sum_val


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
def s491_d_single(
    a: dace.float64[LEN_1D],
    b: dace.float64[LEN_1D],
    c: dace.float64[LEN_1D],
    d: dace.float64[LEN_1D],
    ip: dace.int32[LEN_1D],
):
    for i in range(LEN_1D):
        a[ip[i]] = b[i] + c[i] * d[i]


@dace.program
def vag_d_single(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D], ip: dace.int32[LEN_1D]):
    for i in range(LEN_1D):
        a[i] = b[ip[i]]


@dace.program
def vas_d_single(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D], ip: dace.int32[LEN_1D]):
    for i in range(LEN_1D):
        a[ip[i]] = b[i]


# (kernel, {argname: kind}, {scalar_param: value}) — N4 = LEN//4
_KERNELS = [
    (s114_d_single, {
        "aa": "F2",
        "bb": "F2"
    }, {}),
    (s122_d_single, {
        "a": "F1",
        "b": "F1"
    }, {
        "n1": 1,
        "n3": 2
    }),
    (s1232_d_single, {
        "aa": "F2",
        "bb": "F2",
        "cc": "F2"
    }, {}),
    (s162_d_single, {
        "a": "F1",
        "b": "F1",
        "c": "F1"
    }, {
        "k": 3
    }),
    (s171_d_single, {
        "a": "F1",
        "b": "F1"
    }, {
        "inc": 1
    }),
    (s172_d_single, {
        "a": "F1",
        "b": "F1"
    }, {
        "n1": 1,
        "n3": 2
    }),
    (s174_d_single, {
        "a": "F1",
        "b": "F1"
    }, {
        "M": "N4"
    }),
    (s175_d_single, {
        "a": "F1",
        "b": "F1"
    }, {
        "inc": 2
    }),
    (s258_d_single, {
        "a": "F2v",
        "b": "F2v",
        "c": "F2v",
        "d": "F2v",
        "e": "F2v",
        "aa": "F1L2"
    }, {}),
    (s272_d_single, {
        "a": "F1",
        "b": "F1",
        "c": "F1",
        "d": "F1",
        "e": "F1"
    }, {
        "threshold": 0
    }),
    (s3110_d_single, {
        "aa": "F2",
        "bb": "F22"
    }, {}),
    (s318_d_single, {
        "a": "F1",
        "result": "R1"
    }, {
        "inc": 1
    }),
    (s332_d_single, {
        "a": "F1",
        "result": "R1"
    }, {
        "threshold": 0
    }),
    (s353_d_single, {
        "a": "F1",
        "b": "F1",
        "c": "F1",
        "ip": "I1"
    }, {}),
    (s4113_d_single, {
        "a": "F1",
        "b": "F1",
        "c": "F1",
        "ip": "I1"
    }, {}),
    (s4114_d_single, {
        "a": "F1",
        "b": "F1",
        "c": "F1",
        "d_": "F1",
        "ip": "I1"
    }, {
        "n1": 1
    }),
    (s4115_d_single, {
        "a": "F1",
        "b": "F1",
        "ip": "I1",
        "sum_out": "R1"
    }, {}),
    (s4116_d_single, {
        "a": "F1",
        "aa": "F2",
        "ip": "I2",
        "sum_out": "R1"
    }, {
        "j": 1,
        "inc": 1
    }),
    (s442_d_single, {
        "a": "F1",
        "b": "F1",
        "c": "F1",
        "d": "F1",
        "e": "F1",
        "indx": "I1"
    }, {}),
    (s481_d_single, {
        "a": "F1",
        "b": "F1",
        "c": "F1",
        "d": "F1"
    }, {}),
    (s482_d_single, {
        "a": "F1",
        "b": "F1",
        "c": "F1"
    }, {}),
    (s491_d_single, {
        "a": "F1",
        "b": "F1",
        "c": "F1",
        "d": "F1",
        "ip": "I1"
    }, {}),
    (vag_d_single, {
        "a": "F1",
        "b": "F1",
        "ip": "I1"
    }, {}),
    (vas_d_single, {
        "a": "F1",
        "b": "F1",
        "ip": "I1"
    }, {}),
]


def _needs_2d(argspec) -> bool:
    return any(v in ("F2", "F2v", "F1L2", "F22", "I2") for v in argspec.values())


def _alloc(kind: str, L1: int, L2: int, rng):
    """Allocate one argument by kind with safe in-bounds data.

    :param kind: arg-kind code (see module docstring / _KERNELS).
    :param L1: LEN_1D value.
    :param L2: LEN_2D value.
    :param rng: seeded numpy Generator (ref/vec share the same data).
    :returns: the ndarray for this argument.
    """
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


_MATRIX, _IDS = build_tsvc_matrix(_KERNELS, (64, 65))


@pytest.mark.parametrize("kernel,argspec,params,remainder_strategy,branch_mode,len_1d_val", _MATRIX, ids=_IDS)
def test_tsvc_block4(kernel, argspec, params, remainder_strategy, branch_mode, len_1d_val):

    L1 = len_1d_val
    L2 = _LEN_2D_VAL
    rng = np.random.default_rng(seed=L1)
    arrays_ref = {name: _alloc(kind, L1, L2, rng) for name, kind in argspec.items()}
    arrays_vec = {name: arr.copy() for name, arr in arrays_ref.items()}

    # Resolve scalar params (``N4`` -> LEN//4 keeps shifted writes in bounds).
    scalar_params = {pn: (L1 // 4 if pv == "N4" else pv) for pn, pv in params.items()}

    # Symbols the kernel actually needs (passing an unused symbol errors).
    symbols = {}
    if any(k in ("F1", "I1") for k in argspec.values()):
        symbols["LEN_1D"] = L1
    if _needs_2d(argspec):
        symbols["LEN_2D"] = L2

    sdfg_name = f"{kernel.name}_b4_{branch_mode}_{remainder_strategy}_{L1}"
    # Isolate each parametrized variant from the @dace.program SDFG
    # cache: to_sdfg() can return a shared cached SDFG that a prior
    # variant already mutated in place (simplify/LoopToMap), so deep-
    # copy before any mutation. Variant tag is the name suffix.
    import copy as _copy
    sdfg = _copy.deepcopy(kernel.to_sdfg(simplify=False))
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
            continue  # index inputs are not mutated
        diff = np.max(np.abs(arrays_ref[name] - arrays_vec[name]))
        assert diff < 1e-10, f"{kernel.name}/{name}: max abs diff = {diff}"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
