# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
TSVC ``_d_single`` kernels imported in bulk from VectraArtifacts, block 1.

Each kernel is parametrised across:
- ``LEN_1D`` in ``{64, 65}`` — 64 is divisible-by-W=8 (P2 proves
  divisibility and emits no remainder); 65 forces a non-empty
  remainder, exercising the ``scalar`` (step-1 postamble) and
  ``masked`` (W-wide iter_mask) paths end-to-end.
- ``remainder_strategy`` in ``{scalar, masked}`` — selects the
  remainder *shape*; P2 itself decides whether a remainder is needed
  via symbolic divisibility analysis.
- ``branch_mode`` in ``{merge, fp_factor}`` — ``fp_factor`` paired
  with ``masked`` is rejected by VectorizeCPU's locked plan rule and
  skipped.

Reference correctness pinned against the unvectorized SDFG; all
kernels are 1D ``dace.float64[LEN_1D]`` so the matrix is the same
shape for every entry.
"""
import copy

import dace
import numpy as np
import pytest

from dace.transformation.interstate import LoopToMap
from dace.transformation.passes.vectorization.vectorize_cpu import VectorizeCPU

LEN_1D = dace.symbol("LEN_1D")


@dace.program
def s116_d_single(a: dace.float64[LEN_1D]):
    for i in range(0, LEN_1D - 4, 4):
        a[i] = a[i + 1] * a[i]
        a[i + 1] = a[i + 2] * a[i + 1]
        a[i + 2] = a[i + 3] * a[i + 2]
        a[i + 3] = a[i + 4] * a[i + 3]


@dace.program
def s1161_d_single(
    a: dace.float64[LEN_1D],
    b: dace.float64[LEN_1D],
    c: dace.float64[LEN_1D],
    d: dace.float64[LEN_1D],
    e: dace.float64[LEN_1D],
):
    for i in range(LEN_1D):
        if c[i] < 0.0:
            b[i] = a[i] + d[i] * d[i]
        else:
            a[i] = c[i] + d[i] * e[i]


@dace.program
def s1213_d_single(
    a: dace.float64[LEN_1D],
    b: dace.float64[LEN_1D],
    c: dace.float64[LEN_1D],
    d: dace.float64[LEN_1D],
):
    for i in range(1, LEN_1D - 1):
        a[i] = b[i - 1] + c[i]
        b[i] = a[i + 1] * d[i]


@dace.program
def s1221_d_single(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D]):
    for i in range(4, LEN_1D):
        b[i] = b[i - 4] + a[i]


@dace.program
def s123_d_single(
    a: dace.float64[LEN_1D],
    b: dace.float64[LEN_1D],
    c: dace.float64[LEN_1D],
    d: dace.float64[LEN_1D],
    e: dace.float64[LEN_1D],
):
    j = -1
    for i in range(LEN_1D // 2):
        j = j + 1
        a[j] = b[i] + d[i] * e[i]
        if c[i] > 0.0:
            j = j + 1
            a[j] = c[i] + d[i] * e[i]


@dace.program
def s124_d_single(
    a: dace.float64[LEN_1D],
    b: dace.float64[LEN_1D],
    c: dace.float64[LEN_1D],
    d: dace.float64[LEN_1D],
    e: dace.float64[LEN_1D],
):
    j = -1
    for i in range(LEN_1D):
        if b[i] > 0.0:
            j = j + 1
            a[j] = b[i] + d[i] * e[i]
        else:
            j = j + 1
            a[j] = c[i] + d[i] * e[i]


@dace.program
def s1244_d_single(
    a: dace.float64[LEN_1D],
    b: dace.float64[LEN_1D],
    c: dace.float64[LEN_1D],
    d: dace.float64[LEN_1D],
):
    for i in range(LEN_1D - 1):
        a[i] = b[i] + c[i] * c[i] + b[i] * b[i] + c[i]
        d[i] = a[i] + a[i + 1]


@dace.program
def s1251_d_single(
    a: dace.float64[LEN_1D],
    b: dace.float64[LEN_1D],
    c: dace.float64[LEN_1D],
    d: dace.float64[LEN_1D],
    e: dace.float64[LEN_1D],
):
    for i in range(LEN_1D):
        s = b[i] + c[i]
        b[i] = a[i] + d[i]
        a[i] = s * e[i]


@dace.program
def s1279_d_single(
    a: dace.float64[LEN_1D],
    b: dace.float64[LEN_1D],
    c: dace.float64[LEN_1D],
    d: dace.float64[LEN_1D],
    e: dace.float64[LEN_1D],
):
    for i in range(LEN_1D):
        if a[i] < 0.0:
            if b[i] > a[i]:
                c[i] = c[i] + d[i] * e[i]


@dace.program
def s128_d_single(
    a: dace.float64[LEN_1D],
    b: dace.float64[LEN_1D],
    c: dace.float64[LEN_1D],
    d: dace.float64[LEN_1D],
):
    j = -1
    for i in range(LEN_1D // 2):
        k = j + 1
        a[i] = b[k] - d[i]
        j = k + 1
        b[k] = a[i] + c[k]


@dace.program
def s1281_d_single(
    a: dace.float64[LEN_1D],
    b: dace.float64[LEN_1D],
    c: dace.float64[LEN_1D],
    d: dace.float64[LEN_1D],
    e: dace.float64[LEN_1D],
):
    for i in range(LEN_1D):
        x = (b[i] * c[i]) + (a[i] * d[i]) + e[i]
        a[i] = x - 1.0
        b[i] = x


@dace.program
def s1421_d_single(b: dace.float64[LEN_1D], a: dace.float64[LEN_1D]):
    half = LEN_1D // 2
    for i in range(half):
        b[i] = b[half + i] + a[i]


@dace.program
def s151_d_single(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D]):
    for i in range(LEN_1D - 1):
        a[i] = a[i + 1] + b[i]


@dace.program
def s152_d_single(
    a: dace.float64[LEN_1D],
    b: dace.float64[LEN_1D],
    c: dace.float64[LEN_1D],
    d: dace.float64[LEN_1D],
    e: dace.float64[LEN_1D],
):
    for i in range(LEN_1D):
        b[i] = d[i] * e[i]
    for i in range(LEN_1D):
        a[i] = a[i] + b[i] * c[i]


@dace.program
def s161_d_single(
    a: dace.float64[LEN_1D],
    b: dace.float64[LEN_1D],
    c: dace.float64[LEN_1D],
    d: dace.float64[LEN_1D],
    e: dace.float64[LEN_1D],
):
    for i in range(LEN_1D):
        if b[i] < 0.0:
            c[i + 1] = a[i] + d[i] * d[i]
        else:
            a[i] = c[i] + d[i] * e[i]


@dace.program
def s176_d_single(
    a: dace.float64[LEN_1D], b: dace.float64[LEN_1D], c: dace.float64[LEN_1D]
):
    m = LEN_1D // 2
    for j in range(LEN_1D // 2):
        for i in range(m):
            a[i] = a[i] + b[i + m - j - 1] * c[j]


@dace.program
def s221_d_single(
    a: dace.float64[LEN_1D],
    b: dace.float64[LEN_1D],
    c: dace.float64[LEN_1D],
    d: dace.float64[LEN_1D],
):
    for i in range(1, LEN_1D):
        a[i] = a[i] + c[i] * d[i]
        b[i] = b[i - 1] + a[i] + d[i]


@dace.program
def s222_d_single(
    a: dace.float64[LEN_1D],
    b: dace.float64[LEN_1D],
    c: dace.float64[LEN_1D],
    e: dace.float64[LEN_1D],
):
    for i in range(1, LEN_1D):
        a[i] = a[i] + b[i] * c[i]
        e[i] = e[i - 1] * e[i - 1]
        a[i] = a[i] - b[i] * c[i]


@dace.program
def s2244_d_single(
    a: dace.float64[LEN_1D],
    b: dace.float64[LEN_1D],
    c: dace.float64[LEN_1D],
    e: dace.float64[LEN_1D],
):
    a[LEN_1D - 1] = b[LEN_1D - 2] + e[LEN_1D - 2]
    for i in range(LEN_1D - 1):
        a[i] = b[i] + c[i]


@dace.program
def s2251_d_single(
    a: dace.float64[LEN_1D],
    b: dace.float64[LEN_1D],
    c: dace.float64[LEN_1D],
    d: dace.float64[LEN_1D],
    e: dace.float64[LEN_1D],
):
    s = 0.0
    for i in range(LEN_1D):
        a[i] = s * e[i]
        s = b[i] + c[i]
        b[i] = a[i] + d[i]


@dace.program
def s242_d_single(
    a: dace.float64[LEN_1D],
    b: dace.float64[LEN_1D],
    c: dace.float64[LEN_1D],
    d: dace.float64[LEN_1D],
):
    for i in range(1, LEN_1D):
        a[i] = a[i - 1] + 0.5 + 1.0 + b[i] + c[i] + d[i]


@dace.program
def s253_d_single(
    a: dace.float64[LEN_1D],
    b: dace.float64[LEN_1D],
    c: dace.float64[LEN_1D],
    d: dace.float64[LEN_1D],
):
    for i in range(LEN_1D):
        if a[i] > b[i]:
            s = a[i] - b[i] * d[i]
            c[i] = c[i] + s
            a[i] = s


@dace.program
def s254_d_single(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D]):
    x = b[LEN_1D - 1]
    for i in range(LEN_1D):
        a[i] = (b[i] + x) * 0.5
        x = b[i]


@dace.program
def s255_d_single(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D]):
    x = b[LEN_1D - 1]
    y = b[LEN_1D - 2]
    for i in range(LEN_1D):
        a[i] = (b[i] + x + y) * 0.333
        y = x
        x = b[i]


@dace.program
def s261_d_single(
    a: dace.float64[LEN_1D],
    b: dace.float64[LEN_1D],
    c: dace.float64[LEN_1D],
    d: dace.float64[LEN_1D],
):
    for i in range(1, LEN_1D):
        t = a[i] + b[i]
        a[i] = t + c[i - 1]
        c[i] = c[i] * d[i]

_KERNELS = [
    (s116_d_single, ['a']),
    (s1161_d_single, ['a', 'b', 'c', 'd', 'e']),
    (s1213_d_single, ['a', 'b', 'c', 'd']),
    (s1221_d_single, ['a', 'b']),
    (s123_d_single, ['a', 'b', 'c', 'd', 'e']),
    (s124_d_single, ['a', 'b', 'c', 'd', 'e']),
    (s1244_d_single, ['a', 'b', 'c', 'd']),
    (s1251_d_single, ['a', 'b', 'c', 'd', 'e']),
    (s1279_d_single, ['a', 'b', 'c', 'd', 'e']),
    (s128_d_single, ['a', 'b', 'c', 'd']),
    (s1281_d_single, ['a', 'b', 'c', 'd', 'e']),
    (s1421_d_single, ['b', 'a']),
    (s151_d_single, ['a', 'b']),
    (s152_d_single, ['a', 'b', 'c', 'd', 'e']),
    (s161_d_single, ['a', 'b', 'c', 'd', 'e']),
    (s176_d_single, ['a', 'b', 'c']),
    (s221_d_single, ['a', 'b', 'c', 'd']),
    (s222_d_single, ['a', 'b', 'c', 'e']),
    (s2244_d_single, ['a', 'b', 'c', 'e']),
    (s2251_d_single, ['a', 'b', 'c', 'd', 'e']),
    (s242_d_single, ['a', 'b', 'c', 'd']),
    (s253_d_single, ['a', 'b', 'c', 'd']),
    (s254_d_single, ['a', 'b']),
    (s255_d_single, ['a', 'b']),
    (s261_d_single, ['a', 'b', 'c', 'd']),
]


def _allocate(name: str, n: int) -> np.ndarray:
    """Length-1 outputs are written via subset ``[0]`` on a full
    ``LEN_1D`` array; allocate full ``n`` so kernels using either
    ``out[0]`` or ``out[i]`` can do so without OOB."""
    if name in ("result", "sum_out", "dot_out", "max_out", "min_out", "prod"):
        return np.zeros(n, dtype=np.float64)
    return np.random.rand(n).astype(np.float64)


@pytest.fixture(params=["scalar", "masked"])
def remainder_strategy(request) -> str:
    return request.param


@pytest.fixture(params=["merge", "fp_factor"])
def branch_mode(request) -> str:
    return request.param


@pytest.fixture(params=[64, 65])
def len_1d_val(request) -> int:
    return request.param


@pytest.mark.parametrize("kernel,argnames", _KERNELS, ids=[k.name for k, _ in _KERNELS])
def test_tsvc_block1(kernel, argnames, remainder_strategy, branch_mode, len_1d_val):
    # Locked-plan-rule skip: fp_factor cannot combine with masked iter_mask.
    if branch_mode == "fp_factor" and remainder_strategy == "masked":
        pytest.skip("fp_factor + masked rejected by VectorizeCPU (locked plan rule)")

    arrays_ref = {name: _allocate(name, len_1d_val) for name in argnames}
    arrays_vec = {name: arr.copy() for name, arr in arrays_ref.items()}

    sdfg_name = f"{kernel.name}_b1_{branch_mode}_{remainder_strategy}_{len_1d_val}"
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

    c_ref(**arrays_ref, LEN_1D=len_1d_val)
    c_vec(**arrays_vec, LEN_1D=len_1d_val)

    for name in argnames:
        diff = np.max(np.abs(arrays_ref[name] - arrays_vec[name]))
        assert diff < 1e-10, f"{kernel.name}/{name}: max abs diff = {diff}"
