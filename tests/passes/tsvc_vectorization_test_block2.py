# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
TSVC ``_d_single`` kernels imported in bulk from VectraArtifacts, block 2.

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
from tests.passes._tsvc_harness_helper import build_tsvc_matrix

LEN_1D = dace.symbol("LEN_1D")

@dace.program
def s271_d_single(
    a: dace.float64[LEN_1D], b: dace.float64[LEN_1D], c: dace.float64[LEN_1D]
):
    for i in range(LEN_1D):
        if b[i] > 0.0:
            a[i] = a[i] + b[i] * c[i]


@dace.program
def s2710_d_single(
    a: dace.float64[LEN_1D],
    b: dace.float64[LEN_1D],
    c: dace.float64[LEN_1D],
    d: dace.float64[LEN_1D],
    e: dace.float64[LEN_1D],
    x: dace.float64[LEN_1D],
):
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
def s2711_d_single(
    a: dace.float64[LEN_1D], b: dace.float64[LEN_1D], c: dace.float64[LEN_1D]
):
    for i in range(LEN_1D):
        if b[i] != 0.0:
            a[i] = a[i] + b[i] * c[i]


@dace.program
def s2712_d_single(
    a: dace.float64[LEN_1D], b: dace.float64[LEN_1D], c: dace.float64[LEN_1D]
):
    for i in range(LEN_1D):
        if a[i] > b[i]:
            a[i] = a[i] + b[i] * c[i]


@dace.program
def s273_d_single(
    a: dace.float64[LEN_1D],
    b: dace.float64[LEN_1D],
    c: dace.float64[LEN_1D],
    d: dace.float64[LEN_1D],
    e: dace.float64[LEN_1D],
):
    for i in range(LEN_1D):
        a[i] = a[i] + d[i] * e[i]
        if a[i] < 0.0:
            b[i] = b[i] + d[i] * e[i]
        c[i] = c[i] + a[i] * d[i]


@dace.program
def s274_d_single(
    a: dace.float64[LEN_1D],
    b: dace.float64[LEN_1D],
    c: dace.float64[LEN_1D],
    d: dace.float64[LEN_1D],
    e: dace.float64[LEN_1D],
):
    for i in range(LEN_1D):
        a[i] = c[i] + e[i] * d[i]
        if a[i] > 0.0:
            b[i] = a[i] + b[i]
        else:
            a[i] = d[i] * e[i]


@dace.program
def s276_d_single(
    a: dace.float64[LEN_1D],
    b: dace.float64[LEN_1D],
    c: dace.float64[LEN_1D],
    d: dace.float64[LEN_1D],
):
    mid = LEN_1D // 2
    for i in range(LEN_1D):
        if i + 1 < mid:
            a[i] = a[i] + b[i] * c[i]
        else:
            a[i] = a[i] + b[i] * d[i]


@dace.program
def s277_d_single(
    a: dace.float64[LEN_1D],
    b: dace.float64[LEN_1D],
    c: dace.float64[LEN_1D],
    d: dace.float64[LEN_1D],
    e: dace.float64[LEN_1D],
):
    for i in range(LEN_1D - 1):
        if a[i] < 0.0:
            if b[i] < 0.0:
                a[i] = a[i] + c[i] * d[i]
            b[i + 1] = c[i] + d[i] * e[i]


@dace.program
def s278_d_single(
    a: dace.float64[LEN_1D],
    b: dace.float64[LEN_1D],
    c: dace.float64[LEN_1D],
    d: dace.float64[LEN_1D],
    e: dace.float64[LEN_1D],
):
    for i in range(LEN_1D):
        if a[i] > 0.0:
            c[i] = -c[i] + d[i] * e[i]
        else:
            b[i] = -b[i] + d[i] * e[i]
        a[i] = b[i] + c[i] * d[i]


@dace.program
def s279_d_single(
    a: dace.float64[LEN_1D],
    b: dace.float64[LEN_1D],
    c: dace.float64[LEN_1D],
    d: dace.float64[LEN_1D],
    e: dace.float64[LEN_1D],
):
    for i in range(LEN_1D):
        if a[i] > 0.0:
            c[i] = -c[i] + e[i] * e[i]
        else:
            b[i] = -b[i] + d[i] * d[i]
            if b[i] > a[i]:
                c[i] = c[i] + d[i] * e[i]
        a[i] = b[i] + c[i] * d[i]


@dace.program
def s281_d_single(
    a: dace.float64[LEN_1D], b: dace.float64[LEN_1D], c: dace.float64[LEN_1D]
):
    for i in range(LEN_1D):
        x = a[LEN_1D - i - 1] + b[i] * c[i]
        a[i] = x - 1.0
        b[i] = x


@dace.program
def s291_d_single(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D]):
    a[0] = (b[0] + b[LEN_1D - 1]) * 0.5
    for i in range(1, LEN_1D):
        a[i] = (b[i] + b[i - 1]) * 0.5


@dace.program
def s292_d_single(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D]):
    a[0] = (b[0] + b[LEN_1D - 1] + b[LEN_1D - 2]) * 0.333
    a[1] = (b[1] + b[0] + b[LEN_1D - 1]) * 0.333
    for i in range(2, LEN_1D):
        a[i] = (b[i] + b[i - 1] + b[i - 2]) * 0.333


@dace.program
def s293_d_single(a: dace.float64[LEN_1D]):
    a0 = a[0]
    for i in range(LEN_1D):
        a[i] = a0


@dace.program
def s3111_d_single(a: dace.float64[LEN_1D], b: dace.float64[2]):
    sum_val = 0.0
    for i in range(LEN_1D):
        if a[i] > 0.0:
            sum_val = sum_val + a[i]
    b[0] = sum_val


@dace.program
def s31111_d_single(a: dace.float64[LEN_1D], b: dace.float64[2]):
    sum_val = 0.0
    for base in range(0, LEN_1D, 4):
        partial = 0.0
        partial = partial + a[base + 0]
        partial = partial + a[base + 1]
        partial = partial + a[base + 2]
        partial = partial + a[base + 3]
        sum_val = sum_val + partial
    b[0] = sum_val


@dace.program
def s3112_d_single(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D]):
    sum = 0.0
    for i in range(LEN_1D):
        sum = sum + a[i]
        b[i] = sum


@dace.program
def s3113_d_single(a: dace.float64[LEN_1D], b: dace.float64[2]):
    maxv = dace.float64(0)
    maxv = abs(a[0])
    for i in range(LEN_1D):
        av = abs(a[i])
        if av > maxv:
            maxv = av
    b[0] = maxv


@dace.program
def s313_d_single(
    a: dace.float64[LEN_1D], b: dace.float64[LEN_1D], dot: dace.float64[1]
):
    dot[0] = 0.0
    for i in range(LEN_1D):
        dot[0] = dot[0] + a[i] * b[i]


@dace.program
def s314_d_single(a: dace.float64[LEN_1D], result: dace.float64[1]):
    x = a[0]
    for i in range(1, LEN_1D):
        if a[i] > x:
            x = a[i]
    result[0] = x


@dace.program
def s315_d_single(a: dace.float64[LEN_1D], result: dace.float64[1]):
    for i in range(LEN_1D):
        a[i] = float((i * 7) % LEN_1D)
    x = a[0]
    index = 0
    for i in range(LEN_1D):
        if a[i] > x:
            x = a[i]
            index = i
    a[0] = x + float(index)
    result[0] = a[0]


@dace.program
def s316_d_single(a: dace.float64[LEN_1D], result: dace.float64[1]):
    x = a[0]
    for i in range(1, LEN_1D):
        if a[i] < x:
            x = a[i]
    result[0] = x


@dace.program
def s317_d_single(q: dace.float64[LEN_1D]):
    q[0] = 1.0
    for i in range(LEN_1D // 2):
        q[0] = q[0] * 0.99


@dace.program
def s319_d_single(
    a: dace.float64[LEN_1D],
    b: dace.float64[LEN_1D],
    c: dace.float64[LEN_1D],
    d: dace.float64[LEN_1D],
    e: dace.float64[LEN_1D],
):
    sum_val = 0.0
    for i in range(LEN_1D):
        a[i] = c[i] + d[i]
        sum_val = sum_val + a[i]
        b[i] = c[i] + e[i]
        sum_val = sum_val + b[i]
    b[0] = sum_val


@dace.program
def s321_d_single(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D]):
    for i in range(1, LEN_1D):
        a[i] = a[i] + a[i - 1] * b[i]


@dace.program
def s322_d_single(
    a: dace.float64[LEN_1D], b: dace.float64[LEN_1D], c: dace.float64[LEN_1D]
):
    for i in range(2, LEN_1D):
        a[i] = a[i] + a[i - 1] * b[i] + a[i - 2] * c[i]


@dace.program
def s323_d_single(
    a: dace.float64[LEN_1D],
    b: dace.float64[LEN_1D],
    c: dace.float64[LEN_1D],
    d: dace.float64[LEN_1D],
    e: dace.float64[LEN_1D],
):
    for i in range(1, LEN_1D):
        a[i] = b[i - 1] + c[i] * d[i]
        b[i] = a[i] + c[i] * e[i]


@dace.program
def s3251_d_single(
    a: dace.float64[LEN_1D],
    b: dace.float64[LEN_1D],
    c: dace.float64[LEN_1D],
    d: dace.float64[LEN_1D],
    e: dace.float64[LEN_1D],
):
    for i in range(LEN_1D - 1):
        a[i + 1] = b[i] + c[i]
        b[i] = c[i] * e[i]
        d[i] = a[i] * e[i]

_KERNELS = [
    (s271_d_single, ['a', 'b', 'c']),
    (s2710_d_single, ['a', 'b', 'c', 'd', 'e', 'x']),
    (s2711_d_single, ['a', 'b', 'c']),
    (s2712_d_single, ['a', 'b', 'c']),
    (s273_d_single, ['a', 'b', 'c', 'd', 'e']),
    (s274_d_single, ['a', 'b', 'c', 'd', 'e']),
    (s276_d_single, ['a', 'b', 'c', 'd']),
    (s277_d_single, ['a', 'b', 'c', 'd', 'e']),
    (s278_d_single, ['a', 'b', 'c', 'd', 'e']),
    (s279_d_single, ['a', 'b', 'c', 'd', 'e']),
    (s281_d_single, ['a', 'b', 'c']),
    (s291_d_single, ['a', 'b']),
    (s292_d_single, ['a', 'b']),
    (s293_d_single, ['a']),
    (s3111_d_single, ['a', 'b']),
    (s31111_d_single, ['a', 'b']),
    (s3112_d_single, ['a', 'b']),
    (s3113_d_single, ['a', 'b']),
    (s313_d_single, ['a', 'b', 'dot']),
    (s314_d_single, ['a', 'result']),
    (s315_d_single, ['a', 'result']),
    (s316_d_single, ['a', 'result']),
    (s317_d_single, ['q']),
    (s319_d_single, ['a', 'b', 'c', 'd', 'e']),
    (s321_d_single, ['a', 'b']),
    (s322_d_single, ['a', 'b', 'c']),
    (s323_d_single, ['a', 'b', 'c', 'd', 'e']),
    (s3251_d_single, ['a', 'b', 'c', 'd', 'e']),
]


def _allocate(name: str, n: int) -> np.ndarray:
    """Length-1 outputs are written via subset ``[0]`` on a full
    ``LEN_1D`` array; allocate full ``n`` so kernels using either
    ``out[0]`` or ``out[i]`` can do so without OOB."""
    if name in ("result", "sum_out", "dot_out", "max_out", "min_out", "prod"):
        return np.zeros(n, dtype=np.float64)
    return np.random.rand(n).astype(np.float64)




_MATRIX, _IDS = build_tsvc_matrix(_KERNELS, (64, 65))


@pytest.mark.parametrize("kernel,argnames,remainder_strategy,branch_mode,len_1d_val", _MATRIX, ids=_IDS)
def test_tsvc_block2(kernel, argnames, remainder_strategy, branch_mode, len_1d_val):

    arrays_ref = {name: _allocate(name, len_1d_val) for name in argnames}
    arrays_vec = {name: arr.copy() for name, arr in arrays_ref.items()}

    sdfg_name = f"{kernel.name}_b2_{branch_mode}_{remainder_strategy}_{len_1d_val}"
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

    c_ref(**arrays_ref, LEN_1D=len_1d_val)
    c_vec(**arrays_vec, LEN_1D=len_1d_val)

    for name in argnames:
        diff = np.max(np.abs(arrays_ref[name] - arrays_vec[name]))
        assert diff < 1e-10, f"{kernel.name}/{name}: max abs diff = {diff}"
