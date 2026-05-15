# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
TSVC ``_d_single`` kernels imported in bulk from VectraArtifacts, block 3.

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
def s331_d_single(a: dace.float64[LEN_1D], b: dace.float64[2]):
    j = -1
    j = -1
    for i in range(LEN_1D):
        if a[i] < 0.0:
            j = i
    b[0] = j


@dace.program
def s341_d_single(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D]):
    j = -1
    for i in range(LEN_1D):
        if b[i] > 0.0:
            j = j + 1
            a[j] = b[i]


@dace.program
def s342_d_single(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D]):
    j = -1
    for i in range(LEN_1D):
        if a[i] > 0.0:
            j = j + 1
            a[i] = b[j]


@dace.program
def s351_d_single(
    a: dace.float64[LEN_1D], b: dace.float64[LEN_1D], c: dace.float64[LEN_1D]
):
    alpha = c[0]
    for i in range(0, LEN_1D, 4):
        a[i] = a[i] + alpha * b[i]
        a[i + 1] = a[i + 1] + alpha * b[i + 1]
        a[i + 2] = a[i + 2] + alpha * b[i + 2]
        a[i + 3] = a[i + 3] + alpha * b[i + 3]


@dace.program
def s352_d_single(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D], c: dace.float64[2]):
    dot = 0.0
    dot = 0.0
    for i in range(0, LEN_1D, 4):
        dot = dot + (
            a[i] * b[i]
            + a[i + 1] * b[i + 1]
            + a[i + 2] * b[i + 2]
            + a[i + 3] * b[i + 3]
            + a[i + 4] * b[i + 4]
        )
    c[0] = dot


@dace.program
def s4117_d_single(
    a: dace.float64[LEN_1D],
    b: dace.float64[LEN_1D],
    c: dace.float64[LEN_1D],
    d: dace.float64[LEN_1D],
):
    for i in range(LEN_1D):
        j = i // 2
        a[i] = b[i] + c[j] * d[i]


@dace.program
def s421_d_single(a: dace.float64[LEN_1D], flat_2d_array: dace.float64[LEN_1D]):
    for i in range(LEN_1D - 1):
        flat_2d_array[i] = flat_2d_array[i + 1] + a[i]


@dace.program
def s422_d_single(a: dace.float64[LEN_1D], flat_2d_array: dace.float64[LEN_1D * LEN_1D]):
    for i in range(LEN_1D):
        flat_2d_array[4 + i] = flat_2d_array[8 + i] + a[i]


@dace.program
def s423_d_single(a: dace.float64[LEN_1D], flat_2d_array: dace.float64[LEN_1D]):
    vl = 64
    for i in range(LEN_1D - 1):
        flat_2d_array[i + 1] = flat_2d_array[vl + i] + a[i]


@dace.program
def s424_d_single(
    a: dace.float64[LEN_1D], xx: dace.float64[LEN_1D], flat: dace.float64[LEN_1D]
):
    for i in range(LEN_1D - 1):
        xx[i + 1] = flat[i] + a[i]


@dace.program
def s441_d_single(
    a: dace.float64[LEN_1D],
    b: dace.float64[LEN_1D],
    c: dace.float64[LEN_1D],
    d: dace.float64[LEN_1D],
):
    for i in range(LEN_1D):
        if d[i] < 0.0:
            a[i] = a[i] + b[i] * c[i]
        elif d[i] == 0.0:
            a[i] = a[i] + b[i] * b[i]
        else:
            a[i] = a[i] + c[i] * c[i]


@dace.program
def s443_d_single(
    a: dace.float64[LEN_1D],
    b: dace.float64[LEN_1D],
    c: dace.float64[LEN_1D],
    d: dace.float64[LEN_1D],
):
    for i in range(LEN_1D):
        if d[i] <= 0.0:
            a[i] = a[i] + b[i] * c[i]
        else:
            a[i] = a[i] + b[i] * b[i]


@dace.program
def s451_d_single(
    a: dace.float64[LEN_1D], b: dace.float64[LEN_1D], c: dace.float64[LEN_1D]
):
    for i in range(LEN_1D):
        a[i] = sin(b[i]) + cos(c[i])


@dace.program
def s452_d_single(
    a: dace.float64[LEN_1D], b: dace.float64[LEN_1D], c: dace.float64[LEN_1D]
):
    for i in range(LEN_1D):
        a[i] = b[i] + c[i] * (i + 1)


@dace.program
def s471_d_single(
    x: dace.float64[LEN_1D],
    b: dace.float64[LEN_1D],
    c: dace.float64[LEN_1D],
    d: dace.float64[LEN_1D],
    e: dace.float64[LEN_1D],
):
    for i in range(LEN_1D):
        x[i] = b[i] + d[i] * d[i]
        b[i] = c[i] + d[i] * e[i]


@dace.program
def vdotr_d_single(
    a: dace.float64[LEN_1D], b: dace.float64[LEN_1D], dot_out: dace.float64[LEN_1D]
):
    dot_out[0] = 0.0
    dot_out[0] = 0.0
    for i in range(LEN_1D):
        dot_out[0] = dot_out[0] + a[i] * b[i]


@dace.program
def vif_d_single(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D]):
    for i in range(LEN_1D):
        if b[i] > 0.0:
            a[i] = b[i]


@dace.program
def vpv_d_single(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D]):
    for i in range(LEN_1D):
        a[i] = a[i] + b[i]


@dace.program
def vpvpv_d_single(
    a: dace.float64[LEN_1D], b: dace.float64[LEN_1D], c: dace.float64[LEN_1D]
):
    for i in range(LEN_1D):
        a[i] = a[i] + b[i] + c[i]


@dace.program
def vpvtv_d_single(
    a: dace.float64[LEN_1D], b: dace.float64[LEN_1D], c: dace.float64[LEN_1D]
):
    for i in range(LEN_1D):
        a[i] = a[i] + b[i] * c[i]


@dace.program
def vsumr_d_single(a: dace.float64[LEN_1D], sum_out: dace.float64[1]):
    s = 0.0
    s = 0.0
    for i in range(LEN_1D):
        s = s + a[i]
    sum_out[0] = s


@dace.program
def vtv_d_single(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D]):
    for i in range(LEN_1D):
        a[i] = a[i] * b[i]


@dace.program
def vtvtv_d_single(
    a: dace.float64[LEN_1D], b: dace.float64[LEN_1D], c: dace.float64[LEN_1D]
):
    for i in range(LEN_1D):
        a[i] = a[i] * b[i] * c[i]

_KERNELS = [
    (s331_d_single, ['a', 'b']),
    (s341_d_single, ['a', 'b']),
    (s342_d_single, ['a', 'b']),
    (s351_d_single, ['a', 'b', 'c']),
    (s352_d_single, ['a', 'b', 'c']),
    (s4117_d_single, ['a', 'b', 'c', 'd']),
    (s421_d_single, ['a', 'flat_2d_array']),
    (s422_d_single, ['a', 'flat_2d_array']),
    (s423_d_single, ['a', 'flat_2d_array']),
    (s424_d_single, ['a', 'xx', 'flat']),
    (s441_d_single, ['a', 'b', 'c', 'd']),
    (s443_d_single, ['a', 'b', 'c', 'd']),
    (s451_d_single, ['a', 'b', 'c']),
    (s452_d_single, ['a', 'b', 'c']),
    (s471_d_single, ['x', 'b', 'c', 'd', 'e']),
    (vdotr_d_single, ['a', 'b', 'dot_out']),
    (vif_d_single, ['a', 'b']),
    (vpv_d_single, ['a', 'b']),
    (vpvpv_d_single, ['a', 'b', 'c']),
    (vpvtv_d_single, ['a', 'b', 'c']),
    (vsumr_d_single, ['a', 'sum_out']),
    (vtv_d_single, ['a', 'b']),
    (vtvtv_d_single, ['a', 'b', 'c']),
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
def test_tsvc_block3(kernel, argnames, remainder_strategy, branch_mode, len_1d_val):

    arrays_ref = {name: _allocate(name, len_1d_val) for name in argnames}
    arrays_vec = {name: arr.copy() for name, arr in arrays_ref.items()}

    sdfg_name = f"{kernel.name}_b3_{branch_mode}_{remainder_strategy}_{len_1d_val}"
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
