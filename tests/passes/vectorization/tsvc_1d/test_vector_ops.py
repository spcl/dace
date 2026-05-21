# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""TSVC 1D ``s``-prefixed kernels (bulk), consolidated from the former
block1/block2/block3 batches. Curated kernels live in ``test_selected``;
the non-``s`` vector kernels live in ``test_vector_ops`` — neither is
duplicated here.

Each kernel is parametrised across ``LEN_1D`` in ``{{64, 65}}`` (64
divisible-by-W=8, 65 forces a remainder), ``remainder_strategy`` in
``{{scalar, masked}}`` and ``branch_mode`` in ``{{merge, fp_factor}}``.
Reference correctness is pinned against the unvectorized SDFG.
"""
import copy

import dace
import numpy as np
import pytest

from dace.transformation.interstate import LoopToMap
from dace.transformation.passes.vectorization.vectorize_cpu import VectorizeCPU
from tests.passes.vectorization.helpers.tsvc_matrix import build_tsvc_matrix

LEN_1D = dace.symbol("LEN_1D")


@dace.program
def vdotr_d_single(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D], dot_out: dace.float64[LEN_1D]):
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
def vpvpv_d_single(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D], c: dace.float64[LEN_1D]):
    for i in range(LEN_1D):
        a[i] = a[i] + b[i] + c[i]

@dace.program
def vpvtv_d_single(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D], c: dace.float64[LEN_1D]):
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
def vtvtv_d_single(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D], c: dace.float64[LEN_1D]):
    for i in range(LEN_1D):
        a[i] = a[i] * b[i] * c[i]

_KERNELS = [
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
def test_tsvc_1d_vector_ops(kernel, argnames, remainder_strategy, branch_mode, len_1d_val):
    arrays_ref = {name: _allocate(name, len_1d_val) for name in argnames}
    arrays_vec = {name: arr.copy() for name, arr in arrays_ref.items()}

    sdfg_name = f"{kernel.name}_1dv_{branch_mode}_{remainder_strategy}_{len_1d_val}"
    # Deep-copy before any mutation: to_sdfg() may return a shared cached
    # SDFG a prior variant already mutated in place.
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

    c_ref(**arrays_ref, LEN_1D=len_1d_val)
    c_vec(**arrays_vec, LEN_1D=len_1d_val)

    for name in argnames:
        diff = np.max(np.abs(arrays_ref[name] - arrays_vec[name]))
        assert diff < 1e-10, f"{kernel.name}/{name}: max abs diff = {diff}"
