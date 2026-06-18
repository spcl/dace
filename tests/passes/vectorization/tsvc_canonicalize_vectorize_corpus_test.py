# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Full TSVC corpus: canonicalize -> vectorize, value-preserving.

For every TSVC kernel the test:

1. **Canonicalizes** the SDFG (loop nests -> parallel maps / reduces / scans).
2. **Verifies post-canonicalization correctness** end-to-end against the numpy
   reference -- canonicalization alone must preserve semantics.
3. **Vectorizes** the canonicalized SDFG with ONE config drawn round-robin (by
   kernel index) from the legacy ``VectorizeCPU`` knob set (``test_..._legacy``)
   and ONE from the multi-dim tile-op ``VectorizeCPUMultiDim`` set
   (``test_..._multidim``), then re-checks the output against numpy.

Round-robin (rather than the full knob cross-product) keeps the corpus-wide run
tractable while still exercising every knob across the suite. A kernel that
canonicalization renders as a 2-D nested map additionally gets a ``K=2`` multidim
config.

Known multidim gaps (legacy passes them) are marked ``xfail`` with the tracking
reason -- see ``_MULTIDIM_XFAIL``.
"""

import numpy as np
import pytest

from dace.sdfg import nodes as nd
from dace.transformation.passes.canonicalize import canonicalize
from dace.transformation.passes.vectorization.vectorize_cpu import VectorizeCPU
from dace.transformation.passes.vectorization.vectorize_cpu_multi_dim import VectorizeCPUMultiDim
from tests.corpus import tsvc
from tests.corpus.tsvc_numpy import REFERENCES

_KERNELS = [k.name for k in tsvc.collect()]

# Round-robin knob sets (valid combinations only; see VectorizeCPU /
# VectorizeCPUMultiDim constructors).
_LEGACY_KNOBS = [
    dict(remainder_strategy="scalar", branch_normalization=True, use_fp_factor=False),
    dict(remainder_strategy="masked", branch_normalization=True, use_fp_factor=False),
    dict(remainder_strategy="scalar", branch_normalization=False, use_fp_factor=True),
]
_MULTIDIM_KNOBS = [
    dict(target_isa="AVX512", remainder_strategy="masked_tail", branch_mode="merge"),
    dict(target_isa="SCALAR", remainder_strategy="scalar_postamble", branch_mode="merge"),
    dict(target_isa="AVX512", remainder_strategy="full_mask", branch_mode="merge"),
    dict(target_isa="SCALAR", remainder_strategy="masked_tail", branch_mode="fp_factor"),
]

# Multidim kernels with a known tile-emit gap (legacy handles them). Populated
# from the corpus harness; each entry is the tracking reason.
_MULTIDIM_XFAIL: dict = {
    # filled in after the harness matrix run
}


def _max_float_diff(got: dict, ref: dict) -> float:
    md = 0.0
    for n, a in got.items():
        if isinstance(a, np.ndarray) and np.issubdtype(a.dtype, np.floating) and a.size:
            md = max(md, float(np.nanmax(np.abs(a - ref[n]))))
    return md


def _canonicalized(name):
    """Canonicalize ``name`` and assert post-canon e2e correctness; return the SDFG."""
    kernel = tsvc.collect(name=name)[0]
    sdfg = tsvc.to_sdfg(kernel, tag="cvc", simplify=True)
    canonicalize(sdfg, validate=True, peel_limit=4, break_anti_dependence=True)
    arrays, ck = tsvc.make_inputs(kernel, seed=1234)
    ref = {n: a.copy() for n, a in arrays.items()}
    REFERENCES[kernel.name](**ref, **ck)
    work = {n: a.copy() for n, a in arrays.items()}
    sdfg.compile()(**work, **ck)
    assert _max_float_diff(work, ref) < 1e-9, f"{name}: canonicalization changed the result"
    return kernel, sdfg, arrays, ck, ref


def _vectorize_and_check(name, sdfg, kernel, arrays, ck, ref, vec_pass):
    vec_pass.apply_pass(sdfg, {})
    sdfg.validate()
    work = {n: a.copy() for n, a in arrays.items()}
    sdfg.compile()(**work, **ck)
    assert _max_float_diff(work, ref) < 1e-9, f"{name}: vectorized result diverged from numpy"


@pytest.mark.parametrize("idx,name", list(enumerate(_KERNELS)))
def test_tsvc_canonicalize_then_legacy_vectorize(idx, name):
    """Canonicalize -> verify -> legacy VectorizeCPU (round-robin knob) -> verify."""
    kernel, sdfg, arrays, ck, ref = _canonicalized(name)
    knobs = _LEGACY_KNOBS[idx % len(_LEGACY_KNOBS)]
    _vectorize_and_check(name, sdfg, kernel, arrays, ck, ref, VectorizeCPU(8, fail_on_unvectorizable=False, **knobs))


@pytest.mark.parametrize("idx,name", list(enumerate(_KERNELS)))
def test_tsvc_canonicalize_then_multidim_vectorize(idx, name):
    """Canonicalize -> verify -> multidim VectorizeCPUMultiDim (round-robin knob,
    K=2 when the canonicalized body is a 2-D nested map) -> verify."""
    if name in _MULTIDIM_XFAIL:
        pytest.xfail(_MULTIDIM_XFAIL[name])
    kernel, sdfg, arrays, ck, ref = _canonicalized(name)
    map_param_counts = [len(n.map.params) for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nd.MapEntry)]
    # K=2 only when EVERY inner map is a genuine collapsed 2-D map. A kernel with
    # any 1-D map (an init / reduction / boundary beside a 2-D body) cannot be
    # tiled with a uniform K=2 -- mixed-K within one SDFG is unsupported by the
    # tile pipeline (it aborts) -- so such kernels fall back to K=1.
    if map_param_counts and min(map_param_counts) >= 2:
        # 2-D nested map -> K=2 tile (merge/masked_tail; fp_factor+scalar are K=1 only).
        vec = VectorizeCPUMultiDim(widths=(8, 8),
                                   target_isa="SCALAR",
                                   remainder_strategy="masked_tail",
                                   branch_mode="merge")
    else:
        vec = VectorizeCPUMultiDim(widths=(8, ), **_MULTIDIM_KNOBS[idx % len(_MULTIDIM_KNOBS)])
    _vectorize_and_check(name, sdfg, kernel, arrays, ck, ref, vec)
