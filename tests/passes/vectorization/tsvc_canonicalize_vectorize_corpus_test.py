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
import os

# dace lazily ``from mpi4py import MPI`` during ``to_sdfg``. Left to auto-init,
# mpi4py installs MPI's abort-on-error handler; the compile step's fork+exec
# (cmake/g++) then aborts the whole interpreter partway through a single-process
# sweep (SIGABRT in codegen). Skip MPI_Init -- nothing here uses MPI -- and steer
# Open MPI off UCX, matching the canonicalize sibling. ``setdefault`` defers to
# any externally-provided configuration.
os.environ.setdefault("MPI4PY_RC_INITIALIZE", "0")
os.environ.setdefault("OMPI_MCA_pml", "ob1")
os.environ.setdefault("OMPI_MCA_btl", "self,vader")
os.environ.setdefault("UCX_VFS_ENABLE", "n")

import numpy as np
import pytest

from dace.sdfg import nodes as nd
from dace.transformation.passes.canonicalize import canonicalize
from dace.transformation.passes.vectorization.vectorize_cpu import VectorizeCPU
from dace.transformation.passes.vectorization.vectorize_cpu_multi_dim import VectorizeCPUMultiDim
from tests.corpus.tsvc import tsvc
from tests.corpus.tsvc.tsvc_numpy import REFERENCES

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

# Known vectorize gaps per arm; each entry is the tracking reason. Populated from
# the corpus matrix run (see the module docstring / the project memory). The
# ``xfail`` is imperative and fires before canonicalize/compile, so a kernel whose
# vectorized codegen aborts cannot crash the run. Entries are removed as the
# underlying gap is fixed (Group A miscompiles, Group B loop2map recurrences, and
# the structural ``codegen gap`` / ``tile gap`` lowerings).
_LEGACY_XFAIL: dict = {
    's1112_d_single': 'canon/shape gap: negative shape in data descriptor',
    's112_d_single': 'canon/shape gap: negative shape in data descriptor',
    's115_d_single': 'legacy Min-rebasing breaks invariant a[j] (s173 family); WCR lowered, multidim green',
    's118_d_single': 'numerical: vectorized output diverges from numpy reference',
    's1221_d_single': 'numerical: vectorized output diverges from numpy reference',
    's123_d_single': 'codegen gap: BranchNormalization IF-arm carries an unsupported body',
    's124_d_single': 'codegen gap: BranchNormalization leaves a ConditionalBlock',
    's126_d_single': "pass gap: KeyError 'pop from an empty set'",
    's162_d_single': 'codegen gap: BranchNormalization leaves a ConditionalBlock',
    's171_d_single': 'legacy: cannot widen symbolic-strided write a[i*inc] (multidim ok via trap-guard)',
    's2101_d_single': 'codegen gap: memlet subset/descriptor dimensionality mismatch',
    's2275_d_single': 'codegen gap: memlet subset/descriptor dimensionality mismatch',
    's233_d_single': 'codegen gap: non-contiguous strided edge unsupported',
    's235_d_single': "pass gap: KeyError 'pop from an empty set'",
    's2710_d_single': 'numerical: vectorized output diverges from numpy reference',
    's272_d_single': 'codegen gap: missing symbol on nested SDFG',
    's275_d_single': "pass gap: KeyError 'pop from an empty set'",
    's3111_d_single': 'numerical (flaky): masked-sum diverges nondeterministically (uninit-mask remainder)',
    's315_d_single': 'numerical: vectorized output diverges from numpy reference',
    's318_d_single': 'codegen gap: missing symbol on nested SDFG',
    's331_d_single': 'codegen gap: BranchNormalization leaves a ConditionalBlock',
    's332_d_single': 'Group B (loop2map): race condition on interstate edge (recurrence over-parallelized)',
    's4115_d_single': 'codegen gap: reduction lift inside nested SDFG unsupported',
    's4116_d_single': 'codegen gap: reduction lift inside nested SDFG unsupported',
    's442_d_single': 'numerical: vectorized output diverges from numpy reference',
    's482_d_single': 'tile gap: GenerateIterationMask requires a uniform innermost map',
}

_MULTIDIM_XFAIL: dict = {
    's118_d_single': 'numerical: vectorized output diverges from numpy reference',
    's1221_d_single': 'numerical: vectorized output diverges from numpy reference',
    's122_d_single': 'numerical: vectorized output diverges from numpy reference',
    's123_d_single': 'codegen gap: BranchNormalization IF-arm carries an unsupported body',
    's124_d_single': 'numerical: vectorized output diverges from numpy reference',
    's128_d_single': 'numerical: vectorized output diverges from numpy reference',
    's141_d_single': 'tile gap: could not build a tile-op gather index',
    's172_d_single': 'tile gap: MarkTileDims ineligible map (non-uniform/illegal tile dims)',
    's175_d_single': 'tile gap: MarkTileDims ineligible map (non-uniform/illegal tile dims)',
    's2101_d_single': 'codegen gap: memlet subset/descriptor dimensionality mismatch',
    's2275_d_single': 'codegen gap: memlet subset/descriptor dimensionality mismatch',
    's232_d_single': 'numerical: vectorized output diverges from numpy reference',
    's272_d_single': 'codegen gap: missing symbol on nested SDFG',
    's274_d_single': 'codegen gap: predicated if/else cond not fully tiled (scalar cond + tile_unop ! output)',
    's3111_d_single': 'numerical: vectorized output diverges from numpy reference',
    's331_d_single': 'numerical: vectorized output diverges from numpy reference',
    's332_d_single': 'Group B (loop2map): race condition on interstate edge (recurrence over-parallelized)',
    's341_d_single': 'numerical: vectorized output diverges from numpy reference',
    's342_d_single': 'numerical: vectorized output diverges from numpy reference',
    's343_d_single': 'numerical: vectorized output diverges from numpy reference',
    's4115_d_single': 'codegen gap: augassign_binop output-kind rule violated',
    's4116_d_single': 'codegen gap: augassign_binop output-kind rule violated',
    's481_d_single': 'numerical: find-first exit index off-by-~1 (scalar-store+ITE-tiling fixed; cause TBD)',
    's482_d_single': 'numerical: find-first exit index off-by-~1 (scalar-store+ITE-tiling fixed; cause TBD)',
}


def _assert_matches(name: str, got: dict, ref: dict, stage: str):
    """Assert every float output matches the reference, ``nan``/``inf`` equal.

    ``np.allclose(equal_nan=True)`` (not a raw ``max|diff|``) so a recurrence that
    legitimately overflows to ``inf`` everywhere -- e.g. s232's squaring scan --
    compares ``inf == inf`` instead of ``nanmax(inf - inf) == nan``; the latter is
    never ``< tol`` and spuriously fails. Integer arrays are read-only indices.
    """
    for n, a in got.items():
        if not (isinstance(a, np.ndarray) and np.issubdtype(a.dtype, np.floating) and a.size):
            continue
        assert np.allclose(np.asarray(a), np.asarray(ref[n]), rtol=1e-9, atol=1e-9,
                           equal_nan=True), (f"{name}/{n}: {stage} diverges from numpy reference, "
                                             f"max|diff|={np.nanmax(np.abs(np.asarray(a) - np.asarray(ref[n]))):.3e}")


def _canonicalized(name, tag="cvc"):
    """Canonicalize ``name`` and assert post-canon e2e correctness; return the SDFG.

    ``tag`` distinguishes the build (.dacecache) directory per test variant -- the
    ``name`` cache policy keys the folder on sdfg.name, so legacy and multidim must
    not share a tag or they collide under a parallel sweep.
    """
    kernel = tsvc.collect(name=name)[0]
    sdfg = tsvc.to_sdfg(kernel, tag=tag, simplify=True)
    canonicalize(sdfg, validate=True, peel_limit=4, break_anti_dependence=True)
    arrays, ck = tsvc.make_inputs(kernel, seed=1234)
    ref = {n: a.copy() for n, a in arrays.items()}
    REFERENCES[kernel.name](**ref, **ck)
    work = {n: a.copy() for n, a in arrays.items()}
    sdfg.compile()(**work, **ck)
    _assert_matches(name, work, ref, "canonicalization")
    return kernel, sdfg, arrays, ck, ref


def _vectorize_and_check(name, sdfg, kernel, arrays, ck, ref, vec_pass):
    vec_pass.apply_pass(sdfg, {})
    sdfg.validate()
    work = {n: a.copy() for n, a in arrays.items()}
    sdfg.compile()(**work, **ck)
    _assert_matches(name, work, ref, "vectorization")


@pytest.mark.parametrize("idx,name", list(enumerate(_KERNELS)))
def test_tsvc_canonicalize(idx, name):
    """Canonicalize -> verify e2e against numpy. Canonicalization alone is
    value-preserving; this is the first of the three corpus paths (this, then
    ``+legacy`` / ``+multidim`` vectorize). ``_canonicalized`` asserts the
    post-canon output matches the reference."""
    _canonicalized(name, tag="canon")


@pytest.mark.parametrize("idx,name", list(enumerate(_KERNELS)))
def test_tsvc_canonicalize_then_legacy_vectorize(idx, name):
    """Canonicalize -> verify -> legacy VectorizeCPU (round-robin knob) -> verify."""
    if name in _LEGACY_XFAIL:
        pytest.xfail(_LEGACY_XFAIL[name])
    kernel, sdfg, arrays, ck, ref = _canonicalized(name)
    knobs = _LEGACY_KNOBS[idx % len(_LEGACY_KNOBS)]
    _vectorize_and_check(name, sdfg, kernel, arrays, ck, ref, VectorizeCPU(8, fail_on_unvectorizable=False, **knobs))


@pytest.mark.parametrize("idx,name", list(enumerate(_KERNELS)))
def test_tsvc_canonicalize_then_multidim_vectorize(idx, name):
    """Canonicalize -> verify -> multidim VectorizeCPUMultiDim (round-robin knob,
    K=2 when the canonicalized body is a 2-D nested map) -> verify."""
    if name in _MULTIDIM_XFAIL:
        pytest.xfail(_MULTIDIM_XFAIL[name])
    kernel, sdfg, arrays, ck, ref = _canonicalized(name, tag="cvc_multidim")
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
