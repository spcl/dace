# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""End-to-end NUMERICAL corpus test: ``canonicalize`` (and its ``fast=True``
variant) must be value-preserving on every kernel of the COMBINED polybench +
npbench corpus.

For each kernel a fresh SDFG is canonicalized + ``finalize_for_target('cpu')`` and
run; its output must match the suite's reference (polybench: the untransformed
SDFG; npbench: the numpy reference). Kernels with a known *pre-existing* canon gap
are xfail-listed by ``(suite, name)`` -- each is a tracked backlog item, not a
regression; removing an entry as the underlying bug is fixed turns the xfail into
a pass. (xfail is imperative, so a flaky uninit-read kernel is never executed and
cannot flake CI.)

Run as a script for the full canon / fast-canon / auto-opt comparison tables:
    python -m tests.passes.canonicalize.canonicalize_numerical_corpus_test
"""
import os

# Pin a deterministic, single-threaded run before DaCe/OpenMP initialize, so the
# value-preserving assertions don't flake on thread races (the xfail map was
# established at OMP=1). Set at import time, before the compiled kernels load.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MPI4PY_RC_INITIALIZE", "0")
os.environ.setdefault("OMPI_MCA_pml", "ob1")
os.environ.setdefault("OMPI_MCA_btl", "self,vader")
os.environ.setdefault("UCX_VFS_ENABLE", "n")

import pytest

import dace
from dace.transformation.passes.canonicalize import canonicalize
from dace.transformation.passes.canonicalize.finalize import finalize_for_target
from tests.corpus import corpus_suite as CS

_CPU = dict(target='cpu',
            peel_limit=4,
            break_anti_dependence=True,
            interchange_carry_with_map=True,
            scatter_to_guarded_maps=True)

# Pre-existing canon correctness gaps, keyed by ``(suite, name)``. NOT regressions.
# ``WRONG`` = numerical mismatch; ``ERR`` = transform/codegen raise; ``port`` = the
# corpus port is wrong even untransformed (frontend/uninit); ``flaky`` =
# process-dependent uninitialized read.
_CANON_XFAIL = {
    # correlation FIXED (two independent canon bugs, both root-caused + verified bit-exact):
    #   (data) SplitTasklets typed the ``double(N)`` cast as int -> ``sqrt(int)`` truncated
    #     sqrt(32)=5, centering ``data`` by /(5*sd). Fixed by recognizing dtype-cast calls in
    #     ``_infer_ssa_intermediate_types``.
    #   (corr) TrivialTaskletElimination folded the ``symmetrize`` self-copy ``corr[i,j] ->
    #     corr[j,i]``; the same-array merged memlet defaulted ``_is_data_src=True`` and read
    #     the destination subset -> copy direction flipped, zeroing the off-diagonal. Fixed by
    #     building the merged edge from the read side for a self-copy.
    # covariance FIXED earlier: StateFusionExtended treats a WCR edge as a read-modify-write and
    # refuses to fuse the ``mean[:]=0`` seed state into the ``mean(+)=`` accumulate state.
    # --- npbench gaps appended below from _NP_CANON_XFAIL ---
}
# npbench-suite canon gaps (established by the subprocess-isolated corpus sweep).
_NP_CANON_XFAIL = {
    # azimint_naive FIXED by NormalizeNestedReduction. nbody + cavity_flow xfails REMOVED:
    #   nbody: the "GEMM ldc corruption" was a misdiagnosis -- the only BLAS op is a
    #     ``cblas_sgemv`` (no ldc) with correct lda/incy; the stage bisect shows canon error
    #     flat/decreasing (fusion improves the fp32 sum). Canon PASSES at preset S; it is a
    #     chaotic-fp32 kernel where the untransformed baseline sat on the tolerance edge.
    #   cavity_flow: PASSES at preset S in both modes (verified) -- fixed as a side effect of
    #     this session's channel_flow SymbolPropagation/LICM fixes (shared build_up_b/
    #     pressure_poisson structure). The prior "stencil/boundary miscompile" was never
    #     asserted (imperative xfail). Only diverges at full 61x61 by fp32 reassociation at 2
    #     interior cells where canon is *more* accurate than baseline.
    # channel_flow: NOT a canon bug. Canon (and fast) are a FAITHFUL lowering -- bit-identical
    # to the untransformed baseline -- but the baseline ITSELF diverges from numpy: the fp32
    # ``while udiff > 0.001`` convergence test (udiff = (sum(u)-sum(un))/sum(u), a catastrophic
    # cancellation of two ~96-magnitude sums) trips ~5 iterations early because numpy's pairwise
    # ``np.sum`` and the SDFG's reduction order round the near-cancelling sum differently. SDFG-u
    # matches numpy-u bit-exactly at the SDFG's own stopping iteration. This session fixed the two
    # real channel_flow canon bugs (SymbolPropagation loop-condition fold + LICM preheader uninit)
    # and the fast-mode ScalarWriteShadowScopes KeyError; the residual is a port/oracle fp32
    # determinism issue, not canon -- so it is classified ``port:`` like mandelbrot below.
    'channel_flow': 'port: fp32 convergence-count divergence (baseline diverges; canon+fast faithful, bit-identical)',
    # Broken corpus PORTS -- the @dc.program is already wrong (or fails to compile) even
    # UNTRANSFORMED, so this is not a canonicalization gap. Verified: baseline (no canon)
    # does not match the numpy reference. Out of scope for canon; tracked as port bugs.
    'mandelbrot1': 'port: baseline SDFG mismatches numpy reference even untransformed',
    'mandelbrot2': 'port: baseline SDFG mismatches numpy reference even untransformed',
    'stockham_fft': 'port: baseline SDFG mismatches numpy reference even untransformed',
}
_CANON_XFAIL.update({('np', n): r for n, r in _NP_CANON_XFAIL.items()})

# fast-canon shares canon's gaps plus fast-mode-only failures.
_FAST_CANON_XFAIL = {
    **_CANON_XFAIL,
    ('poly', 'durbin'): 'fast-canon flaky KeyError',
}
_NP_FAST_XFAIL = {
    # channel_flow's fast-mode ``KeyError: SDFG (loop_body)`` (ScalarWriteShadowScopes walking
    # a foreign clone's blocks after loop fission) is FIXED; it now inherits the shared canon
    # ``port:`` xfail above (fast lowering is faithful, same fp32-convergence divergence).
}
_FAST_CANON_XFAIL.update({('np', n): r for n, r in _NP_FAST_XFAIL.items()})


def _canon(s):
    return finalize_for_target(canonicalize(s, validate=True, **_CPU), 'cpu')


def _fast_canon(s):
    return finalize_for_target(canonicalize(s, validate=True, fast=True, **_CPU), 'cpu')


def _preserves(suite, name, transform, tag):
    ctx = CS.make(suite, name, 'S')  # small/fast preset for a quick correctness check
    sdfg = CS.build(ctx, transform, tag)
    return CS.run_matches(ctx, sdfg)


@pytest.mark.parametrize("suite,name", CS.kernels())
def test_canon_preserves_semantics(suite, name):
    """``canonicalize(target='cpu')`` is value-preserving across polybench + npbench."""
    if (suite, name) in _CANON_XFAIL:
        pytest.xfail(_CANON_XFAIL[(suite, name)])
    assert _preserves(suite, name, _canon, 'canon'), f"canon changed {suite}:{name} output vs reference"


@pytest.mark.parametrize("suite,name", CS.kernels())
def test_fast_canon_preserves_semantics(suite, name):
    """``canonicalize(target='cpu', fast=True)`` is value-preserving across the corpus."""
    if (suite, name) in _FAST_CANON_XFAIL:
        pytest.xfail(_FAST_CANON_XFAIL[(suite, name)])
    assert _preserves(suite, name, _fast_canon, 'fastcanon'), f"fast-canon changed {suite}:{name} output vs reference"


# ---------------------------------------------------------------------------
# Script entry point: print the canon / fast-canon / auto-opt comparison tables.
# ---------------------------------------------------------------------------
def _generate_tables():
    from dace.transformation.auto.auto_optimize import auto_optimize
    pipelines = {
        'canon': _canon,
        'fast-canon': _fast_canon,
        'auto-opt': lambda s: auto_optimize(s, dace.DeviceType.CPU),
    }
    print(f"{'suite':5} {'kernel':18} {'canon':>10} {'fast-canon':>12} {'auto-opt':>10}", flush=True)
    tally = {lbl: 0 for lbl in pipelines}
    for suite, name in CS.kernels():
        row = {}
        for lbl, transform in pipelines.items():
            try:
                ctx = CS.make(suite, name, 'S')
                s = CS.build(ctx, transform, lbl.replace('-', ''))
                row[lbl] = 'PASS' if CS.run_matches(ctx, s) else 'WRONG'
            except Exception as e:
                row[lbl] = f'ERR:{type(e).__name__}'
            if row[lbl] == 'PASS':
                tally[lbl] += 1
        print(f"{suite:5} {name:18} {row['canon']:>10} {row['fast-canon']:>12} {row['auto-opt']:>10}", flush=True)
    n = len(CS.kernels())
    print(f"\n# SUMMARY ({n} kernels): " + "  ".join(f"{lbl}={tally[lbl]}/{n}" for lbl in pipelines), flush=True)


if __name__ == '__main__':
    _generate_tables()
