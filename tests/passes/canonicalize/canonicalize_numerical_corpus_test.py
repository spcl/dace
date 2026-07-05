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
# channel_flow un-xfailed: the whole corpus now computes in fp64/complex128, so the
# fp32 ``while udiff > 0.001`` convergence-count divergence is gone -- baseline, canon,
# and fast-canon all PASS at preset S.
_NP_CANON_XFAIL = {
    # mandelbrot1 un-xfailed: the canon-created ``C_slice`` (a scalar, not a View) fed a
    # MapExit with the outer array's memlet after TrivialTaskletElimination spliced out the
    # boundary copy tasklet. Fixed by keeping that trivial copy tasklet at the map boundary.
    # stockham_fft un-xfailed: UniqueLoopIterators now renames the loop variable in the owning
    # SDFG's descriptor shapes, so the ``transpose_yv`` transient size loses its stale ``i`` and
    # RelaxIntegerPowers can prove ``R**(K - _loop_it_0 - 1) >= 0``.
    # mandelbrot2 remains: with those fixed, canon now reaches MapToForLoop, which drops the
    # view-defining edge of a boundary View (``C_0``, a reshape of the grid ``C``) when it nests
    # the map -> a dangling View. Tracked; fix in progress in MapToForLoop (copy the view + its
    # viewed node into the nested SDFG).
    'mandelbrot2': 'canon: MapToForLoop drops the view edge of a boundary View (C_0 reshape of C) -> dangling view',
}
_CANON_XFAIL.update({('np', n): r for n, r in _NP_CANON_XFAIL.items()})

# fast-canon shares canon's gaps.
_FAST_CANON_XFAIL = {
    **_CANON_XFAIL,
    # durbin's "fast-canon flaky KeyError" un-xfailed: the crash was a stale Dict[SDFG]
    # (FindSingelUseData) lookup of a LoopToMap-created SDFG('loop_body') NestedSDFG in
    # MapFusionVertical.is_shared_data -- fixed by falling back to a scan on a cache miss
    # instead of asserting. The flake did not reproduce in 1200+ fast-canon runs.
}
_NP_FAST_XFAIL = {
    # channel_flow (incl. the old fast-mode ScalarWriteShadowScopes KeyError) fully passes at
    # fp64; mandelbrot/stockham inherit the shared canon xfails above.
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
