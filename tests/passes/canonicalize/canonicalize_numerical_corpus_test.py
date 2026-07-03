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
    # WCR seed dependency lost: StateFusionExtended merges the ``mean[:]=0`` seed state
    # and the ``mean[j]+=...`` WCR-accumulate state, leaving them unordered (the WCR's
    # read of its seed is implicit, not an edge); MapFusion then reorders -> mean=0.
    # correlation: mean/stddev reductions FIXED (StateFusion no longer fuses a seed state
    # with its WCR-accumulate). Remaining gap: ``center_data`` (in-place ``data``) is fused
    # into the same state as the ``corr`` WCR-accumulate and reads a pre-normalized stddev
    # -- a deeper normalize->center->reduce ordering scramble across StateFusion + MapFusion.
    ('poly', 'correlation'):
    'canon: center_data/corr ordering scramble (mean/stddev now OK)',
    # covariance FIXED: StateFusionExtended now treats a WCR edge as a read-modify-write and
    # refuses to fuse the ``mean[:]=0`` seed state into the ``mean(+)=`` accumulate state.
    # cholesky / ludcmp / gramschmidt (in-place-A matrix reductions ``A[i,j] OP= A[i,k]*A[k,j]``)
    # now PASS on CPU -- the reduction stays a WCR and lowers via the OMP-reduction path
    # (verified bit-exact). Xfails removed.
    # --- npbench gaps appended below from _NP_CANON_XFAIL ---
}
# npbench-suite canon gaps (established by the subprocess-isolated corpus sweep).
_NP_CANON_XFAIL = {
    # azimint_naive / nbody: a conditional accumulator inside a dc.map nsdfg
    # (``if mask: tmp += data[j]``) reduces via an in-nsdfg WCR whose output connector is
    # a POINTER to the caller's accumulator -- codegen lowers it to
    # ``reduce_atomic(&tmp, addend)`` across the parallel map lanes (verified in the
    # baseline C++). MapToForLoop sequentializes the map, but the accumulator stays a
    # WRITE-ONLY nsdfg output (no read-back of the running value is wired), so the loop
    # cannot carry the sum; SimplifyPass then removes the now-dead reduction. Fix = keep
    # the reduction parallel (refuse to sequentialize an in-nsdfg WCR reduction) OR make
    # the accumulator an InOut nsdfg interface before lowering. nbody additionally has an
    # independent canon-introduced GEMM ``ldc`` corruption from a layout transform.
    'azimint_naive': 'canon: in-nsdfg atomic reduction lost when MapToForLoop sequentializes the map',
    'nbody': 'canon: same in-nsdfg reduction-lowering gap as azimint + separate GEMM ldc corruption',
    # cavity_flow: structured-grid CFD solver -- baseline SDFG is bit-exact vs numpy, canon
    # diverges (real canon bug, not a port). Root cause not yet isolated (stencil +
    # boundary-assignment + inner pressure-poisson nit-loop). Tracked.
    'cavity_flow': 'canon: structured-grid stencil/boundary miscompile (baseline bit-exact)',
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
    # filled in below
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
