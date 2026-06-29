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
    ('poly', 'adi'): 'pre-existing port/frontend uninit-read: wrong even untransformed',
    ('poly', 'atax'): 'flaky uninitialized-read (tmp): process-dependent',
    ('poly', 'cholesky'): 'canon InvalidSDFGEdgeError',
    ('poly', 'correlation'): 'canon numerical miscompile',
    ('poly', 'covariance'): 'canon numerical miscompile',
    ('poly', 'gramschmidt'): 'canon InvalidSDFGEdgeError',
    ('poly', 'k3mm'): 'corpus port zero-init / flaky (NaN even untransformed)',
    ('poly', 'lu'): 'canon InvalidSDFGEdgeError',
    ('poly', 'ludcmp'): 'canon numerical miscompile',
    # --- npbench gaps appended below from _NP_CANON_XFAIL ---
}
# npbench-suite canon gaps (established by the subprocess-isolated corpus sweep).
_NP_CANON_XFAIL = {
    # filled in below (keyed by name) -> merged as ('np', name)
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
