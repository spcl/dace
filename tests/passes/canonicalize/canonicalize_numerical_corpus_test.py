# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""End-to-end NUMERICAL corpus test: ``canonicalize`` must be value-preserving on
every kernel of the COMBINED polybench + npbench corpus.

For each kernel a fresh SDFG is canonicalized + ``finalize_for_target('cpu')`` and
run; its output must match the suite's reference (polybench: the untransformed
SDFG; npbench: the numpy reference). Every kernel in the corpus must pass -- there
is no xfail list, so a canon gap is a hard failure rather than a tracked backlog item.

Run as a script for the full canon / auto-opt comparison tables:
    python -m tests.passes.canonicalize.canonicalize_numerical_corpus_test
"""
import os

# No thread pin here: ``tests/conftest.py`` has already set OMP_NUM_THREADS (to at least four,
# deliberately, so a race in a generated kernel is visible), which made the ``setdefault("1")``
# that used to stand here a no-op under pytest -- and had it ever fired it would have hidden the
# very failures this file exists to catch. BandCarriedLoops gave each thread a whole stencil sweep
# with no barrier between the body's two maps, and adi / jacobi_1d / jacobi_2d came out wrong by
# ~1e-2 relative at four threads while staying bit-exact at one.
os.environ.setdefault("MPI4PY_RC_INITIALIZE", "0")
os.environ.setdefault("OMPI_MCA_pml", "ob1")
os.environ.setdefault("OMPI_MCA_btl", "self,vader")
os.environ.setdefault("UCX_VFS_ENABLE", "n")

import pytest

import dace
from dace.transformation.passes.canonicalize import canonicalize
from dace.transformation.passes.canonicalize.finalize import finalize_for_target
from tests.corpus import corpus_suite as CS

CPU = dict(target='cpu',
           peel_limit=4,
           break_anti_dependence=True,
           interchange_carry_with_map=True,
           scatter_to_guarded_maps=True)

# Kernels where the transformed build and the untransformed baseline are the SAME computation
# but differ only in the C compiler's FMA-contraction placement, which the default ``-ffast-math``
# leaves free. That ULP difference is harmless on a well-conditioned kernel but amplifies to a
# large relative error on a rank-deficient one. The set is currently EMPTY: gramschmidt used to
# need it (its polybench input was rank-deficient, so column 13 collapsed to ~3.66e-14 and the ULP
# amplified to ~100% relative error), but its ``init_array`` now builds a well-conditioned matrix
# (diagonal-dominance term, cond ~1.7), so canonicalization is value-preserving with FMA on. The
# fixture below is kept as-is so any future FMA-sensitive kernel can be pinned by adding its key.
FP_CONTRACT_OFF = set()


@pytest.fixture(autouse=True)
def _fp_contract(request):
    """Pin ``-ffp-contract=off`` for the current test iff its kernel is in
    ``FP_CONTRACT_OFF`` (see above); a no-op otherwise. Scoped per-kernel, not gate-wide, and
    the global default is left untouched (FMA stays on for performance elsewhere)."""
    params = request.node.callspec.params
    if (params.get('suite'), params.get('name')) not in FP_CONTRACT_OFF:
        yield
        return
    key = ('compiler', 'cpu', 'args')
    prev = dace.config.Config.get(*key)
    if '-ffp-contract=off' not in prev:
        dace.config.Config.set(*key, value=prev + ' -ffp-contract=off')
    yield
    dace.config.Config.set(*key, value=prev)


def _canon(s):
    return finalize_for_target(canonicalize(s, validate=True, **CPU), 'cpu')


def _preserves(suite, name, transform, tag):
    ctx = CS.make(suite, name, 'S')  # small/fast preset for a quick correctness check
    sdfg = CS.build(ctx, transform, tag)
    return CS.run_matches(ctx, sdfg)


@pytest.mark.parametrize("suite,name", CS.kernels())
def test_canon_preserves_semantics(suite, name):
    """``canonicalize(target='cpu')`` is value-preserving across polybench + npbench."""
    assert _preserves(suite, name, _canon, 'canon'), f"canon changed {suite}:{name} output vs reference"


# ---------------------------------------------------------------------------
# Script entry point: print the canon / auto-opt comparison tables.
# ---------------------------------------------------------------------------
def _generate_tables():
    from dace.transformation.auto.auto_optimize import auto_optimize
    pipelines = {
        'canon': _canon,
        'auto-opt': lambda s: auto_optimize(s, dace.DeviceType.CPU),
    }
    print(f"{'suite':5} {'kernel':18} {'canon':>10} {'auto-opt':>10}", flush=True)
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
        print(f"{suite:5} {name:18} {row['canon']:>10} {row['auto-opt']:>10}", flush=True)
    n = len(CS.kernels())
    print(f"\n# SUMMARY ({n} kernels): " + "  ".join(f"{lbl}={tally[lbl]}/{n}" for lbl in pipelines), flush=True)


if __name__ == '__main__':
    _generate_tables()
