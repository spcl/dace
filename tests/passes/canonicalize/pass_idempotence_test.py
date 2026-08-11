# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Per-unit idempotence of the canonicalize pipeline.

The corpus gate (``canonicalize_idempotence_corpus_test``) proves the PROPERTY end to end but
can only report a digest diff over a whole kernel -- it cannot name the unit that broke it.
This test asks each unit directly: re-applied at its OWN position in the pipeline, does it
(a) report a change and (b) actually change anything?

Both halves matter and they disagree in practice:

* A pass that MUTATES on the second application is the outright bug -- a rewrite that is not a
  fixed point, or one that matched something it had already rewritten.
* A pass that changes NOTHING but still returns non-``None`` stalls
  :class:`~dace.transformation.pass_pipeline.FixedPointPipeline`, which converges only when
  every subpass reports ``None`` in a sweep. It is also the ``d2be1fde4`` bug class read from
  the other side: the return value is the pass's claim about what it did, and a false claim is
  a defect whether or not it is accompanied by a mutation.

Measured offenders are enumerated in ``_REPORTS_CHANGE_WITHOUT_MUTATING`` rather than skipped,
so the set is asserted exactly: a new offender fails the test, and a fixed one fails it too and
gets its entry deleted.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MPI4PY_RC_INITIALIZE", "0")
os.environ.setdefault("OMPI_MCA_pml", "ob1")
os.environ.setdefault("OMPI_MCA_btl", "self,vader")
os.environ.setdefault("UCX_VFS_ENABLE", "n")

from typing import Dict, Set, Tuple

import numpy as np

import dace
from dace.transformation.pass_pipeline import Pipeline
from dace.transformation.passes.pattern_matching import PatternMatchAndApplyRepeated
from dace.transformation.passes.canonicalize.pipeline import _build_stages

N = dace.symbol('N', dtype=dace.int64)

#: Units that leave the SDFG byte-identical on a second application yet still report a change.
#: Value is the stage(s) the unit occupies. Sound, but it prevents a fixed-point pipeline from
#: converging and misreports what the pass did. Reachability is per-fixture -- a unit only shows
#: up if this kernel reaches it -- so the set below is what THIS fixture exercises;
#: ``AssumeSymbolConstraints`` also does it, but only on shapes this kernel does not build.
_REPORTS_CHANGE_WITHOUT_MUTATING: Dict[str, str] = {
    '_PrivatizeScalarsStage': 'reduce, peel, reduction_to_wcr_map',
    '_PrivatizeArraysStage': 'reduce, peel, reduction_to_wcr_map',
    'Pipeline[ScalarFission,ScalarWriteShadowScopes,AccessSets,FindAccessNodes]': 'reduce',
    'Pipeline[ArrayFission,ArrayWriteShadowScopes,AccessSets,FindAccessNodes]': 'reduce',
    'AssignmentAndCopyKernelToMemsetAndMemcpy': 'lift_copy_loops, lift_copy',
}


@dace.program
def _kernel(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N]):
    """A reduction, an elementwise map and a whole-array copy -- enough shape to reach the
    privatize / fission / copy-lift / assumption stages the pipeline is being asked about."""
    total = np.float64(0.0)
    for i in range(N):
        total += a[i] * b[i]
    for i in dace.map[0:N]:
        c[i] = a[i] + b[i] * total
    b[:] = c[:]


def _unit_name(unit) -> str:
    """Stable identity for a pipeline unit: the wrapper plus what it wraps."""
    if isinstance(unit, PatternMatchAndApplyRepeated):
        return f"{type(unit).__name__}[{','.join(type(t).__name__ for t in unit._transformations)}]"
    if isinstance(unit, Pipeline):
        return f"{type(unit).__name__}[{','.join(type(p).__name__ for p in unit.passes[:4])}]"
    return type(unit).__name__


def _digest(sdfg) -> str:
    """Content digest. ``hash_sdfg`` already drops guid / name / cfg_list_id / history."""
    return sdfg.hash_sdfg()


def _second_application_report() -> Tuple[Set[str], Set[str]]:
    """``(mutated, reported_change_only)`` unit names, running the real pipeline once and
    applying each unit a second time at its own position."""
    sdfg = _kernel.to_sdfg(simplify=True)
    mutated: Set[str] = set()
    reported: Set[str] = set()
    for _label, unit in _build_stages():
        unit.apply_pass(sdfg, {})
        before = _digest(sdfg)
        result = unit.apply_pass(sdfg, {})
        if _digest(sdfg) != before:
            mutated.add(_unit_name(unit))
        elif result is not None:
            reported.add(_unit_name(unit))
    return mutated, reported


def test_no_unit_mutates_on_a_second_application():
    """The hard half: re-applying a unit must not rewrite the graph again. No exception list --
    a unit that keeps rewriting is a non-terminating rewrite, not a reporting nit."""
    mutated, _ = _second_application_report()
    assert not mutated, f'units that rewrite the SDFG a second time: {sorted(mutated)}'


def test_units_that_report_a_change_without_making_one_are_exactly_the_known_set():
    """The soft half, asserted as an equality so the set cannot grow silently and a fixed unit
    is noticed. Each entry stalls ``FixedPointPipeline`` convergence."""
    _, reported = _second_application_report()
    assert reported == set(_REPORTS_CHANGE_WITHOUT_MUTATING), (
        f'unexpected: {sorted(reported - set(_REPORTS_CHANGE_WITHOUT_MUTATING))}; '
        f'fixed (delete the entry): {sorted(set(_REPORTS_CHANGE_WITHOUT_MUTATING) - reported)}')


if __name__ == '__main__':
    test_no_unit_mutates_on_a_second_application()
    test_units_that_report_a_change_without_making_one_are_exactly_the_known_set()
    print("OK")
