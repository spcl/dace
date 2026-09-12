# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Assert the dirty-flag cleanup-skip mechanism, not a map count.

Reports the map / pragma counts for a human to read, and asserts the two structural contracts
the dirty flag exists to uphold:

* the recipe splices ``StructuralCleanup`` in at a FIXED number of stage boundaries
  (``_TOTAL_CLEANUP_BOUNDARIES``, read off ``_build_stages`` in ``pipeline.py``: two in
  ``_coalesce``, one each in ``lower``, ``loop_to_scan``, ``reduction_to_wcr_map`` and ``end``,
  two in ``fuse``), so no run on any SDFG can execute more cleanups than that;
* ``changed_the_graph`` only ever filters a bare ``ppl.Pipeline`` result as analysis-only "no
  change" -- never a differently-typed unit. A unit outside that one case landing in ``skipped``
  means the dirty flag went stale behind a real rewrite, which is exactly what would let a later
  ``StructuralCleanup`` boundary be skipped when it should not be.

Both hold for ANY SDFG -- they follow from the pipeline's own control flow, not from what
canonicalize happens to do to CloudSC. The reference SDFG stays ``build_cloudsc_sdfg`` (the same
call ``cloudsc_canonicalize_test.py`` uses for its pinned-count fixture) so this keeps its
original job as a smoke run on the real program too: a fresh parse on a cache miss, a cached
``.sdfgz`` (keyed by dace version + source hash, see
``generate_data_for_cloudsc.cloudsc_cache_dir``) otherwise. Never point this at a hand-picked
file -- an SDFG from a different parse is a different "A" side. The parse this falls back to is
minutes long, so this stays ``integration`` and out of the canonicalization job's 600s-per-test
budget.

The map count itself is printed only, never asserted: it is under active dispute (a 496-vs-491
bisect) and sensitive to the exact dace commit -- read it together with ``git rev-parse HEAD``,
never in isolation.
"""
import contextlib
import os
from typing import Any, Callable

import pytest

from dace import SDFG
from dace.transformation import pass_pipeline as ppl
from dace.transformation.passes.canonicalize import canonicalize, pipeline as canon_pipeline
from tests.corpus.cloudsc.pipelines import map_entries, omp_parallel_for_count
from tests.corpus.cloudsc.cloudsc_target_pipelines_test import SPECIES_CONSTANTS
from tests.corpus.cloudsc.generate_data_for_cloudsc import build_cloudsc_sdfg

#: Stage boundaries the recipe splices ``StructuralCleanup`` into. A code fact read off
#: ``_build_stages``, not a measurement -- it tracks the recipe's own shape, not what
#: canonicalize does to any particular SDFG, so pinning it carries none of the risk a map count
#: does.
_TOTAL_CLEANUP_BOUNDARIES = 8


@pytest.mark.integration
def test_ab_map_counts() -> None:
    skipped: list[str] = []
    applied: list[int] = []
    original_apply: Callable[[canon_pipeline.StructuralCleanup, SDFG, dict[str, Any]],
                             int | None] = (canon_pipeline.StructuralCleanup.apply_pass)

    def counting_apply(self: canon_pipeline.StructuralCleanup, sdfg: SDFG, results: dict[str, Any]) -> int | None:
        applied.append(1)
        return original_apply(self, sdfg, results)

    canon_pipeline.StructuralCleanup.apply_pass = counting_apply
    original_changed: Callable[[ppl.Pass, Any], bool] = canon_pipeline.changed_the_graph

    def counting_changed(unit: ppl.Pass, result: Any) -> bool:
        verdict = original_changed(unit, result)
        if result is not None and not verdict:
            skipped.append(type(unit).__name__)
        return verdict

    canon_pipeline.changed_the_graph = counting_changed
    try:
        sdfg = build_cloudsc_sdfg(simplify=False)
        with open(os.devnull, 'w') as devnull, contextlib.redirect_stdout(devnull):
            canonicalize(sdfg, validate=True, validate_all=False, target='cpu', specialize_constants=SPECIES_CONSTANTS)
        sdfg.validate()
    finally:
        canon_pipeline.StructuralCleanup.apply_pass = original_apply
        canon_pipeline.changed_the_graph = original_changed

    print(f'AB_RESULT maps={len(map_entries(sdfg))} omp_parallel_for={omp_parallel_for_count(sdfg)}')
    print(f'AB_RESULT cleanup_blocks_run={len(applied)} of {_TOTAL_CLEANUP_BOUNDARIES}')
    print(f'AB_RESULT analysis_only_results_filtered={len(skipped)} units={sorted(set(skipped))}')

    assert len(applied) <= _TOTAL_CLEANUP_BOUNDARIES, (
        f'{len(applied)} StructuralCleanup boundaries ran against a recipe that only splices in '
        f'{_TOTAL_CLEANUP_BOUNDARIES} -- a boundary ran more than once, or one was double-counted.')
    assert set(skipped) <= {
        'Pipeline'
    }, (f'changed_the_graph filtered {sorted(set(skipped) - {"Pipeline"})} as analysis-only with no '
        'change, but only a bare ppl.Pipeline result is entitled to that filter -- every other unit '
        'type reporting a non-None result IS a change, and treating it as one is what keeps the next '
        'StructuralCleanup boundary from being skipped behind a real rewrite.')
