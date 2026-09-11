# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Scratch: attribute the CloudSC map-count drop to the F4 cleanup skip or to F3's fusion change.

Reports the map / pragma counts AND how many StructuralCleanup blocks the dirty-flag skip
actually elided. Zero skips exonerates F4 and leaves F3 as the only candidate.
"""
import contextlib
import os
import pathlib

import dace
from dace.transformation.passes.canonicalize import canonicalize, pipeline as canon_pipeline
from tests.corpus.cloudsc.pipelines import map_entries, omp_parallel_for_count
from tests.corpus.cloudsc.cloudsc_target_pipelines_test import SPECIES_CONSTANTS

REF = pathlib.Path(os.path.expanduser('~/.cache/cloudsc_reference.sdfgz'))


def test_ab_map_counts():
    skipped = []
    applied = []
    original = canon_pipeline.StructuralCleanup.apply_pass

    def counting_apply(self, sdfg, results):
        applied.append(1)
        return original(self, sdfg, results)

    canon_pipeline.StructuralCleanup.apply_pass = counting_apply
    original_changed = canon_pipeline.changed_the_graph

    def counting_changed(unit, result):
        verdict = original_changed(unit, result)
        if result is not None and not verdict:
            skipped.append(type(unit).__name__)
        return verdict

    canon_pipeline.changed_the_graph = counting_changed
    try:
        sdfg = dace.SDFG.from_file(str(REF))
        with open(os.devnull, 'w') as devnull, contextlib.redirect_stdout(devnull):
            canonicalize(sdfg, validate=True, validate_all=False, target='cpu', specialize_constants=SPECIES_CONSTANTS)
        sdfg.validate()
    finally:
        canon_pipeline.StructuralCleanup.apply_pass = original
        canon_pipeline.changed_the_graph = original_changed
    print(f'AB_RESULT maps={len(map_entries(sdfg))} omp_parallel_for={omp_parallel_for_count(sdfg)}')
    print(f'AB_RESULT cleanup_blocks_run={len(applied)} of 8')
    print(f'AB_RESULT analysis_only_results_filtered={len(skipped)} units={sorted(set(skipped))}')
