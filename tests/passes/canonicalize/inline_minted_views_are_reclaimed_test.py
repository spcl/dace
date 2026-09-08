# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""An inline mints views, so a reclaim has to run after the last one.

``InlineSDFG`` gives a sliced connector its own ``View`` descriptor. The recipe's reclaim
(``ArrayElimination``, the only pass that folds a view back onto the array it views) used to sit
BEFORE ``_inline_single_state('end')`` and ``InlineControlFlowRegions``, so every view those minted
survived the whole pipeline: CloudSC finished canonicalization holding 18 views of ``zpfplsx``, all
of them FULL-array aliases carrying the base's own shape and strides, and 13 foldable by the
reclaimer that had already run. Downstream that reaches the vectorizer as an alias to reason about
instead of the array itself.
"""
import os

os.environ.setdefault('MPI4PY_RC_INITIALIZE', '0')

import dace
from dace import data
from dace.transformation.pass_pipeline import Pass, Pipeline
from dace.transformation.passes.array_elimination import ArrayElimination
from dace.transformation.passes.canonicalize.pipeline import _build_stages


def flatten(unit: Pass):
    """Every pass inside ``unit``, descending through pipelines and pattern appliers."""
    yield unit
    for attribute in ('_passes', '_transformations'):
        for child in vars(unit).get(attribute, ()) or ():
            if isinstance(child, Pass):
                yield from flatten(child)
            else:
                yield child  # a bare Transformation inside a PatternApplyOnceEverywhere


def stage_kinds():
    """``[{type names of every pass in the stage}]``, one entry per built stage, in order."""
    return [{type(child).__name__ for child in flatten(unit)} for _label, unit in _build_stages()]


def test_a_reclaim_runs_after_the_last_inline():
    """No view-minting inline may be the last word: a reclaim has to follow it."""
    kinds = stage_kinds()
    inlines = [i for i, names in enumerate(kinds) if any('Inline' in name for name in names)]
    reclaims = [i for i, names in enumerate(kinds) if ArrayElimination.__name__ in names]
    assert inlines, 'the recipe no longer inlines at all -- this test is watching the wrong thing'
    assert reclaims, 'the recipe no longer reclaims arrays at all'
    assert max(reclaims) > max(inlines), (f'last inline is stage {max(inlines)} but the last reclaim is stage '
                                          f'{max(reclaims)}: every view that inline mints survives the pipeline')


def test_the_reclaim_folds_a_full_array_view():
    """The shape CloudSC was left holding: a view of the WHOLE array, aliasing it exactly."""
    sdfg = dace.SDFG('full_array_view')
    sdfg.add_array('a', [8], dace.float64)
    sdfg.add_view('a_view', [8], dace.float64)
    state = sdfg.add_state('main', is_start_block=True)
    view = state.add_access('a_view')
    state.add_edge(state.add_read('a'), None, view, 'views', dace.Memlet('a[0:8]'))
    tasklet = state.add_tasklet('read_view', {'_in'}, {'_out'}, '_out = _in')
    state.add_edge(view, None, tasklet, '_in', dace.Memlet('a_view[0]'))
    state.add_edge(tasklet, '_out', state.add_write('a'), None, dace.Memlet('a[1]'))

    Pipeline([ArrayElimination()]).apply_pass(sdfg, {})

    assert not [name for name, desc in sdfg.arrays.items() if isinstance(desc, data.View)], \
        'a full-array view is a pure alias and must fold onto the array it views'
    sdfg.validate()


if __name__ == '__main__':
    test_a_reclaim_runs_after_the_last_inline()
    test_the_reclaim_folds_a_full_array_view()
