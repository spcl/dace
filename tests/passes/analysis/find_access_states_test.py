# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for :class:`FindAccessStates` -- in particular that data referenced by a
control-flow-region condition (not only by an AccessNode or an interstate edge) is
recorded as live in the states the region governs. Without that, a consumer such
as ``ArrayElimination`` would treat a data-dependent bound as unused and drop it,
leaving a dangling reference in the condition."""
import dace
from dace.sdfg.state import LoopRegion
from dace.transformation.passes.analysis import FindAccessStates

N = dace.symbol('N')


def test_condition_referenced_array_marked_live_in_region():
    """A ``LoopRegion`` whose condition directly references ``counts[0]`` (an
    array element, no AccessNode) must mark ``counts`` live in the loop body --
    the codeblock case the interstate-edge branch alone would miss."""
    sdfg = dace.SDFG('cond_data')
    sdfg.add_array('counts', [N], dace.int64)
    sdfg.add_array('out', [N], dace.float64)

    loop = LoopRegion('L', 'i < counts[0]', 'i', 'i = 0', 'i = i + 1')
    sdfg.add_node(loop, is_start_block=True)
    body = loop.add_state('body', is_start_block=True)
    t = body.add_tasklet('w', {}, {'o'}, 'o = 1.0')
    body.add_edge(t, 'o', body.add_access('out'), None, dace.Memlet('out[i]'))

    result = FindAccessStates().apply_pass(sdfg, {})[sdfg.cfg_id]
    assert 'counts' in result, "condition-referenced array dropped from access states"
    assert body in result['counts'], "condition-referenced array not marked live in the governed state"


def test_interstate_referenced_data_still_marked_live():
    """Regression for the pre-existing interstate-edge branch: an array read in an
    interstate-edge assignment stays recorded on both endpoint states."""
    sdfg = dace.SDFG('iedge_data')
    sdfg.add_array('arr', [N], dace.int64)
    sdfg.add_symbol('x', dace.int64)
    st0 = sdfg.add_state('s0', is_start_block=True)
    st1 = sdfg.add_state('s1')
    sdfg.add_edge(st0, st1, dace.InterstateEdge(assignments={'x': 'arr[0]'}))

    result = FindAccessStates().apply_pass(sdfg, {})[sdfg.cfg_id]
    assert 'arr' in result and {st0, st1} <= result['arr']


if __name__ == '__main__':
    import pytest
    pytest.main([__file__, '-v'])
