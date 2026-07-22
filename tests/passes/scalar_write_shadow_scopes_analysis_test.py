# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.
""" Tests the scalar write shadowing analysis pass. """

import pytest
import dace
from dace.properties import CodeBlock
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion, LoopRegion
from dace.transformation.pass_pipeline import Pipeline
from dace.transformation.passes.analysis import ScalarWriteShadowScopes
from dace.transformation.passes.simplification.control_flow_raising import ControlFlowRaising
from dace.transformation.passes.simplification.prune_empty_conditional_branches import PruneEmptyConditionalBranches


@pytest.mark.parametrize('with_raising', (False, True))
def test_scalar_write_shadow_split(with_raising):
    """
    Test the scalar write shadow scopes pass with writes dominating reads across state.
    """
    # Construct the SDFG.
    sdfg = dace.SDFG('scalar_split')

    N = dace.symbol('N')
    sdfg.add_array('A', [N], dace.int32)
    sdfg.add_array('B', [N], dace.int32)
    sdfg.add_array('tmp', [1], dace.int32, transient=True)

    init_state = sdfg.add_state('init')
    guard_1 = sdfg.add_state('guard_1')
    loop_1_1 = sdfg.add_state('loop_1_1')
    loop_1_2 = sdfg.add_state('loop_1_2')
    intermediate = sdfg.add_state('intermediate')
    guard_2 = sdfg.add_state('guard_2')
    loop_2_1 = sdfg.add_state('loop_2_1')
    loop_2_2 = sdfg.add_state('loop_2_2')
    end_state = sdfg.add_state('end')

    init_tasklet = init_state.add_tasklet('init', {}, {'out'}, 'out = 0')
    init_write = init_state.add_write('tmp')
    init_state.add_edge(init_tasklet, 'out', init_write, None, dace.Memlet('tmp[0]'))

    tmp1_tasklet = loop_1_1.add_tasklet('tmp1', {'a', 'b'}, {'out'}, 'out = a * b')
    tmp1_write = loop_1_1.add_write('tmp')
    a1_read = loop_1_1.add_read('A')
    b1_read = loop_1_1.add_read('B')
    loop_1_1.add_edge(a1_read, None, tmp1_tasklet, 'a', dace.Memlet('A[i]'))
    loop_1_1.add_edge(b1_read, None, tmp1_tasklet, 'b', dace.Memlet('B[i]'))
    loop_1_1.add_edge(tmp1_tasklet, 'out', tmp1_write, None, dace.Memlet('tmp[0]'))

    loop1_tasklet_1 = loop_1_2.add_tasklet('loop1_1', {'ap', 't'}, {'a'}, 'a = ap + 2 * t')
    loop1_tasklet_2 = loop_1_2.add_tasklet('loop1_2', {'bp', 't'}, {'b'}, 'b = bp - 2 * t')
    loop1_read_tmp = loop_1_2.add_read('tmp')
    loop1_read_a = loop_1_2.add_read('A')
    loop1_read_b = loop_1_2.add_read('B')
    loop1_write_a = loop_1_2.add_write('A')
    loop1_write_b = loop_1_2.add_write('B')
    loop_1_2.add_edge(loop1_read_tmp, None, loop1_tasklet_1, 't', dace.Memlet('tmp[0]'))
    loop_1_2.add_edge(loop1_read_tmp, None, loop1_tasklet_2, 't', dace.Memlet('tmp[0]'))
    loop_1_2.add_edge(loop1_read_a, None, loop1_tasklet_1, 'ap', dace.Memlet('A[i + 1]'))
    loop_1_2.add_edge(loop1_read_b, None, loop1_tasklet_2, 'bp', dace.Memlet('B[i + 1]'))
    loop_1_2.add_edge(loop1_tasklet_1, 'a', loop1_write_a, None, dace.Memlet('A[i]'))
    loop_1_2.add_edge(loop1_tasklet_2, 'b', loop1_write_b, None, dace.Memlet('B[i]'))

    tmp2_tasklet = loop_2_1.add_tasklet('tmp2', {'a', 'b'}, {'out'}, 'out = a / b')
    tmp2_write = loop_2_1.add_write('tmp')
    a2_read = loop_2_1.add_read('A')
    b2_read = loop_2_1.add_read('B')
    loop_2_1.add_edge(a2_read, None, tmp2_tasklet, 'a', dace.Memlet('A[i + 1]'))
    loop_2_1.add_edge(b2_read, None, tmp2_tasklet, 'b', dace.Memlet('B[i + 1]'))
    loop_2_1.add_edge(tmp2_tasklet, 'out', tmp2_write, None, dace.Memlet('tmp[0]'))

    loop2_tasklet_1 = loop_2_2.add_tasklet('loop2_1', {'ap', 't'}, {'a'}, 'a = ap + t * t')
    loop2_tasklet_2 = loop_2_2.add_tasklet('loop2_2', {'bp', 't'}, {'b'}, 'b = bp - t * t')
    loop2_read_tmp = loop_2_2.add_read('tmp')
    loop2_read_a = loop_2_2.add_read('A')
    loop2_read_b = loop_2_2.add_read('B')
    loop2_write_a = loop_2_2.add_write('A')
    loop2_write_b = loop_2_2.add_write('B')
    loop_2_2.add_edge(loop2_read_tmp, None, loop2_tasklet_1, 't', dace.Memlet('tmp[0]'))
    loop_2_2.add_edge(loop2_read_tmp, None, loop2_tasklet_2, 't', dace.Memlet('tmp[0]'))
    loop_2_2.add_edge(loop2_read_a, None, loop2_tasklet_1, 'ap', dace.Memlet('A[i]'))
    loop_2_2.add_edge(loop2_read_b, None, loop2_tasklet_2, 'bp', dace.Memlet('B[i]'))
    loop_2_2.add_edge(loop2_tasklet_1, 'a', loop2_write_a, None, dace.Memlet('A[i + 1]'))
    loop_2_2.add_edge(loop2_tasklet_2, 'b', loop2_write_b, None, dace.Memlet('B[i + 1]'))

    sdfg.add_edge(init_state, guard_1, dace.InterstateEdge(assignments={'i': 0}))
    sdfg.add_edge(guard_1, loop_1_1, dace.InterstateEdge(condition='i < (N - 1)'))
    sdfg.add_edge(loop_1_1, loop_1_2, dace.InterstateEdge())
    sdfg.add_edge(loop_1_2, guard_1, dace.InterstateEdge(assignments={'i': 'i + 1'}))
    sdfg.add_edge(guard_1, intermediate, dace.InterstateEdge(condition='i >= (N - 1)'))

    sdfg.add_edge(intermediate, guard_2, dace.InterstateEdge(assignments={'i': 0}))
    sdfg.add_edge(guard_2, loop_2_1, dace.InterstateEdge(condition='i < (N - 1)'))
    sdfg.add_edge(loop_2_1, loop_2_2, dace.InterstateEdge())
    sdfg.add_edge(loop_2_2, guard_2, dace.InterstateEdge(assignments={'i': 'i + 1'}))
    sdfg.add_edge(guard_2, end_state, dace.InterstateEdge(condition='i >= (N - 1)'))

    if with_raising:
        Pipeline([ControlFlowRaising(), PruneEmptyConditionalBranches()]).apply_pass(sdfg, {})

    # Test the pass.
    pipeline = Pipeline([ScalarWriteShadowScopes()])
    results = pipeline.apply_pass(sdfg, {})[ScalarWriteShadowScopes.__name__]

    assert results[0]['tmp'][(loop_1_1, tmp1_write)] == {(loop_1_2, loop1_read_tmp)}
    assert results[0]['tmp'][(loop_2_1, tmp2_write)] == {(loop_2_2, loop2_read_tmp)}
    assert results[0]['A'][None] == {(loop_1_1, a1_read), (loop_1_2, loop1_read_a), (loop_2_1, a2_read),
                                     (loop_2_2, loop2_read_a), (loop_2_2, loop2_write_a), (loop_1_2, loop1_write_a)}
    assert results[0]['B'][None] == {(loop_1_1, b1_read), (loop_1_2, loop1_read_b), (loop_2_1, b2_read),
                                     (loop_2_2, loop2_read_b), (loop_2_2, loop2_write_b), (loop_1_2, loop1_write_b)}


@pytest.mark.parametrize('with_raising', (False, True))
def test_scalar_write_shadow_fused(with_raising):
    """
    Test the scalar write shadow scopes pass with writes dominating reads in the same state.
    """
    # Construct the SDFG.
    sdfg = dace.SDFG('scalar_fused')

    N = dace.symbol('N')
    sdfg.add_array('A', [N], dace.int32)
    sdfg.add_array('B', [N], dace.int32)
    sdfg.add_array('tmp', [1], dace.int32, transient=True)

    init_state = sdfg.add_state('init')
    guard_1 = sdfg.add_state('guard_1')
    loop_1 = sdfg.add_state('loop_1')
    intermediate = sdfg.add_state('intermediate')
    guard_2 = sdfg.add_state('guard_2')
    loop_2 = sdfg.add_state('loop_2')
    end_state = sdfg.add_state('end')

    init_tasklet = init_state.add_tasklet('init', {}, {'out'}, 'out = 0')
    init_write = init_state.add_write('tmp')
    init_state.add_edge(init_tasklet, 'out', init_write, None, dace.Memlet('tmp[0]'))

    tmp1_tasklet = loop_1.add_tasklet('tmp1', {'a', 'b'}, {'out'}, 'out = a * b')
    loop1_tasklet_1 = loop_1.add_tasklet('loop1_1', {'ap', 't'}, {'a'}, 'a = ap + 2 * t')
    loop1_tasklet_2 = loop_1.add_tasklet('loop1_2', {'bp', 't'}, {'b'}, 'b = bp - 2 * t')
    tmp1_read_write = loop_1.add_access('tmp')
    a1_read = loop_1.add_read('A')
    b1_read = loop_1.add_read('B')
    a1_write = loop_1.add_write('A')
    b1_write = loop_1.add_write('B')
    loop_1.add_edge(a1_read, None, tmp1_tasklet, 'a', dace.Memlet('A[i]'))
    loop_1.add_edge(b1_read, None, tmp1_tasklet, 'b', dace.Memlet('B[i]'))
    loop_1.add_edge(tmp1_tasklet, 'out', tmp1_read_write, None, dace.Memlet('tmp[0]'))
    loop_1.add_edge(tmp1_read_write, None, loop1_tasklet_1, 't', dace.Memlet('tmp[0]'))
    loop_1.add_edge(tmp1_read_write, None, loop1_tasklet_2, 't', dace.Memlet('tmp[0]'))
    loop_1.add_edge(a1_read, None, loop1_tasklet_1, 'ap', dace.Memlet('A[i + 1]'))
    loop_1.add_edge(b1_read, None, loop1_tasklet_2, 'bp', dace.Memlet('B[i + 1]'))
    loop_1.add_edge(loop1_tasklet_1, 'a', a1_write, None, dace.Memlet('A[i]'))
    loop_1.add_edge(loop1_tasklet_2, 'b', b1_write, None, dace.Memlet('B[i]'))

    tmp2_tasklet = loop_2.add_tasklet('tmp2', {'a', 'b'}, {'out'}, 'out = a / b')
    loop2_tasklet_1 = loop_2.add_tasklet('loop2_1', {'ap', 't'}, {'a'}, 'a = ap + t * t')
    loop2_tasklet_2 = loop_2.add_tasklet('loop2_2', {'bp', 't'}, {'b'}, 'b = bp - t * t')
    tmp2_read_write = loop_2.add_access('tmp')
    a2_read = loop_2.add_read('A')
    b2_read = loop_2.add_read('B')
    a2_write = loop_2.add_write('A')
    b2_write = loop_2.add_write('B')
    loop_2.add_edge(a2_read, None, tmp2_tasklet, 'a', dace.Memlet('A[i + 1]'))
    loop_2.add_edge(b2_read, None, tmp2_tasklet, 'b', dace.Memlet('B[i + 1]'))
    loop_2.add_edge(tmp2_tasklet, 'out', tmp2_read_write, None, dace.Memlet('tmp[0]'))
    loop_2.add_edge(tmp2_read_write, None, loop2_tasklet_1, 't', dace.Memlet('tmp[0]'))
    loop_2.add_edge(tmp2_read_write, None, loop2_tasklet_2, 't', dace.Memlet('tmp[0]'))
    loop_2.add_edge(a2_read, None, loop2_tasklet_1, 'ap', dace.Memlet('A[i]'))
    loop_2.add_edge(b2_read, None, loop2_tasklet_2, 'bp', dace.Memlet('B[i]'))
    loop_2.add_edge(loop2_tasklet_1, 'a', a2_write, None, dace.Memlet('A[i + 1]'))
    loop_2.add_edge(loop2_tasklet_2, 'b', b2_write, None, dace.Memlet('B[i + 1]'))

    sdfg.add_edge(init_state, guard_1, dace.InterstateEdge(assignments={'i': 0}))
    sdfg.add_edge(guard_1, loop_1, dace.InterstateEdge(condition='i < (N - 1)'))
    sdfg.add_edge(loop_1, guard_1, dace.InterstateEdge(assignments={'i': 'i + 1'}))
    sdfg.add_edge(guard_1, intermediate, dace.InterstateEdge(condition='i >= (N - 1)'))

    sdfg.add_edge(intermediate, guard_2, dace.InterstateEdge(assignments={'i': 0}))
    sdfg.add_edge(guard_2, loop_2, dace.InterstateEdge(condition='i < (N - 1)'))
    sdfg.add_edge(loop_2, guard_2, dace.InterstateEdge(assignments={'i': 'i + 1'}))
    sdfg.add_edge(guard_2, end_state, dace.InterstateEdge(condition='i >= (N - 1)'))

    if with_raising:
        Pipeline([ControlFlowRaising(), PruneEmptyConditionalBranches()]).apply_pass(sdfg, {})

    # Test the pass.
    pipeline = Pipeline([ScalarWriteShadowScopes()])
    results = pipeline.apply_pass(sdfg, {})[ScalarWriteShadowScopes.__name__]

    assert results[0]['tmp'][(loop_1, tmp1_read_write)] == {(loop_1, tmp1_read_write)}
    assert results[0]['tmp'][(loop_2, tmp2_read_write)] == {(loop_2, tmp2_read_write)}
    assert results[0]['A'][None] == {(loop_1, a1_read), (loop_2, a2_read), (loop_1, a1_write), (loop_2, a2_write)}
    assert results[0]['B'][None] == {(loop_1, b1_read), (loop_2, b2_read), (loop_1, b1_write), (loop_2, b2_write)}


@pytest.mark.parametrize('with_raising', (False, True))
def test_scalar_write_shadow_interstate_self(with_raising):
    """
    Tests the scalar write shadow pass with interstate edge reads being shadowed by the state they're originating from.
    """
    # Construct the SDFG.
    sdfg = dace.SDFG('scalar_isedge')

    N = dace.symbol('N')
    sdfg.add_array('A', [N], dace.int32)
    sdfg.add_array('B', [N], dace.int32)
    sdfg.add_array('tmp', [1], dace.int32, transient=True)

    init_state = sdfg.add_state('init')
    guard_1 = sdfg.add_state('guard_1')
    loop_1_1 = sdfg.add_state('loop_1_1')
    loop_1_2 = sdfg.add_state('loop_1_2')
    intermediate = sdfg.add_state('intermediate')
    guard_2 = sdfg.add_state('guard_2')
    loop_2_1 = sdfg.add_state('loop_2_1')
    loop_2_2 = sdfg.add_state('loop_2_2')
    end_state = sdfg.add_state('end')

    init_tasklet = init_state.add_tasklet('init', {}, {'out'}, 'out = 0')
    init_write = init_state.add_write('tmp')
    init_state.add_edge(init_tasklet, 'out', init_write, None, dace.Memlet('tmp[0]'))

    tmp1_tasklet = loop_1_1.add_tasklet('tmp1', {'a', 'b'}, {'out'}, 'out = a * b')
    tmp1_write = loop_1_1.add_write('tmp')
    a1_read = loop_1_1.add_read('A')
    b1_read = loop_1_1.add_read('B')
    loop_1_1.add_edge(a1_read, None, tmp1_tasklet, 'a', dace.Memlet('A[i]'))
    loop_1_1.add_edge(b1_read, None, tmp1_tasklet, 'b', dace.Memlet('B[i]'))
    loop_1_1.add_edge(tmp1_tasklet, 'out', tmp1_write, None, dace.Memlet('tmp[0]'))

    loop1_tasklet_1 = loop_1_2.add_tasklet('loop1_1', {'ap', 't'}, {'a'}, 'a = ap + 2 * t')
    loop1_tasklet_2 = loop_1_2.add_tasklet('loop1_2', {'bp', 't'}, {'b'}, 'b = bp - 2 * t')
    loop1_read_tmp = loop_1_2.add_read('tmp')
    loop1_read_a = loop_1_2.add_read('A')
    loop1_read_b = loop_1_2.add_read('B')
    loop1_write_a = loop_1_2.add_write('A')
    loop1_write_b = loop_1_2.add_write('B')
    loop_1_2.add_edge(loop1_read_tmp, None, loop1_tasklet_1, 't', dace.Memlet('tmp[0]'))
    loop_1_2.add_edge(loop1_read_tmp, None, loop1_tasklet_2, 't', dace.Memlet('tmp[0]'))
    loop_1_2.add_edge(loop1_read_a, None, loop1_tasklet_1, 'ap', dace.Memlet('A[i + 1]'))
    loop_1_2.add_edge(loop1_read_b, None, loop1_tasklet_2, 'bp', dace.Memlet('B[i + 1]'))
    loop_1_2.add_edge(loop1_tasklet_1, 'a', loop1_write_a, None, dace.Memlet('A[i]'))
    loop_1_2.add_edge(loop1_tasklet_2, 'b', loop1_write_b, None, dace.Memlet('B[i]'))

    tmp2_tasklet = loop_2_1.add_tasklet('tmp2', {'a', 'b'}, {'out'}, 'out = a / b')
    tmp2_write = loop_2_1.add_write('tmp')
    a2_read = loop_2_1.add_read('A')
    b2_read = loop_2_1.add_read('B')
    loop_2_1.add_edge(a2_read, None, tmp2_tasklet, 'a', dace.Memlet('A[i + 1]'))
    loop_2_1.add_edge(b2_read, None, tmp2_tasklet, 'b', dace.Memlet('B[i + 1]'))
    loop_2_1.add_edge(tmp2_tasklet, 'out', tmp2_write, None, dace.Memlet('tmp[0]'))

    loop2_tasklet_1 = loop_2_2.add_tasklet('loop2_1', {'ap', 't'}, {'a'}, 'a = ap + t * t')
    loop2_tasklet_2 = loop_2_2.add_tasklet('loop2_2', {'bp', 't'}, {'b'}, 'b = bp - t * t')
    loop2_read_tmp = loop_2_2.add_read('tmp')
    loop2_read_a = loop_2_2.add_read('A')
    loop2_read_b = loop_2_2.add_read('B')
    loop2_write_a = loop_2_2.add_write('A')
    loop2_write_b = loop_2_2.add_write('B')
    loop_2_2.add_edge(loop2_read_tmp, None, loop2_tasklet_1, 't', dace.Memlet('tmp[0]'))
    loop_2_2.add_edge(loop2_read_tmp, None, loop2_tasklet_2, 't', dace.Memlet('tmp[0]'))
    loop_2_2.add_edge(loop2_read_a, None, loop2_tasklet_1, 'ap', dace.Memlet('A[i]'))
    loop_2_2.add_edge(loop2_read_b, None, loop2_tasklet_2, 'bp', dace.Memlet('B[i]'))
    loop_2_2.add_edge(loop2_tasklet_1, 'a', loop2_write_a, None, dace.Memlet('A[i + 1]'))
    loop_2_2.add_edge(loop2_tasklet_2, 'b', loop2_write_b, None, dace.Memlet('B[i + 1]'))

    sdfg.add_edge(init_state, guard_1, dace.InterstateEdge(assignments={'i': 0}))
    sdfg.add_edge(guard_1, loop_1_1, dace.InterstateEdge(condition='i < (N - 1)'))
    tmp1_edge = dace.InterstateEdge(assignments={'j': 'tmp'})
    sdfg.add_edge(loop_1_1, loop_1_2, tmp1_edge)
    sdfg.add_edge(loop_1_2, guard_1, dace.InterstateEdge(assignments={'i': 'i + 1'}))
    sdfg.add_edge(guard_1, intermediate, dace.InterstateEdge(condition='i >= (N - 1)'))

    sdfg.add_edge(intermediate, guard_2, dace.InterstateEdge(assignments={'i': 0}))
    sdfg.add_edge(guard_2, loop_2_1, dace.InterstateEdge(condition='i < (N - 1)'))
    tmp2_edge = dace.InterstateEdge(assignments={'j': 'tmp'})
    sdfg.add_edge(loop_2_1, loop_2_2, tmp2_edge)
    sdfg.add_edge(loop_2_2, guard_2, dace.InterstateEdge(assignments={'i': 'i + 1'}))
    sdfg.add_edge(guard_2, end_state, dace.InterstateEdge(condition='i >= (N - 1)'))

    if with_raising:
        Pipeline([ControlFlowRaising(), PruneEmptyConditionalBranches()]).apply_pass(sdfg, {})

    # Test the pass.
    pipeline = Pipeline([ScalarWriteShadowScopes()])
    results = pipeline.apply_pass(sdfg, {})[ScalarWriteShadowScopes.__name__]

    assert results[0]['tmp'][(loop_1_1, tmp1_write)] == {(loop_1_2, loop1_read_tmp), (loop_1_1, tmp1_edge)}
    assert results[0]['tmp'][(loop_2_1, tmp2_write)] == {(loop_2_2, loop2_read_tmp), (loop_2_1, tmp2_edge)}
    assert results[0]['A'][None] == {(loop_1_1, a1_read), (loop_1_2, loop1_read_a), (loop_2_1, a2_read),
                                     (loop_2_2, loop2_read_a), (loop_1_2, loop1_write_a), (loop_2_2, loop2_write_a)}
    assert results[0]['B'][None] == {(loop_1_1, b1_read), (loop_1_2, loop1_read_b), (loop_2_1, b2_read),
                                     (loop_2_2, loop2_read_b), (loop_1_2, loop1_write_b), (loop_2_2, loop2_write_b)}


@pytest.mark.parametrize('with_raising', (False, True))
def test_scalar_write_shadow_interstate_pred(with_raising):
    """
    Tests the scalar write shadow pass with interstate edge reads being shadowed by a predecessor state.
    """
    # Construct the SDFG.
    sdfg = dace.SDFG('scalar_isedge')

    N = dace.symbol('N')
    sdfg.add_array('A', [N], dace.int32)
    sdfg.add_array('B', [N], dace.int32)
    sdfg.add_array('tmp', [1], dace.int32, transient=True)

    init_state = sdfg.add_state('init')
    guard_1 = sdfg.add_state('guard_1')
    loop_1_1 = sdfg.add_state('loop_1_1')
    loop_1_2 = sdfg.add_state('loop_1_2')
    loop_1_3 = sdfg.add_state('loop_1_3')
    intermediate = sdfg.add_state('intermediate')
    guard_2 = sdfg.add_state('guard_2')
    loop_2_1 = sdfg.add_state('loop_2_1')
    loop_2_2 = sdfg.add_state('loop_2_2')
    loop_2_3 = sdfg.add_state('loop_2_3')
    end_state = sdfg.add_state('end')

    init_tasklet = init_state.add_tasklet('init', {}, {'out'}, 'out = 0')
    init_write = init_state.add_write('tmp')
    init_state.add_edge(init_tasklet, 'out', init_write, None, dace.Memlet('tmp[0]'))

    tmp1_tasklet = loop_1_1.add_tasklet('tmp1', {'a', 'b'}, {'out'}, 'out = a * b')
    tmp1_write = loop_1_1.add_write('tmp')
    a1_read = loop_1_1.add_read('A')
    b1_read = loop_1_1.add_read('B')
    loop_1_1.add_edge(a1_read, None, tmp1_tasklet, 'a', dace.Memlet('A[i]'))
    loop_1_1.add_edge(b1_read, None, tmp1_tasklet, 'b', dace.Memlet('B[i]'))
    loop_1_1.add_edge(tmp1_tasklet, 'out', tmp1_write, None, dace.Memlet('tmp[0]'))

    loop1_tasklet_1 = loop_1_3.add_tasklet('loop1_1', {'ap', 't'}, {'a'}, 'a = ap + 2 * t')
    loop1_tasklet_2 = loop_1_3.add_tasklet('loop1_2', {'bp', 't'}, {'b'}, 'b = bp - 2 * t')
    loop1_read_tmp = loop_1_3.add_read('tmp')
    loop1_read_a = loop_1_3.add_read('A')
    loop1_read_b = loop_1_3.add_read('B')
    loop1_write_a = loop_1_3.add_write('A')
    loop1_write_b = loop_1_3.add_write('B')
    loop_1_3.add_edge(loop1_read_tmp, None, loop1_tasklet_1, 't', dace.Memlet('tmp[0]'))
    loop_1_3.add_edge(loop1_read_tmp, None, loop1_tasklet_2, 't', dace.Memlet('tmp[0]'))
    loop_1_3.add_edge(loop1_read_a, None, loop1_tasklet_1, 'ap', dace.Memlet('A[i + 1]'))
    loop_1_3.add_edge(loop1_read_b, None, loop1_tasklet_2, 'bp', dace.Memlet('B[i + 1]'))
    loop_1_3.add_edge(loop1_tasklet_1, 'a', loop1_write_a, None, dace.Memlet('A[i]'))
    loop_1_3.add_edge(loop1_tasklet_2, 'b', loop1_write_b, None, dace.Memlet('B[i]'))

    tmp2_tasklet = loop_2_1.add_tasklet('tmp2', {'a', 'b'}, {'out'}, 'out = a / b')
    tmp2_write = loop_2_1.add_write('tmp')
    a2_read = loop_2_1.add_read('A')
    b2_read = loop_2_1.add_read('B')
    loop_2_1.add_edge(a2_read, None, tmp2_tasklet, 'a', dace.Memlet('A[i + 1]'))
    loop_2_1.add_edge(b2_read, None, tmp2_tasklet, 'b', dace.Memlet('B[i + 1]'))
    loop_2_1.add_edge(tmp2_tasklet, 'out', tmp2_write, None, dace.Memlet('tmp[0]'))

    loop2_tasklet_1 = loop_2_3.add_tasklet('loop2_1', {'ap', 't'}, {'a'}, 'a = ap + t * t')
    loop2_tasklet_2 = loop_2_3.add_tasklet('loop2_2', {'bp', 't'}, {'b'}, 'b = bp - t * t')
    loop2_read_tmp = loop_2_3.add_read('tmp')
    loop2_read_a = loop_2_3.add_read('A')
    loop2_read_b = loop_2_3.add_read('B')
    loop2_write_a = loop_2_3.add_write('A')
    loop2_write_b = loop_2_3.add_write('B')
    loop_2_3.add_edge(loop2_read_tmp, None, loop2_tasklet_1, 't', dace.Memlet('tmp[0]'))
    loop_2_3.add_edge(loop2_read_tmp, None, loop2_tasklet_2, 't', dace.Memlet('tmp[0]'))
    loop_2_3.add_edge(loop2_read_a, None, loop2_tasklet_1, 'ap', dace.Memlet('A[i]'))
    loop_2_3.add_edge(loop2_read_b, None, loop2_tasklet_2, 'bp', dace.Memlet('B[i]'))
    loop_2_3.add_edge(loop2_tasklet_1, 'a', loop2_write_a, None, dace.Memlet('A[i + 1]'))
    loop_2_3.add_edge(loop2_tasklet_2, 'b', loop2_write_b, None, dace.Memlet('B[i + 1]'))

    sdfg.add_edge(init_state, guard_1, dace.InterstateEdge(assignments={'i': 0}))
    sdfg.add_edge(guard_1, loop_1_1, dace.InterstateEdge(condition='i < (N - 1)'))
    sdfg.add_edge(loop_1_1, loop_1_2, dace.InterstateEdge())
    tmp1_edge = dace.InterstateEdge(assignments={'j': 'tmp'})
    sdfg.add_edge(loop_1_2, loop_1_3, tmp1_edge)
    sdfg.add_edge(loop_1_3, guard_1, dace.InterstateEdge(assignments={'i': 'i + 1'}))
    sdfg.add_edge(guard_1, intermediate, dace.InterstateEdge(condition='i >= (N - 1)'))

    sdfg.add_edge(intermediate, guard_2, dace.InterstateEdge(assignments={'i': 0}))
    sdfg.add_edge(guard_2, loop_2_1, dace.InterstateEdge(condition='i < (N - 1)'))
    sdfg.add_edge(loop_2_1, loop_2_2, dace.InterstateEdge())
    tmp2_edge = dace.InterstateEdge(assignments={'j': 'tmp'})
    sdfg.add_edge(loop_2_2, loop_2_3, tmp2_edge)
    sdfg.add_edge(loop_2_3, guard_2, dace.InterstateEdge(assignments={'i': 'i + 1'}))
    sdfg.add_edge(guard_2, end_state, dace.InterstateEdge(condition='i >= (N - 1)'))

    if with_raising:
        Pipeline([ControlFlowRaising(), PruneEmptyConditionalBranches()]).apply_pass(sdfg, {})

    # Test the pass.
    pipeline = Pipeline([ScalarWriteShadowScopes()])
    results = pipeline.apply_pass(sdfg, {})[ScalarWriteShadowScopes.__name__]

    assert results[0]['tmp'][(loop_1_1, tmp1_write)] == {(loop_1_3, loop1_read_tmp), (loop_1_2, tmp1_edge)}
    assert results[0]['tmp'][(loop_2_1, tmp2_write)] == {(loop_2_3, loop2_read_tmp), (loop_2_2, tmp2_edge)}
    assert results[0]['A'][None] == {(loop_1_1, a1_read), (loop_1_3, loop1_read_a), (loop_2_1, a2_read),
                                     (loop_2_3, loop2_read_a), (loop_1_3, loop1_write_a), (loop_2_3, loop2_write_a)}
    assert results[0]['B'][None] == {(loop_1_1, b1_read), (loop_1_3, loop1_read_b), (loop_2_1, b2_read),
                                     (loop_2_3, loop2_read_b), (loop_1_3, loop1_write_b), (loop_2_3, loop2_write_b)}


@pytest.mark.parametrize('with_raising', (False, True))
def test_loop_fake_shadow(with_raising):
    sdfg = dace.SDFG('loop_fake_shadow')
    sdfg.add_array('A', [1], dace.float64, transient=True)
    sdfg.add_array('B', [1], dace.float64)
    init = sdfg.add_state('init')
    guard = sdfg.add_state('guard')
    loop = sdfg.add_state('loop')
    loop2 = sdfg.add_state('loop2')
    end = sdfg.add_state('end')

    init_access = init.add_access('A')
    init_tasklet = init.add_tasklet('init', {}, {'a'}, 'a = 0')
    init.add_edge(init_tasklet, 'a', init_access, None, dace.Memlet('A[0]'))

    loop_access = loop.add_access('A')
    loop_access_b = loop.add_access('B')
    loop_tasklet_1 = loop.add_tasklet('loop_1', {}, {'a'}, 'a = 1')
    loop_tasklet_2 = loop.add_tasklet('loop_2', {'a'}, {'b'}, 'b = a')
    loop.add_edge(loop_tasklet_1, 'a', loop_access, None, dace.Memlet('A[0]'))
    loop.add_edge(loop_access, None, loop_tasklet_2, 'a', dace.Memlet('A[0]'))
    loop.add_edge(loop_tasklet_2, 'b', loop_access_b, None, dace.Memlet('B[0]'))

    loop2_access = loop2.add_access('A')
    loop2_access_b = loop2.add_access('B')
    loop2_tasklet_1 = loop2.add_tasklet('loop_1', {}, {'a'}, 'a = 2')
    loop2_tasklet_2 = loop2.add_tasklet('loop_2', {'a'}, {'b'}, 'b = a')
    loop2.add_edge(loop2_tasklet_1, 'a', loop2_access, None, dace.Memlet('A[0]'))
    loop2.add_edge(loop2_access, None, loop2_tasklet_2, 'a', dace.Memlet('A[0]'))
    loop2.add_edge(loop2_tasklet_2, 'b', loop2_access_b, None, dace.Memlet('B[0]'))

    end_access = end.add_access('A')
    end_access_b = end.add_access('B')
    end_tasklet = end.add_tasklet('end', {'a'}, {'b'}, 'b = a')
    end.add_edge(end_access, None, end_tasklet, 'a', dace.Memlet('A[0]'))
    end.add_edge(end_tasklet, 'b', end_access_b, None, dace.Memlet('B[0]'))

    sdfg.add_edge(init, guard, dace.InterstateEdge(assignments={'i': 0}))
    sdfg.add_edge(guard, loop, dace.InterstateEdge(condition='i < 10'))
    sdfg.add_edge(loop, loop2, dace.InterstateEdge())
    sdfg.add_edge(loop2, guard, dace.InterstateEdge(assignments={'i': 'i + 1'}))
    sdfg.add_edge(guard, end, dace.InterstateEdge(condition='i >= 10'))

    if with_raising:
        Pipeline([ControlFlowRaising(), PruneEmptyConditionalBranches()]).apply_pass(sdfg, {})

    ppl = Pipeline([ScalarWriteShadowScopes()])
    res = ppl.apply_pass(sdfg, {})[ScalarWriteShadowScopes.__name__]

    if with_raising:
        # Raised into a LoopRegion, the trip count of ``i = 0; i < 10`` is a concrete 10, so the
        # loop provably runs and the read in ``end`` sees what ``loop2`` wrote on the last
        # iteration -- the shadow is not fake here at all, and ``init``'s value is dead. There is
        # no upward-exposed read anywhere in the body (both accesses are written before they are
        # read in their own state), which is what would make a body write a fake shadow; see
        # ``test_loop_fake_complex_shadow`` for that shape, which still collapses.
        assert res[0]['A'][(loop, loop_access)] == {(loop, loop_access)}
        assert res[0]['A'][(loop2, loop2_access)] == {(loop2, loop2_access), (end, end_access)}
        assert res[0]['A'][None] == {(init, init_access)}
    else:
        # Without raising the guard is an ordinary state and there is no loop bound to analyze, so
        # the read in ``end`` may still be reached with zero iterations and everything collapses
        # into ``init``'s scope.
        assert res[0]['A'][(init, init_access)] == {(loop, loop_access), (loop2, loop2_access), (end, end_access)}


@pytest.mark.parametrize('with_raising', (False, True))
def test_loop_fake_complex_shadow(with_raising):
    sdfg = dace.SDFG('loop_fake_shadow')
    sdfg.add_array('A', [1], dace.float64, transient=True)
    sdfg.add_array('B', [1], dace.float64)
    init = sdfg.add_state('init')
    guard = sdfg.add_state('guard')
    loop = sdfg.add_state('loop')
    loop2 = sdfg.add_state('loop2')
    end = sdfg.add_state('end')

    init_access = init.add_access('A')
    init_tasklet = init.add_tasklet('init', {}, {'a'}, 'a = 0')
    init.add_edge(init_tasklet, 'a', init_access, None, dace.Memlet('A[0]'))

    loop_access = loop.add_access('A')
    loop_access_b = loop.add_access('B')
    loop_tasklet_2 = loop.add_tasklet('loop_2', {'a'}, {'b'}, 'b = a')
    loop.add_edge(loop_access, None, loop_tasklet_2, 'a', dace.Memlet('A[0]'))
    loop.add_edge(loop_tasklet_2, 'b', loop_access_b, None, dace.Memlet('B[0]'))

    loop2_access = loop2.add_access('A')
    loop2_access_b = loop2.add_access('B')
    loop2_tasklet_1 = loop2.add_tasklet('loop_1', {}, {'a'}, 'a = 2')
    loop2_tasklet_2 = loop2.add_tasklet('loop_2', {'a'}, {'b'}, 'b = a')
    loop2.add_edge(loop2_tasklet_1, 'a', loop2_access, None, dace.Memlet('A[0]'))
    loop2.add_edge(loop2_access, None, loop2_tasklet_2, 'a', dace.Memlet('A[0]'))
    loop2.add_edge(loop2_tasklet_2, 'b', loop2_access_b, None, dace.Memlet('B[0]'))

    sdfg.add_edge(init, guard, dace.InterstateEdge(assignments={'i': 0}))
    sdfg.add_edge(guard, loop, dace.InterstateEdge(condition='i < 10'))
    sdfg.add_edge(loop, loop2, dace.InterstateEdge())
    sdfg.add_edge(loop2, guard, dace.InterstateEdge(assignments={'i': 'i + 1'}))
    sdfg.add_edge(guard, end, dace.InterstateEdge(condition='i >= 10'))

    if with_raising:
        Pipeline([ControlFlowRaising(), PruneEmptyConditionalBranches()]).apply_pass(sdfg, {})

    ppl = Pipeline([ScalarWriteShadowScopes()])
    res = ppl.apply_pass(sdfg, {})[ScalarWriteShadowScopes.__name__]

    assert res[0]['A'][(init, init_access)] == {(loop, loop_access), (loop2, loop2_access)}


@pytest.mark.parametrize('with_raising', (False, True))
def test_loop_real_shadow(with_raising):
    sdfg = dace.SDFG('loop_fake_shadow')
    sdfg.add_array('A', [1], dace.float64, transient=True)
    sdfg.add_array('B', [1], dace.float64)
    init = sdfg.add_state('init')
    guard = sdfg.add_state('guard')
    loop = sdfg.add_state('loop')
    loop2 = sdfg.add_state('loop2')
    end = sdfg.add_state('end')

    init_access = init.add_access('A')
    init_tasklet = init.add_tasklet('init', {}, {'a'}, 'a = 0')
    init.add_edge(init_tasklet, 'a', init_access, None, dace.Memlet('A[0]'))

    loop_access = loop.add_access('A')
    loop_access_b = loop.add_access('B')
    loop_tasklet_1 = loop.add_tasklet('loop_1', {}, {'a'}, 'a = 1')
    loop_tasklet_2 = loop.add_tasklet('loop_2', {'a'}, {'b'}, 'b = a')
    loop.add_edge(loop_tasklet_1, 'a', loop_access, None, dace.Memlet('A[0]'))
    loop.add_edge(loop_access, None, loop_tasklet_2, 'a', dace.Memlet('A[0]'))
    loop.add_edge(loop_tasklet_2, 'b', loop_access_b, None, dace.Memlet('B[0]'))

    loop2_access = loop2.add_access('A')
    loop2_access_b = loop2.add_access('B')
    loop2_tasklet_1 = loop2.add_tasklet('loop_1', {}, {'a'}, 'a = 2')
    loop2_tasklet_2 = loop2.add_tasklet('loop_2', {'a'}, {'b'}, 'b = a')
    loop2.add_edge(loop2_tasklet_1, 'a', loop2_access, None, dace.Memlet('A[0]'))
    loop2.add_edge(loop2_access, None, loop2_tasklet_2, 'a', dace.Memlet('A[0]'))
    loop2.add_edge(loop2_tasklet_2, 'b', loop2_access_b, None, dace.Memlet('B[0]'))

    sdfg.add_edge(init, guard, dace.InterstateEdge(assignments={'i': 0}))
    sdfg.add_edge(guard, loop, dace.InterstateEdge(condition='i < 10'))
    sdfg.add_edge(loop, loop2, dace.InterstateEdge())
    sdfg.add_edge(loop2, guard, dace.InterstateEdge(assignments={'i': 'i + 1'}))
    sdfg.add_edge(guard, end, dace.InterstateEdge(condition='i >= 10'))

    if with_raising:
        Pipeline([ControlFlowRaising(), PruneEmptyConditionalBranches()]).apply_pass(sdfg, {})

    ppl = Pipeline([ScalarWriteShadowScopes()])
    res = ppl.apply_pass(sdfg, {})[ScalarWriteShadowScopes.__name__]

    assert res[0]['A'][(loop, loop_access)] == {(loop, loop_access)}
    assert res[0]['A'][(loop2, loop2_access)] == {(loop2, loop2_access)}


@pytest.mark.parametrize('with_raising', (False, True))
def test_dominationless_write_branch(with_raising):
    sdfg = dace.SDFG('dominationless_write_branch')
    sdfg.add_array('A', [1], dace.float64, transient=True)
    sdfg.add_array('B', [1], dace.float64)

    init = sdfg.add_state('init')
    guard = sdfg.add_state('guard')
    left = sdfg.add_state('left')
    merge = sdfg.add_state('merge')

    init_a = init.add_access('A')
    init_b = init.add_access('B')
    init_t1 = init.add_tasklet('init_1', {}, {'a'}, 'a = 0')
    init_t2 = init.add_tasklet('init_1', {'a'}, {'b'}, 'b = a + 1')
    init.add_edge(init_t1, 'a', init_a, None, dace.Memlet('A[0]'))
    init.add_edge(init_a, None, init_t2, 'a', dace.Memlet('A[0]'))
    init.add_edge(init_t2, 'b', init_b, None, dace.Memlet('B[0]'))

    guard_a = guard.add_access('A')
    guard_t1 = guard.add_tasklet('guard_1', {}, {'a'}, 'a = 1')
    guard.add_edge(guard_t1, 'a', guard_a, None, dace.Memlet('A[0]'))

    left_a = left.add_access('A')
    left_t1 = left.add_tasklet('left_1', {}, {'a'}, 'a = 2')
    left.add_edge(left_t1, 'a', left_a, None, dace.Memlet('A[0]'))

    merge_a = merge.add_access('A')
    merge_b = merge.add_access('B')
    merge_t1 = merge.add_tasklet('merge_1', {'a'}, {'b'}, 'b = a + 1')
    merge.add_edge(merge_a, None, merge_t1, 'a', dace.Memlet('A[0]'))
    merge.add_edge(merge_t1, 'b', merge_b, None, dace.Memlet('B[0]'))

    sdfg.add_edge(init, guard, dace.InterstateEdge())
    sdfg.add_edge(guard, left, dace.InterstateEdge(condition='B[0] < 10'))
    sdfg.add_edge(guard, merge, dace.InterstateEdge(condition='B[0] >= 10'))
    sdfg.add_edge(left, merge, dace.InterstateEdge())

    if with_raising:
        Pipeline([ControlFlowRaising(), PruneEmptyConditionalBranches()]).apply_pass(sdfg, {})

    ppl = Pipeline([ScalarWriteShadowScopes()])
    res = ppl.apply_pass(sdfg, {})[ScalarWriteShadowScopes.__name__]

    assert res[0]['A'][(init, init_a)] == {(init, init_a)}
    assert res[0]['A'][(guard, guard_a)] == {(merge, merge_a), (left, left_a)}


# ---------------------------------------------------------------------------------------------- #
#  Dominating writes found INSIDE a control flow region
#
#  A canonicalized kernel's top-level blocks are typically all ``LoopRegion``s. Accepting only an
#  ``SDFGState`` as a dominating write block means the walk up the idom chain never finds one, and
#  EVERY access of such a container -- the producing write included -- falls into the undominated
#  (``None``) scope. ``must_write_state`` closes that, but only where a must-def can be PROVEN;
#  each refusal below is the difference between a missed optimization and a wrong answer.
# ---------------------------------------------------------------------------------------------- #


def region_scope_fixture(name: str, condition: str, init: str = 'i = 0', guard_body_write: bool = False):
    """``for i in <condition>: A = 1`` followed by a state reading ``A``.

    :param name: SDFG name.
    :param condition: The loop condition, which decides whether the trip count is provable.
    :param init: The loop's init statement.
    :param guard_body_write: Put the body's write under a non-exhaustive ``if`` inside the loop.
    :returns: ``(sdfg, loop_write_access, end_read_access)``.
    """
    sdfg = dace.SDFG(name)
    sdfg.add_array('A', [1], dace.float64, transient=True)
    sdfg.add_array('B', [1], dace.float64)
    sdfg.add_symbol('i', dace.int64)

    loop = LoopRegion('loop', condition, 'i', init, 'i = i + 1')
    sdfg.add_node(loop, is_start_block=True)
    body = loop.add_state('body', is_start_block=True)
    if guard_body_write:
        branch = ConditionalBlock('guarded')
        loop.add_node(branch)
        loop.add_edge(body, branch, dace.InterstateEdge())
        body = ControlFlowRegion('then', sdfg=sdfg)
        branch.add_branch(CodeBlock('i > 2'), body)
        body = body.add_state('then_body', is_start_block=True)

    loop_write = body.add_access('A')
    write_tasklet = body.add_tasklet('w', {}, {'a'}, 'a = 1')
    body.add_edge(write_tasklet, 'a', loop_write, None, dace.Memlet('A[0]'))

    end = sdfg.add_state('end')
    sdfg.add_edge(loop, end, dace.InterstateEdge())
    end_read = end.add_access('A')
    end_b = end.add_access('B')
    end_tasklet = end.add_tasklet('r', {'a'}, {'b'}, 'b = a')
    end.add_edge(end_read, None, end_tasklet, 'a', dace.Memlet('A[0]'))
    end.add_edge(end_tasklet, 'b', end_b, None, dace.Memlet('B[0]'))

    sdfg.validate()
    return sdfg, loop_write, end_read


def write_scopes(sdfg: dace.SDFG):
    return Pipeline([ScalarWriteShadowScopes()]).apply_pass(sdfg, {})[ScalarWriteShadowScopes.__name__][0]


def state_named(sdfg: dace.SDFG, label: str):
    return next(s for s in sdfg.all_states() if s.label == label)


def test_loop_region_write_dominates_read_after_loop():
    """Positive: a concrete trip count makes the body's write a must-def after the loop."""
    sdfg, loop_write, end_read = region_scope_fixture('region_write_concrete', 'i < 4')
    res = write_scopes(sdfg)

    body, end = state_named(sdfg, 'body'), state_named(sdfg, 'end')
    assert res['A'][(body, loop_write)] == {(end, end_read)}
    assert not res['A'][None], 'the producing write must not fall into the undominated scope'


def test_symbolic_trip_count_loop_is_refused():
    """Refusal: ``for i in range(N)`` may run zero times.

    The nonnegative-symbol assumption gives ``N >= 0``, not ``N >= 1``, so the write inside the
    body defines nothing after the loop and the read stays undominated.
    """
    sdfg, loop_write, end_read = region_scope_fixture('region_write_symbolic', 'i < N')
    sdfg.add_symbol('N', dace.int64)
    res = write_scopes(sdfg)

    body, end = state_named(sdfg, 'body'), state_named(sdfg, 'end')
    assert (end, end_read) in res['A'][None]
    assert (body, loop_write) in res['A'][None]


def test_conditional_write_inside_provable_loop_is_refused():
    """Refusal: the write is unconditional in the LOOP but not in the loop BODY.

    A block that must-write has to dominate every sink of its region; a write nested under a
    further conditional does not, however many times the loop runs.
    """
    sdfg, _, end_read = region_scope_fixture('region_write_guarded', 'i < 4', guard_body_write=True)
    res = write_scopes(sdfg)

    assert (state_named(sdfg, 'end'), end_read) in res['A'][None]


def conditional_block_fixture(name: str, exhaustive: bool):
    """``if c: A = 1 [else: A = 2]`` followed by a state reading ``A``."""
    sdfg = dace.SDFG(name)
    sdfg.add_array('A', [1], dace.float64, transient=True)
    sdfg.add_array('B', [1], dace.float64)
    sdfg.add_symbol('c', dace.int64)

    branch = ConditionalBlock('branch')
    sdfg.add_node(branch, is_start_block=True)
    writes = []
    conditions = [CodeBlock('c > 0')] + ([None] if exhaustive else [])
    for idx, cond in enumerate(conditions):
        region = ControlFlowRegion(f'br{idx}', sdfg=sdfg)
        branch.add_branch(cond, region)
        state = region.add_state(f'br{idx}_body', is_start_block=True)
        access = state.add_access('A')
        tasklet = state.add_tasklet(f'w{idx}', {}, {'a'}, f'a = {idx}')
        state.add_edge(tasklet, 'a', access, None, dace.Memlet('A[0]'))
        writes.append((state, access))

    end = sdfg.add_state('end')
    sdfg.add_edge(branch, end, dace.InterstateEdge())
    end_read = end.add_access('A')
    end_b = end.add_access('B')
    end_tasklet = end.add_tasklet('r', {'a'}, {'b'}, 'b = a')
    end.add_edge(end_read, None, end_tasklet, 'a', dace.Memlet('A[0]'))
    end.add_edge(end_tasklet, 'b', end_b, None, dace.Memlet('B[0]'))

    sdfg.validate()
    return sdfg, writes, (end, end_read)


@pytest.mark.parametrize('exhaustive', (False, True))
def test_conditional_block_is_refused(exhaustive):
    """Refusal: a ``ConditionalBlock`` is never a dominating write, exhaustive or not.

    Without an ``else`` the premise itself fails -- no branch is guaranteed to run. WITH an
    ``else`` in which every branch writes, the must-def premise holds but the result dict keys a
    scope on exactly ONE ``(state, node)`` pair; naming one branch's write would let a consumer
    version it while leaving its sibling behind, so the block is refused on shape as well.
    """
    sdfg, writes, end_access = conditional_block_fixture(f'conditional_{exhaustive}', exhaustive)
    res = write_scopes(sdfg)

    assert end_access in res['A'][None]
    for write in writes:
        assert write not in res['A'] or res['A'][write] == {write}


def test_loop_carried_read_is_not_split_from_the_body_write():
    """The producer loop is a must-def, yet the consumer loop's own carry must stay in one scope.

    ``loop A`` (provable trip count) writes ``A``; ``loop B`` reads it at the top of its body and
    rewrites it at the bottom, so from the second iteration on the read sees B's own write, not
    A's. Attributing the read to A while keeping B's write as a separate root would let the
    consumer version them apart and leave the read on a container nobody writes -- the coarsening
    step has to fold them back into a single scope.
    """
    sdfg = dace.SDFG('loop_carried_after_region_write')
    sdfg.add_array('A', [1], dace.float64, transient=True)
    sdfg.add_array('B', [1], dace.float64)
    sdfg.add_symbol('i', dace.int64)
    sdfg.add_symbol('j', dace.int64)

    producer = LoopRegion('producer', 'i < 4', 'i', 'i = 0', 'i = i + 1')
    sdfg.add_node(producer, is_start_block=True)
    pbody = producer.add_state('pbody', is_start_block=True)
    pw = pbody.add_access('A')
    pt = pbody.add_tasklet('pw', {}, {'a'}, 'a = 1')
    pbody.add_edge(pt, 'a', pw, None, dace.Memlet('A[0]'))

    consumer = LoopRegion('consumer', 'j < 4', 'j', 'j = 0', 'j = j + 1')
    sdfg.add_node(consumer)
    sdfg.add_edge(producer, consumer, dace.InterstateEdge())
    cread = consumer.add_state('cread', is_start_block=True)
    cwrite = consumer.add_state('cwrite')
    cafter = consumer.add_state('cafter')
    consumer.add_edge(cread, cwrite, dace.InterstateEdge())
    consumer.add_edge(cwrite, cafter, dace.InterstateEdge())

    cr = cread.add_access('A')
    cb = cread.add_access('B')
    crt = cread.add_tasklet('cr', {'a'}, {'b'}, 'b = a')
    cread.add_edge(cr, None, crt, 'a', dace.Memlet('A[0]'))
    cread.add_edge(crt, 'b', cb, None, dace.Memlet('B[0]'))

    cw = cwrite.add_access('A')
    cwt = cwrite.add_tasklet('cw', {}, {'a'}, 'a = 2')
    cwrite.add_edge(cwt, 'a', cw, None, dace.Memlet('A[0]'))

    ca = cafter.add_access('A')
    cab = cafter.add_access('B')
    cat = cafter.add_tasklet('ca', {'a'}, {'b'}, 'b = a')
    cafter.add_edge(ca, None, cat, 'a', dace.Memlet('A[0]'))
    cafter.add_edge(cat, 'b', cab, None, dace.Memlet('B[0]'))
    sdfg.validate()

    res = write_scopes(sdfg)

    roots = [w for w in res['A'] if w is not None and res['A'][w]]
    assert len(roots) == 1, f'the loop-carried chain was split into {len(roots)} scopes: {roots}'
    scope = res['A'][roots[0]]
    assert (cread, cr) in scope and (cafter, ca) in scope
    assert (cwrite, cw) in scope


if __name__ == '__main__':
    test_scalar_write_shadow_split(False)
    test_scalar_write_shadow_fused(False)
    test_scalar_write_shadow_interstate_self(False)
    test_scalar_write_shadow_interstate_pred(False)
    test_loop_fake_shadow(False)
    test_loop_fake_complex_shadow(False)
    test_loop_real_shadow(False)
    test_dominationless_write_branch(False)
    test_scalar_write_shadow_split(True)
    test_scalar_write_shadow_fused(True)
    test_scalar_write_shadow_interstate_self(True)
    test_scalar_write_shadow_interstate_pred(True)
    test_loop_fake_shadow(True)
    test_loop_fake_complex_shadow(True)
    test_loop_real_shadow(True)
    test_dominationless_write_branch(True)
