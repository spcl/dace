# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
""" Tests the scalar write shadowing analysis pass. """

import pytest

import dace
from dace.transformation.pass_pipeline import Pipeline
from dace.transformation.passes.analysis import ScalarWriteShadowScopes


def test_scalar_write_shadow_split():
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

    # Test the pass.
    pipeline = Pipeline([ScalarWriteShadowScopes()])
    results = pipeline.apply_pass(sdfg, {})[ScalarWriteShadowScopes.__name__]

    assert results[0]['tmp'][(loop_1_1, tmp1_write)] == set([(loop_1_2, loop1_read_tmp)])
    assert results[0]['tmp'][(loop_2_1, tmp2_write)] == set([(loop_2_2, loop2_read_tmp)])
    assert results[0]['A'][None] == set([(loop_1_1, a1_read), (loop_1_2, loop1_read_a), (loop_2_1, a2_read),
                                         (loop_2_2, loop2_read_a)])
    assert results[0]['B'][None] == set([(loop_1_1, b1_read), (loop_1_2, loop1_read_b), (loop_2_1, b2_read),
                                         (loop_2_2, loop2_read_b)])


def test_scalar_write_shadow_fused():
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

    # Test the pass.
    pipeline = Pipeline([ScalarWriteShadowScopes()])
    results = pipeline.apply_pass(sdfg, {})[ScalarWriteShadowScopes.__name__]

    assert results[0]['tmp'][(loop_1, tmp1_read_write)] == set([(loop_1, tmp1_read_write)])
    assert results[0]['tmp'][(loop_2, tmp2_read_write)] == set([(loop_2, tmp2_read_write)])
    assert results[0]['A'][None] == set([(loop_1, a1_read), (loop_2, a2_read)])
    assert results[0]['B'][None] == set([(loop_1, b1_read), (loop_2, b2_read)])


def test_scalar_write_shadow_interstate_self():
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

    # Test the pass.
    pipeline = Pipeline([ScalarWriteShadowScopes()])
    results = pipeline.apply_pass(sdfg, {})[ScalarWriteShadowScopes.__name__]

    assert results[0]['tmp'][(loop_1_1, tmp1_write)] == set([(loop_1_2, loop1_read_tmp), (loop_1_1, tmp1_edge)])
    assert results[0]['tmp'][(loop_2_1, tmp2_write)] == set([(loop_2_2, loop2_read_tmp), (loop_2_1, tmp2_edge)])
    assert results[0]['A'][None] == set([(loop_1_1, a1_read), (loop_1_2, loop1_read_a), (loop_2_1, a2_read),
                                         (loop_2_2, loop2_read_a)])
    assert results[0]['B'][None] == set([(loop_1_1, b1_read), (loop_1_2, loop1_read_b), (loop_2_1, b2_read),
                                         (loop_2_2, loop2_read_b)])


def test_scalar_write_shadow_interstate_pred():
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

    # Test the pass.
    pipeline = Pipeline([ScalarWriteShadowScopes()])
    results = pipeline.apply_pass(sdfg, {})[ScalarWriteShadowScopes.__name__]

    assert results[0]['tmp'][(loop_1_1, tmp1_write)] == set([(loop_1_3, loop1_read_tmp), (loop_1_2, tmp1_edge)])
    assert results[0]['tmp'][(loop_2_1, tmp2_write)] == set([(loop_2_3, loop2_read_tmp), (loop_2_2, tmp2_edge)])
    assert results[0]['A'][None] == set([(loop_1_1, a1_read), (loop_1_3, loop1_read_a), (loop_2_1, a2_read),
                                         (loop_2_3, loop2_read_a)])
    assert results[0]['B'][None] == set([(loop_1_1, b1_read), (loop_1_3, loop1_read_b), (loop_2_1, b2_read),
                                         (loop_2_3, loop2_read_b)])


def test_loop_fake_shadow():
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

    ppl = Pipeline([ScalarWriteShadowScopes()])
    res = ppl.apply_pass(sdfg, {})[ScalarWriteShadowScopes.__name__]

    assert res[0]['A'][(init, init_access)] == {(loop, loop_access), (loop2, loop2_access), (end, end_access)}


def test_loop_fake_complex_shadow():
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

    ppl = Pipeline([ScalarWriteShadowScopes()])
    res = ppl.apply_pass(sdfg, {})[ScalarWriteShadowScopes.__name__]

    assert res[0]['A'][(init, init_access)] == {(loop, loop_access), (loop2, loop2_access)}


def test_loop_real_shadow():
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

    ppl = Pipeline([ScalarWriteShadowScopes()])
    res = ppl.apply_pass(sdfg, {})[ScalarWriteShadowScopes.__name__]

    assert res[0]['A'][(loop, loop_access)] == {(loop, loop_access)}
    assert res[0]['A'][(loop2, loop2_access)] == {(loop2, loop2_access)}


if __name__ == '__main__':
    test_scalar_write_shadow_split()
    test_scalar_write_shadow_fused()
    test_scalar_write_shadow_interstate_self()
    test_scalar_write_shadow_interstate_pred()
    test_loop_fake_shadow()
    test_loop_fake_complex_shadow()
    test_loop_real_shadow()
