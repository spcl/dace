# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import copy
import dace
import pytest
from dace.sdfg import nodes, utils as sdutils
from dace.transformation.dataflow import MapFission
from dace.transformation.interstate import InlineSDFG
from dace.transformation.helpers import nest_state_subgraph
import numpy as np


def mapfission_sdfg():
    sdfg = dace.SDFG('mapfission')
    sdfg.add_array('A', [4], dace.float64)
    sdfg.add_array('B', [2], dace.float64)
    sdfg.add_scalar('scal', dace.float64, transient=True)
    sdfg.add_scalar('s1', dace.float64, transient=True)
    sdfg.add_transient('s2', [2], dace.float64)
    sdfg.add_transient('s3out', [1], dace.float64)
    state = sdfg.add_state()

    # Nodes
    rnode = state.add_read('A')
    ome, omx = state.add_map('outer', dict(i='0:2'))
    t1 = state.add_tasklet('one', {'a'}, {'b'}, 'b = a[0] + a[1]')
    ime2, imx2 = state.add_map('inner', dict(j='0:2'))
    t2 = state.add_tasklet('two', {'a'}, {'b'}, 'b = a * 2')
    s24node = state.add_access('s2')
    s34node = state.add_access('s3out')
    ime3, imx3 = state.add_map('inner', dict(j='0:2'))
    t3 = state.add_tasklet('three', {'a'}, {'b'}, 'b = a[0] * 3')
    scalar = state.add_tasklet('scalar', {}, {'out'}, 'out = 5.0')
    t4 = state.add_tasklet('four', {'ione', 'itwo', 'ithree', 'sc'}, {'out'},
                           'out = ione + itwo[0] * itwo[1] + ithree + sc')
    wnode = state.add_write('B')

    # Edges
    state.add_nedge(ome, scalar, dace.Memlet())
    state.add_memlet_path(rnode, ome, t1, memlet=dace.Memlet.simple('A', '2*i:2*i+2'), dst_conn='a')
    state.add_memlet_path(rnode, ome, ime2, t2, memlet=dace.Memlet.simple('A', '2*i+j'), dst_conn='a')
    state.add_memlet_path(t2, imx2, s24node, memlet=dace.Memlet.simple('s2', 'j'), src_conn='b')
    state.add_memlet_path(rnode, ome, ime3, t3, memlet=dace.Memlet.simple('A', '2*i:2*i+2'), dst_conn='a')
    state.add_memlet_path(t3, imx3, s34node, memlet=dace.Memlet.simple('s3out', '0'), src_conn='b')

    state.add_edge(t1, 'b', t4, 'ione', dace.Memlet.simple('s1', '0'))
    state.add_edge(s24node, None, t4, 'itwo', dace.Memlet.simple('s2', '0:2'))
    state.add_edge(s34node, None, t4, 'ithree', dace.Memlet.simple('s3out', '0'))
    state.add_edge(scalar, 'out', t4, 'sc', dace.Memlet.simple('scal', '0'))
    state.add_memlet_path(t4, omx, wnode, memlet=dace.Memlet.simple('B', 'i'), src_conn='out')

    sdfg.validate()
    return sdfg


def config():
    A = np.random.rand(4)
    expected = np.zeros([2], dtype=np.float64)
    expected[0] = (A[0] + A[1]) + (A[0] * 2 * A[1] * 2) + (A[0] * 3) + 5.0
    expected[1] = (A[2] + A[3]) + (A[2] * 2 * A[3] * 2) + (A[2] * 3) + 5.0
    return A, expected


def test_subgraph():
    A, expected = config()
    B = np.random.rand(2)

    graph = mapfission_sdfg()
    assert graph.apply_transformations(MapFission) > 0
    graph(A=A, B=B)

    assert np.allclose(B, expected)


def test_nested_sdfg():
    A, expected = config()
    B = np.random.rand(2)

    # Nest the subgraph within the outer map, then apply transformation
    graph = mapfission_sdfg()
    state = graph.nodes()[0]
    topmap = next(node for node in state.nodes() if isinstance(node, nodes.MapEntry) and node.label == 'outer')
    subgraph = state.scope_subgraph(topmap, include_entry=False, include_exit=False)
    nest_state_subgraph(graph, state, subgraph)
    assert graph.apply_transformations(MapFission) > 0
    graph(A=A, B=B)
    assert np.allclose(B, expected)


def test_nested_transient():
    """ Test nested SDFGs with transients. """

    # Inner SDFG
    nsdfg = dace.SDFG('nested')
    nsdfg.add_array('a', [1], dace.float64)
    nsdfg.add_array('b', [1], dace.float64)
    nsdfg.add_transient('t', [1], dace.float64)

    # a->t state
    nstate = nsdfg.add_state()
    irnode = nstate.add_read('a')
    task = nstate.add_tasklet('t1', {'inp'}, {'out'}, 'out = 2*inp')
    iwnode = nstate.add_write('t')
    nstate.add_edge(irnode, None, task, 'inp', dace.Memlet.simple('a', '0'))
    nstate.add_edge(task, 'out', iwnode, None, dace.Memlet.simple('t', '0'))

    # t->a state
    first_state = nstate
    nstate = nsdfg.add_state()
    irnode = nstate.add_read('t')
    task = nstate.add_tasklet('t2', {'inp'}, {'out'}, 'out = 3*inp')
    iwnode = nstate.add_write('b')
    nstate.add_edge(irnode, None, task, 'inp', dace.Memlet.simple('t', '0'))
    nstate.add_edge(task, 'out', iwnode, None, dace.Memlet.simple('b', '0'))

    nsdfg.add_edge(first_state, nstate, dace.InterstateEdge())

    # Outer SDFG
    sdfg = dace.SDFG('nested_transient_fission')
    sdfg.add_array('A', [2], dace.float64)
    state = sdfg.add_state()
    rnode = state.add_read('A')
    wnode = state.add_write('A')
    me, mx = state.add_map('outer', dict(i='0:2'))
    nsdfg_node = state.add_nested_sdfg(nsdfg, {'a'}, {'b'})
    state.add_memlet_path(rnode, me, nsdfg_node, dst_conn='a', memlet=dace.Memlet.simple('A', 'i'))
    state.add_memlet_path(nsdfg_node, mx, wnode, src_conn='b', memlet=dace.Memlet.simple('A', 'i'))

    assert sdfg.apply_transformations_repeated(MapFission) > 0

    # Test
    A = np.random.rand(2)
    expected = A * 6
    sdfg(A=A)
    assert np.allclose(A, expected)


def test_inputs_outputs():
    """
    Test subgraphs where the computation modules that are in the middle
    connect to the outside.
    """

    sdfg = dace.SDFG('inputs_outputs_fission')
    sdfg.add_array('in1', [2], dace.float64)
    sdfg.add_array('in2', [2], dace.float64)
    sdfg.add_scalar('tmp', dace.float64, transient=True)
    sdfg.add_array('out1', [2], dace.float64)
    sdfg.add_array('out2', [2], dace.float64)
    state = sdfg.add_state()
    in1 = state.add_read('in1')
    in2 = state.add_read('in2')
    out1 = state.add_write('out1')
    out2 = state.add_write('out2')
    me, mx = state.add_map('outer', dict(i='0:2'))
    t1 = state.add_tasklet('t1', {'i1'}, {'o1', 'o2'}, 'o1 = i1 * 2; o2 = i1 * 5')
    t2 = state.add_tasklet('t2', {'i1', 'i2'}, {'o1'}, 'o1 = i1 * i2')
    state.add_memlet_path(in1, me, t1, dst_conn='i1', memlet=dace.Memlet.simple('in1', 'i'))
    state.add_memlet_path(in2, me, t2, dst_conn='i2', memlet=dace.Memlet.simple('in2', 'i'))
    state.add_edge(t1, 'o1', t2, 'i1', dace.Memlet.simple('tmp', '0'))
    state.add_memlet_path(t2, mx, out1, src_conn='o1', memlet=dace.Memlet.simple('out1', 'i'))
    state.add_memlet_path(t1, mx, out2, src_conn='o2', memlet=dace.Memlet.simple('out2', 'i'))

    assert sdfg.apply_transformations(MapFission) > 0

    # Test
    A, B, C, D = tuple(np.random.rand(2) for _ in range(4))
    expected_C = (A * 2) * B
    expected_D = A * 5
    sdfg(in1=A, in2=B, out1=C, out2=D)
    assert np.allclose(C, expected_C)
    assert np.allclose(D, expected_D)


def test_multidim():
    sdfg = dace.SDFG('mapfission_multidim')
    sdfg.add_array('A', [2, 3], dace.float64)
    state = sdfg.add_state()
    me, mx = state.add_map('outer', dict(i='0:2', j='0:3'))

    nsdfg = dace.SDFG('nested')
    nsdfg.add_array('a', [1], dace.float64)
    nstate = nsdfg.add_state()
    t = nstate.add_tasklet('reset', {}, {'out'}, 'out = 0')
    a = nstate.add_write('a')
    nstate.add_edge(t, 'out', a, None, dace.Memlet.simple('a', '0'))
    nsdfg_node = state.add_nested_sdfg(nsdfg, {}, {'a'})

    state.add_edge(me, None, nsdfg_node, None, dace.Memlet())
    anode = state.add_write('A')
    state.add_memlet_path(nsdfg_node, mx, anode, src_conn='a', memlet=dace.Memlet.simple('A', 'i,j'))

    assert sdfg.apply_transformations_repeated(MapFission) > 0

    # Test
    A = np.random.rand(2, 3)
    sdfg(A=A)
    assert np.allclose(A, np.zeros_like(A))


def test_offsets():
    sdfg = dace.SDFG('mapfission_offsets')
    sdfg.add_array('A', [20], dace.float64)
    sdfg.add_scalar('interim', dace.float64, transient=True)
    state = sdfg.add_state()
    me, mx = state.add_map('outer', dict(i='10:20'))

    t1 = state.add_tasklet('addone', {'a'}, {'b'}, 'b = a + 1')
    t2 = state.add_tasklet('addtwo', {'a'}, {'b'}, 'b = a + 2')

    aread = state.add_read('A')
    awrite = state.add_write('A')
    state.add_memlet_path(aread, me, t1, dst_conn='a', memlet=dace.Memlet.simple('A', 'i'))
    state.add_edge(t1, 'b', t2, 'a', dace.Memlet.simple('interim', '0'))
    state.add_memlet_path(t2, mx, awrite, src_conn='b', memlet=dace.Memlet.simple('A', 'i'))

    assert sdfg.apply_transformations(MapFission) > 0

    dace.propagate_memlets_sdfg(sdfg)
    sdfg.validate()

    # Test
    A = np.random.rand(20)
    expected = A.copy()
    expected[10:] += 3
    sdfg(A=A)
    assert np.allclose(A, expected)


def test_offsets_array():
    sdfg = dace.SDFG('mapfission_offsets2')
    sdfg.add_array('A', [20], dace.float64)
    sdfg.add_array('interim', [1], dace.float64, transient=True)
    state = sdfg.add_state()
    me, mx = state.add_map('outer', dict(i='10:20'))

    t1 = state.add_tasklet('addone', {'a'}, {'b'}, 'b = a + 1')
    interim = state.add_access('interim')
    t2 = state.add_tasklet('addtwo', {'a'}, {'b'}, 'b = a + 2')

    aread = state.add_read('A')
    awrite = state.add_write('A')
    state.add_memlet_path(aread, me, t1, dst_conn='a', memlet=dace.Memlet.simple('A', 'i'))
    state.add_edge(t1, 'b', interim, None, dace.Memlet.simple('interim', '0'))
    state.add_edge(interim, None, t2, 'a', dace.Memlet.simple('interim', '0'))
    state.add_memlet_path(t2, mx, awrite, src_conn='b', memlet=dace.Memlet.simple('A', 'i'))

    assert sdfg.apply_transformations(MapFission) > 0

    dace.propagate_memlets_sdfg(sdfg)
    sdfg.validate()

    # Test
    A = np.random.rand(20)
    expected = A.copy()
    expected[10:] += 3
    sdfg(A=A)
    assert np.allclose(A, expected)


def test_mapfission_with_symbols():
    """
    Tests MapFission in the case of a Map containing a NestedSDFG that is using some symbol from the top-level SDFG
    missing from the NestedSDFG's symbol mapping. Please note that this is an unusual case that is difficult to
    reproduce and ultimately unrelated to MapFission. Consider solving the underlying issue and then deleting this
    test and the corresponding (obsolete) code in MapFission.
    """

    M, N = dace.symbol('M'), dace.symbol('N')

    sdfg = dace.SDFG('tasklet_code_with_symbols')
    sdfg.add_array('A', (M, N), dace.int32)
    sdfg.add_array('B', (M, N), dace.int32)

    state = sdfg.add_state('parent', is_start_block=True)
    me, mx = state.add_map('parent_map', {'i': '0:N'})

    nsdfg = dace.SDFG('nested_sdfg')
    nsdfg.add_scalar('inner_A', dace.int32)
    nsdfg.add_scalar('inner_B', dace.int32)

    nstate = nsdfg.add_state('child', is_start_block=True)
    na = nstate.add_access('inner_A')
    nb = nstate.add_access('inner_B')
    ta = nstate.add_tasklet('tasklet_A', {}, {'__out'}, '__out = M')
    tb = nstate.add_tasklet('tasklet_B', {}, {'__out'}, '__out = M')
    nstate.add_edge(ta, '__out', na, None, dace.Memlet.from_array('inner_A', nsdfg.arrays['inner_A']))
    nstate.add_edge(tb, '__out', nb, None, dace.Memlet.from_array('inner_B', nsdfg.arrays['inner_B']))

    a = state.add_access('A')
    b = state.add_access('B')
    t = state.add_nested_sdfg(nsdfg, {}, {'inner_A', 'inner_B'})
    state.add_nedge(me, t, dace.Memlet())
    state.add_memlet_path(t, mx, a, memlet=dace.Memlet('A[0, i]'), src_conn='inner_A')
    state.add_memlet_path(t, mx, b, memlet=dace.Memlet('B[0, i]'), src_conn='inner_B')

    num = sdfg.apply_transformations_repeated(MapFission)
    assert num == 1

    A = np.ndarray((2, 10), dtype=np.int32)
    B = np.ndarray((2, 10), dtype=np.int32)
    sdfg(A=A, B=B, M=2, N=10)

    ref = np.full((10, ), fill_value=2, dtype=np.int32)

    assert np.array_equal(A[0], ref)
    assert np.array_equal(B[0], ref)


def test_two_edges_through_map():
    """
    Tests MapFission in the case of a Map with a component that has two inputs from a single data container. In such
    cases, using `fill_scope_connectors` will lead to broken Map connectors. The tests confirms that new code in the
    transformation manually adding the appropriate Map connectors works properly.
    """

    N = dace.symbol('N')

    sdfg = dace.SDFG('two_edges_through_map')
    sdfg.add_array('A', (N, ), dace.int32)
    sdfg.add_array('B', (N, ), dace.int32)

    state = sdfg.add_state('parent', is_start_block=True)
    me, mx = state.add_map('parent_map', {'i': '0:N'})

    nsdfg = dace.SDFG('nested_sdfg')
    nsdfg.add_array('inner_A', (N, ), dace.int32)
    nsdfg.add_scalar('inner_B', dace.int32)

    nstate = nsdfg.add_state('child', is_start_block=True)
    na = nstate.add_access('inner_A')
    nb = nstate.add_access('inner_B')
    t = nstate.add_tasklet('tasklet', {'__in1', '__in2'}, {'__out'}, '__out = __in1 + __in2')
    nstate.add_edge(na, None, t, '__in1', dace.Memlet('inner_A[i]'))
    nstate.add_edge(na, None, t, '__in2', dace.Memlet('inner_A[N-i-1]'))
    nstate.add_edge(t, '__out', nb, None, dace.Memlet.from_array('inner_B', nsdfg.arrays['inner_B']))

    a = state.add_access('A')
    b = state.add_access('B')
    t = state.add_nested_sdfg(nsdfg, {'inner_A'}, {'inner_B'}, {'N': 'N', 'i': 'i'})
    state.add_memlet_path(a, me, t, memlet=dace.Memlet.from_array('A', sdfg.arrays['A']), dst_conn='inner_A')
    state.add_memlet_path(t, mx, b, memlet=dace.Memlet('B[i]'), src_conn='inner_B')

    num = sdfg.apply_transformations_repeated(MapFission)
    assert num == 1

    A = np.arange(10, dtype=np.int32)
    B = np.ndarray((10, ), dtype=np.int32)
    sdfg(A=A, B=B, N=10)

    ref = np.full((10, ), fill_value=9, dtype=np.int32)

    assert np.array_equal(B, ref)


def test_if_scope():

    @dace.program
    def map_with_if(A: dace.int32[10]):
        for i in dace.map[0:10]:
            if i < 5:
                A[i] = 0
            else:
                A[i] = 1

    ref = np.array([0] * 5 + [1] * 5, dtype=np.int32)

    sdfg = map_with_if.to_sdfg()
    val0 = np.ndarray((10, ), dtype=np.int32)
    sdfg(A=val0)
    assert np.array_equal(val0, ref)

    sdfg.apply_transformations_repeated(MapFission)

    val1 = np.ndarray((10, ), dtype=np.int32)
    sdfg(A=val1)
    assert np.array_equal(val1, ref)


def test_if_scope_2():

    @dace.program
    def map_with_if_2(A: dace.int32[10]):
        for i in dace.map[0:10]:
            j = i < 5
            if j:
                A[i] = 0
            else:
                A[i] = 1

    ref = np.array([0] * 5 + [1] * 5, dtype=np.int32)

    sdfg = map_with_if_2.to_sdfg()
    val0 = np.ndarray((10, ), dtype=np.int32)
    sdfg(A=val0)
    assert np.array_equal(val0, ref)

    sdfg.apply_transformations_repeated(MapFission)

    val1 = np.ndarray((10, ), dtype=np.int32)
    sdfg(A=val1)
    assert np.array_equal(val1, ref)


def test_array_copy_outside_scope():
    """
    This test checks for two issues occuring when MapFission applies on a NestedSDFG with a state-subgraph
    containing copies among AccessNodes. In such cases, these copies may end up outside the scope of the generated
    Maps (after MapFssion), potentially leading to the following errors:
    1. The memlet subset corresponding to a NestedSDFG connector (input/output) may have its dimensionality
    erroneously increased.
    2. The memlet subset corresponding to a NestedSDFG connector (input/output) may not be propagated even if it uses
    the Map's parameters.
    """

    sdfg = dace.SDFG('array_copy_outside_scope')
    iname, _ = sdfg.add_array('inp', (10, ), dtype=dace.int32)
    oname, _ = sdfg.add_array('out', (10, ), dtype=dace.int32)

    nsdfg = dace.SDFG('nested_sdfg')
    niname, nidesc = nsdfg.add_array('ninp', (1, ), dtype=dace.int32)
    ntname, ntdesc = nsdfg.add_scalar('ntmp', dtype=dace.int32, transient=True)
    noname, nodesc = nsdfg.add_array('nout', (1, ), dtype=dace.int32)

    nstate = nsdfg.add_state('nmain')
    ninode = nstate.add_access(niname)
    ntnode = nstate.add_access(ntname)
    nonode = nstate.add_access(noname)
    tasklet = nstate.add_tasklet('tasklet', {'__inp'}, {'__out'}, '__out = __inp + 1')
    nstate.add_edge(ninode, None, tasklet, '__inp', dace.Memlet.from_array(niname, nidesc))
    nstate.add_edge(tasklet, '__out', ntnode, None, dace.Memlet.from_array(ntname, ntdesc))
    nstate.add_nedge(ntnode, nonode, dace.Memlet.from_array(noname, nodesc))

    state = sdfg.add_state('main')
    inode = state.add_access(iname)
    onode = state.add_access(oname)
    me, mx = state.add_map('map', {'i': '0:10'})
    snode = state.add_nested_sdfg(nsdfg, {'ninp'}, {'nout'})
    state.add_memlet_path(inode, me, snode, memlet=dace.Memlet(data=iname, subset='i'), dst_conn='ninp')
    state.add_memlet_path(snode, mx, onode, memlet=dace.Memlet(data=oname, subset='i'), src_conn='nout')

    # Issue no. 1 will be caught by validation after MapFission
    sdfg.apply_transformations(MapFission)

    # Issue no. 2 will be caught by code-generation due to `i` existing in a memlet outside the Map's scope.
    A = np.arange(10, dtype=np.int32)
    B = np.empty((10, ), dtype=np.int32)
    sdfg(inp=A, out=B)
    assert np.array_equal(A + 1, B)


def test_single_data_multiple_connectors():

    outer_sdfg = dace.SDFG('single_data_multiple_connectors')
    outer_sdfg.add_array('A', (2, 10), dtype=dace.int32)
    outer_sdfg.add_array('B', (2, 10), dtype=dace.int32)

    inner_sdfg = dace.SDFG('inner')
    inner_sdfg.add_array('A0', (10, ), dtype=dace.int32)
    inner_sdfg.add_array('A1', (10, ), dtype=dace.int32)
    inner_sdfg.add_array('B0', (10, ), dtype=dace.int32)
    inner_sdfg.add_array('B1', (10, ), dtype=dace.int32)

    inner_state = inner_sdfg.add_state('inner_state', is_start_block=True)

    inner_state.add_mapped_tasklet(name='plus',
                                   map_ranges={'j': '0:10'},
                                   inputs={
                                       '__a0': dace.Memlet(data='A0', subset='j'),
                                       '__a1': dace.Memlet(data='A1', subset='j')
                                   },
                                   outputs={'__b0': dace.Memlet(data='B0', subset='j')},
                                   code='__b0 = __a0 + __a1',
                                   external_edges=True)
    inner_state.add_mapped_tasklet(name='minus',
                                   map_ranges={'j': '0:10'},
                                   inputs={
                                       '__a0': dace.Memlet(data='A0', subset='j'),
                                       '__a1': dace.Memlet(data='A1', subset='j')
                                   },
                                   outputs={'__b1': dace.Memlet(data='B1', subset='j')},
                                   code='__b1 = __a0 - __a1',
                                   external_edges=True)

    outer_state = outer_sdfg.add_state('outer_state', is_start_block=True)

    a = outer_state.add_access('A')
    b = outer_state.add_access('B')

    me, mx = outer_state.add_map('map', {'i': '0:2'})
    inner_sdfg_node = outer_state.add_nested_sdfg(inner_sdfg, {'A0', 'A1'}, {'B0', 'B1'})

    outer_state.add_memlet_path(a, me, inner_sdfg_node, memlet=dace.Memlet(data='A', subset='0, 0:10'), dst_conn='A0')
    outer_state.add_memlet_path(a, me, inner_sdfg_node, memlet=dace.Memlet(data='A', subset='1, 0:10'), dst_conn='A1')
    outer_state.add_memlet_path(inner_sdfg_node, mx, b, memlet=dace.Memlet(data='B', subset='0, 0:10'), src_conn='B0')
    outer_state.add_memlet_path(inner_sdfg_node, mx, b, memlet=dace.Memlet(data='B', subset='1, 0:10'), src_conn='B1')

    sdutils.consolidate_edges(outer_sdfg)

    A = np.arange(20, dtype=np.int32).reshape((2, 10)).copy()
    ref = np.empty_like(A)
    ref_sdfg = copy.deepcopy(outer_sdfg)
    ref_sdfg.name = f"{ref_sdfg.name}_ref"
    ref_sdfg(A=A, B=ref)

    MapFission.apply_to(outer_sdfg, expr_index=1, map_entry=me, nested_sdfg=inner_sdfg_node)
    val = np.empty_like(A)
    outer_sdfg(A=A, B=val)

    assert np.array_equal(val, ref)


def test_dependent_symbol():

    outer_sdfg = dace.SDFG('map_fission_with_dependent_symbol')

    outer_sdfg.add_symbol('fidx', dace.int32)
    outer_sdfg.add_symbol('lidx', dace.int32)

    outer_sdfg.add_array('A', (2, 10), dtype=dace.int32)
    outer_sdfg.add_array('B', (2, 10), dtype=dace.int32)

    inner_sdfg = dace.SDFG('inner')

    inner_sdfg.add_symbol('first', dace.int32)
    inner_sdfg.add_symbol('last', dace.int32)

    inner_sdfg.add_array('A0', (10, ), dtype=dace.int32)
    inner_sdfg.add_array('A1', (10, ), dtype=dace.int32)
    inner_sdfg.add_array('B0', (10, ), dtype=dace.int32)
    inner_sdfg.add_array('B1', (10, ), dtype=dace.int32)

    inner_state = inner_sdfg.add_state('inner_state', is_start_block=True)

    inner_state.add_mapped_tasklet(name='plus',
                                   map_ranges={'j': 'first:last'},
                                   inputs={
                                       '__a0': dace.Memlet(data='A0', subset='j'),
                                       '__a1': dace.Memlet(data='A1', subset='j')
                                   },
                                   outputs={'__b0': dace.Memlet(data='B0', subset='j')},
                                   code='__b0 = __a0 + __a1',
                                   external_edges=True)

    inner_sdfg2 = dace.SDFG('inner2')

    inner_sdfg2.add_symbol('first', dace.int32)
    inner_sdfg2.add_symbol('last', dace.int32)

    inner_sdfg2.add_array('A0', (10, ), dtype=dace.int32)
    inner_sdfg2.add_array('A1', (10, ), dtype=dace.int32)
    inner_sdfg2.add_array('B1', (10, ), dtype=dace.int32)

    inner_state2 = inner_sdfg2.add_state('inner_state2', is_start_block=True)

    inner_state2.add_mapped_tasklet(name='minus',
                                    map_ranges={'j': 'first:last'},
                                    inputs={
                                        '__a0': dace.Memlet(data='A0', subset='j'),
                                        '__a1': dace.Memlet(data='A1', subset='j')
                                    },
                                    outputs={'__b1': dace.Memlet(data='B1', subset='j')},
                                    code='__b1 = __a0 - __a1',
                                    external_edges=True)

    nsdfg = inner_state.add_nested_sdfg(inner_sdfg2, {'A0', 'A1'}, {'B1'})
    a0 = inner_state.add_access('A0')
    a1 = inner_state.add_access('A1')
    b1 = inner_state.add_access('B1')

    inner_state.add_edge(a0, None, nsdfg, 'A0', dace.Memlet(data='A0', subset='0:10'))
    inner_state.add_edge(a1, None, nsdfg, 'A1', dace.Memlet(data='A1', subset='0:10'))
    inner_state.add_edge(nsdfg, 'B1', b1, None, dace.Memlet(data='B1', subset='0:10'))

    outer_state = outer_sdfg.add_state('outer_state', is_start_block=True)

    a = outer_state.add_access('A')
    b = outer_state.add_access('B')

    me, mx = outer_state.add_map('map', {'i': '0:2'})
    inner_sdfg_node = outer_state.add_nested_sdfg(inner_sdfg, {'A0', 'A1'}, {'B0', 'B1'},
                                                  symbol_mapping={
                                                      'first': 'max(0, i - fidx)',
                                                      'last': 'min(10, i + lidx)'
                                                  })

    outer_state.add_memlet_path(a, me, inner_sdfg_node, memlet=dace.Memlet(data='A', subset='0, 0:10'), dst_conn='A0')
    outer_state.add_memlet_path(a, me, inner_sdfg_node, memlet=dace.Memlet(data='A', subset='1, 0:10'), dst_conn='A1')
    outer_state.add_memlet_path(inner_sdfg_node, mx, b, memlet=dace.Memlet(data='B', subset='0, 0:10'), src_conn='B0')
    outer_state.add_memlet_path(inner_sdfg_node, mx, b, memlet=dace.Memlet(data='B', subset='1, 0:10'), src_conn='B1')

    sdutils.consolidate_edges(outer_sdfg)
    A = np.arange(20, dtype=np.int32).reshape((2, 10)).copy()
    ref = np.zeros_like(A)
    ref_sdfg = copy.deepcopy(outer_sdfg)
    ref_sdfg.name = f"{ref_sdfg.name}_ref"
    ref_sdfg(A=A, B=ref, fidx=1, lidx=5)

    MapFission.apply_to(outer_sdfg, expr_index=1, map_entry=me, nested_sdfg=inner_sdfg_node)
    outer_sdfg.apply_transformations_repeated(InlineSDFG)
    val = np.zeros_like(A)
    outer_sdfg(A=A, B=val, fidx=1, lidx=5)

    assert np.array_equal(val, ref)


@dace.program
def strided_two_ops(A: dace.float64[10], B: dace.float64[10]):
    for i in dace.map[0:9:2]:
        tmp = A[i] * 2.0
        B[i] = tmp + 1.0


def test_strided_fission_step2():
    """ MapFission on a map with step=2 and a scalar border transient. """
    A = np.arange(10, dtype=np.float64)
    B_ref = np.zeros(10, dtype=np.float64)
    B_test = np.zeros(10, dtype=np.float64)

    strided_two_ops(A=A, B=B_ref)

    sdfg = strided_two_ops.to_sdfg()
    sdfg.save("before.sdfg")
    assert sdfg.apply_transformations(MapFission, validate=True, validate_all=True) > 0

    sdfg(A=A, B=B_test)
    sdfg.save("after.sdfg")
    assert np.allclose(B_test, B_ref)


@dace.program
def strided_offset_ops(A: dace.float64[30], B: dace.float64[30]):
    for i in dace.map[10:29:3]:
        tmp = A[i] + 100.0
        B[i] = tmp * 0.5


def test_strided_fission_offset_and_step():
    """ MapFission on a map with offset=10 and step=3. """
    A = np.arange(30, dtype=np.float64)
    B_ref = np.zeros(30, dtype=np.float64)
    B_test = np.zeros(30, dtype=np.float64)

    strided_offset_ops(A=A, B=B_ref)

    sdfg = strided_offset_ops.to_sdfg()
    assert sdfg.apply_transformations(MapFission, validate=True, validate_all=True) > 0

    sdfg(A=A, B=B_test)
    assert np.allclose(B_test, B_ref)


_N = dace.symbol('N')
_START = dace.symbol('START')
_STEP = dace.symbol('STEP')
_STOP = dace.symbol('STOP')


@dace.program
def symbolic_strided(A: dace.float64[_N], B: dace.float64[_N]):
    for i in dace.map[_START:_STOP:_STEP]:
        B[i] = A[i] + 2.0
        A[i] = B[i] * 0.5


def test_symbolic_strided_fission():
    """ MapFission on a symbolic strided map. """
    sdfg = symbolic_strided.to_sdfg()
    assert sdfg.apply_transformations(MapFission, validate=True, validate_all=True) > 0

    A_ref = np.arange(10, dtype=np.float64)
    A_test = A_ref.copy()
    B_ref = np.zeros(10, dtype=np.float64)
    B_test = np.zeros(10, dtype=np.float64)

    symbolic_strided(A=A_ref, B=B_ref, N=10, START=1, STOP=9, STEP=2)
    sdfg(A=A_test, B=B_test, N=10, START=1, STOP=9, STEP=2)

    assert np.allclose(B_test, B_ref)
    assert np.allclose(A_test, A_ref)


@dace.program
def trivial_1d(A: dace.float64[16], B: dace.float64[16]):
    for i in dace.map[0:16]:
        tmp = A[i] + 1.0
        B[i] = tmp * 2.0


def test_trivial_1d_step1():
    """ MapFission on a simple 1D map with step=1. """
    A = np.arange(16, dtype=np.float64)
    B_ref = np.zeros(16, dtype=np.float64)
    B_test = np.zeros(16, dtype=np.float64)

    trivial_1d(A=A, B=B_ref)

    sdfg = trivial_1d.to_sdfg()
    assert sdfg.apply_transformations(MapFission, validate=True, validate_all=True) > 0

    sdfg(A=A, B=B_test)
    assert np.allclose(B_test, B_ref)


@dace.program
def trivial_3d(A: dace.float64[4, 5, 6], B: dace.float64[4, 5, 6]):
    for i, j, k in dace.map[0:4, 0:5, 0:6]:
        tmp = A[i, j, k] + 1.0
        B[i, j, k] = tmp + 2.0


def test_trivial_3d_step1():
    """ MapFission on a 3D map with step=1. """
    A = np.arange(4 * 5 * 6, dtype=np.float64).reshape(4, 5, 6).copy()
    B_ref = np.zeros((4, 5, 6), dtype=np.float64)
    B_test = np.zeros((4, 5, 6), dtype=np.float64)

    trivial_3d(A=A, B=B_ref)

    sdfg = trivial_3d.to_sdfg()
    assert sdfg.apply_transformations(MapFission, validate=True, validate_all=True) > 0

    sdfg(A=A, B=B_test)
    assert np.allclose(B_test, B_ref)


@dace.program
def map_with_data_cond(A: dace.float64[10]):
    for i in dace.map[0:10]:
        if A[i] > 0.0:
            A[i] = A[i] + 1.9


def test_map_with_if_nested_sdfg():
    """ MapFission must refuse maps whose body's interstate assignments depend on the map iterator. """
    # The number of applications depends on whether auto-opt is enabled.
    # We only check numerical correctness.
    sdfg = map_with_data_cond.to_sdfg()

    A_ref = np.array([-1.0, 2.0, -3.0, 4.0, 5.0, -6.0, 7.0, -8.0, 9.0, 10.0], dtype=np.float64)
    A_test = A_ref.copy()
    map_with_data_cond(A=A_ref)
    sdfg(A=A_test)
    assert np.allclose(A_test, A_ref)


N, M, P, K = (dace.symbol(s) for s in ('N', 'M', 'P', 'K'))


@dace.program
def _nested_parent(x: dace.float64[N, M], y: dace.float64[N, M]):
    for j in dace.map[0:M]:
        for jj in dace.map[0:P]:
            for i in dace.map[0:N]:
                x[i, j] = 1.0
            for i in dace.map[0:N]:
                y[i, j] = 2.0


@dace.program
def _toplevel_two_components(x: dace.float64[N, M], y: dace.float64[N, M]):
    for j in dace.map[0:M]:
        for i in dace.map[0:N]:
            x[i, j] = 1.0
        for i in dace.map[0:N]:
            y[i, j] = 2.0


@dace.program
def _only_conditional(x: dace.float64[N]):
    for i in dace.map[0:N]:
        if i % 2 == 0:
            x[i] = 1.0
        else:
            x[i] = -1.0


@dace.program
def _control_flow_in_scope(x: dace.float64[N, M], y: dace.float64[N, M]):
    for j in dace.map[0:M]:
        for i in dace.map[0:N]:
            if j % 2 == 0:
                x[i, j] = 1.0
            else:
                x[i, j] = -1.0
        for i in dace.map[0:N]:
            y[i, j] = 2.0


@dace.program
def _five_set_five_cpy(s0: dace.float64[K], s1: dace.float64[K], s2: dace.float64[K], s3: dace.float64[K],
                       s4: dace.float64[K], a0: dace.float64[K], a1: dace.float64[K], a2: dace.float64[K],
                       a3: dace.float64[K], a4: dace.float64[K], c0: dace.float64[K], c1: dace.float64[K],
                       c2: dace.float64[K], c3: dace.float64[K], c4: dace.float64[K]):
    for i in dace.map[0:K]:
        s0[i] = 0.0
        s1[i] = 1.0
        s2[i] = 2.0
        s3[i] = 3.0
        s4[i] = 4.0
        c0[i] = a0[i]
        c1[i] = a1[i]
        c2[i] = a2[i]
        c3[i] = a3[i]
        c4[i] = a4[i]


@dace.program
def _three_set_two_cpy(s0: dace.float64[K], s1: dace.float64[K], s2: dace.float64[K], a0: dace.float64[K],
                       a1: dace.float64[K], c0: dace.float64[K], c1: dace.float64[K]):
    for i in dace.map[0:K]:
        s0[i] = 0.0
        s1[i] = 1.0
        s2[i] = 2.0
        c0[i] = a0[i]
        c1[i] = a1[i]


def _find_map_entry(sdfg, param, nested):
    """Find a single-parameter map entry by nesting.

    :param sdfg: The SDFG to search.
    :param param: The map's sole iteration parameter name.
    :param nested: Whether the map must be inside another scope.
    :returns: A ``(state, map_entry)`` pair, or ``(None, None)``.
    """
    for st in sdfg.all_states():
        for n in st.nodes():
            if isinstance(n, nodes.MapEntry) and n.map.params == [param]:
                if (st.entry_node(n) is not None) == nested:
                    return st, n
    return None, None


def _fission_maps(sdfg):
    """Return the top-level ``(state, map_entry)`` pairs.

    :param sdfg: The SDFG to scan.
    :returns: One pair per outermost map entry.
    """
    return [(st, n) for st in sdfg.all_states() for n in st.nodes()
            if isinstance(n, nodes.MapEntry) and st.entry_node(n) is None]


def _assert_no_spurious_connectors(sdfg, n_set, n_cpy):
    """Assert each fissioned map carries exactly its component's connectors.

    :param sdfg: The fissioned SDFG.
    :param n_set: Expected count of memset (zero-input) maps.
    :param n_cpy: Expected count of memcpy (one-input) maps.
    :raises AssertionError: On a wrong map count or an unused connector.
    """
    maps = _fission_maps(sdfg)
    assert len(maps) == n_set + n_cpy, f"expected {n_set + n_cpy} maps, got {len(maps)}"
    n_in0 = n_in1 = 0
    for st, me in maps:
        mx = st.exit_node(me)
        in_c = {c for c in me.in_connectors if c.startswith('IN_')}
        out_c = {c for c in me.out_connectors if c.startswith('OUT_')}
        assert {e.dst_conn for e in st.in_edges(me) if e.dst_conn} == in_c
        assert {e.src_conn for e in st.out_edges(me) if e.src_conn} == out_c
        assert all(st.out_edges(mx)), "map exit has a dangling output"
        deg = len(in_c)
        assert deg in (0, 1), f"unexpected input arity {deg}"
        n_in0 += deg == 0
        n_in1 += deg == 1
    assert n_in0 == n_set and n_in1 == n_cpy


def test_mapfission_handles_nested_parent_correctly():
    """Fission a map nested inside another map (regression: Leftover nodes)."""
    n, m, p = 4, 3, 2
    sdfg = _nested_parent.to_sdfg(simplify=True)
    _, jj = _find_map_entry(sdfg, 'jj', nested=True)
    assert jj is not None
    assert MapFission.can_be_applied_to(sdfg, map_entry=jj) is True

    x0, y0 = np.zeros((n, m)), np.zeros((n, m))
    copy.deepcopy(sdfg)(x=x0, y=y0, N=n, M=m, P=p)

    assert sdfg.apply_transformations_repeated(MapFission) >= 1
    sdfg.validate()

    x1, y1 = np.zeros((n, m)), np.zeros((n, m))
    sdfg(x=x1, y=y1, N=n, M=m, P=p)
    assert np.allclose(x1, x0) and np.allclose(y1, y0)
    assert np.allclose(x1, 1.0) and np.allclose(y1, 2.0)


def test_mapfission_still_applies_to_toplevel_map():
    """Top-level two-component fission is unaffected by the nested fix."""
    n, m = 6, 5
    sdfg = _toplevel_two_components.to_sdfg(simplify=True)
    _, j = _find_map_entry(sdfg, 'j', nested=False)
    assert j is not None and MapFission.can_be_applied_to(sdfg, map_entry=j) is True

    assert sdfg.apply_transformations_repeated(MapFission) >= 1
    sdfg.validate()

    x, y = np.zeros((n, m)), np.zeros((n, m))
    sdfg(x=x, y=y, N=n, M=m)
    assert np.allclose(x, 1.0) and np.allclose(y, 2.0)


@pytest.mark.parametrize("prog,n_set,n_cpy", [(_five_set_five_cpy, 5, 5), (_three_set_two_cpy, 3, 2)])
def test_memset_memcpy_fission_clean_connectors(prog, n_set, n_cpy):
    """Many memset/memcpy components fission into maps with no spurious connectors."""
    k = 8
    sdfg = prog.to_sdfg(simplify=True)
    names = list(sdfg.arg_names)
    args = {nm: (np.zeros(k) if nm.startswith(('s', 'c')) else np.random.rand(k)) for nm in names}
    ref = {nm: v.copy() for nm, v in args.items()}
    copy.deepcopy(sdfg)(**ref, K=k)

    assert sdfg.apply_transformations_repeated(MapFission) >= 1
    sdfg.validate()
    _assert_no_spurious_connectors(sdfg, n_set, n_cpy)

    out = {nm: v.copy() for nm, v in args.items()}
    sdfg(**out, K=k)
    for nm in names:
        assert np.allclose(out[nm], ref[nm]), f"mismatch on {nm}"


def test_mapfission_does_not_apply_to_conditional_map():
    """A map whose sole body is a conditional: MapFission must not apply."""
    n = 6
    sdfg = _only_conditional.to_sdfg(simplify=True)
    assert sdfg.apply_transformations_repeated(MapFission) == 0
    sdfg.validate()

    x = np.zeros(n)
    sdfg(x=x, N=n)
    assert np.allclose(x, np.where(np.arange(n) % 2 == 0, 1.0, -1.0))


def test_mapfission_refuses_conditional_component_stays_valid():
    """A conditional component is refused; the rest stays valid (regression: Leftover nodes)."""
    n, m = 5, 4
    sdfg = _control_flow_in_scope.to_sdfg(simplify=True)

    x0, y0 = np.zeros((n, m)), np.zeros((n, m))
    copy.deepcopy(sdfg)(x=x0, y=y0, N=n, M=m)

    for st in sdfg.all_states():
        for me in st.nodes():
            if not isinstance(me, nodes.MapEntry):
                continue
            has_cond = any(
                isinstance(e.dst, nodes.NestedSDFG) and any(
                    type(b).__name__ == 'ConditionalBlock' for b in e.dst.sdfg.all_control_flow_regions(recursive=True))
                for e in st.out_edges(me))
            if has_cond:
                assert MapFission.can_be_applied_to(sdfg, map_entry=me) is False

    sdfg.apply_transformations_repeated(MapFission)
    sdfg.validate()

    x1, y1 = np.zeros((n, m)), np.zeros((n, m))
    sdfg(x=x1, y=y1, N=n, M=m)
    assert np.allclose(x1, x0) and np.allclose(y1, y0)


@dace.program
def _if_two_components(a: dace.float64[N], A: dace.float64[N], B: dace.float64[N], c: dace.int32[1]):
    for i in dace.map[0:N]:
        if c[0] > 0:
            A[i] = a[i] + 1.0
            B[i] = a[i] * 2.0


@dace.program
def _if_three_components(a: dace.float64[N], A: dace.float64[N], B: dace.float64[N], C: dace.float64[N],
                         c: dace.int32[1]):
    for i in dace.map[0:N]:
        if c[0] > 0:
            A[i] = a[i] + 1.0
            B[i] = a[i] * 2.0
            C[i] = a[i] - 3.0


@dace.program
def _if_single_component(x: dace.float64[N], c: dace.int32[1]):
    for i in dace.map[0:N]:
        if c[0] > 0:
            x[i] = 1.0
        else:
            x[i] = -1.0


def _toplevel_map_count(sdfg):
    """Number of outermost map entries across all states.

    :param sdfg: The SDFG to scan.
    :returns: The count of top-level map entries.
    """
    return sum(1 for st in sdfg.all_states() for n in st.nodes()
               if isinstance(n, nodes.MapEntry) and st.entry_node(n) is None)


def _run_branch_fission(prog, args, n, expect_maps):
    """Apply SplitStatements + MapFission and check e2e.

    :param prog: The dace program.
    :param args: Keyword arrays for the reference/post runs (no ``N``).
    :param n: The value bound to symbol ``N``.
    :param expect_maps: Minimum number of top-level maps expected after.
    :returns: The post-transformation SDFG.
    """
    from dace.transformation.passes.canonicalize.split_statements import SplitStatements
    from dace.transformation.passes.simplify import SimplifyPass
    sdfg = prog.to_sdfg(simplify=True)

    ref = {k: v.copy() for k, v in args.items()}
    copy.deepcopy(sdfg)(**ref, N=n)

    SplitStatements().apply_pass(sdfg, {})
    sdfg.apply_transformations_repeated(MapFission)
    SimplifyPass().apply_pass(sdfg, {})
    sdfg.validate()
    assert _toplevel_map_count(sdfg) >= expect_maps, \
        f"expected >= {expect_maps} maps, got {_toplevel_map_count(sdfg)}"

    out = {k: v.copy() for k, v in args.items()}
    sdfg(**out, N=n)
    for k in args:
        assert np.allclose(out[k], ref[k]), f"mismatch on {k}"
    return sdfg


def test_conditional_component_fission_two():
    """if c: {A;B} -> two maps, each a conditional writing one output."""
    n = 16
    a = np.random.rand(n)
    _run_branch_fission(_if_two_components, dict(a=a, A=np.zeros(n), B=np.zeros(n), c=np.array([1], np.int32)), n, 2)
    # condition false: outputs stay zero (replicated condition still guards).
    s = _if_two_components.to_sdfg(simplify=True)
    from dace.transformation.passes.canonicalize.split_statements import SplitStatements
    SplitStatements().apply_pass(s, {})
    s.apply_transformations_repeated(MapFission)
    s.validate()
    A, B = np.full(n, 9.0), np.full(n, 9.0)
    s(a=a.copy(), A=A, B=B, c=np.array([0], np.int32), N=n)
    assert np.allclose(A, 9.0) and np.allclose(B, 9.0)


def test_conditional_component_fission_three():
    """if c: {A;B;C} -> three independent condition-replicated maps."""
    n = 12
    a = np.random.rand(n)
    _run_branch_fission(_if_three_components,
                        dict(a=a, A=np.zeros(n), B=np.zeros(n), C=np.zeros(n), c=np.array([1], np.int32)), n, 3)


def test_conditional_component_fission_single_is_noop():
    """A lone single-output conditional has nothing to fission: no-op, valid."""
    from dace.transformation.passes.canonicalize.split_statements import SplitStatements
    n = 8
    sdfg = _if_single_component.to_sdfg(simplify=True)
    assert SplitStatements().apply_pass(sdfg, {}) is None
    sdfg.validate()
    x = np.zeros(n)
    sdfg(x=x, c=np.array([1], np.int32), N=n)
    assert np.allclose(x, 1.0)


if __name__ == '__main__':
    test_subgraph()
    test_nested_sdfg()
    test_nested_transient()
    test_inputs_outputs()
    test_multidim()
    test_offsets()
    test_offsets_array()
    test_mapfission_with_symbols()
    test_two_edges_through_map()
    test_if_scope()
    test_if_scope_2()
    test_array_copy_outside_scope()
    test_single_data_multiple_connectors()
    test_dependent_symbol()
    test_strided_fission_step2()
    test_strided_fission_offset_and_step()
    test_symbolic_strided_fission()
    test_trivial_1d_step1()
    test_trivial_3d_step1()
    test_map_with_if_nested_sdfg()
    test_mapfission_handles_nested_parent_correctly()
    test_mapfission_still_applies_to_toplevel_map()
    test_memset_memcpy_fission_clean_connectors(_five_set_five_cpy, 5, 5)
    test_memset_memcpy_fission_clean_connectors(_three_set_two_cpy, 3, 2)
    test_mapfission_does_not_apply_to_conditional_map()
    test_mapfission_refuses_conditional_component_stays_valid()
    test_conditional_component_fission_two()
    test_conditional_component_fission_three()
    test_conditional_component_fission_single_is_noop()
