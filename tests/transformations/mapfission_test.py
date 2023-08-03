# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import copy
import dace
from dace.sdfg import nodes, utils as sdutils
from dace.transformation.dataflow import MapFission
from dace.transformation.interstate import InlineSDFG
from dace.transformation.helpers import nest_state_subgraph
import numpy as np
import unittest


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
    nsdfg_node = state.add_nested_sdfg(nsdfg, None, {'a'}, {'b'})
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
    nsdfg_node = state.add_nested_sdfg(nsdfg, None, {}, {'a'})

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

    state = sdfg.add_state('parent', is_start_state=True)
    me, mx = state.add_map('parent_map', {'i': '0:N'})

    nsdfg = dace.SDFG('nested_sdfg')
    nsdfg.add_scalar('inner_A', dace.int32)
    nsdfg.add_scalar('inner_B', dace.int32)

    nstate = nsdfg.add_state('child', is_start_state=True)
    na = nstate.add_access('inner_A')
    nb = nstate.add_access('inner_B')
    ta = nstate.add_tasklet('tasklet_A', {}, {'__out'}, '__out = M')
    tb = nstate.add_tasklet('tasklet_B', {}, {'__out'}, '__out = M')
    nstate.add_edge(ta, '__out', na, None, dace.Memlet.from_array('inner_A', nsdfg.arrays['inner_A']))
    nstate.add_edge(tb, '__out', nb, None, dace.Memlet.from_array('inner_B', nsdfg.arrays['inner_B']))

    a = state.add_access('A')
    b = state.add_access('B')
    t = state.add_nested_sdfg(nsdfg, None, {}, {'inner_A', 'inner_B'})
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

    state = sdfg.add_state('parent', is_start_state=True)
    me, mx = state.add_map('parent_map', {'i': '0:N'})

    nsdfg = dace.SDFG('nested_sdfg')
    nsdfg.add_array('inner_A', (N, ), dace.int32)
    nsdfg.add_scalar('inner_B', dace.int32)

    nstate = nsdfg.add_state('child', is_start_state=True)
    na = nstate.add_access('inner_A')
    nb = nstate.add_access('inner_B')
    t = nstate.add_tasklet('tasklet', {'__in1', '__in2'}, {'__out'}, '__out = __in1 + __in2')
    nstate.add_edge(na, None, t, '__in1', dace.Memlet('inner_A[i]'))
    nstate.add_edge(na, None, t, '__in2', dace.Memlet('inner_A[N-i-1]'))
    nstate.add_edge(t, '__out', nb, None, dace.Memlet.from_array('inner_B', nsdfg.arrays['inner_B']))

    a = state.add_access('A')
    b = state.add_access('B')
    t = state.add_nested_sdfg(nsdfg, None, {'inner_A'}, {'inner_B'}, {'N': 'N', 'i': 'i'})
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
    snode = state.add_nested_sdfg(nsdfg, None, {'ninp'}, {'nout'})
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

    inner_state = inner_sdfg.add_state('inner_state', is_start_state=True)

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

    outer_state = outer_sdfg.add_state('outer_state', is_start_state=True)

    a = outer_state.add_access('A')
    b = outer_state.add_access('B')

    me, mx = outer_state.add_map('map', {'i': '0:2'})
    inner_sdfg_node = outer_state.add_nested_sdfg(inner_sdfg, None, {'A0', 'A1'}, {'B0', 'B1'})

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

    inner_state = inner_sdfg.add_state('inner_state', is_start_state=True)

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

    inner_state2 = inner_sdfg2.add_state('inner_state2', is_start_state=True)

    inner_state2.add_mapped_tasklet(name='minus',
                                    map_ranges={'j': 'first:last'},
                                    inputs={
                                        '__a0': dace.Memlet(data='A0', subset='j'),
                                        '__a1': dace.Memlet(data='A1', subset='j')
                                    },
                                    outputs={'__b1': dace.Memlet(data='B1', subset='j')},
                                    code='__b1 = __a0 - __a1',
                                    external_edges=True)

    nsdfg = inner_state.add_nested_sdfg(inner_sdfg2, None, {'A0', 'A1'}, {'B1'})
    a0 = inner_state.add_access('A0')
    a1 = inner_state.add_access('A1')
    b1 = inner_state.add_access('B1')

    inner_state.add_edge(a0, None, nsdfg, 'A0', dace.Memlet(data='A0', subset='0:10'))
    inner_state.add_edge(a1, None, nsdfg, 'A1', dace.Memlet(data='A1', subset='0:10'))
    inner_state.add_edge(nsdfg, 'B1', b1, None, dace.Memlet(data='B1', subset='0:10'))

    outer_state = outer_sdfg.add_state('outer_state', is_start_state=True)

    a = outer_state.add_access('A')
    b = outer_state.add_access('B')

    me, mx = outer_state.add_map('map', {'i': '0:2'})
    inner_sdfg_node = outer_state.add_nested_sdfg(inner_sdfg,
                                                  None, {'A0', 'A1'}, {'B0', 'B1'},
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
