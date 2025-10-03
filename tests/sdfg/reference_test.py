# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
""" Tests the use of Reference data descriptors. """
import dace
from dace.sdfg import validation
from dace.transformation.pass_pipeline import Pipeline
from dace.transformation.passes.analysis import FindReferenceSources
from dace.transformation.passes.reference_reduction import ReferenceToView
import numpy as np
import pytest
import networkx as nx


def test_frontend_reference():
    N = dace.symbol('N')
    M = dace.symbol('M')
    mystruct = dace.data.Structure(members={
        "data": dace.data.Array(dace.float32, (N, M), strides=(1, N)),
        "arrA": dace.data.ArrayReference(dace.float32, (N, )),
        "arrB": dace.data.ArrayReference(dace.float32, (N, )),
    },
                                   name="MyStruct")

    @dace.program
    def init_prog(mydat: mystruct, fill_value: int) -> None:
        mydat.arrA = mydat.data[:, 2]
        mydat.arrB = mydat.data[:, 0]

        # loop over all arrays and initialize them with `fill_value`
        for index in range(M):
            mydat.data[:, index] = fill_value

        # Initialize the two named ones by name
        mydat.arrA[:] = fill_value + 1
        mydat.arrB[:] = fill_value + 2

    dat = np.zeros((10, 5), dtype=np.float32)
    inp_struct = mystruct.dtype._typeclass.as_ctypes()(data=dat.__array_interface__['data'][0])

    func = init_prog.compile()
    func(mydat=inp_struct, fill_value=3, N=10, M=5)

    assert np.allclose(dat[0, :], 5) and np.allclose(dat[1, :], 5)
    assert np.allclose(dat[2, :], 3) and np.allclose(dat[3, :], 3)
    assert np.allclose(dat[4, :], 4) and np.allclose(dat[5, :], 4)
    assert np.allclose(dat[6, :], 3) and np.allclose(dat[7, :], 3)
    assert np.allclose(dat[8, :], 3) and np.allclose(dat[9, :], 3)


def test_type_annotation_reference():
    N = dace.symbol('N')

    @dace.program
    def ref(A: dace.float64[N], B: dace.float64[N], T: dace.int32, out: dace.float64[N]):
        ref1: dace.data.ArrayReference(A.dtype, A.shape) = A
        ref2: dace.data.ArrayReference(A.dtype, A.shape) = B
        if T <= 0:
            out[:] = ref1[:] + 1
        else:
            out[:] = ref2[:] + 1

    a = np.random.rand(20)
    a_verif = a.copy()
    b = np.random.rand(20)
    b_verif = b.copy()
    out = np.random.rand(20)
    out_verif = out.copy()

    ref(a, b, 1, out, N=20)
    ref.f(a_verif, b_verif, 1, out_verif)
    assert np.allclose(out, out_verif)

    ref(a, b, -1, out, N=20)
    ref.f(a_verif, b_verif, -1, out_verif)
    assert np.allclose(out, out_verif)


def test_unset_reference():
    sdfg = dace.SDFG('tester')
    sdfg.add_reference('ref', [20], dace.float64)
    state = sdfg.add_state()
    t = state.add_tasklet('doit', {'a'}, {'b'}, 'b = a + 1')
    state.add_edge(state.add_read('ref'), None, t, 'a', dace.Memlet('ref[0]'))
    state.add_edge(t, 'b', state.add_write('ref'), None, dace.Memlet('ref[1]'))

    with pytest.raises(validation.InvalidSDFGNodeError):
        sdfg.validate()


def _create_branch_sdfg():
    """
    An SDFG in which a reference is set conditionally.
    """
    sdfg = dace.SDFG('refbranch')
    sdfg.add_array('A', [20], dace.float64)
    sdfg.add_array('B', [20], dace.float64)
    sdfg.add_symbol('i', dace.int64)
    sdfg.add_reference('ref', [20], dace.float64)
    sdfg.add_array('out', [20], dace.float64)

    # Branch to a or b depending on i
    start = sdfg.add_state()
    a = sdfg.add_state()
    b = sdfg.add_state()
    finish = sdfg.add_state()
    sdfg.add_edge(start, a, dace.InterstateEdge('i < 5'))
    sdfg.add_edge(start, b, dace.InterstateEdge('i >= 5'))
    sdfg.add_edge(a, finish, dace.InterstateEdge())
    sdfg.add_edge(b, finish, dace.InterstateEdge())

    # Copy from reference to output
    a.add_edge(a.add_read('A'), None, a.add_write('ref'), 'set', dace.Memlet('A'))
    b.add_edge(b.add_read('B'), None, b.add_write('ref'), 'set', dace.Memlet('B'))

    r = finish.add_read('ref')
    w = finish.add_write('out')
    finish.add_nedge(r, w, dace.Memlet('ref'))
    return sdfg


def _create_tasklet_assignment_sdfg():
    """
    An SDFG in which a reference is set by a tasklet.
    """
    sdfg = dace.SDFG('refta')
    sdfg.add_array('A', [20], dace.float64)
    sdfg.add_array('B', [19], dace.float64)
    sdfg.add_reference('ref', [19], dace.float64)

    state = sdfg.add_state()
    t = state.add_tasklet('ptrset', {'a': dace.pointer(dace.float64)}, {'o'}, 'o = a + 1')
    state.add_edge(state.add_read('A'), None, t, 'a', dace.Memlet('A'))
    ref = state.add_access('ref')
    state.add_edge(t, 'o', ref, 'set', dace.Memlet('ref'))
    t2 = state.add_tasklet('addone', {'a'}, {'o'}, 'o = a + 1')
    state.add_edge(ref, None, t2, 'a', dace.Memlet('ref[0]'))
    state.add_edge(t2, 'o', state.add_write('B'), None, dace.Memlet('B[0]'))
    return sdfg


def _create_twostate_sdfg():
    """
    An SDFG in which a reference set happens on another state.
    """
    sdfg = dace.SDFG('reftest')
    sdfg.add_array('A', [20], dace.float64)
    sdfg.add_reference('ref', [10], dace.float64)

    setstate = sdfg.add_state()
    computestate = sdfg.add_state_after(setstate)

    setstate.add_edge(setstate.add_read('A'), None, setstate.add_write('ref'), 'set', dace.Memlet('A[10:20]'))

    # Read from A[10], write to A[11]
    t = computestate.add_tasklet('addone', {'a'}, {'b'}, 'b = a + 1')
    computestate.add_edge(computestate.add_write('ref'), None, t, 'a', dace.Memlet('ref[0]'))
    computestate.add_edge(t, 'b', computestate.add_write('ref'), None, dace.Memlet('ref[1]'))
    return sdfg


def _create_multisubset_sdfg():
    """
    A Jacobi-2d style SDFG to test multi-dimensional subsets and the use of an empty memlet
    as a dependency edge.
    """
    sdfg = dace.SDFG('reftest')
    sdfg.add_array('A', [22, 22], dace.float64)
    sdfg.add_array('B', [20, 20], dace.float64)
    sdfg.add_reference('ref1', [22], dace.float64)
    sdfg.add_reference('ref2', [22], dace.float64, strides=[22])
    sdfg.add_reference('ref3', [22], dace.float64, strides=[22])
    sdfg.add_reference('ref4', [22], dace.float64, strides=[22])
    sdfg.add_reference('refB', [20], dace.float64)

    state = sdfg.add_state()

    # Access nodes
    a = state.add_read('A')
    b = state.add_read('B')
    r1 = state.add_access('ref1')
    r2 = state.add_access('ref2')
    r3 = state.add_access('ref3')
    r4 = state.add_access('ref4')
    rbset = state.add_access('refB')
    rbwrite = state.add_write('refB')

    # Add reference sets
    state.add_edge(a, None, r1, 'set', dace.Memlet('A[5, 0:22]'))
    state.add_edge(a, None, r2, 'set', dace.Memlet('A[0:22, 5]'))
    state.add_edge(a, None, r3, 'set', dace.Memlet('A[0:22, 4]'))
    state.add_edge(a, None, r4, 'set', dace.Memlet('A[0:22, 3]'))
    state.add_edge(b, None, rbset, 'set', dace.Memlet('B[4, 0:20]'))

    # Add tasklet
    t = state.add_tasklet('stencil', {'a', 'b', 'c', 'd'}, {'o'}, 'o = 0.25 * (a + b + c + d)')

    # Connect tasklet
    state.add_nedge(rbset, t, dace.Memlet())  # Happens-before edge
    state.add_edge(r1, None, t, 'a', dace.Memlet('ref1[4]'))  # (5,4)
    state.add_edge(r2, None, t, 'b', dace.Memlet('ref2[4]'))  # (4,5)
    state.add_edge(r3, None, t, 'c', dace.Memlet('ref3[3]'))  # (3,4)
    state.add_edge(r4, None, t, 'd', dace.Memlet('ref4[4]'))  # (4,3)
    state.add_edge(t, 'o', rbwrite, None, dace.Memlet('refB[4]'))

    return sdfg


def _create_scoped_sdfg():
    """
    An SDFG in which a reference is used inside a scope.
    """
    sdfg = dace.SDFG('reftest')
    sdfg.add_array('A', [20, 20], dace.float64)
    sdfg.add_array('B', [20, 20], dace.float64)
    sdfg.add_reference('ref', [2], dace.float64, strides=[20])

    istate = sdfg.add_state()
    state = sdfg.add_state_after(istate)

    istate.add_edge(istate.add_read('A'), None, istate.add_write('ref'), 'set', dace.Memlet('A[2:4, 3]'))

    me, mx = state.add_map('mapit', dict(i='0:2'))
    ref = state.add_access('ref')
    inp = state.add_read('B')
    t = state.add_tasklet('doit', {'r'}, {'w'}, 'w = r + 1')
    out = state.add_write('A')
    state.add_memlet_path(inp, me, ref, memlet=dace.Memlet('B[1, i] -> [i]'))
    state.add_edge(ref, None, t, 'r', dace.Memlet('ref[i]'))
    state.add_edge_pair(mx, t, out, internal_connector='w', internal_memlet=dace.Memlet('A[10, i]'))

    return sdfg


def _create_scoped_empty_memlet_sdfg():
    """
    An SDFG in which a reference is used inside a scope with no inputs.
    """
    sdfg = dace.SDFG('reftest')
    sdfg.add_array('A', [20, 20], dace.float64)
    sdfg.add_array('B', [20, 20], dace.float64)
    sdfg.add_reference('ref', [2], dace.float64, strides=[20])

    istate = sdfg.add_state()
    state = sdfg.add_state_after(istate)

    istate.add_edge(istate.add_read('A'), None, istate.add_write('ref'), 'set', dace.Memlet('A[2:4, 3]'))

    me, mx = state.add_map('mapit', dict(i='0:2'))
    ref = state.add_access('ref')
    t = state.add_tasklet('doit', {'r'}, {'w'}, 'w = r + 1')
    out = state.add_write('B')
    state.add_edge(me, None, ref, None, memlet=dace.Memlet())
    state.add_edge(ref, None, t, 'r', dace.Memlet('ref[i]'))
    state.add_edge_pair(mx, t, out, internal_connector='w', internal_memlet=dace.Memlet('B[10, i]'))

    return sdfg


def _create_neighbor_sdfg():
    """
    An SDFG where a reference has both predecessors and successors.
    """
    sdfg = dace.SDFG('reftest')
    sdfg.add_array('A', [20, 20], dace.float64)
    sdfg.add_array('B', [2, 2], dace.float64)
    sdfg.add_reference('ref', [2, 2], dace.float64, strides=[20, 1])

    istate = sdfg.add_state()
    state = sdfg.add_state_after(istate)

    istate.add_edge(istate.add_read('A'), None, istate.add_write('ref'), 'set', dace.Memlet('A[2:4, 3:5]'))

    b = state.add_read('B')
    ref1 = state.add_access('ref')
    ref2 = state.add_write('ref')
    state.add_mapped_tasklet('addtwo',
                             dict(i='0:2', j='0:2'),
                             dict(r=dace.Memlet('B[i, j]')),
                             'w = r + 2',
                             dict(w=dace.Memlet('ref[i, j]')),
                             external_edges=True,
                             input_nodes=dict(B=b),
                             output_nodes=dict(ref=ref1))
    state.add_mapped_tasklet('sum',
                             dict(i='0:2'),
                             dict(r=dace.Memlet('ref[0, i]')),
                             'w = r',
                             dict(w=dace.Memlet('ref[1, 0]', wcr='lambda a,b: a+b')),
                             external_edges=True,
                             input_nodes=dict(ref=ref1),
                             output_nodes=dict(ref=ref2))
    state.add_mapped_tasklet('addone',
                             dict(i='1:2'),
                             dict(r=dace.Memlet('ref[i - 1, i - 1]')),
                             'w = r + 1',
                             dict(w=dace.Memlet('ref[i, i]')),
                             external_edges=True,
                             input_nodes=dict(ref=ref1),
                             output_nodes=dict(ref=ref2))
    return sdfg


def _create_loop_nonfree_symbols_sdfg():
    """
    An SDFG where a reference is set inside a loop and used outside.
    """
    sdfg = dace.SDFG('reftest')
    sdfg.add_array('A', [20], dace.float64)
    sdfg.add_reference('ref', [1], dace.float64)

    # Create state machine
    istate = sdfg.add_state()
    state = sdfg.add_state()
    after = sdfg.add_state()
    sdfg.add_loop(istate, state, after, 'i', '0', 'i < 20', 'i + 1')

    # Reference set inside loop
    state.add_edge(state.add_read('A'), None, state.add_write('ref'), 'set', dace.Memlet('A[i] -> [0]'))

    # Use outisde loop
    t = after.add_tasklet('setone', {}, {'out'}, 'out = 1')
    after.add_edge(t, 'out', after.add_write('ref'), None, dace.Memlet('ref[0]'))

    return sdfg


def _create_loop_reference_internal_use():
    """
    An SDFG where a reference is set and used inside a loop.
    """
    sdfg = dace.SDFG('reftest')
    sdfg.add_array('A', [20], dace.float64)
    sdfg.add_reference('ref', [1], dace.float64)

    # Create state machine
    istate = sdfg.add_state()
    state = sdfg.add_state()
    after = sdfg.add_state()
    sdfg.add_edge(state, after, dace.InterstateEdge())
    sdfg.add_loop(istate, state, None, 'i', '0', 'i < 20', 'i + 1', loop_end_state=after)

    # Reference set inside loop
    state.add_edge(state.add_read('A'), None, state.add_write('ref'), 'set', dace.Memlet('A[i]'))

    # Use inside loop
    t = after.add_tasklet('setone', {}, {'out'}, 'out = 1')
    after.add_edge(t, 'out', after.add_write('ref'), None, dace.Memlet('ref[0]'))

    return sdfg


def _create_loop_reference_nonfree_internal_use():
    """
    An SDFG where a reference is set inside one loop and used in another, with
    the same symbol name.
    """
    sdfg = dace.SDFG('reftest')
    sdfg.add_array('A', [20], dace.float64)
    sdfg.add_reference('ref', [1], dace.float64)

    # Create state machine
    istate = sdfg.add_state()
    between_loops = sdfg.add_state()

    # First loop
    state1 = sdfg.add_state()
    sdfg.add_loop(istate, state1, between_loops, 'i', '0', 'i < 20', 'i + 1')

    # Second loop
    state2 = sdfg.add_state()
    sdfg.add_loop(between_loops, state2, None, 'i', '0', 'i < 20', 'i + 1')

    # Reference set inside first loop
    state1.add_edge(state1.add_read('A'), None, state1.add_write('ref'), 'set', dace.Memlet('A[i]'))

    # Use inside second loop
    t = state2.add_tasklet('setone', {}, {'out'}, 'out = 1')
    state2.add_edge(t, 'out', state2.add_write('ref'), None, dace.Memlet('ref[0]'))

    return sdfg


def test_reference_branch():
    sdfg = _create_branch_sdfg()

    A = np.random.rand(20)
    B = np.random.rand(20)
    out = np.random.rand(20)

    sdfg(A=A, B=B, out=out, i=10)
    assert np.allclose(out, B)

    sdfg(A=A, B=B, out=out, i=1)
    assert np.allclose(out, A)

    # Test reference-to-view - should fail to apply
    result = Pipeline([ReferenceToView()]).apply_pass(sdfg, {})
    assert 'ReferenceToView' not in result or not result['ReferenceToView']


def test_reference_sources_pass():
    sdfg = _create_branch_sdfg()
    sources = FindReferenceSources().apply_pass(sdfg, {})
    assert len(sources) == 1  # There is only one SDFG
    sources = sources[0]
    assert len(sources) == 1 and 'ref' in sources  # There is one reference
    sources = sources['ref']
    assert sources == {dace.Memlet('A[0:20]', volume=1), dace.Memlet('B[0:20]', volume=1)}


def test_reference_tasklet_assignment():
    sdfg = _create_tasklet_assignment_sdfg()

    A = np.random.rand(20)
    B = np.random.rand(19)
    ref = np.copy(B)
    ref[0] = A[1] + 1

    sdfg(A=A, B=B)
    assert np.allclose(ref, B)


def test_reference_tasklet_assignment_analysis():
    sdfg = _create_tasklet_assignment_sdfg()
    sources = FindReferenceSources().apply_pass(sdfg, {})
    assert len(sources) == 1  # There is only one SDFG
    sources = sources[0]
    assert len(sources) == 1 and 'ref' in sources  # There is one reference
    sources = sources['ref']
    assert sources == {
        next(n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.Tasklet) and n.label == 'ptrset')
    }


def test_reference_tasklet_assignment_stree():
    from dace.sdfg.analysis.schedule_tree import sdfg_to_tree as s2t, treenodes as tn
    sdfg = _create_tasklet_assignment_sdfg()
    stree = s2t.as_schedule_tree(sdfg)
    assert [type(n) for n in stree.children] == [tn.TaskletNode, tn.RefSetNode, tn.TaskletNode]


def test_reference_tasklet_assignment_reftoview():
    sdfg = _create_tasklet_assignment_sdfg()

    # Test reference-to-view - should fail to apply
    result = Pipeline([ReferenceToView()]).apply_pass(sdfg, {})
    assert 'ReferenceToView' not in result or not result['ReferenceToView']


@pytest.mark.parametrize('reftoview', (False, True))
def test_twostate(reftoview):
    sdfg = _create_twostate_sdfg()

    # Test sources
    sources = FindReferenceSources().apply_pass(sdfg, {})
    assert len(sources) == 1  # There is only one SDFG
    sources = sources[0]
    assert len(sources) == 1 and 'ref' in sources  # There is one reference
    sources = sources['ref']
    assert sources == {dace.Memlet('A[10:20]')}

    if reftoview:
        sdfg.simplify()
        assert not any(isinstance(v, dace.data.Reference) for v in sdfg.arrays.values())

    # Test correctness
    A = np.random.rand(20)
    ref = np.copy(A)
    ref[11] = ref[10] + 1
    sdfg(A=A)
    assert np.allclose(A, ref)


@pytest.mark.parametrize('reftoview', (False, True))
def test_multisubset(reftoview):
    sdfg = _create_multisubset_sdfg()

    # Test sources
    sources = FindReferenceSources().apply_pass(sdfg, {})
    assert len(sources) == 1  # There is only one SDFG
    sources = sources[0]
    assert len(sources) == 5
    assert sources['ref1'] == {dace.Memlet('A[5, 0:22]')}
    assert sources['ref2'] == {dace.Memlet('A[0:22, 5]')}
    assert sources['refB'] == {dace.Memlet('B[4, 0:20]')}

    if reftoview:
        sdfg.simplify()
        assert not any(isinstance(v, dace.data.Reference) for v in sdfg.arrays.values())

    # Test correctness
    A = np.random.rand(22, 22)
    B = np.random.rand(20, 20)
    ref = np.copy(B)
    ref[4, 4] = 0.25 * (A[5, 4] + A[4, 5] + A[3, 4] + A[4, 3])
    sdfg(A=A, B=B)
    assert np.allclose(B, ref)


@pytest.mark.parametrize('reftoview', (False, True))
def test_scoped(reftoview):
    sdfg = _create_scoped_sdfg()

    # Test sources
    sources = FindReferenceSources().apply_pass(sdfg, {})
    assert len(sources) == 1  # There is only one SDFG
    sources = sources[0]
    assert len(sources) == 1
    assert sources['ref'] == {dace.Memlet('A[2:4, 3]')}

    if reftoview:
        sdfg.simplify()
        assert not any(isinstance(v, dace.data.Reference) for v in sdfg.arrays.values())

    # Test correctness
    A = np.random.rand(20, 20)
    B = np.random.rand(20, 20)
    ref = np.copy(A)

    ref[2:4, 3] = B[1, 0:2]
    ref[10, 0:2] = ref[2:4, 3] + 1

    sdfg(A=A, B=B)
    assert np.allclose(A, ref)


@pytest.mark.parametrize('reftoview', (False, True))
def test_scoped_empty_memlet(reftoview):
    sdfg = _create_scoped_empty_memlet_sdfg()

    # Test sources
    sources = FindReferenceSources().apply_pass(sdfg, {})
    assert len(sources) == 1  # There is only one SDFG
    sources = sources[0]
    assert len(sources) == 1
    assert sources['ref'] == {dace.Memlet('A[2:4, 3]')}

    if reftoview:
        sdfg.simplify()
        assert not any(isinstance(v, dace.data.Reference) for v in sdfg.arrays.values())

    # Test correctness
    A = np.random.rand(20, 20)
    B = np.random.rand(20, 20)
    ref = np.copy(B)
    ref[10, 0:2] = A[2:4, 3] + 1

    sdfg(A=A, B=B)
    assert np.allclose(B, ref)


@pytest.mark.parametrize('reftoview', (False, True))
def test_reference_neighbors(reftoview):
    sdfg = _create_neighbor_sdfg()

    # Test sources
    sources = FindReferenceSources().apply_pass(sdfg, {})
    assert len(sources) == 1  # There is only one SDFG
    sources = sources[0]
    assert len(sources) == 1
    assert sources['ref'] == {dace.Memlet('A[2:4, 3:5]')}

    if reftoview:
        sdfg.simplify()
        assert not any(isinstance(v, dace.data.Reference) for v in sdfg.arrays.values())

    # Test correctness
    A = np.random.rand(20, 20)
    B = np.random.rand(2, 2)
    ref = np.copy(A)
    ref[2:4, 3:5] = B + 2
    ref[3, 3] += np.sum(ref[2, 3:5])
    ref[3, 4] = ref[2, 3] + 1

    sdfg(A=A, B=B)
    assert np.allclose(A, ref)


def test_reference_loop_nonfree():
    sdfg = _create_loop_nonfree_symbols_sdfg()

    # Test sources
    sources = FindReferenceSources().apply_pass(sdfg, {})
    assert len(sources) == 1  # There is only one SDFG
    sources = sources[0]
    assert len(sources) == 1
    assert sources['ref'] == {dace.Memlet('A[i] -> [0]')}

    # Test loop-to-map - should fail to apply
    from dace.transformation.interstate import LoopToMap
    assert sdfg.apply_transformations(LoopToMap) == 0

    # Test reference-to-view - should fail to apply
    result = Pipeline([ReferenceToView()]).apply_pass(sdfg, {})
    assert 'ReferenceToView' not in result or not result['ReferenceToView']

    # Test correctness
    A = np.random.rand(20)
    ref = np.copy(A)
    ref[-1] = 1
    sdfg(A=A)
    assert np.allclose(ref, A)


@pytest.mark.parametrize('reftoview', (False, True))
def test_reference_loop_internal_use(reftoview):
    sdfg = _create_loop_reference_internal_use()

    # Test sources
    sources = FindReferenceSources().apply_pass(sdfg, {})
    assert len(sources) == 1  # There is only one SDFG
    sources = sources[0]
    assert len(sources) == 1
    assert sources['ref'] == {dace.Memlet('A[i]')}

    if reftoview:
        sdfg.simplify()
        assert not any(isinstance(v, dace.data.Reference) for v in sdfg.arrays.values())

    # Test correctness
    A = np.random.rand(20)
    ref = np.copy(A)
    ref[:] = 1
    sdfg(A=A)
    assert np.allclose(ref, A)


def test_reference_loop_nonfree_internal_use():
    sdfg = _create_loop_reference_nonfree_internal_use()

    # Test sources
    sources = FindReferenceSources().apply_pass(sdfg, {})
    assert len(sources) == 1  # There is only one SDFG
    sources = sources[0]
    assert len(sources) == 1
    assert sources['ref'] == {dace.Memlet('A[i]')}

    # Test reference-to-view - should fail to apply
    result = Pipeline([ReferenceToView()]).apply_pass(sdfg, {})
    assert 'ReferenceToView' not in result or not result['ReferenceToView']

    # Test correctness
    A = np.random.rand(20)
    ref = np.copy(A)
    ref[-1] = 1
    sdfg(A=A)
    assert np.allclose(ref, A)


@pytest.mark.parametrize(('array_outside_scope', 'depends_on_iterate'), ((False, True), (False, True)))
def test_ref2view_refset_in_scope(array_outside_scope, depends_on_iterate):
    sdfg = dace.SDFG('reftest')
    sdfg.add_array('A', [20], dace.float64)
    sdfg.add_array('B', [20], dace.float64)
    sdfg.add_reference('ref', [1], dace.float64)

    memlet_string = 'A[i]' if depends_on_iterate else 'A[3]'

    state = sdfg.add_state()
    me, mx = state.add_map('somemap', dict(i='0:20'))
    arr = state.add_access('A')
    ref = state.add_access('ref')
    write = state.add_write('B')

    if array_outside_scope:
        state.add_edge_pair(me, ref, arr, dace.Memlet(memlet_string), internal_connector='set')
    else:
        state.add_nedge(me, arr, dace.Memlet())
        state.add_edge(arr, None, ref, 'set', dace.Memlet(memlet_string))

    t = state.add_tasklet('addone', {'inp'}, {'out'}, 'out = inp + 1')
    state.add_edge(ref, None, t, 'inp', dace.Memlet('ref'))
    state.add_edge_pair(mx, t, write, dace.Memlet('B[i]'), internal_connector='out')

    # Test sources
    sources = FindReferenceSources().apply_pass(sdfg, {})
    assert len(sources) == 1  # There is only one SDFG
    sources = sources[0]
    assert len(sources) == 1
    assert sources['ref'] == {dace.Memlet(memlet_string)}

    # Test correctness before pass
    A = np.random.rand(20)
    B = np.random.rand(20)
    ref = (A + 1) if depends_on_iterate else (A[3] + 1)
    sdfg(A=A, B=B)
    assert np.allclose(B, ref)

    # Test reference-to-view - should fail to apply
    result = Pipeline([ReferenceToView()]).apply_pass(sdfg, {})
    if depends_on_iterate:
        assert 'ReferenceToView' not in result or not result['ReferenceToView']
    else:
        assert result['ReferenceToView'] == {'ref'}

    # Test correctness after pass
    if not depends_on_iterate:
        A = np.random.rand(20)
        B = np.random.rand(20)
        ref = (A + 1) if depends_on_iterate else (A[3] + 1)
        sdfg(A=A, B=B)
        assert np.allclose(B, ref)


def test_ref2view_reconnection():
    """
    Tests a regression in which ReferenceToView disconnects an existing weakly-connected state
    and thus creating a race condition.
    """
    sdfg = dace.SDFG('reftest')
    sdfg.add_array('A', [2], dace.float64)
    sdfg.add_array('B', [1], dace.float64)
    sdfg.add_reference('ref', [1], dace.float64)

    state = sdfg.add_state()
    a2 = state.add_access('A')
    ref = state.add_access('ref')
    b = state.add_access('B')

    t2 = state.add_tasklet('addone', {'inp'}, {'out'}, 'out = inp + 1')
    state.add_edge(ref, None, t2, 'inp', dace.Memlet('ref[0]'))
    state.add_edge(t2, 'out', b, None, dace.Memlet('B[0]'))
    state.add_edge(a2, None, ref, 'set', dace.Memlet('A[1]'))

    t1 = state.add_tasklet('addone', {'inp'}, {'out'}, 'out = inp + 1')
    a1 = state.add_access('A')
    state.add_edge(a1, None, t1, 'inp', dace.Memlet('A[1]'))
    state.add_edge(t1, 'out', a2, None, dace.Memlet('A[1]'))

    # Test correctness before pass
    A = np.random.rand(2)
    B = np.random.rand(1)
    ref = (A[1] + 2)
    sdfg(A=A, B=B)
    assert np.allclose(B, ref)

    # Test reference-to-view
    result = Pipeline([ReferenceToView()]).apply_pass(sdfg, {})
    assert result['ReferenceToView'] == {'ref'}

    # Pass should not break order
    assert len(list(nx.weakly_connected_components(state.nx))) == 1

    # Test correctness after pass
    ref = (A[1] + 2)
    sdfg(A=A, B=B)
    assert np.allclose(B, ref)


if __name__ == '__main__':
    test_frontend_reference()
    test_type_annotation_reference()
    test_unset_reference()
    test_reference_branch()
    test_reference_sources_pass()
    test_reference_tasklet_assignment()
    test_reference_tasklet_assignment_analysis()
    test_reference_tasklet_assignment_stree()
    test_reference_tasklet_assignment_reftoview()
    test_twostate(False)
    test_twostate(True)
    test_multisubset(False)
    test_multisubset(True)
    test_scoped(False)
    test_scoped(True)
    test_scoped_empty_memlet(False)
    test_scoped_empty_memlet(True)
    test_reference_neighbors(False)
    test_reference_neighbors(True)
    test_reference_loop_nonfree()
    test_reference_loop_internal_use(False)
    test_reference_loop_internal_use(True)
    test_reference_loop_nonfree_internal_use()
    test_ref2view_refset_in_scope(False, False)
    test_ref2view_refset_in_scope(False, True)
    test_ref2view_refset_in_scope(True, False)
    test_ref2view_refset_in_scope(True, True)
    test_ref2view_reconnection()
