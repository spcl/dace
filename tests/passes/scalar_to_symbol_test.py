# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" Tests the scalar to symbol promotion functionality. """
import dace
from dace.transformation.passes import scalar_to_symbol
from dace.sdfg.state import SDFGState
from dace.transformation import transformation as xf, interstate as isxf
from dace.transformation.interstate import loop_detection as ld
from dace import registry
from dace.transformation import helpers as xfh

import collections
import numpy as np
import pytest


def test_find_promotable():
    """ Find promotable and non-promotable symbols. """
    @dace.program
    def testprog1(A: dace.float32[20, 20], scal: dace.float32):
        tmp = dace.ndarray([20, 20], dtype=dace.float32)
        m = dace.define_local_scalar(dace.float32)
        j = dace.ndarray([1], dtype=dace.int64)
        i = 1
        i = 2
        j[:] = 0
        while j[0] < 5:
            tmp[:] = A + j
            for k in dace.map[0:20]:
                with dace.tasklet:
                    inp << scal
                    out >> m(1, lambda a, b: a + b)
                    out = inp
            j += 1
            i += j

    sdfg: dace.SDFG = testprog1.to_sdfg(simplify=False)
    scalars = scalar_to_symbol.find_promotable_scalars(sdfg)
    assert 'i' in scalars
    assert 'j' in scalars


def test_promote_simple():
    """ Simple promotion with Python tasklets. """
    @dace.program
    def testprog2(A: dace.float64[20, 20]):
        j = 5
        A[:] += j

    sdfg: dace.SDFG = testprog2.to_sdfg(simplify=False)
    assert scalar_to_symbol.find_promotable_scalars(sdfg) == {'j'}
    scalar_to_symbol.promote_scalars_to_symbols(sdfg)
    sdfg.apply_transformations_repeated(isxf.StateFusion)

    # There should be two states:
    # [empty] --j=5--> [A->MapEntry->Tasklet->MapExit->A]
    assert sdfg.number_of_nodes() == 2
    assert sdfg.source_nodes()[0].number_of_nodes() == 0
    assert sdfg.sink_nodes()[0].number_of_nodes() == 5
    tasklet = next(n for n in sdfg.sink_nodes()[0] if isinstance(n, dace.nodes.Tasklet))
    assert '+ j' in tasklet.code.as_string

    # Program should produce correct result
    A = np.random.rand(20, 20)
    expected = A + 5
    sdfg(A=A)
    assert np.allclose(A, expected)


def test_promote_simple_c():
    """ Simple promotion with C++ tasklets. """
    @dace.program
    def testprog3(A: dace.float32[20, 20]):
        i = 0
        j = 0
        k = dace.ndarray([1], dtype=dace.int32)
        with dace.tasklet(dace.Language.CPP):
            jj << j
            """
            ii = jj + 1;
            """
            ii >> i
        with dace.tasklet(dace.Language.CPP):
            jin << j
            """
            int something = (int)jin;
            jout = something + 1;
            """
            jout >> j
        with dace.tasklet(dace.Language.CPP):
            """
            kout[0] = 0;
            """
            kout >> k

    sdfg: dace.SDFG = testprog3.to_sdfg(simplify=False)
    scalars = scalar_to_symbol.find_promotable_scalars(sdfg)
    assert scalars == {'i'}
    scalar_to_symbol.promote_scalars_to_symbols(sdfg)
    sdfg.apply_transformations_repeated(isxf.StateFusion)

    # SDFG: [empty] --i=0--> [Tasklet->j] --i=j+1--> [j->Tasklet->j, Tasklet->k]
    assert sdfg.number_of_nodes() == 3
    src_state = sdfg.source_nodes()[0]
    sink_state = sdfg.sink_nodes()[0]
    middle_state = next(s for s in sdfg.nodes() if s not in [src_state, sink_state])
    assert src_state.number_of_nodes() == 0
    assert middle_state.number_of_nodes() == 2
    assert sink_state.number_of_nodes() == 5


def test_promote_disconnect():
    """ Promotion that disconnects tasklet from map. """
    @dace.program
    def testprog4(A: dace.float64[20, 20]):
        j = 5
        A[:] = j

    sdfg: dace.SDFG = testprog4.to_sdfg(simplify=False)
    assert scalar_to_symbol.find_promotable_scalars(sdfg) == {'j'}
    scalar_to_symbol.promote_scalars_to_symbols(sdfg)
    sdfg.apply_transformations_repeated(isxf.StateFusion)

    # There should be two states:
    # [empty] --j=5--> [MapEntry->Tasklet->MapExit->A]
    assert sdfg.number_of_nodes() == 2
    assert sdfg.source_nodes()[0].number_of_nodes() == 0
    assert sdfg.sink_nodes()[0].number_of_nodes() == 4

    # Program should produce correct result
    A = np.random.rand(20, 20)
    expected = np.zeros_like(A)
    expected[:] = 5
    sdfg(A=A)
    assert np.allclose(A, expected)


def test_promote_copy():
    """ Promotion that has a connection to an array and to another symbol. """
    # Create SDFG
    sdfg = dace.SDFG('testprog5')
    sdfg.add_array('A', [20, 20], dace.float64)
    sdfg.add_transient('i', [1], dace.int32)
    sdfg.add_transient('j', [1], dace.int32)
    state = sdfg.add_state()
    state.add_edge(state.add_tasklet('seti', {}, {'out'}, 'out = 0'), 'out', state.add_write('i'), None,
                   dace.Memlet('i'))
    state = sdfg.add_state_after(state)
    state.add_edge(state.add_tasklet('setj', {}, {'out'}, 'out = 5'), 'out', state.add_write('j'), None,
                   dace.Memlet('j'))
    state = sdfg.add_state_after(state)
    state.add_nedge(state.add_read('j'), state.add_write('i'), dace.Memlet('i'))
    state = sdfg.add_state_after(state)
    state.add_nedge(state.add_read('i'), state.add_write('A'), dace.Memlet('A[5, 5]'))

    assert scalar_to_symbol.find_promotable_scalars(sdfg) == {'i', 'j'}
    scalar_to_symbol.promote_scalars_to_symbols(sdfg)
    sdfg.apply_transformations_repeated(isxf.StateFusion)

    # There should be two states:
    # [empty] --i=0,j=5--> [empty] --j=i--> [Tasklet->A]
    assert sdfg.number_of_nodes() == 3
    src_state = sdfg.source_nodes()[0]
    sink_state = sdfg.sink_nodes()[0]
    middle_state = next(s for s in sdfg.nodes() if s not in [src_state, sink_state])
    assert src_state.number_of_nodes() == 0
    assert middle_state.number_of_nodes() == 0
    assert sink_state.number_of_nodes() == 2

    # Program should produce correct result
    A = np.random.rand(20, 20)
    expected = np.copy(A)
    expected[5, 5] = 5.0
    sdfg(A=A)
    assert np.allclose(A, expected)


def test_promote_array_assignment():
    """ Simple promotion with array assignment. """
    @dace.program
    def testprog6(A: dace.float64[20, 20]):
        j = A[1, 1]
        if j >= 0.0:
            A[:] += j

    sdfg: dace.SDFG = testprog6.to_sdfg(simplify=False)
    assert scalar_to_symbol.find_promotable_scalars(sdfg) == {'j'}
    scalar_to_symbol.promote_scalars_to_symbols(sdfg)
    sdfg.apply_transformations_repeated(isxf.StateFusion)

    # There should be 4 states:
    # [empty] --j=A[1, 1]--> [A->MapEntry->Tasklet->MapExit->A] --> [empty]
    #                   \--------------------------------------------/
    assert sdfg.number_of_nodes() == 4
    ctr = collections.Counter(s.number_of_nodes() for s in sdfg)
    assert ctr[0] == 3
    assert ctr[5] == 1

    # Program should produce correct result
    A = np.random.rand(20, 20)
    expected = A + A[1, 1]
    sdfg(A=A)
    assert np.allclose(A, expected)


def test_promote_array_assignment_tasklet():
    """ Simple promotion with array assignment. """
    @dace.program
    def testprog7(A: dace.float64[20, 20]):
        j = dace.define_local_scalar(dace.int64)
        with dace.tasklet:
            inp << A[1, 1]
            out >> j
            out = inp
        A[:] += j

    sdfg: dace.SDFG = testprog7.to_sdfg(simplify=False)
    assert scalar_to_symbol.find_promotable_scalars(sdfg) == {'j'}
    scalar_to_symbol.promote_scalars_to_symbols(sdfg)
    sdfg.apply_transformations_repeated(isxf.StateFusion)

    # There should be two states:
    # [empty] --j=A[1, 1]--> [A->MapEntry->Tasklet->MapExit->A]
    assert sdfg.number_of_nodes() == 2
    assert sdfg.source_nodes()[0].number_of_nodes() == 0
    assert sdfg.sink_nodes()[0].number_of_nodes() == 5

    # Program should produce correct result
    A = np.random.rand(20, 20)
    expected = A + int(A[1, 1])
    sdfg(A=A)
    assert np.allclose(A, expected)


class LoopTester(ld.DetectLoop, xf.MultiStateTransformation):
    """ Tester method that sets loop index on a guard state. """
    def can_be_applied(self, graph, expr_index, sdfg, permissive):
        if super().can_be_applied(graph, expr_index, sdfg, permissive):
            return False
        guard = self.loop_guard
        if hasattr(guard, '_LOOPINDEX'):
            return False
        return True

    def apply(self, graph: dace.SDFGState, sdfg: dace.SDFG):
        guard = self.loop_guard
        edge = sdfg.in_edges(guard)[0]
        loopindex = next(iter(edge.data.assignments.keys()))
        guard._LOOPINDEX = loopindex


def test_promote_loop():
    """ Loop promotion. """
    N = dace.symbol('N')

    @dace.program
    def testprog8(A: dace.float32[20, 20]):
        i = dace.ndarray([1], dtype=dace.int32)
        i = 0
        while i[0] < N:
            A += i
            i += 2

    sdfg: dace.SDFG = testprog8.to_sdfg(simplify=False)
    assert 'i' in scalar_to_symbol.find_promotable_scalars(sdfg)
    scalar_to_symbol.promote_scalars_to_symbols(sdfg)
    sdfg.simplify()
    # TODO: LoopDetection does not apply to loops with a multi-state guard
    # assert sdfg.apply_transformations_repeated(LoopTester) == 1


def test_promote_loops():
    """ Nested loops. """
    N = dace.symbol('N')

    @dace.program
    def testprog9(A: dace.float32[20, 20]):
        i = 0
        while i < N:
            A += i
            for j in range(5):
                k = 0
                while k < N / 2:
                    A += k
                    k += 1
            i += 2

    sdfg: dace.SDFG = testprog9.to_sdfg(simplify=False)
    scalars = scalar_to_symbol.find_promotable_scalars(sdfg)
    assert 'i' in scalars
    assert 'k' in scalars
    scalar_to_symbol.promote_scalars_to_symbols(sdfg)
    sdfg.simplify()
    # TODO: LoopDetection does not apply to loops with a multi-state guard
    # xfh.split_interstate_edges(sdfg)
    # assert sdfg.apply_transformations_repeated(LoopTester) == 3


def test_promote_indirection():
    """ Indirect access in promotion. """
    @dace.program
    def testprog10(A: dace.float64[2, 3, 4, 5], B: dace.float64[4]):
        i = 2
        j = 1
        k = 0

        # Complex numpy expression with double indirection
        B[:] = A[:, i, :, j][k, :]

        # Include tasklet with more than one statement (one with indirection
        # and one without) and two outputs.
        for m in dace.map[0:2]:
            with dace.tasklet:
                a << A(1)[:]
                ii << i
                b1in << B[m]
                b2in << B[m + 2]

                c = a[0, 0, 0, ii]
                b1 = c + 1
                b2 = b2in + 1

                b1 >> B[m]
                b2 >> B[m + 2]

    sdfg: dace.SDFG = testprog10.to_sdfg(simplify=False)
    assert scalar_to_symbol.find_promotable_scalars(sdfg) == {'i', 'j', 'k'}
    scalar_to_symbol.promote_scalars_to_symbols(sdfg)
    for cursdfg in sdfg.all_sdfgs_recursive():
        scalar_to_symbol.remove_symbol_indirection(cursdfg)
    sdfg.simplify()

    assert sdfg.number_of_nodes() == 1
    assert all(e.data.subset.num_elements() == 1 for e in sdfg.node(0).edges()
               if isinstance(e.src, dace.nodes.Tasklet) or isinstance(e.dst, dace.nodes.Tasklet))

    # Check result
    A = np.random.rand(2, 3, 4, 5)
    B = np.random.rand(4)
    expected = np.copy(A[0, 2, :, 1])
    expected[0:2] = A[0, 0, 0, 2] + 1
    expected[2:4] += 1

    sdfg(A=A, B=B)

    assert np.allclose(B, expected)


def test_promote_output_indirection():
    """ Indirect output access in promotion. """
    @dace.program
    def testprog11(A: dace.float64[10]):
        i = 2
        with dace.tasklet:
            ii << i
            a >> A(2)[:]
            a[ii] = ii
            a[ii + 1] = ii + 1

    sdfg: dace.SDFG = testprog11.to_sdfg(simplify=False)
    assert scalar_to_symbol.find_promotable_scalars(sdfg) == {'i'}
    scalar_to_symbol.promote_scalars_to_symbols(sdfg)
    sdfg.simplify()

    assert sdfg.number_of_nodes() == 1

    # Check result
    A = np.random.rand(10)
    expected = np.copy(A)
    expected[2] = 2
    expected[3] = 3
    sdfg(A=A)

    assert np.allclose(A, expected)


def test_promote_indirection_c():
    """ Indirect access in promotion with C++ tasklets. """
    @dace.program
    def testprog12(A: dace.float64[10]):
        i = 2
        with dace.tasklet(dace.Language.CPP):
            ii << i
            a << A(1)[:]
            aout >> A(2)[:]
            '''
            aout[ii] = a[ii + 1];
            aout[ii + 1] = ii + 1;
            '''

    sdfg: dace.SDFG = testprog12.to_sdfg(simplify=False)
    assert scalar_to_symbol.find_promotable_scalars(sdfg) == {'i'}
    scalar_to_symbol.promote_scalars_to_symbols(sdfg)
    assert all('i' in e.data.free_symbols for e in sdfg.sink_nodes()[0].edges())

    sdfg.simplify()
    assert sdfg.number_of_nodes() == 1

    # Check result
    A = np.random.rand(10)
    expected = np.copy(A)
    expected[2] = A[3]
    expected[3] = 3
    sdfg(A=A)

    assert np.allclose(A, expected)


def test_promote_indirection_impossible():
    """ Indirect access that cannot be promoted. """
    @dace.program
    def testprog13(A: dace.float64[20, 20], scal: dace.int32):
        i = 2
        with dace.tasklet:
            s << scal
            a << A(1)[:, :]
            out >> A(1)[:, :]
            out[i, s] = a[s, i]

    sdfg: dace.SDFG = testprog13.to_sdfg(simplify=False)
    assert scalar_to_symbol.find_promotable_scalars(sdfg) == {'i'}
    scalar_to_symbol.promote_scalars_to_symbols(sdfg)
    sdfg.simplify()

    # [A,scal->Tasklet->A]
    assert sdfg.number_of_nodes() == 1
    assert sdfg.sink_nodes()[0].number_of_nodes() == 4

    A = np.random.rand(20, 20)
    expected = np.copy(A)
    expected[2, 1] = expected[1, 2]
    sdfg(A=A, scal=1)

    assert np.allclose(A, expected)


@pytest.mark.parametrize('with_subscript', [False, True])
def test_nested_promotion_connector(with_subscript):
    # Construct SDFG
    postfix = 'a'
    if with_subscript:
        postfix = 'b'
    sdfg = dace.SDFG('testprog14{}'.format(postfix))
    sdfg.add_array('A', [20, 20], dace.float64)
    sdfg.add_array('B', [1], dace.float64)
    sdfg.add_transient('scal', [1], dace.int32)
    initstate = sdfg.add_state()
    initstate.add_edge(initstate.add_tasklet('do', {}, {'out'}, 'out = 5'), 'out', initstate.add_write('scal'), None,
                       dace.Memlet('scal'))
    state = sdfg.add_state_after(initstate)

    nsdfg = dace.SDFG('nested')
    nsdfg.add_array('a', [20, 20], dace.float64)
    nsdfg.add_array('b', [1], dace.float64)
    nsdfg.add_array('s', [1], dace.int32)
    nsdfg.add_symbol('s2', dace.int32)
    nstate1 = nsdfg.add_state()
    nstate2 = nsdfg.add_state()
    nsdfg.add_edge(nstate1, nstate2, dace.InterstateEdge(assignments=dict(s2='s[0]' if with_subscript else 's')))
    a = nstate2.add_read('a')
    t = nstate2.add_tasklet('do', {'inp'}, {'out'}, 'out = inp')
    b = nstate2.add_write('b')
    nstate2.add_edge(a, None, t, 'inp', dace.Memlet('a[s2, s2 + 1]'))
    nstate2.add_edge(t, 'out', b, None, dace.Memlet('b[0]'))

    nnode = state.add_nested_sdfg(nsdfg, None, {'a', 's'}, {'b'})
    aouter = state.add_read('A')
    souter = state.add_read('scal')
    bouter = state.add_write('B')
    state.add_edge(aouter, None, nnode, 'a', dace.Memlet('A'))
    state.add_edge(souter, None, nnode, 's', dace.Memlet('scal'))
    state.add_edge(nnode, 'b', bouter, None, dace.Memlet('B'))
    #######################################################

    # Promotion
    assert scalar_to_symbol.find_promotable_scalars(sdfg) == {'scal'}
    scalar_to_symbol.promote_scalars_to_symbols(sdfg)
    sdfg.simplify()

    assert sdfg.number_of_nodes() == 1
    assert sdfg.node(0).number_of_nodes() == 3
    assert not any(isinstance(n, dace.nodes.NestedSDFG) for n in sdfg.node(0))

    # Correctness
    A = np.random.rand(20, 20)
    B = np.random.rand(1)
    sdfg(A=A, B=B)
    assert B[0] == A[5, 6]


@pytest.mark.parametrize('language', [dace.Language.CPP, dace.Language.Python])
def test_indirection_with_reindex(language):

    N = dace.symbol('N')
    S = dace.symbol('S')

    sdfg = dace.SDFG(f"test_indirection_with_reindex")
    sdfg.add_array('A', shape=[N], dtype=dace.float32, transient=False)
    sdfg.add_array('index_0', shape=[1], dtype=dace.int32, transient=True)
    sdfg.add_array('index_1', shape=[1], dtype=dace.int32, transient=True)
    sdfg.add_array('index_2', shape=[1], dtype=dace.int32, transient=True)
    sdfg.add_array('out', shape=[N], dtype=dace.float32, transient=False)
    sdfg.add_symbol('S', S.dtype)

    state_init1 = sdfg.add_state()
    state_init2 = sdfg.add_state()
    state_init3 = sdfg.add_state()
    state_compute = sdfg.add_state()

    sdfg.add_edge(state_init1, state_init2, dace.InterstateEdge())
    sdfg.add_edge(state_init2, state_init3, dace.InterstateEdge())
    sdfg.add_edge(state_init3, state_compute, dace.InterstateEdge())

    tasklet1 = state_init1.add_tasklet(name="init1",
                                       inputs=[],
                                       outputs=["out"],
                                       code="out = 1;",
                                       language=dace.Language.CPP)
    tasklet2 = state_init2.add_tasklet(name="init2",
                                       inputs=[],
                                       outputs=["out"],
                                       code="out = 2;",
                                       language=dace.Language.CPP)
    tasklet3 = state_init3.add_tasklet(name="init3",
                                       inputs=[],
                                       outputs=["out"],
                                       code="out = 3;",
                                       language=dace.Language.CPP)

    dst = state_init1.add_write("index_0")
    memlet = dace.Memlet(expr="index_0", subset="0")
    state_init1.add_memlet_path(tasklet1, dst, src_conn="out", memlet=memlet)

    dst = state_init2.add_write("index_1")
    memlet = dace.Memlet(expr="index_1", subset="0")
    state_init2.add_memlet_path(tasklet2, dst, src_conn="out", memlet=memlet)

    dst = state_init3.add_write("index_2")
    memlet = dace.Memlet(expr="index_2", subset="0")
    state_init3.add_memlet_path(tasklet3, dst, src_conn="out", memlet=memlet)

    semicolon = ';' if language == dace.Language.CPP else ''
    tasklet = state_compute.add_tasklet(name="add",
                                        inputs=["_A", "_index_0", "_index_1", "_index_2"],
                                        outputs=["_out"],
                                        code=f"_out[_index_2] = _A[_index_0] + _A[_index_1]{semicolon}",
                                        language=language)

    src = state_compute.add_read("A")
    memlet = dace.Memlet(expr="A", subset="S:N")
    state_compute.add_memlet_path(src, tasklet, dst_conn="_A", memlet=memlet)

    src = state_compute.add_read("index_0")
    memlet = dace.Memlet(expr="index_0", subset="0")
    state_compute.add_memlet_path(src, tasklet, dst_conn="_index_0", memlet=memlet)

    src = state_compute.add_read("index_1")
    memlet = dace.Memlet(expr="index_1", subset="0")
    state_compute.add_memlet_path(src, tasklet, dst_conn="_index_1", memlet=memlet)

    src = state_compute.add_read("index_2")
    memlet = dace.Memlet(expr="index_2", subset="0")
    state_compute.add_memlet_path(src, tasklet, dst_conn="_index_2", memlet=memlet)

    dst = state_compute.add_write("out")
    memlet = dace.Memlet(expr="out", subset="S:N")
    state_compute.add_memlet_path(tasklet, dst, src_conn="_out", memlet=memlet)

    scalar_to_symbol.promote_scalars_to_symbols(sdfg)
    sdfg.simplify()

    A = np.array(list(range(10)), dtype=np.float32)
    out = np.zeros((10, ), dtype=np.float32)
    sdfg(A=A, out=out, N=10, S=5)

    assert (np.allclose(A[6] + A[7], out[8]))


def test_multiple_boolop():
    @dace.program
    def tester():
        a = 1
        b = 0
        c = 1
        if a and b and c:
            return 1
        else:
            return 0

    sdfg = tester.to_sdfg(simplify=False)
    scalar_to_symbol.promote_scalars_to_symbols(sdfg)
    sdfg.validate()

    assert tester() == 0


def test_multidim_cpp():
    sdfg = dace.SDFG('tester')
    sdfg.add_array('A', [20, 10], dace.float64)
    sdfg.add_scalar('sz1', dace.int32, transient=True)
    sdfg.add_scalar('sz2', dace.int32, transient=True)
    sdfg.add_scalar('ind1', dace.int32, transient=True)
    sdfg.add_scalar('ind2', dace.int32, transient=True)

    state = sdfg.add_state()
    state.add_edge(state.add_tasklet('s1', {}, {'o'}, 'o = 20;', language=dace.Language.CPP), 'o',
                   state.add_write('sz1'), None, dace.Memlet('sz1'))
    state.add_edge(state.add_tasklet('s2', {}, {'o'}, 'o = 10;', language=dace.Language.CPP), 'o',
                   state.add_write('sz2'), None, dace.Memlet('sz2'))

    state = sdfg.add_state_after(state)
    t1 = state.add_tasklet('w1', {'i'}, {'o'}, 'o = i - 5;', language=dace.Language.CPP)
    t2 = state.add_tasklet('w2', {'i'}, {'o'}, 'o = i - 3;', language=dace.Language.CPP)
    state.add_edge(state.add_read('sz1'), None, t1, 'i', dace.Memlet('sz1'))
    state.add_edge(state.add_read('sz2'), None, t2, 'i', dace.Memlet('sz2'))
    state.add_edge(t1, 'o', state.add_write('ind1'), None, dace.Memlet('ind1'))
    state.add_edge(t2, 'o', state.add_write('ind2'), None, dace.Memlet('ind2'))

    state = sdfg.add_state_after(state)
    t3 = state.add_tasklet('warr', {'i1', 'i2'}, {'arr'}, 'arr[i1][i2] = 1.0;', language=dace.Language.CPP)
    state.add_edge(state.add_read('ind1'), None, t3, 'i1', dace.Memlet('ind1'))
    state.add_edge(state.add_read('ind2'), None, t3, 'i2', dace.Memlet('ind2'))
    state.add_edge(t3, 'arr', state.add_write('A'), None, dace.Memlet('A'))

    scalar_to_symbol.ScalarToSymbolPromotion().apply_pass(sdfg, {})
    new_edge = sdfg.sink_nodes()[0].edges()[0]

    assert new_edge.data.data == 'A'
    assert str(new_edge.data.subset) == 'ind1, ind2'


def test_dynamic_mapind():
    @dace.program
    def prog(inp: dace.int32[4, 2], out: dace.float64[5, 5]):
        A = np.zeros((5, 5))
        E = inp.shape[1]

        for e in dace.map[0:E]:
            with dace.tasklet:
                # Multiple edges are allowed.
                a << inp[0, e]
                b << inp[1, e]
                o[a, b] = 1
                o >> A(-1, lambda a, b: a + b)

        out[:] = A

    sdfg = prog.to_sdfg(simplify=False)
    promoted = scalar_to_symbol.ScalarToSymbolPromotion().apply_pass(sdfg, {})
    assert 'E' in promoted
    sdfg.compile()


if __name__ == '__main__':
    test_find_promotable()
    test_promote_simple()
    test_promote_simple_c()
    test_promote_disconnect()
    test_promote_copy()
    test_promote_array_assignment()
    test_promote_array_assignment_tasklet()
    test_promote_loop()
    test_promote_loops()
    test_promote_indirection()
    test_promote_indirection_c()
    test_promote_output_indirection()
    test_promote_indirection_impossible()
    test_nested_promotion_connector(False)
    test_nested_promotion_connector(True)
    test_indirection_with_reindex(dace.Language.CPP)
    test_indirection_with_reindex(dace.Language.Python)
    test_multiple_boolop()
    test_multidim_cpp()
    test_dynamic_mapind()
