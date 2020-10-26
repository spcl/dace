# Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
""" Tests the scalar to symbol promotion functionality. """
import dace
from dace.sdfg.analysis import scalar_to_symbol
from dace.transformation import interstate as isxf
from dace.transformation.interstate import loop_detection as ld
from dace import registry
from dace.transformation import helpers as xfh

import numpy as np


def test_find_promotable():
    """ Find promotable and non-promotable symbols. """
    @dace.program
    def testprog(A: dace.float32[20, 20], scal: dace.float32):
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

    sdfg: dace.SDFG = testprog.to_sdfg(strict=False)
    scalars = scalar_to_symbol.find_promotable_scalars(sdfg)
    assert scalars == {'i', 'j'}


def test_promote_simple():
    """ Simple promotion with Python tasklets. """
    @dace.program
    def testprog(A: dace.float64[20, 20]):
        j = 5
        A[:] += j

    sdfg: dace.SDFG = testprog.to_sdfg(strict=False)
    assert scalar_to_symbol.find_promotable_scalars(sdfg) == {'j'}
    scalar_to_symbol.promote_scalars_to_symbols(sdfg)
    sdfg.apply_transformations_repeated(isxf.StateFusion)

    # There should be two states:
    # [empty] --j=5--> [A->MapEntry->Tasklet->MapExit->A]
    assert sdfg.number_of_nodes() == 2
    assert sdfg.source_nodes()[0].number_of_nodes() == 0
    assert sdfg.sink_nodes()[0].number_of_nodes() == 5
    tasklet = next(n for n in sdfg.sink_nodes()[0]
                   if isinstance(n, dace.nodes.Tasklet))
    assert '+ j' in tasklet.code.as_string

    # Program should produce correct result
    A = np.random.rand(20, 20)
    expected = A + 5
    sdfg(A=A)
    assert np.allclose(A, expected)


def test_promote_simple_c():
    """ Simple promotion with C++ tasklets. """
    @dace.program
    def testprog(A: dace.float32[20, 20]):
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

    sdfg: dace.SDFG = testprog.to_sdfg(strict=False)
    scalars = scalar_to_symbol.find_promotable_scalars(sdfg)
    assert scalars == {'i'}
    scalar_to_symbol.promote_scalars_to_symbols(sdfg)
    sdfg.apply_transformations_repeated(isxf.StateFusion)

    # SDFG: [empty] --i=0--> [Tasklet->j] --i=j+1--> [j->Tasklet->j, Tasklet->k]
    assert sdfg.number_of_nodes() == 3
    src_state = sdfg.source_nodes()[0]
    sink_state = sdfg.sink_nodes()[0]
    middle_state = next(s for s in sdfg.nodes()
                        if s not in [src_state, sink_state])
    assert src_state.number_of_nodes() == 0
    assert middle_state.number_of_nodes() == 2
    assert sink_state.number_of_nodes() == 5


def test_promote_disconnect():
    """ Promotion that disconnects tasklet from map. """
    @dace.program
    def testprog(A: dace.float64[20, 20]):
        j = 5
        A[:] = j

    sdfg: dace.SDFG = testprog.to_sdfg(strict=False)
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
    sdfg = dace.SDFG('testprog')
    sdfg.add_array('A', [20, 20], dace.float64)
    sdfg.add_transient('i', [1], dace.int32)
    sdfg.add_transient('j', [1], dace.int32)
    state = sdfg.add_state()
    state.add_edge(state.add_tasklet('seti', {}, {'out'}, 'out = 0'), 'out',
                   state.add_write('i'), None, dace.Memlet('i'))
    state = sdfg.add_state_after(state)
    state.add_edge(state.add_tasklet('setj', {}, {'out'}, 'out = 5'), 'out',
                   state.add_write('j'), None, dace.Memlet('j'))
    state = sdfg.add_state_after(state)
    state.add_nedge(state.add_read('j'), state.add_write('i'), dace.Memlet('i'))
    state = sdfg.add_state_after(state)
    state.add_nedge(state.add_read('i'), state.add_write('A'),
                    dace.Memlet('A[5, 5]'))

    assert scalar_to_symbol.find_promotable_scalars(sdfg) == {'i', 'j'}
    scalar_to_symbol.promote_scalars_to_symbols(sdfg)
    sdfg.apply_transformations_repeated(isxf.StateFusion)

    # There should be two states:
    # [empty] --i=0,j=5--> [empty] --j=i--> [Tasklet->A]
    assert sdfg.number_of_nodes() == 3
    src_state = sdfg.source_nodes()[0]
    sink_state = sdfg.sink_nodes()[0]
    middle_state = next(s for s in sdfg.nodes()
                        if s not in [src_state, sink_state])
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
    def testprog(A: dace.float64[20, 20]):
        j = A[1, 1]
        A[:] += j

    sdfg: dace.SDFG = testprog.to_sdfg(strict=False)
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
    expected = A + A[1, 1]
    sdfg(A=A)
    assert np.allclose(A, expected)


def test_promote_array_assignment_tasklet():
    """ Simple promotion with array assignment. """
    @dace.program
    def testprog(A: dace.float64[20, 20]):
        j = dace.define_local_scalar(dace.float64)
        with dace.tasklet:
            inp << A[1, 1]
            out >> j
            out = inp
        A[:] += j

    sdfg: dace.SDFG = testprog.to_sdfg(strict=False)
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
    expected = A + A[1, 1]
    sdfg(A=A)
    assert np.allclose(A, expected)


@registry.autoregister
class LoopTester(ld.DetectLoop):
    """ Tester method that sets loop index on a guard state. """
    @staticmethod
    def can_be_applied(graph, candidate, expr_index, sdfg, strict):
        if not ld.DetectLoop.can_be_applied(graph, candidate, expr_index, sdfg,
                                            strict):
            return False
        guard = graph.node(candidate[ld.DetectLoop._loop_guard])
        if hasattr(guard, '_LOOPINDEX'):
            return False
        return True

    def apply(self, sdfg: dace.SDFG):
        guard = sdfg.node(self.subgraph[ld.DetectLoop._loop_guard])
        edge = sdfg.in_edges(guard)[0]
        loopindex = next(iter(edge.data.assignments.keys()))
        guard._LOOPINDEX = loopindex


def test_promote_loop():
    """ Loop promotion. """
    N = dace.symbol('N')

    @dace.program
    def testprog(A: dace.float32[20, 20]):
        i = dace.ndarray([1], dtype=dace.int32)
        i = 0
        while i[0] < N:
            A += i
            i += 2

    sdfg: dace.SDFG = testprog.to_sdfg(strict=False)
    assert scalar_to_symbol.find_promotable_scalars(sdfg) == {'i'}
    scalar_to_symbol.promote_scalars_to_symbols(sdfg)
    sdfg.apply_strict_transformations()
    assert sdfg.apply_transformations_repeated(LoopTester) == 1


def test_promote_loops():
    """ Nested loops. """
    N = dace.symbol('N')

    @dace.program
    def testprog(A: dace.float32[20, 20]):
        i = 0
        while i < N:
            A += i
            for j in range(5):
                k = 0
                while k < N / 2:
                    A += k
                    k += 1
            i += 2

    sdfg: dace.SDFG = testprog.to_sdfg(strict=False)
    assert scalar_to_symbol.find_promotable_scalars(sdfg) == {'i', 'k'}
    scalar_to_symbol.promote_scalars_to_symbols(sdfg)
    sdfg.apply_strict_transformations()
    xfh.split_interstate_edges(sdfg)
    assert sdfg.apply_transformations_repeated(LoopTester) == 3


def test_promote_indirection():
    """ Indirect access in promotion. """
    @dace.program
    def testprog(A: dace.float64[2, 3, 4, 5], B: dace.float64[4]):
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

    sdfg: dace.SDFG = testprog.to_sdfg(strict=False)
    assert scalar_to_symbol.find_promotable_scalars(sdfg) == {'i', 'j', 'k'}
    scalar_to_symbol.promote_scalars_to_symbols(sdfg)
    sdfg.apply_strict_transformations()

    assert sdfg.number_of_nodes() == 2

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
    def testprog(A: dace.float64[10]):
        i = 2
        with dace.tasklet:
            ii << i
            a >> A(2)[:]
            a[ii] = ii
            a[ii + 1] = ii + 1

    sdfg: dace.SDFG = testprog.to_sdfg(strict=False)
    assert scalar_to_symbol.find_promotable_scalars(sdfg) == {'i'}
    scalar_to_symbol.promote_scalars_to_symbols(sdfg)
    sdfg.apply_strict_transformations()

    assert sdfg.number_of_nodes() == 2

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
    def testprog(A: dace.float64[10]):
        i = 2
        with dace.tasklet(dace.Language.CPP):
            ii << i
            a << A(1)[:]
            aout >> A(2)[:]
            '''
            aout[ii] = a[ii + 1];
            aout[ii + 1] = ii + 1;
            '''

    sdfg: dace.SDFG = testprog.to_sdfg(strict=False)
    assert scalar_to_symbol.find_promotable_scalars(sdfg) == {'i'}
    scalar_to_symbol.promote_scalars_to_symbols(sdfg)
    sdfg.apply_strict_transformations()

    assert sdfg.number_of_nodes() == 2
    assert all('i' in e.data.free_symbols for e in sdfg.sink_nodes()[0].edges())

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
    def testprog(A: dace.float64[20, 20], scal: dace.int32):
        i = 2
        with dace.tasklet:
            s << scal
            a << A(1)[:, :]
            out >> A(1)[:, :]
            out[i, s] = a[s, i]

    sdfg: dace.SDFG = testprog.to_sdfg(strict=False)
    assert scalar_to_symbol.find_promotable_scalars(sdfg) == {'i'}
    scalar_to_symbol.promote_scalars_to_symbols(sdfg)
    sdfg.apply_strict_transformations()

    # [A,scal->Tasklet->A]
    assert sdfg.number_of_nodes() == 2
    assert sdfg.sink_nodes()[0].number_of_nodes() == 4

    A = np.random.rand(20, 20)
    expected = np.copy(A)
    expected[2, 1] = expected[1, 2]
    sdfg(A=A, scal=1)

    assert np.allclose(A, expected)


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
