# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Tests conversion of schedule trees to SDFGs.
"""
import dace
import numpy as np
import pytest

from dace.sdfg.analysis.schedule_tree import treenodes as tn
from dace.sdfg.state import LoopRegion
from dace.transformation.pass_pipeline import FixedPointPipeline
from dace.transformation.passes.simplification.control_flow_raising import ControlFlowRaising


def _roundtrip(sdfg: dace.SDFG, expected_node_type: type, simplify: bool) -> dace.SDFG:
    """
    Converts an SDFG to a schedule tree and back, ensuring the tree contains a node of the given type.
    """
    stree = sdfg.as_schedule_tree()
    assert any(type(node) is expected_node_type for node in stree.preorder_traversal())
    return stree.as_sdfg(simplify=simplify)


def test_implicit_inline_and_constants():
    """
    Tests implicit inlining upon roundtrip conversion, as well as constants with conflicting names.
    """

    @dace
    def nester(A: dace.float64[20]):
        A[:] = 12

    @dace.program
    def tester(A: dace.float64[20, 20]):
        for i in dace.map[0:20]:
            nester(A[:, i])

    sdfg = tester.to_sdfg(simplify=False)

    # Inject constant into nested SDFG
    assert len(list(sdfg.all_sdfgs_recursive())) > 1
    sdfg.add_constant('cst', 13)  # Add an unused constant
    sdfg.cfg_list[-1].add_constant('cst', 1, dace.data.Scalar(dace.float64))
    tasklet = next(n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.Tasklet))
    tasklet.code.as_string = tasklet.code.as_string.replace('12', 'cst')

    # Perform a roundtrip conversion
    stree = sdfg.as_schedule_tree()
    new_sdfg = stree.as_sdfg()

    assert len(list(new_sdfg.all_sdfgs_recursive())) == 1
    assert new_sdfg.constants['cst_0'].dtype == np.float64

    # Test SDFG
    a = np.random.rand(20, 20)
    new_sdfg(A=a)  # Tests arg_names
    assert np.allclose(a, 1)


def test_name_propagation():
    name = "my_complicated_sdfg_test_name"
    sdfg = dace.SDFG(name)
    sdfg.add_state("empty", is_start_block=True)

    stree = sdfg.as_schedule_tree()
    assert stree.name == name

    sdfg = stree.as_sdfg()
    assert sdfg.name == name


@pytest.mark.parametrize('simplify', (False, True))
def test_view_of_slice(simplify: bool):

    @dace.program
    def tester(a: dace.float64[30]):
        b = a[1:21]
        b[:] = 5

    new_sdfg = _roundtrip(tester.to_sdfg(simplify=False), tn.ViewNode, simplify)

    a = np.random.rand(30)
    expected = a.copy()
    expected[1:21] = 5
    new_sdfg(a=a)
    assert np.allclose(a, expected)


@pytest.mark.parametrize('simplify', (False, True))
def test_view_read_after_write(simplify: bool):

    @dace.program
    def tester(a: dace.float64[30]):
        b = a[5:25]
        b[3] = b[2] + a[7]
        a[9] = b[3] * 2

    new_sdfg = _roundtrip(tester.to_sdfg(simplify=False), tn.ViewNode, simplify)

    a = np.random.rand(30)
    expected = a.copy()
    expected[8] = expected[7] + expected[7]
    expected[9] = expected[8] * 2
    new_sdfg(a=a)
    assert np.allclose(a, expected)


@pytest.mark.parametrize('simplify', (False, True))
def test_view_of_view(simplify: bool):
    sdfg = dace.SDFG('tester')
    sdfg.add_array('A', [20, 20], dace.float64)
    sdfg.add_array('B', [20, 20], dace.float64)
    sdfg.add_view('Av', [400], dace.float64)
    sdfg.add_view('Avv', [10, 40], dace.float64)
    sdfg.add_view('Bv', [400], dace.float64)
    sdfg.add_view('Bvv', [10, 40], dace.float64)
    state = sdfg.add_state()
    av = state.add_access('Av')
    bv = state.add_access('Bv')
    bvv = state.add_access('Bvv')
    avv = state.add_access('Avv')
    state.add_edge(state.add_read('A'), None, av, None, dace.Memlet('A[0:20, 0:20]'))
    state.add_edge(av, None, avv, 'views', dace.Memlet('Av[0:400]'))
    state.add_edge(avv, None, bvv, None, dace.Memlet('Avv[0:10, 0:40]'))
    state.add_edge(bvv, 'views', bv, None, dace.Memlet('Bv[0:400]'))
    state.add_edge(bv, 'views', state.add_write('B'), None, dace.Memlet('Bv[0:400]'))

    new_sdfg = _roundtrip(sdfg, tn.ViewNode, simplify)

    a = np.random.rand(20, 20)
    b = np.random.rand(20, 20)
    new_sdfg(A=a, B=b)
    assert np.allclose(a, b)


@pytest.mark.parametrize('simplify', (False, True))
def test_view_in_map_scope(simplify: bool):

    @dace.program
    def tester(a: dace.float64[10, 30], b: dace.float64[10]):
        for i in dace.map[0:10]:
            r = a[i]
            b[i] = r[4] + r[5]

    new_sdfg = _roundtrip(tester.to_sdfg(simplify=False), tn.ViewNode, simplify)

    a = np.random.rand(10, 30)
    b = np.random.rand(10)
    new_sdfg(a=a, b=b)
    assert np.allclose(b, a[:, 4] + a[:, 5])


@pytest.mark.parametrize('simplify', (False, True))
def test_view_write_in_map_scope(simplify: bool):

    @dace.program
    def tester(a: dace.float64[10, 30]):
        for i in dace.map[0:10]:
            r = a[i]
            r[3] = i

    new_sdfg = _roundtrip(tester.to_sdfg(simplify=False), tn.ViewNode, simplify)

    a = np.random.rand(10, 30)
    expected = a.copy()
    expected[:, 3] = np.arange(10)
    new_sdfg(a=a)
    assert np.allclose(a, expected)


@pytest.mark.parametrize('simplify', (False, True))
def test_view_passed_to_nested_sdfg(simplify: bool):

    @dace.program
    def nested(x: dace.float64[20]):
        x[:] = x + 1

    @dace.program
    def tester(a: dace.float64[30]):
        nested(a[1:21])

    new_sdfg = _roundtrip(tester.to_sdfg(simplify=False), tn.ViewNode, simplify)

    a = np.random.rand(30)
    expected = a.copy()
    expected[1:21] += 1
    new_sdfg(a=a)
    assert np.allclose(a, expected)


@pytest.mark.parametrize('simplify', (False, True))
def test_dynamic_map_range(simplify: bool):
    H = dace.symbol('H')
    nnz = dace.symbol('nnz')

    @dace.program
    def tester(A_row: dace.uint32[H + 1], A_val: dace.float32[nnz], b: dace.float32[H]):
        for i in dace.map[0:H]:
            for j in dace.map[A_row[i]:A_row[i + 1]]:
                b[i] += A_val[j]

    new_sdfg = _roundtrip(tester.to_sdfg(), tn.DynScopeCopyNode, simplify)

    A_row = np.array([0, 2, 3, 5], dtype=np.uint32)
    A_val = np.random.rand(5).astype(np.float32)
    b = np.zeros(3, dtype=np.float32)
    new_sdfg(A_row=A_row, A_val=A_val, b=b, H=3, nnz=5)
    assert np.allclose(b, [A_val[0] + A_val[1], A_val[2], A_val[3] + A_val[4]])


@pytest.mark.parametrize('simplify', (False, True))
def test_reference_set(simplify: bool):
    sdfg = dace.SDFG('tester')
    sdfg.add_symbol('n', dace.int32)
    sdfg.add_array('A', [20], dace.float64)
    sdfg.add_array('B', [20], dace.float64)
    sdfg.add_array('C', [20], dace.float64)
    sdfg.add_reference('ref', [20], dace.float64)

    init = sdfg.add_state()
    s1 = sdfg.add_state()
    s2 = sdfg.add_state()
    end = sdfg.add_state()
    sdfg.add_edge(init, s1, dace.InterstateEdge('n > 0'))
    sdfg.add_edge(init, s2, dace.InterstateEdge('n <= 0'))
    sdfg.add_edge(s1, end, dace.InterstateEdge())
    sdfg.add_edge(s2, end, dace.InterstateEdge())

    s1.add_edge(s1.add_access('A'), None, s1.add_access('ref'), 'set', dace.Memlet('A[0:20]'))
    s2.add_edge(s2.add_access('B'), None, s2.add_access('ref'), 'set', dace.Memlet('B[0:20]'))
    end.add_nedge(end.add_access('ref'), end.add_access('C'), dace.Memlet('ref[0:20]'))

    FixedPointPipeline([ControlFlowRaising()]).apply_pass(sdfg, {})
    new_sdfg = _roundtrip(sdfg, tn.RefSetNode, simplify)

    a = np.random.rand(20)
    b = np.random.rand(20)
    c = np.random.rand(20)
    new_sdfg(A=a, B=b, C=c, n=1)
    assert np.allclose(c, a)
    new_sdfg(A=a, B=b, C=c, n=0)
    assert np.allclose(c, b)


@pytest.mark.parametrize('simplify', (False, True))
def test_library_call(simplify: bool):

    @dace.program
    def tester(a: dace.float64[5, 4], b: dace.float64[4, 3]):
        return a @ b

    new_sdfg = _roundtrip(tester.to_sdfg(), tn.LibraryCall, simplify)

    a = np.random.rand(5, 4)
    b = np.random.rand(4, 3)
    assert np.allclose(new_sdfg(a=a, b=b), a @ b)


def _inverted_loop_sdfg(name: str, loop: LoopRegion) -> dace.SDFG:
    """
    Creates an SDFG that increments ``A[i]`` in the body of the given (inverted) loop.
    """
    sdfg = dace.SDFG(name)
    sdfg.add_array('A', [10], dace.float64)
    sdfg.add_symbol('i', dace.int64)
    init = sdfg.add_state('init', is_start_block=True)
    sdfg.add_node(loop)
    sdfg.add_edge(init, loop, dace.InterstateEdge(assignments={} if loop.loop_variable else {'i': '0'}))
    body = loop.add_state('body', is_start_block=True)
    tasklet = body.add_tasklet('increment', {'inp'}, {'out'}, 'out = inp + 1')
    body.add_edge(body.add_read('A'), None, tasklet, 'inp', dace.Memlet('A[i]'))
    body.add_edge(tasklet, 'out', body.add_write('A'), None, dace.Memlet('A[i]'))
    if not loop.loop_variable:
        loop.add_state_after(body, 'increment_i', assignments={'i': 'i + 1'})
    return sdfg


@pytest.mark.parametrize('simplify', (False, True))
def test_do_while_loop(simplify: bool):
    # The condition never holds, the body is executed once
    sdfg = _inverted_loop_sdfg('tester', LoopRegion('loop', 'i < 0', inverted=True))
    new_sdfg = _roundtrip(sdfg, tn.DoWhileScope, simplify)

    a = np.zeros(10)
    new_sdfg(A=a)
    assert np.allclose(a, [1] + [0] * 9)


@pytest.mark.parametrize('update_before_condition', (False, True))
@pytest.mark.parametrize('simplify', (False, True))
def test_do_for_loop(update_before_condition: bool, simplify: bool):
    loop = LoopRegion('loop',
                      'i < 3',
                      'i',
                      'i = 0',
                      'i = i + 1',
                      inverted=True,
                      update_before_condition=update_before_condition)
    sdfg = _inverted_loop_sdfg('tester', loop)
    new_sdfg = _roundtrip(sdfg, tn.LoopScope, simplify)

    a = np.zeros(10)
    expected = np.zeros(10)
    sdfg(A=expected)
    new_sdfg(A=a)
    assert np.allclose(a, expected)


if __name__ == '__main__':
    test_implicit_inline_and_constants()
    test_name_propagation()
    for simplify in (False, True):
        test_view_of_slice(simplify)
        test_view_read_after_write(simplify)
        test_view_of_view(simplify)
        test_view_in_map_scope(simplify)
        test_view_write_in_map_scope(simplify)
        test_view_passed_to_nested_sdfg(simplify)
        test_dynamic_map_range(simplify)
        test_reference_set(simplify)
        test_library_call(simplify)
        test_do_while_loop(simplify)
        test_do_for_loop(False, simplify)
        test_do_for_loop(True, simplify)
