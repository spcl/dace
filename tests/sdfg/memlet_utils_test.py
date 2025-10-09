# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.

import dace
import numpy as np
import pytest
from dace import symbolic
from dace.sdfg import memlet_utils as mu
import re
from typing import Tuple, Optional


def _replace_zero_with_one(memlet: dace.Memlet) -> dace.Memlet:
    for i, s in enumerate(memlet.subset):
        if s == 0:
            memlet.subset[i] = 1
    return memlet


@pytest.mark.parametrize('filter_type', ['none', 'same_array', 'different_array'])
def test_replace_memlet(filter_type):
    # Prepare SDFG
    sdfg = dace.SDFG('replace_memlet')
    sdfg.add_array('A', [2, 2], dace.float64)
    sdfg.add_array('B', [1], dace.float64)
    state1 = sdfg.add_state()
    state2 = sdfg.add_state()
    state3 = sdfg.add_state()
    end_state = sdfg.add_state()
    sdfg.add_edge(state1, state2, dace.InterstateEdge('A[0, 0] > 0'))
    sdfg.add_edge(state1, state3, dace.InterstateEdge('A[0, 0] <= 0'))
    sdfg.add_edge(state2, end_state, dace.InterstateEdge())
    sdfg.add_edge(state3, end_state, dace.InterstateEdge())

    t2 = state2.add_tasklet('write_one', {}, {'out'}, 'out = 1')
    t3 = state3.add_tasklet('write_two', {}, {'out'}, 'out = 2')
    w2 = state2.add_write('B')
    w3 = state3.add_write('B')
    state2.add_memlet_path(t2, w2, src_conn='out', memlet=dace.Memlet('B'))
    state3.add_memlet_path(t3, w3, src_conn='out', memlet=dace.Memlet('B'))

    # Filter memlets
    if filter_type == 'none':
        filter = set()
    elif filter_type == 'same_array':
        filter = {'A'}
    elif filter_type == 'different_array':
        filter = {'B'}

    # Replace memlets in conditions
    replacer = mu.MemletReplacer(sdfg.arrays, _replace_zero_with_one, filter)
    for e in sdfg.edges():
        e.data.condition.code[0] = replacer.visit(e.data.condition.code[0])

    # Compile and run
    sdfg.compile()

    A = np.array([[1, 1], [1, -1]], dtype=np.float64)
    B = np.array([0], dtype=np.float64)
    sdfg(A=A, B=B)

    if filter_type in {'none', 'same_array'}:
        assert B[0] == 2
    else:
        assert B[0] == 1


def _perform_non_lin_delin_test(sdfg: dace.SDFG, edge) -> bool:
    assert sdfg.number_of_nodes() == 1
    state: dace.SDFGState = sdfg.states()[0]
    assert state.number_of_nodes() == 2
    assert state.number_of_edges() == 1
    assert all(isinstance(node, dace.nodes.AccessNode) for node in state.nodes())
    assert isinstance(edge.src, dace.nodes.AccessNode)
    assert isinstance(edge.dst, dace.nodes.AccessNode)
    sdfg.validate()

    a = np.random.rand(*sdfg.arrays["a"].shape)
    b_unopt = np.random.rand(*sdfg.arrays["b"].shape)
    b_opt = b_unopt.copy()
    sdfg(a=a, b=b_unopt)

    assert mu.can_memlet_be_turned_into_a_map(edge, state, sdfg)
    mu.memlet_to_map(edge, state, sdfg)
    assert state.number_of_nodes() == 5

    # Now looking for the tasklet and checking if the memlets follows the expected
    #  simple pattern.
    tasklet: dace.nodes.Tasklet = next(iter([node for node in state.nodes() if isinstance(node, dace.nodes.Tasklet)]))
    pattern: re.Pattern = re.compile(r"(__j[0-9])|(__j[0-9]+\s*\+\s*[0-9]+)|([0-9]+)")

    assert state.in_degree(tasklet) == 1
    assert state.out_degree(tasklet) == 1
    in_edge = next(iter(state.in_edges(tasklet)))
    out_edge = next(iter(state.out_edges(tasklet)))

    assert all(pattern.fullmatch(str(idxs[0]).strip())
               for idxs in in_edge.data.src_subset), f"IN: {in_edge.data.src_subset}"
    assert all(pattern.fullmatch(str(idxs[0]).strip())
               for idxs in out_edge.data.dst_subset), f"OUT: {out_edge.data.dst_subset}"

    # Now call it again after the optimization.
    sdfg(a=a, b=b_opt)
    assert np.allclose(b_unopt, b_opt)

    return True


def _make_non_lin_delin_sdfg(
    shape_a: Tuple[int, ...],
    shape_b: Optional[Tuple[int, ...]] = None
) -> Tuple[dace.SDFG, dace.SDFGState, dace.nodes.AccessNode, dace.nodes.AccessNode]:

    if shape_b is None:
        shape_b = shape_a

    sdfg = dace.SDFG("bypass1")
    state = sdfg.add_state(is_start_block=True)

    ac = []
    for name, shape in [('a', shape_a), ('b', shape_b)]:
        sdfg.add_array(
            name=name,
            shape=shape,
            dtype=dace.float64,
            transient=False,
        )
        ac.append(state.add_access(name))

    return sdfg, state, ac[0], ac[1]


def test_non_lin_delin_1():
    sdfg, state, a, b = _make_non_lin_delin_sdfg((10, 10))
    e = state.add_nedge(
        a,
        b,
        dace.Memlet("a[0:10, 0:10] -> [0:10, 0:10]"),
    )
    _perform_non_lin_delin_test(sdfg, e)


def test_non_lin_delin_2():
    sdfg, state, a, b = _make_non_lin_delin_sdfg((10, 10), (100, 100))
    e = state.add_nedge(
        a,
        b,
        dace.Memlet("a[0:10, 0:10] -> [50:60, 40:50]"),
    )
    _perform_non_lin_delin_test(sdfg, e)


def test_non_lin_delin_3():
    sdfg, state, a, b = _make_non_lin_delin_sdfg((100, 100), (100, 100))
    e = state.add_nedge(
        a,
        b,
        dace.Memlet("a[1:11, 20:30] -> [50:60, 40:50]"),
    )
    _perform_non_lin_delin_test(sdfg, e)


def test_non_lin_delin_4():
    sdfg, state, a, b = _make_non_lin_delin_sdfg((100, 4, 100), (100, 100))
    e = state.add_nedge(
        a,
        b,
        dace.Memlet("a[1:11, 2, 20:30] -> [50:60, 40:50]"),
    )
    _perform_non_lin_delin_test(sdfg, e)


def test_non_lin_delin_5():
    sdfg, state, a, b = _make_non_lin_delin_sdfg((100, 4, 100), (100, 10, 100))
    e = state.add_nedge(
        a,
        b,
        dace.Memlet("a[1:11, 2, 20:30] -> [50:60, 4, 40:50]"),
    )
    _perform_non_lin_delin_test(sdfg, e)


def test_non_lin_delin_6():
    sdfg, state, a, b = _make_non_lin_delin_sdfg((100, 100), (100, 10, 100))
    e = state.add_nedge(
        a,
        b,
        dace.Memlet("a[1:11, 20:30] -> [50:60, 4, 40:50]"),
    )
    _perform_non_lin_delin_test(sdfg, e)


def test_non_lin_delin_7():
    sdfg, state, a, b = _make_non_lin_delin_sdfg((10, 10), (20, 20))
    e = state.add_nedge(
        a,
        b,
        dace.Memlet("b[5:15, 6:16]"),
    )
    _perform_non_lin_delin_test(sdfg, e)


def test_non_lin_delin_8():
    sdfg, state, a, b = _make_non_lin_delin_sdfg((20, 20), (10, 10))
    e = state.add_nedge(
        a,
        b,
        dace.Memlet("a[5:15, 6:16]"),
    )
    _perform_non_lin_delin_test(sdfg, e)


if __name__ == '__main__':
    test_replace_memlet('none')
    test_replace_memlet('same_array')
    test_replace_memlet('different_array')

    test_non_lin_delin_1()
    test_non_lin_delin_2()
    test_non_lin_delin_3()
    test_non_lin_delin_4()
    test_non_lin_delin_5()
    test_non_lin_delin_6()
    test_non_lin_delin_7()
    test_non_lin_delin_8()
