# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.

import pytest

import dace
from dace.sdfg import utils as sdutil
from dace.transformation.pass_pipeline import Pipeline
from dace.transformation.passes.array_elimination import ArrayElimination


def test_redundant_simple():

    @dace.program
    def tester(A: dace.float64[20], B: dace.float64[20]):
        e = dace.ndarray([20], dace.float64)
        f = dace.ndarray([20], dace.float64)
        g = dace.ndarray([20], dace.float64)
        h = dace.ndarray([20], dace.float64)
        c = A + 1
        d = A + 2
        e[:] = c
        f[:] = d
        g[:] = f
        h[:] = d
        B[:] = g + e

    sdfg = tester.to_sdfg(simplify=False)
    sdutil.inline_sdfgs(sdfg)
    sdutil.fuse_states(sdfg)
    sdutil.inline_sdfgs(sdfg)
    sdutil.fuse_states(sdfg)
    Pipeline([ArrayElimination()]).apply_pass(sdfg, {})
    assert len(sdfg.arrays) == 4


def test_merge_simple():
    sdfg = dace.SDFG('tester')
    sdfg.add_array('A', [20], dace.float64)
    sdfg.add_array('B', [20], dace.float64)

    state = sdfg.add_state()
    a1 = state.add_read('A')
    a2 = state.add_read('A')
    b1 = state.add_write('B')
    b2 = state.add_write('B')
    t1 = state.add_tasklet('doit1', {'a'}, {'b'}, 'b = a')
    t2 = state.add_tasklet('doit2', {'a'}, {'b'}, 'b = a')
    state.add_edge(a1, None, t1, 'a', dace.Memlet('A[0]'))
    state.add_edge(a2, None, t2, 'a', dace.Memlet('A[1]'))
    state.add_edge(t1, 'b', b1, None, dace.Memlet('B[0]'))
    state.add_edge(t2, 'b', b2, None, dace.Memlet('B[1]'))

    Pipeline([ArrayElimination()]).apply_pass(sdfg, {})
    assert len(state.data_nodes()) == 2


if __name__ == '__main__':
    test_redundant_simple()
    test_merge_simple()
# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.

import pytest

import dace
from dace.sdfg import utils as sdutil
from dace.transformation.pass_pipeline import Pipeline
from dace.transformation.passes.array_elimination import ArrayElimination
from dace.sdfg.state import LoopRegion
from dace.transformation.passes import analysis as ap


def test_redundant_simple():

    @dace.program
    def tester(A: dace.float64[20], B: dace.float64[20]):
        e = dace.ndarray([20], dace.float64)
        f = dace.ndarray([20], dace.float64)
        g = dace.ndarray([20], dace.float64)
        h = dace.ndarray([20], dace.float64)
        c = A + 1
        d = A + 2
        e[:] = c
        f[:] = d
        g[:] = f
        h[:] = d
        B[:] = g + e

    sdfg = tester.to_sdfg(simplify=False)
    sdutil.inline_sdfgs(sdfg)
    sdutil.fuse_states(sdfg)
    sdutil.inline_sdfgs(sdfg)
    sdutil.fuse_states(sdfg)
    Pipeline([ArrayElimination()]).apply_pass(sdfg, {})
    assert len(sdfg.arrays) == 4


def test_merge_simple():
    sdfg = dace.SDFG('tester')
    sdfg.add_array('A', [20], dace.float64)
    sdfg.add_array('B', [20], dace.float64)

    state = sdfg.add_state()
    a1 = state.add_read('A')
    a2 = state.add_read('A')
    b1 = state.add_write('B')
    b2 = state.add_write('B')
    t1 = state.add_tasklet('doit1', {'a'}, {'b'}, 'b = a')
    t2 = state.add_tasklet('doit2', {'a'}, {'b'}, 'b = a')
    state.add_edge(a1, None, t1, 'a', dace.Memlet('A[0]'))
    state.add_edge(a2, None, t2, 'a', dace.Memlet('A[1]'))
    state.add_edge(t1, 'b', b1, None, dace.Memlet('B[0]'))
    state.add_edge(t2, 'b', b2, None, dace.Memlet('B[1]'))

    Pipeline([ArrayElimination()]).apply_pass(sdfg, {})
    assert len(state.data_nodes()) == 2

def test_nested_edge_view():
    """
    Tests if the ArrayElimination pass works correctly with nested eges using views.
    """
    sdfg = dace.SDFG("tester")
    sdfg.add_array("A", [64], dace.float32)
    sdfg.add_view("A_view", [1], dace.float32)

    loop = LoopRegion("loop1", "i < 64", "i", "i = 0", "i = i + 1")
    sdfg.add_node(loop)
    s = loop.add_state(is_start_block=True)
    access = s.add_access("A")
    access_view = s.add_access("A_view")
    access_view.add_in_connector("views")
    s.add_edge(access, None, access_view, "views", dace.Memlet("A[0]"))
    s2 = loop.add_state()
    loop.add_edge(s, s2, dace.InterstateEdge(assignments={"v": "A_view[0]"}))

    s = sdfg.add_state()
    access = s.add_access("A")
    access_view = s.add_access("A_view")
    access_view.add_in_connector("views")
    s.add_edge(access, None, access_view, "views", dace.Memlet("A[0]"))
    sdfg.add_edge(loop, s, dace.InterstateEdge())

    sdfg.validate()
    prev_arrays = list(sdfg.arrays.keys())

    # Apply ArrayElimination
    try:
        res1 = ap.StateReachability().apply_pass(sdfg, {})
        res2 = ap.FindAccessStates().apply_pass(sdfg, {})
        # combine both dicts
        pipe_res = {
            ap.StateReachability.__name__: res1,
            ap.FindAccessStates.__name__: res2,
        }
        ArrayElimination().apply_pass(sdfg, pipe_res)
    except Exception as e:
        assert False, f"ArrayElimination failed: {e}"

    # Should not remove anything
    assert len(list(sdfg.arrays.keys())) == len(prev_arrays)


def test_view():
    """
    Tests if the ArrayElimination pass works correctly with views.
    """
    sdfg = dace.SDFG("tester")
    sdfg.add_array("A", [64], dace.float32)
    sdfg.add_view("A_view", [1], dace.float32)

    s = sdfg.add_state()
    access = s.add_access("A")
    access_view = s.add_access("A_view")
    access_view.add_in_connector("views")
    s.add_edge(access, None, access_view, "views", dace.Memlet("A[0]"))

    sdfg.validate()

    # Apply ArrayElimination
    try:
        res1 = ap.StateReachability().apply_pass(sdfg, {})
        res2 = ap.FindAccessStates().apply_pass(sdfg, {})
        # combine both dicts
        pipe_res = {
            ap.StateReachability.__name__: res1,
            ap.FindAccessStates.__name__: res2,
        }
        ArrayElimination().apply_pass(sdfg, pipe_res)
        sdfg.validate()
    except Exception as e:
        assert False, f"ArrayElimination failed: {e}"

    # Should remove everything
    assert len(list(sdfg.all_nodes_recursive())) == 1
    assert isinstance(sdfg.nodes()[0], dace.sdfg.SDFGState)

def test_loop_header():
    """
    Tests if the ArrayElimination pass considers uses in loop headers.
    """
    sdfg = dace.SDFG("tester")
    sdfg.add_array("A", [64], dace.float32)
    sdfg.add_view("A_view", [1], dace.float32)

    s = sdfg.add_state()
    access = s.add_access("A")
    access_view = s.add_access("A_view")
    access_view.add_in_connector("views")
    s.add_edge(access, None, access_view, "views", dace.Memlet("A[0]"))

    loop = LoopRegion("loop1", "i < A_view", "i", "i = 0", "i = i + 1")
    sdfg.add_node(loop)
    loop.add_state(is_start_block=True)
    sdfg.add_edge(s, loop, dace.InterstateEdge())

    sdfg.validate()
    prev_arrays = list(sdfg.all_nodes_recursive())

    # Apply ArrayElimination
    try:
        res1 = ap.StateReachability().apply_pass(sdfg, {})
        res2 = ap.FindAccessStates().apply_pass(sdfg, {})
        # combine both dicts
        pipe_res = {
            ap.StateReachability.__name__: res1,
            ap.FindAccessStates.__name__: res2,
        }
        ArrayElimination().apply_pass(sdfg, pipe_res)
    except Exception as e:
        assert False, f"ArrayElimination failed: {e}"

    # Should not remove anything
    assert len(list(sdfg.all_nodes_recursive())) == len(prev_arrays)

if __name__ == '__main__':
    test_redundant_simple()
    test_merge_simple()
    test_nested_edge_view()
    test_view()
    test_loop_header()
