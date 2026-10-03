# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Loop-aware global layout assignment: a LoopRegion whose body is a flat line is admitted by line_graph,
its kernels are pinned to one layout by the body-uniform model, and apply_assignment hoists the entry/exit
relayout OUTSIDE the region (a conversion before the body state would re-run every iteration). Runtime is
checked bit-exact across several iterations against a NumPy oracle, and a non-uniform loop assignment (which
would feed the wrong layout across the back-edge) is refused loudly.
"""
import numpy
import pytest

import dace
from dace.sdfg.state import LoopRegion
from dace.libraries.layout.algebra import Permute
from dace.transformation.layout.apply_assignment import IDENTITY_LAYOUT, Layout, apply_assignment
from dace.transformation.layout.line_graph import line_graph, locked_transitions, loop_spans

N = dace.symbol("N")
T = dace.symbol("T")
CM = Layout("perm10", (Permute((1, 0)), ))


def elementwise_body(loop, label, expr, start=False):
    """A 2D map ``X[i,j] = expr(X[i,j])`` state inside ``loop`` -- reads then writes X (so an entry conversion
    is needed) with a permutable rank-2 access."""
    state = loop.add_state(label, is_start_block=start)
    me, mx = state.add_map(f"{label}_m", {"i": "0:N", "j": "0:N"})
    tasklet = state.add_tasklet("t", {"a"}, {"b"}, f"b = {expr}")
    state.add_memlet_path(state.add_read("X"), me, tasklet, dst_conn="a", memlet=dace.Memlet("X[i, j]"))
    state.add_memlet_path(tasklet, mx, state.add_write("X"), src_conn="b", memlet=dace.Memlet("X[i, j]"))
    return state


def loop_sdfg(bodies):
    """An SDFG that is a single ``for k in 0:T`` loop over the given body expressions on X[N,N]."""
    sdfg = dace.SDFG("looprelayout")
    sdfg.add_array("X", [N, N], dace.float64)
    sdfg.add_symbol("T", dace.int64)
    loop = LoopRegion("loop", condition_expr="k < T", loop_var="k", initialize_expr="k = 0", update_expr="k = k + 1")
    sdfg.add_node(loop, is_start_block=True)
    prev = None
    for idx, expr in enumerate(bodies):
        state = elementwise_body(loop, f"body{idx}", expr, start=(idx == 0))
        if prev is not None:
            loop.add_edge(prev, state, dace.InterstateEdge())
        prev = state
    sdfg.validate()
    return sdfg, loop


def relayout_states_inside(loop):
    """Labels of states inside the loop region whose name marks an inserted relayout boundary."""
    return [s.label for s in loop.all_states() if s.label.startswith("relayout_")]


def test_loop_relayout_bit_exact():
    """One body kernel, permuted uniformly: the entry conversion is hoisted before the loop, the body runs on
    the transposed clone across all iterations, and the exit restores X -- bit-exact with the untouched run."""
    _N, _Tv = 4, 3
    X0 = numpy.random.rand(_N, _N)
    ref = X0 * (2.0**_Tv)  # X <- X*2, T times

    sdfg, loop = loop_sdfg(["a * 2.0"])
    kernels = line_graph(sdfg)
    assert loop_spans(kernels) == [(0, 1)] and locked_transitions(kernels) == set()

    applied = apply_assignment(sdfg, kernels, {"X": [CM] * len(kernels)})
    sdfg.validate()
    # The relayout landed OUTSIDE the loop (before the region), never inside it.
    assert [s.label for s in applied.boundary_states] == ["relayout_before_loop"]
    assert relayout_states_inside(loop) == []

    X = X0.copy()
    sdfg(X=X, N=_N, T=_Tv)
    assert numpy.allclose(X, ref)


def test_two_body_kernels_uniform_bit_exact():
    """Two body kernels sharing one layout: no conversion is inserted between them (body-uniform), the single
    entry/exit conversions sit outside the loop, and f(x)=2x+1 iterated T times comes back bit-exact."""
    _N, _Tv = 5, 4
    X0 = numpy.random.rand(_N, _N)
    ref = X0.copy()
    for _ in range(_Tv):
        ref = ref * 2.0 + 1.0  # body0 then body1, per iteration

    sdfg, loop = loop_sdfg(["a * 2.0", "a + 1.0"])
    kernels = line_graph(sdfg)
    assert loop_spans(kernels) == [(0, 2)] and locked_transitions(kernels) == {1}

    apply_assignment(sdfg, kernels, {"X": [CM, CM]})
    sdfg.validate()
    assert relayout_states_inside(loop) == []  # one layout across the body -> no inner boundary

    X = X0.copy()
    sdfg(X=X, N=_N, T=_Tv)
    assert numpy.allclose(X, ref)


def test_nonuniform_loop_assignment_refused():
    """An assignment that changes X's layout between two kernels of the same loop body is refused: the back-edge
    would feed the wrong layout into the next iteration (a silent miscompile)."""
    sdfg, _ = loop_sdfg(["a * 2.0", "a + 1.0"])
    kernels = line_graph(sdfg)
    with pytest.raises(NotImplementedError, match="body-uniform"):
        apply_assignment(sdfg, kernels, {"X": [IDENTITY_LAYOUT, CM]})


if __name__ == "__main__":
    test_loop_relayout_bit_exact()
    test_two_body_kernels_uniform_bit_exact()
    test_nonuniform_loop_assignment_refused()
    print("apply_assignment loop tests PASS")
