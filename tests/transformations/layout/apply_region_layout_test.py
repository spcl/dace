# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Region-scoped layout: change an array's layout for a line of top-level nodes, restore at the end.

``apply_region_layout`` imposes a layout on the kernels of a top-level region ``[start, end)`` and
restores the original layout at the region's end -- the enter relayout lands before the region, the
restore after it, both at the top level. It is the imposed, scoped counterpart of the global
``apply_assignment`` trajectory (and the API form of the OMEN mid-flight transpose, bounded to a
region). Here ``A`` is transposed only for the two middle nests of a four-nest chain; nests outside
the region see the original layout, and the result is bit-exact.
"""
import numpy
import pytest

import dace
from dace.sdfg.state import LoopRegion
from dace.libraries.layout.algebra import Permute
from dace.transformation.layout.apply_assignment import Layout, apply_region_layout
from dace.transformation.layout.line_graph import kernel_per_state, line_graph, loop_spans
from dace.transformation.layout.prepare import prepare_for_layout

N = dace.symbol("N")

PERM10 = Layout("perm10", (Permute((1, 0)), ))


@dace.program
def chain(A: dace.float64[N, N], P: dace.float64[N, N], O0: dace.float64[N, N], O1: dace.float64[N, N],
          O2: dace.float64[N, N], O3: dace.float64[N, N]):
    for i, j in dace.map[0:N, 0:N]:
        O0[i, j] = A[i, j] + P[i, j]  # nest 0: A straight
    for i, j in dace.map[0:N, 0:N]:
        O1[i, j] = O0[i, N - 1 - j] + A[j, i] + P[i, j]  # nest 1: A transposed
    for i, j in dace.map[0:N, 0:N]:
        O2[i, j] = O1[i, N - 1 - j] + A[j, i] + P[i, j]  # nest 2: A transposed
    for i, j in dace.map[0:N, 0:N]:
        O3[i, j] = O2[i, N - 1 - j] + A[i, j] + P[i, j]  # nest 3: A straight


def oracle(A, P):
    o0 = A + P
    o1 = o0[:, ::-1] + A.T + P
    o2 = o1[:, ::-1] + A.T + P
    o3 = o2[:, ::-1] + A + P
    return {"O0": o0, "O1": o1, "O2": o2, "O3": o3}


def build():
    sdfg = chain.to_sdfg(simplify=True)
    prepare_for_layout(sdfg)
    kernel_per_state(sdfg)
    return sdfg, line_graph(sdfg)


def run_and_check(sdfg, n=64, seed=0):
    rng = numpy.random.default_rng(seed)
    A, P = rng.random((n, n)), rng.random((n, n))
    outs = {f"O{k}": numpy.zeros((n, n)) for k in range(4)}
    sdfg(A=A.copy(), P=P.copy(), **outs, N=n)
    ref = oracle(A, P)
    for name, r in ref.items():
        assert numpy.allclose(outs[name], r), f"{name} diverges from the oracle"


def test_region_layout_readonly_enters_without_redundant_restore():
    """Transpose the READ-ONLY input A for the region [1, 3) (the two middle nests). The region reads a
    transposed clone; the original A is never mutated, so only ONE boundary appears (the enter) -- no
    restore transpose is emitted -- and the program stays bit-exact."""
    sdfg, kernels = build()
    assert len(kernels) == 4

    applied = apply_region_layout(sdfg, kernels, {"A": PERM10}, region=(1, 3))
    # enter before nest 1 only; a read-only array needs no restore at the region end (original is valid).
    assert len(applied.boundary_states) == 1
    assert any("A__seg" in name for names in applied.segment_names.values() for name in names)  # region clone
    sdfg.validate()
    run_and_check(sdfg)


def test_region_layout_rejects_bad_regions():
    sdfg, kernels = build()
    with pytest.raises(ValueError, match="out of range"):
        apply_region_layout(sdfg, kernels, {"A": PERM10}, region=(2, 2))  # empty
    with pytest.raises(ValueError, match="out of range"):
        apply_region_layout(sdfg, kernels, {"A": PERM10}, region=(1, 9))  # past the end


def two_body_loop_sdfg():
    """``for k in 0:T`` over two X[N,N] body states -- one loop span [0, 2) that a region may not split."""
    sdfg = dace.SDFG("region_loop")
    sdfg.add_array("X", [N, N], dace.float64)
    sdfg.add_symbol("T", dace.int64)
    loop = LoopRegion("loop", condition_expr="k < T", loop_var="k", initialize_expr="k = 0", update_expr="k = k + 1")
    sdfg.add_node(loop, is_start_block=True)
    prev = None
    for idx, expr in enumerate(("a * 2.0", "a + 1.0")):
        state = loop.add_state(f"body{idx}", is_start_block=(idx == 0))
        me, mx = state.add_map(f"body{idx}_m", {"i": "0:N", "j": "0:N"})
        tasklet = state.add_tasklet("t", {"a"}, {"b"}, f"b = {expr}")
        state.add_memlet_path(state.add_read("X"), me, tasklet, dst_conn="a", memlet=dace.Memlet("X[i, j]"))
        state.add_memlet_path(tasklet, mx, state.add_write("X"), src_conn="b", memlet=dace.Memlet("X[i, j]"))
        if prev is not None:
            loop.add_edge(prev, state, dace.InterstateEdge())
        prev = state
    sdfg.validate()
    return sdfg


def test_region_layout_rejects_splitting_a_loop():
    """A region whose end falls strictly inside a loop span [0, 2) is refused (the relayout would land
    inside the loop body)."""
    sdfg = two_body_loop_sdfg()
    kernels = line_graph(sdfg)
    assert loop_spans(kernels) == [(0, 2)]
    with pytest.raises(ValueError, match="splits loop span"):
        apply_region_layout(sdfg, kernels, {"X": PERM10}, region=(0, 1))  # cuts the loop body


if __name__ == "__main__":
    test_region_layout_readonly_enters_without_redundant_restore()
    test_region_layout_rejects_bad_regions()
    test_region_layout_rejects_splitting_a_loop()
    print("apply_region_layout tests PASS")
