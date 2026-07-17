# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A2 kernel_per_state + A3 line_graph + the A6 invariant guard (GLOBAL_LAYOUT_DESIGN.md): the
fixture programs split into one kernel per state and stay bit-exact, the line graph comes back in
dependency order, and every v1 refusal (multi-kernel state, branch, LoopRegion) is loud."""
import numpy
import pytest

import dace
from dace.sdfg.state import LoopRegion
from dace.transformation.layout.line_graph import (check_kernel_per_state, kernel_per_state, line_graph)
from dace.transformation.layout.prepare import prepare_for_layout

from tests.transformations.layout import multinest_programs as fixtures
from tests.transformations.layout.multinest_fixtures_test import run_and_check

EXPECTED_ORDER = {"conflict2": ["B", "C"], "conflict3": ["B", "C", "D"], "agree2": ["B", "C"]}


def prepared(program_name):
    program, _, _ = fixtures.PROGRAMS[program_name]
    sdfg = program.to_sdfg(simplify=True)
    prepare_for_layout(sdfg)
    return sdfg


def kernel_output(kernel):
    """The single non-transient array a fixture kernel writes."""
    sdfg = kernel.state.sdfg
    written = {
        n.data
        for n in kernel.state.data_nodes() if kernel.state.in_degree(n) > 0 and not sdfg.arrays[n.data].transient
    }
    assert len(written) == 1, written
    return written.pop()


@pytest.mark.parametrize("program_name", sorted(fixtures.PROGRAMS))
def test_kernel_per_state_splits_and_stays_bitexact(program_name):
    sdfg = prepared(program_name)
    kernel_per_state(sdfg)
    check_kernel_per_state(sdfg)  # must not raise
    kernels = line_graph(sdfg)
    assert [kernel_output(k) for k in kernels] == EXPECTED_ORDER[program_name]
    assert [k.index for k in kernels] == list(range(len(kernels)))
    run_and_check(sdfg, program_name, seed=2)


def test_line_graph_refuses_multi_kernel_state():
    """conflict2 keeps both nests in one fused state until A2 runs -- the A6 guard must fire."""
    sdfg = prepared("conflict2")
    with pytest.raises(RuntimeError, match="kernel-per-state invariant"):
        line_graph(sdfg)


def test_line_graph_refuses_branch():
    sdfg = dace.SDFG("branchy")
    first = sdfg.add_state("first", is_start_block=True)
    left = sdfg.add_state("left")
    right = sdfg.add_state("right")
    join = sdfg.add_state("join")
    sdfg.add_edge(first, left, dace.InterstateEdge(condition="1 > 0"))
    sdfg.add_edge(first, right, dace.InterstateEdge(condition="0 > 1"))
    sdfg.add_edge(left, join, dace.InterstateEdge())
    sdfg.add_edge(right, join, dace.InterstateEdge())
    with pytest.raises(NotImplementedError, match="branches"):
        line_graph(sdfg)


def test_line_graph_refuses_loop_region():
    sdfg = dace.SDFG("loopy")
    sdfg.add_state("pre", is_start_block=True)
    loop = LoopRegion("loop", condition_expr="i < 10", loop_var="i", initialize_expr="i = 0", update_expr="i = i + 1")
    sdfg.add_node(loop)
    sdfg.add_edge(sdfg.start_block, loop, dace.InterstateEdge())
    with pytest.raises(NotImplementedError, match="LoopRegion"):
        line_graph(sdfg)


def test_line_graph_refuses_nonmap_work():
    sdfg = dace.SDFG("copyonly")
    sdfg.add_array("X", [8], dace.float64)
    state = sdfg.add_state("s", is_start_block=True)
    t = sdfg.add_state("t")
    sdfg.add_edge(state, t, dace.InterstateEdge())
    tasklet = t.add_tasklet("lone", {}, {"o"}, "o = 1.0")
    t.add_edge(tasklet, "o", t.add_write("X"), None, dace.Memlet("X[0]"))
    with pytest.raises(NotImplementedError, match="non-map work"):
        line_graph(sdfg)


def test_line_graph_refuses_bare_copy_state():
    """A state of two access nodes joined by a copy memlet MOVES DATA: it must be refused as
    non-map work, not slip through as structural -- apply_assignment's liveness planning never
    sees pass-through states, so a silent pass-through would corrupt its read/write analysis."""
    sdfg = dace.SDFG("an_copy")
    sdfg.add_array("B", [8, 8], dace.float64)
    sdfg.add_array("C", [8, 8], dace.float64)
    state = sdfg.add_state("copy", is_start_block=True)
    state.add_nedge(state.add_read("B"), state.add_write("C"),
                    dace.Memlet(data="B", subset="0:8, 0:8", other_subset="0:8, 0:8"))
    sdfg.validate()
    with pytest.raises(NotImplementedError, match="non-map work"):
        line_graph(sdfg)


def test_kernel_per_state_refuses_undraggable_shared_sink():
    """Two nests writing the SAME sink access node: state_fission pulls the second nest along into
    the state it creates, so no split can succeed -- kernel_per_state must raise instead of
    reporting a split that did not happen (and leaving a misleading A6 failure for later)."""
    M = dace.symbol("M")
    sdfg = dace.SDFG("shared_sink")
    sdfg.add_array("A", [M], dace.float64)
    sdfg.add_array("B", [M], dace.float64)
    sdfg.add_array("C", [M, 2], dace.float64)
    state = sdfg.add_state("s", is_start_block=True)
    c_node = state.add_access("C")
    for col, src in ((0, "A"), (1, "B")):
        me, mx = state.add_map(f"m{col}", {"j": "0:M"})
        tasklet = state.add_tasklet(f"t{col}", {"a"}, {"b"}, "b = a * 2.0")
        state.add_memlet_path(state.add_access(src), me, tasklet, dst_conn="a", memlet=dace.Memlet(f"{src}[j]"))
        state.add_memlet_path(tasklet, mx, c_node, src_conn="b", memlet=dace.Memlet(f"C[j, {col}]"))
    sdfg.validate()
    with pytest.raises(RuntimeError, match="no progress"):
        kernel_per_state(sdfg)


if __name__ == "__main__":
    for name in sorted(fixtures.PROGRAMS):
        test_kernel_per_state_splits_and_stays_bitexact(name)
    test_line_graph_refuses_multi_kernel_state()
    test_line_graph_refuses_branch()
    test_line_graph_refuses_loop_region()
    test_line_graph_refuses_nonmap_work()
    test_line_graph_refuses_bare_copy_state()
    test_kernel_per_state_refuses_undraggable_shared_sink()
    print("line_graph tests PASS")
