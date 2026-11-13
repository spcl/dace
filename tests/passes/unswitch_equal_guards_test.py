# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Equal invariant guards around unguarded work unswitch together out of a loop (``MoveLoopInvariantIfUp``) or a
map (``MoveMapInvariantIfUp``): ``if c: A; B; if c: C`` becomes ``if c: {A; B; C} else: {B}``."""

import numpy as np

import dace
from dace.sdfg import nodes
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion, LoopRegion
from dace.transformation.interstate.move_loop_invariant_if_up import MoveLoopInvariantIfUp
from dace.transformation.interstate.move_map_invariant_if_up import MoveMapInvariantIfUp
from dace.transformation.passes.canonicalize import canonicalize

N = dace.symbol("N")
C = dace.symbol("C")


@dace.program
def two_equal_guards_in_loop(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N]):
    for i in range(N):
        if C > 0:
            b[i] = a[i] + 1.0
        c[i] = a[i] * 2.0
        if 0 < C:
            b[i] = b[i] + c[i]


@dace.program
def three_equal_guards_in_loop(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N]):
    for i in range(N):
        if C > 0:
            b[i] = a[i] + 1.0
        c[i] = a[i] * 2.0
        if C > 0:
            b[i] = b[i] * c[i]
        c[i] = c[i] + 3.0
        if 0 < C:
            b[i] = b[i] - c[i]


@dace.program
def unequal_guards_in_loop(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N]):
    for i in range(N):
        if C > 0:
            b[i] = a[i] + 1.0
        c[i] = a[i] * 2.0
        if C > 1:
            b[i] = b[i] + c[i]


@dace.program
def guard_data_written_in_loop(a: dace.float64[N], b: dace.float64[N], g: dace.float64[1]):
    for i in range(N):
        if g[0] > 0.0:
            b[i] = a[i] + 1.0
        g[0] = a[i]
        if g[0] > 0.0:
            b[i] = b[i] * 2.0


@dace.program
def guard_data_read_only_in_loop(a: dace.float64[N], b: dace.float64[N], g: dace.float64[1], h: dace.float64[1]):
    for i in range(N):
        if g[0] > 0.0:
            b[i] = a[i] + 1.0
        h[0] = a[i]
        if g[0] > 0.0:
            b[i] = b[i] * 2.0


@dace.program
def equal_guards_in_map(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N]):
    for i in dace.map[0:N]:
        if C > 0:
            b[i] = a[i] + 1.0
        c[i] = a[i] * 2.0
        if 0 < C:
            b[i] = b[i] + c[i]


@dace.program
def unequal_guards_in_map(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N]):
    for i in dace.map[0:N]:
        if C > 0:
            b[i] = a[i] + 1.0
        c[i] = a[i] * 2.0
        if C > 1:
            b[i] = b[i] + c[i]


def outputs(sdfg: dace.SDFG, c: int, **extra) -> list[np.ndarray]:
    arrays = {"a": np.linspace(-2.0, 2.0, 9), "b": np.full(9, 7.0)}
    arrays.update({k: v.copy() for k, v in extra.items()})
    if "c" in sdfg.arrays:
        arrays["c"] = np.zeros(9)
    sdfg(**arrays, N=9, C=c)
    return [v for k, v in sorted(arrays.items()) if k != "a"]


def assert_same_outputs(sdfg: dace.SDFG, program, values=(1, 0), **extra) -> None:
    reference = program.to_sdfg(simplify=True)
    reference.name = sdfg.name + "_reference"
    for c in values:
        for got, want in zip(outputs(sdfg, c, **extra), outputs(reference, c, **extra)):
            assert np.array_equal(got, want)


def loops_per_arm(sdfg: dace.SDFG) -> list[int]:
    guard = next(b for b in sdfg.nodes() if isinstance(b, ConditionalBlock))
    return [sum(isinstance(r, LoopRegion) for r in arm.all_control_flow_regions()) for _, arm in guard.branches]


def test_two_equal_invariant_guards_unswitch_out_of_a_loop():
    sdfg = two_equal_guards_in_loop.to_sdfg(simplify=True)
    assert MoveLoopInvariantIfUp().apply_pass(sdfg, {}) == 1
    sdfg.validate()
    assert [type(b) for b in sdfg.nodes()] == [ConditionalBlock]
    assert loops_per_arm(sdfg) == [1, 1]
    assert_same_outputs(sdfg, two_equal_guards_in_loop)


def test_three_equal_invariant_guards_unswitch_out_of_a_loop():
    sdfg = three_equal_guards_in_loop.to_sdfg(simplify=True)
    assert MoveLoopInvariantIfUp().apply_pass(sdfg, {}) == 1
    sdfg.validate()
    assert loops_per_arm(sdfg) == [1, 1]
    guard = next(b for b in sdfg.nodes() if isinstance(b, ConditionalBlock))
    assert not any(isinstance(b, ConditionalBlock) for _, arm in guard.branches for b in arm.all_control_flow_blocks())
    assert_same_outputs(sdfg, three_equal_guards_in_loop)


def test_guards_not_provably_equal_stay_in_the_loop():
    sdfg = unequal_guards_in_loop.to_sdfg(simplify=True)
    assert MoveLoopInvariantIfUp().apply_pass(sdfg, {}) is None
    assert any(isinstance(b, LoopRegion) for b in sdfg.nodes())
    assert_same_outputs(sdfg, unequal_guards_in_loop, values=(2, 1, 0))


def test_a_guard_reading_data_the_loop_writes_stays_in_the_loop():
    sdfg = guard_data_written_in_loop.to_sdfg(simplify=True)
    assert MoveLoopInvariantIfUp().apply_pass(sdfg, {}) is None
    assert any(isinstance(b, LoopRegion) for b in sdfg.nodes())
    assert_same_outputs(sdfg, guard_data_written_in_loop, g=np.array([1.0]))


def test_guards_reading_the_same_unwritten_element_unswitch_out_of_a_loop():
    """Each guard reads ``g[0]`` through its own prep symbol; the conditions are compared with the prep expanded."""
    sdfg = guard_data_read_only_in_loop.to_sdfg(simplify=True)
    assert MoveLoopInvariantIfUp().apply_pass(sdfg, {}) == 1
    sdfg.validate()
    assert loops_per_arm(sdfg) == [1, 1]
    for g in (1.0, -1.0):
        assert_same_outputs(sdfg, guard_data_read_only_in_loop, values=(0,), g=np.array([g]), h=np.zeros(1))


def test_equal_invariant_guards_unswitch_out_of_a_map():
    sdfg = equal_guards_in_map.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True)
    assert MoveMapInvariantIfUp().apply_pass(sdfg, {}) == 1
    sdfg.validate()
    guard = next(b for b in sdfg.nodes() if isinstance(b, ConditionalBlock))
    maps = [sum(isinstance(n, nodes.MapEntry) for n, _ in arm.all_nodes_recursive()) for _, arm in guard.branches]
    assert maps == [1, 1]
    assert_same_outputs(sdfg, equal_guards_in_map)


def test_guards_not_provably_equal_stay_in_the_map():
    sdfg = unequal_guards_in_map.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True)
    assert MoveMapInvariantIfUp().apply_pass(sdfg, {}) is None
    assert_same_outputs(sdfg, unequal_guards_in_map, values=(2, 1, 0))


def test_work_reading_a_guards_in_place_value_keeps_the_guard_in_the_map():
    """The arm without the guard would have to route ``x`` around the removed nest; the guard stays in the map."""
    inner = dace.SDFG("in_place_guard_inner")
    inner.add_scalar("x", dace.float64)
    inner.add_symbol("c", dace.int64)
    branch = ControlFlowRegion("taken")
    state = branch.add_state(is_start_block=True)
    tasklet = state.add_tasklet("double", {"q": None}, {"o": None}, "o = q * 2")
    state.add_edge(state.add_read("x"), None, tasklet, "q", dace.Memlet("x[0]"))
    state.add_edge(tasklet, "o", state.add_write("x"), None, dace.Memlet("x[0]"))
    guard = ConditionalBlock("guard")
    guard.add_branch(dace.properties.CodeBlock("c > 0"), branch)
    inner.add_node(guard, is_start_block=True)

    sdfg = dace.SDFG("in_place_guard")
    sdfg.add_array("a", [4], dace.float64)
    sdfg.add_array("b", [4], dace.float64)
    sdfg.add_scalar("x", dace.float64, transient=True)
    sdfg.add_symbol("c", dace.int64)
    state = sdfg.add_state(is_start_block=True)
    entry, exit_node = state.add_map("m", {"i": "0:4"})
    first = state.add_tasklet("first", {"q": None}, {"o": None}, "o = q + 1")
    state.add_memlet_path(state.add_read("a"), entry, first, dst_conn="q", memlet=dace.Memlet("a[i]"))
    before, after = state.add_access("x"), state.add_access("x")
    state.add_edge(first, "o", before, None, dace.Memlet("x[0]"))
    nest = state.add_nested_sdfg(inner, {"x": None}, {"x": None}, symbol_mapping={"c": "c"})
    state.add_edge(before, None, nest, "x", dace.Memlet("x[0]"))
    state.add_edge(nest, "x", after, None, dace.Memlet("x[0]"))
    last = state.add_tasklet("last", {"q": None}, {"o": None}, "o = q + 100")
    state.add_edge(after, None, last, "q", dace.Memlet("x[0]"))
    state.add_memlet_path(last, exit_node, state.add_write("b"), src_conn="o", memlet=dace.Memlet("b[i]"))
    sdfg.validate()

    MoveMapInvariantIfUp().apply_pass(sdfg, {})

    sdfg.validate()
    assert state in sdfg.nodes()


def test_the_arm_without_the_guard_drops_the_access_node_it_wrote():
    """The guard writes ``b`` through an access node in the map body; the arm without it drops that node too."""
    inner = dace.SDFG("written_guard_inner")
    inner.add_array("y", [4], dace.float64)
    inner.add_symbol("i", dace.int64)
    inner.add_symbol("c", dace.int64)
    branch = ControlFlowRegion("taken")
    state = branch.add_state(is_start_block=True)
    tasklet = state.add_tasklet("inc", {"q": None}, {"o": None}, "o = q + 1")
    state.add_edge(state.add_read("y"), None, tasklet, "q", dace.Memlet("y[i]"))
    state.add_edge(tasklet, "o", state.add_write("y"), None, dace.Memlet("y[i]"))
    guard = ConditionalBlock("guard")
    guard.add_branch(dace.properties.CodeBlock("c > 0"), branch)
    inner.add_node(guard, is_start_block=True)

    sdfg = dace.SDFG("written_guard")
    for name in ("a", "b", "d"):
        sdfg.add_array(name, [4], dace.float64)
    sdfg.add_symbol("c", dace.int64)
    state = sdfg.add_state(is_start_block=True)
    entry, exit_node = state.add_map("m", {"i": "0:4"})
    work = state.add_tasklet("work", {"q": None}, {"o": None}, "o = q * 2")
    state.add_memlet_path(state.add_read("a"), entry, work, dst_conn="q", memlet=dace.Memlet("a[i]"))
    state.add_memlet_path(work, exit_node, state.add_write("d"), src_conn="o", memlet=dace.Memlet("d[i]"))
    nest = state.add_nested_sdfg(inner, {"y"}, {"y"}, symbol_mapping={"c": "c", "i": "i"})
    state.add_memlet_path(state.add_read("b"), entry, nest, dst_conn="y", memlet=dace.Memlet("b[0:4]"))
    written = state.add_access("b")
    state.add_edge(nest, "y", written, None, dace.Memlet("b[0:4]"))
    state.add_memlet_path(written, exit_node, state.add_write("b"), memlet=dace.Memlet("b[0:4]"))
    sdfg.validate()

    assert MoveMapInvariantIfUp().apply_pass(sdfg, {}) == 1

    sdfg.validate()
    for c, want in ((1, 6.0), (0, 5.0)):
        a, b, d = np.arange(4.0), np.full(4, 5.0), np.zeros(4)
        sdfg(a=a, b=b, d=d, c=c)
        assert np.array_equal(b, np.full(4, want)) and np.array_equal(d, 2 * a)


if __name__ == "__main__":
    test_two_equal_invariant_guards_unswitch_out_of_a_loop()
    test_three_equal_invariant_guards_unswitch_out_of_a_loop()
    test_guards_not_provably_equal_stay_in_the_loop()
    test_a_guard_reading_data_the_loop_writes_stays_in_the_loop()
    test_guards_reading_the_same_unwritten_element_unswitch_out_of_a_loop()
    test_equal_invariant_guards_unswitch_out_of_a_map()
    test_guards_not_provably_equal_stay_in_the_map()
    test_work_reading_a_guards_in_place_value_keeps_the_guard_in_the_map()
    test_the_arm_without_the_guard_drops_the_access_node_it_wrote()
