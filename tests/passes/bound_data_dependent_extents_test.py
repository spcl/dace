# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A transient sized by a symbol read from data takes the extent that bounds it."""

import numpy as np

import dace
from dace.properties import CodeBlock
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion, LoopRegion
from dace.sdfg.utils import has_literal_capacity
from dace.transformation.passes.bound_data_dependent_extents import BoundDataDependentExtents, set_capacities


def table_copy(table_extent: int | None) -> dace.SDFG:
    """``n = sizes[0]``, then ``vals[0:n]`` of a transient ``vals[n]`` copied into a transient ``table[table_extent]``;
    a ``table`` of symbolic extent ``m`` when ``table_extent`` is ``None``."""
    sdfg = dace.SDFG(f"table_copy_{table_extent}")
    sdfg.add_symbol("n", dace.int64)
    sdfg.add_symbol("m", dace.int64)
    sdfg.add_array("sizes", [1], dace.int64)
    sdfg.add_array("A", ["m"], dace.float64)
    sdfg.add_transient("vals", ["n"], dace.float64)
    sdfg.add_transient("table", [table_extent or "m"], dace.float64)
    first = sdfg.add_state("first", is_start_block=True)
    second = sdfg.add_state_after(first, "second", assignments={"n": "sizes[0]"})
    second.add_edge(second.add_read("A"), None, second.add_write("vals"), None, dace.Memlet("A[0:n]"))
    third = sdfg.add_state_after(second, "third")
    third.add_edge(third.add_read("vals"), None, third.add_write("table"), None, dace.Memlet("vals[0:n] -> [0:n]"))
    return sdfg


def test_an_in_bounds_access_of_a_literal_array_gives_a_capacity_and_keeps_the_shape():
    """``table[0:n]`` of a five-element ``table`` shows ``n <= 5``: ``vals`` holds five, its shape stays ``n``."""
    sdfg = table_copy(5)
    assert set_capacities(sdfg, {"n"}) == 1
    assert sdfg.arrays["vals"].total_size == 5
    assert str(sdfg.arrays["vals"].shape[0]) == "n"
    assert has_literal_capacity(sdfg.arrays["vals"])


def test_an_access_of_a_symbolic_array_gives_no_capacity():
    sdfg = table_copy(None)
    assert set_capacities(sdfg, {"n"}) == 0
    assert not has_literal_capacity(sdfg.arrays["vals"])


def per_row_window(sign: int) -> dace.SDFG:
    """``lo = 0 if A[p] > 0 else 1`` per row ``p``, then ``out[p] = sum(win)`` over ``win[N - lo]``
    (``N + lo`` for ``sign = 1``), the window ``A[lo:N]`` filled by a map."""
    sdfg = dace.SDFG("per_row_window_" + ("up" if sign > 0 else "down"))
    sdfg.add_symbol("N", dace.int64)
    sdfg.add_array("A", ["N + 1"], dace.float64)
    sdfg.add_array("out", ["N"], dace.float64)
    sdfg.add_array("low", ["N"], dace.int64, transient=True)
    sdfg.add_symbol("lo", dace.int64)
    sdfg.add_array("win", [f"N {'+' if sign > 0 else '-'} lo"], dace.float64, transient=True)
    first = sdfg.add_state("first", is_start_block=True)
    first.add_mapped_tasklet(
        "low",
        {"i": "0:N"},
        {"a": dace.Memlet("A[i]")},
        "o = 0 if a > 0 else 1",
        {"o": dace.Memlet("low[i]")},
        external_edges=True,
    )
    loop = LoopRegion("rows", "p < N", "p", "p = 0", "p = p + 1")
    sdfg.add_node(loop)
    sdfg.add_edge(first, loop, dace.InterstateEdge())
    head = loop.add_state("head", is_start_block=True)
    fill = loop.add_state("fill")
    loop.add_edge(head, fill, dace.InterstateEdge(assignments={"lo": "low[p]"}))
    fill.add_mapped_tasklet(
        "window",
        {"j": "0:N - lo"},
        {"a": dace.Memlet("A[j + lo]")},
        "o = a",
        {"o": dace.Memlet("win[j]")},
        external_edges=True,
    )
    total = loop.add_state_after(fill, "total")
    tasklet = total.add_tasklet("zero", {}, {"o"}, "o = 0")
    total.add_edge(tasklet, "o", total.add_write("out"), None, dace.Memlet("out[p]"))
    acc = loop.add_state_after(total, "acc")
    acc.add_mapped_tasklet(
        "sum",
        {"j": "0:N - lo"},
        {"w": dace.Memlet("win[j]")},
        "o = w",
        {"o": dace.Memlet("out[p]", wcr="lambda x, y: x + y")},
        external_edges=True,
    )
    return sdfg


def test_a_nonnegative_data_symbol_subtracted_from_an_extent_bounds_it_by_zero():
    sdfg = per_row_window(-1)
    A = np.arange(9, dtype=np.float64) - 3.0
    expected = np.zeros(8)
    sdfg(A=A.copy(), out=expected, N=8)

    assert BoundDataDependentExtents().apply_pass(sdfg, {}) == 1
    assert [str(extent) for extent in sdfg.arrays["win"].shape] == ["N"]
    got = np.zeros(8)
    sdfg(A=A.copy(), out=got, N=8)
    assert np.allclose(got, expected)


def test_a_nest_the_bound_names_a_new_symbol_for_is_given_it():
    """The nest's ``win[W]``, ``W = N - lo``, takes ``N``, which it was never given."""
    sdfg = per_row_window(-1)
    body = dace.SDFG("nested_fill")
    body.add_array("win", ["W"], dace.float64)
    inner = body.add_state("write", is_start_block=True)
    tasklet = inner.add_tasklet("one", {}, {"o"}, "o = 1.0")
    inner.add_edge(tasklet, "o", inner.add_write("win"), None, dace.Memlet("win[0]"))
    last = next(state for state in sdfg.states() if state.label == "acc")
    state = last.parent_graph.add_state_after(last, "nested")
    nest = state.add_nested_sdfg(body, set(), {"win"}, symbol_mapping={"W": "N - lo"})
    state.add_edge(nest, "win", state.add_write("win"), None, dace.Memlet("win[0]"))

    assert BoundDataDependentExtents().apply_pass(sdfg, {}) == 1
    assert [str(extent) for extent in body.arrays["win"].shape] == ["N"]
    assert str(nest.symbol_mapping["N"]) == "N"
    sdfg.validate()


def test_an_extent_growing_with_the_data_symbol_is_left_alone():
    sdfg = per_row_window(1)
    assert BoundDataDependentExtents().apply_pass(sdfg, {}) is None
    assert [str(extent) for extent in sdfg.arrays["win"].shape] == ["N + lo"]


def test_an_in_bounds_access_of_an_argument_bounds_the_extent_by_the_argument_extent():
    """``A[0:n]`` of an ``A[m]`` argument shows ``n <= m``: ``vals[n]`` takes ``m``, one size for the whole call."""
    sdfg = table_copy(None)
    sdfg.name = "table_copy_argument_bound"
    assert BoundDataDependentExtents().apply_pass(sdfg, {}) == 1
    assert [str(extent) for extent in sdfg.arrays["vals"].shape] == ["m"]
    sdfg.validate()


def test_an_extent_scaled_by_an_argument_extent_takes_the_bound():
    """``vals[n * k]``, ``k`` an extent of the argument ``B``: ``A[0:n]`` of ``A[m]`` gives ``vals[m * k]``."""
    sdfg = dace.SDFG("scaled_table_copy")
    sdfg.add_symbol("n", dace.int64)
    sdfg.add_symbol("m", dace.int64)
    sdfg.add_symbol("k", dace.int64)
    sdfg.add_array("sizes", [1], dace.int64)
    sdfg.add_array("A", ["m"], dace.float64)
    sdfg.add_array("B", ["k"], dace.float64)
    sdfg.add_transient("vals", ["n * k"], dace.float64)
    first = sdfg.add_state("first", is_start_block=True)
    second = sdfg.add_state_after(first, "second", assignments={"n": "sizes[0]"})
    second.add_edge(second.add_read("A"), None, second.add_write("vals"), None, dace.Memlet("A[0:n] -> [0:n]"))
    assert BoundDataDependentExtents().apply_pass(sdfg, {}) == 1
    assert [str(extent) for extent in sdfg.arrays["vals"].shape] == ["k*m"]


def test_an_access_on_one_branch_only_gives_no_bound():
    """``A[0:n]`` read only when ``n > 2``: the other path never shows ``n <= m``, so ``vals[n]`` keeps ``n``."""
    sdfg = dace.SDFG("guarded_table_copy")
    sdfg.add_symbol("n", dace.int64)
    sdfg.add_symbol("m", dace.int64)
    sdfg.add_array("sizes", [1], dace.int64)
    sdfg.add_array("A", ["m"], dace.float64)
    sdfg.add_transient("vals", ["n"], dace.float64)
    first = sdfg.add_state("first", is_start_block=True)
    second = sdfg.add_state_after(first, "second", assignments={"n": "sizes[0]"})
    copy = sdfg.add_state("copy")
    done = sdfg.add_state("done")
    sdfg.add_edge(second, copy, dace.InterstateEdge("n > 2"))
    sdfg.add_edge(second, done, dace.InterstateEdge("n <= 2"))
    sdfg.add_edge(copy, done, dace.InterstateEdge())
    copy.add_edge(copy.add_read("A"), None, copy.add_write("vals"), None, dace.Memlet("A[0:n]"))
    assert BoundDataDependentExtents().apply_pass(sdfg, {}) is None
    assert [str(extent) for extent in sdfg.arrays["vals"].shape] == ["n"]


def trip_guarded_copy(name: str, trips: str) -> dace.SDFG:
    """``n = sizes[0]``, ``c = sizes[1]``, then a loop of ``trips`` iterations copying ``A[0:n]`` into ``vals[n]``;
    a second transient ``other[c]`` is sized by ``c``."""
    sdfg = dace.SDFG(name)
    for symbol in ("n", "c", "m"):
        sdfg.add_symbol(symbol, dace.int64)
    sdfg.add_array("sizes", [2], dace.int64)
    sdfg.add_array("A", ["m"], dace.float64)
    sdfg.add_transient("vals", ["n"], dace.float64)
    sdfg.add_transient("other", ["c"], dace.float64)
    first = sdfg.add_state("first", is_start_block=True)
    second = sdfg.add_state_after(first, "second", assignments={"n": "sizes[0]", "c": "sizes[1]"})
    loop = LoopRegion("trips", f"j < {trips}", "j", "j = 0", "j = j + 1")
    sdfg.add_node(loop)
    sdfg.add_edge(second, loop, dace.InterstateEdge())
    body = loop.add_state("copy", is_start_block=True)
    body.add_edge(body.add_read("A"), None, body.add_write("vals"), None, dace.Memlet("A[0:n]"))
    sdfg.reset_cfg_list()
    return sdfg


def test_a_loop_over_the_transients_own_extent_bounds_it():
    """``n = 0`` leaves ``vals`` empty, and otherwise the loop runs the copy, which shows ``n <= m``."""
    sdfg = trip_guarded_copy("trip_guarded_own_extent", "n")
    assert BoundDataDependentExtents().apply_pass(sdfg, {}) == 1
    assert [str(extent) for extent in sdfg.arrays["vals"].shape] == ["m"]


def test_a_loop_over_another_transients_extent_bounds_nothing():
    """With ``c = 0`` the copy never runs, so ``n`` may exceed ``m`` while ``vals`` still holds ``n``."""
    sdfg = trip_guarded_copy("trip_guarded_other_extent", "c")
    assert BoundDataDependentExtents().apply_pass(sdfg, {}) is None
    assert [str(extent) for extent in sdfg.arrays["vals"].shape] == ["n"]


def test_a_loop_of_no_trips_bounds_nothing():
    sdfg = trip_guarded_copy("trip_guarded_no_trips", "0")
    assert BoundDataDependentExtents().apply_pass(sdfg, {}) is None
    assert [str(extent) for extent in sdfg.arrays["vals"].shape] == ["n"]


def test_a_symbol_assigned_twice_in_one_arm_bounds_nothing():
    """Both arms copy ``A[0:n]`` with ``n = sizes[0]``, but one arm then sets ``n = sizes[1]`` and writes ``vals``:
    that value of ``n`` no access bounds."""
    sdfg = dace.SDFG("reassigned_in_one_arm")
    for symbol in ("n", "m"):
        sdfg.add_symbol(symbol, dace.int64)
    sdfg.add_array("sizes", [2], dace.int64)
    sdfg.add_array("A", ["m"], dace.float64)
    sdfg.add_transient("vals", ["n"], dace.float64)
    first = sdfg.add_state("first", is_start_block=True)
    branch = ConditionalBlock("branch")
    sdfg.add_node(branch)
    sdfg.add_edge(first, branch, dace.InterstateEdge())
    for label, condition in (("then", CodeBlock("m > 3")), ("otherwise", None)):
        arm = ControlFlowRegion(label, sdfg=sdfg)
        branch.add_branch(condition, arm)
        head = arm.add_state(f"{label}_head", is_start_block=True)
        copy = arm.add_state(f"{label}_copy")
        arm.add_edge(head, copy, dace.InterstateEdge(assignments={"n": "sizes[0]"}))
        copy.add_edge(copy.add_read("A"), None, copy.add_write("vals"), None, dace.Memlet("A[0:n]"))
        if condition is not None:
            again = arm.add_state(f"{label}_again")
            arm.add_edge(copy, again, dace.InterstateEdge(assignments={"n": "sizes[1]"}))
            tasklet = again.add_tasklet("one", {}, {"o"}, "o = 1.0")
            again.add_edge(tasklet, "o", again.add_write("vals"), None, dace.Memlet("vals[0]"))
    sdfg.reset_cfg_list()
    assert BoundDataDependentExtents().apply_pass(sdfg, {}) is None
    assert [str(extent) for extent in sdfg.arrays["vals"].shape] == ["n"]


def test_bounds_set_then_overwritten_before_any_read_bound_a_trip_count_extent():
    """ICON's ``get_indices``: ``lo``/``hi`` set by a three-way branch, ``hi`` overwritten on the last block, then
    ``A[lo - 1:hi]`` of ``A[m]`` shows ``1 <= lo`` and ``hi <= m``: ``vals[Max(0, hi - lo + 1)]`` takes ``m``."""
    sdfg = dace.SDFG("get_indices_window")
    for symbol in ("lo", "hi", "m", "b"):
        sdfg.add_symbol(symbol, dace.int64)
    sdfg.add_array("sizes", [2], dace.int64)
    sdfg.add_array("A", ["m"], dace.float64)
    sdfg.add_transient("vals", ["Max(0, hi - lo + 1)"], dace.float64)
    first = sdfg.add_state("first", is_start_block=True)
    previous = first
    for label, arms in (
        (
            "bounds",
            (
                ("b == 0", {"lo": "sizes[0]", "hi": "m"}),
                ("b == 1", {"lo": "1", "hi": "sizes[1]"}),
                (None, {"lo": "1", "hi": "m"}),
            ),
        ),
        ("last", (("b == 1", {"hi": "sizes[1]"}),)),
    ):
        branch = ConditionalBlock(label)
        sdfg.add_node(branch)
        sdfg.add_edge(previous, branch, dace.InterstateEdge())
        for index, (condition, assignments) in enumerate(arms):
            arm = ControlFlowRegion(f"{label}_{index}", sdfg=sdfg)
            branch.add_branch(None if condition is None else CodeBlock(condition), arm)
            head = arm.add_state(f"{label}_{index}_head", is_start_block=True)
            arm.add_state_after(head, f"{label}_{index}_set", assignments=assignments)
        previous = branch
    copy = sdfg.add_state("copy")
    sdfg.add_edge(previous, copy, dace.InterstateEdge())
    copy.add_edge(
        copy.add_read("A"), None, copy.add_write("vals"), None, dace.Memlet("A[lo - 1:hi] -> [0:hi - lo + 1]")
    )
    sdfg.reset_cfg_list()
    assert BoundDataDependentExtents().apply_pass(sdfg, {}) == 1
    assert [str(extent) for extent in sdfg.arrays["vals"].shape] == ["Max(0, m)"]


if __name__ == "__main__":
    test_an_in_bounds_access_of_a_literal_array_gives_a_capacity_and_keeps_the_shape()
    test_an_access_of_a_symbolic_array_gives_no_capacity()
    test_a_nonnegative_data_symbol_subtracted_from_an_extent_bounds_it_by_zero()
    test_a_nest_the_bound_names_a_new_symbol_for_is_given_it()
    test_an_extent_growing_with_the_data_symbol_is_left_alone()
    test_an_in_bounds_access_of_an_argument_bounds_the_extent_by_the_argument_extent()
    test_an_extent_scaled_by_an_argument_extent_takes_the_bound()
    test_an_access_on_one_branch_only_gives_no_bound()
    test_a_loop_over_the_transients_own_extent_bounds_it()
    test_a_loop_over_another_transients_extent_bounds_nothing()
    test_a_loop_of_no_trips_bounds_nothing()
    test_a_symbol_assigned_twice_in_one_arm_bounds_nothing()
    test_bounds_set_then_overwritten_before_any_read_bound_a_trip_count_extent()
