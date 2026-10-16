# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Unit tests for the ``UniqueLoopIterators`` pass."""

import numpy as np

import dace
from dace.sdfg.state import LoopRegion
from dace.transformation.passes.analysis import loop_analysis
from dace.transformation.passes.unique_loop_iterators import UniqueLoopIterators


@dace.program
def foo(A: dace.float64[10, 10], idx: dace.int32[10, 10], B: dace.float64[5, 10]):
    for i in range(5):
        for j, k in dace.map[0:10, 0:10]:
            A[j, k] = 1.1 * A[j, k] + 1.2 * B[i, idx[j, k]]


def test_nested_sdfg_symbol_mapping():
    """
    The map inside the loop body with nested SDFG.
    The loop variable ``i`` must appear in the nested SDFG's symbol_mapping.
    After UniqueLoopIterators, the symbol_mapping should reference the new
    ``_loop_it_<N>`` name, not the original ``i``.
    """

    sdfg = foo.to_sdfg(simplify=False)

    # Before: confirm ``i`` is the loop variable and appears in a nested SDFG mapping
    loops_before = [cfg for cfg in sdfg.all_control_flow_regions() if isinstance(cfg, LoopRegion)]
    assert len(loops_before) == 1
    assert loops_before[0].loop_variable == "i"

    found_i_in_mapping = False
    for state in sdfg.states():
        for node in state.nodes():
            if isinstance(node, dace.nodes.NestedSDFG):
                if "i" in node.symbol_mapping:
                    found_i_in_mapping = True
    assert found_i_in_mapping, "Expected 'i' in nested SDFG symbol_mapping before pass"

    # Apply pass
    UniqueLoopIterators().apply_pass(sdfg, None)
    sdfg.validate()

    # After: the nested SDFG symbol_mapping should have _loop_it_0, not i
    for state in sdfg.states():
        for node in state.nodes():
            if isinstance(node, dace.nodes.NestedSDFG):
                assert "i" not in node.symbol_mapping, (
                    f"Original loop var 'i' should not be in symbol_mapping, got {node.symbol_mapping}"
                )
                assert "_loop_it_0" in node.symbol_mapping, (
                    f"renamed loop var '_loop_it_0' should be in symbol_mapping, got {node.symbol_mapping}"
                )

    A = np.random.rand(10, 10)
    idx = np.random.randint(0, 10, size=(10, 10), dtype=np.int32)
    B = np.random.rand(5, 10)

    A_ref = A.copy()
    for i in range(5):
        for j in range(10):
            for k in range(10):
                A_ref[j, k] = 1.1 * A_ref[j, k] + 1.2 * B[i, idx[j, k]]

    csdfg = sdfg.compile()
    csdfg(A=A, idx=idx, B=B)
    assert np.allclose(A, A_ref), f"Max error: {np.max(np.abs(A - A_ref))}"


@dace.program
def loop_var_unread_after(A: dace.float64[10], B: dace.float64[10]):
    for i in range(10):
        A[i] = 2.0 * B[i]


def loop_then_read(read_after: bool) -> dace.SDFG:
    """``for i in range(10)``, then a state that writes ``i`` (or a constant) to ``out[0]``."""
    sdfg = dace.SDFG(f"loop_then_read_{int(read_after)}")
    sdfg.add_array("out", [1], dace.int64)
    sdfg.add_symbol("i", dace.int64)
    loop = LoopRegion("loop", condition_expr="i < 10", loop_var="i", initialize_expr="i = 0", update_expr="i = i + 1")
    sdfg.add_node(loop, is_start_block=True)
    loop.add_state("body", is_start_block=True)
    after = sdfg.add_state("after")
    sdfg.add_edge(loop, after, dace.InterstateEdge())
    t = after.add_tasklet("read", set(), {"_o"}, "_o = i" if read_after else "_o = 0")
    after.add_edge(t, "_o", after.add_access("out"), None, dace.Memlet("out[0]"))
    return sdfg


def post_value_states(sdfg: dace.SDFG) -> list:
    return [s for s in sdfg.states() if "loop_iter_post_value" in s.label]


def test_loop_var_reconstruction():
    """A read of the iterator after the loop sees its exit value, one step past the last iteration (Fortran ``DO``)."""
    sdfg = loop_then_read(read_after=True)
    UniqueLoopIterators().apply_pass(sdfg, None)
    sdfg.validate()
    assert len(post_value_states(sdfg)) == 1

    loop = next(cfg for cfg in sdfg.all_control_flow_regions() if isinstance(cfg, LoopRegion))
    out_edges = loop.parent_graph.out_edges(loop)
    assert len(out_edges) == 1
    assignments = out_edges[0].data.assignments
    assert "i" in assignments, f"Expected assignment to 'i', got {assignments}"

    loop_end = loop_analysis.get_loop_end(loop)
    init = loop_analysis.get_init_assignment(loop)
    stride = loop_analysis.get_loop_stride(loop)
    import sympy as sp

    expected_post = init + (loop_end - init + stride) - sp.Mod(loop_end - init + stride, stride)
    assert str(assignments["i"]) == f"({expected_post})", f"Expected post-value {expected_post}, got {assignments['i']}"

    out = np.zeros(1, dtype=np.int64)
    sdfg(out=out)
    assert out[0] == 10


def test_no_post_value_without_a_read_after_the_loop():
    for sdfg in (loop_then_read(read_after=False), loop_var_unread_after.to_sdfg(simplify=False)):
        UniqueLoopIterators().apply_pass(sdfg, None)
        sdfg.validate()
        assert post_value_states(sdfg) == []


@dace.program
def nested_loops(A: dace.float64[8, 8]):
    for i in range(8):
        for j in range(8):
            A[i, j] = A[i, j] + 1.0


def test_loop_bound_with_indirect_array():
    """Loop bound is ``row_ptr(i+1) - 1`` -- sympy's default ``str``
    would render the array access as a function call.  The pass must use
    ``arrayexprs=`` so the reconstruction assignment carries the
    Python subscript form."""

    sdfg = dace.SDFG("indirect_bound")
    sdfg.add_array("row_ptr", [11], dace.int64)
    sdfg.add_array("values", [20], dace.float64)
    sdfg.add_array("acc", [10], dace.float64)
    sdfg.add_symbol("i", dace.int64)
    sdfg.add_symbol("j", dace.int64)

    outer = LoopRegion("outer", condition_expr="i < 10", loop_var="i", initialize_expr="i = 0", update_expr="i = i + 1")
    sdfg.add_node(outer, is_start_block=True)
    body = outer.add_state("outer_body", is_start_block=True)
    inner = LoopRegion(
        "inner",
        condition_expr="j < row_ptr[i + 1]",
        loop_var="j",
        initialize_expr="j = row_ptr[i]",
        update_expr="j = j + 1",
    )
    outer.add_node(inner)
    outer.add_edge(body, inner, dace.InterstateEdge())
    # Reads ``j`` after the inner loop, so the loop needs its exit value
    sdfg.add_symbol("last_j", dace.int64)
    outer.add_edge(inner, outer.add_state("after_inner"), dace.InterstateEdge(assignments={"last_j": "j"}))

    inner_body = inner.add_state("inner_body", is_start_block=True)
    rd = inner_body.add_access("values")
    wr = inner_body.add_access("acc")
    t = inner_body.add_tasklet("add", {"_v", "_a"}, {"_o"}, "_o = _v + _a")
    inner_body.add_edge(rd, None, t, "_v", dace.Memlet("values[j]"))
    inner_body.add_edge(wr, None, t, "_a", dace.Memlet("acc[i]"))
    wr2 = inner_body.add_access("acc")
    inner_body.add_edge(t, "_o", wr2, None, dace.Memlet("acc[i]"))

    pass_ = UniqueLoopIterators()
    pass_.apply_pass(sdfg, None)
    sdfg.validate()

    # The inner-loop reconstruction state must use Python subscript form.
    recon = [s for s in sdfg.states() if hasattr(s, "label") and "loop_iter_post_value" in s.label]
    rhs_strings = [str(rhs) for s in recon for e in s.parent_graph.in_edges(s) for rhs in e.data.assignments.values()]

    assert any("row_ptr[" in s for s in rhs_strings), (
        f"Reconstruction RHS must use Python subscript form: {rhs_strings}"
    )
    assert all("row_ptr(" not in s for s in rhs_strings), (
        f"Reconstruction RHS must not use function-call form: {rhs_strings}"
    )


def test_while_loop_no_induction_var():
    """A LoopRegion that wasn't synthesised from a counted ``for`` (no
    ``loop_variable``) has nothing to rename.  The pass must skip it
    cleanly (there's no induction variable to leave a post-value for)."""

    sdfg = dace.SDFG("whileloop")
    sdfg.add_symbol("flag", dace.int64)

    body = LoopRegion("body", condition_expr="flag != 0", loop_var=None)
    sdfg.add_node(body, is_start_block=True)
    inner = body.add_state("inner", is_start_block=True)
    t = inner.add_tasklet("flip", set(), {"_o"}, "_o = 0")
    sdfg.add_array("out", [1], dace.int64)
    a = inner.add_access("out")
    inner.add_edge(t, "_o", a, None, dace.Memlet("out[0]"))

    pass_ = UniqueLoopIterators()
    pass_.apply_pass(sdfg, None)

    recon = [s for s in sdfg.states() if hasattr(s, "label") and "loop_iter_post_value" in s.label]
    assert recon == [], f"Unexpected reconstruction states: {[s.label for s in recon]}"


def test_no_postamble_drops_dead_symbol_declaration():
    """Nothing reads ``i`` after the loops, so the renamed iterator
    has no surviving reader, so the stale ``sdfg.symbols[old_name]``
    declaration left by the frontend must be removed -- otherwise it leaks
    as a phantom free symbol on the enclosing NestedSDFG boundary
    (validation: ``Missing symbols on nested SDFG: ['i']``)."""

    @dace.program
    def sibling_for_loops(x: dace.float64[10, 10]):
        for j in dace.map[0:10]:
            for i in range(10):
                x[i, j] = 1.0
            for i in range(10):
                x[i, j] += 2.0

    sdfg = sibling_for_loops.to_sdfg(simplify=False)

    # A loop iterator is never a declared symbol, so the frontend leaves no declaration of ``i`` to go stale; the
    # renaming must not introduce one either.
    body_nsdfgs = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.NestedSDFG)]

    p = UniqueLoopIterators()
    p.apply_pass(sdfg, {})

    # Every LoopRegion now has a unique ``_loop_it_<N>`` variable.
    loopvars = [n.loop_variable for n, _ in sdfg.all_nodes_recursive() if isinstance(n, LoopRegion)]
    assert loopvars == sorted(set(loopvars))
    assert all(v.startswith("_loop_it_") for v in loopvars)

    # An SDFG whose last use of ``i`` was renamed no longer declares it.
    for n in body_nsdfgs:
        assert "i" not in n.sdfg.symbols, f"stale 'i' still declared in {n.sdfg.name}.symbols"

    sdfg.validate()


def test_idempotent_skips_already_unique_iterators():
    """A second run renames nothing: iterators already named ``_loop_it_<N>`` keep their names, so a grandchild
    nested SDFG's import of an outer iterator stays intact."""

    # A map body (nested SDFG) with two nested loops, then a deeper nest --
    # the shape where a grandchild NSDFG imports the outer iterators.
    sdfg = nested_loops.to_sdfg(simplify=False)

    UniqueLoopIterators().apply_pass(sdfg, None)
    sdfg.validate()
    names_after_first = sorted(lr.loop_variable for lr in sdfg.all_control_flow_regions() if isinstance(lr, LoopRegion))
    assert all(n.startswith("_loop_it_") for n in names_after_first), names_after_first

    # Second application: the iterators are already unique; nothing changes
    # and the SDFG stays valid (no "Missing symbols" crash).
    UniqueLoopIterators().apply_pass(sdfg, None)
    sdfg.validate()
    names_after_second = sorted(
        lr.loop_variable for lr in sdfg.all_control_flow_regions() if isinstance(lr, LoopRegion)
    )
    assert names_after_second == names_after_first, (
        f"second pass re-renamed already-unique iterators: {names_after_first} -> {names_after_second}"
    )

    A = np.random.rand(8, 8)
    exp = A + 1.0
    csdfg = sdfg.compile()
    csdfg(A=A)
    assert np.allclose(A, exp)


N = dace.symbol("N")


@dace.program
def sibling_kbound_loops(out: dace.float64[N, N], arr: dace.float64[N, N], x: dace.int32):
    """Map body with two sibling ``k``-loops that share the name ``k`` and
    the same per-``i`` bounds, one of them inside an ``if`` guard. Both
    accumulate into ``out`` over the same sparse ``[beg:end)`` range."""
    for i in dace.map[0:N]:
        beg = i // 2 + 1
        end = beg + 2
        for k in range(beg, end):
            out[i, k] += arr[i, k]
        if x > 0:
            for k in range(beg, end):
                out[i, k] += 2.0 * arr[i, k]


def _sibling_kbound_oracle(arr, x, n):
    out = np.zeros((n, n))
    for i in range(n):
        beg, end = i // 2 + 1, i // 2 + 3
        for k in range(beg, min(end, n)):
            out[i, k] += arr[i, k]
        if x > 0:
            for k in range(beg, min(end, n)):
                out[i, k] += 2.0 * arr[i, k]
    return out


def test_value_preserving_sibling_kbound_loops():
    """Renaming two sibling loops that share the iterator ``k`` keeps their memlets apart, including the identical
    ``arr[i, k]`` reads the frontend builds for both, and the guarded accumulation keeps its value."""
    n = 8
    rng = np.random.default_rng(1)
    arr = rng.standard_normal((n, n))
    for x in (1, 0):
        sdfg = sibling_kbound_loops.to_sdfg(simplify=True)
        UniqueLoopIterators().apply_pass(sdfg, {})
        sdfg.validate()
        out = np.zeros((n, n))
        sdfg(out=out, arr=arr.copy(), x=np.int32(x), N=n)
        assert np.allclose(out, _sibling_kbound_oracle(arr, x, n)), f"value corrupted for x={x}"


@dace.program
def descending_loop(A: dace.float64[10], B: dace.float64[10]):
    for i in range(9, -1, -1):
        A[i] = 2.0 * B[i] + i


def test_negative_step_loop_value_preserving():
    """A loop with a negative step (``range(9, -1, -1)``) is renamed like any other and stays value-preserving."""
    sdfg = descending_loop.to_sdfg(simplify=False)
    UniqueLoopIterators().apply_pass(sdfg, None)
    sdfg.validate()

    loopvars = [r.loop_variable for r in sdfg.all_control_flow_regions(recursive=True) if isinstance(r, LoopRegion)]
    assert loopvars and all(v.startswith("_loop_it_") for v in loopvars), loopvars

    A = np.zeros(10)
    B = np.random.rand(10)
    sdfg.compile()(A=A, B=B)
    assert np.allclose(A, 2.0 * B + np.arange(10))


def test_seeds_counter_past_existing_map_params():
    """The counter must seed past iterators already in ``_loop_it_<N>`` form on
    **maps**, not only on loops. ``LoopToMap`` keeps the loop's ``_loop_it_<N>``
    name as the map parameter, so when this pass runs after a LoopToMap stage a
    map carries that name. Here a map parameter ``_loop_it_5`` coexists with a
    loop named ``i``; ``i`` must become ``_loop_it_6`` (not reuse 0 or collide
    with the map's 5 -- the collision aliased two iteration variables and
    silently corrupted results)."""
    sdfg = dace.SDFG("seed_past_map")
    sdfg.add_array("mout", [5], dace.int64)
    sdfg.add_array("out", [10], dace.int64)

    # A map whose parameter is already in unique ``_loop_it_5`` form.
    mstate = sdfg.add_state("mstate", is_start_block=True)
    mstate.add_mapped_tasklet(
        "themap",
        {"_loop_it_5": "0:5"},
        {},
        "_o = _loop_it_5",
        {"_o": dace.Memlet("mout[_loop_it_5]")},
        external_edges=True,
    )

    # A fresh loop to be renamed; it must not reuse the map's id.
    fresh = LoopRegion("fresh", condition_expr="i < 10", loop_var="i", initialize_expr="i = 0", update_expr="i = i + 1")
    sdfg.add_node(fresh)
    sdfg.add_edge(mstate, fresh, dace.InterstateEdge())
    fs = fresh.add_state("fs", is_start_block=True)
    ft = fs.add_tasklet("w", set(), {"_o"}, "_o = i")
    fa = fs.add_access("out")
    fs.add_edge(ft, "_o", fa, None, dace.Memlet("out[i]"))

    UniqueLoopIterators().apply_pass(sdfg, None)
    sdfg.validate()

    loopvars = {r.loop_variable for r in sdfg.all_control_flow_regions() if isinstance(r, LoopRegion)}
    assert loopvars == {"_loop_it_6"}, f"fresh iterator must seed past the map parameter 5 (-> 6): {loopvars}"
    mapparams = {p for n, _ in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.MapEntry) for p in n.map.params}
    assert mapparams == {"_loop_it_5"}, f"map parameter must be left untouched: {mapparams}"


@dace.program
def shadowing_loops(A: dace.float64[4]):
    for i in range(2):
        for i in range(3):
            A[:] = A + 1.0


def test_an_inner_loop_shadowing_the_outer_iterator_keeps_its_own_iterator():
    """An inner loop rebinding the outer loop's name stays a different variable after the rename."""
    sdfg = shadowing_loops.to_sdfg(simplify=False)
    loops = [r for r in sdfg.all_control_flow_regions(recursive=True) if isinstance(r, LoopRegion)]
    assert [loop.loop_variable for loop in loops] == ["i", "i"]

    UniqueLoopIterators().apply_pass(sdfg, None)
    sdfg.validate()
    assert len({loop.loop_variable for loop in loops}) == 2

    A = np.zeros(4)
    sdfg(A=A)
    assert np.allclose(A, 6.0)


if __name__ == "__main__":
    test_nested_sdfg_symbol_mapping()
    test_loop_var_reconstruction()
    test_no_post_value_without_a_read_after_the_loop()
    test_loop_bound_with_indirect_array()
    test_while_loop_no_induction_var()
    test_no_postamble_drops_dead_symbol_declaration()
    test_idempotent_skips_already_unique_iterators()
    test_value_preserving_sibling_kbound_loops()
    test_negative_step_loop_value_preserving()
    test_seeds_counter_past_existing_map_params()
    test_an_inner_loop_shadowing_the_outer_iterator_keeps_its_own_iterator()
