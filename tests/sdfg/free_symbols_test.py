# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
import math

import dace

N, M, K, L, unused = (dace.symbol(s) for s in "NMKLU")


@dace.program
def fsymtest(A: dace.float32[20, N]):
    for i, j in dace.map[0:20, 0:N]:
        with dace.tasklet:
            a << A[i, i + M]
            b >> A[i, j]
            b = a + math.exp(K)


@dace.program
def fsymtest_multistate(A: dace.float32[20, N]):
    for k in range(K):
        for i, j in dace.map[0:20, 0:N]:
            with dace.tasklet:
                a << A[i, k + M]
                b >> A[i, j]
                b = L + a


def test_single_state():
    sdfg: dace.SDFG = fsymtest.to_sdfg()
    sdfg.simplify()
    assert len(sdfg.nodes()) == 1
    state = sdfg.node(0)
    assert state.free_symbols == {"M", "N", "K"}


def test_state_subgraph():
    sdfg = dace.SDFG("fsymtest2")
    state = sdfg.add_state()

    # Add a nested SDFG
    nsdfg = dace.SDFG("nsdfg")
    nstate = nsdfg.add_state()
    me, mx = state.add_map("map", dict(i="0:N"))
    nsdfg = state.add_nested_sdfg(nsdfg, {}, {}, symbol_mapping=dict(l=L / 2, i="i"))
    state.add_nedge(me, nsdfg, dace.Memlet())
    state.add_nedge(nsdfg, mx, dace.Memlet())

    # Entire graph
    assert state.free_symbols == {"L", "N"}

    # Try a subgraph containing only the map contents
    assert state.scope_subgraph(me, include_entry=False, include_exit=False).free_symbols == {"L", "i"}


def test_sdfg():
    sdfg: dace.SDFG = fsymtest_multistate.to_sdfg()
    sdfg.simplify()
    # Test each state separately
    for state in sdfg.states():
        assert state.free_symbols == {"k", "N", "M", "L"} or state.free_symbols == set()
    # The SDFG itself should have another free symbol
    assert sdfg.free_symbols == {"K", "M", "N", "L"}


def test_constants():
    sdfg: dace.SDFG = fsymtest_multistate.to_sdfg()
    sdfg.simplify()
    sdfg.add_constant("K", 5)
    sdfg.add_constant("L", 20)

    for state in sdfg.states():
        assert state.free_symbols == {"k", "N", "M"} or state.free_symbols == set()
    assert sdfg.free_symbols == {"M", "N"}


def test_interstate_edge_symbols():
    i, j, k = (dace.symbol(s) for s in "ijk")

    edge = dace.InterstateEdge(assignments={"i": "j + k"})
    assert "j" in edge.free_symbols
    assert "k" in edge.free_symbols
    assert "i" not in edge.free_symbols

    edge = dace.InterstateEdge(assignments={"i": "i+1"})
    assert "i" in edge.free_symbols

    edge = dace.InterstateEdge(condition="i < j", assignments={"i": "3"})
    assert "i" in edge.free_symbols
    assert "j" in edge.free_symbols

    edge = dace.InterstateEdge(assignments={"j": "i + 1", "i": "3"})
    assert "i" in edge.free_symbols
    assert "j" not in edge.free_symbols


def test_nested_sdfg_free_symbols():
    i, j, k = (dace.symbol(s) for s in "ijk")

    outer_sdfg = dace.SDFG("outer")
    outer_init_state = outer_sdfg.add_state("outer_init")
    outer_guard_state = outer_sdfg.add_state("outer_guard")
    outer_body_state_1 = outer_sdfg.add_state("outer_body_1")
    outer_body_state_2 = outer_sdfg.add_state("outer_body_2")
    outer_exit_state = outer_sdfg.add_state("outer_exit")
    outer_sdfg.add_edge(outer_init_state, outer_guard_state, dace.InterstateEdge(assignments={"i": "0"}))
    outer_sdfg.add_edge(
        outer_guard_state, outer_body_state_1, dace.InterstateEdge(condition="i < 10", assignments={"j": "i + 1"})
    )
    outer_sdfg.add_edge(outer_guard_state, outer_exit_state, dace.InterstateEdge(condition="i >= 10"))
    outer_sdfg.add_edge(
        outer_body_state_1, outer_guard_state, dace.InterstateEdge(condition="j >= 10", assignments={"i": "i + 1"})
    )
    outer_sdfg.add_edge(outer_body_state_1, outer_body_state_2, dace.InterstateEdge(condition="j < 10"))
    outer_sdfg.add_edge(outer_body_state_2, outer_body_state_1, dace.InterstateEdge(assignments={"j": "j + 1"}))

    inner_sdfg = dace.SDFG("inner")
    inner_init_state = inner_sdfg.add_state("inner_init")
    inner_guard_state = inner_sdfg.add_state("inner_guard")
    inner_body_state = inner_sdfg.add_state("inner_body")
    inner_exit_state = inner_sdfg.add_state("inner_exit")
    inner_sdfg.add_edge(inner_init_state, inner_guard_state, dace.InterstateEdge(assignments={"k": "j + 1"}))
    inner_sdfg.add_edge(inner_guard_state, inner_body_state, dace.InterstateEdge(condition="k < 10"))
    inner_sdfg.add_edge(
        inner_guard_state, inner_exit_state, dace.InterstateEdge(condition="k >= 10", assignments={"j": "j + 1"})
    )
    inner_sdfg.add_edge(inner_body_state, inner_guard_state, dace.InterstateEdge(assignments={"k": "k + 1"}))

    outer_body_state_2.add_nested_sdfg(inner_sdfg, {}, {}, symbol_mapping={"j": "j"})

    assert not outer_sdfg.free_symbols
    assert "i" not in inner_sdfg.free_symbols
    assert "j" in inner_sdfg.free_symbols
    assert "k" not in inner_sdfg.free_symbols


def _with_optional_unused_array(create_unused: bool) -> dace.SDFG:
    """Two used arrays plus, optionally, an unused transient whose shape names ``x_shape``."""
    sdfg = dace.SDFG("unused_transient")
    state = sdfg.add_state()
    sdfg.add_array("a", (10,), dace.float64, transient=False)
    sdfg.add_array("b", (10,), dace.float64, transient=False)
    sdfg.add_symbol("x_shape", dace.int32)
    if create_unused:
        sdfg.add_array("x", ("x_shape",), dace.float32, transient=True)
    state.add_mapped_tasklet(
        "map",
        {"__i": "0:10"},
        {"__in": dace.Memlet("a[__i]")},
        "__out = __in + 1.90",
        {"__out": dace.Memlet("b[__i]")},
        external_edges=True,
    )
    return sdfg


def test_unused_array_does_not_leak_shape_symbol():
    """Issue #2382: declaring an unused array must not change what the SDFG needs to be invoked."""
    without = _with_optional_unused_array(False)
    with_unused = _with_optional_unused_array(True)

    assert "x_shape" not in with_unused.used_symbols(all_symbols=False)
    assert "x_shape" not in with_unused.arglist()
    assert list(without.arglist().keys()) == list(with_unused.arglist().keys())
    assert without.init_signature() == with_unused.init_signature()


def test_used_array_keeps_symbolic_extent():
    """The converse: an array used only through a map memlet keeps its shape and stride symbols."""
    n, s = dace.symbol("n"), dace.symbol("s")
    sdfg = dace.SDFG("used_via_map")
    sdfg.add_array("a", (n,), dace.float64, strides=(s,), transient=False)
    sdfg.add_array("b", (n,), dace.float64, transient=False)
    state = sdfg.add_state()
    state.add_mapped_tasklet(
        "m",
        {"__i": "0:n"},
        {"__in": dace.Memlet("a[__i]")},
        "__out = __in + 1.0",
        {"__out": dace.Memlet("b[__i]")},
        external_edges=True,
    )
    sdfg.validate()

    used = sdfg.used_symbols(all_symbols=False)
    assert {"n", "s"} <= used
    assert {"n", "s"} <= sdfg.arglist().keys()


def test_array_used_only_in_a_guard_keeps_its_stride_symbol():
    """An array referenced only by a ConditionalBlock guard is still used, so its stride survives."""
    from dace.properties import CodeBlock
    from dace.sdfg.state import ConditionalBlock, ControlFlowRegion, LoopRegion

    sdfg = dace.SDFG("used_codeblock_array")
    sdfg.add_symbol("S", dace.int32)
    sdfg.add_array("A", (10, 10), dace.int32, strides=(1, dace.symbol("S")))
    sdfg.add_scalar("acc", dace.int32, transient=True)

    loop = LoopRegion("loop", condition_expr="k < 5", loop_var="k", initialize_expr="k = 0", update_expr="k = k + 1")
    sdfg.add_node(loop, is_start_block=True)
    cb = ConditionalBlock("cb")
    loop.add_node(cb, is_start_block=True)
    branch = ControlFlowRegion("branch", sdfg=sdfg)
    cb.add_branch(CodeBlock("A[0, k] == 1"), branch)
    set_one = branch.add_state("set_one", is_start_block=True)
    t = set_one.add_tasklet("t_set", {}, {"o"}, "o = 1")
    set_one.add_edge(t, "o", set_one.add_write("acc"), None, dace.Memlet("acc[0]"))
    sdfg.validate()

    assert "S" in sdfg.used_symbols(all_symbols=False)
    assert "S" in sdfg.init_signature()


def test_descriptor_free_symbols_follow_a_reassigned_extent():
    stream = dace.data.Stream(dace.float64, buffer_size=N, transient=True)
    assert stream.free_symbols == {N}
    stream.buffer_size = M
    assert stream.free_symbols == {M}

    array = dace.data.Array(dace.float64, [N])
    assert array.free_symbols == {N}
    array.set_shape([K])
    assert array.free_symbols == {K}


def test_a_state_reading_a_symbol_before_an_edge_assigns_it_keeps_it_free():
    """``first`` reads ``k`` before any edge assigns it, so the caller must pass ``k``; a later reassignment does not
    make that read defined."""
    sdfg = dace.SDFG("read_before_assignment")
    sdfg.add_array("a", [1], dace.int64)
    sdfg.add_symbol("k", dace.int64)
    first = sdfg.add_state("first", is_start_block=True)
    tasklet = first.add_tasklet("t", {}, {"o"}, "o = k")
    first.add_edge(tasklet, "o", first.add_write("a"), None, dace.Memlet("a[0]"))
    second = sdfg.add_state_after(first, "second", assignments={"k": "0"})
    sdfg.add_state_after(second, "third", assignments={"k": "1"})

    assert "k" in sdfg.free_symbols


def read_into(region, label: str, symbol: str, is_start_block: bool = False) -> dace.SDFGState:
    """A state of ``region`` whose tasklet copies ``symbol`` into ``a[0]``."""
    state = region.add_state(label, is_start_block=is_start_block)
    tasklet = state.add_tasklet(f"{label}_read", {}, {"o": None}, f"o = {symbol}")
    state.add_edge(tasklet, "o", state.add_write("a"), None, dace.Memlet("a[0]"))
    return state


def symbol_sdfg(name: str) -> dace.SDFG:
    sdfg = dace.SDFG(name)
    sdfg.add_array("a", [1], dace.int64)
    sdfg.add_symbol("k", dace.int64)
    return sdfg


def test_a_loop_body_reading_a_symbol_its_own_later_edge_assigns_keeps_it_free():
    """The first iteration reads ``k`` before the body's edge assigns it, so the caller's ``k`` is live."""
    sdfg = symbol_sdfg("loop_read_before_assignment")
    loop = dace.sdfg.state.LoopRegion("loop", "i < 4", "i", "i = 0", "i = i + 1")
    sdfg.add_node(loop, is_start_block=True)
    first = read_into(loop, "first", "k", is_start_block=True)
    loop.add_edge(first, loop.add_state("second"), dace.InterstateEdge(assignments={"k": "i"}))

    assert "k" in sdfg.free_symbols


def test_a_symbol_one_branch_assigns_and_the_join_reads_stays_free():
    """The else arm leaves ``k`` as the caller passed it, so the read after the join needs the caller's ``k``."""
    sdfg = symbol_sdfg("one_branch_assignment")
    block = dace.sdfg.state.ConditionalBlock("if_region", sdfg=sdfg)
    then = dace.sdfg.state.ControlFlowRegion("then", sdfg=sdfg, parent=block)
    then.add_edge(
        then.add_state("then_first", is_start_block=True),
        then.add_state("then_second"),
        dace.InterstateEdge(assignments={"k": "1"}),
    )
    block.add_branch(dace.sdfg.state.CodeBlock("a[0] > 0"), then)
    sdfg.add_node(block, is_start_block=True)
    sdfg.add_edge(block, read_into(sdfg, "join", "k"), dace.InterstateEdge())

    assert "k" in sdfg.free_symbols


def branch_assignments(name: str, conditions: tuple[str, ...]) -> dace.SDFG:
    """Each of ``conditions`` guards a branch that assigns ``k``; a state after the block reads ``k``."""
    sdfg = symbol_sdfg(name)
    block = dace.sdfg.state.ConditionalBlock("if_region", sdfg=sdfg)
    for index, condition in enumerate(conditions):
        arm = dace.sdfg.state.ControlFlowRegion(f"arm_{index}", sdfg=sdfg, parent=block)
        arm.add_edge(
            arm.add_state(f"arm_{index}_first", is_start_block=True),
            arm.add_state(f"arm_{index}_second"),
            dace.InterstateEdge(assignments={"k": "1"}),
        )
        block.add_branch(dace.sdfg.state.CodeBlock(condition), arm)
    sdfg.add_node(block, is_start_block=True)
    sdfg.add_edge(block, read_into(sdfg, "join", "k"), dace.InterstateEdge())
    return sdfg


def test_a_symbol_assigned_by_branches_whose_last_condition_negates_the_others_is_not_free():
    """``a[0] <= 0`` is the negation of ``a[0] > 0``, the else arm ControlFlowRaising would emit, so no path skips both
    assignments."""
    sdfg = branch_assignments("complement_branch_assignment", ("a[0] > 0", "a[0] <= 0"))

    assert "k" not in sdfg.free_symbols


def test_a_symbol_assigned_by_branches_whose_conditions_are_not_a_negation_stays_free():
    """``a[0] > 1`` and ``a[0] < 3`` cover every integer, but neither is the negation of the other, so the block keeps
    a path that skips both assignments."""
    sdfg = branch_assignments("overlapping_branch_assignment", ("a[0] > 1", "a[0] < 3"))

    assert "k" in sdfg.free_symbols


def test_a_symbol_assigned_on_every_path_before_its_read_is_not_free():
    """Every path to the read passes the assignment, so the caller's ``k`` is dead: no over-reporting."""
    sdfg = symbol_sdfg("assigned_before_read")
    first = sdfg.add_state("first", is_start_block=True)
    sdfg.add_edge(first, read_into(sdfg, "reader", "k"), dace.InterstateEdge(assignments={"k": "1"}))

    assert "k" not in sdfg.free_symbols


def test_a_cpp_tasklet_local_listed_in_ignored_symbols_is_not_a_read_of_the_symbol():
    """A C++ tasklet is scanned token by token; a local named like a symbol is a read unless the tasklet lists it in
    ``ignored_symbols``. Listed, the later assignment defines ``k`` and nothing reads the caller's ``k``."""
    sdfg = symbol_sdfg("cpp_local_shadow")
    first = sdfg.add_state("first", is_start_block=True)
    tasklet = first.add_tasklet("local", {}, {"o": None}, "long long k = 2;\no = k;", language=dace.Language.CPP)
    tasklet.ignored_symbols = {"k"}
    first.add_edge(tasklet, "o", first.add_write("a"), None, dace.Memlet("a[0]"))
    sdfg.add_edge(first, read_into(sdfg, "reader", "k"), dace.InterstateEdge(assignments={"k": "1"}))

    assert "k" not in sdfg.free_symbols


if __name__ == "__main__":
    test_single_state()
    test_state_subgraph()
    test_sdfg()
    test_constants()
    test_interstate_edge_symbols()
    test_nested_sdfg_free_symbols()
    test_unused_array_does_not_leak_shape_symbol()
    test_used_array_keeps_symbolic_extent()
    test_array_used_only_in_a_guard_keeps_its_stride_symbol()
    test_descriptor_free_symbols_follow_a_reassigned_extent()
    test_a_state_reading_a_symbol_before_an_edge_assigns_it_keeps_it_free()
    test_a_loop_body_reading_a_symbol_its_own_later_edge_assigns_keeps_it_free()
    test_a_symbol_one_branch_assigns_and_the_join_reads_stays_free()
    test_a_symbol_assigned_on_every_path_before_its_read_is_not_free()
    test_a_cpp_tasklet_local_listed_in_ignored_symbols_is_not_a_read_of_the_symbol()
