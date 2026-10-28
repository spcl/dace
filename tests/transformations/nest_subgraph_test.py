# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
import warnings

import numpy as np
import pytest

import dace
from dace.sdfg import nodes
from dace.sdfg.analysis.schedule_tree import sdfg_to_tree
from dace.sdfg.analysis.schedule_tree import treenodes as tn
from dace.sdfg.graph import SubgraphView
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion, LoopRegion, StateSubgraphView
from dace.transformation.helpers import nest_sdfg_control_flow, nest_sdfg_subgraph, nest_state_subgraph


@dace.program
def nest_subgraph(A: dace.float64[1], B: dace.float64[1]):
    for i in dace.map[0:1]:
        with dace.tasklet:
            a << A[i]
            b >> B[i]
            b = a


def test_nest_oneelementmap():
    A, B = np.random.rand(1), np.random.rand(1)
    sdfg: dace.SDFG = nest_subgraph.to_sdfg()
    state: dace.SDFGState
    # Nest outer region
    for node, state in sdfg.all_nodes_recursive():
        if isinstance(node, dace.nodes.MapEntry):
            subgraph = state.scope_subgraph(node)
            nest_state_subgraph(sdfg, state, subgraph)
    # Nest inner scope
    for node, state in sdfg.all_nodes_recursive():
        if isinstance(node, dace.nodes.MapEntry):
            subgraph = state.scope_subgraph(node, include_entry=False, include_exit=False)
            nest_state_subgraph(state.parent, state, subgraph)

    sdfg(A=A, B=B)
    assert np.allclose(A, B)


def test_internal_outarray():
    sdfg = dace.SDFG("internal_outarr")
    sdfg.add_array("A", [20], dace.float64)
    state = sdfg.add_state()

    me, mx = state.add_map("_", dict(i="0:1"))
    t = state.add_tasklet("doit", {}, {"a"}, "a = 0")
    w = state.add_write("A")
    state.add_nedge(me, t, dace.Memlet())
    state.add_edge(t, "a", w, None, dace.Memlet("A[1]"))
    state.add_nedge(w, mx, dace.Memlet())

    subgraph = StateSubgraphView(state, [t, w])
    nest_state_subgraph(sdfg, state, subgraph)

    a = np.random.rand(20)
    sdfg(A=a)
    assert a[1] == 0


def test_nested_symbol_dtypes():
    M = dace.symbol("M", dace.int64)
    sdfg = dace.SDFG("nested_symbol_dtypes")
    sdfg.add_symbol("M", dace.int64)
    sdfg.add_array("A", [M], dace.float64)
    sdfg.add_array("B", [M], dace.float64)
    state = sdfg.add_state()

    me, mx = state.add_map("m", dict(j="0:M"))
    t = state.add_tasklet("doit", {"a"}, {"b"}, "b = a + j")
    r = state.add_read("A")
    w = state.add_write("B")
    state.add_memlet_path(r, me, t, dst_conn="a", memlet=dace.Memlet("A[j]"))
    state.add_memlet_path(t, mx, w, src_conn="b", memlet=dace.Memlet("B[j]"))

    nsdfg = nest_state_subgraph(sdfg, state, StateSubgraphView(state, [t]))
    # Scope symbols must keep their parent-defined types (j is an int64 map parameter)
    assert nsdfg.sdfg.symbols["M"] == dace.int64
    assert nsdfg.sdfg.symbols["j"] == dace.int64

    a = np.random.rand(10)
    b = np.zeros(10)
    sdfg(A=a, B=b, M=10)
    assert np.allclose(b, a + np.arange(10))


def test_symbolic_return():

    @dace.program
    def symbolic_return():
        a = 6
        for i in range(10):
            a = 5
        a -= 1
        return i, a

    sdfg = symbolic_return.to_sdfg()

    stree = sdfg_to_tree.as_schedule_tree(sdfg)
    for_scope = None
    for i, child in enumerate(stree.children):
        if isinstance(child, tn.LoopScope):
            for_scope = child
            break
    assert for_scope

    assert i < len(stree.children) - 1
    exit_scope = stree.children[i + 1]
    assert isinstance(exit_scope, (tn.AssignNode, tn.TaskletNode))

    states = for_scope.loop.nodes()

    nest_sdfg_subgraph(sdfg, SubgraphView(for_scope.loop, states))

    result = sdfg()
    val = result[1][0]
    _, ref = symbolic_return.f()
    assert val == ref


def test_nest_cf_simple_for_loop():

    @dace.program
    def simple_for_loop():
        A = np.ndarray((10,), dtype=np.int32)
        for i in range(10):
            A[i] = i
        return A

    sdfg = simple_for_loop.to_sdfg()
    nest_sdfg_control_flow(sdfg)

    assert np.array_equal(sdfg(), np.arange(10, dtype=np.int32))


def test_nest_cf_simple_while_loop():

    def force_callback(f):
        return f

    @force_callback
    def update(x):
        return x + 1

    @dace.program
    def simple_while_loop():
        i = 0
        A = np.ndarray((10,), dtype=np.int32)
        while i < 10:
            A[i] = i
            i = update(A[i])
        return A

    with pytest.warns(match="Automatically creating callback"):
        sdfg = simple_while_loop.to_sdfg()
    nest_sdfg_control_flow(sdfg)

    assert np.array_equal(sdfg(update=update), np.arange(10, dtype=np.int32))


def test_nest_cf_simple_if():

    @dace.program
    def simple_if(i: dace.int64):
        if i < 5:
            return 0
        else:
            return 1

    sdfg = simple_if.to_sdfg()
    nest_sdfg_control_flow(sdfg)

    assert sdfg(2)[0] == 0
    assert sdfg(5)[0] == 1


def test_nest_cf_simple_if_elif():

    @dace.program
    def simple_if_elif(i: dace.int64):
        if i < 2:
            return 0
        elif i < 4:
            return 1
        elif i < 6:
            return 2
        elif i < 8:
            return 3
        else:
            return 4

    sdfg = simple_if_elif.to_sdfg()
    nest_sdfg_control_flow(sdfg)

    assert sdfg(0)[0] == 0
    assert sdfg(2)[0] == 1
    assert sdfg(4)[0] == 2
    assert sdfg(7)[0] == 3
    assert sdfg(15)[0] == 4


def test_nest_cf_simple_if_chain():

    @dace.program
    def simple_if_chain(i: dace.int64):
        if i < 2:
            return 0
        if i < 4:
            return 1
        if i < 6:
            return 2
        if i < 8:
            return 3
        return 4

    sdfg = simple_if_chain.to_sdfg()
    nest_sdfg_control_flow(sdfg)

    assert sdfg(0)[0] == 0
    assert sdfg(2)[0] == 1
    assert sdfg(4)[0] == 2
    assert sdfg(7)[0] == 3
    assert sdfg(15)[0] == 4


def test_nest_two_loops_with_same_variable():
    """Nesting a second region that defines the same symbol must name its symbol output consistently."""

    @dace.program
    def two_loops(A: dace.float64[10], B: dace.float64[10]):
        for i in range(10):
            A[i] = A[i] * 2
        for i in range(10):
            B[i] = B[i] + A[i]

    sdfg = two_loops.to_sdfg(simplify=True)
    loops = [b for b in sdfg.bfs_nodes(sdfg.start_block) if isinstance(b, LoopRegion)]
    assert len(loops) == 2
    nest_sdfg_subgraph(sdfg, SubgraphView(sdfg, [loops[0]]))
    nest_sdfg_subgraph(sdfg, SubgraphView(sdfg, [loops[1]]))
    sdfg.reset_cfg_list()
    sdfg.validate()

    A = np.random.rand(10)
    B = np.random.rand(10)
    A_ref = A * 2
    B_ref = B + A_ref
    sdfg(A=A, B=B)
    assert np.allclose(A, A_ref) and np.allclose(B, B_ref)


def test_a_container_moved_across_the_boundary_gets_one_connector():
    """``A`` flows in, through an interior access node of ``A`` (a read-modify-write chain), and out: the
    nested SDFG binds it through one input and one output connector, not a second ``A_0`` pair whose
    writeback is unordered with the first."""
    sdfg = dace.SDFG("rmw_chain")
    sdfg.add_array("A", [8], dace.float64)
    state = sdfg.add_state()
    entry, exit_ = state.add_map("m", {"i": "0:8"})
    first = state.add_tasklet("first", {"x"}, {"y"}, "y = x + 1")
    second = state.add_tasklet("second", {"x"}, {"y"}, "y = x * 2")
    middle = state.add_access("A")
    state.add_memlet_path(state.add_read("A"), entry, first, dst_conn="x", memlet=dace.Memlet("A[i]"))
    state.add_edge(first, "y", middle, None, dace.Memlet("A[i]"))
    state.add_edge(middle, None, second, "x", dace.Memlet("A[i]"))
    state.add_memlet_path(second, exit_, state.add_write("A"), src_conn="y", memlet=dace.Memlet("A[i]"))

    nested = nest_state_subgraph(sdfg, state, state.scope_subgraph(entry))

    assert set(nested.in_connectors) == {"A"} and set(nested.out_connectors) == {"A"}
    sdfg.validate()
    a = np.arange(8.0)
    sdfg(A=a)
    assert np.allclose(a, (np.arange(8.0) + 1) * 2)


def nested_node(state: dace.SDFGState) -> nodes.NestedSDFG:
    return next(n for n in state.nodes() if isinstance(n, nodes.NestedSDFG))


def loop_through_a_scalar(read_after: bool) -> tuple[dace.SDFG, LoopRegion]:
    """``for i: tmp = x[i] * 2; a[i] = tmp + 1``; with ``read_after``, ``tmp`` is also set before the loop and read by
    the edge after it."""
    N = dace.symbol("N")
    sdfg = dace.SDFG(f"loop_through_a_scalar_{int(read_after)}")
    sdfg.add_array("a", [N], dace.float64)
    sdfg.add_array("x", [N], dace.float64)
    sdfg.add_scalar("tmp", dace.float64, transient=True)
    loop = LoopRegion("loop", "i < N", "i", "i = 0", "i = i + 1")
    sdfg.add_node(loop, is_start_block=True)
    st = loop.add_state("body", is_start_block=True)
    t0 = st.add_tasklet("t0", {"inp": None}, {"out": None}, "out = inp * 2.0")
    tmp = st.add_access("tmp")
    st.add_edge(st.add_read("x"), None, t0, "inp", dace.Memlet("x[i]"))
    st.add_edge(t0, "out", tmp, None, dace.Memlet("tmp"))
    t1 = st.add_tasklet("t1", {"v": None}, {"out": None}, "out = v + 1.0")
    st.add_edge(tmp, None, t1, "v", dace.Memlet("tmp"))
    st.add_edge(t1, "out", st.add_write("a"), None, dace.Memlet("a[i]"))
    if read_after:
        init = sdfg.add_state_before(loop, "init", is_start_block=True)
        zero = init.add_tasklet("zero", {}, {"out": None}, "out = 0.0")
        init.add_edge(zero, "out", init.add_write("tmp"), None, dace.Memlet("tmp"))
        sdfg.add_edge(loop, sdfg.add_state("after"), dace.InterstateEdge(condition="tmp > 0.0"))
    return sdfg, loop


@pytest.mark.parametrize("read_after", [False, True], ids=["loop_only", "read_after"])
def test_a_transient_moves_into_the_nest_only_if_nothing_else_uses_it(read_after):
    """Without a use after the loop, ``tmp`` and the loop counter ``i`` stay inside the nested SDFG."""
    sdfg, loop = loop_through_a_scalar(read_after)

    node = nested_node(nest_sdfg_subgraph(sdfg, SubgraphView(sdfg, [loop])))
    sdfg.validate()

    assert ("tmp" in sdfg.arrays) == read_after
    assert set(node.in_connectors) == {"x"} | ({"tmp"} if read_after else set())
    if not read_after:
        assert node.sdfg.arrays["tmp"].transient
        assert sdfg.number_of_nodes() == 1  # No state hands ``i`` out
        x = np.random.default_rng(4).random(16)
        a = np.zeros(16)
        sdfg(a=a, x=x, N=16)
        assert np.allclose(a, x * 2.0 + 1.0)


def test_symbol_needing_an_incoming_value_stays_an_argument():
    """``N`` is read inside the region before the region reassigns it, so its value still comes from the caller."""
    sdfg = dace.SDFG("incoming_symbol")
    sdfg.add_array("A", [4], dace.int32)
    sdfg.add_symbol("N", dace.int32)
    pre = sdfg.add_state("pre", is_start_block=True)
    reg = ControlFlowRegion("reg", sdfg)
    sdfg.add_node(reg)
    sdfg.add_edge(pre, reg, dace.InterstateEdge())
    s0 = reg.add_state("s0", is_start_block=True)
    t0 = s0.add_tasklet("use0", {}, {"o": None}, "o = N")
    s0.add_edge(t0, "o", s0.add_access("A"), None, dace.Memlet("A[0]"))
    s1 = reg.add_state("s1")
    reg.add_edge(s0, s1, dace.InterstateEdge(assignments={"N": "7"}))
    t1 = s1.add_tasklet("use1", {}, {"o": None}, "o = N")
    s1.add_edge(t1, "o", s1.add_access("A"), None, dace.Memlet("A[1]"))
    sdfg.add_edge(reg, sdfg.add_state("post"), dace.InterstateEdge())

    reference = np.zeros(4, dtype=np.int32)
    dace.SDFG.from_json(sdfg.to_json())(A=reference, N=3)

    nest_sdfg_subgraph(sdfg, SubgraphView(sdfg, [reg]))
    sdfg.reset_cfg_list()
    assert "N" in sdfg.symbols and "N" in sdfg.arglist()
    assert "N" in next(n for n, parent in sdfg.all_nodes_recursive() if isinstance(n, nodes.NestedSDFG)).symbol_mapping
    sdfg.validate()

    A = np.zeros(4, dtype=np.int32)
    sdfg(A=A, N=3)
    assert np.array_equal(A, reference)


def test_a_counter_the_loop_sets_and_the_parent_reads_is_exported_not_mapped():
    """The undeclared counter ``loop_i`` is typed from its loop and handed out, since ``after`` reads it; the parent
    does not pass it in."""
    sdfg = dace.SDFG("exported_counter")
    sdfg.add_array("a", [20], dace.float64)
    sdfg.add_array("b", [1], dace.int64)
    for name in ("base", "bound", "step"):
        sdfg.add_symbol(name, dace.int32)
    loop = LoopRegion("loop", "loop_i <= bound", "loop_i", "loop_i = base", "loop_i = loop_i + step")
    sdfg.add_node(loop, is_start_block=True)
    body = loop.add_state("body", is_start_block=True)
    tasklet = body.add_tasklet("t", {}, {"o": None}, "o = 1.0")
    body.add_edge(tasklet, "o", body.add_write("a"), None, dace.Memlet("a[loop_i]"))
    after = sdfg.add_state("after")
    sdfg.add_edge(loop, after, dace.InterstateEdge(assignments={"final": "loop_i"}))
    tasklet = after.add_tasklet("t", {}, {"o": None}, "o = final")
    after.add_edge(tasklet, "o", after.add_write("b"), None, dace.Memlet("b[0]"))

    nsdfg = nested_node(nest_sdfg_subgraph(sdfg, SubgraphView(sdfg, [loop])))
    assert "loop_i" not in nsdfg.symbol_mapping and "loop_i" not in sdfg.symbols
    a, b = np.zeros(20), np.zeros(1, dtype=np.int64)
    sdfg(a=a, b=b, base=2, bound=9, step=3)
    assert np.nonzero(a)[0].tolist() == [2, 5, 8]
    assert b[0] == 11


def symbol_reset_then_loop(start_reads_k: bool) -> tuple[dace.SDFG, list]:
    """``[pre --k = 7-->] start --k = 0--> loop(k = loop_i) --> after(b[0] = k)``; the nested group is
    ``[start, loop]``."""
    sdfg = dace.SDFG(f"symbol_reset_then_loop_{int(start_reads_k)}")
    sdfg.add_array("a", [1], dace.int64)
    sdfg.add_array("b", [1], dace.int64)
    sdfg.add_symbol("k", dace.int64)
    start = sdfg.add_state("start", is_start_block=True)
    if start_reads_k:
        sdfg.add_state_before(start, "pre", is_start_block=True, assignments={"k": "7"})
        tasklet = start.add_tasklet("t", {}, {"o": None}, "o = k")
        start.add_edge(tasklet, "o", start.add_write("a"), None, dace.Memlet("a[0]"))
    loop = LoopRegion("loop", "loop_i < 5", "loop_i", "loop_i = 0", "loop_i = loop_i + 1")
    sdfg.add_node(loop)
    first = loop.add_state("first", is_start_block=True)
    loop.add_edge(first, loop.add_state("second"), dace.InterstateEdge(assignments={"k": "loop_i"}))
    sdfg.add_edge(start, loop, dace.InterstateEdge(assignments={"k": "0"}))
    after = sdfg.add_state("after")
    sdfg.add_edge(loop, after, dace.InterstateEdge())
    tasklet = after.add_tasklet("t", {}, {"o": None}, "o = k")
    after.add_edge(tasklet, "o", after.add_write("b"), None, dace.Memlet("b[0]"))
    return sdfg, [start, loop]


@pytest.mark.parametrize("start_reads_k", [False, True], ids=["dead_on_entry", "read_on_entry"])
def test_a_symbol_the_start_block_resets_is_mapped_only_when_read_first(start_reads_k):
    """``k = 0`` on the edge leaving the start block makes the parent's ``k`` dead unless the start block reads it
    first; the final ``k`` reaches ``after`` either way."""
    sdfg, blocks = symbol_reset_then_loop(start_reads_k)

    nsdfg = nested_node(nest_sdfg_subgraph(sdfg, SubgraphView(sdfg, blocks), start=blocks[0]))
    assert ("k" in nsdfg.symbol_mapping) == start_reads_k
    assert "k" not in sdfg.free_symbols
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sdfg.validate()
    a, b = np.zeros(1, dtype=np.int64), np.zeros(1, dtype=np.int64)
    sdfg(a=a, b=b)
    assert a[0] == (7 if start_reads_k else 0)
    assert b[0] == 4


def test_an_sdfg_nested_in_a_moved_state_gets_the_new_sdfg_as_parent():
    """The loop body calls a nested SDFG; after nesting the loop, that SDFG's parent is the new nested SDFG."""
    sdfg, loop = loop_through_a_scalar(read_after=False)
    body = loop.start_block
    inner = nest_state_subgraph(sdfg, body, SubgraphView(body, body.nodes()))

    outer = nested_node(nest_sdfg_subgraph(sdfg, SubgraphView(sdfg, [loop])))

    assert inner.sdfg.parent_sdfg is outer.sdfg
    sdfg.validate()
    x = np.random.default_rng(5).random(16)
    a = np.zeros(16)
    sdfg(a=a, x=x, N=16)
    assert np.allclose(a, x * 2.0 + 1.0)


def branch_assigned_symbol() -> tuple[dace.SDFG, list]:
    """``i = 1 if first else start`` in both branches of a conditional, then ``A[i] = i``; ``i`` is ``int32``."""
    sdfg = dace.SDFG("branch_assigned_symbol")
    for name in ("first", "start", "i"):
        sdfg.add_symbol(name, dace.int32)
    sdfg.add_array("A", [10], dace.float64)
    begin = sdfg.add_state("begin", is_start_block=True)
    pick = ConditionalBlock("pick")
    sdfg.add_node(pick)
    sdfg.add_edge(begin, pick, dace.InterstateEdge())
    for condition, value in (("first != 0", "1"), (None, "start")):
        branch = ControlFlowRegion(f"pick_{value}")
        pick.add_branch(condition, branch)
        branch.add_edge(
            branch.add_state(is_start_block=True), branch.add_state(), dace.InterstateEdge(assignments={"i": value})
        )
    use = sdfg.add_state("use")
    sdfg.add_edge(pick, use, dace.InterstateEdge())
    tasklet = use.add_tasklet("store", {}, {"out"}, "out = i")
    use.add_edge(tasklet, "out", use.add_write("A"), None, dace.Memlet("A[i]"))
    sdfg.add_edge(use, sdfg.add_state("end"), dace.InterstateEdge())
    return sdfg, [pick, use]


def test_a_symbol_the_subgraph_assigns_keeps_its_declared_type():
    sdfg, blocks = branch_assigned_symbol()

    state = nest_sdfg_subgraph(sdfg, SubgraphView(sdfg, blocks))

    nested = next(n for n in state.nodes() if isinstance(n, nodes.NestedSDFG))
    assert nested.sdfg.symbols["i"] == dace.int32
    code = "\n".join(obj.clean_code for obj in sdfg.generate_code())
    assert "int64_t i;" not in code
    for first, start, index in ((1, 7, 1), (0, 7, 7)):
        data = np.zeros(10)
        sdfg(A=data, first=first, start=start)
        assert data[index] == index and np.count_nonzero(data) == 1


if __name__ == "__main__":
    test_nest_oneelementmap()
    test_internal_outarray()
    test_nested_symbol_dtypes()
    test_symbolic_return()
    test_nest_cf_simple_for_loop()
    test_nest_cf_simple_while_loop()
    test_nest_cf_simple_if()
    test_nest_cf_simple_if_elif()
    test_nest_cf_simple_if_chain()
    test_nest_two_loops_with_same_variable()
    test_a_container_moved_across_the_boundary_gets_one_connector()
    test_a_transient_moves_into_the_nest_only_if_nothing_else_uses_it(False)
    test_a_transient_moves_into_the_nest_only_if_nothing_else_uses_it(True)
    test_symbol_needing_an_incoming_value_stays_an_argument()
    test_a_counter_the_loop_sets_and_the_parent_reads_is_exported_not_mapped()
    test_a_symbol_the_start_block_resets_is_mapped_only_when_read_first(False)
    test_a_symbol_the_start_block_resets_is_mapped_only_when_read_first(True)
    test_an_sdfg_nested_in_a_moved_state_gets_the_new_sdfg_as_parent()
    test_a_symbol_the_subgraph_assigns_keeps_its_declared_type()
