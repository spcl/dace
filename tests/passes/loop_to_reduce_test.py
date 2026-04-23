# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for LoopToReduce."""
import dace
from dace import memlet as mm
from dace.libraries.standard.nodes.reduce import Reduce
from dace.properties import CodeBlock
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion, LoopRegion
from dace.transformation.passes.loop_to_reduce import LoopToReduce

N = dace.symbol("N")
M = dace.symbol("M")


def _count_loops(sdfg: dace.SDFG) -> int:
    return sum(1 for n, _ in sdfg.all_nodes_recursive() if isinstance(n, LoopRegion))


def _assert_single_sum_reduce_identity_none(sdfg: dace.SDFG):
    reduces = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, Reduce)]
    assert len(reduces) == 1, reduces
    (red, ) = reduces
    assert red.wcr == "lambda a, b: a + b"
    assert red.identity is None


def test_sdfg_api_sum_reduction_is_lifted():
    """Hand-built ``for i in range(N): A[0] = A[0] + B[i]``."""
    sdfg = dace.SDFG("sum_reduce_api")
    sdfg.add_array("A", [1], dace.float64)
    sdfg.add_array("B", [N], dace.float64)

    loop = LoopRegion("loop", condition_expr="i < N", loop_var="i", initialize_expr="i = 0", update_expr="i = i + 1")
    sdfg.add_node(loop, is_start_block=True)

    body = loop.add_state("body", is_start_block=True)
    acc_r = body.add_read("A")
    arr_r = body.add_read("B")
    task = body.add_tasklet("add", {"in_a", "in_b"}, {"out"}, "out = in_a + in_b")
    acc_w = body.add_write("A")
    body.add_edge(acc_r, None, task, "in_a", mm.Memlet("A[0]"))
    body.add_edge(arr_r, None, task, "in_b", mm.Memlet("B[i]"))
    body.add_edge(task, "out", acc_w, None, mm.Memlet("A[0]"))
    sdfg.validate()
    assert _count_loops(sdfg) == 1

    lifted = LoopToReduce().apply_pass(sdfg, {})
    sdfg.validate()

    assert lifted == 1
    assert _count_loops(sdfg) == 0
    _assert_single_sum_reduce_identity_none(sdfg)


@dace.program
def _frontend_augassign_len1(A: dace.float64[1], B: dace.float64[N]):
    for i in range(N):
        A[0] += B[i]


def test_frontend_augassign_length1_array_is_lifted():
    sdfg = _frontend_augassign_len1.to_sdfg(simplify=True)
    sdfg.validate()
    assert _count_loops(sdfg) >= 1

    lifted = LoopToReduce().apply_pass(sdfg, {})
    sdfg.validate()

    assert lifted and lifted >= 1
    _assert_single_sum_reduce_identity_none(sdfg)


@dace.program
def _frontend_augassign_slice(C: dace.float64[M], B: dace.float64[N]):
    for i in range(N):
        C[3] += B[i]


def test_frontend_augassign_array_slice_is_lifted():
    """``C[k] += B[i]`` with ``C`` multi-element and ``k=3`` loop-invariant.

    Exercises the multi-element-accumulator path: the reduction's output
    memlet must target the single slice ``C[3:4]``, not ``[0:0]``.
    """
    sdfg = _frontend_augassign_slice.to_sdfg(simplify=True)
    sdfg.validate()
    assert _count_loops(sdfg) >= 1

    lifted = LoopToReduce().apply_pass(sdfg, {})
    sdfg.validate()

    assert lifted and lifted >= 1
    _assert_single_sum_reduce_identity_none(sdfg)

    reduces = [(n, g) for n, g in sdfg.all_nodes_recursive() if isinstance(n, Reduce)]
    (red, state) = reduces[0]
    (out_edge, ) = state.out_edges(red)
    assert out_edge.data.data == "C"
    assert str(out_edge.data.subset) in {"3", "3:4", "3:3"}


def _build_interstate_reduction_sdfg(offset_expr: str):
    """Loop body = 2 empty states + interstate edge ``{accum: accum + B[<offset>]}``."""
    sdfg = dace.SDFG(f"interstate_sum_{offset_expr.replace(' ', '').replace('-', 'm')}")
    sdfg.add_symbol("accum", dace.float64)
    sdfg.add_array("B", [N], dace.float64)

    pre = sdfg.add_state("pre", is_start_block=True)
    loop = LoopRegion("loop", condition_expr="i < N", loop_var="i", initialize_expr="i = 0", update_expr="i = i + 1")
    sdfg.add_node(loop)
    sdfg.add_edge(pre, loop, dace.InterstateEdge())
    s1 = loop.add_state("s1", is_start_block=True)
    s2 = loop.add_state("s2")
    loop.add_edge(s1, s2, dace.InterstateEdge(assignments={"accum": f"accum + B[{offset_expr}]"}))
    post = sdfg.add_state("post")
    sdfg.add_edge(loop, post, dace.InterstateEdge())
    return sdfg


def test_interstate_edge_direct_index_is_lifted():
    sdfg = _build_interstate_reduction_sdfg("i")
    sdfg.validate()
    assert _count_loops(sdfg) == 1

    lifted = LoopToReduce().apply_pass(sdfg, {})
    sdfg.validate()

    assert lifted == 1
    assert _count_loops(sdfg) == 0
    reduces = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, Reduce)]
    assert len(reduces) == 1
    (red, ) = reduces
    assert red.wcr == "lambda a, b: a + b"
    assert red.identity is None


def _build_conditional_minmax_sdfg(cond_expr: str):
    """body = ConditionalBlock(cond_expr) > branch of (2 empty states + {accum: B[i]})."""
    safe = "".join(c if c.isalnum() else "_" for c in cond_expr)
    sdfg = dace.SDFG(f"interstate_cond_{safe}")
    sdfg.add_symbol("accum", dace.float64)
    sdfg.add_array("B", [N], dace.float64)

    pre = sdfg.add_state("pre", is_start_block=True)
    loop = LoopRegion("loop", condition_expr="i < N", loop_var="i", initialize_expr="i = 0", update_expr="i = i + 1")
    sdfg.add_node(loop)
    sdfg.add_edge(pre, loop, dace.InterstateEdge())

    cb = ConditionalBlock("cb")
    loop.add_node(cb, is_start_block=True)
    branch = ControlFlowRegion("branch", sdfg=sdfg)
    cb.add_branch(CodeBlock(cond_expr), branch)
    s1 = branch.add_state("s1", is_start_block=True)
    s2 = branch.add_state("s2")
    branch.add_edge(s1, s2, dace.InterstateEdge(assignments={"accum": "B[i]"}))

    post = sdfg.add_state("post")
    sdfg.add_edge(loop, post, dace.InterstateEdge())
    return sdfg


def _assert_single_reduce_with_wcr(sdfg: dace.SDFG, expected_wcr: str):
    reduces = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, Reduce)]
    assert len(reduces) == 1, reduces
    (red, ) = reduces
    assert red.wcr == expected_wcr, red.wcr
    assert red.identity is None


def test_conditional_interstate_gt_lifts_to_max():
    for cond in ("B[i] > accum", "accum < B[i]", "B[i] >= accum", "accum <= B[i]"):
        sdfg = _build_conditional_minmax_sdfg(cond)
        sdfg.validate()
        assert _count_loops(sdfg) == 1
        lifted = LoopToReduce().apply_pass(sdfg, {})
        sdfg.validate()
        assert lifted == 1, cond
        assert _count_loops(sdfg) == 0
        _assert_single_reduce_with_wcr(sdfg, "lambda a, b: max(a, b)")


def test_conditional_interstate_lt_lifts_to_min():
    for cond in ("B[i] < accum", "accum > B[i]", "B[i] <= accum", "accum >= B[i]"):
        sdfg = _build_conditional_minmax_sdfg(cond)
        sdfg.validate()
        assert _count_loops(sdfg) == 1
        lifted = LoopToReduce().apply_pass(sdfg, {})
        sdfg.validate()
        assert lifted == 1, cond
        assert _count_loops(sdfg) == 0
        _assert_single_reduce_with_wcr(sdfg, "lambda a, b: min(a, b)")


def test_conditional_interstate_unrelated_array_is_not_lifted():
    """Guard compares a different array than the assignment — reject."""
    sdfg = dace.SDFG("interstate_cond_mismatched")
    sdfg.add_symbol("accum", dace.float64)
    sdfg.add_array("B", [N], dace.float64)
    sdfg.add_array("C", [N], dace.float64)
    pre = sdfg.add_state("pre", is_start_block=True)
    loop = LoopRegion("loop", condition_expr="i < N", loop_var="i", initialize_expr="i = 0", update_expr="i = i + 1")
    sdfg.add_node(loop)
    sdfg.add_edge(pre, loop, dace.InterstateEdge())
    cb = ConditionalBlock("cb")
    loop.add_node(cb, is_start_block=True)
    branch = ControlFlowRegion("branch", sdfg=sdfg)
    cb.add_branch(CodeBlock("C[i] > accum"), branch)
    s1 = branch.add_state("s1", is_start_block=True)
    s2 = branch.add_state("s2")
    branch.add_edge(s1, s2, dace.InterstateEdge(assignments={"accum": "B[i]"}))
    post = sdfg.add_state("post")
    sdfg.add_edge(loop, post, dace.InterstateEdge())
    sdfg.validate()

    assert LoopToReduce().apply_pass(sdfg, {}) is None
    assert _count_loops(sdfg) == 1


def _build_interstate_binop_sdfg(rhs_expr: str, accum_dtype=dace.int32, arr_dtype=dace.int32):
    """Loop body = 2 empty states + interstate edge ``{accum: rhs_expr}``."""
    safe = "".join(c if c.isalnum() else "_" for c in rhs_expr)
    sdfg = dace.SDFG(f"interstate_binop_{safe}")
    sdfg.add_symbol("accum", accum_dtype)
    sdfg.add_array("B", [N], arr_dtype)
    pre = sdfg.add_state("pre", is_start_block=True)
    loop = LoopRegion("loop", condition_expr="i < N", loop_var="i", initialize_expr="i = 0", update_expr="i = i + 1")
    sdfg.add_node(loop)
    sdfg.add_edge(pre, loop, dace.InterstateEdge())
    s1 = loop.add_state("s1", is_start_block=True)
    s2 = loop.add_state("s2")
    loop.add_edge(s1, s2, dace.InterstateEdge(assignments={"accum": rhs_expr}))
    post = sdfg.add_state("post")
    sdfg.add_edge(loop, post, dace.InterstateEdge())
    return sdfg


def test_interstate_edge_bitwise_or_is_lifted():
    """``{accum: accum | B[i]}`` -> Reduce with ``|`` WCR."""
    sdfg = _build_interstate_binop_sdfg("accum | B[i]")
    sdfg.validate()
    assert _count_loops(sdfg) == 1
    lifted = LoopToReduce().apply_pass(sdfg, {})
    sdfg.validate()
    assert lifted == 1
    _assert_single_reduce_with_wcr(sdfg, "lambda a, b: a | b")


def test_interstate_edge_bitwise_and_is_lifted():
    """``{accum: accum & B[i]}`` -> Reduce with ``&`` WCR."""
    sdfg = _build_interstate_binop_sdfg("accum & B[i]")
    sdfg.validate()
    assert _count_loops(sdfg) == 1
    lifted = LoopToReduce().apply_pass(sdfg, {})
    sdfg.validate()
    assert lifted == 1
    _assert_single_reduce_with_wcr(sdfg, "lambda a, b: a & b")


def test_interstate_edge_logical_or_is_lifted():
    """``{accum: accum or B[i]}`` -> Reduce with ``|`` WCR."""
    sdfg = _build_interstate_binop_sdfg("accum or B[i]")
    sdfg.validate()
    assert _count_loops(sdfg) == 1
    lifted = LoopToReduce().apply_pass(sdfg, {})
    sdfg.validate()
    assert lifted == 1
    _assert_single_reduce_with_wcr(sdfg, "lambda a, b: a | b")


def test_interstate_edge_logical_and_is_lifted():
    """``{accum: accum and B[i]}`` -> Reduce with ``&`` WCR."""
    sdfg = _build_interstate_binop_sdfg("accum and B[i]")
    sdfg.validate()
    assert _count_loops(sdfg) == 1
    lifted = LoopToReduce().apply_pass(sdfg, {})
    sdfg.validate()
    assert lifted == 1
    _assert_single_reduce_with_wcr(sdfg, "lambda a, b: a & b")


def test_tasklet_body_boolop_or_is_lifted():
    """Tasklet body ``out = a or b`` lifts to an OR reduction."""
    sdfg = dace.SDFG("tasklet_boolop_or")
    sdfg.add_array("A", [1], dace.int32)
    sdfg.add_array("B", [N], dace.int32)
    loop = LoopRegion("loop", condition_expr="i < N", loop_var="i", initialize_expr="i = 0", update_expr="i = i + 1")
    sdfg.add_node(loop, is_start_block=True)
    state = loop.add_state("body", is_start_block=True)
    r_a = state.add_read("A")
    r_b = state.add_read("B")
    w_a = state.add_write("A")
    t = state.add_tasklet("or_t", {"a", "b"}, {"o"}, "o = a or b")
    state.add_edge(r_a, None, t, "a", mm.Memlet("A[0]"))
    state.add_edge(r_b, None, t, "b", mm.Memlet("B[i]"))
    state.add_edge(t, "o", w_a, None, mm.Memlet("A[0]"))
    sdfg.validate()
    assert _count_loops(sdfg) == 1
    lifted = LoopToReduce().apply_pass(sdfg, {})
    sdfg.validate()
    assert lifted == 1
    _assert_single_reduce_with_wcr(sdfg, "lambda a, b: a | b")


def _build_any_pattern_sdfg(assign_const: str = "1", guard: str = "B[i] == 1"):
    """Mirrors FOR_l_600_c_600 (tmp_call_13): a LoopRegion whose body is a
    ConditionalBlock on an array element, with a constant-RHS assignment to
    a symbol."""
    sdfg = dace.SDFG(f"any_pattern_{assign_const}")
    sdfg.add_symbol("tmp_call_13", dace.int32)
    sdfg.add_array("B", [N], dace.int32)
    pre = sdfg.add_state("pre", is_start_block=True)
    loop = LoopRegion("loop", condition_expr="i < N", loop_var="i",
                      initialize_expr="i = 0", update_expr="i = i + 1")
    sdfg.add_node(loop)
    sdfg.add_edge(pre, loop, dace.InterstateEdge())
    cb = ConditionalBlock("cb")
    loop.add_node(cb, is_start_block=True)
    branch = ControlFlowRegion("branch", sdfg=sdfg)
    cb.add_branch(CodeBlock(guard), branch)
    s1 = branch.add_state("s1", is_start_block=True)
    s2 = branch.add_state("s2")
    branch.add_edge(s1, s2, dace.InterstateEdge(assignments={"tmp_call_13": assign_const}))
    post = sdfg.add_state("post")
    sdfg.add_edge(loop, post, dace.InterstateEdge())
    return sdfg


def test_any_pattern_not_lifted_without_permissive():
    """Default (non-permissive) mode leaves the ``any`` pattern alone because
    the lift assumes the guard array is 0/1-valued."""
    sdfg = _build_any_pattern_sdfg()
    sdfg.validate()
    assert _count_loops(sdfg) == 1
    assert LoopToReduce().apply_pass(sdfg, {}) is None
    assert _count_loops(sdfg) == 1


def test_any_pattern_lifts_to_or_in_permissive():
    """``{sym: "1"}`` gated by ``arr[f(i)] == 1`` lifts to a bitwise-OR
    reduction over ``arr`` in permissive mode."""
    sdfg = _build_any_pattern_sdfg(assign_const="1", guard="B[i] == 1")
    sdfg.validate()
    lifted = LoopToReduce(permissive=True).apply_pass(sdfg, {})
    sdfg.validate()
    assert lifted == 1
    assert _count_loops(sdfg) == 0
    _assert_single_reduce_with_wcr(sdfg, "lambda a, b: a | b")


def test_all_pattern_lifts_to_and_in_permissive():
    """``{sym: "0"}`` gated by ``arr[f(i)] == 0`` lifts to a bitwise-AND
    reduction in permissive mode."""
    sdfg = _build_any_pattern_sdfg(assign_const="0", guard="B[i] == 0")
    sdfg.validate()
    lifted = LoopToReduce(permissive=True).apply_pass(sdfg, {})
    sdfg.validate()
    assert lifted == 1
    assert _count_loops(sdfg) == 0
    _assert_single_reduce_with_wcr(sdfg, "lambda a, b: a & b")


def test_any_pattern_symbol_bridge_via_tmp_scalar():
    """When the accumulator is a symbol, ``_lift`` introduces a transient
    ``_red_tmp_<sym>`` scalar, seeds it from the symbol, and assigns the
    symbol back on the outgoing interstate edge."""
    from dace.libraries.standard.nodes.reduce import Reduce as _Reduce
    sdfg = _build_any_pattern_sdfg()
    LoopToReduce(permissive=True).apply_pass(sdfg, {})
    sdfg.validate()

    # Bridge scalar exists and is transient.
    bridge_names = [k for k in sdfg.arrays if k.startswith("_red_tmp_tmp_call_13")]
    assert len(bridge_names) == 1
    assert sdfg.arrays[bridge_names[0]].transient

    # Reduce writes to the bridge scalar.
    reduces = [(n, g) for n, g in sdfg.all_nodes_recursive() if isinstance(n, _Reduce)]
    assert len(reduces) == 1
    red, state = reduces[0]
    (out_edge, ) = state.out_edges(red)
    assert out_edge.data.data == bridge_names[0]

    # Outgoing interstate edge assigns the original symbol from the bridge.
    for e in sdfg.all_interstate_edges():
        if "tmp_call_13" in (e.data.assignments or {}):
            assert e.data.assignments["tmp_call_13"] == bridge_names[0]
            break
    else:
        raise AssertionError("no interstate edge assigning tmp_call_13 from the bridge")


if __name__ == "__main__":
    test_sdfg_api_sum_reduction_is_lifted()
    test_frontend_augassign_length1_array_is_lifted()
    test_frontend_augassign_array_slice_is_lifted()
    test_interstate_edge_direct_index_is_lifted()
    test_conditional_interstate_gt_lifts_to_max()
    test_conditional_interstate_lt_lifts_to_min()
    test_conditional_interstate_unrelated_array_is_not_lifted()
    test_interstate_edge_bitwise_or_is_lifted()
    test_interstate_edge_bitwise_and_is_lifted()
    test_interstate_edge_logical_or_is_lifted()
    test_interstate_edge_logical_and_is_lifted()
    test_tasklet_body_boolop_or_is_lifted()
    test_any_pattern_not_lifted_without_permissive()
    test_any_pattern_lifts_to_or_in_permissive()
    test_all_pattern_lifts_to_and_in_permissive()
    test_any_pattern_symbol_bridge_via_tmp_scalar()
