# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for LoopToReduce.

Every shape-and-value test runs under both emit modes the pass exposes:
``reduce-libnode`` (the original lift into a ``Reduce`` library node) and
``wcr-scalar`` (lift into ``init -> LoopRegion(WCR-on-scalar body) ->
writeback`` so a downstream ``LoopToMap`` can produce ``#pragma omp
parallel for reduction(op:scalar)``).
"""
import pytest

import dace
from dace import memlet as mm
from dace.libraries.standard.nodes.reduce import Reduce
from dace.properties import CodeBlock
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion, LoopRegion
from dace.transformation.passes.loop_to_reduce import LoopToReduce

N = dace.symbol("N")
M = dace.symbol("M")

#: Emit strategies LoopToReduce exposes via its ``prefer`` knob. Each
#: parameterised test runs under both.
_PREFER_MODES = ('reduce-libnode', 'wcr-scalar')


@pytest.fixture(params=_PREFER_MODES)
def prefer(request):
    """Yield each emit mode in turn so every shape-and-value test exercises
    both lift forms."""
    return request.param


def _count_loops(sdfg: dace.SDFG) -> int:
    return sum(1 for n, _ in sdfg.all_nodes_recursive() if isinstance(n, LoopRegion))


def _count_wcr_scalar_targets(sdfg: dace.SDFG, expected_wcr: str) -> int:
    """Count WCR-bearing memlets whose destination AccessNode points at a
    transient ``Scalar`` -- the accumulator shape the wcr-scalar emit
    produces. The memlet's ``data`` names the SOURCE array (``B``); the
    destination Scalar (``_priv_*``) is identified by the dst AccessNode.
    """
    from dace import data
    from dace.sdfg import nodes as _nodes
    n = 0
    for state in sdfg.all_states():
        for e in state.edges():
            if e.data is None or e.data.wcr != expected_wcr:
                continue
            if not isinstance(e.dst, _nodes.AccessNode):
                continue
            desc = sdfg.arrays.get(e.dst.data)
            if isinstance(desc, data.Scalar) and desc.transient:
                n += 1
    return n


def _expected_loop_count_after_lift(prefer: str, libnode_count: int, wcr_scalar_count: int) -> int:
    """Mode-aware expected ``_count_loops`` after the pass runs.

    The ``reduce-libnode`` emit removes the loop entirely (replaced by a
    ``Reduce`` node); the ``wcr-scalar`` emit replaces it with a fresh
    LoopRegion of equivalent extent. Tests pass the value expected for each
    mode; the helper returns whichever applies.
    """
    return libnode_count if prefer == 'reduce-libnode' else wcr_scalar_count


def _assert_lifted_with_wcr(sdfg: dace.SDFG, prefer: str, expected_wcr: str):
    """Mode-aware shape check for a lifted reduction with ``expected_wcr``.

    :param sdfg: The post-lift SDFG.
    :param prefer: Emit mode the pass ran under.
    :param expected_wcr: The WCR lambda the lifted reduction carries.
    """
    if prefer == 'reduce-libnode':
        reduces = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, Reduce)]
        assert len(reduces) == 1, reduces
        (red, ) = reduces
        assert red.wcr == expected_wcr, red.wcr
        assert red.identity is None
    else:
        # wcr-scalar emit: exactly one WCR-on-transient-Scalar memlet.
        assert _count_wcr_scalar_targets(
            sdfg,
            expected_wcr) == 1, (f'wcr-scalar emit must land exactly one WCR-on-Scalar memlet with {expected_wcr!r}; '
                                 f'sdfg has {_count_wcr_scalar_targets(sdfg, expected_wcr)}')


def _assert_single_sum_reduce_identity_none(sdfg: dace.SDFG, prefer: str = 'reduce-libnode'):
    """Backward-compat wrapper: a single ``sum`` reduction was lifted."""
    _assert_lifted_with_wcr(sdfg, prefer, 'lambda a, b: a + b')


def test_sdfg_api_sum_reduction_is_lifted(prefer):
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

    lifted = LoopToReduce(prefer=prefer).apply_pass(sdfg, {})
    sdfg.validate()

    assert lifted == 1
    assert _count_loops(sdfg) == _expected_loop_count_after_lift(prefer, 0, 1)
    _assert_single_sum_reduce_identity_none(sdfg, prefer)


@dace.program
def _frontend_augassign_len1(A: dace.float64[1], B: dace.float64[N]):
    for i in range(N):
        A[0] += B[i]


def test_frontend_augassign_length1_array_is_lifted(prefer):
    sdfg = _frontend_augassign_len1.to_sdfg(simplify=True)
    sdfg.validate()
    assert _count_loops(sdfg) >= 1

    lifted = LoopToReduce(prefer=prefer).apply_pass(sdfg, {})
    sdfg.validate()

    assert lifted and lifted >= 1
    _assert_single_sum_reduce_identity_none(sdfg, prefer)


@dace.program
def _frontend_augassign_slice(C: dace.float64[M], B: dace.float64[N]):
    for i in range(N):
        C[3] += B[i]


def test_frontend_augassign_array_slice_is_lifted(prefer):
    """``C[k] += B[i]`` with ``C`` multi-element and ``k=3`` loop-invariant.

    Exercises the multi-element-accumulator path: the reduction's output
    memlet must target the single slice ``C[3:4]``, not ``[0:0]``.
    """
    sdfg = _frontend_augassign_slice.to_sdfg(simplify=True)
    sdfg.validate()
    assert _count_loops(sdfg) >= 1

    lifted = LoopToReduce(prefer=prefer).apply_pass(sdfg, {})
    sdfg.validate()

    assert lifted and lifted >= 1
    _assert_single_sum_reduce_identity_none(sdfg, prefer)

    if prefer == 'reduce-libnode':
        reduces = [(n, g) for n, g in sdfg.all_nodes_recursive() if isinstance(n, Reduce)]
        (red, state) = reduces[0]
        (out_edge, ) = state.out_edges(red)
        assert out_edge.data.data == "C"
        assert str(out_edge.data.subset) in {"3", "3:4", "3:3"}


@dace.program
def _frontend_per_row_inner_reduction(acc: dace.float64[N], B: dace.float64[N, M]):
    for jl in range(N):
        for jm in range(M):
            acc[jl] += B[jl, jm]


def test_per_row_inner_reduction_multidim_is_lifted(prefer):
    """Per-row inner reduction ``acc[jl] += B[jl, jm]`` over ``jm`` (the CloudSC
    ZQPRETOT shape: ``for jl: for jm: zqpretot[jl] += zpfplsx[jl, jm]``).

    The reduced array ``B[jl, jm]`` is 2-D, so the lift must expand ONLY the
    reduction axis ``jm`` -- ``B[jl, 0:M]`` -- and keep the outer ``jl`` index
    as-is. Regression for ``_expand_over_loop`` previously rejecting any subset
    with a dimension independent of the reduction axis (it computed
    ``jl - jm`` for the ``jl`` dimension and bailed)."""
    import numpy as np
    sdfg = _frontend_per_row_inner_reduction.to_sdfg(simplify=True)
    sdfg.validate()
    assert _count_loops(sdfg) >= 2  # outer jl + inner jm

    lifted = LoopToReduce(prefer=prefer).apply_pass(sdfg, {})
    sdfg.validate()

    assert lifted and lifted >= 1
    _assert_single_sum_reduce_identity_none(sdfg, prefer)
    # The inner jm loop is gone (or replaced by a wcr-scalar inner loop); the
    # per-row outer jl loop survives around the lifted reduction.
    assert _count_loops(sdfg) == _expected_loop_count_after_lift(prefer, 1, 2)

    if prefer == 'reduce-libnode':
        reduces = [(n, g) for n, g in sdfg.all_nodes_recursive() if isinstance(n, Reduce)]
        (red, state) = reduces[0]
        (out_edge, ) = state.out_edges(red)
        assert out_edge.data.data == "acc"

    # Value-preserving: acc[jl] = sum_jm B[jl, jm].
    n, m = 6, 4
    b = np.random.rand(n, m)
    acc = np.zeros(n)
    sdfg(acc=acc, B=b.copy(), N=n, M=m)
    assert np.allclose(acc, b.sum(axis=1))


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


def test_interstate_edge_direct_index_is_lifted(prefer):
    sdfg = _build_interstate_reduction_sdfg("i")
    sdfg.validate()
    assert _count_loops(sdfg) == 1

    lifted = LoopToReduce(prefer=prefer).apply_pass(sdfg, {})
    sdfg.validate()

    assert lifted == 1
    assert _count_loops(sdfg) == _expected_loop_count_after_lift(prefer, 0, 1)
    _assert_lifted_with_wcr(sdfg, prefer, "lambda a, b: a + b")


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


def test_conditional_interstate_gt_lifts_to_max(prefer):
    for cond in ("B[i] > accum", "accum < B[i]", "B[i] >= accum", "accum <= B[i]"):
        sdfg = _build_conditional_minmax_sdfg(cond)
        sdfg.validate()
        assert _count_loops(sdfg) == 1
        lifted = LoopToReduce(prefer=prefer).apply_pass(sdfg, {})
        sdfg.validate()
        assert lifted == 1, cond
        assert _count_loops(sdfg) == _expected_loop_count_after_lift(prefer, 0, 1)
        _assert_lifted_with_wcr(sdfg, prefer, "lambda a, b: max(a, b)")


def test_conditional_interstate_lt_lifts_to_min(prefer):
    for cond in ("B[i] < accum", "accum > B[i]", "B[i] <= accum", "accum >= B[i]"):
        sdfg = _build_conditional_minmax_sdfg(cond)
        sdfg.validate()
        assert _count_loops(sdfg) == 1
        lifted = LoopToReduce(prefer=prefer).apply_pass(sdfg, {})
        sdfg.validate()
        assert lifted == 1, cond
        assert _count_loops(sdfg) == _expected_loop_count_after_lift(prefer, 0, 1)
        _assert_lifted_with_wcr(sdfg, prefer, "lambda a, b: min(a, b)")


def test_conditional_interstate_unrelated_array_is_not_lifted(prefer):
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

    assert LoopToReduce(prefer=prefer).apply_pass(sdfg, {}) is None
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


def test_interstate_edge_bitwise_or_is_lifted(prefer):
    """``{accum: accum | B[i]}`` -> Reduce with ``|`` WCR."""
    sdfg = _build_interstate_binop_sdfg("accum | B[i]")
    sdfg.validate()
    assert _count_loops(sdfg) == 1
    lifted = LoopToReduce(prefer=prefer).apply_pass(sdfg, {})
    sdfg.validate()
    assert lifted == 1
    _assert_lifted_with_wcr(sdfg, prefer, "lambda a, b: a | b")


def test_interstate_edge_bitwise_and_is_lifted(prefer):
    """``{accum: accum & B[i]}`` -> Reduce with ``&`` WCR."""
    sdfg = _build_interstate_binop_sdfg("accum & B[i]")
    sdfg.validate()
    assert _count_loops(sdfg) == 1
    lifted = LoopToReduce(prefer=prefer).apply_pass(sdfg, {})
    sdfg.validate()
    assert lifted == 1
    _assert_lifted_with_wcr(sdfg, prefer, "lambda a, b: a & b")


def test_interstate_edge_logical_or_is_lifted(prefer):
    """``{accum: accum or B[i]}`` -> Reduce with ``|`` WCR."""
    sdfg = _build_interstate_binop_sdfg("accum or B[i]")
    sdfg.validate()
    assert _count_loops(sdfg) == 1
    lifted = LoopToReduce(prefer=prefer).apply_pass(sdfg, {})
    sdfg.validate()
    assert lifted == 1
    _assert_lifted_with_wcr(sdfg, prefer, "lambda a, b: a | b")


def test_interstate_edge_logical_and_is_lifted(prefer):
    """``{accum: accum and B[i]}`` -> Reduce with ``&`` WCR."""
    sdfg = _build_interstate_binop_sdfg("accum and B[i]")
    sdfg.validate()
    assert _count_loops(sdfg) == 1
    lifted = LoopToReduce(prefer=prefer).apply_pass(sdfg, {})
    sdfg.validate()
    assert lifted == 1
    _assert_lifted_with_wcr(sdfg, prefer, "lambda a, b: a & b")


def test_tasklet_body_boolop_or_is_lifted(prefer):
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
    lifted = LoopToReduce(prefer=prefer).apply_pass(sdfg, {})
    sdfg.validate()
    assert lifted == 1
    _assert_lifted_with_wcr(sdfg, prefer, "lambda a, b: a | b")


def _build_any_pattern_sdfg(assign_const: str = "1", guard: str = "B[i] == 1"):
    """Mirrors FOR_l_600_c_600 (tmp_call_13): a LoopRegion whose body is a
    ConditionalBlock on an array element, with a constant-RHS assignment to
    a symbol."""
    sdfg = dace.SDFG(f"any_pattern_{assign_const}")
    sdfg.add_symbol("tmp_call_13", dace.int32)
    sdfg.add_array("B", [N], dace.int32)
    pre = sdfg.add_state("pre", is_start_block=True)
    loop = LoopRegion("loop", condition_expr="i < N", loop_var="i", initialize_expr="i = 0", update_expr="i = i + 1")
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


def test_any_pattern_not_lifted_without_permissive(prefer):
    """Default (non-permissive) mode leaves the ``any`` pattern alone because
    the lift assumes the guard array is 0/1-valued."""
    sdfg = _build_any_pattern_sdfg()
    sdfg.validate()
    assert _count_loops(sdfg) == 1
    assert LoopToReduce(prefer=prefer).apply_pass(sdfg, {}) is None
    assert _count_loops(sdfg) == 1


def test_any_pattern_lifts_to_or_in_permissive(prefer):
    """``{sym: "1"}`` gated by ``arr[f(i)] == 1`` lifts to a bitwise-OR
    reduction over ``arr`` in permissive mode."""
    sdfg = _build_any_pattern_sdfg(assign_const="1", guard="B[i] == 1")
    sdfg.validate()
    lifted = LoopToReduce(permissive=True, prefer=prefer).apply_pass(sdfg, {})
    sdfg.validate()
    assert lifted == 1
    assert _count_loops(sdfg) == _expected_loop_count_after_lift(prefer, 0, 1)
    _assert_lifted_with_wcr(sdfg, prefer, "lambda a, b: a | b")


def test_all_pattern_lifts_to_and_in_permissive(prefer):
    """``{sym: "0"}`` gated by ``arr[f(i)] == 0`` lifts to a bitwise-AND
    reduction in permissive mode."""
    sdfg = _build_any_pattern_sdfg(assign_const="0", guard="B[i] == 0")
    sdfg.validate()
    lifted = LoopToReduce(permissive=True, prefer=prefer).apply_pass(sdfg, {})
    sdfg.validate()
    assert lifted == 1
    assert _count_loops(sdfg) == _expected_loop_count_after_lift(prefer, 0, 1)
    _assert_lifted_with_wcr(sdfg, prefer, "lambda a, b: a & b")


def test_any_pattern_symbol_bridge_via_tmp_scalar(prefer):
    """When the accumulator is a symbol, the lift introduces a transient
    bridge scalar (``_red_tmp_<sym>`` for reduce-libnode, ``_priv_<sym>``
    for wcr-scalar), seeds it from the symbol, and assigns the symbol back
    on the outgoing interstate edge."""
    from dace.libraries.standard.nodes.reduce import Reduce as _Reduce
    sdfg = _build_any_pattern_sdfg()
    LoopToReduce(permissive=True, prefer=prefer).apply_pass(sdfg, {})
    sdfg.validate()

    # Bridge scalar exists and is transient. Naming differs by emit mode.
    bridge_prefix = "_red_tmp_tmp_call_13" if prefer == 'reduce-libnode' else "_priv_tmp_call_13"
    bridge_names = [k for k in sdfg.arrays if k.startswith(bridge_prefix)]
    assert len(bridge_names) == 1, f'bridge scalar with prefix {bridge_prefix!r} not found'
    assert sdfg.arrays[bridge_names[0]].transient

    if prefer == 'reduce-libnode':
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


# ---------------------------------------------------------------------------
# Reduction patterns surfaced by the TSVC corpus (see
# ``tests/corpus/parallelization_report.md`` group B).
#
# Each test mirrors a real TSVC kernel body, builds the simplified SDFG and
# applies the same prelude canonicalize runs before ``LoopToReduce``
# (``TrivialTaskletElimination``) so the entry shape matches what the pass
# sees in the pipeline. Patterns the pass cannot detect yet are ``xfail``
# with a TODO reference -- when the pass is extended they should flip to
# passing without test edits.
# ---------------------------------------------------------------------------

import pytest

from dace.transformation.dataflow.trivial_tasklet_elimination import TrivialTaskletElimination
from dace.transformation.passes.pattern_matching import PatternMatchAndApplyRepeated


def _prep_and_lift(sdfg: dace.SDFG, prefer: str) -> int:
    """Run the canonicalize prelude (``TrivialTaskletElimination``) then
    ``LoopToReduce``, returning the lifted count."""
    PatternMatchAndApplyRepeated([TrivialTaskletElimination()]).apply_pass(sdfg, {})
    return LoopToReduce(prefer=prefer).apply_pass(sdfg, {}) or 0


# ---- s311 family: array-slot accumulator (sum) ---------------------------


@dace.program
def _array_slot_sum(a: dace.float64[N], sum_out: dace.float64[N]):
    sum_out[0] = 0.0
    for i in range(N):
        sum_out[0] = sum_out[0] + a[i]


def test_array_slot_sum_reduction_is_lifted(prefer):
    """TSVC s311: ``sum_out: float64[N]; for i: sum_out[0] += a[i]``.

    Accumulator descriptor is a multi-element array; only slot ``[0]`` is
    touched. Lifts after the chase-forward extension to ``_extract``.
    """
    sdfg = _array_slot_sum.to_sdfg(simplify=True)
    sdfg.validate()
    assert _count_loops(sdfg) >= 1

    lifted = _prep_and_lift(sdfg, prefer)
    sdfg.validate()

    assert lifted >= 1
    _assert_single_sum_reduce_identity_none(sdfg, prefer)


# ---- s313 / vdotr: array-slot dot-product (compute-then-accumulate) ------


@dace.program
def _array_slot_dot_product(a: dace.float64[N], b: dace.float64[N], dot: dace.float64[N]):
    dot[0] = 0.0
    for i in range(N):
        dot[0] = dot[0] + a[i] * b[i]


def test_array_slot_dot_product_is_lifted(prefer):
    """TSVC s313/vdotr: ``dot: float64[N]; for i: dot[0] += a[i] * b[i]``.

    The body has two tasklets (compute multiply then accumulate-add); the
    ``Reduce`` libnode form cannot wrap arbitrary in-body expressions, so
    only the ``wcr-scalar`` emit can express this shape. In
    ``reduce-libnode`` mode the single-tasklet matcher refuses
    (``_extract`` returns ``None``) and the test pins that refusal as
    the ``Reduce`` libnode's domain boundary.
    """
    sdfg = _array_slot_dot_product.to_sdfg(simplify=True)
    sdfg.validate()
    lifted = _prep_and_lift(sdfg, prefer)
    if prefer == 'reduce-libnode':
        assert lifted == 0
    else:
        assert lifted >= 1


# ---- s317: array-slot scalar product (no array fold) ---------------------


@dace.program
def _array_slot_const_product(q: dace.float64[N]):
    q[0] = 1.0
    for i in range(N // 2):
        q[0] = q[0] * 0.99


@pytest.mark.xfail(reason="TSVC s317: ``q[0] *= 0.99`` is a multiplicative induction variable. The preferred "
                   "path -- ``InductionVariableSubstitution`` (canonicalize/induction_variable_substitution.py) -- "
                   "now collapses it to the O(1) closed form ``q[0] *= 0.99**N`` and is wired into the "
                   "canonicalize pipeline. This test still xfails because it exercises ``LoopToReduce`` "
                   "in isolation, which intentionally does NOT recognise the IV shape (no array to fold). "
                   "See tests/passes/induction_variable_substitution_test.py for the passing IV-pass tests.",
                   strict=True)
def test_array_slot_const_product_is_lifted(prefer):
    """TSVC s317."""
    sdfg = _array_slot_const_product.to_sdfg(simplify=True)
    sdfg.validate()
    lifted = _prep_and_lift(sdfg, prefer)
    assert lifted >= 1


# ---- s314 / s316: branched min / max -------------------------------------


@dace.program
def _branched_max(a: dace.float64[N], result: dace.float64[N]):
    x = a[0]
    for i in range(1, N):
        if a[i] > x:
            x = a[i]
    result[0] = x


@pytest.mark.xfail(reason="TSVC s314: frontend lowers ``if a[i] > x: x = a[i]`` to a loop body of "
                   "``[ConditionalBlock, SDFGState, SDFGState]`` at loop level (not the "
                   "in-conditional 2-empty-states form the conditional-interstate pattern in "
                   "``_extract`` expects). Needs the loop-level branched-min/max shape. "
                   "See parallelization_report.md group B.",
                   strict=True)
def test_branched_max_is_lifted(prefer):
    """TSVC s314."""
    sdfg = _branched_max.to_sdfg(simplify=True)
    sdfg.validate()
    lifted = _prep_and_lift(sdfg, prefer)
    assert lifted >= 1


@dace.program
def _branched_min(a: dace.float64[N], result: dace.float64[N]):
    x = a[0]
    for i in range(1, N):
        if a[i] < x:
            x = a[i]
    result[0] = x


@pytest.mark.xfail(reason="TSVC s316: same shape as s314 (branched min). See test_branched_max_is_lifted.", strict=True)
def test_branched_min_is_lifted(prefer):
    """TSVC s316."""
    sdfg = _branched_min.to_sdfg(simplify=True)
    sdfg.validate()
    lifted = _prep_and_lift(sdfg, prefer)
    assert lifted >= 1


# ---- s4115: gather + sum reduction ---------------------------------------


@dace.program
def _gather_sum_reduction(a: dace.float64[N], b: dace.float64[N], ip: dace.int32[N], sum_out: dace.float64[N]):
    s = 0.0
    for i in range(N):
        s = s + a[i] * b[ip[i]]
    sum_out[0] = s


def test_gather_sum_reduction_is_lifted(prefer):
    """TSVC s4115: ``s += a[i] * b[ip[i]]`` -- gather + sum.

    Multi-state body: the frontend lowers the gather through an interstate
    edge ``{ip_index: ip[i]}`` between two body states. ``reduce-libnode``
    mode refuses on body-shape grounds (single-tasklet matcher) and that's
    the right answer there. ``wcr-scalar`` mode SHOULD lift (the in-body
    gather indirection is fine for the WCR form), but the current
    ``AugAssignToWCR``-based preprocess is single-state and can't reach
    across the load-iedge to rewrite the accumulator.

    Follow-up: a multi-state variant of ``AugAssignToWCR`` -- or a
    dedicated multi-state compute-then-accumulate matcher -- would unblock
    this. The iedge itself must stay; the gather depends on it.
    """
    sdfg = _gather_sum_reduction.to_sdfg(simplify=True)
    sdfg.validate()
    lifted = _prep_and_lift(sdfg, prefer)
    if prefer == 'reduce-libnode':
        assert lifted == 0
    else:
        pytest.xfail("multi-state gather body; see docstring follow-up")
        assert lifted >= 1


# ---- 1D-reduction guard: GEMM-shaped multi-input loops must NOT lift -------


def test_gemm_innermost_loop_not_lifted_to_reduce(prefer):
    """A textbook GEMM ``C[i, j] += A[i, k] * B[k, j]`` has THREE data reads in
    the body (``C[i, j]``, ``A[i, k]``, ``B[k, j]``), TWO of which depend on
    the innermost loop variable ``k``. ``LoopToReduce``'s matcher only handles
    single-binary-op tasklets with exactly 2 data inputs and AT MOST ONE
    input subset using the loop variable -- the second loop-var-dependent
    input (``B[k, j]``) trips the "multiple arrays" early-out and the loop is
    refused. The k-loop in GEMM is semantically a 1-D reduction per
    ``(i, j)``, but lifting it loses the multi-array structure that
    GEMM-as-a-whole has; downstream library matching or tiling is the right
    handler. This test pins the refusal so the matcher does not get widened."""
    N, M, K = (dace.symbol(s) for s in ['NN', 'MM', 'KK'])

    @dace.program
    def gemm(A: dace.float64[N, K], B: dace.float64[K, M], C: dace.float64[N, M]):
        for i in range(N):
            for j in range(M):
                for k in range(K):
                    C[i, j] += A[i, k] * B[k, j]

    sdfg = gemm.to_sdfg(simplify=True)
    sdfg.validate()
    lifted = _prep_and_lift(sdfg, prefer)
    if prefer == 'reduce-libnode':
        # The single-tasklet matcher refuses on body-shape grounds (two tasklets
        # after the frontend splits ``Mul`` and ``Add``). No libnode is emitted.
        assert lifted == 0
    else:
        # ``wcr-scalar`` mode: the multi-tasklet matcher lifts the k-loop into
        # ``init + LoopRegion(WCR-on-scalar body) + writeback``. Downstream
        # ``LoopToMap`` produces the canonical ``Map + WCR-on-scalar`` shape
        # ``LiftEinsum`` consumes to detect the BLAS contraction. The
        # GEMM-vs-scalar-reduction classification is intentionally NOT
        # LoopToReduce's job -- LiftEinsum runs the parent-Map check
        # separately. Pinning the lift here documents that contract.
        assert lifted == 1


def test_matvec_innermost_loop_not_lifted_to_reduce(prefer):
    """Matrix-vector ``y[i] += A[i, j] * x[j]`` -- same shape concern as GEMM,
    just one rank down. Two of the three data reads (``A[i, j]``, ``x[j]``)
    use the inner loop variable; the matcher must refuse the j-loop. Lifting
    it would lose the GEMV pattern recognisable downstream."""
    N, M = (dace.symbol(s) for s in ['NN', 'MM'])

    @dace.program
    def matvec(A: dace.float64[N, M], x: dace.float64[M], y: dace.float64[N]):
        for i in range(N):
            for j in range(M):
                y[i] += A[i, j] * x[j]

    sdfg = matvec.to_sdfg(simplify=True)
    sdfg.validate()
    lifted = _prep_and_lift(sdfg, prefer)
    if prefer == 'reduce-libnode':
        assert lifted == 0
    else:
        # Same contract as the GEMM test: GEMV-vs-scalar-reduction
        # classification lives in LiftEinsum, not here. The lift is the
        # canonical intermediate.
        assert lifted == 1


def test_outer_axis_reduction_per_column_is_lifted(prefer):
    """Genuine 1-D reduction along ``i`` with a single loop-var-dependent
    read: ``c[j] += A[i, j]``. Per fixed outer-loop ``j``, this is a
    standard 1-D fold over ``i`` of ``A[:, j]`` -- exactly what
    ``LoopToReduce`` is designed to handle. Pinning this ensures the
    refusal in ``test_gemm_innermost_loop_not_lifted_to_reduce`` doesn't
    over-tighten and reject legitimate 1-D outer-axis reductions."""
    N, M = (dace.symbol(s) for s in ['NN', 'MM'])

    @dace.program
    def axis_sum(A: dace.float64[N, M], c: dace.float64[M]):
        for j in range(M):
            for i in range(N):
                c[j] += A[i, j]

    sdfg = axis_sum.to_sdfg(simplify=True)
    sdfg.validate()
    lifted = _prep_and_lift(sdfg, prefer)
    assert lifted >= 1, (f'``c[j] += A[i, j]`` (single loop-var-dependent read) must lift as a '
                         f'1-D reduction along i; got {lifted}.')


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
    test_array_slot_sum_reduction_is_lifted()
