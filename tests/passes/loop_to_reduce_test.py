# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for LoopToReduce.

Every shape-and-value test runs under both emit modes the pass exposes:
``reduce-libnode`` (the original lift into a ``Reduce`` library node) and
``wcr-scalar`` (lift into ``init -> LoopRegion(WCR-on-scalar body) ->
writeback`` so a downstream ``LoopToMap`` can produce ``#pragma omp
parallel for reduction(op:scalar)``).
"""
import numpy as np
import pytest

import dace
from dace import memlet as mm
from dace.libraries.standard.nodes.reduce import Reduce
from dace.properties import CodeBlock
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion, LoopRegion
from dace.transformation.passes.loop_to_reduce import (AccumulatorCopyChainToWCR, LoopToReduce, RetargetWCRAccumulator)

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


#: Emit mode for MATCHER-only cases. ``apply_pass`` runs ``_extract`` -- and every refusal it
#: makes -- BEFORE it dispatches on ``prefer``, so a case whose accept/refuse decision is taken
#: there re-runs an identical lift (or an identical no-op) under the second mode. Those cases fix
#: the mode here instead of sweeping it; the emit side of every WCR op stays swept by the tests
#: that do vary the mode: ``+`` by ``test_sdfg_api_sum_reduction_is_lifted``, ``max`` by
#: ``test_branched_max_is_lifted`` / ``test_conditional_interstate_gt_lifts_to_max``, ``min`` by
#: the corresponding min pair, ``|``/``&`` by the permissive any/all pair, and the
#: symbol-accumulator bridge by ``test_any_pattern_symbol_bridge_via_tmp_scalar``.
#: ``reduce-libnode`` is the stronger single mode for refusals: only it can emit a ``Reduce``,
#: and only under it does a wrong lift change ``_count_loops``.
_MATCHER_ONLY_MODE = 'reduce-libnode'


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
    """Mode-aware expected ``_count_loops`` after the pass.

    ``reduce-libnode`` removes the loop (replaced by a ``Reduce`` node); ``wcr-scalar``
    replaces it with a fresh LoopRegion of equal extent. Tests pass the value for each
    mode; helper returns whichever applies.
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


def test_frontend_augassign_length1_array_is_lifted():
    """Frontend ``A[0] += B[i]`` with ``A`` length-1: pins the ``scalar_like`` accumulator arm.

    Matcher-only -- that arm is decided in ``_extract``. The frontend-staged, embedded-loop
    emit it produces is swept over both modes by ``test_frontend_augassign_array_slice_is_lifted``.

    Compiles and RUNS the lifted SDFG: ``validate()`` alone accepts a lift that left the
    consumed loop variable free, because a free symbol is only fatal inside a NestedSDFG --
    at top level it survives until ``SDFG.arglist`` raises at codegen. Shape assertions on
    their own therefore cannot tell a correct lift from a broken one.
    """
    prefer = _MATCHER_ONLY_MODE
    sdfg = _frontend_augassign_len1.to_sdfg(simplify=True)
    sdfg.validate()
    assert _count_loops(sdfg) >= 1

    lifted = LoopToReduce(prefer=prefer).apply_pass(sdfg, {})
    sdfg.validate()

    assert lifted and lifted >= 1
    _assert_single_sum_reduce_identity_none(sdfg, prefer)

    n = 12
    rng = np.random.default_rng(21)
    b = rng.standard_normal(n)
    a = np.zeros(1)
    sdfg(A=a, B=b.copy(), N=n)
    assert np.isclose(a[0], b.sum()), f'lifted reduction diverged from the oracle: {a[0]} vs {b.sum()}'


@dace.program
def _frontend_augassign_slice(C: dace.float64[M], B: dace.float64[N]):
    for i in range(N):
        C[3] += B[i]


def test_frontend_augassign_array_slice_is_lifted(prefer):
    """``C[k] += B[i]`` with ``C`` multi-element and ``k=3`` loop-invariant.

    Exercises the multi-element-accumulator path: the reduction's output
    memlet must target the single slice ``C[3:4]``, not ``[0:0]``.

    Compiles and RUNS the result (see ``test_frontend_augassign_length1_array_is_lifted`` for
    why shape assertions alone cannot reject a lift that freed the loop variable), and pins
    that every OTHER slot of ``C`` is left untouched.
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

    n, m = 12, 8
    rng = np.random.default_rng(22)
    b = rng.standard_normal(n)
    c = np.zeros(m)
    sdfg(C=c, B=b.copy(), N=n, M=m)
    assert np.isclose(c[3], b.sum()), f'lifted reduction diverged from the oracle: {c[3]} vs {b.sum()}'
    assert np.allclose(np.delete(c, 3), 0.0), f'the lift wrote outside the accumulator slot: {c}'


@dace.program
def _frontend_per_row_inner_reduction(acc: dace.float64[N], B: dace.float64[N, M]):
    for jl in range(N):
        for jm in range(M):
            acc[jl] += B[jl, jm]


def test_per_row_inner_reduction_multidim_is_lifted(prefer):
    """Per-row inner reduction ``acc[jl] += B[jl, jm]`` over ``jm`` (CloudSC ZQPRETOT:
    ``for jl: for jm: zqpretot[jl] += zpfplsx[jl, jm]``). ``B`` is 2-D, so the lift
    expands ONLY the reduction axis ``jm`` (``B[jl, 0:M]``) and keeps ``jl`` as-is.
    Regression: ``_expand_over_loop`` used to reject any subset with a dim independent
    of the reduction axis (computed ``jl - jm`` for ``jl`` and bailed)."""
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


def _build_interstate_reduction_sdfg(offset_expr: str, iv_dtype=None):
    """Loop body = 2 empty states + interstate edge ``{accum: accum + B[<offset>]}``.

    :param iv_dtype: When given, DECLARE the loop variable with this dtype. Left ``None`` the
                     symbol is never added, so it resolves to the default type -- the only
                     difference between the dtype-mismatch test and its control.
    """
    sdfg = dace.SDFG(f"interstate_sum_{offset_expr.replace(' ', '').replace('-', 'm')}")
    sdfg.add_symbol("accum", dace.float64)
    sdfg.add_array("B", [N], dace.float64)
    if iv_dtype is not None:
        sdfg.add_symbol("i", iv_dtype)

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


def test_interstate_edge_lifts_with_non_default_dtype_loop_variable():
    """The loop variable's DECLARED dtype must not decide whether the lift fires.

    ``symbol._hashable_content`` folds the dtype into sympy identity, so the ``i`` taken from
    ``sdfg.symbols`` (``int64`` here) is a different object than the default-typed ``i`` that
    ``pystr_to_symbolic`` leaves in the parsed index ``B[i]``. Subtracting one from the other
    kept BOTH spellings, the ``offset.has(loop_var_sym)`` guard then saw the loop variable in
    what should have been a constant offset, and ``_extract`` refused a reduction it should
    lift -- a predicate answering False because two spellings differ, not because the shape
    was unsafe. ``test_interstate_edge_direct_index_is_lifted`` is the control: same fixture,
    only the declared dtype differs.
    """
    prefer = _MATCHER_ONLY_MODE
    sdfg = _build_interstate_reduction_sdfg("i", iv_dtype=dace.int64)
    sdfg.validate()
    assert _count_loops(sdfg) == 1

    lifted = LoopToReduce(prefer=prefer).apply_pass(sdfg, {})
    sdfg.validate()

    assert lifted == 1, 'a non-default-typed loop variable must not block the lift'
    assert _count_loops(sdfg) == _expected_loop_count_after_lift(prefer, 0, 1)
    _assert_lifted_with_wcr(sdfg, prefer, "lambda a, b: a + b")
    # ... and it folds the whole iteration space, not one element at a freed symbol.
    ((red, state), ) = [(n, g) for n, g in sdfg.all_nodes_recursive() if isinstance(n, Reduce)]
    assert str(state.in_edges(red)[0].data.subset) == '0:N'


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
    """``>``/``>=`` guard, either operand order -> ``max``.

    Paired with ``test_branched_max_is_lifted``, which pins the same ``max`` WCR under
    ``wcr-scalar``: ``_cmp_to_wcr`` decides before the ``prefer`` dispatch and neither lift
    branches on the WCR op, so the two modes need not be crossed with all four guard forms.
    """
    prefer = 'reduce-libnode'
    for cond in ("B[i] > accum", "accum < B[i]", "B[i] >= accum", "accum <= B[i]"):
        sdfg = _build_conditional_minmax_sdfg(cond)
        sdfg.validate()
        assert _count_loops(sdfg) == 1
        lifted = LoopToReduce(prefer=prefer).apply_pass(sdfg, {})
        sdfg.validate()
        assert lifted == 1, cond
        assert _count_loops(sdfg) == _expected_loop_count_after_lift(prefer, 0, 1)
        _assert_lifted_with_wcr(sdfg, prefer, "lambda a, b: max(a, b)")


def test_conditional_interstate_lt_lifts_to_min():
    """``<``/``<=`` guard, either operand order -> ``min``.

    Runs ``wcr-scalar`` so this pair covers both emit modes; ``min`` under ``reduce-libnode``
    is pinned by ``test_branched_min_is_lifted``.
    """
    prefer = 'wcr-scalar'
    for cond in ("B[i] < accum", "accum > B[i]", "B[i] <= accum", "accum >= B[i]"):
        sdfg = _build_conditional_minmax_sdfg(cond)
        sdfg.validate()
        assert _count_loops(sdfg) == 1
        lifted = LoopToReduce(prefer=prefer).apply_pass(sdfg, {})
        sdfg.validate()
        assert lifted == 1, cond
        assert _count_loops(sdfg) == _expected_loop_count_after_lift(prefer, 0, 1)
        _assert_lifted_with_wcr(sdfg, prefer, "lambda a, b: min(a, b)")


def test_conditional_interstate_unrelated_array_is_not_lifted():
    """Guard compares a different array than the assignment -- reject (in ``_extract``)."""
    prefer = _MATCHER_ONLY_MODE
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


# The four iedge-binop tests below pin ``_extract``'s expression CLASSIFICATION: ``|`` parses to
# ``bitwise_or`` and ``or`` to ``OR`` (likewise ``&``/``and``), four distinct sympy classes across
# two isinstance tuples. That is a matcher decision, and the ``|``/``&`` emit under both modes is
# already pinned by the permissive any/all pair, so they fix the emit mode.


def test_interstate_edge_bitwise_or_is_lifted():
    """``{accum: accum | B[i]}`` -> Reduce with ``|`` WCR."""
    prefer = _MATCHER_ONLY_MODE
    sdfg = _build_interstate_binop_sdfg("accum | B[i]")
    sdfg.validate()
    assert _count_loops(sdfg) == 1
    lifted = LoopToReduce(prefer=prefer).apply_pass(sdfg, {})
    sdfg.validate()
    assert lifted == 1
    _assert_lifted_with_wcr(sdfg, prefer, "lambda a, b: a | b")


def test_interstate_edge_bitwise_and_is_lifted():
    """``{accum: accum & B[i]}`` -> Reduce with ``&`` WCR."""
    prefer = _MATCHER_ONLY_MODE
    sdfg = _build_interstate_binop_sdfg("accum & B[i]")
    sdfg.validate()
    assert _count_loops(sdfg) == 1
    lifted = LoopToReduce(prefer=prefer).apply_pass(sdfg, {})
    sdfg.validate()
    assert lifted == 1
    _assert_lifted_with_wcr(sdfg, prefer, "lambda a, b: a & b")


def test_interstate_edge_logical_or_is_lifted():
    """``{accum: accum or B[i]}`` -> Reduce with ``|`` WCR."""
    prefer = _MATCHER_ONLY_MODE
    sdfg = _build_interstate_binop_sdfg("accum or B[i]")
    sdfg.validate()
    assert _count_loops(sdfg) == 1
    lifted = LoopToReduce(prefer=prefer).apply_pass(sdfg, {})
    sdfg.validate()
    assert lifted == 1
    _assert_lifted_with_wcr(sdfg, prefer, "lambda a, b: a | b")


def test_interstate_edge_logical_and_is_lifted():
    """``{accum: accum and B[i]}`` -> Reduce with ``&`` WCR."""
    prefer = _MATCHER_ONLY_MODE
    sdfg = _build_interstate_binop_sdfg("accum and B[i]")
    sdfg.validate()
    assert _count_loops(sdfg) == 1
    lifted = LoopToReduce(prefer=prefer).apply_pass(sdfg, {})
    sdfg.validate()
    assert lifted == 1
    _assert_lifted_with_wcr(sdfg, prefer, "lambda a, b: a & b")


def test_tasklet_body_boolop_or_is_lifted():
    """Tasklet body ``out = a or b`` lifts to an OR reduction (``_BOOLOP_TO_WCR`` arm).

    Matcher-only: same hand-built accumulator shape as ``test_sdfg_api_sum_reduction_is_lifted``,
    which sweeps both emit modes; only the tasklet-body classification differs.
    """
    prefer = _MATCHER_ONLY_MODE
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


def _build_any_pattern_sdfg(assign_const: str = "1", guard: str = "B[i] == 1", iv_dtype=None):
    """Mirrors FOR_l_600_c_600 (tmp_call_13): a LoopRegion whose body is a
    ConditionalBlock on an array element, with a constant-RHS assignment to
    a symbol.

    :param iv_dtype: When given, DECLARE the loop variable with this dtype (see
                     ``_build_interstate_reduction_sdfg``)."""
    sdfg = dace.SDFG(f"any_pattern_{assign_const}")
    sdfg.add_symbol("tmp_call_13", dace.int32)
    sdfg.add_array("B", [N], dace.int32)
    if iv_dtype is not None:
        sdfg.add_symbol("i", iv_dtype)
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


def test_any_pattern_not_lifted_without_permissive():
    """Default (non-permissive) mode leaves the ``any`` pattern alone because
    the lift assumes the guard array is 0/1-valued. Refused in ``_extract``."""
    prefer = _MATCHER_ONLY_MODE
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


def test_any_pattern_lifts_with_non_default_dtype_loop_variable():
    """Same dtype-identity hazard as the interstate path, on the permissive any/all matcher.

    ``_extract_any_pattern`` located the reduction axis with ``a.has(loop_var_sym)``. Against
    an ``int64``-declared ``i`` that answered False for EVERY axis, so the matcher bailed at
    ``axis_for_iter is None`` and the lift never fired.
    ``test_any_pattern_lifts_to_or_in_permissive`` is the default-dtype control.
    """
    prefer = _MATCHER_ONLY_MODE
    sdfg = _build_any_pattern_sdfg(assign_const="1", guard="B[i] == 1", iv_dtype=dace.int64)
    sdfg.validate()

    lifted = LoopToReduce(permissive=True, prefer=prefer).apply_pass(sdfg, {})
    sdfg.validate()

    assert lifted == 1, 'a non-default-typed loop variable must not block the any/all lift'
    assert _count_loops(sdfg) == _expected_loop_count_after_lift(prefer, 0, 1)
    _assert_lifted_with_wcr(sdfg, prefer, "lambda a, b: a | b")
    ((red, state), ) = [(n, g) for n, g in sdfg.all_nodes_recursive() if isinstance(n, Reduce)]
    assert str(state.in_edges(red)[0].data.subset) == '0:N'


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
# Reduction patterns from the TSVC corpus (group B).
#
# Each test mirrors a real TSVC kernel body and runs the same prelude
# (``TrivialTaskletElimination``) before ``LoopToReduce`` so the entry shape matches
# the pipeline. Patterns the pass can't detect yet are ``xfail`` with a TODO -> flip to
# passing without test edits when the pass is extended.
# ---------------------------------------------------------------------------

import pytest

from dace.transformation.dataflow.trivial_tasklet_elimination import TrivialTaskletElimination
from dace.transformation.dataflow.wcr_conversion import WCRToAugAssign
from dace.transformation.passes.pattern_matching import PatternMatchAndApplyRepeated


def _prep_and_lift(sdfg: dace.SDFG, prefer: str) -> int:
    """Run the canonicalize prelude then ``LoopToReduce``, returning the lifted count.

    ``LoopToReduce`` is a pure matcher, so the caller owns every normalization step -- the
    same ones the pipeline's ``reduction_to_wcr_map`` stage spells out around it."""
    PatternMatchAndApplyRepeated([TrivialTaskletElimination()]).apply_pass(sdfg, {})
    PatternMatchAndApplyRepeated([WCRToAugAssign()]).apply_pass(sdfg, {})
    if prefer != 'wcr-scalar':
        return LoopToReduce(prefer=prefer).apply_pass(sdfg, {}) or 0
    lifted = LoopToReduce(prefer=prefer).apply_pass(sdfg, {}) or 0
    AccumulatorCopyChainToWCR().apply_pass(sdfg, {})
    return lifted + (RetargetWCRAccumulator().apply_pass(sdfg, {}) or 0)


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

    Body has two tasklets (multiply then accumulate-add); the ``Reduce`` libnode can't
    wrap arbitrary in-body expressions, so only ``wcr-scalar`` expresses this. In
    ``reduce-libnode`` the single-tasklet matcher refuses (``_extract`` -> ``None``);
    the test pins that as the libnode's domain boundary.
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


def test_array_slot_const_product_not_lifted_iv(prefer):
    """TSVC s317 ``q[0] *= 0.99`` is a geometric induction variable, NOT a reduction: there
    is no array to fold, so ``LoopToReduce`` alone (either mode) must refuse it. The O(1)
    closed form ``q[0] *= 0.99**N`` is produced by ``InductionVariableSubstitution`` (run
    before LoopToReduce in the pipeline) -- see
    ``test_geometric_iv_handled_by_induction_pass_not_loop_to_reduce`` and
    tests/passes/induction_variable_substitution_test.py for the passing IV-pass coverage.
    """
    sdfg = _array_slot_const_product.to_sdfg(simplify=True)
    sdfg.validate()
    lifted = _prep_and_lift(sdfg, prefer)
    assert lifted == 0


# ---- s314 / s316: branched min / max -------------------------------------


@dace.program
def _branched_max(a: dace.float64[N], result: dace.float64[N]):
    x = a[0]
    for i in range(1, N):
        if a[i] > x:
            x = a[i]
    result[0] = x


def test_branched_max_is_lifted():
    """TSVC s314: ``for i: if a[i] > x: x = a[i]``.

    Frontend lowers the masked update into ``[ConditionalBlock, cond_prep_state,
    post_state]`` joined by iedges ``{arr_sym: a[i]}`` and ``{guard_sym: arr_sym > x}``.
    ``max`` is idempotent so the conditional is redundant at the WCR level
    (``acc = max(acc, a[i])`` matches the masked semantics whether the guard fires or
    not). Both emit modes are correct: this one runs ``wcr-scalar`` -- whose per-iteration
    index inversion has to undo the ``range(1, N)`` offset, the ``a[1:N]`` case
    ``_lift_wcr_scalar`` names s314 for -- and ``test_branched_min_is_lifted`` runs
    ``reduce-libnode`` on the same matcher.
    """
    prefer = 'wcr-scalar'
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


def test_branched_min_is_lifted():
    """TSVC s316: same shape as s314 with ``<`` (branched min), run under ``reduce-libnode``
    so the s314/s316 pair still covers both emit modes."""
    prefer = 'reduce-libnode'
    sdfg = _branched_min.to_sdfg(simplify=True)
    sdfg.validate()
    lifted = _prep_and_lift(sdfg, prefer)
    assert lifted >= 1


@dace.program
def _strided_branched_min(a: dace.float64[N], result: dace.float64[N]):
    x = a[1]
    for i in range(1, N, 2):
        if a[i] < x:
            x = a[i]
    result[0] = x


def test_strided_branched_min_folds_only_visited_elements():
    """A STRIDED branched min must fold only the elements the loop actually visits.

    ``_extract_branched_minmax`` widened the per-iteration read but let ``_expand_over_loop``'s
    ``loop_stride`` fall back to its default of 1, so ``range(1, N, 2)`` produced the reduce
    subset ``a[1:N]`` instead of ``a[1:N:2]`` and every skipped even slot silently entered the
    fold. Unlike the leaked-symbol failures this one is a pure VALUE bug -- the SDFG stays
    valid and compiles clean -- so the fixture parks the global minimum on the even slots the
    loop never reads: a correct fold returns the odd-slot minimum, the unstrided one returns
    ``-100``.
    """
    sdfg = _strided_branched_min.to_sdfg(simplify=True)
    sdfg.validate()
    lifted = _prep_and_lift(sdfg, 'reduce-libnode')
    sdfg.validate()
    assert lifted >= 1

    steps = sorted(
        str(rng[2]) for red, state in [(n, g) for n, g in sdfg.all_nodes_recursive() if isinstance(n, Reduce)]
        for rng in state.in_edges(red)[0].data.subset.ndrange())
    assert steps == ['2'], f'the reduce must read only the stride-2 slots; got {steps}'

    n = 16
    a = np.array([-100.0 if k % 2 == 0 else float(k) for k in range(n)])
    result = np.zeros(n)
    sdfg(a=a.copy(), result=result, N=n)
    assert np.isclose(result[0], a[1:n:2].min()), (f'the lifted strided min folded elements the loop never visits: '
                                                   f'got {result[0]}, expected {a[1:n:2].min()}')


# ---- s4115: gather + sum reduction ---------------------------------------


@dace.program
def _gather_sum_reduction(a: dace.float64[N], b: dace.float64[N], ip: dace.int32[N], sum_out: dace.float64[N]):
    s = 0.0
    for i in range(N):
        s = s + a[i] * b[ip[i]]
    sum_out[0] = s


def test_gather_sum_reduction_is_lifted(prefer):
    """TSVC s4115: ``s += a[i] * b[ip[i]]`` -- gather + sum.

    Multi-state body: frontend lowers the gather through an iedge ``{ip_index: ip[i]}``
    between two body states. ``reduce-libnode`` refuses on body-shape grounds (can't
    express the in-body gather). ``wcr-scalar`` lifts via the multi-state-chain matcher:
    the iedge stays (the gather depends on it), the compute-then-accumulate chain in the
    second state is rewritten to drop the carry input + add WCR on the privatised scalar,
    and init/writeback states seed/drain the original ``s``.
    """
    sdfg = _gather_sum_reduction.to_sdfg(simplify=True)
    sdfg.validate()
    lifted = _prep_and_lift(sdfg, prefer)
    if prefer == 'reduce-libnode':
        assert lifted == 0
    else:
        assert lifted >= 1


# ---- 1D-reduction guard: GEMM-shaped multi-input loops must NOT lift -------


def test_gemm_innermost_loop_not_lifted_to_reduce(prefer):
    """Textbook GEMM ``C[i, j] += A[i, k] * B[k, j]`` has THREE data reads
    (``C[i, j]``, ``A[i, k]``, ``B[k, j]``), TWO depending on the inner var ``k``.
    ``LoopToReduce``'s matcher handles single-binary-op tasklets with exactly 2 data
    inputs and AT MOST ONE input subset using the loop var -- ``B[k, j]`` trips the
    "multiple arrays" early-out and the loop is refused. The k-loop is semantically a
    1-D reduction per ``(i, j)``, but lifting loses GEMM's multi-array structure;
    downstream library matching/tiling is the right handler. Pins the refusal so the
    matcher isn't widened."""
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
        # ``wcr-scalar``: the multi-tasklet matcher lifts the k-loop into
        # ``init + LoopRegion(WCR-on-scalar body) + writeback``. Downstream ``LoopToMap``
        # produces the ``Map + WCR-on-scalar`` shape ``LiftEinsum`` consumes to detect the
        # BLAS contraction. GEMM-vs-scalar-reduction classification is LiftEinsum's job
        # (separate parent-Map check), not LoopToReduce's. Pins the lift documenting that.
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
    """Genuine 1-D reduction along ``i`` with a single loop-var-dependent read:
    ``c[j] += A[i, j]``. Per fixed ``j``, a standard 1-D fold over ``i`` of ``A[:, j]``
    -- exactly what ``LoopToReduce`` handles. Pins that the GEMM-test refusal doesn't
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


# ---- masked compound update with gather (multi-state, wcr-scalar only) ------


@dace.program
def _masked_compound_gather(a: dace.float64[N], idx: dace.int32[N], cond: dace.int32[N], sum_out: dace.float64[1]):
    s = 0.0
    for i in range(N):
        if cond[i] != 0:
            s = s + a[idx[i]]
    sum_out[0] = s


def test_masked_compound_gather_is_lifted(prefer):
    """``for i: if cond[i]: s += a[idx[i]]`` -- non-idempotent (``+``) masked reduction
    with an in-body gather. ``reduce-libnode`` refuses (can't express the in-body mask or
    gather). ``wcr-scalar`` lifts via the chain matcher: the conditional sits inside the
    privatised body and the WCR write fires only when the guard is true.
    """
    import numpy as np
    sdfg = _masked_compound_gather.to_sdfg(simplify=True)
    sdfg.validate()
    lifted = _prep_and_lift(sdfg, prefer)
    if prefer == 'reduce-libnode':
        assert lifted == 0
    else:
        assert lifted >= 1
    sdfg.validate()
    n = 16
    rng = np.random.default_rng(0)
    a_arr = rng.standard_normal(n)
    idx_arr = rng.permutation(n).astype(np.int32)
    cond_arr = rng.integers(0, 2, size=n).astype(np.int32)
    expected = float(sum(a_arr[idx_arr[i]] for i in range(n) if cond_arr[i] != 0))
    sum_out = np.zeros(1)
    sdfg(a=a_arr.copy(), idx=idx_arr.copy(), cond=cond_arr.copy(), sum_out=sum_out, N=n)
    assert np.isclose(sum_out[0], expected), f'got {sum_out[0]}, expected {expected}'


# ---- interleaved two-accumulator single loop --------------------------------


@dace.program
def _interleaved_two_accum(A: dace.float64[N], evens_out: dace.float64[1], odds_out: dace.float64[1]):
    evens = 0.0
    odds = 0.0
    for i in range(N // 2):
        evens = evens + A[2 * i]
        odds = odds + A[2 * i + 1]
    evens_out[0] = evens
    odds_out[0] = odds


def test_interleaved_two_accumulator_partial_lift(prefer):
    """Single loop with two independent accumulators
    (``evens += A[2*i]; odds += A[2*i+1]``).

    LoopToReduce's matchers are single-accumulator by design; it lifts at most one, the
    other stays sequential. Correctness holds (partial privatisation is
    semantics-preserving). Parallelising both needs ``LoopFission`` to split into two
    single-accumulator loops first (see
    ``test_interleaved_two_accumulator_lifts_both_after_loop_fission``).
    """
    import numpy as np
    sdfg = _interleaved_two_accum.to_sdfg(simplify=True)
    sdfg.validate()
    lifted = _prep_and_lift(sdfg, prefer)
    if prefer == 'reduce-libnode':
        assert lifted == 0
    else:
        # At least one accumulator lifted; the matcher is single-accumulator
        # so it returns after the first chain match.
        assert lifted >= 1
    sdfg.validate()
    n = 16
    rng = np.random.default_rng(0)
    A_arr = rng.standard_normal(n)
    evens_out = np.zeros(1)
    odds_out = np.zeros(1)
    sdfg(A=A_arr.copy(), evens_out=evens_out, odds_out=odds_out, N=n)
    assert np.isclose(evens_out[0], float(A_arr[::2].sum()))
    assert np.isclose(odds_out[0], float(A_arr[1::2].sum()))


def test_interleaved_two_accumulator_lifts_both_after_loop_fission():
    """``LoopFission`` splits the interleaved 2-accumulator loop into two
    single-accumulator loops; ``LoopToReduce`` then handles each independently.

    Pipeline order for the interleaved case: interleaved -> fission -> reduce. Post-fission
    the lift target is two parallel WCR-scalar loops that downstream ``LoopToMap`` maps
    independently with ``reduction(+:_priv_X)`` clauses.
    """
    import numpy as np
    from dace.transformation.passes.loop_fission import LoopFission
    sdfg = _interleaved_two_accum.to_sdfg(simplify=True)
    sdfg.validate()
    PatternMatchAndApplyRepeated([TrivialTaskletElimination()]).apply_pass(sdfg, {})
    fissioned = LoopFission().apply_pass(sdfg, {})
    assert fissioned, 'LoopFission must split the interleaved 2-accumulator loop'
    n_loops_after_fission = sum(1 for n, _ in sdfg.all_nodes_recursive() if isinstance(n, LoopRegion))
    assert n_loops_after_fission >= 2, f'expected >= 2 loops after fission, got {n_loops_after_fission}'
    lifted = LoopToReduce(prefer='wcr-scalar').apply_pass(sdfg, {})
    assert lifted == 2, f'both fissioned loops must lift; got {lifted}'
    sdfg.validate()
    rng = np.random.default_rng(0)
    A_arr = rng.standard_normal(n_loops_after_fission * 8)
    n_run = A_arr.size
    evens_out = np.zeros(1)
    odds_out = np.zeros(1)
    sdfg(A=A_arr.copy(), evens_out=evens_out, odds_out=odds_out, N=n_run)
    assert np.isclose(evens_out[0], float(A_arr[::2].sum()))
    assert np.isclose(odds_out[0], float(A_arr[1::2].sum()))


def test_interleaved_dual_strided_lifts_to_two_reduce_nodes():
    """Interleaved dual strided reduction ``evens += A[2*i]; odds += A[2*i+1]``, split by
    ``LoopFission`` into two single-accumulator loops, lifts under ``reduce-libnode`` to
    TWO ``Reduce`` library nodes, each reading a STRIDE-2 subset (``A[0:N:2]`` /
    ``A[1:N:2]``). "Split statements, then detect both as strided reductions": the read
    index ``coeff*i + off`` folds into the reduce subset's step, which the ``Reduce``
    expansions honor.
    """
    import numpy as np
    from dace.transformation.passes.loop_fission import LoopFission
    from dace.transformation.dataflow.trivial_tasklet_elimination import TrivialTaskletElimination
    sdfg = _interleaved_two_accum.to_sdfg(simplify=True)
    sdfg.validate()
    PatternMatchAndApplyRepeated([TrivialTaskletElimination()]).apply_pass(sdfg, {})
    assert LoopFission().apply_pass(sdfg, {}), 'LoopFission must split the interleaved 2-accumulator loop'
    lifted = LoopToReduce(prefer='reduce-libnode').apply_pass(sdfg, {})
    assert lifted == 2, f'both fissioned strided loops must lift to Reduce nodes; got {lifted}'
    sdfg.validate()

    # Two Reduce library nodes, each over a stride-2 input subset.
    reduces = [(n, st) for n, st in sdfg.all_nodes_recursive() if isinstance(n, Reduce)]
    assert len(reduces) == 2, f'expected 2 Reduce nodes, got {len(reduces)}'
    steps = sorted(str(rng[2]) for red, st in reduces for rng in st.in_edges(red)[0].data.subset.ndrange())
    assert steps == ['2', '2'], f'each Reduce must read a stride-2 subset; got {steps}'

    n = 16
    rng = np.random.default_rng(0)
    A_arr = rng.standard_normal(n)
    evens_out = np.zeros(1)
    odds_out = np.zeros(1)
    sdfg(A=A_arr.copy(), evens_out=evens_out, odds_out=odds_out, N=n)
    assert np.isclose(evens_out[0], float(A_arr[::2].sum()))
    assert np.isclose(odds_out[0], float(A_arr[1::2].sum()))


# ---- split two strided loops (independent reductions) -----------------------


@dace.program
def _split_two_strided(A: dace.float64[N], evens_out: dace.float64[1], odds_out: dace.float64[1]):
    evens = 0.0
    odds = 0.0
    for i in range(N // 2):
        evens = evens + A[2 * i]
    for i in range(N // 2):
        odds = odds + A[2 * i + 1]
    evens_out[0] = odds  # intentionally swapped to catch fixture mistakes
    odds_out[0] = odds
    evens_out[0] = evens


def test_geometric_iv_handled_by_induction_pass_not_loop_to_reduce():
    """TSVC s317 ``q[0] *= 0.99`` is NOT a LoopToReduce target -- it's an
    induction-variable closed form ``q[0] *= 0.99**N`` handled by
    ``InductionVariableSubstitution`` (run BEFORE LoopToReduce in the pipeline). After
    ``TTE + IVS`` the loop is gone, so a subsequent LoopToReduce has nothing to lift. The
    ``test_array_slot_const_product_not_lifted_iv`` test pins the inverse: LoopToReduce alone
    must NOT recognise the IV shape.
    """
    import numpy as np
    from dace.transformation.passes.canonicalize.induction_variable_substitution import (InductionVariableSubstitution)

    sdfg = _array_slot_const_product.to_sdfg(simplify=True)
    sdfg.validate()
    assert _count_loops(sdfg) >= 1

    PatternMatchAndApplyRepeated([TrivialTaskletElimination()]).apply_pass(sdfg, {})
    substituted = InductionVariableSubstitution().apply_pass(sdfg, {})
    assert substituted == 1, f'IVS must close-form the geometric IV; got {substituted}'
    assert _count_loops(sdfg) == 0, 'IVS must eliminate the loop entirely'

    # A subsequent LoopToReduce is a no-op (no loops left).
    lifted = LoopToReduce(prefer='wcr-scalar').apply_pass(sdfg, {})
    assert lifted is None or lifted == 0, f'no loops left to lift, got {lifted}'
    sdfg.validate()

    n = 16
    q = np.zeros(n)
    sdfg(q=q, N=n)
    expected = 1.0 * (0.99**(n // 2))
    assert np.isclose(q[0], expected), f'got {q[0]}, expected {expected}'


def test_split_two_strided_loops_both_lift(prefer):
    """Two strided reductions (``evens += A[2*i]``, ``odds += A[2*i+1]``) in two separate
    loops. Each is single-accumulator with stride-2 access; BOTH modes lift both. The read
    index ``coeff*i + off`` folds into a strided reduce subset (``A[0:N:2]`` / ``A[1:N:2]``):
    ``reduce-libnode`` consumes it directly (pure/OpenMP expansions honor the subset step),
    ``wcr-scalar`` keeps a per-iter ``A[2*i]`` WCR write.
    """
    import numpy as np
    sdfg = _split_two_strided.to_sdfg(simplify=True)
    sdfg.validate()
    lifted = _prep_and_lift(sdfg, prefer)
    assert lifted == 2, f'{prefer} must lift both strided loops; got {lifted}'
    sdfg.validate()
    n = 16
    rng = np.random.default_rng(0)
    A_arr = rng.standard_normal(n)
    evens_out = np.zeros(1)
    odds_out = np.zeros(1)
    sdfg(A=A_arr.copy(), evens_out=evens_out, odds_out=odds_out, N=n)
    assert np.isclose(evens_out[0], float(A_arr[::2].sum()))
    assert np.isclose(odds_out[0], float(A_arr[1::2].sum()))


def test_wcr_scalar_refuses_scan_shape_recurrence():
    """``LoopToReduce(prefer='wcr-scalar')`` must NOT lift a loop-carried recurrence
    (``b[i] = b[i+1] + a[i]``) to a WCR write, even after ``LoopToScan`` rewrote the body
    into a scan shape with a similar ``out = in_carry + in_value`` tasklet.

    Regression: a previous version called ``AugAssignToWCR(permissive=True)`` to catch
    multi-tasklet compute-then-accumulate shapes (s313/s4115). The permissive matcher
    couldn't tell a scan from a reduction and rewrote the recurrence into a WCR write;
    downstream parallelisation then lost the carried dependence -> off-by-one answer
    (``b[0]`` = ``sum(a)`` not ``1.0 + sum(a)``). Pins ``permissive=False`` for the
    ``AugAssignToWCR`` inside ``AccumulatorCopyChainToWCR``.
    """
    import numpy as np
    from dace.transformation.passes.canonicalize.pipeline import _build_stages
    NN = dace.symbol("NN")

    @dace.program
    def _recurrence_down(a: dace.float64[NN], b: dace.float64[NN]):
        for i in range(NN - 2, -1, -1):
            b[i] = b[i + 1] + a[i]

    sdfg = _recurrence_down.to_sdfg(simplify=True)
    # Run the canonicalize pre-stages so LoopToScan rewrites the body before
    # LoopToReduce(wcr-scalar) sees it -- the exact shape the regression
    # surfaced under.
    for label, pass_obj in _build_stages():
        if label == 'reduction_to_wcr_map':
            break
        if hasattr(pass_obj, 'apply_pass'):
            try:
                pass_obj.apply_pass(sdfg, {})
            except Exception:
                pass

    # Now run the wcr-scalar lift with its full prelude. It must NOT create any WCR-bearing
    # write on ``b`` (the recurrence target).
    LoopToReduce(prefer='wcr-scalar').apply_pass(sdfg, {})
    AccumulatorCopyChainToWCR().apply_pass(sdfg, {})
    RetargetWCRAccumulator().apply_pass(sdfg, {})
    sdfg.validate()
    bad_wcr = [
        e for st in sdfg.all_states() for e in st.edges()
        if e.data is not None and e.data.wcr is not None and e.data.data == 'b'
    ]
    assert not bad_wcr, (f'LoopToReduce(wcr-scalar) wrongly placed a WCR write on the recurrence '
                         f'accumulator ``b``: {bad_wcr}')

    n = 21
    rng = np.random.default_rng(7)
    a_arr = rng.random(n)
    expected = np.zeros(n)
    expected[n - 1] = 1.0
    for i in range(n - 2, -1, -1):
        expected[i] = expected[i + 1] + a_arr[i]

    out = np.zeros(n)
    out[n - 1] = 1.0
    sdfg(a=a_arr.copy(), b=out, NN=n)
    assert np.allclose(out, expected), f'recurrence value-preservation broke: got {out[:3]}, expected {expected[:3]}'


def _build_scalarised_scan_writeback(n_sym=N):
    """``c = A[0]; for i in 0..N-2: c = c + A[i]; A[i+1] = c`` -- a prefix-sum scan whose
    vertical carry has been scalarised into ``c`` (the shape CloudSC's flux integral
    ``PFLUX(jk+1) = PFLUX(jk) + term`` collapses to after the pipeline stamps out the
    intra-iteration scalar intermediates). The accumulate ``c = c + A[i]`` looks like a plain
    reduction, but the running value is written BACK into the folded array ``A`` at ``i+1``
    every iteration -- a per-iteration output a single ``Reduce`` cannot reproduce.
    """
    sdfg = dace.SDFG("scalarised_scan_writeback")
    sdfg.add_array("A", [n_sym], dace.float64)
    sdfg.add_scalar("c", dace.float64, transient=True)

    pre = sdfg.add_state("pre", is_start_block=True)
    pre.add_edge(pre.add_read("A"), None, pre.add_write("c"), None, mm.Memlet("A[0]"))

    loop = LoopRegion("loop",
                      condition_expr="i < N - 1",
                      loop_var="i",
                      initialize_expr="i = 0",
                      update_expr="i = i + 1")
    sdfg.add_node(loop)
    sdfg.add_edge(pre, loop, dace.InterstateEdge())

    body = loop.add_state("body", is_start_block=True)
    t = body.add_tasklet("acc", {"cin", "ain"}, {"cout"}, "cout = cin + ain")
    c_w = body.add_write("c")
    body.add_edge(body.add_read("c"), None, t, "cin", mm.Memlet("c[0]"))
    body.add_edge(body.add_read("A"), None, t, "ain", mm.Memlet("A[i]"))
    body.add_edge(t, "cout", c_w, None, mm.Memlet("c[0]"))
    # scan output: the running carry written back into the folded array at the next level.
    body.add_edge(c_w, None, body.add_write("A"), None, mm.Memlet("A[i + 1]"))

    post = sdfg.add_state("post")
    sdfg.add_edge(loop, post, dace.InterstateEdge())
    sdfg.validate()
    return sdfg


def test_scalarised_scan_writeback_into_folded_array_not_lifted():
    """A scalarised prefix-sum that writes its running carry BACK into the folded array
    (``c = c + A[i]; A[i + 1] = c``) must NOT be lifted to a ``Reduce``.

    Regression: ``_extract`` used to exempt the folded ``array`` from its "writes only the
    accumulator" guard (``allowed = {accum, array, ...}``), so a writeback into that array
    slipped through and the vertical scan was collapsed to a single ``Reduce`` -- the CloudSC
    flux-integral (PFPLSL/PFHPSL) miscompile, ~2.5% error, because a ``Reduce`` emits one final
    value instead of the running per-level sequence. The array is now NOT exempt; the writeback
    is recognised as a scan output and the loop is left for ``LoopToScan``.

    The refusal is ``_extract``'s, taken before ``apply_pass`` dispatches on ``prefer``, so the
    mode is fixed (``_MATCHER_ONLY_MODE``); ``wcr-scalar`` refusing a scan shape is separately
    pinned -- with the full retarget prelude -- by ``test_wcr_scalar_refuses_scan_shape_recurrence``.
    """
    import numpy as np
    from dace.libraries.standard.nodes.reduce import Reduce

    prefer = _MATCHER_ONLY_MODE
    sdfg = _build_scalarised_scan_writeback()
    sdfg.validate()
    assert _count_loops(sdfg) == 1

    lifted = LoopToReduce(prefer=prefer).apply_pass(sdfg, {})
    sdfg.validate()

    assert not lifted, f'scalarised scan writeback must not lift; got {lifted}'
    assert _count_loops(sdfg) == 1, 'the scan loop must survive for LoopToScan'
    assert not [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, Reduce)], 'no Reduce may be emitted'

    # Value preservation: the surviving loop still computes the in-place prefix scan.
    n = 12
    rng = np.random.default_rng(3)
    a = rng.standard_normal(n)
    expected = a.copy()
    c = expected[0]
    for i in range(n - 1):
        c = c + expected[i]
        expected[i + 1] = c
    got = a.copy()
    sdfg(A=got, N=n)
    assert np.allclose(got, expected), f'scan value broke: got {got[:4]}, expected {expected[:4]}'


def _build_transient_scan_writeback(n_sym=N):
    """``c = A[0]; for i in 0..N-2: c = c + A[i]; ZP[i+1] = c`` where ``ZP`` is a TRANSIENT buffer.
    Same prefix-scan-into-writeback shape as ``_build_scalarised_scan_writeback`` but the running
    carry is stamped into a transient (not the folded array). 881e55d79 refused only NON-transient
    writebacks, so this slipped through and collapsed to a ``Reduce`` -- exactly gpu_scc's vertical
    flux double-buffer, whose ``ZPFPLSX`` carry is a transient scratch array.
    """
    sdfg = dace.SDFG("transient_scan_writeback")
    sdfg.add_array("A", [n_sym], dace.float64)
    sdfg.add_array("ZP", [n_sym], dace.float64, transient=True)
    sdfg.add_scalar("c", dace.float64, transient=True)
    pre = sdfg.add_state("pre", is_start_block=True)
    pre.add_edge(pre.add_read("A"), None, pre.add_write("c"), None, mm.Memlet("A[0]"))
    loop = LoopRegion("loop",
                      condition_expr="i < N - 1",
                      loop_var="i",
                      initialize_expr="i = 0",
                      update_expr="i = i + 1")
    sdfg.add_node(loop)
    sdfg.add_edge(pre, loop, dace.InterstateEdge())
    body = loop.add_state("body", is_start_block=True)
    t = body.add_tasklet("acc", {"cin", "ain"}, {"cout"}, "cout = cin + ain")
    cw = body.add_write("c")
    body.add_edge(body.add_read("c"), None, t, "cin", mm.Memlet("c[0]"))
    body.add_edge(body.add_read("A"), None, t, "ain", mm.Memlet("A[i]"))
    body.add_edge(t, "cout", cw, None, mm.Memlet("c[0]"))
    body.add_edge(cw, None, body.add_write("ZP"), None, mm.Memlet("ZP[i + 1]"))  # carry -> transient buffer
    post = sdfg.add_state("post")
    sdfg.add_edge(loop, post, dace.InterstateEdge())
    sdfg.validate()
    return sdfg


def _build_double_buffer_scan_writeback(n_sym=N):
    """The CloudSC k-caching double-buffer: ``c = c + A[i]; ZP[(i+1)%2] = c`` where ``ZP`` is a 2-slot
    TRANSIENT buffer and the write slot toggles per level via a loop-iedge symbol (``jkip1``). The
    per-level carry is stamped into the ring buffer; a single ``Reduce`` drops the sequence. This is
    the ``ZPFPLSX[jk_ip1]`` flux shape whose *moving transient* slot the plain non-transient guard
    could not see.
    """
    sdfg = dace.SDFG("double_buffer_scan_writeback")
    sdfg.add_array("A", [n_sym], dace.float64)
    sdfg.add_array("ZP", [2], dace.float64, transient=True)
    sdfg.add_scalar("c", dace.float64, transient=True)
    pre = sdfg.add_state("pre", is_start_block=True)
    pre.add_edge(pre.add_read("A"), None, pre.add_write("c"), None, mm.Memlet("A[0]"))
    loop = LoopRegion("loop",
                      condition_expr="i < N - 1",
                      loop_var="i",
                      initialize_expr="i = 0",
                      update_expr="i = i + 1")
    sdfg.add_node(loop)
    sdfg.add_edge(pre, loop, dace.InterstateEdge())
    body = loop.add_state("body", is_start_block=True)
    t = body.add_tasklet("acc", {"cin", "ain"}, {"cout"}, "cout = cin + ain")
    cw = body.add_write("c")
    body.add_edge(body.add_read("c"), None, t, "cin", mm.Memlet("c[0]"))
    body.add_edge(body.add_read("A"), None, t, "ain", mm.Memlet("A[i]"))
    body.add_edge(t, "cout", cw, None, mm.Memlet("c[0]"))
    body.add_edge(cw, None, body.add_write("ZP"), None, mm.Memlet("ZP[(i + 1) % 2]"))  # toggled ring slot
    post = sdfg.add_state("post")
    sdfg.add_edge(loop, post, dace.InterstateEdge())
    sdfg.validate()
    return sdfg


def test_transient_scan_writeback_not_lifted():
    """A prefix-scan carry stamped into a TRANSIENT buffer at a moving slot (``ZP[i+1] = c``) must NOT
    lift to a ``Reduce`` -- the hole 881e55d79 missed (it exempted transient writebacks). gpu_scc's
    flux double-buffer is a transient, so its scan was collapsed and the per-level flux dropped.
    Mode fixed: the refusal is ``_extract``'s, before the ``prefer`` dispatch.
    """
    from dace.libraries.standard.nodes.reduce import Reduce
    prefer = _MATCHER_ONLY_MODE
    sdfg = _build_transient_scan_writeback()
    assert _count_loops(sdfg) == 1
    lifted = LoopToReduce(prefer=prefer).apply_pass(sdfg, {})
    sdfg.validate()
    assert not lifted, f'transient scan writeback must not lift; got {lifted}'
    assert _count_loops(sdfg) == 1, 'the scan loop must survive for LoopToScan'
    assert not [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, Reduce)], 'no Reduce may be emitted'


def test_double_buffer_carry_scan_not_lifted():
    """The CloudSC k-caching double-buffer (``ZP[(i+1)%2] = c``) must NOT lift to a ``Reduce``: it is
    a genuine sequential recurrence over a ring buffer, not a fold. The modulo slot keeps the ``%``
    form of the moving-slot subset in front of the refusal. Mode fixed: ``_extract`` refuses before
    the ``prefer`` dispatch.
    """
    from dace.libraries.standard.nodes.reduce import Reduce
    prefer = _MATCHER_ONLY_MODE
    sdfg = _build_double_buffer_scan_writeback()
    assert _count_loops(sdfg) == 1
    lifted = LoopToReduce(prefer=prefer).apply_pass(sdfg, {})
    sdfg.validate()
    assert not lifted, f'double-buffer carry must not lift; got {lifted}'
    assert _count_loops(sdfg) == 1, 'the recurrence loop must survive'
    assert not [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, Reduce)], 'no Reduce may be emitted'


# ---------------------------------------------------------------------------
# The loop's OWN SDFG decides the lift.
#
# ``all_nodes_recursive`` descends into NestedSDFGs, so the LoopRegions the passes iterate
# may live one level down. Every matcher resolves descriptors BY NAME, so handing it the
# top-level SDFG answers from the wrong repository -- silently, because a same-named outer
# descriptor resolves rather than raising. The fixtures below pin both directions.
# ---------------------------------------------------------------------------

N_NEST = dace.symbol("N_NEST")


@dace.program
def _nested_sum_callee(aa: dace.float64[N_NEST], ss: dace.float64[1]):
    for i in range(N_NEST):
        ss[0] = ss[0] + aa[i]


@dace.program
def _nested_sum_caller(x: dace.float64[N_NEST], t: dace.float64[1]):
    _nested_sum_callee(x, t)


@dace.program
def _nested_dot_callee(aa: dace.float64[N_NEST], bb: dace.float64[N_NEST], dd: dace.float64[1]):
    for i in range(N_NEST):
        dd[0] = dd[0] + aa[i] * bb[i]


@dace.program
def _nested_dot_caller(x: dace.float64[N_NEST], y: dace.float64[N_NEST], t: dace.float64[1]):
    _nested_dot_callee(x, y, t)


@dace.program
def _nested_live_write_callee(a: dace.float64[N_NEST], s: dace.float64[1], b: dace.float64[N_NEST]):
    for i in range(N_NEST):
        s[0] = s[0] + a[i]
        b[:] = a


@dace.program
def _nested_live_write_caller(a: dace.float64[N_NEST], s: dace.float64[1], o: dace.float64[N_NEST]):
    b = np.zeros(N_NEST, dtype=np.float64)
    s[0] = s[0] + a[0]  # mints the callee's staging-transient names in the CALLER too
    _nested_live_write_callee(a, s, b)
    o[:] = b


def _loops_inside_nested_sdfgs(sdfg: dace.SDFG):
    """``[(loop.label, owner.name)]`` for every named loop that lives in a NestedSDFG."""
    return [(r.label, sd.name) for sd in sdfg.all_sdfgs_recursive() if sd.parent_sdfg is not None
            for r in sd.all_control_flow_regions() if isinstance(r, LoopRegion) and r.loop_variable]


def _nest_without_inlining(prog) -> dace.SDFG:
    """Build ``prog``'s SDFG with the callee kept as a NestedSDFG.

    Everything except ``InlineSDFGs`` runs, so the loop body reaches the single-content-state
    shape the matchers want while the reduction stays one level down -- the shape any caller
    that lifts on an SDFG it did not fully inline hands the pass.
    """
    from dace.transformation.passes.lift_preprocess import LiftPreprocess
    from dace.transformation.passes.simplify import SimplifyPass
    sdfg = prog.to_sdfg(simplify=False)
    SimplifyPass(skip={'InlineSDFGs'}, validate=True).apply_pass(sdfg, {})
    LiftPreprocess().apply_pass(sdfg, {})
    return sdfg


def _inner_sdfg(sdfg: dace.SDFG) -> dace.SDFG:
    (inner, ) = [sd for sd in sdfg.all_sdfgs_recursive() if sd.parent_sdfg is not None]
    return inner


def _misowned_data(sdfg: dace.SDFG):
    """``[(sdfg.name, state.label, name)]`` for every AccessNode or memlet naming data its
    OWN SDFG does not define. A lift that resolved descriptors against the wrong SDFG
    allocates its scalars in one repository and wires them in the other, leaving exactly
    this residue."""
    bad = []
    for sd in sdfg.all_sdfgs_recursive():
        for state in sd.states():
            for n in state.data_nodes():
                if n.data not in sd.arrays:
                    bad.append((sd.name, state.label, n.data))
            for e in state.edges():
                if not e.data.is_empty() and e.data.data is not None and e.data.data not in sd.arrays:
                    bad.append((sd.name, state.label, e.data.data))
    return bad


def test_reduce_lift_inside_nested_sdfg_uses_the_nested_sdfgs_arrays(prefer):
    """A reduction inside a NestedSDFG must be matched against THAT SDFG's descriptors.

    ``_extract`` resolves the accumulator with ``accum not in sdfg.arrays -> refuse``. The
    frontend's staging scalar (``ss_slice_plus_aa_slice``) exists only in the callee, so
    against the top-level SDFG the guard fires and a perfectly liftable reduction is
    silently dropped on the floor -- a predicate answering False because it looked in the
    wrong place, not because the shape was unsafe.

    Pins the PROPERTY (the lift happens, lands in the owning SDFG, and the result matches
    the sequential oracle) rather than a count, and guards non-vacuity: if the fixture ever
    stops nesting the loop, or the caller ever gains the callee's names, it proves nothing.
    """
    sdfg = _nest_without_inlining(_nested_sum_caller)
    inner = _inner_sdfg(sdfg)
    assert _loops_inside_nested_sdfgs(sdfg), 'fixture no longer nests the reduction loop'
    assert 'ss' not in sdfg.arrays, 'fixture is vacuous: the caller must NOT define the accumulator'

    lifted = LoopToReduce(prefer=prefer).apply_pass(sdfg, {})
    sdfg.validate()
    assert lifted == 1, f'the nested reduction must lift; got {lifted}'
    # Whatever the emit mode minted belongs to the loop's OWN SDFG, not the top-level one.
    assert not _misowned_data(sdfg), f'the lift wired data across SDFGs: {_misowned_data(sdfg)}'
    if prefer == 'reduce-libnode':
        assert [
            sd.name for sd in sdfg.all_sdfgs_recursive() for st in sd.states() for n in st.nodes()
            if isinstance(n, Reduce)
        ] == [inner.name], 'the Reduce must be emitted in the nested SDFG'

    n = 24
    rng = np.random.default_rng(1147)
    x = rng.standard_normal(n)
    t = np.zeros(1)
    sdfg(x=x, t=t, N_NEST=n)
    assert np.allclose(t[0], x.sum()), 'lifted nested reduction diverged from the sequential oracle'


def test_reduce_refusal_inside_nested_sdfg_ignores_a_same_named_outer_transient():
    """A same-named descriptor in the CALLER must not authorise a lift the callee forbids.

    ``_extract`` refuses a loop that writes any LIVE (non-transient) array besides the
    accumulator: a single ``Reduce`` emits one final value and would drop the per-iteration
    writes. ``b`` is a non-transient connector in the callee -- and a local TRANSIENT in the
    caller. Asked against the top-level SDFG the guard reads the caller's answer, passes,
    and the lift deletes the loop wholesale: ``b`` (hence ``o``) is never written. A wrong
    answer, not a missed one.

    Mode fixed: the live-array guard is ``_extract``'s, evaluated before the ``prefer`` dispatch;
    ``test_reduce_lift_inside_nested_sdfg_uses_the_nested_sdfgs_arrays`` sweeps both modes over the
    positive nested lift (where the emit really does differ per mode).
    """
    prefer = _MATCHER_ONLY_MODE
    sdfg = _nest_without_inlining(_nested_live_write_caller)
    inner = _inner_sdfg(sdfg)
    assert _loops_inside_nested_sdfgs(sdfg), 'fixture no longer nests the reduction loop'
    assert sdfg.arrays['b'].transient and not inner.arrays['b'].transient, (
        'fixture is vacuous: the collision must flip the transient flag between the two SDFGs')

    lifted = LoopToReduce(prefer=prefer).apply_pass(sdfg, {})
    sdfg.validate()
    assert not lifted, f'a loop writing a live array must not lift; got {lifted}'
    assert not [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, Reduce)], 'no Reduce may be emitted'

    n = 24
    rng = np.random.default_rng(1148)
    a = rng.standard_normal(n)
    s = np.zeros(1)
    o = np.zeros(n)
    sdfg(a=a, s=s, o=o, N_NEST=n)
    assert np.allclose(s[0], a[0] + a.sum()), 'accumulator diverged from the sequential oracle'
    assert np.allclose(o, a), 'the per-iteration write to the live array was dropped'


def test_retarget_wcr_accumulator_inside_nested_sdfg_uses_the_nested_sdfgs_arrays():
    """Same ownership property for ``RetargetWCRAccumulator``.

    ``_extract_wcr_body`` looks the WCR write's destination up with ``sdfg.arrays.get`` and
    skips it when the descriptor is missing or transient. ``dd`` lives only in the callee,
    so against the top-level SDFG every candidate is skipped and the multi-tasklet dot
    product keeps its serial in-body accumulate instead of the privatised scalar a
    downstream ``LoopToMap`` lowers to ``reduction(+:scalar)``.
    """
    sdfg = _nest_without_inlining(_nested_dot_caller)
    assert _loops_inside_nested_sdfgs(sdfg), 'fixture no longer nests the reduction loop'
    assert 'dd' not in sdfg.arrays, 'fixture is vacuous: the caller must NOT define the accumulator'

    LoopToReduce(prefer='wcr-scalar').apply_pass(sdfg, {})
    AccumulatorCopyChainToWCR().apply_pass(sdfg, {})
    retargeted = RetargetWCRAccumulator().apply_pass(sdfg, {})
    sdfg.validate()
    assert retargeted == 1, f'the nested WCR accumulator must be retargeted; got {retargeted}'

    n = 24
    rng = np.random.default_rng(1149)
    x = rng.standard_normal(n)
    y = rng.standard_normal(n)
    t = np.zeros(1)
    sdfg(x=x, y=y, t=t, N_NEST=n)
    assert np.allclose(t[0], x @ y), 'retargeted nested dot product diverged from the sequential oracle'


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
    test_reduce_lift_inside_nested_sdfg_uses_the_nested_sdfgs_arrays('reduce-libnode')
    test_reduce_lift_inside_nested_sdfg_uses_the_nested_sdfgs_arrays('wcr-scalar')
    test_reduce_refusal_inside_nested_sdfg_ignores_a_same_named_outer_transient()
    test_retarget_wcr_accumulator_inside_nested_sdfg_uses_the_nested_sdfgs_arrays()

# ---- scatter: the accumulator SLOT moves per iteration -------------------


@dace.program
def scatter_add_by_index(idx: dace.int64[N], val: dace.float64[N], out: dace.float64[M]):
    for i in range(N):
        out[idx[i]] += val[i]


@dace.program
def scatter_add_guarded_bin(radius: dace.float64[N], out: dace.float64[M], lo: dace.float64, hi: dace.float64):
    for i in range(N):
        if lo <= radius[i] and radius[i] <= hi:
            b = int((radius[i] - lo) * M / (hi - lo))
            if b < 0:
                b = 0
            if b > M - 1:
                b = M - 1
            out[b] += 1.0


def test_scatter_by_index_array_is_not_retargeted():
    """``out[idx[i]] += val[i]`` is a SCATTER, not a reduction into one accumulator.

    The slot moves every iteration through a symbol the body assigns, so privatizing it into a
    single scalar collapses every scattered bin onto one. The matcher must refuse it.
    """
    sdfg = scatter_add_by_index.to_sdfg(simplify=True)
    sdfg.validate()
    PatternMatchAndApplyRepeated([TrivialTaskletElimination()]).apply_pass(sdfg, {})
    PatternMatchAndApplyRepeated([WCRToAugAssign()]).apply_pass(sdfg, {})
    AccumulatorCopyChainToWCR().apply_pass(sdfg, {})

    assert RetargetWCRAccumulator().apply_pass(sdfg, {}) is None
    assert _count_wcr_scalar_targets(sdfg, 'lambda a, b: a + b') == 0


def test_scatter_with_guarded_bin_index_is_not_retargeted():
    """The same refusal when the moving symbol is assigned INSIDE a guard.

    A histogram computes its bin on an iedge nested in ``if lo <= r <= hi``, so the symbol is not
    on the loop's own edges and a shallow scan of those edges reports the slot as loop-invariant.
    This is the shape that made azimint_hist lose 999 of its 1000 bins with no error raised.
    """
    sdfg = scatter_add_guarded_bin.to_sdfg(simplify=True)
    sdfg.validate()
    PatternMatchAndApplyRepeated([TrivialTaskletElimination()]).apply_pass(sdfg, {})
    PatternMatchAndApplyRepeated([WCRToAugAssign()]).apply_pass(sdfg, {})
    AccumulatorCopyChainToWCR().apply_pass(sdfg, {})

    assert RetargetWCRAccumulator().apply_pass(sdfg, {}) is None
    assert _count_wcr_scalar_targets(sdfg, 'lambda a, b: a + b') == 0


def test_guarded_scatter_histogram_keeps_every_bin():
    """The refusal has to hold through the whole canonicalize pipeline, not just this matcher.

    Structure alone would not have caught the original bug end to end: the collapse compiled, ran,
    and returned a full-length array whose entries were simply wrong. The histogram shape is the
    one the corpus actually carries (azimint_hist), and its accumulator is a transient the scatter
    stage leaves sequential, so this measures the retarget refusal and nothing else.
    """
    from dace.transformation.passes.canonicalize.pipeline import canonicalize

    n, m = 4096, 64
    rng = np.random.default_rng(0)
    radius = rng.random(n)
    lo, hi = float(radius.min()), float(radius.max())
    bins = np.clip(((radius - lo) * m / (hi - lo)).astype(np.int64), 0, m - 1)
    expected = np.zeros(m)
    np.add.at(expected, bins, 1.0)

    sdfg = scatter_add_guarded_bin.to_sdfg(simplify=True)
    canonicalize(sdfg, target='cpu', validate_all=False)
    out = np.zeros(m)
    sdfg(radius=radius, out=out, lo=lo, hi=hi, N=n, M=m)

    assert np.allclose(out, expected)
