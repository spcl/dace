# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``SameWriteSetIfElseToITECFG`` must REFUSE a guard on the iteration symbol.

The pass rewrites ``if c: arr[s] = f(...)`` into straight-line ``compute_then`` /
``apply_ITE`` dataflow, so ``f``'s READS become unconditional -- every lane evaluates
them. That is sound for a DATA condition (``if a[i] < 0``: the arm's reads are in range
whatever the lane), but UNSOUND when the guard constrains the iteration symbol itself
(``if i < N - 1: s += a[i+1]*a[i+1]``): there the guard is exactly what keeps the arm's
``a[i+1]`` in range, and hoisting it makes lane ``i = N-1`` read ``a[N]`` out of bounds.
Such a guard needs MASKING (a real per-lane predicate), not if-conversion, so the pass
must leave the ``ConditionalBlock`` alone for the masking path downstream.

Covered here: the DIRECT guard (``i < N - 1``), the TRANSITIVE guard (the condition names
a symbol whose interstate-edge assignment chain resolves to ``i``), the MapEntry-param
form (guard inside an NSDFG nested in a map), and -- as the over-refusal control -- a data
condition ``a[i] < 0.0``, whose ``i`` occurs only as a subscript index and which therefore
must STILL be rewritten.

SCOPE: ``BranchNormalization`` is a SECOND flattening site that runs right after this pass; it now
shares the same refusal (via ``condition_guards_iteration_symbol``), so the hole is closed
end-to-end. That is pinned by ``test_branch_normalization_also_refuses_iteration_guard`` at the
bottom; the numeric end-to-end contract lives in ``branch_norm_iteration_guard_test.py``.
"""
import dace
from dace.properties import CodeBlock
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion, LoopRegion
from dace.transformation.passes.vectorization.branch_normalization import BranchNormalization
from dace.transformation.passes.vectorization.same_write_set_if_else_to_ite_cfg import (
    SameWriteSetIfElseToITECFG, )

N = dace.symbol("N", nonnegative=True)


def _add_guarded_arm(sdfg: dace.SDFG, cb: ConditionalBlock, cond: str, read_subset: str) -> None:
    """Attach the single arm ``if <cond>: s[0] = s[0] + a[<read_subset>] * a[<read_subset>]``.

    :param sdfg: the SDFG owning the arm's region.
    :param cb: the conditional block to attach the arm to.
    :param cond: the arm's guard, as Python source.
    :param read_subset: the subset ``a`` is read at inside the arm.
    """
    body = ControlFlowRegion("arm_body", sdfg=sdfg)
    st = body.add_state("arm_state", is_start_block=True)
    ra = st.add_access("a")
    rs = st.add_access("s")
    ws = st.add_access("s")
    tl = st.add_tasklet("acc", {"_a", "_s"}, {"_o"}, "_o = _s + _a * _a")
    st.add_edge(ra, None, tl, "_a", dace.Memlet(f"a[{read_subset}]"))
    st.add_edge(rs, None, tl, "_s", dace.Memlet("s[0]"))
    st.add_edge(tl, "_o", ws, None, dace.Memlet("s[0]"))
    cb.add_branch(CodeBlock(cond), body)


def _build_loop_guard_sdfg(cond: str, read_subset: str, assignments=None) -> dace.SDFG:
    """``for i in range(N): if <cond>: s[0] += a[<read_subset>]**2`` as a LoopRegion.

    :param cond: the guard source attached to the single arm.
    :param read_subset: the subset ``a`` is read at inside the arm.
    :param assignments: optional interstate assignments bound on the edge REACHING the
        conditional block (used to build the transitive-guard case).
    :returns: the constructed SDFG.
    """
    sdfg = dace.SDFG("loop_symbol_guard")
    sdfg.add_array("a", shape=(N, ), dtype=dace.float64)
    sdfg.add_array("s", shape=(1, ), dtype=dace.float64)

    loop = LoopRegion("loop", loop_var="i", initialize_expr="i = 0", condition_expr="i < N", update_expr="i = i + 1")
    sdfg.add_node(loop, is_start_block=True)

    head = loop.add_state("head", is_start_block=True)
    cb = ConditionalBlock("cb", sdfg=sdfg, parent=loop)
    loop.add_node(cb)
    loop.add_edge(head, cb, dace.InterstateEdge(assignments=assignments or {}))
    _add_guarded_arm(sdfg, cb, cond, read_subset)
    return sdfg


def _build_map_param_guard_sdfg() -> dace.SDFG:
    """``for i in dace.map[0:N]: if i < N - 1: s[0] += a[i+1]**2`` -- the guard lives in an
    NSDFG nested inside the map, so ``i`` is a MapEntry PARAM rather than a loop variable.

    :returns: ``(outer, inner)`` -- the map SDFG and the nested body SDFG holding the guard.
    """
    inner = dace.SDFG("map_body")
    inner.add_array("a", shape=(N, ), dtype=dace.float64)
    inner.add_array("s", shape=(1, ), dtype=dace.float64)
    inner.add_symbol("i", dace.int64)
    inner.add_state("head", is_start_block=True)
    cb = ConditionalBlock("cb", sdfg=inner, parent=inner)
    inner.add_node(cb)
    inner.add_edge(inner.start_block, cb, dace.InterstateEdge())
    _add_guarded_arm(inner, cb, "i < N - 1", "i + 1")

    outer = dace.SDFG("map_param_guard")
    outer.add_array("a", shape=(N, ), dtype=dace.float64)
    outer.add_array("s", shape=(1, ), dtype=dace.float64)
    state = outer.add_state("map_state", is_start_block=True)
    me, mx = state.add_map("outer_map", {"i": f"0:{N}"})
    nsdfg = state.add_nested_sdfg(inner, {"a", "s"}, {"s"}, symbol_mapping={"i": "i", "N": N})
    ra = state.add_access("a")
    ws = state.add_access("s")
    state.add_memlet_path(ra, me, nsdfg, dst_conn="a", memlet=dace.Memlet(f"a[0:{N}]"))
    state.add_memlet_path(nsdfg, mx, ws, src_conn="s", memlet=dace.Memlet("s[0]"))
    # The arm also READS ``s`` -- wire it in through the same map entry.
    rs = state.add_access("s")
    state.add_memlet_path(rs, me, nsdfg, dst_conn="s", memlet=dace.Memlet("s[0]"))
    return outer, inner


def _conditional_blocks(sdfg: dace.SDFG):
    return [b for b in sdfg.all_control_flow_blocks() if isinstance(b, ConditionalBlock)]


def test_direct_loop_symbol_guard_is_refused():
    """``if i < N - 1`` names the loop variable free -> must NOT be if-converted."""
    sdfg = _build_loop_guard_sdfg("i < N - 1", "i + 1")
    rewritten = SameWriteSetIfElseToITECFG().apply_pass(sdfg, {})
    assert rewritten is None, "iteration-symbol guard was if-converted; a[N] now read on lane i = N-1"
    assert len(_conditional_blocks(sdfg)) == 1


def test_transitive_loop_symbol_guard_is_refused():
    """``ip1 = i + 1`` on the edge into the block, guard ``ip1 < N``: the symbol chain
    resolves to the loop variable, so the guard is still an iteration guard."""
    sdfg = _build_loop_guard_sdfg("ip1 < N", "ip1", assignments={"ip1": "i + 1"})
    rewritten = SameWriteSetIfElseToITECFG().apply_pass(sdfg, {})
    assert rewritten is None, "transitive iteration-symbol guard was if-converted"
    assert len(_conditional_blocks(sdfg)) == 1


def test_map_param_guard_is_refused():
    """The iteration symbol is a MapEntry param reached across an NSDFG boundary."""
    _outer, inner = _build_map_param_guard_sdfg()
    rewritten = SameWriteSetIfElseToITECFG().apply_pass(inner, {})
    assert rewritten is None, "map-param iteration guard was if-converted"
    assert len(_conditional_blocks(inner)) == 1


def test_multi_hop_symbol_chain_is_refused():
    """``ip1 = j1`` and ``j1 = i + 1``: resolving the chain needs more than one sweep."""
    sdfg = _build_loop_guard_sdfg("ip1 < N", "ip1", assignments={"ip1": "j1", "j1": "i + 1"})
    rewritten = SameWriteSetIfElseToITECFG().apply_pass(sdfg, {})
    assert rewritten is None, "multi-hop iteration-symbol chain was if-converted"
    assert len(_conditional_blocks(sdfg)) == 1


def test_guard_mixed_with_data_condition_is_refused():
    """``a[j] < 0.0 and i < N - 1`` mixes a data predicate with an iteration guard. ``i`` occurs
    both nowhere-as-an-index here and as a genuine guard, so the block must still be refused --
    a free-symbols-MINUS-index-symbols set subtraction would wrongly let this through."""
    sdfg = _build_loop_guard_sdfg("a[i] < 0.0 and i < N - 1", "i + 1")
    rewritten = SameWriteSetIfElseToITECFG().apply_pass(sdfg, {})
    assert rewritten is None, "guard conjoined with a data predicate was if-converted"
    assert len(_conditional_blocks(sdfg)) == 1


def test_unparseable_by_sympy_data_condition_still_rewritten():
    """``math.fabs(a[i]) > 1e-9`` is a plain data condition that ``pystr_to_symbolic`` chokes on.
    The gate must neither crash nor over-refuse it."""
    sdfg = _build_loop_guard_sdfg("math.fabs(a[i]) > 1e-9", "i")
    rewritten = SameWriteSetIfElseToITECFG().apply_pass(sdfg, {})
    assert rewritten == 1, "call-bearing data condition must still be if-converted"
    assert not _conditional_blocks(sdfg)


def test_data_condition_still_rewritten():
    """Over-refusal control: ``i`` occurs ONLY as a subscript index in ``a[i] < 0.0``, so the
    arm's reads are in range on every lane and the if-conversion stays legal."""
    sdfg = _build_loop_guard_sdfg("a[i] < 0.0", "i")
    rewritten = SameWriteSetIfElseToITECFG().apply_pass(sdfg, {})
    assert rewritten == 1, "data condition must still be if-converted (no iteration symbol is free)"
    assert not _conditional_blocks(sdfg)


def test_no_enclosing_iteration_scope_still_rewritten():
    """Over-refusal control: a guard naming a plain SDFG symbol outside any loop/map is
    not an iteration guard."""
    sdfg = dace.SDFG("no_scope")
    sdfg.add_array("a", shape=(N, ), dtype=dace.float64)
    sdfg.add_array("s", shape=(1, ), dtype=dace.float64)
    sdfg.add_symbol("k", dace.int64)
    sdfg.add_state("head", is_start_block=True)
    cb = ConditionalBlock("cb", sdfg=sdfg, parent=sdfg)
    sdfg.add_node(cb)
    sdfg.add_edge(sdfg.start_block, cb, dace.InterstateEdge())
    _add_guarded_arm(sdfg, cb, "k < N - 1", "0")
    rewritten = SameWriteSetIfElseToITECFG().apply_pass(sdfg, {})
    assert rewritten == 1, "non-iteration guard must still be if-converted"
    assert not _conditional_blocks(sdfg)


def test_branch_normalization_also_refuses_iteration_guard():
    """Closes the hole end-to-end: the pipeline runs ``SameWriteSetIfElseToITECFG`` and then
    ``BranchNormalization``, and BOTH now refuse the guard (they share
    ``condition_guards_iteration_symbol``), so the ``ConditionalBlock`` survives instead of being
    flattened into ``s[0] = ITE(i < N-1, s[0] + a[i+1]**2, s[0])`` that reads ``a[N]`` on lane
    ``i = N-1``."""
    sdfg = _build_loop_guard_sdfg("i < N - 1", "i + 1")
    assert SameWriteSetIfElseToITECFG().apply_pass(sdfg, {}) is None
    rewritten = BranchNormalization().apply_pass(sdfg, {})
    assert rewritten is None, "BranchNormalization flattened the iteration guard the first pass refused"
    assert len(_conditional_blocks(sdfg)) == 1


if __name__ == "__main__":
    test_direct_loop_symbol_guard_is_refused()
    test_transitive_loop_symbol_guard_is_refused()
    test_multi_hop_symbol_chain_is_refused()
    test_guard_mixed_with_data_condition_is_refused()
    test_unparseable_by_sympy_data_condition_still_rewritten()
    test_map_param_guard_is_refused()
    test_data_condition_still_rewritten()
    test_no_enclosing_iteration_scope_still_rewritten()
    test_branch_normalization_also_refuses_iteration_guard()
