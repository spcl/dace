# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for ``SimplifyExpressions`` / ``simplify_expressions``."""
import dace
from dace import memlet as mm, subsets as subs, symbolic
from dace.properties import CodeBlock
from dace.sdfg import SDFG
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion, LoopRegion
from dace.transformation.passes.simplify_expressions import (SimplifyExpressions, simplify_expressions)


def _sdfg_with_symbol(name: str = "s", sym: str = "x") -> SDFG:
    """Minimal SDFG with one array and one free symbol so we have a
    symbol to build unsimplified expressions around."""
    sdfg = SDFG(name)
    sdfg.add_array("A", [10], dace.float64)
    sdfg.add_symbol(sym, dace.int32)
    sdfg.add_state("entry", is_start_block=True)
    return sdfg


# sympy auto-simplifies trivial rewrites (``x + 0 -> x``) during
# parsing, so tests pick expressions that survive parsing and only
# collapse under ``sympy.simplify``. ``(x+1)*(x-1) + 1`` stays as
# a product after parsing and simplifies to ``x**2``.
_UNSIMPL = "(x + 1) * (x - 1) + 1"
_SIMPL = "x**2"


def test_memlet_subset_is_simplified():
    """A non-trivially-simplified expression in a memlet subset
    collapses after the pass."""
    sdfg = _sdfg_with_symbol()
    state = sdfg.start_state
    ar = state.add_read("A")
    aw = state.add_write("A")
    m = mm.Memlet(data="A",
                  subset=subs.Range([(symbolic.pystr_to_symbolic(_UNSIMPL), symbolic.pystr_to_symbolic(_UNSIMPL), 1)]))
    state.add_edge(ar, None, aw, None, m)

    count = simplify_expressions(sdfg)

    assert count == 1
    (new_b, new_e, _), = list(state.edges()[0].data.subset.ndrange())
    assert str(new_b) == _SIMPL and str(new_e) == _SIMPL


def test_memlet_volume_is_simplified():
    """A non-trivially-simplified expression in ``Memlet.volume``
    collapses after the pass."""
    sdfg = _sdfg_with_symbol()
    state = sdfg.start_state
    ar = state.add_read("A")
    aw = state.add_write("A")
    m = mm.Memlet(data="A", subset="0:10")
    m.volume = symbolic.pystr_to_symbolic(_UNSIMPL)
    state.add_edge(ar, None, aw, None, m)

    count = simplify_expressions(sdfg)

    assert count == 1
    assert str(state.edges()[0].data.volume) == _SIMPL


def test_interstate_assignment_rhs_is_simplified():
    """Interstate-edge assignment RHS is simplified too."""
    sdfg = _sdfg_with_symbol(sym="k")
    s2 = sdfg.add_state("mid")
    sdfg.add_edge(sdfg.start_state, s2, dace.InterstateEdge(assignments={"k": "2 * 3 + k - k + 1"}))

    count = simplify_expressions(sdfg)

    assert count == 1
    edge = next(iter(sdfg.all_interstate_edges()))
    assert edge.data.assignments["k"] == "7"


def test_interstate_condition_is_simplified():
    """Interstate-edge condition gets simplified but remains a CodeBlock."""
    sdfg = _sdfg_with_symbol()
    s2 = sdfg.add_state("mid")
    ie = dace.InterstateEdge(condition=CodeBlock("x + 0 > 0", "Python"))
    sdfg.add_edge(sdfg.start_state, s2, ie)

    count = simplify_expressions(sdfg)

    assert count == 1
    edge = next(iter(sdfg.all_interstate_edges()))
    assert "+ 0" not in edge.data.condition.as_string


def test_array_subscript_not_rewritten_as_function_call():
    """Regression: an expression like ``A[0]`` (which sympy parses as a
    function call) must survive as a subscript after re-rendering, i.e.
    ``symstr`` is invoked with the SDFG's array names. Without that,
    the pass would emit ``A(0)`` and break generated code."""
    sdfg = _sdfg_with_symbol(sym="n")
    s2 = sdfg.add_state("mid")
    sdfg.add_edge(sdfg.start_state, s2, dace.InterstateEdge(assignments={"n": "A[0] + 0"}))

    simplify_expressions(sdfg)

    edge = next(iter(sdfg.all_interstate_edges()))
    rhs = edge.data.assignments["n"]
    assert "[0]" in rhs
    # ``A(0)`` would indicate a sympy-function printing; we want the subscript.
    assert "A(" not in rhs


def test_loopregion_statements_are_simplified():
    """init / condition / update on a LoopRegion are CodeBlocks with
    their own simplification path."""
    sdfg = SDFG("loop_sdfg")
    sdfg.add_array("A", [10], dace.float64)
    sdfg.add_symbol("i", dace.int32)
    loop = LoopRegion(
        label="loop",
        initialize_expr="i = 0 + 0",
        condition_expr="i + 0 < 10",
        update_expr="i = i + 1 + 0",
        loop_var="i",
    )
    sdfg.add_node(loop, is_start_block=True)
    loop.add_state("body", is_start_block=True)

    count = simplify_expressions(sdfg)

    assert count >= 3
    assert "+ 0" not in loop.init_statement.as_string
    assert "+ 0" not in loop.loop_condition.as_string
    assert "+ 0" not in loop.update_statement.as_string


def test_conditional_block_branch_conditions_are_simplified():
    """Each branch guard on a ConditionalBlock is simplified."""
    sdfg = SDFG("cond_sdfg")
    sdfg.add_symbol("k", dace.int32)
    cb = ConditionalBlock(label="cb")
    sdfg.add_node(cb, is_start_block=True)
    branch_region = ControlFlowRegion("br")
    branch_region.add_state("body", is_start_block=True)
    cb.branches.append((CodeBlock("k + 0 > 0", "Python"), branch_region))

    count = simplify_expressions(sdfg)

    assert count == 1
    new_cond, _ = cb.branches[0]
    assert "+ 0" not in new_cond.as_string


def test_nested_sdfg_is_reached():
    """Expressions inside a NestedSDFG's body are simplified too."""
    inner = SDFG("inner")
    inner.add_array("B", [8], dace.float64)
    inner.add_symbol("x", dace.int32)
    st = inner.add_state(is_start_block=True)
    br = st.add_read("B")
    bw = st.add_write("B")
    st.add_edge(
        br, None, bw, None,
        mm.Memlet(data="B",
                  subset=subs.Range([(symbolic.pystr_to_symbolic(_UNSIMPL), symbolic.pystr_to_symbolic(_UNSIMPL), 1)])))

    top = SDFG("top")
    top.add_array("B", [8], dace.float64)
    top.add_symbol("x", dace.int32)
    tst = top.add_state(is_start_block=True)
    n = tst.add_nested_sdfg(inner, {"B"}, {"B"})
    tr = tst.add_read("B")
    tw = tst.add_write("B")
    tst.add_edge(tr, None, n, "B", mm.Memlet.from_array("B", top.arrays["B"]))
    tst.add_edge(n, "B", tw, None, mm.Memlet.from_array("B", top.arrays["B"]))

    count = simplify_expressions(top)

    assert count >= 1
    inner_edges = list(inner.start_state.edges())
    (b, e, _), = list(inner_edges[0].data.subset.ndrange())
    assert str(b) == _SIMPL and str(e) == _SIMPL


def test_no_change_returns_zero():
    """Already-simplified expressions yield ``count == 0``."""
    sdfg = _sdfg_with_symbol()
    state = sdfg.start_state
    ar = state.add_read("A")
    aw = state.add_write("A")
    state.add_edge(ar, None, aw, None, mm.Memlet(data="A", subset="0:10"))

    assert simplify_expressions(sdfg) == 0


def test_pass_class_apply_pass_returns_count_or_none():
    """``apply_pass`` returns ``None`` when nothing changed, else the count."""
    sdfg = _sdfg_with_symbol()
    state = sdfg.start_state
    ar = state.add_read("A")
    aw = state.add_write("A")
    state.add_edge(
        ar, None, aw, None,
        mm.Memlet(data="A",
                  subset=subs.Range([(symbolic.pystr_to_symbolic(_UNSIMPL), symbolic.pystr_to_symbolic(_UNSIMPL), 1)])))
    # First run: one rewrite.
    assert SimplifyExpressions().apply_pass(sdfg, {}) == 1
    # Second run: nothing to do.
    assert SimplifyExpressions().apply_pass(sdfg, {}) is None


if __name__ == "__main__":
    test_memlet_subset_is_simplified()
    test_memlet_volume_is_simplified()
    test_interstate_assignment_rhs_is_simplified()
    test_interstate_condition_is_simplified()
    test_array_subscript_not_rewritten_as_function_call()
    test_loopregion_statements_are_simplified()
    test_conditional_block_branch_conditions_are_simplified()
    test_nested_sdfg_is_reached()
    test_no_change_returns_zero()
    test_pass_class_apply_pass_returns_count_or_none()
    print("all SimplifyExpressions tests passed")
