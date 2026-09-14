# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for the ``LiftTrivialIf`` simplification pass."""
import dace
from dace import InterstateEdge
from dace.sdfg.sdfg import CodeBlock, ConditionalBlock
from dace import ControlFlowRegion
from dace.transformation.passes.lift_trivial_if import LiftTrivialIf
import pytest

# Conditions the pass must recognize as constant ``True``.
_ALWAYS_TRUE = [
    "True",
    "1 == 1",
    "2 > 1",
    "5 >= 5",
    "not False",
    "2 + 2 == 4",
    "abs(-5) == 5",
    "max(1, 2, 3) == 3",
]

# Conditions the pass must recognize as constant ``False``.
_ALWAYS_FALSE = [
    "False",
    "1 == 2",
    "5 < 3",
    "10 <= 9",
    "not True",
    "2 + 2 == 5",
    "abs(-5) == -5",
]

# Conditions that reference unbound symbols -- the pass must leave these alone.
_CANT_EVAL = ["a < 5", "c == 0", "d >= 1"]

# Conditions where pystr_to_symbolic yields an unevaluated symbolic expression
# (Function/Indexed/Symbol). bool() of such an expression is truthy, which would
# mis-classify dynamic dataflow conditions like ``A[0]`` as trivially true.
_DYNAMIC_RUNTIME_COND = ["A[0]", "tmp_r[0]", "x", "x[0] + 1", "A[i, j]"]


def _get_sdfg(condition: str):
    """Build a one-state SDFG inside a single-branch ``ConditionalBlock``.

    :param condition: Python condition expression for the branch.
    :returns: The SDFG with the conditional as its start block.
    """
    sdfg = dace.SDFG("basic1")
    _, A = sdfg.add_array(name="A", shape=[
        5,
    ], dtype=dace.float64, transient=False)
    _, B = sdfg.add_array(name="B", shape=[
        5,
    ], dtype=dace.float64, transient=False)
    cb = ConditionalBlock(label="cfb1", sdfg=sdfg, parent=sdfg)
    sdfg.add_node(cb, is_start_block=True)
    cfg = ControlFlowRegion(label="cfg1", sdfg=cb.sdfg, parent=cb)
    cb.add_branch(condition=CodeBlock(condition), branch=cfg)
    s1 = cfg.add_state(label="s1")
    aA = s1.add_access("A")
    aB = s1.add_access("B")
    s1.add_edge(aA, None, aB, None, dace.memlet.Memlet.from_array("A", A))

    return sdfg


def _get_if_else_sdfg(condition: str, body_in_else_branch: bool):
    """Build an if/else SDFG with the body on the requested branch.

    :param condition: Python condition expression for the ``if`` arm.
    :param body_in_else_branch: If ``True`` put the access-node body on the ``else``
        branch (otherwise on the ``if`` arm).
    :returns: The SDFG with the conditional as its start block.
    """
    sdfg = dace.SDFG("if_else_1")
    _, A = sdfg.add_array(name="A", shape=[
        5,
    ], dtype=dace.float64, transient=False)
    _, B = sdfg.add_array(name="B", shape=[
        5,
    ], dtype=dace.float64, transient=False)
    cb = ConditionalBlock(label="cfb1", sdfg=sdfg, parent=sdfg)
    sdfg.add_node(cb, is_start_block=True)
    cfg_if_true = ControlFlowRegion(label="cfg_if_true", sdfg=cb.sdfg, parent=cb)
    cfg_if_false = ControlFlowRegion(label="cfg_if_false", sdfg=cb.sdfg, parent=cb)
    cb.add_branch(condition=CodeBlock(condition), branch=cfg_if_true)
    cb.add_branch(condition=None, branch=cfg_if_false)
    if body_in_else_branch:
        s1 = cfg_if_true.add_state(label="s1", is_start_block=True)
        s2 = cfg_if_false.add_state(label="fs1", is_start_block=True)
    else:
        s1 = cfg_if_false.add_state(label="s1", is_start_block=True)
        s2 = cfg_if_true.add_state(label="ts1", is_start_block=True)
    aA = s1.add_access("A")
    aB = s1.add_access("B")
    s1.add_edge(aA, None, aB, None, dace.memlet.Memlet.from_array("A", A))

    return sdfg


def _get_nested_sdfg(condition1: str, condition2: str):
    """Build a conditional nested inside another conditional.

    :param condition1: Outer condition.
    :param condition2: Inner condition.
    :returns: The SDFG with the outer conditional as its start block.
    """
    sdfg = dace.SDFG("nested1")
    sdfg = dace.SDFG("basic1")
    _, A = sdfg.add_array(name="A", shape=[
        5,
    ], dtype=dace.float64, transient=False)
    _, B = sdfg.add_array(name="B", shape=[
        5,
    ], dtype=dace.float64, transient=False)
    cb = ConditionalBlock(label="cfb1", sdfg=sdfg, parent=sdfg)
    sdfg.add_node(cb, is_start_block=True)
    cfg1 = ControlFlowRegion(label="cfg1", sdfg=cb.sdfg, parent=cb)
    cb.add_branch(condition=CodeBlock(condition1), branch=cfg1)
    cb2 = ConditionalBlock(label="cfb2", sdfg=cfg1.sdfg, parent=cfg1)
    cfg1.add_node(cb2, is_start_block=True)
    cfg2 = ControlFlowRegion(label="cfg2", sdfg=cb2.sdfg, parent=cb2)
    cb2.add_branch(condition=CodeBlock(condition2), branch=cfg2)
    s1 = cfg2.add_state(label="s1", is_start_block=True)
    aA = s1.add_access("A")
    aB = s1.add_access("B")
    s1.add_edge(aA, None, aB, None, dace.memlet.Memlet.from_array("A", A))

    return sdfg


def _get_sdfg_with_many_states():
    """Build an SDFG with non-conditional states surrounding a nested conditional.

    :returns: An SDFG where the conditional sits between ``so1`` and ``so2``.
    """
    sdfg = dace.SDFG("nested1")
    sdfg = dace.SDFG("basic1")
    _, A = sdfg.add_array(name="A", shape=[
        5,
    ], dtype=dace.float64, transient=False)
    _, B = sdfg.add_array(name="B", shape=[
        5,
    ], dtype=dace.float64, transient=False)
    cb = ConditionalBlock(label="cfb1", sdfg=sdfg, parent=sdfg)
    so1 = sdfg.add_state("so1", is_start_block=True)
    so2 = sdfg.add_state("so2")
    sdfg.add_node(cb, is_start_block=False)
    sdfg.add_edge(so1, cb, InterstateEdge())
    sdfg.add_edge(cb, so2, InterstateEdge())
    cfg1 = ControlFlowRegion(label="cfg1", sdfg=cb.sdfg, parent=cb)
    cb.add_branch(condition=CodeBlock("1 == 1"), branch=cfg1)
    cb2 = ConditionalBlock(label="cfb2", sdfg=cfg1.sdfg, parent=cfg1)
    cfg1.add_node(cb2, is_start_block=True)
    cfg2 = ControlFlowRegion(label="cfg2", sdfg=cb2.sdfg, parent=cb2)
    cb2.add_branch(condition=CodeBlock("0 == 1"), branch=cfg2)
    s1 = cfg2.add_state(label="s1", is_start_block=True)
    aA = s1.add_access("A")
    aB = s1.add_access("B")
    s1.add_edge(aA, None, aB, None, dace.memlet.Memlet.from_array("A", A))
    return sdfg


@pytest.mark.parametrize("condition", _ALWAYS_TRUE)
def test_single_condition(condition: str):
    """Trivially-true single-branch ``if`` is removed."""
    sdfg = _get_sdfg(condition)
    sdfg.validate()
    LiftTrivialIf().apply_pass(sdfg, {})
    sdfg.validate()
    assert len({n for n in sdfg.all_control_flow_blocks() if isinstance(n, ConditionalBlock)}) == 0


@pytest.mark.parametrize("condition", _CANT_EVAL)
def test_single_condition_cant_eval(condition: str):
    """Free-symbol condition is left in place -- the pass can't prove it constant."""
    sdfg = _get_sdfg(condition)
    sdfg.validate()
    LiftTrivialIf().apply_pass(sdfg, {})
    sdfg.validate()
    assert len({n for n in sdfg.all_control_flow_blocks() if isinstance(n, ConditionalBlock)}) == 1


@pytest.mark.parametrize("condition1,condition2",
                         [(_ALWAYS_TRUE[i], _ALWAYS_TRUE[i + 1]) for i in range(len(_ALWAYS_TRUE) - 1)])
def test_nested_condition(condition1: str, condition2: str):
    """Trivially-true ``if`` inside trivially-true ``if`` collapses to plain dataflow."""
    sdfg = _get_nested_sdfg(condition1, condition2)
    sdfg.validate()
    LiftTrivialIf().apply_pass(sdfg, {})
    sdfg.validate()
    assert len({n for n in sdfg.all_control_flow_blocks() if isinstance(n, ConditionalBlock)}) == 0


@pytest.mark.parametrize("condition1,condition2",
                         [(_CANT_EVAL[i], _CANT_EVAL[i + 1]) for i in range(len(_CANT_EVAL) - 1)])
def test_nested_condition_cant_eval(condition1: str, condition2: str):
    """Nested unresolvable conditions are both preserved."""
    sdfg = _get_nested_sdfg(condition1, condition2)
    sdfg.validate()
    LiftTrivialIf().apply_pass(sdfg, {})
    sdfg.validate()
    assert len({n for n in sdfg.all_control_flow_blocks() if isinstance(n, ConditionalBlock)}) == 2


@pytest.mark.parametrize("condition", _ALWAYS_TRUE)
def test_if_else_cond_is_trivially_true(condition: str):
    """``if/else`` with provably-true ``if`` keeps the ``if`` body."""
    sdfg = _get_if_else_sdfg(condition, False)
    sdfg.validate()
    LiftTrivialIf().apply_pass(sdfg, {})
    sdfg.validate()
    assert len({n for n in sdfg.all_control_flow_blocks() if isinstance(n, ConditionalBlock)}) == 0


@pytest.mark.parametrize("condition", _ALWAYS_FALSE)
def test_if_else_cond_is_trivially_false(condition: str):
    """``if/else`` with provably-false ``if`` keeps the ``else`` body."""
    sdfg = _get_if_else_sdfg(condition, True)
    sdfg.validate()
    LiftTrivialIf().apply_pass(sdfg, {})
    sdfg.validate()
    assert len({n for n in sdfg.all_control_flow_blocks() if isinstance(n, ConditionalBlock)}) == 0


def test_cfg_is_a_middle_node():
    """The conditional being mid-graph (with predecessor + successor) doesn't break the splice."""
    sdfg = _get_sdfg_with_many_states()
    sdfg.validate()
    LiftTrivialIf().apply_pass(sdfg, {})
    sdfg.validate()
    assert len({n for n in sdfg.all_control_flow_blocks() if isinstance(n, ConditionalBlock)}) == 0


# Fortran frontends emit conditions of the form ``(x == y) == 0/1`` (a
# comparison of a comparison to an integer). Sympy fails to evaluate this due to mismatching types.
# If these ever stop folding to True/False the pass would silently miss real trivial-if cases in Fortran SDFGs.
@pytest.mark.parametrize("condition", ["(1 == 1) == 1", "(2 == 2) == 1", "(0 == 1) == 0"])
def test_fortran_style_nested_comparison_true(condition: str):
    """The boolean-arithmetic fallback handles the Fortran-frontend ``(x == y) == k`` shape."""
    sdfg = _get_sdfg(condition)
    sdfg.validate()
    LiftTrivialIf().apply_pass(sdfg, {})
    sdfg.validate()
    assert len({n for n in sdfg.all_control_flow_blocks() if isinstance(n, ConditionalBlock)}) == 0


@pytest.mark.parametrize("condition", _DYNAMIC_RUNTIME_COND)
def test_dynamic_runtime_cond_not_trivial(condition: str):
    """Dynamic (runtime-dataflow) conditions must NOT be treated as trivially true/false."""
    sdfg = _get_sdfg(condition)
    sdfg.validate()
    LiftTrivialIf().apply_pass(sdfg, {})
    sdfg.validate()
    # The conditional must survive: it depends on data we don't know at
    # transform time.
    assert len({n for n in sdfg.all_control_flow_blocks() if isinstance(n, ConditionalBlock)}) == 1


def test_simplify_pipeline_includes_lift_trivial_if():
    """``SimplifyPass`` wires ``LiftTrivialIf`` in -- running simplify removes trivial ifs."""
    from dace.transformation.passes.simplify import SimplifyPass
    sdfg = _get_sdfg("True")
    sdfg.validate()
    SimplifyPass().apply_pass(sdfg, {})
    sdfg.validate()
    assert len({n for n in sdfg.all_control_flow_blocks() if isinstance(n, ConditionalBlock)}) == 0


from dace.sdfg.state import LoopRegion


def _get_loop_with_conditional(branch_cond: str,
                               with_else: bool = False,
                               init: str = "i = 2",
                               cond: str = "i < 10",
                               update: str = "i = i + 1"):
    """Build ``for i in [init..cond): if (branch_cond) [else ...]`` with real
    dataflow on the live arm, so the conditional sits inside a ``LoopRegion`` whose
    iteration range governs whether ``branch_cond`` is a range tautology/contradiction.

    :param branch_cond: Condition on the guarded ``if`` arm.
    :param with_else: If ``True`` add an else arm carrying the dataflow (so the
        guarded arm being a contradiction must keep the else).
    :returns: The SDFG with the loop as its start block.
    """
    sdfg = dace.SDFG("loopcond")
    _, A = sdfg.add_array("A", [10], dace.float64)
    sdfg.add_array("B", [10], dace.float64)
    loop = LoopRegion(label="myloop",
                      condition_expr=cond,
                      loop_var="i",
                      initialize_expr=init,
                      update_expr=update,
                      sdfg=sdfg)
    sdfg.add_node(loop, is_start_block=True)
    cb = ConditionalBlock(label="cfb", sdfg=sdfg, parent=loop)
    loop.add_node(cb, is_start_block=True)
    guarded = ControlFlowRegion(label="guarded", sdfg=sdfg, parent=cb)
    cb.add_branch(condition=CodeBlock(branch_cond), branch=guarded)
    guarded.add_state(label="g_s", is_start_block=True)  # every arm needs a start block to be valid
    live = guarded
    if with_else:
        els = ControlFlowRegion(label="els", sdfg=sdfg, parent=cb)
        cb.add_branch(condition=None, branch=els)
        els.add_state(label="e_s", is_start_block=True)
        live = els
    s1 = live.start_block  # the dataflow goes on the branch that survives
    s1.add_edge(s1.add_access("A"), None, s1.add_access("B"), None, dace.memlet.Memlet.from_array("A", A))
    return sdfg


def _num_conditionals(sdfg) -> int:
    return len({n for n in sdfg.all_control_flow_blocks() if isinstance(n, ConditionalBlock)})


# Guards that are a contradiction over ``i in [2, 9]`` -- the guarded branch never runs.
_RANGE_CONTRADICTION = ["i == 0", "i == 1", "i == 10", "i == 100", "i < 2", "i <= 1", "i > 9", "i >= 10", "0 == i"]

# Guards that are a tautology over ``i in [2, 9]`` -- the guarded branch always runs.
_RANGE_TAUTOLOGY = ["not (i == 0)", "i != 0", "i < 10", "i <= 9", "i > 1", "i >= 2", "i != 100"]

# Guards genuinely data-dependent over ``i in [2, 9]`` -- must NOT be folded.
_RANGE_RUNTIME = ["i == 5", "i < 4", "i > 6", "i == 9"]


@pytest.mark.parametrize("cond", _RANGE_CONTRADICTION)
def test_iteration_range_contradiction_single_branch_removed(cond: str):
    """A single-branch ``if`` whose guard is false for every iteration of the
    enclosing loop is dropped (its body never runs)."""
    sdfg = _get_loop_with_conditional(cond)
    sdfg.validate()
    LiftTrivialIf().apply_pass(sdfg, {})
    sdfg.validate()
    assert _num_conditionals(sdfg) == 0


@pytest.mark.parametrize("cond", _RANGE_TAUTOLOGY)
def test_iteration_range_tautology_single_branch_lifted(cond: str):
    """A single-branch ``if`` whose guard holds for every iteration of the enclosing
    loop is lifted: the body always runs, so the redundant guard is discarded."""
    sdfg = _get_loop_with_conditional(cond)
    sdfg.validate()
    LiftTrivialIf().apply_pass(sdfg, {})
    sdfg.validate()
    assert _num_conditionals(sdfg) == 0


@pytest.mark.parametrize("cond", _RANGE_CONTRADICTION)
def test_iteration_range_contradiction_if_else_keeps_else(cond: str):
    """An ``if/else`` whose guard is a range contradiction keeps only the else body."""
    sdfg = _get_loop_with_conditional(cond, with_else=True)
    sdfg.validate()
    LiftTrivialIf().apply_pass(sdfg, {})
    sdfg.validate()
    assert _num_conditionals(sdfg) == 0


@pytest.mark.parametrize("cond", _RANGE_RUNTIME)
def test_iteration_range_runtime_guard_preserved(cond: str):
    """A guard that is neither always-true nor always-false over the loop range is a
    genuine per-iteration branch and must be left in place."""
    sdfg = _get_loop_with_conditional(cond)
    sdfg.validate()
    LiftTrivialIf().apply_pass(sdfg, {})
    sdfg.validate()
    assert _num_conditionals(sdfg) == 1


def test_iteration_range_symbolic_upper_bound():
    """The range reasoning works with a symbolic upper bound: ``i in [2, N-1]`` makes
    ``i == 0`` a contradiction and ``not(i == 1)`` a tautology, regardless of ``N``."""
    N = dace.symbol("N")
    contra = _get_loop_with_conditional("i == 0", cond="i < N")
    contra.add_symbol("N", N.dtype)
    LiftTrivialIf().apply_pass(contra, {})
    contra.validate()
    assert _num_conditionals(contra) == 0

    tauto = _get_loop_with_conditional("not (i == 1)", cond="i < N")
    tauto.add_symbol("N", N.dtype)
    LiftTrivialIf().apply_pass(tauto, {})
    tauto.validate()
    assert _num_conditionals(tauto) == 0


def test_iteration_range_no_enclosing_loop_is_not_folded():
    """Without an enclosing loop there is no iteration range, so ``i == 0`` is a
    free-symbol condition the pass must leave alone."""
    sdfg = _get_sdfg("i == 0")
    sdfg.validate()
    LiftTrivialIf().apply_pass(sdfg, {})
    sdfg.validate()
    assert _num_conditionals(sdfg) == 1


def test_trivial_if_with_nested_sdfg_reparents_correctly():
    """Lifting a trivial-if whose branch holds a NestedSDFG must leave that nested SDFG's parent
    pointers consistent after the branch is spliced up -- the per-splice local repair. The SDFG
    must still validate, and the nested SDFG must now be parented to the outer SDFG."""
    inner = dace.SDFG("inner")
    inner.add_array("A", [5], dace.float64)
    ns = inner.add_state("ns", is_start_block=True)
    t = ns.add_tasklet("w", set(), {"o"}, "o = 1.0")
    ns.add_edge(t, "o", ns.add_access("A"), None, dace.memlet.Memlet("A[0]"))

    sdfg = dace.SDFG("outer_ns")
    sdfg.add_array("A", [5], dace.float64)
    cb = ConditionalBlock(label="cfb", sdfg=sdfg, parent=sdfg)
    sdfg.add_node(cb, is_start_block=True)
    body = ControlFlowRegion(label="body", sdfg=sdfg, parent=cb)
    cb.add_branch(condition=CodeBlock("1 == 1"), branch=body)
    st = body.add_state("bs", is_start_block=True)
    nnode = st.add_nested_sdfg(inner, set(), {"A"})
    st.add_edge(nnode, "A", st.add_access("A"), None, dace.memlet.Memlet("A[0:5]"))

    sdfg.validate()
    LiftTrivialIf().apply_pass(sdfg, {})
    sdfg.validate()

    assert _num_conditionals(sdfg) == 0
    found = [n for s in sdfg.all_states() for n in s.nodes() if isinstance(n, dace.nodes.NestedSDFG)]
    assert len(found) == 1
    assert found[0].sdfg.parent_sdfg is sdfg
    assert found[0].sdfg.parent is not None
    assert found[0].sdfg.parent_nsdfg_node is found[0]


def test_trivial_cond_check_is_cached():
    """The per-condition check is memoized (module-level, ``typed=True``) so repeated guards -- across
    conditionals and across FixedPointPipeline re-invocations -- skip the sympy work without changing
    the verdict."""
    from dace.transformation.passes.lift_trivial_if import _trivial_cond_check_cached
    _trivial_cond_check_cached.cache_clear()
    assert _trivial_cond_check_cached("1 == 1", True) is True
    base = _trivial_cond_check_cached.cache_info()
    for _ in range(5):
        assert _trivial_cond_check_cached("1 == 1", True) is True
    after = _trivial_cond_check_cached.cache_info()
    assert after.hits - base.hits == 5  # repeats served from cache
    assert after.misses == base.misses  # no new sympy evaluation
    # Distinct string and distinct truth value are distinct keys, and verdicts are still correct.
    assert _trivial_cond_check_cached("1 == 2", True) is False
    assert _trivial_cond_check_cached("1 == 1", False) is False


def test_should_reapply_only_on_block_count_change():
    """The pass re-runs only when the number of control-flow blocks changed (a conditional added or
    removed, or a loop peel shifting an enclosing range), not when interstate-edge contents change --
    those never create or destroy a conditional."""
    from dace.transformation import pass_pipeline as ppl
    p = LiftTrivialIf()
    assert p.should_reapply(ppl.Modifies.States)
    assert p.should_reapply(ppl.Modifies.CFG)  # CFG includes States
    assert not p.should_reapply(ppl.Modifies.InterstateEdges)
    assert not p.should_reapply(ppl.Modifies.Nothing)
    assert not p.should_reapply(ppl.Modifies.Symbols | ppl.Modifies.Memlets)


if __name__ == "__main__":
    for c in _ALWAYS_TRUE:
        test_single_condition(c)

    for i in range(len(_ALWAYS_TRUE) - 1):
        c1 = _ALWAYS_TRUE[i]
        c2 = _ALWAYS_TRUE[i + 1]
        test_nested_condition(c1, c2)

    for c in _ALWAYS_TRUE:
        test_if_else_cond_is_trivially_true(c)

    for c in _ALWAYS_FALSE:
        test_if_else_cond_is_trivially_false(c)

    test_cfg_is_a_middle_node()
    test_simplify_pipeline_includes_lift_trivial_if()
    test_trivial_if_with_nested_sdfg_reparents_correctly()
    test_trivial_cond_check_is_cached()
    test_should_reapply_only_on_block_count_change()
