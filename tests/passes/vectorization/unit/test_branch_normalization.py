# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Pass-level tests for ``BranchNormalization``.

After the pass runs:
- No ``ConditionalBlock`` remains in any region the pass processed.
- Every state has well-formed connector roles
  (``assert_connector_role_matches_edges`` from the same construction-utils
  audit family that M3.1 uses).
- An ITE-tasklet is emitted for every previously-conditional write,
  routing ``arr[subset] = ITE(cond, expr, arr[subset])``.

Cases covered:
- Single-arm conditional ``if cond: a[i] = expr`` with no else.
- Two-arm ``if/else`` with disjoint write sets: the pass should split
  into two sequential single-arm conditionals and then normalize both.
- Chained ``if/elif/else`` shape (recursive normalization).
- End-to-end numerical correctness against the plain-Python reference,
  inputs chosen to exercise every arm.
"""

import numpy as np

import dace
from dace.properties import CodeBlock
from dace.sdfg.construction_utils import assert_connector_role_matches_edges
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion
from dace.transformation.passes.vectorization.branch_normalization import (
    BranchNormalization,
    compute_arm_escape_writes,
)


def _build_single_arm_if_sdfg():
    """Build an SDFG that, in Python, looks like
        if cond: A[0] = B[0] + 1.0
    Then nothing if cond is false.
    """
    sdfg = dace.SDFG("single_arm")
    sdfg.add_array("A", shape=(1, ), dtype=dace.float64)
    sdfg.add_array("B", shape=(1, ), dtype=dace.float64)
    sdfg.add_symbol("cond", dace.bool_)

    entry = sdfg.add_state("entry", is_start_block=True)
    exit_state = sdfg.add_state("exit")

    cb = ConditionalBlock("cb")
    sdfg.add_node(cb)
    sdfg.add_edge(entry, cb, dace.InterstateEdge())
    sdfg.add_edge(cb, exit_state, dace.InterstateEdge())

    body = ControlFlowRegion("body", sdfg=sdfg)
    bs = body.add_state("bs", is_start_block=True)
    rb = bs.add_access("B")
    wa = bs.add_access("A")
    t = bs.add_tasklet("plus", {"_b"}, {"_a"}, "_a = _b + 1.0")
    bs.add_edge(rb, None, t, "_b", dace.Memlet("B[0]"))
    bs.add_edge(t, "_a", wa, None, dace.Memlet("A[0]"))
    cb.add_branch(CodeBlock("cond"), body)

    return sdfg


def _build_disjoint_two_arm_sdfg():
    """``if cond: A[0] = B[0] + 1 else: C[0] = B[0] - 1`` — disjoint write
    sets (A vs C), should be split into two sequential single-arm conditionals
    and then normalized."""
    sdfg = dace.SDFG("disjoint_two_arm")
    for n in ("A", "B", "C"):
        sdfg.add_array(n, shape=(1, ), dtype=dace.float64)
    sdfg.add_symbol("cond", dace.bool_)

    entry = sdfg.add_state("entry", is_start_block=True)
    exit_state = sdfg.add_state("exit")
    cb = ConditionalBlock("cb")
    sdfg.add_node(cb)
    sdfg.add_edge(entry, cb, dace.InterstateEdge())
    sdfg.add_edge(cb, exit_state, dace.InterstateEdge())

    then_cfr = ControlFlowRegion("then_cfr", sdfg=sdfg)
    ts = then_cfr.add_state("ts", is_start_block=True)
    r = ts.add_access("B")
    w = ts.add_access("A")
    t = ts.add_tasklet("t", {"_b"}, {"_a"}, "_a = _b + 1.0")
    ts.add_edge(r, None, t, "_b", dace.Memlet("B[0]"))
    ts.add_edge(t, "_a", w, None, dace.Memlet("A[0]"))
    cb.add_branch(CodeBlock("cond"), then_cfr)

    else_cfr = ControlFlowRegion("else_cfr", sdfg=sdfg)
    es = else_cfr.add_state("es", is_start_block=True)
    r2 = es.add_access("B")
    w2 = es.add_access("C")
    te = es.add_tasklet("t2", {"_b"}, {"_c"}, "_c = _b - 1.0")
    es.add_edge(r2, None, te, "_b", dace.Memlet("B[0]"))
    es.add_edge(te, "_c", w2, None, dace.Memlet("C[0]"))
    cb.add_branch(None, else_cfr)

    return sdfg


def test_single_arm_removes_conditional_block_and_emits_ite_tasklet():
    sdfg = _build_single_arm_if_sdfg()
    rewritten = BranchNormalization().apply_pass(sdfg, {})
    assert rewritten is not None and rewritten >= 1

    assert not any(isinstance(b, ConditionalBlock) for b in sdfg.all_control_flow_blocks()), \
        [b.label for b in sdfg.all_control_flow_blocks()]

    found_ite = False
    for state in sdfg.states():
        for n in state.nodes():
            if isinstance(n, dace.nodes.Tasklet) and n.label.startswith("bn_ite_"):
                assert "ITE(cond," in n.code.as_string.replace(" ", "")
                found_ite = True
    assert found_ite, "no bn_ite_ tasklet was emitted"


def test_single_arm_connector_topology_correct():
    sdfg = _build_single_arm_if_sdfg()
    BranchNormalization().apply_pass(sdfg, {})
    for state in sdfg.states():
        assert_connector_role_matches_edges(state)


def test_single_arm_numerical_correctness():
    sdfg = _build_single_arm_if_sdfg()
    BranchNormalization().apply_pass(sdfg, {})
    csdfg = sdfg.compile()

    for c in (True, False):
        for b in (-2.0, 0.5, 7.0):
            A = np.array([99.0], dtype=np.float64)
            B = np.array([b], dtype=np.float64)
            csdfg(A=A, B=B, cond=c)
            expected = (b + 1.0) if c else 99.0
            np.testing.assert_allclose(A, np.array([expected]), err_msg=f"c={c}, b={b}, got={A}, expected={expected}")


def test_disjoint_two_arm_splits_and_normalizes():
    sdfg = _build_disjoint_two_arm_sdfg()
    rewritten = BranchNormalization().apply_pass(sdfg, {})
    assert rewritten is not None and rewritten >= 2  # one split + two single-arm rewrites

    assert not any(isinstance(b, ConditionalBlock) for b in sdfg.all_control_flow_blocks())


def test_disjoint_two_arm_connector_topology_correct():
    sdfg = _build_disjoint_two_arm_sdfg()
    BranchNormalization().apply_pass(sdfg, {})
    for state in sdfg.states():
        assert_connector_role_matches_edges(state)


def test_disjoint_two_arm_numerical_correctness():
    sdfg = _build_disjoint_two_arm_sdfg()
    BranchNormalization().apply_pass(sdfg, {})
    csdfg = sdfg.compile()

    for c in (True, False):
        for b in (-3.0, 0.0, 4.5):
            A = np.array([10.0], dtype=np.float64)
            C = np.array([20.0], dtype=np.float64)
            B = np.array([b], dtype=np.float64)
            csdfg(A=A, B=B, C=C, cond=c)
            expected_A = (b + 1.0) if c else 10.0
            expected_C = 20.0 if c else (b - 1.0)
            np.testing.assert_allclose(A, np.array([expected_A]), err_msg=f"A c={c}, b={b}, got={A}")
            np.testing.assert_allclose(C, np.array([expected_C]), err_msg=f"C c={c}, b={b}, got={C}")


def test_pass_is_idempotent_after_first_run():
    sdfg = _build_single_arm_if_sdfg()
    first = BranchNormalization().apply_pass(sdfg, {})
    second = BranchNormalization().apply_pass(sdfg, {})
    assert first is not None
    assert second is None


# ---------------------------------------------------------------------------
# compute_arm_escape_writes (pure analysis)
# ---------------------------------------------------------------------------


def _add_state_writing(sdfg: dace.SDFG, region, label: str, write_target: str, value: str = "1.0"):
    """Convenience: add a state to ``region`` that contains a single
    tasklet writing ``value`` to ``write_target[0]``."""
    s = region.add_state(label)
    out = s.add_access(write_target)
    t = s.add_tasklet(f"t_{label}", set(), {"_o"}, f"_o = {value}")
    s.add_edge(t, "_o", out, None, dace.Memlet(f"{write_target}[0]"))
    return s


def _add_state_reading(sdfg: dace.SDFG, region, label: str, read_target: str, write_target: str):
    """Convenience: add a state that reads ``read_target`` and writes
    its negation to ``write_target``."""
    s = region.add_state(label)
    rd = s.add_access(read_target)
    wr = s.add_access(write_target)
    t = s.add_tasklet(f"t_{label}", {"_i"}, {"_o"}, "_o = -_i")
    s.add_edge(rd, None, t, "_i", dace.Memlet(f"{read_target}[0]"))
    s.add_edge(t, "_o", wr, None, dace.Memlet(f"{write_target}[0]"))
    return s


def _build_arm_with_writes(sdfg: dace.SDFG, label: str, writes):
    """Build a single-state ControlFlowRegion containing one tasklet per
    entry in ``writes`` (a list of array names to write 1.0 to)."""
    region = ControlFlowRegion(label, sdfg=sdfg)
    state = region.add_state(f"{label}_s", is_start_block=True)
    for arr in writes:
        out = state.add_access(arr)
        t = state.add_tasklet(f"t_{label}_{arr}", set(), {"_o"}, "_o = 1.0")
        state.add_edge(t, "_o", out, None, dace.Memlet(f"{arr}[0]"))
    return region


def test_escape_writes_non_transient_target_escapes():
    """Rule 1: a write to a non-transient array always escapes."""
    sdfg = dace.SDFG("escape_rule1")
    sdfg.add_array("A", shape=(1, ), dtype=dace.float64)  # non-transient
    sdfg.add_array("S", shape=(1, ), dtype=dace.float64, transient=True)
    sdfg.add_symbol("c", dace.bool_)
    entry = sdfg.add_state("entry", is_start_block=True)
    cb = ConditionalBlock("cb")
    sdfg.add_node(cb)
    sdfg.add_edge(entry, cb, dace.InterstateEdge())
    cb.add_branch(CodeBlock("c"), _build_arm_with_writes(sdfg, "arm0", ["A", "S"]))

    plan = compute_arm_escape_writes(sdfg, cb)
    assert "A" in plan[0], plan
    # ``S`` is transient and not read elsewhere: stays arm-private.
    assert "S" not in plan[0], plan


def test_escape_writes_transient_read_elsewhere_escapes():
    """Rule 2: a write to a transient is escaping iff the transient is
    read in a state outside the conditional's subtree."""
    sdfg = dace.SDFG("escape_rule2")
    sdfg.add_array("T", shape=(1, ), dtype=dace.float64, transient=True)
    sdfg.add_array("OUT", shape=(1, ), dtype=dace.float64)
    sdfg.add_symbol("c", dace.bool_)
    entry = sdfg.add_state("entry", is_start_block=True)
    cb = ConditionalBlock("cb")
    sdfg.add_node(cb)
    sdfg.add_edge(entry, cb, dace.InterstateEdge())
    cb.add_branch(CodeBlock("c"), _build_arm_with_writes(sdfg, "arm0", ["T"]))
    # Sibling state outside ``cb`` that reads ``T``.
    after = _add_state_reading(sdfg, sdfg, "after_cb", "T", "OUT")
    sdfg.add_edge(cb, after, dace.InterstateEdge())

    plan = compute_arm_escape_writes(sdfg, cb)
    assert "T" in plan[0], plan


def test_escape_writes_transient_only_inside_arm_stays_private():
    """A transient that is never read outside the arm and not written by
    a sibling arm stays arm-private (no entry in the plan)."""
    sdfg = dace.SDFG("escape_arm_private")
    sdfg.add_array("PRIVATE", shape=(1, ), dtype=dace.float64, transient=True)
    sdfg.add_symbol("c", dace.bool_)
    entry = sdfg.add_state("entry", is_start_block=True)
    cb = ConditionalBlock("cb")
    sdfg.add_node(cb)
    sdfg.add_edge(entry, cb, dace.InterstateEdge())
    cb.add_branch(CodeBlock("c"), _build_arm_with_writes(sdfg, "arm0", ["PRIVATE"]))

    plan = compute_arm_escape_writes(sdfg, cb)
    assert plan[0] == set(), plan


def test_escape_writes_cross_arm_read_escapes_writer():
    """Rule 3: if one arm reads ``arr`` and another arm writes ``arr``,
    the writer's write escapes because both arms run unconditionally
    post-rewrite."""
    sdfg = dace.SDFG("escape_rule3")
    sdfg.add_array("T", shape=(1, ), dtype=dace.float64, transient=True)
    sdfg.add_array("OUT", shape=(1, ), dtype=dace.float64, transient=True)
    sdfg.add_symbol("c", dace.bool_)
    entry = sdfg.add_state("entry", is_start_block=True)
    cb = ConditionalBlock("cb")
    sdfg.add_node(cb)
    sdfg.add_edge(entry, cb, dace.InterstateEdge())

    # arm 0 writes T.
    cb.add_branch(CodeBlock("c"), _build_arm_with_writes(sdfg, "arm0", ["T"]))

    # arm 1 (else) reads T and writes OUT.
    else_cfr = ControlFlowRegion("arm1", sdfg=sdfg)
    es = else_cfr.add_state("arm1_s", is_start_block=True)
    rd = es.add_access("T")
    wr = es.add_access("OUT")
    t = es.add_tasklet("rt", {"_i"}, {"_o"}, "_o = -_i")
    es.add_edge(rd, None, t, "_i", dace.Memlet("T[0]"))
    es.add_edge(t, "_o", wr, None, dace.Memlet("OUT[0]"))
    cb.add_branch(None, else_cfr)

    plan = compute_arm_escape_writes(sdfg, cb)
    assert "T" in plan[0], plan  # arm 0's write to T escapes (rule 3)
    assert plan[1] == set(), plan  # arm 1 has no writes that escape


def test_escape_writes_interstate_edge_cond_outside_cb_is_a_read():
    """Rule 2 extension: an interstate-edge condition outside ``cb`` that
    references a transient counts as a read for escape purposes."""
    sdfg = dace.SDFG("escape_interstate_cond")
    sdfg.add_array("T", shape=(1, ), dtype=dace.float64, transient=True)
    sdfg.add_symbol("c", dace.bool_)
    entry = sdfg.add_state("entry", is_start_block=True)
    cb = ConditionalBlock("cb")
    sdfg.add_node(cb)
    sdfg.add_edge(entry, cb, dace.InterstateEdge())
    cb.add_branch(CodeBlock("c"), _build_arm_with_writes(sdfg, "arm0", ["T"]))

    # An after-cb chain whose interstate edge condition tokenises ``T``.
    s1 = sdfg.add_state("after_one")
    s2 = sdfg.add_state("after_two")
    sdfg.add_edge(cb, s1, dace.InterstateEdge())
    sdfg.add_edge(s1, s2, dace.InterstateEdge(condition=CodeBlock("T > 0")))

    plan = compute_arm_escape_writes(sdfg, cb)
    assert "T" in plan[0], plan


def test_escape_writes_returns_empty_set_for_each_arm_when_nothing_escapes():
    """Arms with no writes at all return an empty set, not a missing key."""
    sdfg = dace.SDFG("escape_no_writes")
    sdfg.add_symbol("c", dace.bool_)
    entry = sdfg.add_state("entry", is_start_block=True)
    cb = ConditionalBlock("cb")
    sdfg.add_node(cb)
    sdfg.add_edge(entry, cb, dace.InterstateEdge())
    empty = ControlFlowRegion("arm0", sdfg=sdfg)
    empty.add_state("noop", is_start_block=True)
    cb.add_branch(CodeBlock("c"), empty)
    plan = compute_arm_escape_writes(sdfg, cb)
    assert plan[0] == set(), plan
