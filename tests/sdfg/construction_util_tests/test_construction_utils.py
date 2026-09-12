# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Unit tests for the construction utilities in :mod:`dace.sdfg.construction_utils`
used by the upcoming branch-normalization passes (M3.1+):

* ``copy_state_contents`` — deep-copies nodes/edges from one state into another
  and returns a node map.
* ``copy_graph_contents`` — same, but for a ``ControlFlowRegion`` (preserves
  the start-block flag, copies interstate edges).
* ``move_state_after`` / ``move_state_before`` — relocates a state inside its
  parent region, inserting an empty hull state where it used to be.
* ``move_branch_cfg_up_discard_conditions`` — replaces a ``ConditionalBlock``
  with the contents of one of its branches, rewiring in/out edges.

These helpers are used by ``SameWriteSetIfElseToITECFG`` (M3.1) to lift the
two arms of an if-else into their own CFGs and then drop the conditional
shell. Bugs in any of them silently corrupt control flow downstream; the
tests below pin the contracts that pass requires.
"""
import numpy as np

import dace
import pytest
from dace.properties import CodeBlock
from dace.sdfg.construction_utils import (
    copy_state_contents,
    copy_graph_contents,
    move_state_after,
    move_state_before,
    move_branch_cfg_up_discard_conditions,
)
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion


def _build_scalar_add_state(sdfg: dace.SDFG, state_label: str, lhs: str, rhs: str, out: str) -> dace.SDFGState:
    """Builds a state ``out = lhs + rhs`` over scalar arrays of shape (1,).
    All three arrays must already exist on ``sdfg``."""
    s = sdfg.add_state(state_label, is_start_block=False)
    a = s.add_access(lhs)
    b = s.add_access(rhs)
    c = s.add_access(out)
    t = s.add_tasklet("add", {"_a", "_b"}, {"_c"}, "_c = _a + _b")
    s.add_edge(a, None, t, "_a", dace.Memlet(f"{lhs}[0]"))
    s.add_edge(b, None, t, "_b", dace.Memlet(f"{rhs}[0]"))
    s.add_edge(t, "_c", c, None, dace.Memlet(f"{out}[0]"))
    return s


def _scalar_sdfg(name: str, arrays=("a", "b", "c")) -> dace.SDFG:
    sdfg = dace.SDFG(name)
    for n in arrays:
        sdfg.add_array(n, shape=(1, ), dtype=dace.float64)
    return sdfg


# ---------------------------------------------------------------------------
# copy_state_contents
# ---------------------------------------------------------------------------


def test_copy_state_contents_preserves_nodes_and_edges():
    sdfg = _scalar_sdfg("src")
    src = _build_scalar_add_state(sdfg, "src_state", "a", "b", "c")
    dst = sdfg.add_state("dst_state")

    node_map = copy_state_contents(src, dst)

    assert len(node_map) == len(src.nodes())
    assert set(node_map.keys()) == set(src.nodes())
    # Same number of nodes/edges, and the new ones are *not* the same objects.
    assert dst.number_of_nodes() == src.number_of_nodes()
    assert dst.number_of_edges() == src.number_of_edges()
    for old, new in node_map.items():
        assert old is not new


def test_copy_state_contents_edge_connectors_preserved():
    """Every edge's src_conn/dst_conn names must survive the copy verbatim."""
    sdfg = _scalar_sdfg("src")
    src = _build_scalar_add_state(sdfg, "src_state", "a", "b", "c")
    dst = sdfg.add_state("dst_state")

    node_map = copy_state_contents(src, dst)

    src_edges = {(e.src.label if hasattr(e.src, "label") else e.src.data, e.src_conn,
                  e.dst.label if hasattr(e.dst, "label") else e.dst.data, e.dst_conn)
                 for e in src.edges()}
    dst_edges = {(e.src.label if hasattr(e.src, "label") else e.src.data, e.src_conn,
                  e.dst.label if hasattr(e.dst, "label") else e.dst.data, e.dst_conn)
                 for e in dst.edges()}
    assert src_edges == dst_edges


def test_copy_state_contents_into_empty_state_idempotent_on_source():
    """Source state must be untouched by the copy."""
    sdfg = _scalar_sdfg("src")
    src = _build_scalar_add_state(sdfg, "src_state", "a", "b", "c")
    nodes_before = list(src.nodes())
    edges_before = list(src.edges())
    dst = sdfg.add_state("dst_state")

    copy_state_contents(src, dst)

    assert list(src.nodes()) == nodes_before
    assert list(src.edges()) == edges_before


# ---------------------------------------------------------------------------
# copy_graph_contents
# ---------------------------------------------------------------------------


def test_copy_graph_contents_preserves_start_block():
    """The start block of ``old_graph`` must map to the start block of ``new_graph``."""
    src = dace.SDFG("src")
    src.add_array("a", shape=(1, ), dtype=dace.float64)
    s0 = src.add_state("s0", is_start_block=True)
    s1 = src.add_state("s1")
    src.add_edge(s0, s1, dace.InterstateEdge())

    dst = dace.SDFG("dst")
    dst.add_array("a", shape=(1, ), dtype=dace.float64)

    node_map = copy_graph_contents(src, dst)

    assert dst.start_block is node_map[s0]
    assert dst.number_of_nodes() == 2
    assert dst.number_of_edges() == 1


def test_copy_graph_contents_copies_interstate_edge_assignments():
    """Interstate edge ``assignments`` must come across deep-copied, not aliased."""
    src = dace.SDFG("src")
    src.add_symbol("k", dace.int64)
    s0 = src.add_state("s0", is_start_block=True)
    s1 = src.add_state("s1")
    src.add_edge(s0, s1, dace.InterstateEdge(assignments={"k": "5"}))

    dst = dace.SDFG("dst")
    dst.add_symbol("k", dace.int64)

    copy_graph_contents(src, dst)

    dst_edges = list(dst.edges())
    assert len(dst_edges) == 1
    assert dict(dst_edges[0].data.assignments) == {"k": "5"}
    assert dst_edges[0].data is not list(src.edges())[0].data


def test_copy_graph_contents_rejects_non_cfr():
    """The helper has explicit ``assert isinstance(... ControlFlowRegion)`` guards."""
    src = dace.SDFG("src")
    state = src.add_state("only_state", is_start_block=True)
    dst = dace.SDFG("dst")
    with pytest.raises(AssertionError):
        copy_graph_contents(state, dst)
    with pytest.raises(AssertionError):
        copy_graph_contents(src, state)


# ---------------------------------------------------------------------------
# move_state_after / move_state_before
# ---------------------------------------------------------------------------


def _linear_chain(sdfg: dace.SDFG, labels):
    """Adds states with the given labels in a linear chain; first is the start block."""
    states = []
    for i, lab in enumerate(labels):
        s = sdfg.add_state(lab, is_start_block=(i == 0))
        states.append(s)
    for a, b in zip(states[:-1], states[1:]):
        sdfg.add_edge(a, b, dace.InterstateEdge())
    return states


def test_move_state_after_inserts_hull_and_reorders():
    """move_state_after(graph, X, Y): after the call, the in-graph edges should
    include ``Y -> X``. A hull state is created where ``X`` used to sit."""
    sdfg = dace.SDFG("mv_after")
    s_start, s_a, s_b, s_end = _linear_chain(sdfg, ["start", "a", "b", "end"])
    # Move 'b' to come after 'start' (originally 'b' was after 'a').

    move_state_after(sdfg, state_to_move=s_b, target_predecessor=s_start)

    # 'start' -> 'b' edge now exists.
    succ_of_start = [e.dst for e in sdfg.out_edges(s_start)]
    assert s_b in succ_of_start

    # Hull state was created in 'b's old slot (between 'a' and 'end').
    hull_labels = [n.label for n in sdfg.nodes() if n.label.endswith("_hull")]
    assert len(hull_labels) == 1
    assert hull_labels[0] == "b_hull"


def test_move_state_after_noop_when_already_in_position():
    """If the state already comes directly after the predecessor, the helper
    returns without mutating the graph."""
    sdfg = dace.SDFG("mv_after_noop")
    s0, s1 = _linear_chain(sdfg, ["s0", "s1"])
    nodes_before = set(sdfg.nodes())
    edges_before = {(e.src, e.dst) for e in sdfg.edges()}

    move_state_after(sdfg, state_to_move=s1, target_predecessor=s0)

    assert set(sdfg.nodes()) == nodes_before
    assert {(e.src, e.dst) for e in sdfg.edges()} == edges_before


def test_move_state_after_rejects_self_move():
    sdfg = dace.SDFG("mv_after_self")
    (s, ) = _linear_chain(sdfg, ["only"])
    with pytest.raises(AssertionError):
        move_state_after(sdfg, state_to_move=s, target_predecessor=s)


def test_move_state_after_rejects_foreign_state():
    sdfg = dace.SDFG("mv_after_foreign")
    other = dace.SDFG("other")
    s_local, = _linear_chain(sdfg, ["local"])
    s_foreign, = _linear_chain(other, ["foreign"])
    with pytest.raises(ValueError):
        move_state_after(sdfg, state_to_move=s_foreign, target_predecessor=s_local)


def test_move_state_before_inserts_hull_and_reorders():
    """move_state_before(graph, X, Y): after the call, ``X -> Y`` exists, and a
    hull state replaces X's original slot."""
    sdfg = dace.SDFG("mv_before")
    s_start, s_a, s_b, s_end = _linear_chain(sdfg, ["start", "a", "b", "end"])

    move_state_before(sdfg, state_to_move=s_a, target_successor=s_end)

    pred_of_end = [e.src for e in sdfg.in_edges(s_end)]
    assert s_a in pred_of_end

    hull_labels = [n.label for n in sdfg.nodes() if n.label.endswith("_hull")]
    assert len(hull_labels) == 1
    assert hull_labels[0] == "a_hull"


def test_move_state_before_noop_when_already_in_position():
    sdfg = dace.SDFG("mv_before_noop")
    s0, s1 = _linear_chain(sdfg, ["s0", "s1"])
    nodes_before = set(sdfg.nodes())
    edges_before = {(e.src, e.dst) for e in sdfg.edges()}

    move_state_before(sdfg, state_to_move=s0, target_successor=s1)

    assert set(sdfg.nodes()) == nodes_before
    assert {(e.src, e.dst) for e in sdfg.edges()} == edges_before


# ---------------------------------------------------------------------------
# move_branch_cfg_up_discard_conditions
# ---------------------------------------------------------------------------


def _build_conditional_with_two_branches(sdfg: dace.SDFG):
    """Inserts a ConditionalBlock with two single-state branches into ``sdfg``.
    The conditional sits between an entry state and an exit state."""
    entry = sdfg.add_state("entry", is_start_block=True)
    exit_state = sdfg.add_state("exit")

    cb = ConditionalBlock("cb", sdfg=sdfg, parent=sdfg)
    sdfg.add_node(cb)
    sdfg.add_edge(entry, cb, dace.InterstateEdge())
    sdfg.add_edge(cb, exit_state, dace.InterstateEdge())

    # then-branch: a single CFR with one state.
    then_cfr = ControlFlowRegion("then_cfr", sdfg=sdfg)
    then_state = then_cfr.add_state("then_state", is_start_block=True)
    cb.add_branch(CodeBlock("cond"), then_cfr)

    # else-branch.
    else_cfr = ControlFlowRegion("else_cfr", sdfg=sdfg)
    else_state = else_cfr.add_state("else_state", is_start_block=True)
    cb.add_branch(None, else_cfr)

    return entry, cb, then_cfr, then_state, else_cfr, else_state, exit_state


def test_move_branch_cfg_up_replaces_conditional_with_then_body():
    sdfg = dace.SDFG("mbcu_then")
    entry, cb, then_cfr, then_state, else_cfr, else_state, exit_state = (_build_conditional_with_two_branches(sdfg))

    move_branch_cfg_up_discard_conditions(if_block=cb, body_to_take=then_cfr)

    labels = {n.label for n in sdfg.nodes()}
    # Conditional gone, its then-branch state landed in the parent.
    assert "cb" not in labels
    assert any(lbl.startswith("then_state") for lbl in labels)
    # entry now points at a copy of then_state; that copy points at exit.
    entry_succs = [e.dst for e in sdfg.out_edges(entry)]
    assert len(entry_succs) == 1
    promoted = entry_succs[0]
    assert promoted.label.startswith("then_state")
    promoted_succs = [e.dst for e in sdfg.out_edges(promoted)]
    assert exit_state in promoted_succs


def test_move_branch_cfg_up_replaces_conditional_with_else_body():
    sdfg = dace.SDFG("mbcu_else")
    entry, cb, then_cfr, then_state, else_cfr, else_state, exit_state = (_build_conditional_with_two_branches(sdfg))

    move_branch_cfg_up_discard_conditions(if_block=cb, body_to_take=else_cfr)

    labels = {n.label for n in sdfg.nodes()}
    assert "cb" not in labels
    assert any(lbl.startswith("else_state") for lbl in labels)
    entry_succs = [e.dst for e in sdfg.out_edges(entry)]
    assert len(entry_succs) == 1
    promoted = entry_succs[0]
    assert promoted.label.startswith("else_state")


def test_move_branch_cfg_up_rejects_foreign_branch():
    """``body_to_take`` must be one of the conditional's branches."""
    sdfg = dace.SDFG("mbcu_foreign")
    entry, cb, then_cfr, then_state, else_cfr, else_state, exit_state = (_build_conditional_with_two_branches(sdfg))
    foreign = ControlFlowRegion("foreign", sdfg=sdfg)
    foreign.add_state("foreign_state", is_start_block=True)

    with pytest.raises(AssertionError):
        move_branch_cfg_up_discard_conditions(if_block=cb, body_to_take=foreign)


def test_move_branch_cfg_up_preserves_start_block_when_conditional_was_start():
    """If the conditional was the start block of its parent graph, the promoted
    branch's start state must take over as the new start block."""
    sdfg = dace.SDFG("mbcu_start")
    cb = ConditionalBlock("cb", sdfg=sdfg, parent=sdfg)
    sdfg.add_node(cb, is_start_block=True)
    exit_state = sdfg.add_state("exit")
    sdfg.add_edge(cb, exit_state, dace.InterstateEdge())

    then_cfr = ControlFlowRegion("then_cfr", sdfg=sdfg)
    then_state = then_cfr.add_state("then_state", is_start_block=True)
    cb.add_branch(CodeBlock("cond"), then_cfr)
    else_cfr = ControlFlowRegion("else_cfr", sdfg=sdfg)
    else_cfr.add_state("else_state", is_start_block=True)
    cb.add_branch(None, else_cfr)

    move_branch_cfg_up_discard_conditions(if_block=cb, body_to_take=then_cfr)

    assert sdfg.start_block is not cb
    assert sdfg.start_block.label.startswith("then_state")


# ---------------------------------------------------------------------------
# Numerical correctness — compile + run each helper's output and compare
# against the plain-Python reference for the same computation. Per project
# rule: SDFG-producing tests compare against a non-transformed reference.
# ---------------------------------------------------------------------------


def _build_add_state_into(sdfg: dace.SDFG, state_label: str, out_arr: str = "c"):
    """Builds a state that computes ``<out_arr>[0] = a[0] + b[0]`` on the
    given sdfg. ``a``, ``b`` and ``out_arr`` must already exist."""
    s = sdfg.add_state(state_label)
    a = s.add_access("a")
    b = s.add_access("b")
    c = s.add_access(out_arr)
    t = s.add_tasklet("add", {"_a", "_b"}, {"_c"}, "_c = _a + _b")
    s.add_edge(a, None, t, "_a", dace.Memlet("a[0]"))
    s.add_edge(b, None, t, "_b", dace.Memlet("b[0]"))
    s.add_edge(t, "_c", c, None, dace.Memlet(f"{out_arr}[0]"))
    return s


def test_copy_state_contents_numerical_correctness():
    """Build SDFG ``c = a + b``, copy that state's contents into a fresh
    state, redirect the start to the copy, and verify the SDFG still
    produces ``a + b``."""
    sdfg = dace.SDFG("copy_state_num")
    for n in ("a", "b", "c"):
        sdfg.add_array(n, shape=(1, ), dtype=dace.float64)
    src = _build_add_state_into(sdfg, "src", out_arr="c")
    sdfg.start_block = sdfg.node_id(src)

    dst = sdfg.add_state("dst")
    copy_state_contents(src, dst)

    # Rewire: dst replaces src as the start state.
    sdfg.start_block = sdfg.node_id(dst)
    for e in list(sdfg.out_edges(src)) + list(sdfg.in_edges(src)):
        sdfg.remove_edge(e)
    sdfg.remove_node(src)

    a = np.array([3.5], dtype=np.float64)
    b = np.array([-1.25], dtype=np.float64)
    c = np.zeros((1, ), dtype=np.float64)
    sdfg(a=a, b=b, c=c)
    np.testing.assert_allclose(c, a + b)


def test_copy_graph_contents_numerical_correctness():
    """Compute ``c = (a + b) * 2`` in a source SDFG (two states), copy that
    graph into a fresh SDFG, and verify the copied SDFG produces the
    same result."""
    src = dace.SDFG("src")
    for n in ("a", "b", "tmp", "c"):
        src.add_array(n, shape=(1, ), dtype=dace.float64)
    src.arrays["tmp"].transient = True

    # Compute tmp = a + b in state 0.
    s0 = src.add_state("s0", is_start_block=True)
    sa = s0.add_access("a")
    sb = s0.add_access("b")
    stmp = s0.add_access("tmp")
    t0 = s0.add_tasklet("add", {"_a", "_b"}, {"_t"}, "_t = _a + _b")
    s0.add_edge(sa, None, t0, "_a", dace.Memlet("a[0]"))
    s0.add_edge(sb, None, t0, "_b", dace.Memlet("b[0]"))
    s0.add_edge(t0, "_t", stmp, None, dace.Memlet("tmp[0]"))

    # Compute c = tmp * 2 in state 1.
    s1 = src.add_state("s1")
    src.add_edge(s0, s1, dace.InterstateEdge())
    stmp2 = s1.add_access("tmp")
    sc = s1.add_access("c")
    t1 = s1.add_tasklet("mul", {"_t"}, {"_c"}, "_c = _t * 2.0")
    s1.add_edge(stmp2, None, t1, "_t", dace.Memlet("tmp[0]"))
    s1.add_edge(t1, "_c", sc, None, dace.Memlet("c[0]"))

    # Fresh SDFG with the same descriptors and an empty body.
    dst = dace.SDFG("dst")
    for n in ("a", "b", "tmp", "c"):
        dst.add_array(n, shape=(1, ), dtype=dace.float64)
    dst.arrays["tmp"].transient = True
    copy_graph_contents(src, dst)

    a = np.array([2.0], dtype=np.float64)
    b = np.array([0.5], dtype=np.float64)
    c = np.zeros((1, ), dtype=np.float64)
    dst(a=a, b=b, c=c)
    np.testing.assert_allclose(c, (a + b) * 2.0)


def _build_two_writers_with_shared_input(sdfg: dace.SDFG):
    """Builds two states each writing to a *different* output from a shared
    input ``inp``. Their relative order is therefore semantically
    irrelevant, which is what makes them safe inputs for ``move_state_*``
    numerical tests."""
    # state s_a writes A[0] = inp[0] + 1
    sa = sdfg.add_state("s_a")
    ra = sa.add_access("inp")
    wa = sa.add_access("A")
    ta = sa.add_tasklet("plus", {"_i"}, {"_o"}, "_o = _i + 1.0")
    sa.add_edge(ra, None, ta, "_i", dace.Memlet("inp[0]"))
    sa.add_edge(ta, "_o", wa, None, dace.Memlet("A[0]"))
    # state s_b writes B[0] = inp[0] - 1
    sb = sdfg.add_state("s_b")
    rb = sb.add_access("inp")
    wb = sb.add_access("B")
    tb = sb.add_tasklet("minus", {"_i"}, {"_o"}, "_o = _i - 1.0")
    sb.add_edge(rb, None, tb, "_i", dace.Memlet("inp[0]"))
    sb.add_edge(tb, "_o", wb, None, dace.Memlet("B[0]"))
    return sa, sb


def test_move_state_after_numerical_correctness():
    """Reorder two semantically-independent states and verify the SDFG still
    computes the correct ``A`` and ``B``."""
    sdfg = dace.SDFG("mv_after_num")
    for n in ("inp", "A", "B"):
        sdfg.add_array(n, shape=(1, ), dtype=dace.float64)
    s_start = sdfg.add_state("s_start", is_start_block=True)
    sa, sb = _build_two_writers_with_shared_input(sdfg)
    s_end = sdfg.add_state("s_end")
    sdfg.add_edge(s_start, sa, dace.InterstateEdge())
    sdfg.add_edge(sa, sb, dace.InterstateEdge())
    sdfg.add_edge(sb, s_end, dace.InterstateEdge())

    move_state_after(sdfg, state_to_move=sb, target_predecessor=s_start)

    inp = np.array([4.0], dtype=np.float64)
    A = np.zeros((1, ), dtype=np.float64)
    B = np.zeros((1, ), dtype=np.float64)
    sdfg(inp=inp, A=A, B=B)
    np.testing.assert_allclose(A, inp + 1.0)
    np.testing.assert_allclose(B, inp - 1.0)


def test_move_state_before_numerical_correctness():
    sdfg = dace.SDFG("mv_before_num")
    for n in ("inp", "A", "B"):
        sdfg.add_array(n, shape=(1, ), dtype=dace.float64)
    s_start = sdfg.add_state("s_start", is_start_block=True)
    sa, sb = _build_two_writers_with_shared_input(sdfg)
    s_end = sdfg.add_state("s_end")
    sdfg.add_edge(s_start, sa, dace.InterstateEdge())
    sdfg.add_edge(sa, sb, dace.InterstateEdge())
    sdfg.add_edge(sb, s_end, dace.InterstateEdge())

    move_state_before(sdfg, state_to_move=sa, target_successor=s_end)

    inp = np.array([-2.5], dtype=np.float64)
    A = np.zeros((1, ), dtype=np.float64)
    B = np.zeros((1, ), dtype=np.float64)
    sdfg(inp=inp, A=A, B=B)
    np.testing.assert_allclose(A, inp + 1.0)
    np.testing.assert_allclose(B, inp - 1.0)


def test_move_branch_cfg_up_then_branch_numerical_correctness():
    """After promoting the then-branch and discarding the else-branch, the
    SDFG must unconditionally execute the then-branch's body — regardless
    of what the discarded condition would have evaluated to."""
    sdfg = dace.SDFG("mbcu_num")
    sdfg.add_array("A", shape=(1, ), dtype=dace.float64)
    sdfg.add_symbol("cond", dace.bool_)
    cb = ConditionalBlock("cb", sdfg=sdfg, parent=sdfg)
    sdfg.add_node(cb, is_start_block=True)
    exit_state = sdfg.add_state("exit")
    sdfg.add_edge(cb, exit_state, dace.InterstateEdge())

    then_cfr = ControlFlowRegion("then_cfr", sdfg=sdfg)
    ts = then_cfr.add_state("then_state", is_start_block=True)
    wa = ts.add_access("A")
    tt = ts.add_tasklet("set17", set(), {"_o"}, "_o = 17.0")
    ts.add_edge(tt, "_o", wa, None, dace.Memlet("A[0]"))
    cb.add_branch(CodeBlock("cond"), then_cfr)

    else_cfr = ControlFlowRegion("else_cfr", sdfg=sdfg)
    es = else_cfr.add_state("else_state", is_start_block=True)
    wa2 = es.add_access("A")
    te = es.add_tasklet("set99", set(), {"_o"}, "_o = 99.0")
    es.add_edge(te, "_o", wa2, None, dace.Memlet("A[0]"))
    cb.add_branch(None, else_cfr)

    move_branch_cfg_up_discard_conditions(if_block=cb, body_to_take=then_cfr)

    for c in (True, False):
        A = np.zeros((1, ), dtype=np.float64)
        sdfg(A=A, cond=c)
        # Regardless of cond, the then-branch was forced — A == 17.
        np.testing.assert_allclose(A, np.array([17.0]))
