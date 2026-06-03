# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Pass-level tests for ``SameWriteSetIfElseToITECFG``.

After running, an SDFG that originally contained a same-write-set ``if/else``
must:
- have *no* remaining ``ConditionalBlock``,
- have three new states ``compute_then`` / ``compute_else`` / ``apply_ITE``
  wired in sequence,
- produce numerically the same result as the original SDFG.

The third assertion uses ``sdfg.compile()`` end-to-end. Per the project rule,
the reference is an unfolded scalar Python evaluation -- not a different
SDFG variant.
"""
import os

import numpy as np

import dace
from dace.properties import CodeBlock
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion
from dace.transformation.passes.vectorization.same_write_set_if_else_to_ite_cfg import (
    SameWriteSetIfElseToITECFG,
    _symbol_has_external_consumer,
)

# This pass emits ``ITE(...)`` tasklets which need ``dace/ITE.h``.
os.environ.setdefault("DACE_compiler_cpu_args", "")


def _build_same_write_if_else_sdfg():
    """Builds an SDFG that computes
        if c: A[0] = B[0] + 1.0
        else: A[0] = B[0] - 1.0
    where ``c`` is a scalar bool symbol.
    """
    sdfg = dace.SDFG("if_else_same_write")
    sdfg.add_array("A", shape=(1, ), dtype=dace.float64)
    sdfg.add_array("B", shape=(1, ), dtype=dace.float64)
    sdfg.add_symbol("c", dace.bool_)

    entry = sdfg.add_state("entry", is_start_block=True)
    exit_state = sdfg.add_state("exit")

    cb = ConditionalBlock("cb")
    sdfg.add_node(cb)
    sdfg.add_edge(entry, cb, dace.InterstateEdge())
    sdfg.add_edge(cb, exit_state, dace.InterstateEdge())

    then_cfr = ControlFlowRegion("then_cfr", sdfg=sdfg)
    ts = then_cfr.add_state("then_s", is_start_block=True)
    rB = ts.add_access("B")
    wA = ts.add_access("A")
    tt = ts.add_tasklet("plus", {"_b"}, {"_a"}, "_a = _b + 1.0")
    ts.add_edge(rB, None, tt, "_b", dace.Memlet("B[0]"))
    ts.add_edge(tt, "_a", wA, None, dace.Memlet("A[0]"))
    cb.add_branch(CodeBlock("c"), then_cfr)

    else_cfr = ControlFlowRegion("else_cfr", sdfg=sdfg)
    es = else_cfr.add_state("else_s", is_start_block=True)
    rB2 = es.add_access("B")
    wA2 = es.add_access("A")
    te = es.add_tasklet("minus", {"_b"}, {"_a"}, "_a = _b - 1.0")
    es.add_edge(rB2, None, te, "_b", dace.Memlet("B[0]"))
    es.add_edge(te, "_a", wA2, None, dace.Memlet("A[0]"))
    cb.add_branch(None, else_cfr)

    return sdfg


def test_pass_removes_conditional_block_and_inserts_three_states():
    sdfg = _build_same_write_if_else_sdfg()
    rewritten = SameWriteSetIfElseToITECFG().apply_pass(sdfg, {})
    assert rewritten == 1

    blocks = list(sdfg.all_control_flow_blocks())
    assert not any(isinstance(b, ConditionalBlock) for b in blocks), [b.label for b in blocks]
    labels = {s.label for s in sdfg.states()}
    assert any(l.startswith("compute_then_") for l in labels)
    assert any(l.startswith("compute_else_") for l in labels)
    assert any(l.startswith("apply_ITE_") for l in labels)


def test_pass_creates_then_else_transients_with_matching_dtype_and_shape():
    sdfg = _build_same_write_if_else_sdfg()
    SameWriteSetIfElseToITECFG().apply_pass(sdfg, {})
    base = sdfg.arrays["A"]
    then_names = [n for n in sdfg.arrays if n.startswith("_then_A")]
    else_names = [n for n in sdfg.arrays if n.startswith("_else_A")]
    assert len(then_names) == 1
    assert len(else_names) == 1
    for n in (then_names[0], else_names[0]):
        arr = sdfg.arrays[n]
        assert arr.dtype == base.dtype
        assert tuple(arr.shape) == tuple(base.shape)
        assert arr.transient is True
        assert arr.storage == dace.dtypes.StorageType.Register


def test_pass_does_not_match_disjoint_write_set():
    """When the two arms write to *different* arrays, the pass leaves the
    ConditionalBlock alone."""
    sdfg = dace.SDFG("if_else_disjoint")
    sdfg.add_array("A", shape=(1, ), dtype=dace.float64)
    sdfg.add_array("B", shape=(1, ), dtype=dace.float64)
    sdfg.add_array("Src", shape=(1, ), dtype=dace.float64)
    sdfg.add_symbol("c", dace.bool_)
    entry = sdfg.add_state("entry", is_start_block=True)
    cb = ConditionalBlock("cb")
    sdfg.add_node(cb)
    sdfg.add_edge(entry, cb, dace.InterstateEdge())
    then_cfr = ControlFlowRegion("then_cfr", sdfg=sdfg)
    ts = then_cfr.add_state("ts", is_start_block=True)
    r = ts.add_access("Src")
    w = ts.add_access("A")
    t = ts.add_tasklet("t", {"_b"}, {"_a"}, "_a = _b + 1.0")
    ts.add_edge(r, None, t, "_b", dace.Memlet("Src[0]"))
    ts.add_edge(t, "_a", w, None, dace.Memlet("A[0]"))
    cb.add_branch(CodeBlock("c"), then_cfr)
    else_cfr = ControlFlowRegion("else_cfr", sdfg=sdfg)
    es = else_cfr.add_state("es", is_start_block=True)
    r2 = es.add_access("Src")
    w2 = es.add_access("B")
    te = es.add_tasklet("t2", {"_b"}, {"_a"}, "_a = _b - 1.0")
    es.add_edge(r2, None, te, "_b", dace.Memlet("Src[0]"))
    es.add_edge(te, "_a", w2, None, dace.Memlet("B[0]"))
    cb.add_branch(None, else_cfr)

    rewritten = SameWriteSetIfElseToITECFG().apply_pass(sdfg, {})
    assert rewritten is None
    assert any(isinstance(b, ConditionalBlock) for b in sdfg.all_control_flow_blocks())


def test_pass_emits_ite_tasklet_with_cond_in_code():
    sdfg = _build_same_write_if_else_sdfg()
    SameWriteSetIfElseToITECFG().apply_pass(sdfg, {})
    found = False
    for state in sdfg.states():
        for n in state.nodes():
            if isinstance(n, dace.nodes.Tasklet) and n.label.startswith("ITE_"):
                assert "ITE(c," in n.code.as_string.replace(" ", "")
                found = True
    assert found, "no ITE_ tasklet emitted"


def test_pass_numerical_correctness():
    """End-to-end compile-run. Reference is the scalar branch expressed in
    plain Python -- not another SDFG."""
    sdfg = _build_same_write_if_else_sdfg()
    SameWriteSetIfElseToITECFG().apply_pass(sdfg, {})

    def reference(c: bool, B: np.ndarray) -> np.ndarray:
        return np.array([B[0] + 1.0 if c else B[0] - 1.0], dtype=np.float64)

    csdfg = sdfg.compile()
    for c in (True, False):
        for b in (-2.0, 0.5, 7.0):
            A = np.zeros((1, ), dtype=np.float64)
            B = np.array([b], dtype=np.float64)
            csdfg(A=A, B=B, c=c)
            expected = reference(c, B)
            np.testing.assert_allclose(A, expected, err_msg=f"c={c}, b={b}, got={A}, want={expected}")


def test_pass_is_idempotent_after_first_run():
    sdfg = _build_same_write_if_else_sdfg()
    first = SameWriteSetIfElseToITECFG().apply_pass(sdfg, {})
    second = SameWriteSetIfElseToITECFG().apply_pass(sdfg, {})
    assert first == 1
    assert second is None


# ---------------------------------------------------------------------------
# Use-count gating, _symbol_has_external_consumer
# ---------------------------------------------------------------------------


def _build_sdfg_with_cb_only(sym_name: str):
    """Skeleton: entry -> cb (cond=sym_name) -> exit, where sym_name is
    assigned on the entry->cb interstate edge. The cb cond is the *only*
    consumer of sym_name."""
    sdfg = dace.SDFG(f"only_{sym_name}")
    sdfg.add_array("A", shape=(1, ), dtype=dace.float64)
    sdfg.add_symbol(sym_name, dace.bool_)
    entry = sdfg.add_state("entry", is_start_block=True)
    exit_state = sdfg.add_state("exit")
    cb = ConditionalBlock("cb")
    sdfg.add_node(cb)
    sdfg.add_edge(entry, cb, dace.InterstateEdge(assignments={sym_name: "True"}))
    sdfg.add_edge(cb, exit_state, dace.InterstateEdge())
    body = ControlFlowRegion("then_body", sdfg=sdfg)
    body.add_state("body_state", is_start_block=True)
    cb.add_branch(CodeBlock(sym_name), body)
    defining = next(e for e in sdfg.edges() if sym_name in (e.data.assignments or {}))
    return sdfg, cb, defining


def test_symbol_has_external_consumer_when_only_cb_uses_it():
    """Only consumer is the cb's own branch condition (which the pass is
    about to rewrite away). With ``skip_cb=cb`` the helper sees no
    external consumer."""
    sdfg, cb, defining = _build_sdfg_with_cb_only("flag")
    assert _symbol_has_external_consumer(sdfg, "flag", defining, skip_cb=cb) is False


def test_symbol_has_external_consumer_when_downstream_assignment_reads_it():
    """A second interstate edge after the cb assigns ``out_sym = flag``,
    that second use counts as external."""
    sdfg, cb, defining = _build_sdfg_with_cb_only("flag")
    sdfg.add_symbol("out_sym", dace.bool_)
    after = sdfg.add_state("after_cb")
    exit_state = next(s for s in sdfg.states() if s.label == "exit")
    for e in list(sdfg.out_edges(cb)):
        sdfg.remove_edge(e)
    sdfg.add_edge(cb, after, dace.InterstateEdge(assignments={"out_sym": "flag"}))
    sdfg.add_edge(after, exit_state, dace.InterstateEdge())
    assert _symbol_has_external_consumer(sdfg, "flag", defining, skip_cb=cb) is True


def test_symbol_has_external_consumer_when_sibling_cb_condition_reads_it():
    sdfg, cb, defining = _build_sdfg_with_cb_only("flag")
    sibling = ConditionalBlock("cb_sibling")
    sdfg.add_node(sibling)
    sibling.add_branch(CodeBlock("flag"), ControlFlowRegion("sib_body", sdfg=sdfg))
    sibling.branches[0][1].add_state("sib_state", is_start_block=True)
    exit_state = next(s for s in sdfg.states() if s.label == "exit")
    for e in list(sdfg.out_edges(cb)):
        sdfg.remove_edge(e)
    sdfg.add_edge(cb, sibling, dace.InterstateEdge())
    sdfg.add_edge(sibling, exit_state, dace.InterstateEdge())
    assert _symbol_has_external_consumer(sdfg, "flag", defining, skip_cb=cb) is True


def test_symbol_has_external_consumer_when_tasklet_body_reads_it():
    sdfg, cb, defining = _build_sdfg_with_cb_only("flag")
    exit_state = next(s for s in sdfg.states() if s.label == "exit")
    exit_state.add_tasklet("uses_flag", set(), set(), "x = flag")
    assert _symbol_has_external_consumer(sdfg, "flag", defining, skip_cb=cb) is True


def test_symbol_has_external_consumer_when_interstate_condition_reads_it():
    sdfg, cb, defining = _build_sdfg_with_cb_only("flag")
    exit_state = next(s for s in sdfg.states() if s.label == "exit")
    for e in list(sdfg.out_edges(cb)):
        sdfg.remove_edge(e)
    sdfg.add_edge(cb, exit_state, dace.InterstateEdge(condition=CodeBlock("flag")))
    assert _symbol_has_external_consumer(sdfg, "flag", defining, skip_cb=cb) is True
