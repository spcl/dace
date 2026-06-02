# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Pass-level tests for ``SameWriteSetIfElseToMergeCFG``.

After running, an SDFG that originally contained a same-write-set ``if/else``
must:
- have *no* remaining ``ConditionalBlock``,
- have three new states ``compute_then`` / ``compute_else`` / ``apply_merge``
  wired in sequence,
- produce numerically the same result as the original SDFG.

The third assertion uses ``sdfg.compile()`` end-to-end. Per the project rule,
the reference is an unfolded scalar Python evaluation — not a different
SDFG variant.
"""
import os

import numpy as np

import dace
from dace.properties import CodeBlock
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion
from dace.transformation.passes.vectorization.same_write_set_if_else_to_merge_cfg import (
    SameWriteSetIfElseToMergeCFG,
    _symbol_has_external_consumer,
)

# This pass emits ``merge(...)`` tasklets which need ``dace/merge.h``.
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
    rewritten = SameWriteSetIfElseToMergeCFG().apply_pass(sdfg, {})
    assert rewritten == 1

    blocks = list(sdfg.all_control_flow_blocks())
    assert not any(isinstance(b, ConditionalBlock) for b in blocks), [b.label for b in blocks]
    labels = {s.label for s in sdfg.states()}
    assert any(l.startswith("compute_then_") for l in labels)
    assert any(l.startswith("compute_else_") for l in labels)
    assert any(l.startswith("apply_merge_") for l in labels)


def test_pass_creates_then_else_transients_with_matching_dtype_and_shape():
    sdfg = _build_same_write_if_else_sdfg()
    SameWriteSetIfElseToMergeCFG().apply_pass(sdfg, {})
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

    rewritten = SameWriteSetIfElseToMergeCFG().apply_pass(sdfg, {})
    assert rewritten is None
    assert any(isinstance(b, ConditionalBlock) for b in sdfg.all_control_flow_blocks())


def test_pass_emits_merge_tasklet_with_cond_in_code():
    sdfg = _build_same_write_if_else_sdfg()
    SameWriteSetIfElseToMergeCFG().apply_pass(sdfg, {})
    found = False
    for state in sdfg.states():
        for n in state.nodes():
            if isinstance(n, dace.nodes.Tasklet) and n.label.startswith("merge_"):
                assert "merge(c," in n.code.as_string.replace(" ", "")
                found = True
    assert found, "no merge_ tasklet emitted"


def test_pass_numerical_correctness():
    """End-to-end compile-run. Reference is the scalar branch expressed in
    plain Python — not another SDFG."""
    sdfg = _build_same_write_if_else_sdfg()
    SameWriteSetIfElseToMergeCFG().apply_pass(sdfg, {})

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
    first = SameWriteSetIfElseToMergeCFG().apply_pass(sdfg, {})
    second = SameWriteSetIfElseToMergeCFG().apply_pass(sdfg, {})
    assert first == 1
    assert second is None


# ---------------------------------------------------------------------------
# Two-arm same-write-set with the python-frontend empty entry-state pattern.
# Each arm begins with an EMPTY state whose only out-edge carries an
# interstate symbol binding (e.g. ``__sym_z = z``); the substantive compute
# state follows. Without the pre-match hoist that collapses the entry state,
# ``_matches`` (single-state guard) would refuse this kernel and the broken
# sequential single-arm fallback in ``BranchNormalization`` would silently
# wire ``_old`` to the original output array, breaking the dataflow.
# These tests pin the hoist + entry-state removal.
# ---------------------------------------------------------------------------


def _build_same_write_if_else_with_empty_entry_states_sdfg():
    """Builds an SDFG matching the cloudsc-snippet-one shape: a two-arm
    same-write-set ``if/else`` whose IF arm starts with an empty state
    that only carries an interstate symbol binding ``__sym_z = z``."""
    sdfg = dace.SDFG("if_else_with_empty_entry")
    sdfg.add_array("A", shape=(1, ), dtype=dace.float64)
    sdfg.add_array("B", shape=(1, ), dtype=dace.float64)
    sdfg.add_symbol("c", dace.bool_)
    sdfg.add_symbol("z", dace.int64)
    sdfg.add_symbol("__sym_z", dace.int64)

    entry = sdfg.add_state("entry", is_start_block=True)
    exit_state = sdfg.add_state("exit")

    cb = ConditionalBlock("cb")
    sdfg.add_node(cb)
    sdfg.add_edge(entry, cb, dace.InterstateEdge())
    sdfg.add_edge(cb, exit_state, dace.InterstateEdge())

    # IF arm: empty entry state with assignment ``__sym_z = z``, then a
    # compute state that writes A = B + 1.0.
    then_cfr = ControlFlowRegion("then_cfr", sdfg=sdfg)
    then_entry = then_cfr.add_state("then_entry", is_start_block=True)
    then_compute = then_cfr.add_state("then_compute")
    then_cfr.add_edge(then_entry, then_compute, dace.InterstateEdge(assignments={"__sym_z": "z"}))
    rB_t = then_compute.add_access("B")
    wA_t = then_compute.add_access("A")
    tt = then_compute.add_tasklet("plus", {"_b"}, {"_a"}, "_a = _b + 1.0")
    then_compute.add_edge(rB_t, None, tt, "_b", dace.Memlet("B[0]"))
    then_compute.add_edge(tt, "_a", wA_t, None, dace.Memlet("A[0]"))
    cb.add_branch(CodeBlock("c"), then_cfr)

    # ELSE arm: single substantive state A = B - 1.0.
    else_cfr = ControlFlowRegion("else_cfr", sdfg=sdfg)
    es = else_cfr.add_state("else_compute", is_start_block=True)
    rB_e = es.add_access("B")
    wA_e = es.add_access("A")
    te = es.add_tasklet("minus", {"_b"}, {"_a"}, "_a = _b - 1.0")
    es.add_edge(rB_e, None, te, "_b", dace.Memlet("B[0]"))
    es.add_edge(te, "_a", wA_e, None, dace.Memlet("A[0]"))
    cb.add_branch(None, else_cfr)

    return sdfg


def test_empty_entry_state_pre_hoist_blocks_match():
    """Without the hoist + entry-state removal, ``_matches`` would refuse
    this kernel (the IF arm has 2 nodes, not 1). Verify the bare match
    check fails before ``apply_pass`` runs the hoist."""
    sdfg = _build_same_write_if_else_with_empty_entry_states_sdfg()
    cbs = [b for b in sdfg.all_control_flow_blocks() if isinstance(b, ConditionalBlock)]
    assert len(cbs) == 1
    matches = SameWriteSetIfElseToMergeCFG()._matches(cbs[0])
    assert matches is False, "raw _matches must reject multi-state arms"


def test_apply_pass_hoists_empty_entry_state_and_matches():
    """``apply_pass`` runs the hoist preprocessor first, dropping the
    empty entry state. The arm becomes single-state, ``_matches``
    accepts the kernel, and M3.1b rewrites it into the per-arm temp +
    merge shape."""
    sdfg = _build_same_write_if_else_with_empty_entry_states_sdfg()
    rewritten = SameWriteSetIfElseToMergeCFG().apply_pass(sdfg, {})
    assert rewritten == 1
    assert not any(isinstance(b, ConditionalBlock) for b in sdfg.all_control_flow_blocks())
    labels = {s.label for s in sdfg.states()}
    assert any(l.startswith("compute_then_") for l in labels)
    assert any(l.startswith("compute_else_") for l in labels)
    assert any(l.startswith("apply_merge_") for l in labels)
    # The per-arm temp transients must exist (single-arm fallback never
    # allocates ``_then_*`` / ``_else_*`` — its presence is what proves
    # the kernel took the right path).
    assert any(n.startswith("_then_A") for n in sdfg.arrays)
    assert any(n.startswith("_else_A") for n in sdfg.arrays)


def test_empty_entry_state_hoists_assignment_to_pre_cb_edge():
    """After the hoist, the ``__sym_z = z`` assignment must live on the
    edge entering the ConditionalBlock (or the post-hoist state structure
    it became), not inside the arm. Otherwise it would still be a per-arm
    binding the merge dataflow can't honour."""
    sdfg = _build_same_write_if_else_with_empty_entry_states_sdfg()
    SameWriteSetIfElseToMergeCFG().apply_pass(sdfg, {})
    # After rewrite, the cb is gone — collect every assignment on every
    # interstate edge that survived. The hoisted ``__sym_z = z`` must
    # appear somewhere in the SDFG's interstate-edge assignments.
    found = False
    for e in sdfg.all_interstate_edges():
        if "__sym_z" in e.data.assignments and str(e.data.assignments["__sym_z"]) == "z":
            found = True
            break
    assert found, "__sym_z = z must be hoisted onto an interstate edge surviving the rewrite"


def test_empty_entry_state_numerical_correctness():
    """End-to-end compile-run on the multi-state-arm shape; reference is
    plain Python."""
    sdfg = _build_same_write_if_else_with_empty_entry_states_sdfg()
    SameWriteSetIfElseToMergeCFG().apply_pass(sdfg, {})

    def reference(c: bool, B: np.ndarray) -> np.ndarray:
        return np.array([B[0] + 1.0 if c else B[0] - 1.0], dtype=np.float64)

    csdfg = sdfg.compile()
    for c in (True, False):
        for b in (-2.0, 0.5, 7.0):
            A = np.zeros((1, ), dtype=np.float64)
            B = np.array([b], dtype=np.float64)
            csdfg(A=A, B=B, c=c, z=1)
            expected = reference(c, B)
            np.testing.assert_allclose(A, expected, err_msg=f"c={c}, b={b}, got={A}, want={expected}")


def _build_two_writes_per_arm_sdfg():
    """Builds the cloudsc-snippet-one structural shape directly: two-arm
    same-write-set where each arm writes TWO arrays (A and C, with
    C = 1 - A in the IF arm). Single-state arms (this test verifies the
    plain two-write match — the multi-state companion is covered above)."""
    sdfg = dace.SDFG("if_else_two_writes")
    sdfg.add_array("A", shape=(1, ), dtype=dace.float64)
    sdfg.add_array("B", shape=(1, ), dtype=dace.float64)
    sdfg.add_array("C", shape=(1, ), dtype=dace.float64)
    sdfg.add_symbol("c", dace.bool_)

    entry = sdfg.add_state("entry", is_start_block=True)
    exit_state = sdfg.add_state("exit")

    cb = ConditionalBlock("cb")
    sdfg.add_node(cb)
    sdfg.add_edge(entry, cb, dace.InterstateEdge())
    sdfg.add_edge(cb, exit_state, dace.InterstateEdge())

    then_cfr = ControlFlowRegion("then_cfr", sdfg=sdfg)
    ts = then_cfr.add_state("ts", is_start_block=True)
    rB = ts.add_access("B")
    wA = ts.add_access("A")
    rA = ts.add_access("A")
    wC = ts.add_access("C")
    tA = ts.add_tasklet("compA", {"_b"}, {"_a"}, "_a = _b + 1.0")
    tC = ts.add_tasklet("compC", {"_a"}, {"_c"}, "_c = 1.0 - _a")
    ts.add_edge(rB, None, tA, "_b", dace.Memlet("B[0]"))
    ts.add_edge(tA, "_a", wA, None, dace.Memlet("A[0]"))
    ts.add_edge(wA, None, rA, None, dace.Memlet())
    ts.add_edge(rA, None, tC, "_a", dace.Memlet("A[0]"))
    ts.add_edge(tC, "_c", wC, None, dace.Memlet("C[0]"))
    cb.add_branch(CodeBlock("c"), then_cfr)

    else_cfr = ControlFlowRegion("else_cfr", sdfg=sdfg)
    es = else_cfr.add_state("es", is_start_block=True)
    wA2 = es.add_access("A")
    wC2 = es.add_access("C")
    tA2 = es.add_tasklet("zA", {}, {"_a"}, "_a = 0.0")
    tC2 = es.add_tasklet("zC", {}, {"_c"}, "_c = 0.0")
    es.add_edge(tA2, "_a", wA2, None, dace.Memlet("A[0]"))
    es.add_edge(tC2, "_c", wC2, None, dace.Memlet("C[0]"))
    cb.add_branch(None, else_cfr)

    return sdfg


def test_two_writes_per_arm_uses_per_arm_temps():
    """Two arms each writing TWO arrays must produce TWO per-arm temp
    pairs and TWO merge tasklets — the proper per-arm-temp path, not the
    broken sequential single-arm chain that would write through the
    original arrays back-to-back."""
    sdfg = _build_two_writes_per_arm_sdfg()
    SameWriteSetIfElseToMergeCFG().apply_pass(sdfg, {})
    for arr in ("A", "C"):
        assert any(n.startswith(f"_then_{arr}") for n in sdfg.arrays), f"missing _then_{arr}"
        assert any(n.startswith(f"_else_{arr}") for n in sdfg.arrays), f"missing _else_{arr}"
    merge_tasklets = [n for state in sdfg.states() for n in state.nodes()
                      if isinstance(n, dace.nodes.Tasklet) and n.label.startswith("merge_")]
    assert len(merge_tasklets) == 2, f"expected 2 merges (one per written array), got {len(merge_tasklets)}"


def test_two_writes_per_arm_numerical_correctness():
    """End-to-end on the two-writes-per-arm shape; reference is plain Python.
    Each arm's writes flow through their per-arm temps and combine in the
    merge — verifying the second merge does not clobber the first."""
    sdfg = _build_two_writes_per_arm_sdfg()
    SameWriteSetIfElseToMergeCFG().apply_pass(sdfg, {})
    csdfg = sdfg.compile()

    def reference(c: bool, B: np.ndarray):
        if c:
            A = np.array([B[0] + 1.0], dtype=np.float64)
            C = np.array([1.0 - A[0]], dtype=np.float64)
        else:
            A = np.array([0.0], dtype=np.float64)
            C = np.array([0.0], dtype=np.float64)
        return A, C

    for c in (True, False):
        for b in (-2.0, 0.5, 7.0):
            A = np.zeros((1, ), dtype=np.float64)
            B = np.array([b], dtype=np.float64)
            C = np.zeros((1, ), dtype=np.float64)
            csdfg(A=A, B=B, C=C, c=c)
            exp_A, exp_C = reference(c, B)
            np.testing.assert_allclose(A, exp_A, err_msg=f"A: c={c}, b={b}, got={A}, want={exp_A}")
            np.testing.assert_allclose(C, exp_C, err_msg=f"C: c={c}, b={b}, got={C}, want={exp_C}")


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
