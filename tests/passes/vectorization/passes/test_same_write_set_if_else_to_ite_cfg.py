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
the reference is an unfolded scalar Python evaluation — not a different
SDFG variant.
"""
import numpy as np
import pytest

import dace
from dace.config import set_temporary
from dace.properties import CodeBlock
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion
from dace.transformation.passes.vectorization.same_write_set_if_else_to_ite_cfg import (
    SameWriteSetIfElseToITECFG,
    _symbol_has_external_consumer,
)


@pytest.fixture(autouse=True)
def blank_cpu_args():
    """This pass emits ``merge(...)`` tasklets which need ``dace/ITE.h``.

    Scoped to this module rather than written into ``os.environ`` at import time: a marker
    expression only DESELECTS, so the module is still imported in every xdist worker of every
    lane, and ``Config.get`` reads the environment ahead of the config -- which pinned empty
    compiler flags process-wide and left nothing able to restore them.
    """
    with set_temporary('compiler', 'cpu', 'args', value=''):
        yield


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
    assert any(lbl.startswith("compute_then_") for lbl in labels)
    assert any(lbl.startswith("compute_else_") for lbl in labels)
    assert any(lbl.startswith("apply_ITE_") for lbl in labels)


def test_pass_creates_then_else_transients_with_matching_dtype():
    """The per-arm temps inherit the base array's dtype but are always
    shape ``(1,)`` (element-wise writes only need one scratch element)."""
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
        assert tuple(
            arr.shape) == (1, ), (f"per-arm temp {n!r} shape must be (1,) (element-wise scratch); got {arr.shape}")
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


def test_pass_emits_ITE_tasklet_with_cond_in_code():
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
    plain Python — not another SDFG."""
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
    matches = SameWriteSetIfElseToITECFG()._matches(cbs[0])
    assert matches is False, "raw _matches must reject multi-state arms"


def test_apply_pass_hoists_empty_entry_state_and_matches():
    """``apply_pass`` runs the hoist preprocessor first, dropping the
    empty entry state. The arm becomes single-state, ``_matches``
    accepts the kernel, and M3.1b rewrites it into the per-arm temp +
    merge shape."""
    sdfg = _build_same_write_if_else_with_empty_entry_states_sdfg()
    rewritten = SameWriteSetIfElseToITECFG().apply_pass(sdfg, {})
    assert rewritten == 1
    assert not any(isinstance(b, ConditionalBlock) for b in sdfg.all_control_flow_blocks())
    labels = {s.label for s in sdfg.states()}
    assert any(lbl.startswith("compute_then_") for lbl in labels)
    assert any(lbl.startswith("compute_else_") for lbl in labels)
    assert any(lbl.startswith("apply_ITE_") for lbl in labels)
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
    SameWriteSetIfElseToITECFG().apply_pass(sdfg, {})
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
    SameWriteSetIfElseToITECFG().apply_pass(sdfg, {})

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
    pairs and TWO ITE tasklets — the proper per-arm-temp path, not the
    broken sequential single-arm chain that would write through the
    original arrays back-to-back."""
    sdfg = _build_two_writes_per_arm_sdfg()
    SameWriteSetIfElseToITECFG().apply_pass(sdfg, {})
    for arr in ("A", "C"):
        assert any(n.startswith(f"_then_{arr}") for n in sdfg.arrays), f"missing _then_{arr}"
        assert any(n.startswith(f"_else_{arr}") for n in sdfg.arrays), f"missing _else_{arr}"
    ITE_tasklets = [
        n for state in sdfg.states() for n in state.nodes()
        if isinstance(n, dace.nodes.Tasklet) and n.label.startswith("ITE_")
    ]
    assert len(ITE_tasklets) == 2, f"expected 2 ITEs (one per written array), got {len(ITE_tasklets)}"


def _build_2d_base_array_sdfg():
    """Build the cloudsc-snippet-one shape: per-arm temps for arrays with
    symbolic 2D shape ``(N, M)`` and per-element writes ``arr[j, i]``.
    Pre-shape-fix this allocated each temp as a kernel-sized symbolic
    array (which the codegen would heap-allocate, defeating the
    Register storage hint). The (1,)-shape fix keeps the temp Register-
    allocable on every backend even when the BASE array is symbolic."""
    sdfg = dace.SDFG("if_else_2d_base")
    N = dace.symbol("N", dace.int64)
    M = dace.symbol("M", dace.int64)
    sdfg.add_array("A", shape=(N, M), dtype=dace.float64)
    sdfg.add_array("B", shape=(N, M), dtype=dace.float64)
    sdfg.add_symbol("c", dace.bool_)
    sdfg.add_symbol("j", dace.int64)
    sdfg.add_symbol("i", dace.int64)

    entry = sdfg.add_state("entry", is_start_block=True)
    cb = ConditionalBlock("cb")
    sdfg.add_node(cb)
    sdfg.add_edge(entry, cb, dace.InterstateEdge())

    then_cfr = ControlFlowRegion("then_cfr", sdfg=sdfg)
    ts = then_cfr.add_state("ts", is_start_block=True)
    rB = ts.add_access("B")
    wA = ts.add_access("A")
    tt = ts.add_tasklet("plus", {"_b"}, {"_a"}, "_a = _b + 1.0")
    ts.add_edge(rB, None, tt, "_b", dace.Memlet("B[j, i]"))
    ts.add_edge(tt, "_a", wA, None, dace.Memlet("A[j, i]"))
    cb.add_branch(CodeBlock("c"), then_cfr)

    else_cfr = ControlFlowRegion("else_cfr", sdfg=sdfg)
    es = else_cfr.add_state("es", is_start_block=True)
    rB2 = es.add_access("B")
    wA2 = es.add_access("A")
    te = es.add_tasklet("minus", {"_b"}, {"_a"}, "_a = _b - 1.0")
    es.add_edge(rB2, None, te, "_b", dace.Memlet("B[j, i]"))
    es.add_edge(te, "_a", wA2, None, dace.Memlet("A[j, i]"))
    cb.add_branch(None, else_cfr)

    return sdfg


def test_2d_base_array_yields_scalar_per_arm_temps():
    """Per-arm temps for a symbolic-2D-shaped base ``A[N, M]`` must STILL
    be (1,)-shaped — otherwise codegen would heap-allocate them and the
    K-dim tile path could not register-promote the scratch."""
    sdfg = _build_2d_base_array_sdfg()
    SameWriteSetIfElseToITECFG().apply_pass(sdfg, {})
    for prefix in ("_then_A", "_else_A"):
        names = [n for n in sdfg.arrays if n.startswith(prefix)]
        assert len(names) == 1, f"expected one {prefix} transient, got {names}"
        arr = sdfg.arrays[names[0]]
        assert tuple(
            arr.shape) == (1, ), (f"per-arm temp for 2D base must remain (1,) (Register-allocable); got {arr.shape}")


def test_2d_base_array_memlets_subset_zero_on_temps():
    """The cloned arm writes ``A[j, i]`` and the merge reads ``A[j, i]``
    before redirect; after redirect the writes/reads target the (1,)-shaped
    temps, so the subsets must become ``[0]`` — otherwise the codegen
    would emit out-of-bounds ``temp[j, i]`` accesses."""
    sdfg = _build_2d_base_array_sdfg()
    SameWriteSetIfElseToITECFG().apply_pass(sdfg, {})
    for state in sdfg.states():
        for e in state.edges():
            if e.data.data is None:
                continue
            if e.data.data.startswith("_then_") or e.data.data.startswith("_else_"):
                assert str(e.data.subset) == "0", (
                    f"per-arm temp memlet must read/write ``[0]``; got {e.data.subset} on {e.data.data}")


def test_two_writes_per_arm_numerical_correctness():
    """End-to-end on the two-writes-per-arm shape; reference is plain Python.
    Each arm's writes flow through their per-arm temps and combine in the
    merge — verifying the second merge does not clobber the first."""
    sdfg = _build_two_writes_per_arm_sdfg()
    SameWriteSetIfElseToITECFG().apply_pass(sdfg, {})
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


def test_promote_gather_indices_rewrites_nested_subscript():
    """A gather cond value ``w[idx[i], k]`` (nested subscript no plain memlet can express)
    is rewritten to ``w[_gidx_0, k]`` by promoting the inner index read ``idx[i]`` to a
    fresh interstate INTEGER symbol assigned on the defining edge -- the body-gather
    representation the tile machinery vectorizes (TSVC ``loop_to_map_threshold_gather``)."""
    n = dace.symbol("n")
    sdfg = dace.SDFG("gather_cond")
    sdfg.add_array("w", [n, n], dace.float64)
    sdfg.add_array("idx", [n], dace.int64)
    sdfg.add_symbol("i", dace.int64)  # index symbols must be in scope to hoist idx[i] onto the edge
    sdfg.add_symbol("k", dace.int64)
    s0 = sdfg.add_state("s0", is_start_block=True)
    s1 = sdfg.add_state("s1")
    edge = sdfg.add_edge(s0, s1, dace.InterstateEdge(assignments={"w_index": "w[idx[i], k]"}))

    p = SameWriteSetIfElseToITECFG()
    new_rhs = p._promote_gather_indices(sdfg, [edge], "w[idx[i], k]")
    assert new_rhs == "w[_gidx_0, k]", new_rhs
    # The nested index was promoted to a fresh int symbol defined on the edge.
    assert edge.data.assignments.get("_gidx_0") == "idx[i]"
    assert "_gidx_0" in sdfg.symbols and sdfg.symbols["_gidx_0"] == dace.int64
    # No-op on an affine (non-gather) read -- the subset has no nested subscript.
    assert p._promote_gather_indices(sdfg, [edge], "w[i, k]") == "w[i, k]"


def test_promote_gather_indices_noop_without_edge():
    """No defining edge (dominance target) -> the rewrite is a safe no-op."""
    sdfg = dace.SDFG("no_edge")
    sdfg.add_array("w", [dace.symbol("n")], dace.float64)
    sdfg.add_array("idx", [dace.symbol("n")], dace.int64)
    p = SameWriteSetIfElseToITECFG()
    assert p._promote_gather_indices(sdfg, None, "w[idx[i]]") == "w[idx[i]]"


# ---------------------------------------------------------------------------
# Array-predicate guard lifting (``_lift_array_predicate_cond``).
# ---------------------------------------------------------------------------


def test_lift_array_predicate_cond_stages_array_read_as_connector():
    """An array-subscript guard is staged via an in-connector, never inlined
    (regression: tsvc_2_5 ``masked_store_sym`` / ``move_if_data_dep_nest``)."""
    sdfg = dace.SDFG("lift_array_pred")
    sdfg.add_array("A", (16, ), dace.float64)
    sdfg.add_symbol("K", dace.float64)
    state = sdfg.add_state("s", is_start_block=True)
    res = SameWriteSetIfElseToITECFG()._lift_array_predicate_cond(sdfg, state, "A > K", "i")
    assert res is not None, "an array-reading guard must be staged, not left as inline free-symbol text"
    cond_name, _ = res
    assert sdfg.arrays[cond_name].dtype == dace.bool_, "lifted guard transient must be bool"
    tasklets = [n for n in state.nodes() if isinstance(n, dace.nodes.Tasklet)]
    assert len(tasklets) == 1
    body = tasklets[0].code.as_string
    assert "A" not in body.replace("_in_A", ""), f"array head A must not be inlined in the tasklet body: {body!r}"
    a_edges = [e for e in state.in_edges(tasklets[0]) if e.data is not None and e.data.data == "A"]
    assert len(a_edges) == 1, "A must be wired as exactly one in-connector"
    assert str(a_edges[0].data.subset) == "i", "the bare array read is staged at the write's per-lane subset"


def test_lift_array_predicate_cond_skips_pure_symbol_condition():
    """A pure-symbol guard has nothing to stage; helper returns None."""
    sdfg = dace.SDFG("lift_pure_sym")
    sdfg.add_symbol("K", dace.float64)
    state = sdfg.add_state("s", is_start_block=True)
    assert SameWriteSetIfElseToITECFG()._lift_array_predicate_cond(sdfg, state, "K > 0", "i") is None


# ---------------------------------------------------------------------------
# s279-shaped mixed cond: array transient vs interstate staged array read.
# ---------------------------------------------------------------------------


def build_s279_shaped_guard_sdfg():
    """Single-arm guard whose condition MIXES a directly-read array transient
    (``b_index_0``) with an interstate-defined staged read of an array
    (``a_index_0 = a[0]``) -- the shape TSVC s279's nested ``if b[i] > a[i]`` takes
    after canonicalization::

        b_index_0 = b[0]              (staged in entry)
        a_index_0 = a[0]              (interstate assignment)
        if b_index_0 > a_index_0:
            c[0] = c[0] + 1.0

    The array-predicate recipe fires (``b_index_0`` is an array) but must also inline
    the interstate staged read ``a_index_0`` into ``a[0]`` and stage it through a
    connector -- else ``a_index_0`` is left as free-symbol text and orphaned when the
    apply-ITE state is relocated ("Missing symbols on nested SDFG").
    """
    sdfg = dace.SDFG("s279_shaped_guard")
    sdfg.add_array("a", (1, ), dace.float64)
    sdfg.add_array("b", (1, ), dace.float64)
    sdfg.add_array("c", (1, ), dace.float64)
    sdfg.add_array("b_index_0", (1, ), dace.float64, transient=True)
    sdfg.add_symbol("a_index_0", dace.float64)

    entry = sdfg.add_state("entry", is_start_block=True)
    rb = entry.add_access("b")
    wbi = entry.add_access("b_index_0")
    tb = entry.add_tasklet("stage_b", {"_b"}, {"_bi"}, "_bi = _b")
    entry.add_edge(rb, None, tb, "_b", dace.Memlet("b[0]"))
    entry.add_edge(tb, "_bi", wbi, None, dace.Memlet("b_index_0[0]"))

    mid = sdfg.add_state("mid")
    sdfg.add_edge(entry, mid, dace.InterstateEdge(assignments={"a_index_0": "a[0]"}))

    cb = ConditionalBlock("guard")
    sdfg.add_node(cb)
    sdfg.add_edge(mid, cb, dace.InterstateEdge())
    exit_s = sdfg.add_state("exit")
    sdfg.add_edge(cb, exit_s, dace.InterstateEdge())

    arm = ControlFlowRegion("arm", sdfg=sdfg)
    s = arm.add_state("arm_s", is_start_block=True)
    rc = s.add_access("c")
    wc = s.add_access("c")
    t = s.add_tasklet("body", {"_c"}, {"_o"}, "_o = _c + 1.0")
    s.add_edge(rc, None, t, "_c", dace.Memlet("c[0]"))
    s.add_edge(t, "_o", wc, None, dace.Memlet("c[0]"))
    cb.add_branch(CodeBlock("(b_index_0 > a_index_0)"), arm)
    return sdfg


def test_pass_converts_mixed_array_and_interstate_staged_read_cond():
    """s279's nested guard: conversion fires and the interstate staged read
    ``a_index_0 = a[0]`` is inlined + dropped (no orphaned free symbol)."""
    sdfg = build_s279_shaped_guard_sdfg()
    rewritten = SameWriteSetIfElseToITECFG().apply_pass(sdfg, {})
    assert rewritten == 1, "the s279-shaped guard must be converted, not refused"

    # No ConditionalBlock remains; the three ITE states are in place.
    assert not any(isinstance(b, ConditionalBlock) for b in sdfg.all_control_flow_blocks())

    # The staged-read symbol is fully eliminated -- not a symbol, not free, and no
    # interstate edge still assigns it (that is what caused the missing-symbol error).
    assert "a_index_0" not in sdfg.symbols
    assert "a_index_0" not in set(map(str, sdfg.free_symbols))
    assert not [
        e for cfg in sdfg.all_control_flow_regions(recursive=True)
        for e in cfg.edges() if "a_index_0" in (e.data.assignments or {})
    ], "the dead a_index_0 assignment must be pruned"

    # The lifted guard stages BOTH operands through in-connectors (never the bare
    # symbol / array head) so both become per-lane tiles downstream.
    lift = [
        n for st in sdfg.states() for n in st.nodes() if isinstance(n, dace.nodes.Tasklet) and "lift_cond" in n.label
    ]
    assert len(lift) == 1, "exactly one lifted cond tasklet expected"
    lift_state = next(st for st in sdfg.states() if lift[0] in st.nodes())
    assert "a_index_0" not in lift[0].code.as_string
    staged = {e.data.data for e in lift_state.in_edges(lift[0])}
    assert staged == {"a", "b_index_0"}, f"both operands must be staged reads, got {staged}"

    sdfg.validate()


def test_pass_mixed_cond_numerical_correctness():
    """End-to-end compile-run of the s279-shaped guard; reference = scalar branch."""
    sdfg = build_s279_shaped_guard_sdfg()
    SameWriteSetIfElseToITECFG().apply_pass(sdfg, {})
    csdfg = sdfg.compile()
    for av in (-1.0, 0.5, 3.0):
        for bv in (-2.0, 1.0, 5.0):
            for cv in (7.0, -4.0):
                a = np.array([av], dtype=np.float64)
                b = np.array([bv], dtype=np.float64)
                c = np.array([cv], dtype=np.float64)
                csdfg(a=a, b=b, c=c)
                expected = cv + 1.0 if bv > av else cv
                np.testing.assert_allclose(c[0], expected, err_msg=f"a={av} b={bv} c={cv}")


# ---------------------------------------------------------------------------
# Extended gather-index promotion (``_promote_gather_indices``) edge cases.
# A guard reading a gather (``w[idx[i], k] > K``) carries a nested subscript no
# plain memlet can express; the lift promotes each nested index ``idx[i]`` to a
# fresh interstate symbol ``_gidx`` assigned on the edge(s) feeding the merge
# state, so the staged read becomes ``w[_gidx, k]``.
# ---------------------------------------------------------------------------


def _gather_merge_two_preds_sdfg():
    """SDFG whose merge state has TWO interstate predecessors, plus arrays ``w``
    (gathered), ``idx`` (index), and a registered loop-iterator symbol ``i``."""
    sdfg = dace.SDFG("gather_two_preds")
    sdfg.add_array("w", (8, 8), dace.float64)
    sdfg.add_array("idx", (8, ), dace.int64)
    sdfg.add_symbol("i", dace.int64)
    start = sdfg.add_state("start", is_start_block=True)
    p1 = sdfg.add_state("p1")
    p2 = sdfg.add_state("p2")
    merge = sdfg.add_state("merge")
    sdfg.add_edge(start, p1, dace.InterstateEdge(condition="i < 4"))
    sdfg.add_edge(start, p2, dace.InterstateEdge(condition="i >= 4"))
    sdfg.add_edge(p1, merge, dace.InterstateEdge())
    sdfg.add_edge(p2, merge, dace.InterstateEdge())
    return sdfg, merge


def test_promote_gather_indices_defines_symbol_on_every_predecessor():
    """The promoted ``_gidx`` must be assigned on EVERY edge into the merge state --
    a merge with several predecessors would otherwise read an unbound ``_gidx`` on
    the path that skipped the single (``[0]``) edge."""
    sdfg, merge = _gather_merge_two_preds_sdfg()
    in_edges = list(merge.parent_graph.in_edges(merge))
    assert len(in_edges) == 2

    out = SameWriteSetIfElseToITECFG()._promote_gather_indices(sdfg, in_edges, "w[idx[i], 0] > 0.0")

    assert "idx[" not in out and "_gidx_0" in out, f"gather index not promoted: {out!r}"
    assigned = [("_gidx_0" in (e.data.assignments or {})) for e in in_edges]
    assert all(assigned), f"_gidx_0 must be assigned on all predecessors, got {assigned}"


def test_promote_gather_indices_refuses_without_a_def_edge():
    """With no edge to hoist the assignment onto (a CFG start block), the nested
    subscript survives unchanged and ``_has_nested_subscript`` flags it so the
    caller refuses the lift instead of emitting a bare-pointer read."""
    sdfg, _merge = _gather_merge_two_preds_sdfg()
    p = SameWriteSetIfElseToITECFG()
    rhs = "w[idx[i], 0] > 0.0"

    out = p._promote_gather_indices(sdfg, [], rhs)

    assert out == rhs, "no def edge -> promotion must be a no-op"
    assert p._has_nested_subscript(sdfg, out), "surviving gather must be flagged so the caller refuses"


def test_promote_gather_indices_refuses_out_of_scope_index_symbol():
    """An index symbol that is not a registered SDFG symbol is not in scope to assign
    ``_gidx = idx[i]`` on the edge; the promotion must leave the nested subscript (so
    the caller refuses) rather than plant an out-of-scope assignment."""
    sdfg = dace.SDFG("gather_oos")
    sdfg.add_array("w", (8, 8), dace.float64)
    sdfg.add_array("idx", (8, ), dace.int64)
    # 'i' intentionally NOT registered as an SDFG symbol -> out of scope.
    s0 = sdfg.add_state("s0", is_start_block=True)
    s1 = sdfg.add_state("s1")
    sdfg.add_edge(s0, s1, dace.InterstateEdge())
    edge = list(sdfg.out_edges(s0))[0]
    p = SameWriteSetIfElseToITECFG()
    rhs = "w[idx[i], 0] > 0.0"

    out = p._promote_gather_indices(sdfg, [edge], rhs)

    assert out == rhs and p._has_nested_subscript(sdfg, out)
    assert "_gidx_0" not in (edge.data.assignments or {}), "must not plant an out-of-scope assignment"
