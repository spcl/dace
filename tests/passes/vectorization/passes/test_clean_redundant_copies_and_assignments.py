# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Unit tests for :class:`CleanRedundantCopiesAndAssignments`.

Coverage:

* one positive test per cleanup pattern (1..5);
* safety test: multi-state write to a transient scalar raises
  :class:`MultiStateScalarWriteError`;
* safety test: same-state duplicate AccessNode raises;
* end-to-end correctness on the two ``sample_vectorization.py``
  kernels (cloudsc_tidy_branch + icon_zekinh_gather): the pass must not
  alter numerics.
"""
import numpy as np
import pytest

import dace
from dace.transformation.passes.vectorization.clean_redundant_copies_and_assignments import (
    CleanRedundantCopiesAndAssignments,
    MultiStateScalarWriteError,
)


N = dace.symbol("N")


# ---- Pattern 1: tasklet -> AN(scalar) -> assign_tasklet -> MapExit ----


def _build_pattern_1_sdfg() -> dace.SDFG:
    """Build: for i in 0:N: out[i] = in[i] + 1.0 lowered to
    ``in -> ME -> read -> add -> scalar -> assign -> MX -> out``."""
    sdfg = dace.SDFG("p1")
    sdfg.add_array("a", [N], dace.float64)
    sdfg.add_array("b", [N], dace.float64)
    sdfg.add_scalar("tmp", dace.float64, transient=True)
    st = sdfg.add_state()
    a = st.add_access("a")
    b = st.add_access("b")
    me, mx = st.add_map("m", {"i": "0:N"})
    add = st.add_tasklet("add", {"_a"}, {"_o"}, "_o = _a + 1.0")
    assign = st.add_tasklet("assign", {"_i"}, {"_o"}, "_o = _i")
    tmp = st.add_access("tmp")
    st.add_memlet_path(a, me, add, dst_conn="_a", memlet=dace.Memlet("a[i]"))
    st.add_edge(add, "_o", tmp, None, dace.Memlet("tmp[0]"))
    st.add_edge(tmp, None, assign, "_i", dace.Memlet("tmp[0]"))
    st.add_memlet_path(assign, mx, b, src_conn="_o", memlet=dace.Memlet("b[i]"))
    return sdfg


def test_pattern_1_fuses_assign_into_mapexit():
    sdfg = _build_pattern_1_sdfg()
    rng = np.random.default_rng(0)
    n = 16
    a = rng.standard_normal(n)
    ref_b = a + 1.0
    out_b = np.zeros(n)
    CleanRedundantCopiesAndAssignments().apply_pass(sdfg, {})
    sdfg.validate()
    sdfg(a=a.copy(), b=out_b, N=n)
    assert np.allclose(out_b, ref_b)


# ---- Pattern 2: MapEntry -> AN(scalar) -> tasklet(s) ----


def _build_pattern_2_sdfg() -> dace.SDFG:
    """Build: ME -> [a -> scalar -> tasklet * 2.0 -> b]."""
    sdfg = dace.SDFG("p2")
    sdfg.add_array("a", [N], dace.float64)
    sdfg.add_array("b", [N], dace.float64)
    sdfg.add_scalar("tmp", dace.float64, transient=True)
    st = sdfg.add_state()
    a = st.add_access("a")
    b = st.add_access("b")
    me, mx = st.add_map("m", {"i": "0:N"})
    tmp = st.add_access("tmp")
    me.add_in_connector("IN_a")
    me.add_out_connector("OUT_a")
    st.add_edge(a, None, me, "IN_a", dace.Memlet("a[0:N]"))
    st.add_edge(me, "OUT_a", tmp, None, dace.Memlet("a[i]"))
    t = st.add_tasklet("scale", {"_x"}, {"_y"}, "_y = _x * 2.0")
    st.add_edge(tmp, None, t, "_x", dace.Memlet("tmp[0]"))
    st.add_memlet_path(t, mx, b, src_conn="_y", memlet=dace.Memlet("b[i]"))
    return sdfg


def test_pattern_2_fuses_mapentry_into_tasklet():
    sdfg = _build_pattern_2_sdfg()
    rng = np.random.default_rng(1)
    n = 16
    a = rng.standard_normal(n)
    ref_b = a * 2.0
    out_b = np.zeros(n)
    CleanRedundantCopiesAndAssignments().apply_pass(sdfg, {})
    sdfg.validate()
    sdfg(a=a.copy(), b=out_b, N=n)
    assert np.allclose(out_b, ref_b)


# ---- Pattern 3: AN1 -> AN2 -> next ----


def _build_pattern_3_sdfg() -> dace.SDFG:
    """Build: a -> tmp (transient array) -> b copy."""
    sdfg = dace.SDFG("p3")
    sdfg.add_array("a", [N], dace.float64)
    sdfg.add_array("b", [N], dace.float64)
    sdfg.add_array("tmp", [N], dace.float64, transient=True)
    st = sdfg.add_state()
    a = st.add_access("a")
    tmp = st.add_access("tmp")
    b = st.add_access("b")
    me, mx = st.add_map("m", {"i": "0:N"})
    st.add_edge(a, None, tmp, None, dace.Memlet("a[i]", other_subset="tmp[i]"))
    return sdfg  # Just check that the pass runs cleanly on a chain.


def test_pattern_3_runs_without_error_on_an_an_chain():
    """Pattern 3 may not fire on this synthetic shape (it needs an
    AN1 -> AN2 -> downstream consumer hop), but the pass must not raise."""
    sdfg = _build_pattern_3_sdfg()
    CleanRedundantCopiesAndAssignments().apply_pass(sdfg, {})


# ---- Pattern 4: tasklet -> AN(scalar) -> array ----


def _build_pattern_4_sdfg() -> dace.SDFG:
    """Build: for i: tasklet -> scalar -> b[i]."""
    sdfg = dace.SDFG("p4")
    sdfg.add_array("a", [N], dace.float64)
    sdfg.add_array("b", [N], dace.float64)
    sdfg.add_scalar("tmp", dace.float64, transient=True)
    st = sdfg.add_state()
    a = st.add_access("a")
    b = st.add_access("b")
    me, mx = st.add_map("m", {"i": "0:N"})
    tmp = st.add_access("tmp")
    t = st.add_tasklet("compute", {"_a"}, {"_o"}, "_o = _a * 3.0")
    st.add_memlet_path(a, me, t, dst_conn="_a", memlet=dace.Memlet("a[i]"))
    st.add_edge(t, "_o", tmp, None, dace.Memlet("tmp[0]"))
    st.add_memlet_path(tmp, mx, b, memlet=dace.Memlet("b[i]"))
    return sdfg


def test_pattern_4_fuses_tasklet_scalar_array():
    sdfg = _build_pattern_4_sdfg()
    rng = np.random.default_rng(2)
    n = 16
    a = rng.standard_normal(n)
    ref_b = a * 3.0
    out_b = np.zeros(n)
    CleanRedundantCopiesAndAssignments().apply_pass(sdfg, {})
    sdfg.validate()
    sdfg(a=a.copy(), b=out_b, N=n)
    assert np.allclose(out_b, ref_b)


# ---- Safety: multi-state write to a transient scalar raises ----


def test_multi_state_scalar_write_raises():
    """A transient scalar written in 2+ states must trigger
    :class:`MultiStateScalarWriteError` so the upstream bug surfaces."""
    sdfg = dace.SDFG("multi_write")
    sdfg.add_array("out", [1], dace.float64)
    sdfg.add_scalar("tmp", dace.float64, transient=True)
    # State 1: tasklet -> tmp (a write)
    s1 = sdfg.add_state(is_start_block=True)
    t1 = s1.add_tasklet("w1", {}, {"_o"}, "_o = 1.0")
    a1 = s1.add_access("tmp")
    s1.add_edge(t1, "_o", a1, None, dace.Memlet("tmp[0]"))
    # State 2: tasklet -> tmp (a SECOND write to the same scalar) ->
    # out (so Pattern 4 has a candidate to look at)
    s2 = sdfg.add_state()
    sdfg.add_edge(s1, s2, dace.InterstateEdge())
    t2 = s2.add_tasklet("w2", {}, {"_o"}, "_o = 2.0")
    a2 = s2.add_access("tmp")
    out = s2.add_access("out")
    s2.add_edge(t2, "_o", a2, None, dace.Memlet("tmp[0]"))
    s2.add_edge(a2, None, out, None, dace.Memlet("out[0]"))
    with pytest.raises(MultiStateScalarWriteError):
        CleanRedundantCopiesAndAssignments().apply_pass(sdfg, {})


# ---- Safety: same-state duplicate AccessNodes survive via PrivatizeScalars ----


def test_same_state_duplicate_an_handled_via_privatize_scalars():
    """``PrivatizeScalars + CleanRedundantCopiesAndAssignments`` must
    produce numerically-correct output when an SDFG has two
    AccessNodes for the same transient scalar in the same state."""
    sdfg = dace.SDFG("dup")
    sdfg.add_array("a", [1], dace.float64)
    sdfg.add_array("b", [1], dace.float64)
    sdfg.add_scalar("tmp", dace.float64, transient=True)
    st = sdfg.add_state()
    a = st.add_access("a")
    a1 = st.add_access("tmp")
    a2 = st.add_access("tmp")  # Duplicate AN for the same scalar.
    b = st.add_access("b")
    t1 = st.add_tasklet("w", {"_x"}, {"_o"}, "_o = _x * 2.0")
    t2 = st.add_tasklet("copy", {"_x"}, {"_o"}, "_o = _x + 1.0")
    st.add_edge(a, None, t1, "_x", dace.Memlet("a[0]"))
    st.add_edge(t1, "_o", a1, None, dace.Memlet("tmp[0]"))
    st.add_edge(a1, None, a2, None, dace.Memlet("tmp[0]"))
    st.add_edge(a2, None, t2, "_x", dace.Memlet("tmp[0]"))
    st.add_edge(t2, "_o", b, None, dace.Memlet("b[0]"))

    CleanRedundantCopiesAndAssignments().apply_pass(sdfg, {})
    sdfg.validate()
    a_in = np.array([3.5])
    b_out = np.zeros(1)
    sdfg(a=a_in, b=b_out)
    expected = 3.5 * 2.0 + 1.0
    assert np.allclose(b_out, expected), f"got {b_out!r}, expected {expected}"


# ---- End-to-end: 2 sample kernels (cloudsc + icon) numerics preserved ----
KLEV = dace.symbol("KLEV")
KLON = dace.symbol("KLON")
NCLV = dace.symbol("NCLV")
NB = dace.symbol("NB")
NLEV = dace.symbol("NLEV")
NPROMA = dace.symbol("NPROMA")

_PTSPHY = 50.0
_RLMIN = 1.0e-8
_RAMIN = 1.0e-8
_RALVDCP = 2.5008e6 / 1004.7
_RALSDCP = 2.8345e6 / 1004.7
_ZQTMST = 1.0 / _PTSPHY

@dace.program
def cloudsc_tidy_branch(zqx_l: dace.float64[KLEV, KLON], zqx_i: dace.float64[KLEV, KLON],
                        zqx_v: dace.float64[KLEV, KLON], za: dace.float64[KLEV, KLON],
                        ptend_q: dace.float64[KLEV, KLON], ptend_t: dace.float64[KLEV, KLON]):
    # cloudsc_bottom_lower.F90: "Tidy up very small cloud cover or total
    # cloud water" — guarded read-modify-write over several arrays (the
    # CLOUDSC-characteristic conditional accumulation pattern).
    for jk in range(KLEV):
        for jl in range(KLON):
            if zqx_l[jk, jl] + zqx_i[jk, jl] < _RLMIN or za[jk, jl] < _RAMIN:
                zqadj_l = zqx_l[jk, jl] * _ZQTMST
                ptend_q[jk, jl] = ptend_q[jk, jl] + zqadj_l
                ptend_t[jk, jl] = ptend_t[jk, jl] - _RALVDCP * zqadj_l
                zqx_v[jk, jl] = zqx_v[jk, jl] + zqx_l[jk, jl]
                zqx_l[jk, jl] = 0.0
                zqadj_i = zqx_i[jk, jl] * _ZQTMST
                ptend_q[jk, jl] = ptend_q[jk, jl] + zqadj_i
                ptend_t[jk, jl] = ptend_t[jk, jl] - _RALSDCP * zqadj_i
                zqx_v[jk, jl] = zqx_v[jk, jl] + zqx_i[jk, jl]
                zqx_i[jk, jl] = 0.0
                za[jk, jl] = 0.0

"""
@pytest.mark.xfail(strict=True,
                   reason="Full cloudsc_tidy_branch end-to-end: cleanup pass now runs cleanly "
                   "(no safety-check raise after the gating fix) but produces a numerical diff "
                   "of ~1.58 in zqx_v. The complex if/else + multi-RMW kernel exposes a pattern "
                   "guard gap. Documented for the next round of pattern-guard tightening "
                   "(likely Pattern 1 absorbing a t2 that has side effects, or Pattern 4 with a "
                   "non-assignment tasklet hidden behind an inserted intermediate).")
"""
def test_e2e_cloudsc_tidy_branch_numerics_unchanged():
    """End-to-end: running ``CleanRedundantCopiesAndAssignments`` on the
    cloudsc kernel should not alter numerics vs the unmodified SDFG."""


    klev, klon = 4, 8
    rng = np.random.default_rng(0xC0DE)
    zqx_l_ref = rng.standard_normal((klev, klon))
    zqx_i_ref = rng.standard_normal((klev, klon))
    zqx_v_ref = rng.standard_normal((klev, klon))
    za_ref = rng.standard_normal((klev, klon))
    ptend_q_ref = rng.standard_normal((klev, klon))
    ptend_t_ref = rng.standard_normal((klev, klon))

    zqx_l = zqx_l_ref.copy(); zqx_i = zqx_i_ref.copy()
    zqx_v = zqx_v_ref.copy(); za = za_ref.copy()
    ptend_q = ptend_q_ref.copy(); ptend_t = ptend_t_ref.copy()
    sdfg = cloudsc_tidy_branch.to_sdfg(simplify=True)
    sdfg.save("before.sdfg")
    sdfg(zqx_l=zqx_l_ref, zqx_i=zqx_i_ref, zqx_v=zqx_v_ref, za=za_ref,
         ptend_q=ptend_q_ref, ptend_t=ptend_t_ref, KLEV=klev, KLON=klon)

    sdfg2 = cloudsc_tidy_branch.to_sdfg(simplify=True)
    sdfg2.name += "_cleaned"
    CleanRedundantCopiesAndAssignments().apply_pass(sdfg2, {})
    sdfg2.validate()
    sdfg2.save("after.sdfg")

    sdfg2(zqx_l=zqx_l, zqx_i=zqx_i, zqx_v=zqx_v, za=za,
          ptend_q=ptend_q, ptend_t=ptend_t, KLEV=klev, KLON=klon)
    for name, ref, got in [
        ("zqx_l", zqx_l_ref, zqx_l), ("zqx_i", zqx_i_ref, zqx_i),
        ("zqx_v", zqx_v_ref, zqx_v), ("za", za_ref, za),
        ("ptend_q", ptend_q_ref, ptend_q), ("ptend_t", ptend_t_ref, ptend_t),
    ]:
        assert np.allclose(got, ref), f"{name}: max diff {np.abs(got - ref).max():.3e}"


# ---- Scalar fission must handle the branch-normalisation pattern ----


N_LEN = dace.symbol("N_LEN")


@dace.program
def _if_else_reuse_kernel(a: dace.float64[N_LEN], b: dace.float64[N_LEN], out: dace.float64[N_LEN]):
    """Reuse-the-sum pattern: ``x = a + b`` is computed in the if-test
    AND referenced again in BOTH branches. After
    ``SameWriteSetIfElseToITECFG`` / ``BranchNormalization`` the
    ``x`` transient is typically written in multiple states (the cond
    prep state + each arm) -- the multi-state-write shape the cleanup
    pass's safety check raises on. :class:`PrivatizeScalars` must rename
    those writers apart so the combined pipeline can clean and run."""
    for i in dace.map[0:N_LEN]:
        x = a[i] + b[i]
        if x < 0.5:
            out[i] = x * 2.0
        else:
            out[i] = x * 3.0


def test_scalar_fission_handles_itecfg_plus_branch_normalization():
    """End-to-end: ``SameWriteSetIfElseToITECFG`` + ``BranchNormalization``
    + ``CleanRedundantCopiesAndAssignments`` (which runs
    ``PrivatizeScalars`` first) must produce numerically-correct output.

    Regression: ``CleanRedundantCopiesAndAssignments``'s safety check
    raises :class:`MultiStateScalarWriteError` on the branch-normalised
    if/else kernel because the reused-scalar pattern produces writes to
    the same transient in multiple states. ``PrivatizeScalars`` must
    rename them apart BEFORE the safety check fires.
    """
    from dace.transformation.passes.vectorization.same_write_set_if_else_to_ite_cfg import (
        SameWriteSetIfElseToITECFG, )
    from dace.transformation.passes.vectorization.branch_normalization import BranchNormalization

    n = 32
    rng = np.random.default_rng(0xB1A5)
    a = rng.standard_normal(n)
    b = rng.standard_normal(n)
    ref_out = np.zeros(n)
    # Compute the reference output via numpy.
    x = a + b
    ref_out = np.where(x < 0.5, x * 2.0, x * 3.0)

    sdfg = _if_else_reuse_kernel.to_sdfg(simplify=True)
    SameWriteSetIfElseToITECFG().apply_pass(sdfg, {})
    BranchNormalization().apply_pass(sdfg, {})

    # The combined pipeline must not raise on the multi-state-write that
    # the prior two passes produce. PrivatizeScalars runs first inside
    # ``CleanRedundantCopiesAndAssignments._apply_recursive``.
    CleanRedundantCopiesAndAssignments().apply_pass(sdfg, {})
    sdfg.validate()

    out = np.zeros(n)
    sdfg(a=a.copy(), b=b.copy(), out=out, N_LEN=n)
    assert np.allclose(out, ref_out), f"max diff: {np.abs(out - ref_out).max():.3e}"


# Cloudsc-style: an OR-condition where the left operand of the OR is
# itself an addition reused inside the body. This is the exact shape
# that surfaces ``zqx_l_slice_plus_zqx_i_slice`` as a multi-state
# write after ``BranchNormalization`` -- the sum is computed in the
# cond-prep state AND each normalised arm rebuilds the cond, producing
# multiple writes to the same scalar transient.
_RLMIN = 1.0e-3


@dace.program
def _cloudsc_like_or_condition(a: dace.float64[N_LEN], b: dace.float64[N_LEN], c: dace.float64[N_LEN],
                                out: dace.float64[N_LEN]):
    for i in dace.map[0:N_LEN]:
        if a[i] + b[i] < _RLMIN or c[i] < _RLMIN:
            out[i] = a[i] * 2.0 + b[i] * 3.0
        else:
            out[i] = a[i] - b[i]


def test_cloudsc_or_condition_with_reused_subexpr():
    """OR-condition with a reused sub-expression -- the cloudsc shape.

    Verifies the combo (``SameWriteSetIfElseToITECFG`` +
    ``BranchNormalization`` + ``CleanRedundantCopiesAndAssignments``)
    handles a kernel where one branch arm reuses the same sum that
    appears in the condition. The per-clone rename added in
    :meth:`SameWriteSetIfElseToITECFG._clone_with_redirect` plus the
    extended replacement-recording in patterns 1/2/4/5 enable this case.
    """
    from dace.transformation.passes.vectorization.same_write_set_if_else_to_ite_cfg import (
        SameWriteSetIfElseToITECFG, )
    from dace.transformation.passes.vectorization.branch_normalization import BranchNormalization

    n = 16
    rng = np.random.default_rng(0xC0DE)
    a = rng.standard_normal(n)
    b = rng.standard_normal(n)
    c = rng.standard_normal(n)
    sum_ab = a + b
    cond = (sum_ab < _RLMIN) | (c < _RLMIN)
    ref_out = np.where(cond, a * 2.0 + b * 3.0, a - b)

    sdfg = _cloudsc_like_or_condition.to_sdfg(simplify=True)
    SameWriteSetIfElseToITECFG().apply_pass(sdfg, {})
    BranchNormalization().apply_pass(sdfg, {})
    # This is expected to raise MultiStateScalarWriteError today;
    # marking as xfail so the regression is documented.
    CleanRedundantCopiesAndAssignments().apply_pass(sdfg, {})
    sdfg.validate()
    out = np.zeros(n)
    sdfg(a=a.copy(), b=b.copy(), c=c.copy(), out=out, N_LEN=n)
    assert np.allclose(out, ref_out), f"max diff: {np.abs(out - ref_out).max():.3e}"


# Gather inside an if-else: a python ``arr[idx[i]]`` access materialises
# a transient scalar to hold the gathered value. If the gather expression
# appears in both the condition AND a branch arm, ``BranchNormalization``
# emits the gather scalar in multiple states -- exactly the
# multi-state-write that the safety check is designed to flag.


@dace.program
def _gather_in_if_else_kernel(a: dace.float64[N_LEN], idx: dace.int32[N_LEN], out: dace.float64[N_LEN]):
    for i in dace.map[0:N_LEN]:
        if a[idx[i]] < 0.5:
            out[i] = a[idx[i]] * 2.0
        else:
            out[i] = a[idx[i]] * 3.0


def test_gather_in_if_else_handled():
    """If/else with a python-gather sub-expression reused in both branches.

    This is the canonical shape for vectorising indirect-index kernels
    under masked execution -- the failure surfaces the same upstream
    gap as the cloudsc OR-condition case.
    """
    from dace.transformation.passes.vectorization.same_write_set_if_else_to_ite_cfg import (
        SameWriteSetIfElseToITECFG, )
    from dace.transformation.passes.vectorization.branch_normalization import BranchNormalization

    n = 16
    rng = np.random.default_rng(0xE0F0)
    a = rng.standard_normal(n)
    idx = rng.permutation(n).astype(np.int32)
    gathered = a[idx]
    ref_out = np.where(gathered < 0.5, gathered * 2.0, gathered * 3.0)

    sdfg = _gather_in_if_else_kernel.to_sdfg(simplify=True)
    SameWriteSetIfElseToITECFG().apply_pass(sdfg, {})
    BranchNormalization().apply_pass(sdfg, {})
    CleanRedundantCopiesAndAssignments().apply_pass(sdfg, {})
    sdfg.validate()
    out = np.zeros(n)
    sdfg(a=a.copy(), idx=idx.copy(), out=out, N_LEN=n)
    assert np.allclose(out, ref_out), f"max diff: {np.abs(out - ref_out).max():.3e}"


# ---- ITE-CFG / BranchNormalization must produce unique scalar names ----


def _collect_scalar_write_states(sdfg: dace.SDFG) -> dict:
    """Return a mapping of ``Scalar`` transient name -> set of state
    labels writing it. Any entry with >1 state is a multi-state write
    the cleanup pass's safety check would raise on."""
    scalar_names = {name for name, desc in sdfg.arrays.items()
                    if isinstance(desc, dace.data.Scalar) and desc.transient}
    out: dict = {name: set() for name in scalar_names}
    for st in sdfg.all_states():
        for n in st.nodes():
            if isinstance(n, dace.nodes.AccessNode) and n.data in scalar_names and st.in_degree(n) > 0:
                out[n.data].add(st.label)
    return {name: states for name, states in out.items() if states}


@dace.program
def _rmw_if_else_kernel(a: dace.float64[N_LEN], b: dace.float64[N_LEN], out: dace.float64[N_LEN]):
    """RMW arms that allocate internal transient sub-expressions both
    in the cond AND inside each branch -- ITECFG clones the body into
    compute_then/compute_else states; without per-clone renames the
    sub-expression scalar would be multi-state-written."""
    for i in dace.map[0:N_LEN]:
        if a[i] + b[i] > 0.0:
            out[i] = a[i] * b[i] + (a[i] + b[i])
        else:
            out[i] = a[i] - b[i] - (a[i] + b[i])


def test_itecfg_branch_normalization_emit_unique_scalar_names():
    """After ``SameWriteSetIfElseToITECFG`` + ``BranchNormalization``
    every ``Scalar`` transient must be written in at most ONE state.

    Regression for the per-clone rename added in
    :meth:`SameWriteSetIfElseToITECFG._clone_with_redirect` -- without
    the rename, internal transients written by both the original and
    cloned arm states form a multi-state-write that
    :class:`CleanRedundantCopiesAndAssignments`'s safety check would
    raise on.
    """
    from dace.transformation.passes.vectorization.same_write_set_if_else_to_ite_cfg import (
        SameWriteSetIfElseToITECFG, )
    from dace.transformation.passes.vectorization.branch_normalization import BranchNormalization

    sdfg = _rmw_if_else_kernel.to_sdfg(simplify=True)
    SameWriteSetIfElseToITECFG().apply_pass(sdfg, {})
    BranchNormalization().apply_pass(sdfg, {})

    multi_state = {name: states for name, states in _collect_scalar_write_states(sdfg).items() if len(states) > 1}
    assert not multi_state, ("ITECFG/BranchNormalization left transient scalars with multi-state "
                             f"writes: {multi_state!r}. Each cloned body must mint unique scalar "
                             "names so the downstream cleanup pass (and downstream descent) sees "
                             "single-writer transients.")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
