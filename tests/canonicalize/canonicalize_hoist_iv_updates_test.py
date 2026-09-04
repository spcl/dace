# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for :class:`HoistInductionVariableUpdates`.

The pass fissions IV-eligible updates out of compound loop bodies so the
downstream :class:`InductionVariableSubstitution` matcher (which requires a
single-tasklet body) catches them and collapses to ``O(1)``.

Fission needs the IV component to be INDEPENDENT of the rest of the body. When the
body READS the accumulator the two are coupled and this pass must refuse -- that
shape is handled instead by ``InductionVariableSubstitution``'s use-site expansion,
which is covered here alongside the refusal so the hand-off stays honest.
"""
import numpy as np
import pytest

import dace
from dace.sdfg import nodes
from dace.sdfg.state import LoopRegion
from dace.transformation.dataflow.trivial_tasklet_elimination import TrivialTaskletElimination
from dace.transformation.passes.canonicalize.hoist_iv_updates import HoistInductionVariableUpdates
from dace.transformation.passes.canonicalize.induction_variable_substitution import InductionVariableSubstitution
from dace.transformation.passes.pattern_matching import PatternMatchAndApplyRepeated

N = dace.symbol('N')


def _setup(program):
    """Build the SDFG and apply the pre-pass that ``canonicalize`` runs before
    ``HoistInductionVariableUpdates`` -- ``TrivialTaskletElimination`` collapses
    the frontend's ``compute -> tmp -> copy -> accum`` staging so that what
    reaches the IV passes is the bare compute/per-element body."""
    sdfg = program.to_sdfg(simplify=True)
    PatternMatchAndApplyRepeated([TrivialTaskletElimination()]).apply_pass(sdfg, {})
    return sdfg


def _nloops(sdfg):
    return sum(1 for r in sdfg.all_control_flow_regions() if isinstance(r, LoopRegion) and r.loop_variable)


def _ntasklets_in_loop_bodies(sdfg):
    """Tasklet count *inside* loop bodies, across the whole SDFG."""
    total = 0
    for r in sdfg.all_control_flow_regions():
        if isinstance(r, LoopRegion) and r.loop_variable:
            for blk in r.nodes():
                if hasattr(blk, 'nodes'):
                    total += sum(1 for n in blk.nodes() if isinstance(n, nodes.Tasklet))
    return total


@dace.program
def compound_iv_and_perelem(a: dace.float64[1], b: dace.float64[N]):
    """Loop with an IV update on a loop-invariant slot (``a[0] *= 0.99``) and an
    independent per-iteration update on ``b[i]``."""
    for i in range(N):
        a[0] = a[0] * 0.99
        b[i] = b[i] + 1.0


def test_hoist_iv_updates_splits_compound_body():
    sdfg = _setup(compound_iv_and_perelem)
    n_loops_before = _nloops(sdfg)
    n_tasklets_before = _ntasklets_in_loop_bodies(sdfg)
    res = HoistInductionVariableUpdates().apply_pass(sdfg, {})
    sdfg.validate()
    # Either nothing matched (and the result is None) or at least one loop was
    # fissioned -- in the latter case there's now one more loop and one fewer
    # tasklet per body (the IV statement moved out).
    n_loops_after = _nloops(sdfg)
    n_tasklets_after = _ntasklets_in_loop_bodies(sdfg)
    assert res == n_loops_after - n_loops_before, (f"reported {res} hoists, but "
                                                   f"loop count went {n_loops_before} -> {n_loops_after}")
    if res:
        assert n_tasklets_after == n_tasklets_before, ("split should not duplicate or drop tasklets; "
                                                       f"{n_tasklets_before} -> {n_tasklets_after}")


def test_hoist_iv_updates_value_preserving():
    n = 16
    rng = np.random.default_rng(0)
    a0, b0 = rng.standard_normal(1), rng.standard_normal(n)
    sdfg = _setup(compound_iv_and_perelem)
    HoistInductionVariableUpdates().apply_pass(sdfg, {})
    sdfg.validate()
    a, b = a0.copy(), b0.copy()
    sdfg(a=a, b=b, N=n)
    # Reference: same semantics as the unrolled loop.
    a_ref, b_ref = a0.copy(), b0.copy()
    for i in range(n):
        a_ref[0] = a_ref[0] * 0.99
        b_ref[i] = b_ref[i] + 1.0
    assert np.allclose(a, a_ref) and np.allclose(b, b_ref)


def test_hoist_then_ivsub_collapses_iv_loop():
    """End-to-end: hoist + IV substitution turns the IV update into a closed
    form, leaving the per-iteration loop with the remaining body."""
    sdfg = _setup(compound_iv_and_perelem)
    HoistInductionVariableUpdates().apply_pass(sdfg, {})
    n_subs = InductionVariableSubstitution().apply_pass(sdfg, {})
    sdfg.validate()
    assert n_subs is not None and n_subs >= 1, ("expected at least one IV-substituted loop after hoist; "
                                                "the hoisted single-statement loop should have collapsed")


@dace.program
def coupled_iv_and_perelem(a: dace.float64[1], b: dace.float64[N]):
    """Refusal case: ``a[0]`` is both read by the per-element update AND IV-updated,
    so the IV statement is NOT independent of the rest -- the pass must leave it alone."""
    for i in range(N):
        b[i] = b[i] + a[0]  # reads a[0]
        a[0] = a[0] * 0.5  # IV update on the same slot the loop body reads


def test_hoist_refuses_when_iv_slot_is_loop_dependency():
    sdfg = _setup(coupled_iv_and_perelem)
    res = HoistInductionVariableUpdates().apply_pass(sdfg, {})
    sdfg.validate()
    assert res is None, ("hoist must refuse when the IV-eligible slot is also read elsewhere in the body; "
                         f"got result {res}")


def _iv_slot_written_in_loop(sdfg, name: str) -> bool:
    """Whether any loop body still WRITES ``name`` -- i.e. the loop-carried recurrence survived."""
    for r in sdfg.all_control_flow_regions():
        if not (isinstance(r, LoopRegion) and r.loop_variable):
            continue
        for blk in r.nodes():
            if not isinstance(blk, dace.SDFGState):
                continue
            for n in blk.nodes():
                if isinstance(n, nodes.AccessNode) and n.data == name and blk.in_degree(n) > 0:
                    return True
    return False


@dace.program
def use_after_update(a: dace.float64[N], b: dace.float64[N]):
    """TSVC ``s453``'s shape: the accumulator is updated, then READ by the per-element statement,
    so the read must see this iteration's update (``2.0 * (i + 1)``, not ``2.0 * i``)."""
    s = 0.0
    for i in range(N):
        s = s + 2.0
        a[i] = s * b[i]


@dace.program
def use_before_update(a: dace.float64[N], b: dace.float64[N]):
    """The mirror image: the per-element statement reads the accumulator BEFORE the update, so it
    must see the PREVIOUS iterations' value only (``0.5 ** i``, not ``0.5 ** (i + 1)``)."""
    s = 1.0
    for i in range(N):
        a[i] = s * b[i]
        s = s * 0.5


def test_use_site_substitution_post_update_read():
    """The shape ``HoistInductionVariableUpdates`` cannot fission (the accumulator is READ by the
    statement beside it) and ``_try_substitute`` cannot collapse (the body has two statements):
    the closed form is expanded at the read instead."""
    n = 16
    b0 = np.random.default_rng(1).standard_normal(n)

    sdfg = _setup(use_after_update)
    assert HoistInductionVariableUpdates().apply_pass(sdfg, {}) is None, \
        "the IV component is not independent here, so fission must refuse"
    assert InductionVariableSubstitution().apply_pass(sdfg, {}) is not None, \
        "s453's read-after-update data IV must be expanded at its use site"
    sdfg.validate()
    assert not _iv_slot_written_in_loop(sdfg, 's'), "the loop still writes s -> recurrence survived"

    a = np.zeros(n)
    sdfg(a=a, b=b0.copy(), N=n)
    a_ref, s_ref = np.zeros(n), 0.0
    for i in range(n):
        s_ref = s_ref + 2.0
        a_ref[i] = s_ref * b0[i]
    # 2.0*(i+1) is exact in binary at these lengths, so an off-by-one step cannot hide in rounding.
    assert np.array_equal(a, a_ref), f"post-update read got the wrong step:\nref={a_ref}\ngot={a}"


def test_use_site_substitution_pre_update_read():
    """Same body, opposite order: the offset is ``t``, not ``t + 1``. Reading the source order
    instead of the dataflow would give every element one extra halving."""
    n = 16
    b0 = np.random.default_rng(2).standard_normal(n)

    sdfg = _setup(use_before_update)
    assert InductionVariableSubstitution().apply_pass(sdfg, {}) is not None, \
        "a read BEFORE the update has a closed form too (one step behind) -- it must lift"
    sdfg.validate()
    assert not _iv_slot_written_in_loop(sdfg, 's'), "the loop still writes s -> recurrence survived"

    a = np.zeros(n)
    sdfg(a=a, b=b0.copy(), N=n)
    a_ref, s_ref = np.zeros(n), 1.0
    for i in range(n):
        a_ref[i] = s_ref * b0[i]
        s_ref = s_ref * 0.5
    # Halving is exact in binary, so demand bit-equality: an off-by-one step doubles every element.
    assert np.array_equal(a, a_ref), f"pre-update read got the wrong step:\nref={a_ref}\ngot={a}"


def test_an_ordering_edge_does_not_make_a_statement_part_of_the_iv_component():
    """An empty memlet decides ORDER, never membership: the copy statement is not the IV's."""
    import dace as _dace
    from dace.sdfg.state import LoopRegion

    n = 8
    sdfg = _dace.SDFG("hoist_iv_ordering")
    sdfg.add_array("a", [n], _dace.float64)
    sdfg.add_array("b", [n], _dace.float64)
    sdfg.add_array("scale", [1], _dace.float64)
    sdfg.add_symbol("i", _dace.int64)

    loop = LoopRegion("loop", condition_expr=f"i < {n}", loop_var="i", initialize_expr="i = 0", update_expr="i = i + 1")
    st = loop.add_state("body", is_start_block=True)
    scale_r, scale_w = st.add_access("scale"), st.add_access("scale")
    iv = st.add_tasklet("iv", {"__in"}, {"__out"}, "__out = __in * 0.99")
    st.add_edge(scale_r, None, iv, "__in", _dace.Memlet("scale[0]"))
    st.add_edge(iv, "__out", scale_w, None, _dace.Memlet("scale[0]"))
    cp = st.add_tasklet("cp", {"__inp"}, {"__out"}, "__out = __inp")
    st.add_edge(st.add_access("b"), None, cp, "__inp", _dace.Memlet("b[i]"))
    st.add_edge(cp, "__out", st.add_access("a"), None, _dace.Memlet("a[i]"))
    st.add_nedge(scale_w, cp, _dace.Memlet())
    sdfg.add_node(loop, is_start_block=True)
    sdfg.validate()

    assert HoistInductionVariableUpdates().apply_pass(sdfg, {}) is None
    assert [b.label for b in sdfg.nodes()] == ["loop"]
    assert sorted(t.label for t in st.nodes() if isinstance(t, _dace.nodes.Tasklet)) == ["cp", "iv"]


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
