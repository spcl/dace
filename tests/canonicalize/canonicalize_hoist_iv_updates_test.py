# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for :class:`HoistInductionVariableUpdates`.

The pass fissions IV-eligible updates out of compound loop bodies so the
downstream :class:`InductionVariableSubstitution` matcher (which requires a
single-tasklet body) catches them and collapses to ``O(1)``.
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


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
