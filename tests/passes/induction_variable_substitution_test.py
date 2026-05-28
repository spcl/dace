# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for ``InductionVariableSubstitution``."""
import numpy as np

import dace
from dace.sdfg.state import LoopRegion
from dace.transformation.dataflow.trivial_tasklet_elimination import TrivialTaskletElimination
from dace.transformation.passes.canonicalize.induction_variable_substitution import InductionVariableSubstitution
from dace.transformation.passes.pattern_matching import PatternMatchAndApplyRepeated

N = dace.symbol("N")


def _prep_and_run(prog) -> dace.SDFG:
    """Build simplified SDFG, run ``TrivialTaskletElim`` (matches canon order), then IV substitution."""
    sdfg = prog.to_sdfg(simplify=True)
    PatternMatchAndApplyRepeated([TrivialTaskletElimination()]).apply_pass(sdfg, {})
    InductionVariableSubstitution().apply_pass(sdfg, {})
    sdfg.validate()
    return sdfg


def _n_loops(sdfg: dace.SDFG) -> int:
    return sum(1 for c in sdfg.all_control_flow_regions() if isinstance(c, LoopRegion))


# ---------------------------------------------------------------------------
# Arithmetic IV: ``s[0] += c`` -> ``s[0] += c*N``
# ---------------------------------------------------------------------------


@dace.program
def _sum_iv_scalar_slot(s: dace.float64[1]):
    for i in range(N):
        s[0] = s[0] + 2.5


def test_arithmetic_iv_scalar_slot_collapses_to_closed_form():
    """``for i in range(N): s[0] += 2.5`` -> ``s[0] = s[0] + 2.5*N``.

    Loop is eliminated; the result matches the closed form numerically.
    """
    sdfg = _prep_and_run(_sum_iv_scalar_slot)
    assert _n_loops(sdfg) == 0

    s = np.array([1.0])
    sdfg(s=s, N=10)
    assert np.isclose(s[0], 1.0 + 2.5 * 10)


# ---------------------------------------------------------------------------
# Geometric IV: ``q[0] *= c`` -> ``q[0] *= c**N`` (TSVC s317)
# ---------------------------------------------------------------------------


@dace.program
def _mult_iv_scalar_slot(q: dace.float64[1]):
    for i in range(N):
        q[0] = q[0] * 0.99


def test_geometric_iv_scalar_slot_collapses_to_closed_form():
    """TSVC s317 shape: ``for i in range(N): q[0] *= 0.99`` -> ``q[0] *= 0.99**N``."""
    sdfg = _prep_and_run(_mult_iv_scalar_slot)
    assert _n_loops(sdfg) == 0

    q = np.array([1.0])
    sdfg(q=q, N=50)
    expected = 1.0 * 0.99**50
    assert np.isclose(q[0], expected)


# ---------------------------------------------------------------------------
# Array-slot accumulator on a multi-element array: the IV touches only [0]
# ---------------------------------------------------------------------------


@dace.program
def _mult_iv_array_slot(q: dace.float64[N]):
    for i in range(N // 2):
        q[0] = q[0] * 0.99


def test_geometric_iv_array_slot_collapses_to_closed_form():
    """Accumulator descriptor is ``float64[N]`` but only ``[0]`` is touched.

    Mirrors the chase-forward path that ``LoopToReduce`` also uses: the IV pass
    must accept an array-slot accumulator (not just a scalar-like descriptor).
    """
    sdfg = _prep_and_run(_mult_iv_array_slot)
    assert _n_loops(sdfg) == 0

    n = 100
    q = np.zeros(n)
    q[0] = 1.0
    sdfg(q=q, N=n)
    expected = 1.0 * 0.99**(n // 2)
    assert np.isclose(q[0], expected)
    # The rest of ``q`` should be untouched (still zeros).
    assert np.allclose(q[1:], 0.0)


# ---------------------------------------------------------------------------
# Negative cases: the pass MUST refuse these (they are NOT IV recurrences).
# ---------------------------------------------------------------------------


@dace.program
def _not_iv_input_is_array(s: dace.float64[1], a: dace.float64[N]):
    for i in range(N):
        s[0] = s[0] + a[i]


def test_array_indexed_input_is_not_lifted():
    """``s += a[i]`` is a fold over ``a``, not an IV. Must be left for LoopToReduce."""
    sdfg = _prep_and_run(_not_iv_input_is_array)
    assert _n_loops(sdfg) == 1


@dace.program
def _not_iv_write_index_uses_loop_var(a: dace.float64[N]):
    for i in range(N):
        a[i] = a[i] * 0.5


def test_loop_indexed_write_is_not_lifted():
    """``a[i] *= 0.5`` is a per-element map, not an accumulator IV. Must be left for LoopToMap."""
    sdfg = _prep_and_run(_not_iv_write_index_uses_loop_var)
    assert _n_loops(sdfg) == 1


@dace.program
def _not_iv_non_const_operand(s: dace.float64[1], k: dace.float64[N]):
    for i in range(N):
        s[0] = s[0] * k[0]


def test_non_const_operand_is_not_lifted():
    """``s *= k[0]`` may look like an IV but the operand is a (loop-invariant) DATA read,
    not a numeric constant. Today's pass refuses; an extension could accept loop-invariant
    array-slot operands (mark as TODO if useful in the corpus)."""
    sdfg = _prep_and_run(_not_iv_non_const_operand)
    assert _n_loops(sdfg) == 1


# ---------------------------------------------------------------------------
# Symbolic loop-invariant operand: ``s[0] += step`` with ``step`` a SDFG symbol.
# Closed form ``s[0] = s[0] + step * N`` is computable; the pass should lift it.
# ---------------------------------------------------------------------------

step = dace.symbol("step")


@dace.program
def _sum_iv_scalar_slot_symbolic(s: dace.float64[1]):
    for i in range(N):
        s[0] = s[0] + step


def test_arithmetic_iv_symbolic_operand_collapses_to_closed_form():
    """``s[0] += step`` (``step`` a SDFG int symbol; frontend wraps it in
    ``dace.float64(step)`` for type-consistency, which the matcher unwraps).
    Closed form: ``s[0] = s[0] + step * N``."""
    sdfg = _prep_and_run(_sum_iv_scalar_slot_symbolic)
    assert _n_loops(sdfg) == 0
    s = np.array([3.0])
    sdfg(s=s, N=8, step=2)
    assert np.isclose(s[0], 3.0 + 2 * 8)


# ---------------------------------------------------------------------------
# Symbolic stride: ``for i in range(0, N, stride): s[0] += c`` over a symbolic
# step. Trip count uses ``stride`` -- ``s[0] = s[0] + c * trip``. The closed
# form is identical to the unit-stride case; only the trip count differs.
# ---------------------------------------------------------------------------

stride_sym = dace.symbol("stride_sym", positive=True)


@dace.program
def _sum_iv_scalar_slot_symbolic_stride(s: dace.float64[1]):
    for i in range(0, N, stride_sym):
        s[0] = s[0] + 1.5


def test_arithmetic_iv_symbolic_stride_collapses_to_closed_form():
    """``for i in range(0, N, stride): s[0] += 1.5`` -> ``s[0] = s[0] + 1.5 * trip``."""
    sdfg = _prep_and_run(_sum_iv_scalar_slot_symbolic_stride)
    assert _n_loops(sdfg) == 0
    s = np.array([0.0])
    N_v, stride_v = 20, 4
    sdfg(s=s, N=N_v, stride_sym=stride_v)
    trip = (N_v + stride_v - 1) // stride_v  # ceil division for range(0, N, stride)
    assert np.isclose(s[0], 1.5 * trip)


if __name__ == "__main__":
    test_arithmetic_iv_scalar_slot_collapses_to_closed_form()
    test_geometric_iv_scalar_slot_collapses_to_closed_form()
    test_geometric_iv_array_slot_collapses_to_closed_form()
    test_array_indexed_input_is_not_lifted()
    test_loop_indexed_write_is_not_lifted()
    test_non_const_operand_is_not_lifted()
    test_arithmetic_iv_symbolic_operand_collapses_to_closed_form()
    test_arithmetic_iv_symbolic_stride_collapses_to_closed_form()
