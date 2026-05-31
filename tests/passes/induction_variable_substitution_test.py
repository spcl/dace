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


# ---------------------------------------------------------------------------
# Iedge-IV substitution: ``k := k + literal_const`` on an interstate edge
# inside a multi-statement body. Substitute reads of ``k`` with the closed
# form; surviving loop becomes parallelisable.
# ---------------------------------------------------------------------------


@dace.program
def _iedge_iv_counter_in_inner_loop(flat: dace.float64[N * N],
                                     a: dace.float64[N, N], b: dace.float64[N, N], c: dace.float64[N, N]):
    """TSVC ``s125`` minimal repro: scalar counter ``k`` incremented each
    inner iter, indexing a flat output array. After IV substitution the
    inner-body reference to ``k`` becomes ``k + (loop_var - start + 1)``
    (a per-iter affine expression) so the inner loop is parallelisable.
    """
    k = -1
    for i in range(N):
        for j in range(N):
            k = k + 1
            flat[k] = a[i, j] + b[i, j] * c[i, j]


def test_iedge_iv_counter_in_inner_loop_substitution():
    """``k := k + 1`` on an inner-loop iedge gets substituted; the inner-
    body reference ``flat[k]`` becomes ``flat[k + (i+1)]`` (loop-var
    affine). Numerics match the row-major flatten of ``a + b*c``."""
    sdfg = _prep_and_run(_iedge_iv_counter_in_inner_loop)
    sdfg.validate()
    # The inner loop survives but its body is no longer carrying ``k``;
    # downstream LoopToMap (run separately) would lift it.
    n = 6
    rng = np.random.default_rng(125)
    a = rng.standard_normal((n, n))
    b = rng.standard_normal((n, n))
    c = rng.standard_normal((n, n))
    flat = np.zeros(n * n)
    sdfg(flat=flat, a=a, b=b, c=c, N=n)
    expected = (a + b * c).ravel(order='C')
    assert np.allclose(flat, expected), f"got {flat}, expected {expected}"


@dace.program
def _iedge_iv_counter_step_two(flat: dace.float64[2 * N], a: dace.float64[N]):
    """Iedge IV with step != 1: ``k := k + 2``. Closed-form must use the
    actual step value when substituting reads."""
    k = 0
    for i in range(N):
        k = k + 2
        flat[k - 2] = a[i]
        flat[k - 1] = a[i]


def test_iedge_iv_counter_step_two_substitution():
    """``k := k + 2`` -- closed form must scale by 2 in the substituted
    references."""
    sdfg = _prep_and_run(_iedge_iv_counter_step_two)
    sdfg.validate()
    n = 5
    a = np.arange(n, dtype=np.float64) + 1.0
    flat = np.zeros(2 * n)
    sdfg(flat=flat, a=a, N=n)
    expected = np.repeat(a, 2)
    assert np.allclose(flat, expected), f"got {flat}, expected {expected}"


@dace.program
def _iedge_iv_post_loop_value_consumed(out: dace.float64[1], a: dace.float64[N]):
    """``k`` is updated in the loop AND read after the loop via an
    interstate-edge symbol use. The substitution must materialise the
    post-loop value of ``k`` so the read after the loop sees the right
    value."""
    k = 10
    for i in range(N):
        k = k + 3
        a[i] = a[i] + 1.0
    out[0] = k


def test_iedge_iv_post_loop_value_materialised():
    """After substitution, the post-loop read of ``k`` must equal the
    closed-form final value (init + N*step)."""
    sdfg = _prep_and_run(_iedge_iv_post_loop_value_consumed)
    sdfg.validate()
    n = 7
    a = np.zeros(n)
    out = np.array([0.0])
    sdfg(out=out, a=a, N=n)
    # k starts at 10, +3 each iter, N iters -> final k = 10 + 3*N
    assert np.isclose(out[0], 10 + 3 * n), f"post-loop k mismatch: got {out[0]}"
    # And the array writes happened correctly.
    assert np.allclose(a, 1.0)


@dace.program
def _iedge_iv_two_assignments_on_same_edge(flat: dace.float64[N], a: dace.float64[N]):
    """An iedge with TWO assignments (only one is the IV) -- v1 refuses
    because the pass requires the IV iedge to carry ONLY the IV
    assignment. The loop stays sequential, numerics still correct."""
    k = 0
    j = 0
    for i in range(N):
        k = k + 1
        j = j + 2
        flat[i] = a[i] + j


def test_iedge_iv_refuses_when_iedge_has_other_assignments():
    """When the IV iedge carries multiple assignments, v1 refuses and the
    loop stays carried. Numerics still match (the refusal is conservative,
    not incorrect)."""
    sdfg = _prep_and_run(_iedge_iv_two_assignments_on_same_edge)
    sdfg.validate()
    n = 5
    a = np.zeros(n)
    flat = np.zeros(n)
    sdfg(flat=flat, a=a, N=n)
    # j starts at 0, becomes 2, 4, 6, ... -- flat[i] = a[i] + (2*(i+1)) = 2*(i+1)
    expected = 2.0 * (np.arange(n) + 1)
    assert np.allclose(flat, expected), f"got {flat}, expected {expected}"


if __name__ == "__main__":
    test_arithmetic_iv_scalar_slot_collapses_to_closed_form()
    test_geometric_iv_scalar_slot_collapses_to_closed_form()
    test_geometric_iv_array_slot_collapses_to_closed_form()
    test_array_indexed_input_is_not_lifted()
    test_loop_indexed_write_is_not_lifted()
    test_non_const_operand_is_not_lifted()
    test_arithmetic_iv_symbolic_operand_collapses_to_closed_form()
    test_arithmetic_iv_symbolic_stride_collapses_to_closed_form()
