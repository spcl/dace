# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for :class:`~dace.transformation.passes.canonicalize.loop_to_conditional_reduce.LoopToConditionalReduce`.

Covers TSVC ``s3111`` (the conditional ``+=`` accumulator) and the refusal
contracts (non-accumulator conditional bodies, multi-write true-branches,
unsupported ops, etc.).
"""

import copy

import numpy as np
import pytest

import dace
from dace.sdfg import nodes as nd
from dace.sdfg.state import ConditionalBlock, LoopRegion
from dace.transformation.interstate.loop_to_map import LoopToMap
from dace.transformation.passes.canonicalize.loop_to_conditional_reduce import LoopToConditionalReduce
from tests.sdfg.cfg_list_in_place_test import assert_tree_consistent


def assert_cfg_list_matches_reset(sdfg: dace.SDFG) -> None:
    """The kept CFG list and ids equal a fresh copy's after ``reset_cfg_list``, and every parent pointer holds."""
    fresh = copy.deepcopy(sdfg)
    fresh.reset_cfg_list()
    kept = [(type(r).__name__, r.label, r.cfg_id) for r in sdfg.cfg_list]
    assert kept == [(type(r).__name__, r.label, r.cfg_id) for r in fresh.cfg_list]
    assert_tree_consistent(sdfg)


N = dace.symbol("N")


def _num_loops(sdfg):
    return sum(1 for r in sdfg.all_control_flow_regions() if isinstance(r, LoopRegion) and r.loop_variable)


# Positive: TSVC s3111-style conditional accumulators.


def test_tsvc_s3111_conditional_sum():
    """``if a[i] > 0: sum += a[i]`` -- the canonical conditional reduction.
    After the rewrite the ``ConditionalBlock`` is gone and the body is an
    UNCONDITIONAL masked reduction: a mask tasklet (``__out = __addend if cond
    else 0.0``) writing a per-iteration transient, then a plain accumulate
    (``__out = __acc + __masked``). The loop keeps its scalar RMW carry so it
    stays sequential here; the full ``reduction_to_wcr_map`` stage of the
    canonicalize pipeline is what lifts it to a parallel WCR-map + OMP
    ``reduction(+:s)`` clause (covered end-to-end in
    ``tests/passes/canonicalize/conditional_reduction_tree_reduce_test.py``).
    This test pins the rewrite's structural contract + value preservation.
    """

    @dace.program
    def s3111(a: dace.float64[N], b: dace.float64[1]):
        sum_val = 0.0
        for i in range(N):
            if a[i] > 0.0:
                sum_val = sum_val + a[i]
        b[0] = sum_val

    sdfg = s3111.to_sdfg(simplify=True)
    assert _num_loops(sdfg) == 1
    res = LoopToConditionalReduce().apply_pass(sdfg, {})
    assert res == 1
    sdfg.validate()
    assert_cfg_list_matches_reset(sdfg)

    # The guard is folded into the accumulated value: no ConditionalBlock left.
    assert not any(
        isinstance(r, ConditionalBlock) for sd in sdfg.all_sdfgs_recursive() for r in sd.all_control_flow_regions()
    )
    # The unconditional masked-reduction tasklets replace the guarded update.
    bodies = [n.code.as_string for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nd.Tasklet)]
    assert any("if " in b and "else" in b for b in bodies), "expected a masked (ternary) tasklet"

    n = 32
    rng = np.random.default_rng(3111)
    a = rng.standard_normal(n)
    expected = float(np.sum(a[a > 0.0]))
    b = np.zeros(1)
    sdfg(a=a, b=b, N=n)
    assert np.isclose(b[0], expected), f"got {b[0]}, expected {expected}"


def test_no_positives_returns_zero():
    """Edge case: all ``a[i] <= 0`` -- the cond is false at every iteration,
    so every WCR contribution is the identity ``0.0``. The accumulator stays
    at its pre-loop initial value."""

    @dace.program
    def kernel_no_positives_returns_zero(a: dace.float64[N], b: dace.float64[1]):
        sum_val = 0.0
        for i in range(N):
            if a[i] > 0.0:
                sum_val = sum_val + a[i]
        b[0] = sum_val

    sdfg = kernel_no_positives_returns_zero.to_sdfg(simplify=True)
    assert LoopToConditionalReduce().apply_pass(sdfg, {}) == 1, "the rewrite refused; the original loop also sums"
    assert not any(
        isinstance(r, ConditionalBlock) for sd in sdfg.all_sdfgs_recursive() for r in sd.all_control_flow_regions()
    )
    sdfg.apply_transformations_repeated(LoopToMap, validate=False)

    a = -np.abs(np.random.default_rng(0).standard_normal(8))  # all negative
    b = np.zeros(1)
    sdfg(a=a, b=b, N=8)
    assert np.isclose(b[0], 0.0)


def test_all_positives_acts_like_unconditional_sum():
    """Edge case: every ``a[i] > 0`` -- the cond is always true, so the
    rewrite behaves like an unconditional sum reduction. Numerics match
    ``np.sum(a)``."""

    @dace.program
    def kernel_all_positives_acts_like_unconditional_sum(a: dace.float64[N], b: dace.float64[1]):
        sum_val = 0.0
        for i in range(N):
            if a[i] > 0.0:
                sum_val = sum_val + a[i]
        b[0] = sum_val

    sdfg = kernel_all_positives_acts_like_unconditional_sum.to_sdfg(simplify=True)
    assert LoopToConditionalReduce().apply_pass(sdfg, {}) == 1, "the rewrite refused; the original loop also sums"
    assert not any(
        isinstance(r, ConditionalBlock) for sd in sdfg.all_sdfgs_recursive() for r in sd.all_control_flow_regions()
    )
    sdfg.apply_transformations_repeated(LoopToMap, validate=False)

    a = np.abs(np.random.default_rng(1).standard_normal(16))  # all positive
    b = np.zeros(1)
    sdfg(a=a, b=b, N=16)
    assert np.isclose(b[0], float(np.sum(a)))


# Refusal contracts.


def test_refuses_unconditional_accumulator():
    """``sum = sum + a[i]`` (no ConditionalBlock) is NOT in this pass's
    scope -- it's the plain reduction shape that ``LoopToReduce`` already
    handles. Refuse."""

    @dace.program
    def kernel_refuses_unconditional_accumulator(a: dace.float64[N], b: dace.float64[1]):
        sum_val = 0.0
        for i in range(N):
            sum_val = sum_val + a[i]
        b[0] = sum_val

    sdfg = kernel_refuses_unconditional_accumulator.to_sdfg(simplify=True)
    res = LoopToConditionalReduce().apply_pass(sdfg, {})
    assert res is None, "unconditional accumulator must be left for LoopToReduce"


def test_refuses_true_branch_writes_extra_array():
    """The true-branch writes both the accumulator AND another non-transient
    array (``if cond: sum += a[i]; out[i] = sum`` -- a conditional prefix-scan
    plus output). The rewrite would silently drop the ``out[i] = sum`` write,
    so refuse."""

    @dace.program
    def kernel_refuses_true_branch_writes_extra_array(a: dace.float64[N], out: dace.float64[N], b: dace.float64[1]):
        sum_val = 0.0
        for i in range(N):
            if a[i] > 0.0:
                sum_val = sum_val + a[i]
                out[i] = sum_val
        b[0] = sum_val

    sdfg = kernel_refuses_true_branch_writes_extra_array.to_sdfg(simplify=True)
    res = LoopToConditionalReduce().apply_pass(sdfg, {})
    assert res is None


def test_argument_written_before_the_guarded_accumulate_keeps_its_last_taken_value():

    @dace.program
    def kernel_argument_written_before_the_guarded_accumulate_keeps_its_last_taken_value(
        a: dace.float64[N], c: dace.float64[1], b: dace.float64[1]
    ):
        s = 0.0
        for i in range(N):
            if a[i] > 0.0:
                c[0] = a[i] * 2.0
                s = s + c[0]
        b[0] = s

    sdfg = kernel_argument_written_before_the_guarded_accumulate_keeps_its_last_taken_value.to_sdfg(simplify=True)
    a, b, c = np.array([2.0, 1.0, -1.0]), np.zeros(1), np.zeros(1)
    assert LoopToConditionalReduce().apply_pass(sdfg, {}) is None
    sdfg(a=a, b=b, c=c, N=3)
    assert (b[0], c[0]) == (6.0, 2.0)


def test_refuses_else_branch_with_content():
    """An else branch with side effects (``if cond: sum += a[i] else: ...``)
    isn't handled by the inline-ternary mask. Refuse."""

    @dace.program
    def kernel_refuses_else_branch_with_content(a: dace.float64[N], b: dace.float64[N]):
        sum_val = 0.0
        for i in range(N):
            if a[i] > 0.0:
                sum_val = sum_val + a[i]
            else:
                b[i] = -1.0
        # Final consumer of sum_val so it isn't optimised away.
        b[0] = sum_val + b[0]

    sdfg = kernel_refuses_else_branch_with_content.to_sdfg(simplify=True)
    res = LoopToConditionalReduce().apply_pass(sdfg, {})
    assert res is None


# Guard-protected addends: the rewrite runs the true-branch on every iteration.

K = dace.symbol("K")


def _has_conditional_block(sdfg: dace.SDFG) -> bool:
    return any(isinstance(r, ConditionalBlock) for r in sdfg.all_control_flow_regions(recursive=True))


def test_refuses_an_addend_the_guard_keeps_in_bounds():
    """``a[i + K]`` leaves the array for ``i >= N - K``; only the guard keeps the product from reading there. Lifting
    would compute ``a[i + K] * 2.0`` on every iteration, so the loop must stay guarded."""

    @dace.program
    def kernel_guarded_out_of_range_read(a: dace.float64[N], out: dace.float64[1]):
        s = 0.0
        for i in range(N):
            if i + K < N:
                s = s + a[i + K] * 2.0
        out[0] = s

    sdfg = kernel_guarded_out_of_range_read.to_sdfg(simplify=True)
    assert LoopToConditionalReduce().apply_pass(sdfg, {}) is None
    assert _has_conditional_block(sdfg)

    a = np.arange(1.0, 9.0)
    out = np.zeros(1)
    sdfg(a=a, out=out, N=8, K=3)
    assert out[0] == 2.0 * a[3:].sum()


def test_refuses_an_integer_modulo_the_guard_keeps_off_zero():
    """C's ``fmod(n[i], d[i])`` on integers divides by zero where the guard fails; lifting would evaluate it there."""

    @dace.program
    def kernel_guarded_integer_modulo(n: dace.int64[N], d: dace.int64[N], out: dace.float64[1]):
        s = 0.0
        for i in range(N):
            if d[i] > 0:
                s = s + np.fmod(n[i], d[i]) * 1.0
        out[0] = s

    sdfg = kernel_guarded_integer_modulo.to_sdfg(simplify=True)
    assert LoopToConditionalReduce().apply_pass(sdfg, {}) is None
    assert _has_conditional_block(sdfg)

    n = np.arange(1, 9, dtype=np.int64)
    d = np.array([3, 0, 2, 0, 3, 0, 4, 0], dtype=np.int64)
    out = np.zeros(1)
    sdfg(n=n, d=d, out=out, N=8)
    assert out[0] == float(sum(x % y for x, y in zip(n, d) if y > 0))


def test_a_python_integer_modulo_keeps_the_symbols_its_reads_are_promoted_to():
    """Python's ``n[i] % d[i]`` is ``PyMod``, which does not trap; simplify promotes the integer reads to symbols the
    loop body assigns on interstate edges (``d_index_0 = d[i]``). Rebuilding the body without those edges left them
    free, so such a loop is refused."""

    @dace.program
    def kernel_guarded_python_modulo(n: dace.int64[N], d: dace.int64[N], out: dace.float64[1]):
        s = 0.0
        for i in range(N):
            if d[i] > 0:
                s = s + (n[i] % d[i]) * 1.0
        out[0] = s

    sdfg = kernel_guarded_python_modulo.to_sdfg(simplify=True)
    free_before = set(sdfg.free_symbols)
    LoopToConditionalReduce().apply_pass(sdfg, {})
    assert set(sdfg.free_symbols) <= free_before

    n = np.arange(1, 9, dtype=np.int64)
    d = np.array([3, 0, 2, 0, 3, 0, 4, 0], dtype=np.int64)
    out = np.zeros(1)
    sdfg(n=n, d=d, out=out, N=8)
    assert out[0] == float(sum(x % y for x, y in zip(n, d) if y > 0))


def test_lifts_a_floating_point_division_the_guard_keeps_off_zero():
    """A floating-point division by zero yields an infinity, not a trap, and the mask discards it, so the guard does
    not protect anything and the loop lifts."""

    @dace.program
    def kernel_guarded_float_division(n: dace.float64[N], d: dace.float64[N], out: dace.float64[1]):
        s = 0.0
        for i in range(N):
            if d[i] != 0.0:
                s = s + n[i] / d[i]
        out[0] = s

    sdfg = kernel_guarded_float_division.to_sdfg(simplify=True)
    assert LoopToConditionalReduce().apply_pass(sdfg, {}) == 1
    assert not _has_conditional_block(sdfg)

    # Quotients and partial sums are exact in float64, so the reference holds under any order.
    n = np.array([2.0, 1.0, 6.0, 1.0, 12.0, 1.0, 8.0, 1.0])
    d = np.array([1.0, 0.0, 2.0, 0.0, 4.0, 0.0, 8.0, 0.0])
    out = np.zeros(1)
    sdfg(n=n, d=d, out=out, N=8)
    assert out[0] == 2.0 + 3.0 + 3.0 + 1.0


@pytest.mark.parametrize("descending", [False, True])
def test_lifts_a_shifted_read_that_stays_in_bounds(descending: bool):
    """``a[i + 1]`` over ``[0, N - 1)`` stays in bounds in either loop direction, so the bound check accepts it."""

    @dace.program
    def kernel_shifted_read_ascending(a: dace.float64[N], out: dace.float64[1]):
        s = 0.0
        for i in range(N - 1):
            if a[i] > 0.0:
                s = s + a[i + 1] * 2.0
        out[0] = s

    @dace.program
    def kernel_shifted_read_descending(a: dace.float64[N], out: dace.float64[1]):
        s = 0.0
        for i in range(N - 2, -1, -1):
            if a[i] > 0.0:
                s = s + a[i + 1] * 2.0
        out[0] = s

    kernel = kernel_shifted_read_descending if descending else kernel_shifted_read_ascending
    sdfg = kernel.to_sdfg(simplify=True)
    assert LoopToConditionalReduce().apply_pass(sdfg, {}) == 1
    assert not _has_conditional_block(sdfg)

    a = np.array([1.0, -2.0, 3.0, 4.0, -5.0, 6.0, 7.0, 8.0])
    out = np.zeros(1)
    sdfg(a=a, out=out, N=8)
    assert out[0] == sum(2.0 * a[i + 1] for i in range(7) if a[i] > 0.0)


# Cross-pass non-interference.


def test_doesnt_lift_an_argmax_loop():
    """Argmax conditional has a different shape (writes the array's value
    AND optionally an index, not an accumulator OP). LoopToConditionalReduce
    must refuse so ArgMaxLift handles it."""

    @dace.program
    def s314(a: dace.float64[N], result: dace.float64[1]):
        x = a[0]
        for i in range(1, N):
            if a[i] > x:
                x = a[i]
        result[0] = x

    sdfg = s314.to_sdfg(simplify=True)
    res = LoopToConditionalReduce().apply_pass(sdfg, {})
    assert res is None


def test_doesnt_lift_a_break_loop():
    """A break-loop has a BreakBlock in the conditional, not an accumulator
    update; the matcher refuses (the true-branch's terminal AN check
    eliminates this shape -- a BreakBlock is not an AccessNode)."""

    @dace.program
    def s481(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N], d: dace.float64[N]):
        for i in range(N):
            if d[i] < 0.0:
                break
            a[i] = a[i] + b[i] * c[i]

    sdfg = s481.to_sdfg(simplify=True)
    res = LoopToConditionalReduce().apply_pass(sdfg, {})
    assert res is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
