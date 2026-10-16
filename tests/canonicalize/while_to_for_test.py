# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""WhileToFor: a while loop that counts gets its counter, and the program computes what it did."""

import numpy as np

import dace
from dace.sdfg.state import LoopRegion
from dace.transformation.passes.canonicalize import canonicalize
from dace.transformation.passes.canonicalize.while_to_for import WhileToFor

N = dace.symbol("N")


def loops(sdfg: dace.SDFG) -> list[LoopRegion]:
    return [r for r in sdfg.all_control_flow_regions(recursive=True) if isinstance(r, LoopRegion)]


@dace.program
def unrolled_then_remainder(a: dace.float64[N], b: dace.float64[N]):
    i = 0
    while i + 4 <= N:
        b[i] = a[i] + 1.0
        b[i + 1] = a[i + 1] + 1.0
        b[i + 2] = a[i + 2] + 1.0
        b[i + 3] = a[i + 3] + 1.0
        i += 4
    while i < N:
        b[i] = a[i] + 1.0
        i += 1


@dace.program
def unrolled_sum_then_tail(a: dace.float64[N], out: dace.float64[1]):
    i = 0
    s0 = 0.0
    s1 = 0.0
    while i + 2 <= N:
        s0 += a[i]
        s1 += a[i + 1]
        i += 2
    tail = 0.0
    while i < N:
        tail += a[i]
        i += 1
    out[0] = s0 + s1 + tail


@dace.program
def counter_read_after(a: dace.float64[N], out: dace.int64[1]):
    i = 0
    while i < N:
        a[i] = a[i] * 2.0
        i += 3
    out[0] = i


@dace.program
def bump_then_read(a: dace.float64[N], b: dace.float64[N]):
    i = 0
    while i < N - 1:
        i += 1
        b[i] = a[i]


@dace.program
def doubling(a: dace.float64[N]):
    i = 1
    while i < N:
        a[i] = 1.0
        i = i * 2


def test_both_loops_of_an_unrolled_body_and_its_remainder_get_a_counter():
    sdfg = unrolled_then_remainder.to_sdfg(simplify=True)
    assert WhileToFor().apply_pass(sdfg, {}) == 2
    sdfg.validate()
    assert all(loop.loop_variable for loop in loops(sdfg))


def test_canonical_unrolled_loops_compute_the_reference_and_go_parallel():
    sdfg = unrolled_then_remainder.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True, validate_all=False, target="cpu")
    assert not [loop for loop in loops(sdfg) if not loop.loop_variable], "a while loop survived canonicalization"
    assert any(isinstance(n, dace.nodes.MapEntry) for n, _ in sdfg.all_nodes_recursive())
    rng = np.random.default_rng(3)
    for n in (0, 1, 3, 4, 7, 9, 64):
        a, b = rng.standard_normal(n), np.zeros(n)
        sdfg(a=a, b=b, N=n)
        assert np.array_equal(b, a + 1.0), n


def test_a_remainder_loop_starts_where_the_unrolled_loop_stopped_across_a_state():
    """unroll_reduction_11_accs: ``tail = 0`` sits between the loops, so the remainder's start is one edge further
    back than its entry edge; every element is summed exactly once."""
    sdfg = unrolled_sum_then_tail.to_sdfg(simplify=True)
    assert WhileToFor().apply_pass(sdfg, {}) == 2
    canonicalize(sdfg, validate=True, validate_all=False, target="cpu")
    for n in (0, 1, 2, 5, 8):
        a, out = np.arange(1.0, n + 1.0), np.zeros(1)
        sdfg(a=a, out=out, N=n)
        assert out[0] == n * (n + 1) / 2, (n, out)


def test_the_counter_ends_at_its_exit_value():
    """``i`` read after the loop is the first value that fails ``i < N``: a multiple of 3, not the last one taken."""
    sdfg = counter_read_after.to_sdfg(simplify=True)
    assert WhileToFor().apply_pass(sdfg, {}) == 1
    canonicalize(sdfg, validate=True, validate_all=False, target="cpu")
    for n in (0, 1, 3, 5, 9):
        a, out = np.ones(n), np.zeros(1, np.int64)
        sdfg(a=a, out=out, N=n)
        assert out[0] == -(-n // 3) * 3, (n, out)
        assert np.array_equal(a, np.where(np.arange(n) % 3 == 0, 2.0, 1.0)), n


def test_a_counter_read_after_its_bump_is_left_alone():
    """The body reads ``i`` after bumping it, so the bump is not the last thing an iteration does to it."""
    sdfg = bump_then_read.to_sdfg(simplify=True)
    assert WhileToFor().apply_pass(sdfg, {}) is None


def test_a_counter_that_does_not_step_by_a_constant_is_left_alone():
    sdfg = doubling.to_sdfg(simplify=True)
    assert WhileToFor().apply_pass(sdfg, {}) is None


if __name__ == "__main__":
    test_both_loops_of_an_unrolled_body_and_its_remainder_get_a_counter()
    test_canonical_unrolled_loops_compute_the_reference_and_go_parallel()
    test_a_remainder_loop_starts_where_the_unrolled_loop_stopped_across_a_state()
    test_the_counter_ends_at_its_exit_value()
    test_a_counter_read_after_its_bump_is_left_alone()
    test_a_counter_that_does_not_step_by_a_constant_is_left_alone()
