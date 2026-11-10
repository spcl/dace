# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A loop leaves its data where its head wants it: the loop's tail copies it back, so an enclosing loop must not copy
it a second time.

The shape is seidel's before the wavefront: per row a device map over the whole array (one moving no less than the
row loop's copies of ``A``, so it stays a kernel), then a host recurrence over the row. The row loop
copies ``A`` back to the device at its tail; the time loop around it used to read the row loop's last use (``A`` on the
host) and copy the whole of ``A`` to the device once more per step."""

import numpy as np
import pytest

import dace
from dace.sdfg.state import LoopRegion

N = dace.symbol("N")


@dace.program
def rows(A: dace.float64[N, N], T: dace.int64):
    for t in range(T):
        for i in range(1, N):
            A[:, :] = 0.5 * A[:, :]
            for j in range(1, N):
                A[i, j] = A[i, j] + A[i, j - 1]


def offloaded() -> dace.SDFG:
    sdfg = rows.to_sdfg(simplify=True)
    sdfg.apply_gpu_transformations()
    return sdfg


def copies_to_device(region) -> list[str]:
    """Copy states to the device directly in ``region``'s own graph."""
    return [b.label for b in region.nodes() if isinstance(b, dace.SDFGState) and b.label.startswith("copy_A_to_gpu")]


def test_an_enclosing_loop_does_not_copy_back_what_the_inner_loop_already_did():
    sdfg = offloaded()
    time_loop = next(b for b in sdfg.nodes() if isinstance(b, LoopRegion))
    row_loop = next(b for b in time_loop.nodes() if isinstance(b, LoopRegion))
    assert copies_to_device(row_loop), "the row loop's tail copies A back to the device"
    assert not copies_to_device(time_loop), "the time loop must not copy A to the device again"


@pytest.mark.gpu
def test_the_nest_still_computes_the_reference():
    """``A`` stays a host argument: the offloading stages its device copy itself."""
    n, steps = 17, 3
    a = np.random.default_rng(0).random((n, n))
    expected = a.copy()
    for step in range(steps):  # noqa: B007 -- each step repeats the same sweep
        for i in range(1, n):
            expected[:, :] = 0.5 * expected[:, :]
            for j in range(1, n):
                expected[i, j] = expected[i, j] + expected[i, j - 1]
    got = a.copy()
    offloaded()(A=got, T=steps, N=n)
    assert np.allclose(got, expected)


@dace.program
def guarded_recurrence(A: dace.float64[N], B: dace.float64[N], C: dace.float64[N], c: dace.int64):
    for i in dace.map[0:N]:
        A[i] = B[i] * 2.0
    for k in range(4):
        if k < c:
            for j in range(1, N):
                A[j] = A[j] + A[j - 1]
        else:
            C[0] = C[0] + 1.0


def test_a_loop_using_an_array_only_under_a_conditional_copies_it_at_its_boundary():
    sdfg = guarded_recurrence.to_sdfg(simplify=True)
    sdfg.apply_gpu_transformations()
    loop = next(b for b in sdfg.nodes() if isinstance(b, LoopRegion))
    inside = [s.label for s in loop.states() if s.label.startswith("copy_")]
    assert not inside, inside


@dace.program
def arms_on_both_sides(A: dace.float64[N], B: dace.float64[N], c: dace.int64):
    for i in dace.map[0:N]:
        A[i] = B[i] * 2.0
    for k in range(4):
        if k < c:
            for j in range(1, N):
                A[j] = A[j] + A[j - 1]
        else:
            for i in dace.map[0:N]:
                B[i] = A[i] + 1.0


def test_a_loop_whose_arms_use_an_array_on_both_sides_does_not_move_it_before_the_loop():
    sdfg = arms_on_both_sides.to_sdfg(simplify=True)
    sdfg.apply_gpu_transformations()
    before = [e for e in sdfg.start_block.edges() if (e.src.label, e.dst.label) == ("A_gpu", "A")]
    assert not before, "A is copied to the host ahead of a loop that may never use it there"


if __name__ == "__main__":
    test_a_loop_using_an_array_only_under_a_conditional_copies_it_at_its_boundary()
    test_a_loop_whose_arms_use_an_array_on_both_sides_does_not_move_it_before_the_loop()
    test_an_enclosing_loop_does_not_copy_back_what_the_inner_loop_already_did()
    test_the_nest_still_computes_the_reference()
