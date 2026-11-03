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


if __name__ == "__main__":
    test_an_enclosing_loop_does_not_copy_back_what_the_inner_loop_already_did()
    test_the_nest_still_computes_the_reference()
