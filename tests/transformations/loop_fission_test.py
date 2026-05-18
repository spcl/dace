# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
""" Tests for LoopFission (loop distribution), the LoopRegion equivalent of
    MapFission. Kernels use the dace Python frontend with plain ``for`` loops;
    every test checks numerical equivalence against a deep-copied pre-pass run.
"""
import copy

import numpy as np
import pytest

import dace
from dace.sdfg.state import LoopRegion
from dace.transformation.passes.loop_fission import LoopFission

N = dace.symbol('N')


@dace.program
def loop_two(a: dace.float64[N], A: dace.float64[N], B: dace.float64[N]):
    for i in range(N):
        A[i] = a[i] + 1.0
        B[i] = a[i] * 2.0


@dace.program
def loop_three(a: dace.float64[N], A: dace.float64[N], B: dace.float64[N], C: dace.float64[N]):
    for i in range(N):
        A[i] = a[i] + 1.0
        B[i] = a[i] * 2.0
        C[i] = a[i] - 3.0


@dace.program
def loop_single(a: dace.float64[N], A: dace.float64[N]):
    for i in range(N):
        A[i] = a[i] + 1.0


@dace.program
def loop_dependent(a: dace.float64[N], T: dace.float64[N], B: dace.float64[N]):
    for i in range(N):
        T[i] = a[i] + 1.0
        B[i] = T[i] * 2.0


@dace.program
def loop_carried(A: dace.float64[N]):
    for i in range(1, N):
        A[i] = A[i - 1] + 1.0


def _loop_count(sdfg: dace.SDFG) -> int:
    return sum(1 for cfg in sdfg.all_control_flow_regions(recursive=True) if isinstance(cfg, LoopRegion))


def _run(prog, args, n, expect_loops):
    """Apply LoopFission, validate, and check e2e numerics.

    :param prog: The dace program.
    :param args: Keyword arrays (without ``N``).
    :param n: Value bound to symbol ``N``.
    :param expect_loops: Expected LoopRegion count after fission.
    :returns: The transformed SDFG.
    """
    sdfg = prog.to_sdfg(simplify=True)
    ref = {k: v.copy() for k, v in args.items()}
    copy.deepcopy(sdfg)(**ref, N=n)

    LoopFission().apply_pass(sdfg, {})
    sdfg.validate()
    assert _loop_count(sdfg) == expect_loops, f"expected {expect_loops} loops, got {_loop_count(sdfg)}"

    out = {k: v.copy() for k, v in args.items()}
    sdfg(**out, N=n)
    for k in args:
        assert np.allclose(out[k], ref[k]), f"mismatch on {k}"
    return sdfg


def test_loop_fission_two():
    n = 16
    a = np.random.rand(n)
    _run(loop_two, dict(a=a, A=np.zeros(n), B=np.zeros(n)), n, 2)


def test_loop_fission_three():
    n = 12
    a = np.random.rand(n)
    _run(loop_three, dict(a=a, A=np.zeros(n), B=np.zeros(n), C=np.zeros(n)), n, 3)


def test_loop_fission_single_is_noop():
    n = 8
    a = np.random.rand(n)
    sdfg = loop_single.to_sdfg(simplify=True)
    assert LoopFission().apply_pass(sdfg, {}) is None
    sdfg.validate()
    A = np.zeros(n)
    sdfg(a=a.copy(), A=A, N=n)
    assert np.allclose(A, a + 1.0)


def test_loop_fission_dependent_kept_together():
    """B reads T written in the same loop: the two statements share a written
    container, so they must stay in ONE loop (not distributed)."""
    n = 10
    a = np.random.rand(n)
    sdfg = loop_dependent.to_sdfg(simplify=True)
    assert LoopFission().apply_pass(sdfg, {}) is None  # one group -> no-op
    sdfg.validate()
    T, B = np.zeros(n), np.zeros(n)
    sdfg(a=a.copy(), T=T, B=B, N=n)
    assert np.allclose(T, a + 1.0) and np.allclose(B, (a + 1.0) * 2.0)


def test_loop_fission_loop_carried_is_noop():
    """A loop-carried dependency is a single component: no-op, stays correct."""
    n = 9
    sdfg = loop_carried.to_sdfg(simplify=True)
    assert LoopFission().apply_pass(sdfg, {}) is None
    sdfg.validate()
    A = np.zeros(n)
    A[0] = 5.0
    ref = A.copy()
    for i in range(1, n):
        ref[i] = ref[i - 1] + 1.0
    sdfg(A=A, N=n)
    assert np.allclose(A, ref)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
