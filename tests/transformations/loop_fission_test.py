# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
""" Tests for LoopFission (loop distribution), the LoopRegion equivalent of
    MapFission. Mirrors the map-fission frontend kernels with ``dace.map``
    replaced by ``range`` so the frontend emits loops. Every test checks
    numerical equivalence against a deep-copied pre-pass run; loop counts are
    asserted where the independent-group partition is deterministic.

    LoopFission only distributes a single-body-state loop; data-dependent
    statements (and bodies with control flow / nested loops) stay in one
    loop -- those mirror as value-preserving no-ops.
"""
import copy

import numpy as np
import pytest

import dace
from dace.sdfg.state import LoopRegion
from dace.transformation.passes.loop_fission import LoopFission

N = dace.symbol('N')
START, STOP, STEP = (dace.symbol(s) for s in ('START', 'STOP', 'STEP'))


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


@dace.program
def loop_strided(a: dace.float64[40], A: dace.float64[40], B: dace.float64[40]):
    for i in range(0, 9, 2):
        A[i] = a[i] + 1.0
        B[i] = a[i] * 2.0


@dace.program
def loop_offset_strided(a: dace.float64[40], A: dace.float64[40], B: dace.float64[40]):
    for i in range(10, 29, 3):
        A[i] = a[i] + 1.0
        B[i] = a[i] * 2.0


@dace.program
def loop_symbolic_strided(a: dace.float64[64], A: dace.float64[64], B: dace.float64[64]):
    for i in range(START, STOP, STEP):
        A[i] = a[i] + 1.0
        B[i] = a[i] * 2.0


@dace.program
def loop_five_set_five_cpy(s0: dace.float64[N], s1: dace.float64[N], s2: dace.float64[N], s3: dace.float64[N],
                           s4: dace.float64[N], a0: dace.float64[N], a1: dace.float64[N], a2: dace.float64[N],
                           a3: dace.float64[N], a4: dace.float64[N], c0: dace.float64[N], c1: dace.float64[N],
                           c2: dace.float64[N], c3: dace.float64[N], c4: dace.float64[N]):
    for i in range(N):
        s0[i] = 0.0
        s1[i] = 1.0
        s2[i] = 2.0
        s3[i] = 3.0
        s4[i] = 4.0
        c0[i] = a0[i]
        c1[i] = a1[i]
        c2[i] = a2[i]
        c3[i] = a3[i]
        c4[i] = a4[i]


@dace.program
def loop_nested_two(x: dace.float64[N, N], y: dace.float64[N, N]):
    for j in range(N):
        for i in range(N):
            x[i, j] = 1.0
        for i in range(N):
            y[i, j] = 2.0


@dace.program
def loop_conditional(a: dace.float64[N], A: dace.float64[N], B: dace.float64[N], c: dace.int32[1]):
    for i in range(N):
        if c[0] > 0:
            A[i] = a[i] + 1.0
            B[i] = a[i] * 2.0


def _loop_count(sdfg: dace.SDFG) -> int:
    return sum(1 for cfg in sdfg.all_control_flow_regions(recursive=True) if isinstance(cfg, LoopRegion))


def _run(prog, args, kw, expect_loops):
    """Apply LoopFission, validate, assert loop count and e2e numerics.

    :param prog: The dace program.
    :param args: Keyword arrays compared before/after.
    :param kw: Extra scalar kwargs (symbols) for the calls.
    :param expect_loops: Expected LoopRegion count after fission.
    :returns: The transformed SDFG.
    """
    sdfg = prog.to_sdfg(simplify=True)
    ref = {k: v.copy() for k, v in args.items()}
    copy.deepcopy(sdfg)(**ref, **kw)

    LoopFission().apply_pass(sdfg, {})
    sdfg.validate()
    assert _loop_count(sdfg) == expect_loops, f"expected {expect_loops} loops, got {_loop_count(sdfg)}"

    out = {k: v.copy() for k, v in args.items()}
    sdfg(**out, **kw)
    for k in args:
        assert np.allclose(out[k], ref[k]), f"mismatch on {k}"
    return sdfg


def test_loop_fission_two():
    n = 16
    a = np.random.rand(n)
    _run(loop_two, dict(a=a, A=np.zeros(n), B=np.zeros(n)), dict(N=n), 2)


def test_loop_fission_three():
    n = 12
    a = np.random.rand(n)
    _run(loop_three, dict(a=a, A=np.zeros(n), B=np.zeros(n), C=np.zeros(n)), dict(N=n), 3)


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
    n = 10
    a = np.random.rand(n)
    sdfg = loop_dependent.to_sdfg(simplify=True)
    assert LoopFission().apply_pass(sdfg, {}) is None
    sdfg.validate()
    T, B = np.zeros(n), np.zeros(n)
    sdfg(a=a.copy(), T=T, B=B, N=n)
    assert np.allclose(T, a + 1.0) and np.allclose(B, (a + 1.0) * 2.0)


def test_loop_fission_loop_carried_is_noop():
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


def test_loop_fission_strided():
    a = np.random.rand(40)
    _run(loop_strided, dict(a=a, A=np.full(40, -1.0), B=np.full(40, -1.0)), {}, 2)


def test_loop_fission_offset_strided():
    a = np.random.rand(40)
    _run(loop_offset_strided, dict(a=a, A=np.full(40, -1.0), B=np.full(40, -1.0)), {}, 2)


def test_loop_fission_symbolic_strided():
    a = np.random.rand(64)
    _run(loop_symbolic_strided, dict(a=a, A=np.full(64, -1.0), B=np.full(64, -1.0)),
         dict(START=0, STOP=64, STEP=5), 2)


def test_loop_fission_many_set_cpy():
    n = 8
    arrs = {f'a{i}': np.random.rand(n) for i in range(5)}
    arrs.update({f's{i}': np.zeros(n) for i in range(5)})
    arrs.update({f'c{i}': np.zeros(n) for i in range(5)})
    _run(loop_five_set_five_cpy, arrs, dict(N=n), 10)


def test_loop_fission_perfect_nesting():
    """Parent loop with 2 independent inner loops -> 2 parent loops, each
    wrapping one inner loop (perfect-loop-nesting for loops)."""
    n = 5
    base = loop_nested_two.to_sdfg(simplify=True)
    x0, y0 = np.zeros((n, n)), np.zeros((n, n))
    copy.deepcopy(base)(x=x0, y=y0, N=n)

    sdfg = loop_nested_two.to_sdfg(simplify=True)
    assert LoopFission().apply_pass(sdfg, {}) is not None
    sdfg.validate()
    top = [c for c in sdfg.nodes() if isinstance(c, LoopRegion)]
    assert len(top) == 2, f"expected 2 parent loops, got {len(top)}"
    assert _loop_count(sdfg) == 4  # 2 parent + 2 inner

    x1, y1 = np.zeros((n, n)), np.zeros((n, n))
    sdfg(x=x1, y=y1, N=n)
    assert np.allclose(x1, x0) and np.allclose(y1, y0)
    assert np.allclose(x1, 1.0) and np.allclose(y1, 2.0)


def test_loop_fission_conditional_body_kept():
    """A conditional body is not a single state: conservative no-op, valid."""
    n = 7
    a = np.random.rand(n)
    sdfg = loop_conditional.to_sdfg(simplify=True)
    LoopFission().apply_pass(sdfg, {})
    sdfg.validate()
    A, B = np.zeros(n), np.zeros(n)
    sdfg(a=a.copy(), A=A, B=B, c=np.array([1], np.int32), N=n)
    assert np.allclose(A, a + 1.0) and np.allclose(B, a * 2.0)
    A0, B0 = np.full(n, 9.0), np.full(n, 9.0)
    sdfg(a=a.copy(), A=A0, B=B0, c=np.array([0], np.int32), N=n)
    assert np.allclose(A0, 9.0) and np.allclose(B0, 9.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
