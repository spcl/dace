# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Runtime versioning: a loop parallel under a runtime check runs ``if (check) parallel else sequential``."""

import os

os.environ.setdefault("MPI4PY_RC_INITIALIZE", "0")
os.environ.setdefault("OMPI_MCA_pml", "ob1")
os.environ.setdefault("OMPI_MCA_btl", "self,vader")
os.environ.setdefault("UCX_VFS_ENABLE", "n")

import numpy as np
import pytest

import dace
from dace.sdfg import nodes
from dace.sdfg.state import ConditionalBlock, LoopRegion
from dace.transformation.passes.canonicalize import canonicalize
from dace.transformation.passes.parallelization.contracts.under_constraint import ParallelizeUnderConstraint

N = dace.symbol("N", dtype=dace.int64)
M = dace.symbol("M", dtype=dace.int64)
T = dace.symbol("T", dtype=dace.int64)
K = dace.symbol("K", dtype=dace.int64)


@dace.program
def shift_add(s: dace.float64[3 * N]):
    for i in range(N):
        s[i + K] += s[i]


@dace.program
def shift_add_steps(s: dace.float64[3 * N], q: dace.float64[N]):
    for t in range(T):
        for i in range(N):
            s[i + K] += s[i]
        for i in range(N):
            q[i] += s[i]


@dace.program
def convolve(s: dace.float64[N + M], a: dace.float64[N], b: dace.float64[M]):
    for i in range(N):
        for j in range(M):
            s[i + j] += a[i] * b[j]


@dace.program
def segment_scale(ptr: dace.int64[N + 1], val: dace.float64[M], d: dace.float64[N], z: dace.float64[M]):
    for t in range(T):
        for r in range(N):
            for k in range(ptr[r], ptr[r + 1]):
                z[k] = val[k] * d[r]
        for r in range(N):
            d[r] = d[r] * 0.5


@dace.program
def offset_scatter(s: dace.float64[2 * N], idx: dace.int64[N], a: dace.float64[N]):
    for i in range(N):
        s[i + idx[i]] += a[i]


def segment_scale_reference(ptr, val, d, z, steps: int) -> None:
    for step in range(steps):
        for r in range(d.shape[0]):
            z[ptr[r] : ptr[r + 1]] = val[ptr[r] : ptr[r + 1]] * d[r]
        d *= 0.5


def shift_add_reference(s: np.ndarray, n: int, k: int, steps: int = 1, q: np.ndarray | None = None) -> None:
    for step in range(steps):
        for i in range(n):
            s[i + k] += s[i]
        if q is not None:
            q += s[:n]


def conditionals(sdfg: dace.SDFG) -> list[ConditionalBlock]:
    return [n for n in sdfg.all_control_flow_blocks(recursive=True) if isinstance(n, ConditionalBlock)]


def has_map(region) -> bool:
    return any(isinstance(n, nodes.MapEntry) for n, parent in region.all_nodes_recursive())


# Reproducer: CPF: render and canonicalization fixes for v1.0.1 (6db8721f3d)
def test_symbolic_distance_versions_loop():
    sdfg = shift_add.to_sdfg(simplify=True)
    assert ParallelizeUnderConstraint().apply_pass(sdfg, {}) == 1
    (conditional,) = conditionals(sdfg)
    (par_condition, par_region), (else_condition, seq_region) = conditional.branches
    assert else_condition is None
    assert has_map(par_region), "the checked arm runs the loop as a map"
    fallback = [r for r in seq_region.all_control_flow_regions() if isinstance(r, LoopRegion) and r.loop_variable]
    assert len(fallback) == 1 and fallback[0].pinned_sequential
    assert fallback[0].specialization_hint.startswith("sequential -- the runtime check found")


@pytest.mark.parametrize("k", [0, 3, 8, 16])
# Reproducer: CPF: render and canonicalization fixes for v1.0.1 (6db8721f3d)
def test_symbolic_distance_both_arms_compute(k: int):
    """``K`` = 0 and ``K >= N`` take the parallel arm, ``K = 3`` the fallback."""
    n = 8
    sdfg = shift_add.to_sdfg(simplify=True)
    ParallelizeUnderConstraint().apply_pass(sdfg, {})
    rng = np.random.default_rng(k)
    s = rng.random(3 * n)
    expected = s.copy()
    shift_add_reference(expected, n, k)
    sdfg(s=s, N=n, K=k)
    assert np.allclose(s, expected)


# Reproducer: CPF: render and canonicalization fixes for v1.0.1 (6db8721f3d)
def test_distance_check_runs_once_before_time_loop():
    sdfg = shift_add_steps.to_sdfg(simplify=True)
    assert ParallelizeUnderConstraint().apply_pass(sdfg, {}) == 1
    (conditional,) = conditionals(sdfg)
    flag = conditional.branches[0][0].as_string.strip()
    assert flag in sdfg.symbols, "the conditional branches on the saved check"
    setters = [e for e in sdfg.all_interstate_edges(recursive=True) if flag in e.data.assignments]
    assert len(setters) == 1, "one evaluation of the check"
    time_loop = conditional.parent_graph
    assert isinstance(time_loop, LoopRegion) and time_loop.loop_variable == "t"
    assert setters[0] in sdfg.edges() and setters[0].dst is time_loop, "evaluated outside, right before the t loop"
    for k in (3, 8):
        n, steps = 8, 3
        rng = np.random.default_rng(k)
        s, q = rng.random(3 * n), rng.random(n)
        expected_s, expected_q = s.copy(), q.copy()
        shift_add_reference(expected_s, n, k, steps, expected_q)
        sdfg(s=s, q=q, N=n, K=k, T=steps)
        assert np.allclose(s, expected_s) and np.allclose(q, expected_q)


# Reproducer: CPF: render and canonicalization fixes for v1.0.1 (6db8721f3d)
def test_canonical_shift_add_versions_the_snapshot_fallback():
    """The snapshot covers ``K <= 0``; its fallback is versioned again on ``K >= N``. Every ``K`` computes."""
    sdfg = shift_add.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True)
    hints = [
        r.specialization_hint or ""
        for r in sdfg.all_control_flow_regions(recursive=True)
        if isinstance(r, LoopRegion) and r.loop_variable
    ]
    assert any(h.startswith("sequential -- the runtime check found") for h in hints)
    maps = [n for n, parent in sdfg.all_nodes_recursive() if isinstance(n, nodes.MapEntry)]
    assert any((m.specialization_hint or "").startswith("parallel -- the runtime check found") for m in maps)
    n = 8
    for k in (0, 3, 8, 16):
        rng = np.random.default_rng(k)
        s = rng.random(3 * n)
        expected = s.copy()
        shift_add_reference(expected, n, k)
        sdfg(s=s, N=n, K=k)
        assert np.allclose(s, expected), k


# Reproducer: CPF: render and canonicalization fixes for v1.0.1 (6db8721f3d)
def test_convolution_keeps_reduction_across_outer_map():
    """``s[i + j] +=`` over a parallel ``i`` and an inner ``j``: rows ``i`` and ``i + 1`` meet, so the update stays a
    reduction (WCR) instead of a plain read-modify-write."""
    sdfg = convolve.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True)
    assert not any(isinstance(r, LoopRegion) for r in sdfg.all_control_flow_regions(recursive=True))
    assert any(e.data.data == "s" and e.data.wcr is not None for e, parent in sdfg.all_edges_recursive()), (
        "the update of s must stay a reduction"
    )
    n, m = 300, 200
    rng = np.random.default_rng(0)
    a, b, s = rng.random(n), rng.random(m), rng.random(n + m)
    expected = s.copy()
    expected[: n + m - 1] += np.convolve(a, b)
    sdfg(s=s, a=a, b=b, N=n, M=m)
    assert np.allclose(s, expected)


# Reproducer: CPF: render and canonicalization fixes for v1.0.1 (6db8721f3d)
def test_segment_writes_versioned_on_one_hoisted_pointer_check():
    """Rows write ``z[ptr[r] : ptr[r + 1]]``: parallel when ``ptr`` is non-decreasing, checked once before ``t``."""
    sdfg = segment_scale.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True)
    checks = [
        (n, st)
        for n, st in sdfg.all_nodes_recursive()
        if isinstance(n, nodes.MapEntry) and "runtime check: one pass over ptr" in (n.specialization_hint or "")
    ]
    assert len(checks) == 1, "one check of ptr"
    region = checks[0][1].parent_graph
    while region is not None and not isinstance(region, dace.SDFG):
        assert not isinstance(region, LoopRegion), "the check runs outside every loop"
        region = region.parent_graph
    hints = [n.specialization_hint or "" for n, parent in sdfg.all_nodes_recursive() if isinstance(n, nodes.MapEntry)]
    assert any(h.startswith("parallel -- the runtime check found ptr non-decreasing") for h in hints)
    n, nnz, steps = 6, 12, 3
    rng = np.random.default_rng(1)
    for ptr in (np.array([0, 2, 4, 4, 7, 9, 12]), np.array([0, 5, 2, 8, 6, 10, 12])):
        val, d = rng.random(nnz), rng.random(n)
        z = np.zeros(nnz)
        expected_d, expected_z = d.copy(), z.copy()
        segment_scale_reference(ptr, val, expected_d, expected_z, steps)
        sdfg(ptr=ptr.astype(np.int64), val=val, d=d, z=z, N=n, M=nnz, T=steps)
        assert np.allclose(z, expected_z) and np.allclose(d, expected_d)


# Reproducer: CPF: render and canonicalization fixes for v1.0.1 (6db8721f3d)
def test_offset_scatter_is_a_reduction():
    """``s[i + idx[i]] +=`` combines colliding updates in any order: a parallel reduction, never a kept loop."""
    sdfg = offset_scatter.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True)
    assert not any(isinstance(r, LoopRegion) and r.loop_variable for r in sdfg.all_control_flow_regions(recursive=True))
    n = 64
    rng = np.random.default_rng(2)
    for idx in (rng.permutation(n), rng.integers(0, 4, n)):
        a, s = rng.random(n), rng.random(2 * n)
        expected = s.copy()
        np.add.at(expected, np.arange(n) + idx, a)
        sdfg(s=s, idx=idx.astype(np.int64), a=a, N=n)
        assert np.allclose(s, expected)


if __name__ == "__main__":
    test_symbolic_distance_versions_loop()
    test_symbolic_distance_both_arms_compute(0)
    test_symbolic_distance_both_arms_compute(3)
    test_symbolic_distance_both_arms_compute(8)
    test_symbolic_distance_both_arms_compute(16)
    test_distance_check_runs_once_before_time_loop()
    test_canonical_shift_add_versions_the_snapshot_fallback()
    test_convolution_keeps_reduction_across_outer_map()
    test_segment_writes_versioned_on_one_hoisted_pointer_check()
    test_offset_scatter_is_a_reduction()
