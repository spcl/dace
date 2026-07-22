# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""LoopFusion unit tests written as numpy-style ``@dace.program`` kernels with
explicit ``for`` loops (and a few ``dace.map`` bodies).

Each kernel is built twice: an un-fused reference and a ``LoopFusion``-run copy.
The invariant checked on EVERY case is value-preservation (fused output ==
reference output, bit-exact). Structural assertions (how many LoopRegions remain)
pin the intended fuse / refuse behaviour: consecutive same-range sequential
siblings fuse; a pair whose per-iteration reorder would change a value (a
read-ahead forward flow, a mismatched range) must be refused.
"""
import os

os.environ.setdefault("MPI4PY_RC_INITIALIZE", "0")
os.environ.setdefault("OMPI_MCA_pml", "ob1")
os.environ.setdefault("OMPI_MCA_btl", "self,vader")
os.environ.setdefault("UCX_VFS_ENABLE", "n")

import numpy as np
import pytest

import dace
from dace.sdfg.state import LoopRegion
from dace.transformation.passes.canonicalize.loop_fusion import LoopFusion

N = dace.symbol("N")


def _nloops(sdfg):
    return sum(1 for c in sdfg.all_control_flow_regions(recursive=True) if isinstance(c, LoopRegion))


def _run(prog, inputs, n):
    ref = prog.to_sdfg(simplify=True)
    ref.name = prog.name + "_ref"
    ref_bufs = {k: v.copy() for k, v in inputs.items()}
    ref(**ref_bufs, N=n)

    sd = prog.to_sdfg(simplify=True)
    before = _nloops(sd)
    applied = LoopFusion().apply_pass(sd, {}) or 0
    after = _nloops(sd)
    sd.name = prog.name + "_fused"
    fus_bufs = {k: v.copy() for k, v in inputs.items()}
    sd(**fus_bufs, N=n)

    exact = all(np.allclose(fus_bufs[k], ref_bufs[k], equal_nan=True) for k in inputs)
    return applied, before, after, exact, fus_bufs, ref_bufs


def _mk(n=48, names=("a", "b", "c", "d"), seed=0):
    rng = np.random.default_rng(seed)
    return {k: rng.random(n) for k in names}


# ---------------------------------------------------------------------------
# Fuse: consecutive same-range sequential sibling loops.
# ---------------------------------------------------------------------------


# LoopFusion targets the SEQUENTIAL residual loops LoopToMap refused (recurrences,
# in-place scans) -- a parallel elementwise loop is left to become a Map, NOT fused
# (see ``test_parallel_elementwise_loops_left_for_loop_to_map``). So the fuse cases
# below carry a loop-carried dependence.


def test_two_sequential_recurrence_chain_fuses():
    """Two prefix-sum recurrences chained (body2 reads a[i], same index) -> fuse."""

    @dace.program
    def k(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N]):
        for i in range(1, N):
            a[i] = a[i - 1] + c[i]
        for i in range(1, N):
            b[i] = b[i - 1] + a[i]

    applied, before, after, exact, *_ = _run(k, _mk(names=("a", "b", "c")), 48)
    assert exact
    assert before == 2 and after == 1 and applied == 1


def test_two_independent_recurrences_fuse():
    """Disjoint prefix recurrences (c from a, d from b) -> independent, fuse."""

    @dace.program
    def k(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N], d: dace.float64[N]):
        for i in range(1, N):
            c[i] = c[i - 1] + a[i]
        for i in range(1, N):
            d[i] = d[i - 1] + b[i]

    applied, before, after, exact, *_ = _run(k, _mk(), 48)
    assert exact
    assert before == 2 and after == 1


def test_three_sequential_recurrences_fuse_to_one():
    """A chain of three prefix recurrences collapses to a single loop."""

    @dace.program
    def k(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N], d: dace.float64[N]):
        for i in range(1, N):
            b[i] = b[i - 1] + a[i]
        for i in range(1, N):
            c[i] = c[i - 1] + b[i]
        for i in range(1, N):
            d[i] = d[i - 1] + c[i]

    applied, before, after, exact, *_ = _run(k, _mk(), 48)
    assert exact
    assert before == 3 and after < before and applied >= 1  # at least one pair fused, value-preserving


def test_scan_then_same_index_reuse_fuses():
    """A sequential scan then a second recurrence reading a[i] at the same index."""

    @dace.program
    def k(a: dace.float64[N], b: dace.float64[N]):
        for i in range(1, N):
            a[i] = a[i - 1] * 0.5 + a[i]
        for i in range(1, N):
            b[i] = b[i - 1] + a[i]

    applied, before, after, exact, *_ = _run(k, _mk(names=("a", "b")), 48)
    assert exact
    assert after == 1


def test_parallel_elementwise_loops_left_for_loop_to_map():
    """LoopFusion does NOT fuse two PARALLEL elementwise loops -- they are meant to
    become Maps (LoopToMap), so it leaves them untouched. Still value-preserving."""

    @dace.program
    def k(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N]):
        for i in range(N):
            b[i] = a[i] + 1.0
        for i in range(N):
            c[i] = b[i] * 2.0

    applied, before, after, exact, *_ = _run(k, _mk(names=("a", "b", "c")), 48)
    assert exact
    assert after == before  # parallel loops are not loop-fused


# ---------------------------------------------------------------------------
# Refuse: fusing would change a value, or ranges mismatch.
# ---------------------------------------------------------------------------


def test_forward_read_ahead_refused():
    """body2 reads a[i+1] (the still-old value) -> fusing would read the new
    value -> must NOT fuse; still value-preserving (nothing changed)."""

    @dace.program
    def k(a: dace.float64[N], b: dace.float64[N]):
        for i in range(N):
            a[i] = a[i] * 2.0
        for i in range(0, N - 1):
            b[i] = a[i + 1] + 1.0

    applied, before, after, exact, *_ = _run(k, _mk(names=("a", "b")), 48)
    assert exact
    assert after == 2  # refused


def test_mismatched_ranges_refused():
    """Different loop ranges cannot be fused."""

    @dace.program
    def k(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N]):
        for i in range(N):
            b[i] = a[i] + 1.0
        for i in range(1, N):
            c[i] = a[i] * 2.0

    applied, before, after, exact, *_ = _run(k, _mk(names=("a", "b", "c")), 48)
    assert exact
    assert after == 2  # refused


# ---------------------------------------------------------------------------
# Nested loops + map bodies.
# ---------------------------------------------------------------------------


def test_two_2d_row_loops_fuse():
    """Two same-range outer loops over a 2-D array with elementwise inner work."""
    M = dace.symbol("M")

    @dace.program
    def k(a: dace.float64[N, N], b: dace.float64[N, N]):
        for i in range(N):
            for j in range(N):
                a[i, j] = a[i, j] + 1.0
        for i in range(N):
            for j in range(N):
                b[i, j] = a[i, j] * 2.0

    rng = np.random.default_rng(1)
    inp = {"a": rng.random((16, 16)), "b": rng.random((16, 16))}
    ref = k.to_sdfg(simplify=True)
    ref.name = "k2d_ref"
    rb = {kk: v.copy() for kk, v in inp.items()}
    ref(**rb, N=16)
    sd = k.to_sdfg(simplify=True)
    LoopFusion().apply_pass(sd, {})
    sd.name = "k2d_fused"
    fb = {kk: v.copy() for kk, v in inp.items()}
    sd(**fb, N=16)
    assert np.allclose(fb["a"], rb["a"]) and np.allclose(fb["b"], rb["b"])


def test_does_not_fuse_map_with_loop():
    """A ``dace.map`` is a DATA-PARALLEL loop (MapFusion's domain), not a
    ``LoopRegion``. LoopFusion fuses only sequential loop-with-loop, so a parallel
    map followed by a sequential recurrence loop leaves BOTH untouched (the map is
    never absorbed into the loop). Value-preserving; the single LoopRegion has no
    sibling LoopRegion to fuse with, so nothing fuses."""
    from dace.sdfg import nodes

    @dace.program
    def k(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N]):
        for i in dace.map[0:N]:  # parallel map -- NOT a LoopRegion
            b[i] = a[i] + 1.0
        for i in range(1, N):    # sequential recurrence -- a LoopRegion
            c[i] = c[i - 1] + b[i]

    inputs = _mk(names=("a", "b", "c"))
    sd = k.to_sdfg(simplify=True)
    maps_before = sum(1 for n, _ in sd.all_nodes_recursive() if isinstance(n, nodes.MapEntry))
    loops_before = _nloops(sd)
    applied = LoopFusion().apply_pass(sd, {}) or 0
    maps_after = sum(1 for n, _ in sd.all_nodes_recursive() if isinstance(n, nodes.MapEntry))
    loops_after = _nloops(sd)
    # LoopFusion touched nothing: the map is not a LoopRegion, and there is only one
    # LoopRegion so it has no loop sibling to fuse with.
    assert applied == 0
    assert maps_before == maps_after == 1 and loops_before == loops_after == 1
    # value-preserving
    sd.name = "map_vs_loop_fused"
    fb = {kk: v.copy() for kk, v in inputs.items()}
    sd(**fb, N=48)
    ref = k.to_sdfg(simplify=True)
    ref.name = "map_vs_loop_ref"
    rb = {kk: v.copy() for kk, v in inputs.items()}
    ref(**rb, N=48)
    assert all(np.allclose(fb[kk], rb[kk]) for kk in inputs)


def test_does_not_loop_fuse_two_maps():
    """Two parallel ``dace.map`` blocks are MapFusion's domain, not LoopFusion's:
    LoopFusion sees zero LoopRegions and is a no-op (value-preserving)."""

    @dace.program
    def k(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N]):
        for i in dace.map[0:N]:
            b[i] = a[i] + 1.0
        for i in dace.map[0:N]:
            c[i] = b[i] * 2.0

    sd = k.to_sdfg(simplify=True)
    assert _nloops(sd) == 0  # both are maps, no LoopRegion
    applied = LoopFusion().apply_pass(sd, {}) or 0
    assert applied == 0  # nothing for LoopFusion to do


def test_four_loops_partial_chain():
    """Four loops: a->b->c chain fuses, d independent; all value-preserving."""

    @dace.program
    def k(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N], d: dace.float64[N]):
        for i in range(N):
            b[i] = a[i] + 1.0
        for i in range(N):
            c[i] = b[i] + 1.0
        for i in range(N):
            d[i] = c[i] + 1.0
        for i in range(N):
            a[i] = d[i] * 0.5

    applied, before, after, exact, *_ = _run(k, _mk(), 48)
    assert exact
    assert after < before


@pytest.mark.parametrize("n", [1, 2, 17, 64])
def test_value_preserving_across_sizes(n):
    """Fusion is value-preserving across a spread of sizes incl. tiny ones."""

    @dace.program
    def k(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N]):
        for i in range(N):
            b[i] = a[i] * a[i] + 1.0
        for i in range(N):
            c[i] = b[i] - a[i]

    applied, before, after, exact, *_ = _run(k, _mk(n=n, names=("a", "b", "c")), n)
    assert exact


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
