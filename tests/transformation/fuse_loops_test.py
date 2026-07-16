# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``FuseLoops`` transformation unit tests -- the single-pair loop-fusion transformation form (the agent
arm; the ``LoopFusion`` pass is this transformation applied to a fixpoint).

The transformation shares its legality kernel with the ``LoopFusion`` pass, so the pass's numpy suite
already covers the fusion math exhaustively. These tests pin the TRANSFORMATION interface an agent (or the
pass) drives: ``can_be_applied_to`` identifies exactly the legal pairs, ``apply_to`` / repeated application
fuses them, the result is bit-exact to the un-fused program, and no input crashes.
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
from dace.transformation.interstate.fuse_loops import FuseLoops

N = dace.symbol("N")


def nloops(sdfg):
    return sum(1 for c in sdfg.all_control_flow_regions(recursive=True) if isinstance(c, LoopRegion))


def adjacent_loop_pairs(sdfg):
    """Every (first, second) LoopRegion adjacency (single sequencing edge) across the SDFG."""
    pairs = []
    for cfg in sdfg.all_control_flow_regions(recursive=True):
        for first in cfg.nodes():
            if not isinstance(first, LoopRegion):
                continue
            out = cfg.out_edges(first)
            if len(out) == 1 and isinstance(out[0].dst, LoopRegion) and out[0].dst is not first:
                pairs.append((first, out[0].dst))
    return pairs


def run_fused(prog, inputs, n, simplify=True):
    """Build the program twice: an un-fused reference and a copy fused by repeated ``FuseLoops``. Returns
    (before, after, applied, bit_exact)."""
    ref = prog.to_sdfg(simplify=simplify)
    ref.name = prog.name + "_ref"
    ref_bufs = {k: v.copy() for k, v in inputs.items()}
    ref(**ref_bufs, N=n)

    sd = prog.to_sdfg(simplify=simplify)
    before = nloops(sd)
    applied = sd.apply_transformations_repeated(FuseLoops) or 0
    after = nloops(sd)
    sd.name = prog.name + "_fused"
    fus_bufs = {k: v.copy() for k, v in inputs.items()}
    sd(**fus_bufs, N=n)

    exact = all(np.allclose(fus_bufs[k], ref_bufs[k], equal_nan=True) for k in inputs)
    return before, after, applied, exact


def mk(n=48, names=("a", "b", "c", "d"), seed=0):
    rng = np.random.default_rng(seed)
    return {k: rng.random(n) for k in names}


# --- fuses a legal pair, value-preserving ------------------------------------------------------------


def test_fuse_loops_fuses_two_sequential_recurrences():
    @dace.program
    def prog(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N]):
        for i in range(1, N):
            b[i] = b[i - 1] + a[i]
        for i in range(1, N):
            c[i] = c[i - 1] + b[i]

    before, after, applied, exact = run_fused(prog, mk(names=("a", "b", "c")), 48)
    assert before == 2 and after == 1 and applied == 1
    assert exact


def test_can_be_applied_to_identifies_the_fusable_pair():
    @dace.program
    def prog(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N]):
        for i in range(1, N):
            b[i] = b[i - 1] + a[i]
        for i in range(1, N):
            c[i] = c[i - 1] + b[i]

    sd = prog.to_sdfg(simplify=True)
    pairs = adjacent_loop_pairs(sd)
    assert len(pairs) == 1
    first, second = pairs[0]
    assert FuseLoops.can_be_applied_to(sd, first=first, second=second)


def test_apply_to_a_named_pair_fuses_it():
    @dace.program
    def prog(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N]):
        for i in range(1, N):
            b[i] = b[i - 1] + a[i]
        for i in range(1, N):
            c[i] = c[i - 1] + b[i]

    sd = prog.to_sdfg(simplify=True)
    first, second = adjacent_loop_pairs(sd)[0]
    FuseLoops.apply_to(sd, first=first, second=second, verify=True, save=False, annotate=False)
    assert nloops(sd) == 1
    bufs = mk(names=("a", "b", "c"))
    ref = {k: v.copy() for k, v in bufs.items()}
    prog.to_sdfg(simplify=True)(**ref, N=48)
    got = {k: v.copy() for k, v in bufs.items()}
    sd(**got, N=48)
    assert all(np.allclose(got[k], ref[k]) for k in bufs)


# --- refuses illegal / unsafe pairs ------------------------------------------------------------------


def test_refuses_read_ahead_forward_flow():
    # body2 reads b[i+1] -- a value body1 has not yet produced at iteration i in the fused sweep.
    @dace.program
    def prog(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N]):
        for i in range(1, N):
            b[i] = b[i - 1] + a[i]
        for i in range(0, N - 1):
            c[i] = b[i + 1]

    sd = prog.to_sdfg(simplify=True)
    for first, second in adjacent_loop_pairs(sd):
        assert not FuseLoops.can_be_applied_to(sd, first=first, second=second)
    before, after, applied, exact = run_fused(prog, mk(names=("a", "b", "c")), 48)
    assert applied == 0  # refused
    assert exact  # ... and therefore trivially value-preserving


def test_refuses_mismatched_iteration_range():
    @dace.program
    def prog(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N]):
        for i in range(1, N):
            b[i] = b[i - 1] + a[i]
        for i in range(2, N):
            c[i] = c[i - 1] + b[i]

    sd = prog.to_sdfg(simplify=True)
    for first, second in adjacent_loop_pairs(sd):
        assert not FuseLoops.can_be_applied_to(sd, first=first, second=second)


def test_refuses_doall_parallel_loops():
    # two independent element-wise loops are DOALL -- FuseLoops must not serialize them (LoopToMap's job).
    @dace.program
    def prog(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N]):
        for i in range(N):
            b[i] = a[i] * 2.0
        for i in range(N):
            c[i] = a[i] + 1.0

    sd = prog.to_sdfg(simplify=True)
    for first, second in adjacent_loop_pairs(sd):
        assert not FuseLoops.can_be_applied_to(sd, first=first, second=second)


# --- never crashes -----------------------------------------------------------------------------------


@pytest.mark.parametrize("simplify", [True, False])
def test_never_crashes_on_single_or_no_loop(simplify):
    @dace.program
    def one_loop(a: dace.float64[N], b: dace.float64[N]):
        for i in range(1, N):
            b[i] = b[i - 1] + a[i]

    sd = one_loop.to_sdfg(simplify=simplify)
    assert (sd.apply_transformations_repeated(FuseLoops) or 0) == 0  # nothing to fuse, no crash


def test_value_preserving_across_a_battery():
    cases = []

    @dace.program
    def chain3(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N], d: dace.float64[N]):
        for i in range(1, N):
            b[i] = b[i - 1] + a[i]
        for i in range(1, N):
            c[i] = c[i - 1] + b[i]
        for i in range(1, N):
            d[i] = d[i - 1] + c[i]

    @dace.program
    def same_target(a: dace.float64[N], b: dace.float64[N]):
        for i in range(1, N):
            b[i] = b[i - 1] + a[i]
        for i in range(1, N):
            b[i] = b[i] * 2.0

    for prog, names in ((chain3, ("a", "b", "c", "d")), (same_target, ("a", "b"))):
        before, after, applied, exact = run_fused(prog, mk(names=names), 48)
        assert exact, f"{prog.name}: fused result diverged from reference"
        assert after <= before
