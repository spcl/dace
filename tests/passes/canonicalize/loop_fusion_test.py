# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""LoopFusion: fusing consecutive same-range sequential sibling loops (the
LoopRegion analogue of MapFusion, for the residual loops LoopToMap refused).

Fusion must be value-preserving (compared against the un-fused build) and must
refuse any pair whose per-iteration reorder would change a value (forward flow /
read-behind anti / mismatched output) or that is independently parallel.
"""
import os

os.environ.setdefault("MPI4PY_RC_INITIALIZE", "0")
os.environ.setdefault("OMPI_MCA_pml", "ob1")
os.environ.setdefault("OMPI_MCA_btl", "self,vader")
os.environ.setdefault("UCX_VFS_ENABLE", "n")

import numpy as np

import dace
from dace.sdfg.state import LoopRegion
from dace.transformation.passes.canonicalize.loop_fusion import LoopFusion

N = dace.symbol("N")


def _nloops(sdfg):
    return sum(1 for c in sdfg.all_control_flow_regions(recursive=True) if isinstance(c, LoopRegion))


def _fuse_and_check(prog, n=64, arrays=("a", "b"), seed=0):
    """Return (applied, loops_before, loops_after, bit_exact_vs_unfused)."""
    rng = np.random.default_rng(seed)
    inputs = {name: rng.random(n) for name in ("a", "b", "c", "d")}

    ref = dace.SDFG.from_json(prog.to_sdfg(simplify=True).to_json())  # independent build
    ref_bufs = {k: v.copy() for k, v in inputs.items()}
    ref.name = prog.name + "_ref"
    ref(**{k: ref_bufs[k] for k in ref_bufs}, N=n)

    sd = prog.to_sdfg(simplify=True)
    before = _nloops(sd)
    applied = LoopFusion().apply_pass(sd, {})
    after = _nloops(sd)
    fus_bufs = {k: v.copy() for k, v in inputs.items()}
    sd.name = prog.name + "_fused"
    sd(**{k: fus_bufs[k] for k in fus_bufs}, N=n)

    exact = all(np.allclose(fus_bufs[name], ref_bufs[name]) for name in arrays)
    return applied, before, after, exact


def test_fuse_two_sequential_recurrences():
    """Two sequential recurrences, body2 reads ``a[i]`` (same index) -> fuse, bit-exact."""

    @dace.program
    def fuse_ok(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N]):
        for i in range(1, N):
            a[i] = a[i - 1] + c[i]
        for i in range(1, N):
            b[i] = b[i - 1] + a[i]

    applied, before, after, exact = _fuse_and_check(fuse_ok)
    assert applied == 1
    assert before == 2 and after == 1
    assert exact


def test_refuse_forward_flow_dependence():
    """body2 reads ``a[i+1]`` (read-ahead of body1's write) -> must NOT fuse."""

    @dace.program
    def refuse_fwd(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N]):
        for i in range(1, N - 1):
            a[i] = a[i - 1] + c[i]
        for i in range(1, N - 1):
            b[i] = b[i - 1] + a[i + 1]

    applied, before, after, exact = _fuse_and_check(refuse_fwd)
    assert applied is None
    assert after == before == 2
    assert exact  # unchanged sdfg still runs correctly


def test_refuse_read_behind_anti_dependence():
    """body1 reads ``a[i-1]`` that body2 overwrites earlier in the fused sweep -> refuse."""

    @dace.program
    def refuse_anti(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N]):
        for i in range(1, N):
            b[i] = b[i - 1] + a[i - 1]
        for i in range(1, N):
            a[i] = a[i - 1] + c[i]

    applied, before, after, exact = _fuse_and_check(refuse_anti)
    assert applied is None
    assert exact


def test_doall_pair_not_fused():
    """Two independently-parallel loops are left to LoopToMap, not fused."""

    @dace.program
    def doall_pair(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N], d: dace.float64[N]):
        for i in range(N):
            b[i] = a[i] * c[i]
        for i in range(N):
            d[i] = a[i] + c[i]

    applied, before, after, exact = _fuse_and_check(doall_pair, arrays=("b", "d"))
    assert applied is None
    assert after == before == 2


def test_fused_loop_blocks_uniquely_named():
    """The moved body state must be re-named unique in the fused loop, else a later
    fission/clone trips 'multiple blocks with the same name' (the s233 interaction)."""

    @dace.program
    def fuse_ok(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N]):
        for i in range(1, N):
            a[i] = a[i - 1] + c[i]
        for i in range(1, N):
            b[i] = b[i - 1] + a[i]

    sd = fuse_ok.to_sdfg(simplify=True)
    assert LoopFusion().apply_pass(sd, {}) == 1
    for loop in [c for c in sd.all_control_flow_regions(recursive=True) if isinstance(c, LoopRegion)]:
        labels = [b.label for b in loop.nodes()]
        assert len(labels) == len(set(labels)), f"duplicate block names in fused loop: {labels}"
    sd.validate()


def test_refuse_different_iteration_space():
    """Different start bounds -> not a same-range pair -> not fused."""

    @dace.program
    def diff_range(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N]):
        for i in range(1, N):
            a[i] = a[i - 1] + c[i]
        for i in range(2, N):
            b[i] = b[i - 1] + a[i]

    applied, before, after, exact = _fuse_and_check(diff_range)
    assert applied is None
    assert after == before == 2


if __name__ == "__main__":
    test_fuse_two_sequential_recurrences()
    test_refuse_forward_flow_dependence()
    test_refuse_read_behind_anti_dependence()
    test_doall_pair_not_fused()
    test_fused_loop_blocks_uniquely_named()
    test_refuse_different_iteration_space()
    print("loop_fusion tests ok")
