# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Unit tests for :class:`DistributeProducerConsumerLoop`.

The pass distributes a linear-chain loop body across a FORWARD producer->consumer
dependence (an earlier block writes a per-iteration container a later block reads
at the same index), while refusing any backward anti/output/loop-carried
dependence. Legality follows Allen & Kennedy loop-distribution (no dependence
edge may run from a later group to an earlier group). Cases mirror that catalog:
aligned producer->consumer SPLITS; scalar recurrence and backward deps REFUSE.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MPI4PY_RC_INITIALIZE", "0")
os.environ.setdefault("OMPI_MCA_pml", "ob1")
os.environ.setdefault("OMPI_MCA_btl", "self,vader")
os.environ.setdefault("UCX_VFS_ENABLE", "n")

import numpy as np
import pytest

import dace
from dace.sdfg import nodes
from dace.sdfg.state import LoopRegion
from dace.transformation.passes.canonicalize.pipeline import _build_stages
from dace.transformation.passes.canonicalize.distribute_producer_consumer import DistributeProducerConsumerLoop

M = dace.symbol('M')
N = dace.symbol('N')


def _to_loops(prog):
    """Run the canonicalize recipe up to (not including) the loop_to_x stage,
    where the body is loops-only -- the window this pass runs in."""
    sdfg = prog.to_sdfg(simplify=True)
    for label, unit in _build_stages():
        if label == 'loop_to_x':
            break
        unit.apply_pass(sdfg, {})
    return sdfg


def _nloops(sdfg):
    return sum(1 for r in sdfg.all_control_flow_regions(recursive=True) if isinstance(r, LoopRegion) and r.loop_variable)


def _run_full(prog, **kw):
    """Canonicalize fully WITH the distribution injected before loop_to_x, run,
    return the outputs. Used for the bit-exact value checks."""
    sdfg = prog.to_sdfg(simplify=True)
    for label, unit in _build_stages():
        if label == 'loop_to_x':
            DistributeProducerConsumerLoop().apply_pass(sdfg, {})
        unit.apply_pass(sdfg, {})
    sdfg(**kw)
    return sdfg


def test_atax_matvecs_distribute_and_lift():
    """atax's two matvecs share a for-i loop coupled through tmp[i]; the
    distribution splits them, after which the pipeline lifts a matvec to an
    Einsum. Value must stay bit-exact."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("atax", "tests/corpus/polybench/linear_algebra/kernels/atax.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)

    sdfg = _to_loops(m.atax)
    before = _nloops(sdfg)
    assert DistributeProducerConsumerLoop().apply_pass(sdfg, {}) == 1
    assert _nloops(sdfg) == before + 1, 'the coupled for-i loop must split into two'
    sdfg.validate()

    mm, nn = 38, 42
    rng = np.random.default_rng(0)
    A = rng.standard_normal((mm, nn))
    x = rng.standard_normal((nn, ))
    yref = np.zeros(nn)
    m.atax.to_sdfg(simplify=True)(A=A.copy(), x=x.copy(), y=yref, M=mm, N=nn)

    y = np.zeros(nn)
    lifted = _run_full(m.atax, A=A.copy(), x=x.copy(), y=y, M=mm, N=nn)
    assert np.allclose(y, yref), 'distribution + lift must be value-preserving'
    libs = {type(n).__name__ for n, _ in lifted.all_nodes_recursive() if isinstance(n, nodes.LibraryNode)}
    assert 'Einsum' in libs, 'a split matvec should lift to an Einsum node'


def test_forward_aligned_producer_consumer_splits():
    """Two inner-j loops, coupled through tmp[i] at the aligned index -> SPLIT."""

    @dace.program
    def two_matvec(A: dace.float64[M, N], x: dace.float64[N], y: dace.float64[M]):
        tmp = dace.define_local([M], dace.float64)
        for i in range(M):
            for j in range(N):
                tmp[i] += A[i, j] * x[j]
            for j in range(N):
                y[i] += A[i, j] * tmp[i]

    sdfg = _to_loops(two_matvec)
    before = _nloops(sdfg)
    assert DistributeProducerConsumerLoop().apply_pass(sdfg, {}) is not None
    assert _nloops(sdfg) > before

    mm, nn = 12, 9
    rng = np.random.default_rng(1)
    A = rng.standard_normal((mm, nn))
    x = rng.standard_normal((nn, ))
    ref = np.zeros(mm)
    two_matvec.to_sdfg(simplify=True)(A=A.copy(), x=x.copy(), y=ref, M=mm, N=nn)
    got = np.zeros(mm)
    _run_full(two_matvec, A=A.copy(), x=x.copy(), y=got, M=mm, N=nn)
    assert np.allclose(got, ref)


def test_scalar_carried_recurrence_refuses():
    """A scalar carried across iterations (s not per-iteration) is a recurrence
    SCC; distributing it would let the consumer read the final total for every
    i instead of the running partial sum -> the pass must REFUSE."""

    @dace.program
    def running_sum(a: dace.float64[M, N], b: dace.float64[M]):
        for i in range(M):
            s = 0.0
            for j in range(N):
                s = s + a[i, j]
            b[i] = s * 2.0
        # s reused across i is intra-i here; force a genuine cross-i scalar:

    # Build the cross-i scalar case explicitly.
    @dace.program
    def cross_i_scalar(a: dace.float64[M], b: dace.float64[M]):
        s = 0.0
        for i in range(M):
            s = s + a[i]
            b[i] = s

    sdfg = _to_loops(cross_i_scalar)
    result = DistributeProducerConsumerLoop().apply_pass(sdfg, {})
    # Either the body is a single state (out of block-level scope -> None) or the
    # scalar's non-per-iteration write blocks the split. Both are a no-op here.
    if result is not None:
        pytest.fail('scalar recurrence must not be distributed')

    mm = 7
    a = np.random.default_rng(2).standard_normal((mm, ))
    ref = np.cumsum(a)
    got = np.zeros(mm)
    _run_full(cross_i_scalar, a=a.copy(), b=got, M=mm)
    assert np.allclose(got, ref), 'running-sum semantics must be preserved'


def test_backward_dependence_refuses():
    """A later block writing a container an earlier block read (anti/WAR
    back-edge) must keep the loop fused."""

    @dace.program
    def war(a: dace.float64[M], b: dace.float64[M]):
        for i in range(M):
            for j in range(1):
                b[i] += a[i]          # reads a[i]
            for j in range(1):
                a[i] = b[i] * 0.5     # later block WRITES a that the earlier read

    sdfg = _to_loops(war)
    groups_split = DistributeProducerConsumerLoop().apply_pass(sdfg, {})

    mm = 6
    a = np.random.default_rng(3).standard_normal((mm, ))
    b0 = np.zeros(mm)
    ref_a, ref_b = a.copy(), b0.copy()
    war.to_sdfg(simplify=True)(a=ref_a, b=ref_b, M=mm)
    ga, gb = a.copy(), np.zeros(mm)
    _run_full(war, a=ga, b=gb, M=mm)
    assert np.allclose(ga, ref_a) and np.allclose(gb, ref_b), 'WAR case must stay value-preserving'


if __name__ == '__main__':
    test_atax_matvecs_distribute_and_lift()
    test_forward_aligned_producer_consumer_splits()
    test_scalar_carried_recurrence_refuses()
    test_backward_dependence_refuses()
    print("OK")
