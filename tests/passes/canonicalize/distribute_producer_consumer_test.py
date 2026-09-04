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
    """Run the canonicalize recipe up to (not including) the ``distribute`` stage.

    Stopping at ``distribute`` -- not at the later ``loop_to_x`` -- is what leaves this pass
    something to do: the ``distribute`` stage IS this pass, so running through it hands the tests
    an already-distributed SDFG on which a second application correctly reports ``None``.
    """
    sdfg = prog.to_sdfg(simplify=True)
    for label, unit in _build_stages():
        if label == 'distribute':
            break
        unit.apply_pass(sdfg, {})
    return sdfg


def _nloops(sdfg):
    return sum(1 for r in sdfg.all_control_flow_regions(recursive=True)
               if isinstance(r, LoopRegion) and r.loop_variable)


def _run_full(prog, **kw):
    """Canonicalize fully, run, return the SDFG. Used for the bit-exact value checks.

    The recipe carries its own ``distribute`` stage, so this just runs it. Applying the pass a
    second time before ``loop_to_x`` (as this helper used to) re-splits an already-distributed
    body into a shape the einsum lift then mis-classifies.
    """
    sdfg = prog.to_sdfg(simplify=True)
    for _label, unit in _build_stages():
        unit.apply_pass(sdfg, {})
    sdfg(**kw)
    return sdfg


@dace.program
def atax_loops(A: dace.float64[M, N], x: dace.float64[N], y: dace.float64[N]):
    """atax in its loop form: ``tmp = A @ x`` then ``y = tmp @ A``, sharing one for-i loop."""
    tmp = np.zeros([M], dtype=np.float64)
    for i in range(M):
        for j in range(N):
            tmp[i] += A[i, j] * x[j]
        for j in range(N):
            y[j] += A[i, j] * tmp[i]


def test_atax_matvecs_distribute_and_lift():
    """atax's two matvecs share a for-i loop coupled through tmp[i]; the
    distribution splits them, after which the pipeline lifts a matvec to an
    Einsum. Value must stay bit-exact.

    The kernel is written out here rather than imported from the polybench corpus: that copy is now
    ``y[:] = (A @ x) @ A``, which the frontend lowers straight to two Gemv library nodes, so it
    contains no loop for this pass to distribute.
    """
    sdfg = _to_loops(atax_loops)
    before = _nloops(sdfg)
    assert DistributeProducerConsumerLoop().apply_pass(sdfg, {}) == 1
    assert _nloops(sdfg) == before + 1, 'the coupled for-i loop must split into two'
    sdfg.validate()

    mm, nn = 38, 42
    rng = np.random.default_rng(0)
    A = rng.standard_normal((mm, nn))
    x = rng.standard_normal((nn, ))
    yref = (A @ x) @ A

    y = np.zeros(nn)
    lifted = _run_full(atax_loops, A=A.copy(), x=x.copy(), y=y, M=mm, N=nn)
    assert np.allclose(y, yref), 'distribution + lift must be value-preserving'
    libs = {type(n).__name__ for n, _ in lifted.all_nodes_recursive() if isinstance(n, nodes.LibraryNode)}
    assert 'Einsum' in libs, 'a split matvec should lift to an Einsum node'


def test_forward_aligned_producer_consumer_splits():
    """Two inner-j loops, coupled through tmp[i] at the aligned index -> SPLIT."""

    @dace.program
    def two_matvec(A: dace.float64[M, N], x: dace.float64[N], y: dace.float64[M]):
        tmp = np.zeros([M], dtype=np.float64)
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
                b[i] += a[i]  # reads a[i]
            for j in range(1):
                a[i] = b[i] * 0.5  # later block WRITES a that the earlier read

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


def test_mask_read_through_an_interstate_edge_refuses():
    """A container read only via an interstate edge still blocks the split.

    ``if mask[i]:`` lowers to an assignment ``mask_index = mask[i]`` on the edge into a
    ConditionalBlock plus a condition on the resulting symbol -- never an AccessNode. A
    read/write analysis that walks AccessNodes alone therefore called the second reader group
    independent of the group producing the mask, and split them. The consumer loop then ran
    against the mask left by the producer loop's LAST iteration instead of its own, which is a
    silent wrong answer: npbench mandelbrot1 reported an integer mismatch and nothing else.
    """
    M = dace.symbol('M', dtype=dace.int64)
    K = dace.symbol('K', dtype=dace.int64)

    @dace.program
    def mask_loop(Z: dace.float64[M], C: dace.float64[M], NN: dace.int64[M]):
        for n in range(K):
            mask = Z < 4.0
            for i in range(M):
                if mask[i]:
                    Z[i] = Z[i] * Z[i] + C[i]
            for i in range(M):
                if mask[i]:
                    NN[i] = n

    sdfg = mask_loop.to_sdfg(simplify=True)
    before = len([
        r for sd in sdfg.all_sdfgs_recursive() for r in sd.all_control_flow_regions(recursive=True)
        if isinstance(r, LoopRegion)
    ])
    assert DistributeProducerConsumerLoop().apply_pass(sdfg, {}) is None, 'the mask must keep its readers'
    after = len([
        r for sd in sdfg.all_sdfgs_recursive() for r in sd.all_control_flow_regions(recursive=True)
        if isinstance(r, LoopRegion)
    ])
    assert after == before

    # Values, on inputs whose mask CHANGES per iteration -- with a constant mask both orders agree
    # and the bug is invisible.
    seed_z = np.array([0.1, 0.5, 1.5, 1.9, 2.5, 3.5])
    seed_c = np.full(6, 0.05)
    z, c = seed_z.copy(), seed_c.copy()
    nn = np.zeros(6, dtype=np.int64)
    sdfg(Z=z, C=c, NN=nn, M=6, K=4)

    ref_z, ref_nn = seed_z.copy(), np.zeros(6, dtype=np.int64)
    for n in range(4):
        m = ref_z < 4.0
        for i in range(6):
            if m[i]:
                ref_z[i] = ref_z[i] * ref_z[i] + seed_c[i]
        for i in range(6):
            if m[i]:
                ref_nn[i] = n
    assert np.array_equal(nn, ref_nn), f'{nn} != {ref_nn}'
    assert np.allclose(z, ref_z)


if __name__ == '__main__':
    test_atax_matvecs_distribute_and_lift()
    test_forward_aligned_producer_consumer_splits()
    test_scalar_carried_recurrence_refuses()
    test_backward_dependence_refuses()
    test_mask_read_through_an_interstate_edge_refuses()
    print("OK")
