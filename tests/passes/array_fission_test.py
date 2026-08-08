# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``ArrayFission`` / ``PrivatizeArrays`` may only version an array it can PROVE is fully overwritten.

A transient array reused as a per-iteration scratch buffer is written before it is read on every
iteration, so its iterations share only its *name*. Giving each dominating write its own container
removes that false write-after-write, which is what otherwise makes the shared buffer look
loop-carried and blocks ``LoopToMap``. That is the scalar-privatization argument, and it transfers
to arrays only under one extra premise: the dominating write must be a must-def of the WHOLE array.

The write-shadow analysis decides shadowing by graph dominance alone and never inspects a subset.
For a scalar that is exact -- any write to a scalar writes all of it. For an array it is false: a
dominating write of ``tmp[0:M//2]`` does not shadow a read of ``tmp[M-1]``. Versioning on the back
of it would hand the new container an element the previous iteration wrote, which is a silent
miscompile with no structural symptom. ``ArrayWriteShadowScopes`` therefore reports only arrays for
which every write provably covers the full extent, and the pass refuses everything else.

The negative tests below matter as much as the positive one: both refusals are the difference
between a missed optimization and a wrong answer.
"""
import copy

import numpy as np
import pytest

import dace
from dace import data as dt
from dace.sdfg import nodes as nd
from dace.sdfg.state import LoopRegion
from dace.transformation.interstate import LoopToMap
from dace.transformation.pass_pipeline import Pipeline
from dace.transformation.passes import ArrayFission, PrivatizeArrays, ScalarFission
from dace.transformation.passes.analysis import ArrayWriteShadowScopes, fully_overwritten_arrays

N = 16
M = 8


@dace.program
def full_overwrite(A: dace.float64[N, M], out1: dace.float64[N], out2: dace.float64[N]):
    """``tmp`` is shared by both loops but fully redefined at the top of every iteration."""
    tmp = np.zeros((M, ), dtype=np.float64)
    for i in range(N):
        tmp[:] = A[i] * 2.0
        out1[i] = tmp[0] + tmp[M - 1]
    for i in range(N):
        tmp[:] = A[i] * 3.0
        out2[i] = tmp[0] + tmp[M - 1]


@dace.program
def fanout_reads(A: dace.float64[N, M], out1: dace.float64[N, M], out2: dace.float64[N, M]):
    """Each loop reads ``tmp`` at two offsets, so one map connector fans out to two consumers."""
    tmp = np.zeros((N, M), dtype=np.float64)
    for i in range(4):
        tmp[:] = A * 2.0
        out1[1:N - 1, 1:M - 1] = tmp[0:N - 2, 1:M - 1] + tmp[2:N, 1:M - 1]
    for i in range(4):
        tmp[:] = A * 3.0
        out2[1:N - 1, 1:M - 1] = tmp[1:N - 1, 0:M - 2] + tmp[1:N - 1, 2:M]


@dace.program
def partial_overwrite(A: dace.float64[N, M], out1: dace.float64[N], out2: dace.float64[N]):
    """Only the first half of ``tmp`` is written; ``tmp[M - 1]`` is genuinely loop-carried."""
    tmp = np.zeros((M, ), dtype=np.float64)
    for i in range(N):
        tmp[0:M // 2] = A[i, 0:M // 2] * 2.0
        out1[i] = tmp[0] + tmp[M - 1]
    for i in range(N):
        tmp[0:M // 2] = A[i, 0:M // 2] * 3.0
        out2[i] = tmp[0] + tmp[M - 1]


@dace.program
def carried_between_loops(A: dace.float64[N, M], out: dace.float64[N, M]):
    """``tmp`` is produced by the first loop and consumed by the second before it redefines it.

    Every write covers all of ``tmp`` and every top-level block is a ``LoopRegion``, which is the
    npbench ``vadv`` shape (``data_col``: written by the first step of the backward substitution,
    read by the rest of it). Both trip counts are concrete, so the producer loop IS a dominating
    write and the whole chain now sits in ONE write scope; nothing may be versioned out of it.
    """
    tmp = np.ndarray((N, M), dtype=np.float64)
    for i in range(1):
        tmp[:] = A * 2.0
    for i in range(3):
        out[:] = tmp * 3.0
        tmp[:] = out + 1.0


K = dace.symbol('K')


@dace.program
def region_write_then_consumers(A: dace.float64[N, M], out1: dace.float64[N], out2: dace.float64[N]):
    """Each producer loop writes ``tmp`` in full and a LATER loop consumes it.

    Every top-level block is a ``LoopRegion``, so the only way ``tmp`` gets a dominating write at
    all is to accept a region as one -- and that is sound here because both producers have a
    concrete trip count of 1. With the two producers rooted, the four loops fall into two
    independent versions of ``tmp``.
    """
    tmp = np.ndarray((M, ), dtype=np.float64)
    for _ in range(1):
        tmp[:] = A[0] * 2.0
    for i in range(N):
        out1[i] = tmp[0] + tmp[M - 1]
    for _ in range(1):
        tmp[:] = A[0] * 3.0
    for i in range(N):
        out2[i] = tmp[0] + tmp[M - 1]


@dace.program
def symbolic_producer(A: dace.float64[N, M], out1: dace.float64[N]):
    """The same shape with a SYMBOLIC producer trip count, which may be zero."""
    tmp = np.zeros((M, ), dtype=np.float64)
    for _ in range(K):
        tmp[:] = A[0] * 2.0
    for i in range(N):
        out1[i] = tmp[0] + tmp[M - 1]


@dace.program
def conditional_overwrite(A: dace.float64[N, M], out1: dace.float64[N]):
    """The full write is under a non-exhaustive ``if``, so no iteration is guaranteed to define it."""
    tmp = np.zeros((M, ), dtype=np.float64)
    for i in range(N):
        if A[i, 0] > 0.5:
            tmp[:] = A[i] * 2.0
        out1[i] = tmp[0] + tmp[M - 1]


def loop_count(sdfg: dace.SDFG) -> int:
    return len([c for c in sdfg.all_control_flow_regions() if isinstance(c, LoopRegion)])


def map_count(sdfg: dace.SDFG) -> int:
    return len([n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nd.MapEntry)])


def test_full_overwrite_is_privatized_and_parallelizes():
    """Positive: both loops fully redefine ``tmp``, so each may have its own copy and become a Map."""
    sdfg = full_overwrite.to_sdfg(simplify=True)
    assert 'tmp' in fully_overwritten_arrays(sdfg)
    assert loop_count(sdfg) == 2
    # The shared name is the only thing tying the two loops together: without fission the
    # whole-array write is not indexed by the iteration variable, so LoopToMap refuses.
    assert sdfg.apply_transformations_repeated([LoopToMap]) == 0

    renamed = PrivatizeArrays().apply_pass(sdfg, {})['ArrayFission']
    sdfg.validate()
    assert len(renamed['tmp']) >= 2, f'expected one container per dominating write, got {dict(renamed)}'

    assert sdfg.apply_transformations_repeated([LoopToMap]) == 2
    assert loop_count(sdfg) == 0
    assert map_count(sdfg) > 0


def test_full_overwrite_value_preserving():
    rng = np.random.default_rng(7)
    A = rng.random((N, M))
    want1 = A[:, 0] * 2.0 + A[:, M - 1] * 2.0
    want2 = A[:, 0] * 3.0 + A[:, M - 1] * 3.0

    sdfg = full_overwrite.to_sdfg(simplify=True)
    sdfg.name = 'array_fission_full_value'
    PrivatizeArrays().apply_pass(sdfg, {})
    sdfg.apply_transformations_repeated([LoopToMap])
    got1, got2 = np.zeros(N), np.zeros(N)
    sdfg.compile()(A=A, out1=got1, out2=got2)
    assert np.allclose(got1, want1, rtol=1e-12, atol=1e-12)
    assert np.allclose(got2, want2, rtol=1e-12, atol=1e-12)


def test_fanout_reads_rename_every_branch():
    """The rename must follow the memlet TREE, not one path through it.

    An array read is routed into a map through a single ``OUT_tmp`` connector and then fans out to
    several consumers (``tmp[i - 1, j]`` and ``tmp[i + 1, j]``). ``memlet_path`` is one linear route
    and renames only one branch; the siblings keep naming a container their access node no longer
    references, which validation rejects with "Memlet data does not match source or destination data
    nodes". npbench ``channel_flow`` is the shape (``pn`` in the pressure-poisson stencil).
    """
    sdfg = fanout_reads.to_sdfg(simplify=True)
    fanned = [(state.label, conn) for state in sdfg.all_states() for node in state.nodes()
              if isinstance(node, nd.MapEntry) for conn in node.out_connectors
              if len([e for e in state.out_edges(node) if e.src_conn == conn]) > 1]
    assert fanned, 'fixture no longer produces a fanned-out map connector'

    renamed = PrivatizeArrays().apply_pass(sdfg, {})['ArrayFission']
    assert len(renamed['tmp']) >= 2
    sdfg.validate()


def test_fanout_reads_value_preserving():
    rng = np.random.default_rng(17)
    A = rng.random((N, M))
    want1, want2 = np.zeros((N, M)), np.zeros((N, M))
    ref = A * 2.0
    want1[1:N - 1, 1:M - 1] = ref[0:N - 2, 1:M - 1] + ref[2:N, 1:M - 1]
    ref = A * 3.0
    want2[1:N - 1, 1:M - 1] = ref[1:N - 1, 0:M - 2] + ref[1:N - 1, 2:M]

    sdfg = fanout_reads.to_sdfg(simplify=True)
    sdfg.name = 'array_fission_fanout_value'
    PrivatizeArrays().apply_pass(sdfg, {})
    got1, got2 = np.zeros((N, M)), np.zeros((N, M))
    sdfg.compile()(A=A, out1=got1, out2=got2)
    assert np.allclose(got1, want1, rtol=1e-12, atol=1e-12)
    assert np.allclose(got2, want2, rtol=1e-12, atol=1e-12)


def test_partial_overwrite_is_refused():
    """Negative: a partially written array is never reported, so nothing is versioned."""
    sdfg = partial_overwrite.to_sdfg(simplify=True)
    assert 'tmp' not in fully_overwritten_arrays(sdfg)

    before = sdfg.to_json()
    renamed = PrivatizeArrays().apply_pass(sdfg, {})['ArrayFission']
    assert not renamed, f'partially overwritten array was versioned: {dict(renamed)}'
    # A pass that does not apply must not mutate the SDFG.
    assert sdfg.to_json() == before
    assert loop_count(sdfg) == 2
    assert sdfg.apply_transformations_repeated([LoopToMap]) == 0


def test_partial_overwrite_value_preserving():
    rng = np.random.default_rng(11)
    A = rng.random((N, M))
    want1, want2 = np.zeros(N), np.zeros(N)
    ref = np.zeros(M)
    for i in range(N):
        ref[0:M // 2] = A[i, 0:M // 2] * 2.0
        want1[i] = ref[0] + ref[M - 1]
    for i in range(N):
        ref[0:M // 2] = A[i, 0:M // 2] * 3.0
        want2[i] = ref[0] + ref[M - 1]

    sdfg = partial_overwrite.to_sdfg(simplify=True)
    sdfg.name = 'array_fission_partial_value'
    PrivatizeArrays().apply_pass(sdfg, {})
    got1, got2 = np.zeros(N), np.zeros(N)
    sdfg.compile()(A=A, out1=got1, out2=got2)
    assert np.allclose(got1, want1, rtol=1e-12, atol=1e-12)
    assert np.allclose(got2, want2, rtol=1e-12, atol=1e-12)


def test_conditional_overwrite_is_refused():
    """Negative: every write covers the array, but none is guaranteed to run, so ``tmp`` is carried."""
    sdfg = conditional_overwrite.to_sdfg(simplify=True)
    # The analysis does report it -- every write is a full write -- and the must-def gate in the
    # pass is what refuses: a non-exhaustive ``if`` establishes no definition.
    assert 'tmp' in fully_overwritten_arrays(sdfg)

    before = sdfg.to_json()
    renamed = PrivatizeArrays().apply_pass(sdfg, {})['ArrayFission']
    assert not renamed, f'conditionally written array was versioned: {dict(renamed)}'
    assert sdfg.to_json() == before
    assert loop_count(sdfg) == 1
    assert sdfg.apply_transformations_repeated([LoopToMap]) == 0


def test_conditional_overwrite_value_preserving():
    rng = np.random.default_rng(13)
    A = rng.random((N, M))
    want = np.zeros(N)
    ref = np.zeros(M)
    for i in range(N):
        if A[i, 0] > 0.5:
            ref[:] = A[i] * 2.0
        want[i] = ref[0] + ref[M - 1]

    sdfg = conditional_overwrite.to_sdfg(simplify=True)
    sdfg.name = 'array_fission_conditional_value'
    PrivatizeArrays().apply_pass(sdfg, {})
    got = np.zeros(N)
    sdfg.compile()(A=A, out1=got)
    assert np.allclose(got, want, rtol=1e-12, atol=1e-12)


def carried_reference(A: np.ndarray) -> np.ndarray:
    """Numpy evaluation of ``carried_between_loops``."""
    tmp = np.ndarray((N, M), dtype=np.float64)
    out = np.zeros((N, M))
    for _ in range(1):
        tmp[:] = A * 2.0
    for _ in range(3):
        out[:] = tmp * 3.0
        tmp[:] = out + 1.0
    return out


def test_carried_between_loops_is_refused():
    """Negative: a value handed from one loop to the next may not be privatized per loop.

    The undominated (``None``) write scope is ONE equivalence class of accesses, not a bag of
    independent per-loop temporaries. Here it holds every access of ``tmp``: the producing write in
    the first loop, and the read plus the redefinition in the second. The first loop has no
    upward-exposed use of ``tmp`` on its own, so privatizing it in isolation looks legal and
    validates -- while the second loop is left reading a container nothing writes any more. Nothing
    structural flags it; only the value is wrong.

    Two independent things now stop it. The analysis attributes every access to the producer's
    write (both trip counts are concrete), leaving a single root and so nothing to split; and
    ``_carrier_free`` would refuse the ``None``-scope split anyway, because the SECOND loop reads
    ``tmp`` before defining it. The assertion below pins the first, which is the root fix.
    """
    sdfg = carried_between_loops.to_sdfg(simplify=True)
    # The full-overwrite proof holds -- every write covers all of ``tmp`` -- so the refusal cannot
    # come from that proof.
    assert 'tmp' in fully_overwritten_arrays(sdfg)
    scopes = Pipeline([ArrayWriteShadowScopes()]).apply_pass(sdfg, {})['ArrayWriteShadowScopes'][0]['tmp']
    assert len([w for w in scopes if w is not None]) == 1, f'the carried chain must be one scope, got {scopes}'
    assert None not in scopes, 'no access of a fully attributed chain may stay undominated'
    assert loop_count(sdfg) == 2
    assert all(isinstance(block, LoopRegion) for block in sdfg.nodes()), 'fixture lost the vadv shape'

    before = sdfg.to_json()
    renamed = PrivatizeArrays().apply_pass(sdfg, {})['ArrayFission']
    assert not renamed, f'a loop-carried array was versioned: {dict(renamed)}'
    # A pass that does not apply must not mutate the SDFG.
    assert sdfg.to_json() == before


def test_carried_between_loops_value_preserving():
    rng = np.random.default_rng(3)
    A = rng.random((N, M))
    want = carried_reference(A)

    sdfg = carried_between_loops.to_sdfg(simplify=True)
    sdfg.name = 'array_fission_carried_value'
    PrivatizeArrays().apply_pass(sdfg, {})
    got = np.zeros((N, M))
    sdfg.compile()(A=A, out=got)
    assert np.allclose(got, want, rtol=1e-12, atol=1e-12)


def test_region_write_is_a_dominating_write():
    """Positive: a write inside a provably nonempty ``LoopRegion`` roots the reads after the loop.

    Accepting only an ``SDFGState`` as a dominating write block leaves this shape -- the npbench
    ``vadv`` shape -- with no root at all, so every access lands in the undominated (``None``)
    scope and the pass has nothing to version.
    """
    sdfg = region_write_then_consumers.to_sdfg(simplify=True)
    assert 'tmp' in fully_overwritten_arrays(sdfg)
    assert all(isinstance(block, LoopRegion) for block in sdfg.nodes()), 'fixture lost the vadv shape'

    scopes = Pipeline([ArrayWriteShadowScopes()]).apply_pass(sdfg, {})['ArrayWriteShadowScopes'][0]['tmp']
    assert len([w for w in scopes if w is not None]) == 2, f'expected one root per producer loop, got {scopes}'

    renamed = PrivatizeArrays().apply_pass(sdfg, {})['ArrayFission']
    sdfg.validate()
    assert len(renamed['tmp']) >= 2, f'expected one container per dominating write, got {dict(renamed)}'


def test_region_write_value_preserving():
    rng = np.random.default_rng(23)
    A = rng.random((N, M))
    want1 = np.full(N, A[0, 0] * 2.0 + A[0, M - 1] * 2.0)
    want2 = np.full(N, A[0, 0] * 3.0 + A[0, M - 1] * 3.0)

    sdfg = region_write_then_consumers.to_sdfg(simplify=True)
    sdfg.name = 'array_fission_region_write_value'
    PrivatizeArrays().apply_pass(sdfg, {})
    got1, got2 = np.zeros(N), np.zeros(N)
    sdfg.compile()(A=A, out1=got1, out2=got2)
    assert np.allclose(got1, want1, rtol=1e-12, atol=1e-12)
    assert np.allclose(got2, want2, rtol=1e-12, atol=1e-12)


def test_symbolic_trip_count_producer_is_refused():
    """Refusal: ``for _ in range(K)`` may run zero times, so its write defines nothing after it.

    The nonnegative-symbol assumption gives ``K >= 0``, not ``K >= 1``. Were the write accepted
    anyway, the consumer loop would be versioned onto a container that is never written when
    ``K == 0`` -- the read would silently see the zero-initialised original instead.
    """
    sdfg = symbolic_producer.to_sdfg(simplify=True)
    assert 'tmp' in fully_overwritten_arrays(sdfg)

    scopes = Pipeline([ArrayWriteShadowScopes()]).apply_pass(sdfg, {})['ArrayWriteShadowScopes'][0]['tmp']
    roots = [w for w in scopes if w is not None]
    # The ``np.zeros`` initialiser is a top-level state and a perfectly good dominating write; the
    # point is that the loop body's write must NOT displace it as the root for the consumer, which
    # is what a wrongly accepted zero-trip loop would do (and would then split into two versions).
    assert len(roots) == 1, f'expected only the initialiser to root tmp, got {scopes}'
    assert not isinstance(roots[0][0].parent_graph, LoopRegion), 'a zero-trip loop was accepted as a must-def'

    before = sdfg.to_json()
    renamed = PrivatizeArrays().apply_pass(sdfg, {})['ArrayFission']
    assert not renamed, f'an array behind a symbolic trip count was versioned: {dict(renamed)}'
    # A pass that does not apply must not mutate the SDFG.
    assert sdfg.to_json() == before


@pytest.mark.parametrize('k', (0, 3))
def test_symbolic_trip_count_producer_value_preserving(k):
    rng = np.random.default_rng(29)
    A = rng.random((N, M))
    ref = np.zeros(M)
    for _ in range(k):
        ref[:] = A[0] * 2.0
    want = np.full(N, ref[0] + ref[M - 1])

    sdfg = symbolic_producer.to_sdfg(simplify=True)
    sdfg.name = f'array_fission_symbolic_producer_value_{k}'
    PrivatizeArrays().apply_pass(sdfg, {})
    got = np.zeros(N)
    sdfg.compile()(A=A, out1=got, K=k)
    assert np.allclose(got, want, rtol=1e-12, atol=1e-12)


def test_length_one_arrays_are_handled_as_scalars():
    """A size-1 array is a SCALAR for fission purposes, and belongs to exactly one path.

    ``dace.float64[1]``, ``[1, 1]`` and a true ``Scalar`` are all containers where any write is
    a full write, so the dominance-only shadow analysis is already exact for them and the
    array path's full-extent proof would be redundant work at best. The two ``accepts``
    predicates must therefore PARTITION transients on ``total_size == 1`` -- a container
    claimed by both would be versioned twice under two different analyses, and one claimed by
    neither would silently stop being privatized at all.
    """
    scalar_side, array_side = ScalarFission(), ArrayFission()
    size_one = (
        dt.Scalar(dace.float64, transient=True),
        dt.Array(dace.float64, (1, ), transient=True),
        dt.Array(dace.float64, (1, 1), transient=True),
    )
    for desc in size_one:
        assert scalar_side.accepts(desc), f'size-1 {type(desc).__name__} must take the scalar path'
        assert not array_side.accepts(desc), f'size-1 {type(desc).__name__} must not take the array path'

    larger = dt.Array(dace.float64, (8, ), transient=True)
    assert array_side.accepts(larger)
    assert not scalar_side.accepts(larger)


def test_length_one_arrays_are_not_array_fission_candidates():
    """The analysis side of the same partition: a fully-overwritten size-1 array is not reported.

    ``fully_overwritten_arrays`` would prove full coverage for it trivially; excluding it keeps
    the proof obligation where it belongs and stops the array pass from shadowing the scalar one.
    """
    sdfg = dace.SDFG('len_one_not_array_candidate')
    sdfg.add_array('inp', (4, ), dace.float64)
    sdfg.add_array('out', (4, ), dace.float64)
    sdfg.add_array('one', (1, ), dace.float64, transient=True)
    state = sdfg.add_state()
    src, tmp, dst = state.add_access('inp'), state.add_access('one'), state.add_access('out')
    state.add_nedge(src, tmp, dace.Memlet('inp[0:1] -> [0:1]'))
    state.add_nedge(tmp, dst, dace.Memlet('one[0:1] -> [0:1]'))
    sdfg.validate()
    assert 'one' not in fully_overwritten_arrays(sdfg)


def test_wcr_accumulator_is_refused():
    """Negative: an accumulator whose value must survive the loop is a read-modify-write, not an
    overwrite, and privatizing it would drop the accumulation."""
    sdfg = full_overwrite.to_sdfg(simplify=True)
    accum = copy.deepcopy(sdfg)
    for state in accum.all_states():
        for node in state.data_nodes():
            if node.data != 'tmp':
                continue
            for edge in state.in_edges(node):
                if not edge.data.is_empty():
                    edge.data.wcr = 'lambda a, b: a + b'
    assert 'tmp' not in fully_overwritten_arrays(accum)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
