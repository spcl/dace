# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for ``RematerializeDerivedTemporaries``: it must fire on the fusion artifact and refuse otherwise."""
import numpy as np
import pytest

import dace
from dace.sdfg import nodes
from dace.transformation.dataflow.map_fusion_horizontal import MapFusionHorizontal
from dace.transformation.dataflow.map_fusion_vertical import MapFusionVertical
from dace.transformation.interstate import LoopToMap
from dace.transformation.passes.rematerialize_derived_temporaries import RematerializeDerivedTemporaries

M = dace.symbol('M')
N = dace.symbol('N')


@dace.program
def heat1d(A: dace.float64[M], B: dace.float64[M]):
    B[1:-1] = A[1:-1] + 0.5 * (A[:-2] - 2.0 * A[1:-1] + A[2:])
    A[1:-1] = B[1:-1] + 0.5 * (B[:-2] - 2.0 * B[1:-1] + B[2:])


@dace.program
def heat2d(A: dace.float64[N, N], B: dace.float64[N, N]):
    B[1:-1, 1:-1] = A[1:-1, 1:-1] + 0.5 * (A[:-2, 1:-1] - 2.0 * A[1:-1, 1:-1] + A[2:, 1:-1])
    A[1:-1, 1:-1] = B[1:-1, 1:-1] + 0.5 * (B[:-2, 1:-1] - 2.0 * B[1:-1, 1:-1] + B[2:, 1:-1])


@dace.program
def deep_chain(A: dace.float64[M], B: dace.float64[M], C: dace.float64[M]):
    B[1:-1] = A[1:-1] + 0.5 * (A[:-2] - 2.0 * A[1:-1] + A[2:])
    C[1:-1] = B[1:-1] + 0.5 * (B[:-2] - (B[1:-1] * B[1:-1] * 3.0 + 1.0) + B[2:])


@dace.program
def two_reads(A: dace.float64[M], B: dace.float64[M], C: dace.float64[M]):
    B[1:-1] = A[1:-1] + 0.5 * (A[:-2] - 2.0 * A[1:-1] + A[2:])
    C[1:-1] = B[1:-1] + 0.5 * (B[:-2] - 2.0 * B[1:-1] + B[2:]) + 2.0 * B[1:-1]


@dace.program
def consumer_reads_wrong_index(A: dace.float64[M], B: dace.float64[M], C: dace.float64[M]):
    # The temporary holds ``2 * B[i + 1]`` but the consumer only ever reads ``B[i]``.
    B[1:-1] = A[1:-1] + 0.5 * (A[:-2] - 2.0 * A[1:-1] + A[2:])
    C[1:-1] = 2.0 * B[1:-1] + B[:-2]


@dace.program
def consumer_reads_far_index(A: dace.float64[M], B: dace.float64[M], C: dace.float64[M]):
    # Same, on the other side: the only read is ``B[i + 2]``.
    B[1:-1] = A[1:-1] * 3.0
    C[1:-1] = 2.0 * B[1:-1] + B[2:]


@dace.program
def consumer_writes_container(A: dace.float64[M], B: dace.float64[M]):
    # The consumer writes the container the rematerialization would read back.
    B[1:-1] = A[1:-1] + 0.5 * (A[:-2] - 2.0 * A[1:-1] + A[2:])
    B[1:-1] = 2.0 * B[1:-1] + B[2:]


def fused(program) -> dace.SDFG:
    """The post-fusion shape the pass is a cleanup for."""
    sdfg = program.to_sdfg(simplify=True)
    sdfg.apply_transformations_repeated([MapFusionVertical, MapFusionHorizontal], validate_all=False)
    sdfg.simplify()
    return sdfg


def arrays(sdfg: dace.SDFG):
    return {name for name, desc in sdfg.arrays.items() if desc.transient and desc.total_size != 1}


def candidate(sdfg: dace.SDFG):
    """The transient the pass would act on: written by a map exit, read by a map entry."""
    for state in sdfg.states():
        for node in state.data_nodes():
            if node.data not in arrays(sdfg) or state.in_degree(node) != 1:
                continue
            if isinstance(state.in_edges(node)[0].src, nodes.MapExit) and any(
                    isinstance(e.dst, nodes.MapEntry) for e in state.out_edges(node)):
                return state, node
    raise AssertionError('no candidate temporary in the fused SDFG')


def check_values(program, size: int, dims: int = 1):
    """Run the fused SDFG with and without the pass on identical inputs; require bit equality."""
    rng = np.random.default_rng(4711)
    shape = (size, ) * dims
    base = {name: rng.random(shape) for name in program.f.__annotations__ if name != 'return'}
    symbols = {'M' if dims == 1 else 'N': size}

    outputs = []
    for apply_pass in (False, True):
        sdfg = fused(program)
        if apply_pass:
            assert RematerializeDerivedTemporaries().apply_pass(sdfg, {})
        sdfg.name = program.name + ('_remat' if apply_pass else '_ref')
        args = {name: value.copy() for name, value in base.items()}
        sdfg.compile()(**args, **symbols)
        outputs.append(args)

    for name in base:
        assert np.array_equal(outputs[0][name].view(np.uint64), outputs[1][name].view(np.uint64)), name


def test_removes_stencil_temporary():
    sdfg = fused(heat1d)
    before = arrays(sdfg)
    assert RematerializeDerivedTemporaries().apply_pass(sdfg, {}) == 1
    assert len(arrays(sdfg)) == len(before) - 1
    sdfg.validate()
    # The recomputation reads the container at the shifted index the inversion demands, not at the
    # index the consumer happened to read the temporary at.
    reads = {
        str(e.data.subset)
        for state in sdfg.states()
        for e in state.edges() if e.data.data == 'B' and isinstance(e.src, nodes.MapEntry)
    }
    assert '__i0 + 1' in reads


def test_removes_stencil_temporary_bit_exactly():
    check_values(heat1d, 128)


def test_multidimensional():
    sdfg = fused(heat2d)
    assert RematerializeDerivedTemporaries().apply_pass(sdfg, {}) == 1
    sdfg.validate()
    check_values(heat2d, 32, dims=2)


def test_multi_tasklet_chain():
    sdfg = fused(deep_chain)
    assert RematerializeDerivedTemporaries().apply_pass(sdfg, {}) == 1
    sdfg.validate()
    check_values(deep_chain, 128)


def test_two_reads_share_one_clone():
    sdfg = fused(two_reads)
    assert RematerializeDerivedTemporaries().apply_pass(sdfg, {})
    sdfg.validate()
    check_values(two_reads, 128)


@pytest.mark.parametrize('program', [consumer_reads_wrong_index, consumer_reads_far_index])
def test_refuses_when_the_needed_index_is_not_already_read(program):
    """The user-facing condition is not "the container reaches the consumer" but "that ELEMENT does"."""
    sdfg = fused(program)
    before = arrays(sdfg)
    assert RematerializeDerivedTemporaries().apply_pass(sdfg, {}) is None
    assert arrays(sdfg) == before


def test_refuses_when_the_consumer_writes_the_container():
    sdfg = fused(consumer_writes_container)
    before = arrays(sdfg)
    assert RematerializeDerivedTemporaries().apply_pass(sdfg, {}) is None
    assert arrays(sdfg) == before


def test_refuses_a_temporary_with_a_second_reader():
    """A reader outside the consumer map keeps the temporary alive."""
    sdfg = fused(heat1d)
    state, node = candidate(sdfg)
    desc = sdfg.arrays[node.data]
    escape, _ = sdfg.add_array('escape', desc.shape, desc.dtype, transient=False)
    state.add_edge(node, None, state.add_access(escape), None, dace.Memlet(data=node.data, subset='0:M - 2'))
    before = arrays(sdfg)
    assert RematerializeDerivedTemporaries().apply_pass(sdfg, {}) is None
    assert arrays(sdfg) == before


def test_refuses_a_temporary_named_by_control_flow():
    """A container an interstate edge reads is live outside dataflow."""
    sdfg = fused(heat1d)
    state, node = candidate(sdfg)
    sdfg.add_state_after(state, assignments={'probe': '%s[0]' % node.data})
    before = arrays(sdfg)
    assert RematerializeDerivedTemporaries().apply_pass(sdfg, {}) is None
    assert arrays(sdfg) == before


# ------------------------------------------------------------------ heat3d, on the real transformation path


@dace.program
def heat3d(TSTEPS: dace.int64, A: dace.float64[N, N, N], B: dace.float64[N, N, N]):
    """The npbench/polybench formulation, verbatim -- see ``tests/corpus/polybench/stencils/heat_3d.py``."""
    for _ in range(1, TSTEPS):
        B[1:-1, 1:-1,
          1:-1] = (0.125 * (A[2:, 1:-1, 1:-1] - 2.0 * A[1:-1, 1:-1, 1:-1] + A[:-2, 1:-1, 1:-1]) + 0.125 *
                   (A[1:-1, 2:, 1:-1] - 2.0 * A[1:-1, 1:-1, 1:-1] + A[1:-1, :-2, 1:-1]) + 0.125 *
                   (A[1:-1, 1:-1, 2:] - 2.0 * A[1:-1, 1:-1, 1:-1] + A[1:-1, 1:-1, 0:-2]) + A[1:-1, 1:-1, 1:-1])
        A[1:-1, 1:-1,
          1:-1] = (0.125 * (B[2:, 1:-1, 1:-1] - 2.0 * B[1:-1, 1:-1, 1:-1] + B[:-2, 1:-1, 1:-1]) + 0.125 *
                   (B[1:-1, 2:, 1:-1] - 2.0 * B[1:-1, 1:-1, 1:-1] + B[1:-1, :-2, 1:-1]) + 0.125 *
                   (B[1:-1, 1:-1, 2:] - 2.0 * B[1:-1, 1:-1, 1:-1] + B[1:-1, 1:-1, 0:-2]) + B[1:-1, 1:-1, 1:-1])


def heat3d_numpy(tsteps: int, A: np.ndarray, B: np.ndarray) -> None:
    """Independent oracle: plain numpy, same operand order, updated in place."""
    for _ in range(1, tsteps):
        B[1:-1, 1:-1,
          1:-1] = (0.125 * (A[2:, 1:-1, 1:-1] - 2.0 * A[1:-1, 1:-1, 1:-1] + A[:-2, 1:-1, 1:-1]) + 0.125 *
                   (A[1:-1, 2:, 1:-1] - 2.0 * A[1:-1, 1:-1, 1:-1] + A[1:-1, :-2, 1:-1]) + 0.125 *
                   (A[1:-1, 1:-1, 2:] - 2.0 * A[1:-1, 1:-1, 1:-1] + A[1:-1, 1:-1, 0:-2]) + A[1:-1, 1:-1, 1:-1])
        A[1:-1, 1:-1,
          1:-1] = (0.125 * (B[2:, 1:-1, 1:-1] - 2.0 * B[1:-1, 1:-1, 1:-1] + B[:-2, 1:-1, 1:-1]) + 0.125 *
                   (B[1:-1, 2:, 1:-1] - 2.0 * B[1:-1, 1:-1, 1:-1] + B[1:-1, :-2, 1:-1]) + 0.125 *
                   (B[1:-1, 1:-1, 2:] - 2.0 * B[1:-1, 1:-1, 1:-1] + B[1:-1, 1:-1, 0:-2]) + B[1:-1, 1:-1, 1:-1])


def heat3d_after_loop_to_map_and_fusion() -> dace.SDFG:
    """heat3d driven down the pipeline's real path: LoopToMap, then vertical+horizontal map fusion.

    This is where the redundant transient is manufactured, so it is where the pass has to be exercised --
    a hand-built SDFG would not prove the shape ever occurs. LoopToMap is part of the path and runs here
    even though this formulation gives it nothing to lift (the slice assignments are already maps out of
    the frontend); fusion is what pulls the sub-expression up and strands it in a transient.
    """
    sdfg = heat3d.to_sdfg(simplify=True)
    sdfg.apply_transformations_repeated([LoopToMap], validate_all=False)
    sdfg.simplify()
    sdfg.apply_transformations_repeated([MapFusionVertical, MapFusionHorizontal], validate_all=False)
    sdfg.simplify()
    return sdfg


def test_heat3d_after_loop_to_map_and_fusion_drops_transients():
    sdfg = heat3d_after_loop_to_map_and_fusion()
    before = arrays(sdfg)
    # Guard against a vacuous test: fusion must actually have stranded the two temporaries.
    assert len(before) == 3, before
    assert RematerializeDerivedTemporaries().apply_pass(sdfg, {}) == 2
    assert len(arrays(sdfg)) == 1
    sdfg.validate()


def test_heat3d_after_loop_to_map_and_fusion_is_bit_exact():
    """Bit-exact against the same SDFG without the pass, and against an independent numpy oracle."""
    size, tsteps = 24, 5
    rng = np.random.default_rng(4711)
    base_a = rng.random((size, ) * 3)
    base_b = rng.random((size, ) * 3)

    results = []
    for apply_pass in (False, True):
        sdfg = heat3d_after_loop_to_map_and_fusion()
        if apply_pass:
            assert RematerializeDerivedTemporaries().apply_pass(sdfg, {}) == 2
        sdfg.name = 'heat3d_remat' if apply_pass else 'heat3d_fused_ref'
        a, b = base_a.copy(), base_b.copy()
        sdfg.compile()(TSTEPS=tsteps, A=a, B=b, N=size)
        results.append((a, b))

    oracle_a, oracle_b = base_a.copy(), base_b.copy()
    heat3d_numpy(tsteps, oracle_a, oracle_b)

    for got, want, name in ((results[1][0], results[0][0], 'A'), (results[1][1], results[0][1], 'B')):
        assert np.array_equal(got.view(np.uint64), want.view(np.uint64)), 'vs the same SDFG without the pass: ' + name
    for got, want, name in ((results[1][0], oracle_a, 'A'), (results[1][1], oracle_b, 'B')):
        assert np.array_equal(got.view(np.uint64), want.view(np.uint64)), 'vs numpy oracle: ' + name


if __name__ == '__main__':
    test_removes_stencil_temporary()
    test_removes_stencil_temporary_bit_exactly()
    test_multidimensional()
    test_multi_tasklet_chain()
    test_two_reads_share_one_clone()
    test_refuses_when_the_needed_index_is_not_already_read(consumer_reads_wrong_index)
    test_refuses_when_the_needed_index_is_not_already_read(consumer_reads_far_index)
    test_refuses_when_the_consumer_writes_the_container()
    test_refuses_a_temporary_with_a_second_reader()
    test_refuses_a_temporary_named_by_control_flow()
    test_heat3d_after_loop_to_map_and_fusion_drops_transients()
    test_heat3d_after_loop_to_map_and_fusion_is_bit_exact()
