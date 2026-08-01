# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for ``RematerializeDerivedTemporaries``: it must fire on the fusion artifact and refuse otherwise."""
import numpy as np
import pytest

import dace
from dace.sdfg import nodes
from dace.transformation.dataflow.map_fusion_horizontal import MapFusionHorizontal
from dace.transformation.dataflow.map_fusion_vertical import MapFusionVertical
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
