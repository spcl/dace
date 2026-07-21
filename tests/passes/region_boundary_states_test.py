# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
""" Tests bracketing of control flow regions with states. """
import numpy as np

import dace
from dace.sdfg.state import AbstractControlFlowRegion, ConditionalBlock
from dace.transformation.passes.region_boundary_states import RegionBoundaryStates

N = dace.symbol('N')


@dace.program
def loop_and_branch(a: dace.float64[N], out: dace.float64[N]):
    for i in range(N):
        if a[i] > 0.0:
            out[i] = a[i]
        else:
            out[i] = -a[i]


@dace.program
def calls_loop_and_branch(a: dace.float64[N], out: dace.float64[N]):
    loop_and_branch(a, out)


def regions_of(sdfg):
    for cfg in sdfg.all_control_flow_regions(recursive=True):
        if isinstance(cfg, ConditionalBlock):
            continue
        for node in cfg.nodes():
            if isinstance(node, AbstractControlFlowRegion):
                yield cfg, node


def test_every_region_is_bracketed():
    sdfg = loop_and_branch.to_sdfg(simplify=True)
    assert any(True for _ in regions_of(sdfg))  # guard against a vacuous check

    assert RegionBoundaryStates().apply_pass(sdfg, {}) > 0
    for cfg, region in regions_of(sdfg):
        assert all(isinstance(e.src, dace.SDFGState) for e in cfg.in_edges(region))
        assert all(isinstance(e.dst, dace.SDFGState) for e in cfg.out_edges(region))
    sdfg.validate()


def test_regions_inside_a_nested_sdfg_are_bracketed():
    """The pass descends into nested SDFGs, where the same allocation bug applies."""
    sdfg = calls_loop_and_branch.to_sdfg(simplify=False)
    nested = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.NestedSDFG)]
    assert nested, 'the fixture must keep a nested SDFG'
    inner_regions = [(cfg, r) for cfg, r in regions_of(sdfg) if cfg.sdfg is not sdfg]
    assert inner_regions, 'the nested SDFG must contain a region'

    RegionBoundaryStates().apply_pass(sdfg, {})
    for cfg, region in inner_regions:
        assert all(isinstance(e.src, dace.SDFGState) for e in cfg.in_edges(region))
        assert all(isinstance(e.dst, dace.SDFGState) for e in cfg.out_edges(region))
    sdfg.validate()


def test_leading_region_keeps_start_block():
    """Bracketing a region that starts its parent must hand the start block over, not orphan it."""
    sdfg = dace.SDFG('leading_region')
    sdfg.add_array('out', [1], dace.float64)
    loop = dace.sdfg.state.LoopRegion('loop', 'i < 4', 'i', 'i = 0', 'i = i + 1')
    sdfg.add_node(loop, is_start_block=True)
    state = loop.add_state('body', is_start_block=True)
    tasklet = state.add_tasklet('one', {}, {'o'}, 'o = 1.0')
    state.add_edge(tasklet, 'o', state.add_write('out'), None, dace.Memlet('out[0]'))

    RegionBoundaryStates().apply_pass(sdfg, {})
    assert isinstance(sdfg.start_block, dace.SDFGState)
    assert sdfg.start_block is not loop
    sdfg.validate()

    out = np.zeros(1)
    sdfg(out=out)
    assert out[0] == 1.0


def test_result_is_unchanged():
    """The pass only inserts empty states, so it must not alter what the program computes."""
    a = np.random.default_rng(0).random(16) - 0.5
    expected = np.abs(a)

    sdfg = loop_and_branch.to_sdfg(simplify=True)
    RegionBoundaryStates().apply_pass(sdfg, {})
    out = np.zeros(16)
    sdfg(a=a, out=out, N=16)
    assert np.allclose(out, expected)


if __name__ == '__main__':
    test_every_region_is_bracketed()
    test_regions_inside_a_nested_sdfg_are_bracketed()
    test_leading_region_keeps_start_block()
    test_result_is_unchanged()
