# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import numpy as np
import pytest

import dace
from dace.sdfg import utils as sdutil
from dace.sdfg.analysis import cfg as cfg_analysis


def build_cfg(shape: str) -> tuple[dace.SDFG, list[dace.SDFGState]]:
    sdfg = dace.SDFG(f'dominators_{shape}')
    first = sdfg.add_state('s0', is_start_block=True)
    if shape == 'single':
        return sdfg, [first]
    if shape == 'chain':
        second = sdfg.add_state_after(first, 's1')
        third = sdfg.add_state_after(second, 's2')
        return sdfg, [first, second, third]
    left, right, merge = sdfg.add_state('s1'), sdfg.add_state('s2'), sdfg.add_state('s3')
    sdfg.add_edge(first, left, dace.InterstateEdge(condition='c'))
    sdfg.add_edge(first, right, dace.InterstateEdge(condition='not c'))
    sdfg.add_edge(left, merge, dace.InterstateEdge())
    sdfg.add_edge(right, merge, dace.InterstateEdge())
    return sdfg, [first, left, right, merge]


# Expected immediate dominator of each state (by index into the list returned by build_cfg).
IDOM_BY_SHAPE = {
    'single': [0],
    'chain': [0, 0, 1],
    'diamond': [0, 0, 0, 0],
}

# Expected immediate postdominator of each state.
IPOSTDOM_BY_SHAPE = {
    'single': [0],
    'chain': [1, 2, 2],
    'diamond': [3, 3, 3, 3],
}


@pytest.mark.parametrize('shape', IDOM_BY_SHAPE)
def test_the_start_block_is_its_own_immediate_dominator(shape):
    """networkx 3.6 dropped ``start: start`` from its result and every DaCe consumer indexes it."""
    sdfg, states = build_cfg(shape)
    idom = sdutil.immediate_dominators(sdfg.nx, sdfg.start_block)
    assert idom == {state: states[i] for state, i in zip(states, IDOM_BY_SHAPE[shape])}, idom


@pytest.mark.parametrize('shape', IPOSTDOM_BY_SHAPE)
def test_the_sink_is_its_own_immediate_postdominator(shape):
    sdfg, states = build_cfg(shape)
    ipostdom = sdutil.postdominators(sdfg)
    assert ipostdom == {state: states[i] for state, i in zip(states, IPOSTDOM_BY_SHAPE[shape])}, ipostdom


@pytest.mark.parametrize('shape', IDOM_BY_SHAPE)
def test_all_dominators_of_the_start_block_are_empty_and_cover_every_block(shape):
    sdfg, states = build_cfg(shape)
    alldoms = cfg_analysis.all_dominators(sdfg)
    assert alldoms[states[0]] == set()
    assert set(alldoms) == set(states)


@pytest.mark.parametrize('shape', IDOM_BY_SHAPE)
def test_block_parent_tree_roots_at_the_start_block(shape):
    sdfg, states = build_cfg(shape)
    parents = cfg_analysis.block_parent_tree(sdfg)
    assert parents[states[0]] is None, parents
    assert set(parents) == set(states)


@pytest.mark.parametrize('shape', IDOM_BY_SHAPE)
def test_control_flow_block_dominators_cover_every_block(shape):
    sdfg, states = build_cfg(shape)
    idom, ipostdom = {}, {}
    sdutil.get_control_flow_block_dominators(sdfg, idom=idom, ipostdom=ipostdom)
    assert idom == {state: states[i] for state, i in zip(states, IDOM_BY_SHAPE[shape])}, idom
    assert ipostdom == {state: states[i] for state, i in zip(states, IPOSTDOM_BY_SHAPE[shape])}, ipostdom


def test_code_is_generated_for_a_program_with_two_maps_over_a_transient():
    """Allocation lifetime analysis takes the dominator of every access; it raised KeyError on networkx 3.6."""
    N = dace.symbol('N')

    @dace.program
    def two_maps(A: dace.float64[N], B: dace.float64[N]):
        tmp = np.empty_like(A)
        for i in dace.map[0:N]:
            tmp[i] = A[i] * 2.0
        for i in dace.map[0:N]:
            B[i] = tmp[i] + 1.0

    sdfg = two_maps.to_sdfg()
    assert len(sdfg.generate_code()) > 0


if __name__ == '__main__':
    for shape in IDOM_BY_SHAPE:
        test_the_start_block_is_its_own_immediate_dominator(shape)
        test_the_sink_is_its_own_immediate_postdominator(shape)
        test_all_dominators_of_the_start_block_are_empty_and_cover_every_block(shape)
        test_block_parent_tree_roots_at_the_start_block(shape)
        test_control_flow_block_dominators_cover_every_block(shape)
    test_code_is_generated_for_a_program_with_two_maps_over_a_transient()
