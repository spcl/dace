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
def loop_sized_by_an_assignment(a: dace.float64[N], Nt: dace.int64, out: dace.float64[N]):
    b = np.empty(Nt + 1, dace.float64)  # the size is promoted onto an interstate edge
    for i in range(N):
        b[i] = a[i] * 2.0
    for i in range(N):
        out[i] = b[i]


@dace.program
def conditional_sized_by_an_assignment(a: dace.float64[N], Nt: dace.int64, out: dace.float64[N]):
    b = np.empty(Nt + 1, dace.float64)  # promoted before the branch, so the assignment enters it
    if Nt > 2:
        for i in range(N):
            b[i] = a[i] * 2.0
        for i in range(N):
            out[i] = b[i]
    else:
        for i in range(N):
            out[i] = 0.0


@dace.program
def calls_loop_sized_by_an_assignment(a: dace.float64[N], Nt: dace.int64, out: dace.float64[N]):
    loop_sized_by_an_assignment(a, Nt, out)


def regions_of(sdfg):
    for cfg in sdfg.all_control_flow_regions(recursive=True):
        if isinstance(cfg, ConditionalBlock):
            continue
        for node in cfg.nodes():
            if isinstance(node, AbstractControlFlowRegion):
                yield cfg, node


def bracketed(cfg, region):
    return (all(isinstance(e.src, dace.SDFGState) for e in cfg.in_edges(region))
            and all(isinstance(e.dst, dace.SDFGState) for e in cfg.out_edges(region)))


def sizing_regions(sdfg):
    """The regions the pass must bracket: their incoming edges assign a symbol a transient is sized by.

    A nested SDFG declares its own transients, so the sizes are read from the SDFG owning the region.
    """
    for cfg, region in regions_of(sdfg):
        sized_by = {str(s) for desc in cfg.sdfg.arrays.values() if desc.transient for s in desc.free_symbols}
        if {name for e in cfg.in_edges(region) for name in e.data.assignments} & sized_by:
            yield cfg, region


def test_a_region_whose_assignment_sizes_a_transient_is_bracketed():
    sdfg = loop_sized_by_an_assignment.to_sdfg(simplify=True)
    targets = list(sizing_regions(sdfg))
    assert targets, 'the fixture must size a transient from an interstate assignment'

    assert RegionBoundaryStates().apply_pass(sdfg, {}) > 0
    for cfg, region in targets:
        assert bracketed(cfg, region)
    sdfg.validate()


def test_a_conditional_sized_by_an_assignment_is_bracketed():
    """A ConditionalBlock is a region too: the size assignment can arrive on its incoming edge.

    Its branches are entered without an inter-state edge, so they need no boundary of their own --
    but the block itself does, and skipping it would allocate before the size is defined.
    """
    sdfg = conditional_sized_by_an_assignment.to_sdfg(simplify=True)
    conditionals = [(cfg, n) for cfg, n in regions_of(sdfg) if isinstance(n, ConditionalBlock)]
    assert conditionals, 'the fixture must keep a conditional block'
    assert any((cfg, n) in list(sizing_regions(sdfg)) for cfg, n in conditionals)

    assert RegionBoundaryStates().apply_pass(sdfg, {}) > 0
    for cfg, block in conditionals:
        assert bracketed(cfg, block)
    sdfg.validate()

    n, nt = 6, 9
    a = np.arange(n, dtype=np.float64)
    out = np.zeros(n)
    sdfg(a=a, Nt=np.int64(nt), out=out, N=n)
    assert np.allclose(out, a * 2.0)


def test_a_region_that_needs_no_boundary_is_left_alone():
    """Bracketing unconditionally tripled the state count of a program that sizes nothing this way."""
    sdfg = loop_and_branch.to_sdfg(simplify=True)
    assert any(True for _ in regions_of(sdfg))  # guard against a vacuous check
    assert not list(sizing_regions(sdfg))
    before = sum(1 for _ in sdfg.all_states())

    assert RegionBoundaryStates().apply_pass(sdfg, {}) is None
    assert sum(1 for _ in sdfg.all_states()) == before
    sdfg.validate()


def test_regions_inside_a_nested_sdfg_are_bracketed():
    """The pass descends into nested SDFGs, where the same allocation bug applies."""
    sdfg = calls_loop_sized_by_an_assignment.to_sdfg(simplify=False)
    nested = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.NestedSDFG)]
    assert nested, 'the fixture must keep a nested SDFG'
    # Only simplify the callee: that is what moves the size assignment onto the region's incoming
    # edge, and simplifying the caller too would inline the nested SDFG away.
    for node in nested:
        node.sdfg.simplify()
    inner = [(cfg, r) for cfg, r in sizing_regions(sdfg) if cfg.sdfg is not sdfg]
    assert inner, 'the nested SDFG must contain a region sized by an assignment'

    RegionBoundaryStates().apply_pass(sdfg, {})
    for cfg, region in inner:
        assert bracketed(cfg, region)
    sdfg.validate()


def test_leading_region_keeps_start_block():
    """Bracketing a region that starts its parent must hand the start block over, not orphan it.

    Only a cyclic CFG reaches this: the pass brackets a region whose incoming edge assigns a size
    symbol, and in an acyclic graph a start block has no incoming edge. Hand-built for that reason.
    """
    sdfg = dace.SDFG('cyclic_start')
    sdfg.add_array('out', [1], dace.float64)
    sdfg.add_symbol('K', dace.int64)
    sdfg.add_transient('b', ['K'], dace.float64)

    loop = dace.sdfg.state.LoopRegion('loop', 'i < 4', 'i', 'i = 0', 'i = i + 1')
    sdfg.add_node(loop, is_start_block=True)
    body = loop.add_state('body', is_start_block=True)
    tasklet = body.add_tasklet('one', {}, {'o'}, 'o = 1.0')
    body.add_edge(tasklet, 'o', body.add_write('out'), None, dace.Memlet('out[0]'))

    tail = sdfg.add_state('tail')
    sdfg.add_edge(loop, tail, dace.InterstateEdge(condition='0'))
    sdfg.add_edge(tail, loop, dace.InterstateEdge(assignments={'K': '4'}))

    assert RegionBoundaryStates().apply_pass(sdfg, {}) == 2
    assert isinstance(sdfg.start_block, dace.SDFGState)
    assert sdfg.start_block is not loop
    sdfg.validate()


def test_result_is_unchanged():
    """The pass only inserts empty states, so it must not alter what the program computes."""
    n, nt = 16, 20
    a = np.random.default_rng(0).random(n) - 0.5

    sdfg = loop_sized_by_an_assignment.to_sdfg(simplify=True)
    RegionBoundaryStates().apply_pass(sdfg, {})
    out = np.zeros(n)
    sdfg(a=a, Nt=np.int64(nt), out=out, N=n)
    assert np.allclose(out, a * 2.0)


if __name__ == '__main__':
    test_a_region_whose_assignment_sizes_a_transient_is_bracketed()
    test_a_conditional_sized_by_an_assignment_is_bracketed()
    test_a_region_that_needs_no_boundary_is_left_alone()
    test_regions_inside_a_nested_sdfg_are_bracketed()
    test_leading_region_keeps_start_block()
    test_result_is_unchanged()
