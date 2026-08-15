# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``_start_block`` holds an INDEX into the node list, not the block itself.

Every reader has to agree on that, and the ones that disagreed were invisible until a graph
actually carried an explicit pin: ``SDFG.from_json`` restored the block object while ``to_json``
wrote the index, and ``state_fusion.is_start_block`` compared the field to a block, which an
integer never matches -- so a fusion that removed a region's entry decided the entry had not been
removed and left the region with no answer at all.

The index is derived state that any removal invalidates, so ``remove_node`` re-resolves it around
the removal; that is what the behavioural cases below pin.
"""
import dace
from dace.sdfg.state import ControlFlowRegion
import pytest


def _sdfg_with_explicit_start(n: int = 4) -> dace.SDFG:
    """``n`` chained states whose start block is set explicitly via ``is_start_block``."""
    sdfg = dace.SDFG(f'start_block_repr_{n}')
    states = [sdfg.add_state(f's{i}', is_start_block=(i == 0)) for i in range(n)]
    for a, b in zip(states, states[1:]):
        sdfg.add_edge(a, b, dace.InterstateEdge())
    return sdfg


def test_add_node_stores_an_index_not_the_block():
    sdfg = _sdfg_with_explicit_start()
    assert isinstance(sdfg._start_block, int)
    assert sdfg.node(sdfg._start_block) is sdfg.node(0)


def test_removing_an_earlier_node_keeps_the_start_block():
    """The index of every later node shifts down, so the pin has to be re-resolved by identity."""
    sdfg = _sdfg_with_explicit_start()
    start = sdfg.start_block
    victim = sdfg.node(2)
    sdfg.remove_node(victim)
    assert sdfg.start_block is start
    assert start in sdfg.nodes()


def test_reordering_the_nodes_moves_the_pin_with_the_block():
    """A permutation moves the pinned block without moving the index, so the pin would name
    whichever block landed in that slot -- and ``to_json`` writes the index."""
    sdfg = _sdfg_with_explicit_start()
    start = sdfg.start_block
    sdfg.reorder_nodes(list(reversed(sdfg.nodes())))
    assert sdfg._start_block == len(sdfg.nodes()) - 1
    assert sdfg.node(sdfg._start_block) is start
    assert sdfg.start_block is start


def test_removing_the_start_block_repins_to_the_new_entry():
    """Clearing the pin instead loses information ``to_json`` serializes: the getter falls back to
    ``source_nodes()`` but never writes ``_start_block`` back, so the same graph rewritten twice
    serializes differently."""
    sdfg = _sdfg_with_explicit_start()
    start = sdfg.start_block
    sdfg.remove_node(start)
    assert start not in sdfg.nodes()
    assert sdfg.node(sdfg._start_block) is sdfg.start_block
    assert sdfg.start_block is sdfg.source_nodes()[0]


def _ambiguous_sdfg() -> dace.SDFG:
    """Two source nodes, so ``start_block`` cannot infer and must consult ``_start_block``."""
    sdfg = dace.SDFG('start_block_repr_ambiguous')
    a = sdfg.add_state('a')
    b = sdfg.add_state('b')
    tail = sdfg.add_state('tail')
    sdfg.add_edge(a, tail, dace.InterstateEdge())
    sdfg.add_edge(b, tail, dace.InterstateEdge())
    return sdfg


def test_the_setter_takes_an_index_and_survives_an_earlier_removal():
    """Uses an ambiguous graph, because the getter short-circuits to ``source_nodes()`` whenever
    there is exactly one -- the explicit pin is only ever read when there is not."""
    sdfg = _ambiguous_sdfg()
    b = sdfg.node(1)
    sdfg.start_block = sdfg.node_id(b)
    assert isinstance(sdfg._start_block, int)
    assert sdfg.start_block is b

    sdfg.remove_node(sdfg.node(0))  # drop an EARLIER node: the raw index is now stale
    assert sdfg.start_block is b


def test_remove_node_on_a_region_with_an_explicit_start():
    """The shape the canonicalize cleanup band hits: a region inside an SDFG losing a child."""
    sdfg = dace.SDFG('start_block_repr_region')
    outer = sdfg.add_state('outer', is_start_block=True)
    region = ControlFlowRegion('inner', sdfg=sdfg)
    sdfg.add_node(region)
    sdfg.add_edge(outer, region, dace.InterstateEdge())
    inner = [region.add_state(f'r{i}', is_start_block=(i == 0)) for i in range(3)]
    for a, b in zip(inner, inner[1:]):
        region.add_edge(a, b, dace.InterstateEdge())

    start = region.start_block
    region.remove_node(inner[1])
    assert region.start_block is start
    sdfg.remove_node(region)
    assert sdfg.start_block is outer


def test_round_trip_through_json_keeps_the_representation():
    """``to_json`` writes the index; ``SDFG.from_json`` used to read it back as a block, which made
    every later read of the pin resolve a block through ``node()``."""
    sdfg = _sdfg_with_explicit_start()
    restored = dace.SDFG.from_json(sdfg.to_json())
    assert isinstance(restored._start_block, int)
    assert restored.start_block.label == sdfg.start_block.label
    restored.remove_node(restored.node(2))
    assert restored.start_block.label == sdfg.start_block.label


def test_a_pinned_ambiguous_region_round_trips():
    """The pin only matters when the entry is underivable, which is also the only shape that reads
    it back -- a single-source graph answers from ``source_nodes()`` either way."""
    sdfg = _ambiguous_sdfg()
    sdfg.start_block = sdfg.node_id(sdfg.node(1))
    restored = dace.SDFG.from_json(sdfg.to_json())
    assert isinstance(restored._start_block, int)
    assert restored.start_block.label == 'b'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
