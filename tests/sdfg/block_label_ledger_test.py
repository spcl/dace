# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
""" Tests that a region's block-name ledger stays a superset of its live labels.

``_ensure_unique_block_name`` answers from ``_labels`` alone, so every way a name becomes live in a
region -- ``add_node``, a rename in place, a branch appended to a ``ConditionalBlock``, a block
arriving from JSON -- has to record it there. Each case below issues a name AFTER the event and
checks it did not land on top of a block that is still in the region; a ledger that forgot the event
reissues the colliding name, which ``validate`` refuses as "multiple blocks with the same name".
"""

import dace
from dace.properties import CodeBlock
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion, SDFGState


def chain(sdfg: dace.SDFG, *blocks) -> None:
    for src, dst in zip(blocks, blocks[1:]):
        sdfg.add_edge(src, dst, dace.InterstateEdge())


def live_labels(region) -> list:
    return [block.label for block in region.nodes()]


def test_added_states_never_collide():
    sdfg = dace.SDFG('added')
    chain(sdfg, *[sdfg.add_state('s') for _ in range(4)])

    assert live_labels(sdfg) == ['s', 's_0', 's_1', 's_2']
    sdfg.validate()


def test_rename_in_place_is_recorded():
    sdfg = dace.SDFG('renamed')
    first, second = sdfg.add_state('a'), sdfg.add_state('b')
    first.label = 'c'
    chain(sdfg, first, second, sdfg.add_state('c'))

    assert live_labels(sdfg) == ['c', 'b', 'c_0']
    sdfg.validate()


def test_removed_name_is_not_reissued():
    sdfg = dace.SDFG('removed')
    sdfg.remove_node(sdfg.add_state('a'))

    # The ledger deliberately keeps a removed name: an inlined reference may still point at it.
    assert sdfg.add_state('a').label == 'a_0'


def test_externally_built_block_registers_its_label():
    sdfg = dace.SDFG('external')
    hand_built = SDFGState('hand_built')
    sdfg.add_node(hand_built)
    chain(sdfg, hand_built, sdfg.add_state('hand_built'))

    assert live_labels(sdfg) == ['hand_built', 'hand_built_0']
    sdfg.validate()


def test_branch_label_is_recorded_on_the_conditional():
    sdfg = dace.SDFG('branches')
    conditional = ConditionalBlock('cond')
    sdfg.add_node(conditional, is_start_block=True)
    conditional.add_branch(CodeBlock('1'), ControlFlowRegion('arm', sdfg=sdfg))

    assert conditional._ensure_unique_block_name('arm') == 'arm_0'


def test_ledger_survives_a_json_round_trip():
    sdfg = dace.SDFG('round_trip')
    chain(sdfg, sdfg.add_state('a'), sdfg.add_state('a'))

    restored = dace.SDFG.from_json(sdfg.to_json())
    chain(restored, restored.node(1), restored.add_state('a'))

    assert live_labels(restored) == ['a', 'a_0', 'a_1']
    restored.validate()


if __name__ == '__main__':
    test_added_states_never_collide()
    test_rename_in_place_is_recorded()
    test_removed_name_is_not_reissued()
    test_externally_built_block_registers_its_label()
    test_branch_label_is_recorded_on_the_conditional()
    test_ledger_survives_a_json_round_trip()
