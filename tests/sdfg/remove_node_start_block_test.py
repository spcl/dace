# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Removing a block from a region whose stored start-block index is out of range (as deserialization can leave it)
drops the stale index instead of raising, and the start block falls back to the region's source block."""

import dace


def test_remove_node_with_a_stale_start_block_index():
    sdfg = dace.SDFG("stale_start_block")
    first = sdfg.add_state("first", is_start_block=True)
    second = sdfg.add_state_after(first, "second")
    third = sdfg.add_state_after(second, "third")
    sdfg._start_block = 7

    sdfg.remove_node(third)

    assert sdfg.start_block is first
    sdfg.validate()
