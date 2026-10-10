# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``node_id`` answered from the per-serialization index matches the linear scan."""

import pytest

import dace
from dace.sdfg.graph import NodeNotFoundError, frozen_node_ids


def chain(length: int) -> dace.SDFG:
    sdfg = dace.SDFG("chain")
    sdfg.add_array("A", [length + 1], dace.float64)
    state = sdfg.add_state()
    previous = state.add_access("A")
    for i in range(length):
        tasklet = state.add_tasklet(f"t{i}", {"x"}, {"y"}, "y = x + 1")
        state.add_edge(previous, None, tasklet, "x", dace.Memlet(f"A[{i}]"))
        previous = state.add_access("A")
        state.add_edge(tasklet, "y", previous, None, dace.Memlet(f"A[{i + 1}]"))
    return sdfg


def test_frozen_node_ids_match_the_scan():
    state = chain(5).start_block
    scanned = [state.node_id(n) for n in state.nodes()]
    with frozen_node_ids():
        assert [state.node_id(n) for n in state.nodes()] == scanned


def test_a_foreign_node_is_not_found_while_frozen():
    state = chain(1).start_block
    foreign = chain(1).start_block.nodes()[0]
    with frozen_node_ids(), pytest.raises(NodeNotFoundError):
        state.node_id(foreign)


if __name__ == "__main__":
    test_frozen_node_ids_match_the_scan()
    test_a_foreign_node_is_not_found_while_frozen()
