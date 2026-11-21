# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Fusing away a no-op START block must not re-resolve pattern nodes after removal.

``BlockFusion.apply`` removes the empty first block and then re-pins the start block on the
surviving second block. Pattern-node handles resolve by NODE ID on every attribute access, and
removing a node shifts the ids of every node inserted after it -- so touching
``self.second_block`` AFTER ``remove_node`` raised ``NodeNotFoundError`` on exactly the shape
below (an empty start state followed by the real one, as ``StateFusionExtended`` leaves behind
on the fibonacci consume sample).
"""

import numpy as np
import pytest

import dace
from dace.transformation.interstate import BlockFusion
from dace.transformation.interstate.block_fusion import splice_empty_state


def build_chain() -> dace.SDFG:
    sdfg = dace.SDFG("noop_start_chain")
    sdfg.add_array("a", [4], dace.float64)
    empty = sdfg.add_state("empty_start", is_start_block=True)
    work = sdfg.add_state("work")
    tasklet = work.add_tasklet("w", {}, {"out"}, "out = 7.0")
    work.add_edge(tasklet, "out", work.add_write("a"), None, dace.Memlet("a[0]"))
    sdfg.add_edge(empty, work, dace.InterstateEdge())
    return sdfg


def test_fusing_away_the_noop_start_block():
    sdfg = build_chain()
    applied = sdfg.apply_transformations_repeated(BlockFusion, validate=True)
    assert applied == 1, "BlockFusion refused the empty start block"
    assert sdfg.number_of_nodes() == 1
    assert sdfg.start_block.label == "work", "start block must move to the surviving block"
    sdfg.validate()

    a = np.zeros(4)
    sdfg(a=a)
    assert a[0] == 7.0


def test_block_fusion_keeps_an_assignment_reading_what_the_block_writes():
    """``x = a[0]`` on the edge out of a state writing ``a`` cannot run ahead of the state."""
    sdfg = dace.SDFG("assignment_reads_block_write")
    sdfg.add_array("a", [1], dace.float64)
    start = sdfg.add_state("start", is_start_block=True)
    first = sdfg.add_state_after(start, "first")
    tasklet = first.add_tasklet("w", {}, {"out"}, "out = 7.0")
    first.add_edge(tasklet, "out", first.add_write("a"), None, dace.Memlet("a[0]"))
    second = sdfg.add_state_after(first, "second", assignments={"x": "a[0]"})
    sdfg.add_state_after(second, "last")
    assert not BlockFusion.can_be_applied_to(sdfg, first_block=first, second_block=second)


def test_splicing_an_empty_start_state_keeps_the_assignment_of_its_entry():
    """Predecessors of a looping start state carry its edge's assignment on later visits only."""
    sdfg = dace.SDFG("empty_start_in_a_cycle")
    empty = sdfg.add_state("empty_start", is_start_block=True)
    work = sdfg.add_state_after(empty, "work", assignments={"x": "1"})
    sdfg.add_edge(work, empty, dace.InterstateEdge("x < 0"))
    sdfg.add_state_after(work, "last", condition="x >= 0")
    assert not splice_empty_state(sdfg, empty)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
