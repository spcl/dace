# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.

import copy

import numpy as np
import pytest

import dace
from dace import SDFG
from dace.sdfg import InvalidSDFGError, InterstateEdge
from dace.sdfg.state import ControlFlowRegion


def test_validation_no_state():
    """SDFGs require a start block."""
    sdfg = SDFG("empty_sdfg")

    with pytest.raises(InvalidSDFGError, match="SDFGs are required to contain at least one state."):
        sdfg.validate()


def test_validation_ambiguous_start_block():
    """SDFGs require an unambiguous start block."""
    sdfg = SDFG("ambiguous_start_block")
    state_1 = sdfg.add_state("state_1")
    state_2 = sdfg.add_state("state_2")

    with pytest.raises(InvalidSDFGError, match="Starting block is ambiguous or undefined."):
        sdfg.validate()

    # Disambiguate by adding an edge between state_1 and state_2.
    sdfg.add_edge(state_1, state_2, InterstateEdge())
    assert sdfg.is_valid()


def test_start_block_survives_removal_of_an_earlier_block():
    """Removing a block ordered before the start block must not re-point the start block."""
    cfg = ControlFlowRegion("shift")
    first = cfg.add_state("first")
    second = cfg.add_state("second")
    start = cfg.add_state("start", is_start_block=True)
    cfg.add_state("last")
    cfg.add_edge(first, second, InterstateEdge())

    # More than one source node, so the pinned start block decides rather than ``source_nodes()``.
    assert len(cfg.source_nodes()) == 3
    cfg.remove_node(first)

    assert cfg.start_block is start


def test_removing_the_start_block_leaves_no_dangling_reference():
    """Neither the pinned start block nor its cache may outlive the block itself."""
    cfg = ControlFlowRegion("dangling")
    head = cfg.add_state("head", is_start_block=True)
    body = cfg.add_state("body")
    tail = cfg.add_state("tail")
    cfg.add_edge(head, body, InterstateEdge())
    cfg.add_edge(body, tail, InterstateEdge())
    cfg.add_edge(tail, head, InterstateEdge())  # Back edge, so there are no source nodes to fall back on.

    assert cfg.start_block is head  # Populates the cache.
    cfg.remove_node(head)

    assert cfg.start_block is body


def test_pinned_start_block_survives_a_copy_and_a_serialization_round_trip():
    """A copied or serialized region must come back pointing at the same start block.

    Persistence stores the start block by position, so a removal that shifts positions used to move
    the entry of every SDFG written out or deep-copied afterwards.
    """
    sdfg = SDFG("pinned_start_block")
    sdfg.add_array("a", [1], dace.float64)
    region = ControlFlowRegion("region", sdfg)
    sdfg.add_node(region, is_start_block=True)
    first = region.add_state("first")
    start = region.add_state("start", is_start_block=True)
    region.add_state("other")

    region.remove_node(first)
    assert region.start_block is start

    def region_of(graph):
        return next(b for b in graph.nodes() if isinstance(b, ControlFlowRegion))

    assert region_of(copy.deepcopy(sdfg)).start_block.label == "start"
    assert region_of(SDFG.from_json(sdfg.to_json())).start_block.label == "start"


def test_reloaded_loop_still_enters_at_its_head_and_computes_the_same_values():
    """End-to-end: splice out an earlier block, persist, reload, run.

    Reloading is what makes this bite -- it rebuilds the region from the stored start block alone,
    with none of the in-memory caching that otherwise hides a stale one.
    """
    n = 8
    sdfg = SDFG("late_start_block")
    sdfg.add_array("a", [n], dace.float64)
    sdfg.add_array("b", [n], dace.float64)
    sdfg.add_symbol("i", dace.int64)

    init = sdfg.add_state("init", is_start_block=True)
    region = ControlFlowRegion("region", sdfg)
    sdfg.add_node(region)
    sdfg.add_edge(init, region, InterstateEdge(assignments={"i": "0"}))

    # Insertion order deliberately puts the start block late, so ``nop`` holds position 0.
    nop = region.add_state("nop")
    scale = region.add_state("scale")
    head = region.add_state("head", is_start_block=True)
    tail = region.add_state("tail")

    t = scale.add_tasklet("scale", {"x"}, {"y"}, "y = x * 2.0")
    scale.add_edge(scale.add_read("b"), None, t, "x", dace.Memlet("b[i]"))
    scale.add_edge(t, "y", scale.add_write("a"), None, dace.Memlet("a[i]"))

    region.add_edge(head, nop, InterstateEdge(condition=f"i < {n}"))
    region.add_edge(head, tail, InterstateEdge(condition=f"i >= {n}"))
    region.add_edge(nop, scale, InterstateEdge())
    region.add_edge(scale, head, InterstateEdge(assignments={"i": "i + 1"}))
    sdfg.validate()

    # Splice ``nop`` out: an ordinary rewrite that removes a block ahead of the pinned start.
    region.add_edge(head, scale, InterstateEdge(condition=f"i < {n}"))
    for edge in region.all_edges(nop):
        region.remove_edge(edge)
    region.remove_node(nop)
    sdfg.validate()

    assert region.start_block is head

    revived = SDFG.from_json(sdfg.to_json())
    revived_region = next(b for b in revived.nodes() if isinstance(b, ControlFlowRegion))
    assert revived_region.start_block.label == "head"

    b = np.arange(n, dtype=np.float64)
    a = np.zeros(n, dtype=np.float64)
    revived(a=a, b=b, i=0)
    assert np.allclose(a, b * 2.0)


if __name__ == "__main__":
    test_validation_no_state()
    test_validation_ambiguous_start_block()
    test_start_block_survives_removal_of_an_earlier_block()
    test_removing_the_start_block_leaves_no_dangling_reference()
    test_pinned_start_block_survives_a_copy_and_a_serialization_round_trip()
    test_reloaded_loop_still_enters_at_its_head_and_computes_the_same_values()
