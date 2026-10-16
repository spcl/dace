# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``all_control_flow_regions(parent_first=True)`` yields the regions of the recursive definition, in its order."""

from collections.abc import Iterator

import dace
from dace.properties import CodeBlock
from dace.sdfg import nodes
from dace.sdfg.state import AbstractControlFlowRegion, ConditionalBlock, ControlFlowRegion, LoopRegion


def recursive_regions(cfg: AbstractControlFlowRegion, recursive: bool) -> Iterator[AbstractControlFlowRegion]:
    yield cfg
    for block in cfg.nodes():
        if isinstance(block, dace.SDFGState) and recursive:
            for node in block.nodes():
                if isinstance(node, nodes.NestedSDFG) and node.sdfg:
                    yield from recursive_regions(node.sdfg, recursive)
        elif isinstance(block, AbstractControlFlowRegion):
            yield from recursive_regions(block, recursive)


def nested_sdfg(label: str, depth: int) -> dace.SDFG:
    """A loop holding a conditional holding a state with a nested SDFG ``depth`` more levels down."""
    sdfg = dace.SDFG(label)
    sdfg.add_array("a", [10], dace.float64)
    first = sdfg.add_state(f"{label}_first", is_start_block=True)
    loop = LoopRegion(f"{label}_loop", "i < 10", "i", "i = 0", "i = i + 1")
    sdfg.add_node(loop)
    sdfg.add_edge(first, loop, dace.InterstateEdge())
    body = loop.add_state(f"{label}_body", is_start_block=True)
    cond = ConditionalBlock(f"{label}_cond")
    loop.add_node(cond)
    loop.add_edge(body, cond, dace.InterstateEdge())
    for branch_index in range(2):
        branch = ControlFlowRegion(f"{label}_branch{branch_index}", sdfg=sdfg)
        cond.add_branch(CodeBlock("i < 5") if branch_index == 0 else None, branch)
        state = branch.add_state(f"{label}_bstate{branch_index}", is_start_block=True)
        if depth > 0 and branch_index == 0:
            inner = nested_sdfg(f"{label}_n", depth - 1)
            state.add_nested_sdfg(inner, {}, {"a"})
            state.add_nested_sdfg(nested_sdfg(f"{label}_m", 0), {}, {"a"})
            tasklet = state.add_tasklet(f"{label}_t", {}, {"__out"}, "__out = 1")
            state.add_edge(tasklet, "__out", state.add_write("a"), None, dace.Memlet("a[0]"))
    last = sdfg.add_state(f"{label}_last")
    sdfg.add_edge(loop, last, dace.InterstateEdge())
    return sdfg


def test_preorder_walk_matches_the_recursive_definition():
    sdfg = nested_sdfg("root", 2)
    for recursive in (True, False):
        expected = [cfg.label for cfg in recursive_regions(sdfg, recursive)]
        assert [cfg.label for cfg in sdfg.all_control_flow_regions(recursive=recursive)] == expected
    labels = [cfg.label for cfg in sdfg.all_control_flow_regions(recursive=True)]
    assert labels[:4] == ["root", "root_loop", "root_cond", "root_branch0"]
    assert "root_n_n" in labels and "root_n_m" in labels and len(labels) == len(set(labels)) == 25


if __name__ == "__main__":
    test_preorder_walk_matches_the_recursive_definition()
