# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The thread-block size around a node in device code, also from SDFGs nested in control flow regions."""

import dace
from dace.sdfg import nodes
from dace.sdfg.scope import devicelevel_block_size
from dace.sdfg.state import ConditionalBlock


@dace.program
def inner(A: dace.float64[32], i: dace.int64):
    A[i] = A[i] + 1


@dace.program
def outer(A: dace.float64[32], flag: dace.int32[32]):
    for i in dace.map[0:32]:
        if flag[i] > 0:
            inner(A, i)


def test_nested_sdfg_in_a_branch():
    sdfg = outer.to_sdfg(simplify=False)
    for node, _ in sdfg.all_nodes_recursive():
        if isinstance(node, nodes.MapEntry):
            node.map.schedule = dace.ScheduleType.GPU_Device
    # The innermost SDFG, nested in a state of a branch
    nested = [n for n, s in sdfg.all_nodes_recursive() if isinstance(n, nodes.NestedSDFG) and "inner" in n.sdfg.name]
    assert nested
    region = nested[0].sdfg.parent.parent_graph
    while region is not None and not isinstance(region, ConditionalBlock):
        region = region.parent_graph
    assert region is not None, "The nested SDFG is not in a branch"
    inner_sdfg = nested[0].sdfg
    state, node = next((s, n) for s in inner_sdfg.states() for n in s.nodes() if isinstance(n, nodes.Tasklet))
    expected = tuple(int(s) for s in dace.Config.get("compiler", "cuda", "default_block_size").split(","))
    assert devicelevel_block_size(inner_sdfg, state, node) == expected


if __name__ == "__main__":
    test_nested_sdfg_in_a_branch()
