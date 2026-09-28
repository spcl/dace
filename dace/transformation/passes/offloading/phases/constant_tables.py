# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Turn a table filled once with literals into an SDFG constant, which every kernel reads without a copy."""
import ast
from typing import Dict, Optional, Tuple

import numpy as np
from ordered_set import OrderedSet

from dace import data, dtypes, symbolic
from dace.sdfg import nodes, SDFG
from dace.sdfg.state import LoopRegion, SDFGState

#: Per table: the state that fills it, and the fill tasklets with the index and value each writes.
Fill = Tuple[SDFGState, Dict[nodes.Tasklet, Tuple[int, object]]]


def fold_constant_tables(sdfg: SDFG) -> None:
    """Replace each literal-filled table (CloudSC's ``imelt[0:5] = 2, 3, 4, 3, -99``) by an SDFG constant.

    Every constant array becomes a register: host code and each kernel declare it locally, so any side reads it.
    """
    for name, (state, writes) in literal_fills(sdfg).items():
        desc = sdfg.arrays[name]
        values = np.zeros(desc.total_size, dtype=desc.dtype.type)
        for index, value in writes.values():
            values[index] = value
        sdfg.add_constant(name, values, desc)
        for tasklet in writes:
            targets = [edge.dst for edge in state.out_edges(tasklet)]
            state.remove_node(tasklet)
            state.remove_nodes_from([target for target in targets if state.degree(target) == 0])
    for name in sdfg.constants:
        if name in sdfg.arrays:
            sdfg.arrays[name].storage = dtypes.StorageType.Register


def literal_fills(sdfg: SDFG) -> Dict[str, Fill]:
    """Transient 1-D tables whose every element is written once, by a tasklet assigning a literal, in one state
    outside every loop, and that nothing else writes."""
    fills: Dict[str, Fill] = {}
    refused: OrderedSet[str] = OrderedSet(sdfg.constants)
    for state in sdfg.states():
        for node in state.data_nodes():
            desc = sdfg.arrays[node.data]
            if type(desc) is not data.Array or not desc.transient or len(desc.shape) != 1:
                continue
            for edge in state.in_edges(node):
                write = literal_write(state, edge)
                owner = fills.setdefault(node.data, (state, {}))[0]
                if write is None or owner is not state or in_a_loop(state):
                    refused.add(node.data)
                else:
                    fills[node.data][1][edge.src] = write
    return {
        name: fill
        for name, fill in fills.items() if name not in refused and sorted(
            index for index, _ in fill[1].values()) == list(range(int(sdfg.arrays[name].total_size)))
    }


def literal_write(state: SDFGState, edge) -> Optional[Tuple[int, object]]:
    """``(index, value)`` when ``edge`` comes from an input-less tasklet assigning a literal to one element."""
    tasklet = edge.src
    if not isinstance(tasklet, nodes.Tasklet) or state.in_degree(tasklet) or state.out_degree(tasklet) != 1:
        return None
    index = edge.data.subset.min_element()[0] if edge.data.subset.num_elements() == 1 else None
    try:
        statement, = ast.parse(tasklet.code.as_string.strip()).body
        value = ast.literal_eval(statement.value)
    except (SyntaxError, ValueError, AttributeError):
        return None
    if index is None or symbolic.issymbolic(index) or not isinstance(statement, ast.Assign):
        return None
    return int(index), value


def in_a_loop(block: SDFGState) -> bool:
    """A ``LoopRegion`` of the block's own SDFG encloses it."""
    region = block.parent_graph
    while region is not None and not isinstance(region, SDFG):
        if isinstance(region, LoopRegion):
            return True
        region = region.parent_graph
    return False
