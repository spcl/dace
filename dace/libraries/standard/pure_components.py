# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Building blocks for ``pure`` library expansions made only of SDFG components.

A ``pure`` expansion built from states, loops and Python tasklets is readable by every consumer of an SDFG: it
compiles through the C++ generator like any other SDFG, and a tool that walks the graph (a Python emitter, an
analysis) needs no knowledge of the library. These helpers keep such expansions short.
"""
import copy
from typing import Dict, Iterable, Optional

import dace
from dace import subsets, symbolic
from dace.memlet import Memlet
from dace.sdfg.state import ControlFlowBlock, ControlFlowRegion, LoopRegion
from dace.sdfg.graph import MultiConnectorEdge


def operand_array(nsdfg: dace.SDFG, name: str, edge: MultiConnectorEdge, outer: dace.SDFG) -> dace.data.Array:
    """Declare in ``nsdfg`` the array a connector sees through ``edge``: the subset with its length-1 dimensions
    dropped, each kept dimension striding by the outer stride times the subset step.

    :param nsdfg: The expansion's SDFG.
    :param name: The connector, which names the inner array.
    :param edge: The library node's edge on that connector.
    :param outer: The SDFG holding the library node.
    :returns: The declared descriptor; a single element is a one-element array.
    """
    desc = outer.arrays[edge.data.data]
    subset = subsets.Range.from_indices(edge.data.subset) if isinstance(edge.data.subset,
                                                                        subsets.Indices) else edge.data.subset
    kept = [dim for dim, (begin, end, step) in enumerate(subset.ranges) if begin != end]
    shape = [subset.size()[dim] for dim in kept] or [1]
    strides = [desc.strides[dim] * subset.ranges[dim][2] for dim in kept] or [1]
    nsdfg.add_array(name, shape, desc.dtype, strides=strides, storage=desc.storage)
    return nsdfg.arrays[name]


def element(nsdfg: dace.SDFG, name: str, position: str) -> str:
    """The memlet text of element ``position`` of array ``name`` counted in row-major order, the order the C++
    lowerings walk an operand through one pointer."""
    shape = nsdfg.arrays[name].shape
    index = [f'{position}']
    for extent in reversed(shape[1:]):
        extent_text = symbolic.symstr(extent, cpp_mode=False)
        index[0:1] = [f'({index[0]}) // ({extent_text})', f'({index[0]}) % ({extent_text})']
    return f'{name}[{", ".join(index)}]'


def chain(region: ControlFlowRegion, blocks: Iterable[ControlFlowBlock]) -> None:
    """Add ``blocks`` to ``region`` to run one after the other, the first as its start block."""
    previous: Optional[ControlFlowBlock] = None
    for block in blocks:
        region.add_node(block, is_start_block=previous is None)
        if previous is not None:
            region.add_edge(previous, block, dace.InterstateEdge())
        previous = block


def counted_loop(label: str, var: str, start: str, stop: str, step: str = '1') -> LoopRegion:
    """``for var in range(start, stop, step)`` with a positive ``step``, as a loop region."""
    return LoopRegion(label, f'{var} < {stop}', var, f'{var} = {start}', f'{var} = {var} + {step}')


def tasklet_state(sdfg: dace.SDFG, label: str, code: str, reads: Dict[str, Memlet],
                  writes: Dict[str, Memlet]) -> dace.SDFGState:
    """A detached state of ``sdfg`` holding one Python tasklet; ``reads`` and ``writes`` map each connector to
    the memlet it moves, whose data names the access node."""
    state = dace.SDFGState(label, sdfg=sdfg)
    tasklet = state.add_tasklet(label, dict.fromkeys(reads), dict.fromkeys(writes), code)
    # a memlet object is never shared between edges
    for conn, memlet in reads.items():
        state.add_edge(state.add_read(memlet.data), None, tasklet, conn, copy.deepcopy(memlet))
    for conn, memlet in writes.items():
        state.add_edge(tasklet, conn, state.add_write(memlet.data), None, copy.deepcopy(memlet))
    return state
