# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Lower the ``pipeline_depth`` of loops into multiple buffering made of plain SDFG constructs.

A loop ``i = s; i <= e; i += k`` with ``pipeline_depth = D`` and ``N`` iterations, whose body fills staged buffers
(see :func:`find_staged_buffers`) before it reads them, becomes three loops. With the normalized iteration index
``n = (i - s) / k``::

    prologue  for p in [0, min(D-1, N)):          PRODUCE tile s + p*k        into slot p
    main      for i with n in [0, N - (D-1)):     CONSUME tile i              from slot n % D
                                                  PRODUCE tile i + (D-1)*k    into slot (n + D-1) % D
    epilogue  for i with n in [max(N-(D-1), 0), N): CONSUME tile i            from slot n % D

Every staged buffer gets a leading dimension of size ``D``, so tile ``n`` lives in slot ``n % D``. ``PRODUCE`` holds
the staged copies and ``CONSUME`` the rest of the body. Each is its own control flow region, so later passes can make
the copies asynchronous, place the waits between the two, or run them on different warps. The copies stay synchronous
here, so the result computes exactly what the loop did.

The main loop consumes before it produces: ``PRODUCE`` of iteration ``n`` overwrites the slot ``CONSUME`` of iteration
``n-1`` read, and the copies of iteration ``n-1`` lie between the two. When threads share the buffer, the barrier that
completes those copies (needed anyway before their tile is read) therefore also orders the reads before the overwrite.
TODO: Switch order of main-loop once asynchronous waits are supported
TODO: Re-write this docstring once asynchronous waits are supported through their own pass
"""

import copy
import warnings

import sympy

import dace
from dace import SDFG, InterstateEdge, properties, subsets, symbolic
from dace.properties import CodeBlock
from dace.sdfg import nodes
from dace.sdfg.state import (
    AbstractControlFlowRegion,
    BreakBlock,
    ContinueBlock,
    ControlFlowRegion,
    LoopRegion,
    ReturnBlock,
    SDFGState,
)
from dace.transformation import pass_pipeline as ppl
from dace.transformation import transformation
from dace.transformation.passes.analysis.loop_analysis import get_init_assignment, get_loop_end, get_loop_stride
from dace.transformation.passes.multiple_buffering.staging_analysis import StagedBuffer, find_staged_buffers


@properties.make_properties
@transformation.explicit_cf_compatible
class LowerMultipleBuffering(ppl.Pass):
    """Lower every loop with a ``pipeline_depth`` of 2 or more into a prologue, a main and an epilogue loop that
    rotate its staged buffers through ``pipeline_depth`` slots."""

    CATEGORY: str = "Optimization Preparation"

    buffers = properties.ListProperty(
        element_type=str,
        default=None,
        allow_none=True,
        desc="If set, only these arrays are staged, and naming one that cannot be staged warns.",
    )

    def __init__(self, buffers: list[str] | None = None):
        super().__init__()
        self.buffers = buffers

    def modifies(self) -> ppl.Modifies:
        return (
            ppl.Modifies.CFG
            | ppl.Modifies.Descriptors
            | ppl.Modifies.AccessNodes
            | ppl.Modifies.Memlets
            | ppl.Modifies.Symbols
        )

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def apply_pass(self, sdfg: SDFG, _) -> dict[str, int] | None:
        """
        :return: The depth of every lowered loop, by label, or None if no loop was lowered.
        """
        # Innermost loops first, so the copies of a loop body carry its inner loops already lowered
        loops = [
            region
            for region in sdfg.all_control_flow_regions(recursive=True, parent_first=False)
            if isinstance(region, LoopRegion) and region.pipeline_depth >= 2
        ]
        names = set(self.buffers) if self.buffers is not None else None
        lowered = {}
        for loop in loops:
            depth = loop.pipeline_depth
            loop.pipeline_depth = 0
            reason = lower_loop(loop, depth, names)
            if reason is None:
                lowered[loop.label] = depth
            else:
                warnings.warn(f"Multiple buffering of loop {loop.label} skipped: {reason}")
        if lowered:
            sdfg.reset_cfg_list()
        return lowered or None


def lower_loop(loop: LoopRegion, depth: int, names: set[str] | None = None) -> str | None:
    """Lower ``loop`` into multiple buffering with ``depth`` slots.

    :param loop: The loop to lower. Its ``pipeline_depth`` is ignored.
    :param depth: The number of slots of each staged buffer, at least 2.
    :param names: If given, only these arrays are staged.
    :return: None if the loop was lowered, else the reason it was left unchanged.
    """
    sdfg = loop.sdfg
    reason = _check_loop(loop)
    if reason is not None:
        return reason
    staged, rejected = find_staged_buffers(loop, sdfg, names)
    for name in sorted((names or set()) & set(rejected)):
        warnings.warn(f"Buffer {name} of loop {loop.label} cannot be staged: {rejected[name]}")
    if not staged:
        return "no staged buffer"

    var = loop.loop_variable
    start = get_init_assignment(loop)
    stride = symbolic.simplify(get_loop_stride(loop))
    end = get_loop_end(loop)
    slot = symbolic.symbol(sdfg.find_new_symbol(f"__mb_slot_{var}"))

    produce, consume = _split_body(loop, staged)
    for buffer in staged:
        _add_slot_dimension(sdfg, loop, buffer.name, depth, slot)

    itersym = symbolic.pystr_to_symbolic(var)
    ahead = depth - 1
    index = _normalized_index(itersym, start, stride)
    trip_count = _normalized_index(end, start, stride) + 1

    # Prologue: load the first D-1 tiles
    prologue_var = sdfg.find_new_symbol(f"_mb_{var}")
    prologue_sym = symbolic.symbol(prologue_var)
    if var in sdfg.symbols:
        sdfg.add_symbol(prologue_var, sdfg.symbols[var])
    prologue = LoopRegion(
        f"{loop.label}_prologue",
        condition_expr=f"{prologue_var} < {_str(sympy.Min(ahead, trip_count))}",
        loop_var=prologue_var,
        initialize_expr=f"{prologue_var} = 0",
        update_expr=f"{prologue_var} = {prologue_var} + 1",
    )
    prologue_body = _clone(produce, sdfg)
    prologue_body.replace_dict({var: _str(start + prologue_sym * stride)})
    prologue_body.replace_dict({slot.name: prologue_var})
    prologue.add_node(prologue_body, is_start_block=True)

    # Epilogue: consume the last D-1 tiles
    epilogue = LoopRegion(
        f"{loop.label}_epilogue",
        condition_expr=copy.deepcopy(loop.loop_condition),
        loop_var=var,
        initialize_expr=f"{var} = {_str(start + sympy.Max(trip_count - ahead, 0) * stride)}",
        update_expr=copy.deepcopy(loop.update_statement),
    )
    epilogue_body = _clone(consume, sdfg)
    epilogue_body.replace_dict({slot.name: _str(sympy.Mod(index, depth))})
    epilogue.add_node(epilogue_body, is_start_block=True)

    # Main loop: compute on tile n, then load tile n + D-1
    produce.replace_dict({var: _str(itersym + ahead * stride)})
    produce.replace_dict({slot.name: _str(sympy.Mod(index + ahead, depth))})
    consume.replace_dict({slot.name: _str(sympy.Mod(index, depth))})
    loop.loop_condition = CodeBlock(f"{var} <= {_str(end - ahead * stride)}")

    _splice(loop, prologue, epilogue)
    return None


def _check_loop(loop: LoopRegion) -> str | None:
    """The reason ``loop`` cannot be lowered, or None."""
    if not loop.loop_variable:
        return "no loop variable"
    if loop.inverted:
        return "inverted loop"
    if get_init_assignment(loop) is None or get_loop_end(loop) is None:
        return "loop range cannot be analyzed"
    stride = get_loop_stride(loop)
    if stride is None:
        return "loop stride cannot be analyzed"
    stride = symbolic.simplify(stride)
    if not stride.is_number or stride <= 0:
        return f"stride {stride} is not a positive constant"
    for block in _own_blocks(loop):
        if isinstance(block, (BreakBlock, ContinueBlock, ReturnBlock)):
            return f"loop body holds {type(block).__name__} {block.label}"
    return None


def _own_blocks(region: AbstractControlFlowRegion):
    """The blocks of ``region`` that a break or continue in it would apply to: none of nested loops."""
    for block in region.nodes():
        yield block
        if isinstance(block, AbstractControlFlowRegion) and not isinstance(block, LoopRegion):
            yield from _own_blocks(block)
        elif isinstance(block, LoopRegion):
            # A return in a nested loop still leaves this loop
            yield from (b for b in block.all_control_flow_blocks() if isinstance(b, ReturnBlock))


def _normalized_index(value, start, stride):
    """The iteration index of ``value``, for a loop from ``start`` in steps of ``stride``."""
    if stride == 1:
        return value - start
    return symbolic.int_floor(value - start, stride)


def _str(expr) -> str:
    return symbolic.symstr(expr)


def _split_body(loop: LoopRegion, staged: list[StagedBuffer]) -> tuple[ControlFlowRegion, ControlFlowRegion]:
    """Move the staged copies of ``loop`` into a new produce region and the rest of the body into a consume region,
    which become the body of the loop, in that order: consume, then produce."""
    sdfg = loop.sdfg
    blocks = list(loop.nodes())
    edges = list(loop.edges())
    start = loop.start_block
    for block in blocks:
        loop.remove_node(block)

    produce = ControlFlowRegion(f"{loop.label}_produce", sdfg)
    consume = ControlFlowRegion(f"{loop.label}_consume", sdfg)
    for block in blocks:
        consume.add_node(block, is_start_block=block is start, reset_cfg_list=False)
    for edge in edges:
        consume.add_edge(edge.src, edge.dst, edge.data)
    loop.add_node(consume, is_start_block=True, reset_cfg_list=False)
    loop.add_node(produce, reset_cfg_list=False)
    loop.add_edge(consume, produce, InterstateEdge())

    state = produce.add_state(f"{loop.label}_produce_state", is_start_block=True)
    emptied = set()
    for buffer in staged:
        for staged_copy in buffer.copies:
            _move_copy(staged_copy.state, state, staged_copy)
            emptied |= {(staged_copy.state, staged_copy.src), (staged_copy.state, staged_copy.dst)}
    for old_state, node in emptied:
        if node in old_state.nodes() and old_state.degree(node) == 0:
            old_state.remove_node(node)
    return produce, consume


def _move_copy(old_state: SDFGState, state: SDFGState, staged_copy) -> None:
    """Move one copy from ``old_state`` into ``state``, with new access nodes at both ends."""
    src = state.add_access(staged_copy.src.data)
    dst = state.add_access(staged_copy.dst.data)
    for edge in staged_copy.edges:
        old_state.remove_edge(edge)
    if staged_copy.copy_node is None:
        (edge,) = staged_copy.edges
        state.add_edge(src, edge.src_conn, dst, edge.dst_conn, edge.data)
        return
    first, second = staged_copy.edges
    old_state.remove_node(staged_copy.copy_node)
    state.add_node(staged_copy.copy_node)
    state.add_edge(src, first.src_conn, staged_copy.copy_node, first.dst_conn, first.data)
    state.add_edge(staged_copy.copy_node, second.src_conn, dst, second.dst_conn, second.data)


def _add_slot_dimension(sdfg: SDFG, loop: LoopRegion, name: str, depth: int, slot: symbolic.symbol) -> None:
    """Give ``name`` a leading dimension of ``depth`` slots, and index it with ``slot`` everywhere in ``loop``."""
    _prepend_dimension(sdfg.arrays[name], depth)
    _index_with_slot(loop.states(), name, depth, slot)


def _prepend_dimension(desc, depth: int) -> None:
    desc.set_shape(
        (depth, *desc.shape),
        strides=(desc.total_size, *desc.strides),
        total_size=desc.total_size * depth,
        offset=(0, *desc.offset),
    )


def _index_with_slot(states, name: str, depth: int, slot: symbolic.symbol) -> None:
    """Index every access to ``name`` in ``states`` with ``slot``, now that it has a leading dimension of ``depth``.

    A nested SDFG gets the whole buffer, since its descriptor must match, and the slot as a symbol to index it with.
    """
    for state in states:
        whole = set()
        for node in state.nodes():
            if isinstance(node, nodes.NestedSDFG):
                whole |= _pass_slot_to_nested(state, node, name, depth, slot)
        for edge in state.edges():
            memlet = edge.data
            if memlet.is_empty():
                continue
            slot_range = (0, depth - 1, 1) if edge in whole else (slot, slot, 1)
            if memlet.data == name:
                memlet.subset = subsets.Range([slot_range] + list(memlet.subset.ranges))
                continue
            path = state.memlet_path(edge)
            ends = (path[0].src, path[-1].dst)
            if any(isinstance(end, nodes.AccessNode) and end.data == name for end in ends):
                # The buffer is the other side of a copy
                other = memlet.other_subset
                if other is None:
                    shape = state.sdfg.arrays[name].shape[1:]
                    other = subsets.Range([(0, s - 1, 1) for s in shape])
                memlet.other_subset = subsets.Range([slot_range] + list(other.ranges))


def _pass_slot_to_nested(
    state: SDFGState, nsdfg: nodes.NestedSDFG, name: str, depth: int, slot: symbolic.symbol
) -> set:
    """Give the arrays of ``nsdfg`` connected to ``name`` the slot dimension, and map ``slot`` into it.

    :return: The edges of the memlet paths between ``name`` and ``nsdfg``, which must carry all slots.
    """
    edges = set()
    for edge in [*state.in_edges(nsdfg), *state.out_edges(nsdfg)]:
        if edge.data.data != name:
            continue
        connector = edge.dst_conn if edge.dst is nsdfg else edge.src_conn
        inner = nsdfg.sdfg
        inner_slot = symbolic.symbol(inner.find_new_symbol(slot.name))
        inner.add_symbol(inner_slot.name, dace.int64)
        nsdfg.symbol_mapping[inner_slot.name] = slot
        _prepend_dimension(inner.arrays[connector], depth)
        _index_with_slot(inner.states(), connector, depth, inner_slot)
        edges |= set(state.memlet_path(edge))
    return edges


def _clone(region: ControlFlowRegion, sdfg: SDFG) -> ControlFlowRegion:
    """Deep copy of ``region``, with the parent pointers deepcopy drops restored."""
    new_region = copy.deepcopy(region)
    for block in [new_region, *new_region.all_control_flow_blocks()]:
        block.sdfg = sdfg
        if isinstance(block, SDFGState):
            for node in block.nodes():
                if isinstance(node, nodes.NestedSDFG):
                    node.sdfg.parent_sdfg = sdfg
    return new_region


def _splice(loop: LoopRegion, prologue: LoopRegion, epilogue: LoopRegion) -> None:
    """Put ``prologue`` before and ``epilogue`` after ``loop`` in its parent region."""
    parent = loop.parent_graph
    was_start = parent.start_block is loop
    in_edges = list(parent.in_edges(loop))
    out_edges = list(parent.out_edges(loop))
    parent.add_node(prologue, ensure_unique_name=True, reset_cfg_list=False)
    parent.add_node(epilogue, ensure_unique_name=True, reset_cfg_list=False)
    for edge in in_edges:
        parent.remove_edge(edge)
        parent.add_edge(edge.src, prologue, edge.data)
    for edge in out_edges:
        parent.remove_edge(edge)
        parent.add_edge(epilogue, edge.dst, edge.data)
    parent.add_edge(prologue, loop, InterstateEdge())
    parent.add_edge(loop, epilogue, InterstateEdge())
    if was_start:
        parent.start_block = parent.node_id(prologue)
