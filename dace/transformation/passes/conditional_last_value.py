# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Parallelize a loop that carries a scalar from the last iteration that stored it under a condition (TSVC s258)::

    for i in [s, e]:                       map i:  priv[i - s] = f(i) where c(i);  mark[i - s] = i - s if c(i) else -1
        if c(i):                           last = inclusive max-scan of mark
            x = f(i)              ->       map i:  held = priv[last[i - s]] if last[i - s] >= 0 else x
        use(x)                                     use(held)
                                           if e >= s: x = priv[last[e - s]] if last[e - s] >= 0 else x

The body splits at the conditional store into two loops, each of which ``LoopToMap`` then lifts. The pass applies only
when ``LoopToMap`` refuses the loop on ``x`` alone and accepts it with ``x`` iteration-private.
"""

import copy
from typing import Any, NamedTuple

from dace import SDFG, InterstateEdge, data, dtypes, properties, subsets, symbolic
from dace.libraries.standard.nodes import scan
from dace.memlet import Memlet
from dace.ordered import OrderedSet
from dace.sdfg import nodes
from dace.sdfg.state import ConditionalBlock, ControlFlowBlock, ControlFlowRegion, LoopRegion, SDFGState
from dace.transformation import pass_pipeline as ppl
from dace.transformation.passes.analysis import loop_analysis
from dace.transformation.passes.promote_constant_index_access import PromoteConstantIndexAccess

#: Picks the stored value of the last iteration that wrote, or the value from before the loop
PICK = "_out = _values[_last] if _last >= 0 else _before"


class LastValueStore(NamedTuple):
    """The conditional store of scalar ``name`` in ``loop``, and the body blocks after its conditional block."""

    name: str
    branch: SDFGState
    store: nodes.AccessNode
    after: list[SDFGState]


def chain(loop: LoopRegion) -> list[ControlFlowBlock] | None:
    """``loop``'s blocks when they run one after another over unconditional edges that assign nothing."""
    order = [loop.start_block]
    # Follow the single out-edge until the chain ends; a branch, an assignment or a cycle refuses
    while loop.out_degree(order[-1]) == 1:
        edge = loop.out_edges(order[-1])[0]
        if edge.data.assignments or not edge.data.is_unconditional() or edge.dst in order:
            return None
        order.append(edge.dst)
    return order if len(order) == loop.number_of_nodes() else None


def accessed(states: list[SDFGState]) -> tuple[OrderedSet[str], OrderedSet[str]]:
    """The containers ``states`` read and those they write."""
    read: OrderedSet[str] = OrderedSet()
    written: OrderedSet[str] = OrderedSet()
    # Collect every container each state reads or writes
    for state in states:
        # An out-edge reads the container, an in-edge writes it
        for node in state.data_nodes():
            if state.out_degree(node) > 0:
                read.add(node.data)
            if state.in_degree(node) > 0:
                written.add(node.data)
    return read, written


def single_store(branch: SDFGState, sdfg: SDFG) -> nodes.AccessNode | None:
    """The one top-level write of a transient scalar in ``branch`` that nothing in ``branch`` reads back."""
    stores = [
        n
        for n in branch.data_nodes()
        if isinstance(sdfg.arrays[n.data], data.Scalar)
        and sdfg.arrays[n.data].transient
        and branch.in_degree(n) == 1
        and branch.out_degree(n) == 0
        and branch.entry_node(n) is None
    ]
    if len(stores) != 1 or len([n for n in branch.data_nodes() if n.data == stores[0].data]) != 1:
        return None
    return stores[0]


def states_of(blocks: list[ControlFlowBlock]) -> list[SDFGState]:
    states = [b for b in blocks if isinstance(b, SDFGState)]
    # States nested in control-flow blocks count as the block's states
    for block in blocks:
        if not isinstance(block, SDFGState):
            states += [s for r in block.all_control_flow_regions() for s in r.nodes() if isinstance(s, SDFGState)]
    return states


def separable(before: list[ControlFlowBlock], condition: Any, after: list[SDFGState], name: str) -> bool:
    """Whether the blocks up to the store (``before``, guarded by ``condition``) and those ``after`` it share no
    container other than ``name``, which only ``after`` reads."""
    read1, written1 = accessed(states_of(before))
    read1 |= OrderedSet(str(s) for s in condition.get_free_symbols())
    read2, written2 = accessed(after)
    if name in read1 or name in written2 or name not in read2:
        return False
    return not ((written1 - {name}) & (read2 | written2) or written2 & read1)


def conditional_store(sdfg: SDFG, loop: LoopRegion) -> LastValueStore | None:
    """The single-branch conditional store of a scalar that only later blocks of ``loop`` read (:func:`separable`)."""
    order = chain(loop)
    if order is None or loop_analysis.get_loop_stride(loop) != 1 or loop_analysis.get_loop_end(loop) is None:
        return None
    blocks = [b for b in order if isinstance(b, ConditionalBlock)]
    if len(blocks) != 1 or len(blocks[0].branches) != 1:
        return None
    condition, region = blocks[0].branches[0]
    if condition is None or region.number_of_nodes() != 1 or not isinstance(region.start_block, SDFGState):
        return None
    store = single_store(region.start_block, sdfg)
    position = order.index(blocks[0]) + 1
    after = order[position:]
    if store is None or not after or not all(isinstance(b, SDFGState) for b in after):
        return None
    if not separable(order[:position], condition, after, store.data):
        return None
    return LastValueStore(store.data, region.start_block, store, after)


def add_pick(sdfg: SDFG, state: SDFGState, values: str, last: str, index: Any, before: str, out: str) -> None:
    """``out = values[last[index]] if last[index] >= 0 else before`` in ``state``."""
    pick = state.add_tasklet("pick_last_value", {"_values": None, "_last": None, "_before": None}, {"_out": None}, PICK)
    state.add_edge(state.add_read(values), None, pick, "_values", Memlet.from_array(values, sdfg.arrays[values]))
    state.add_edge(state.add_read(last), None, pick, "_last", Memlet(data=last, subset=subsets.Range([index])))
    state.add_edge(state.add_read(before), None, pick, "_before", Memlet(data=before, subset="0"))
    state.add_edge(pick, "_out", state.add_write(out), None, Memlet(data=out, subset="0"))


def add_mark(state: SDFGState, mark: str, offset: Any, value: str) -> None:
    tasklet = state.add_tasklet("mark_iteration", {}, {"_out": None}, f"_out = {value}")
    state.add_edge(tasklet, "_out", state.add_write(mark), None, Memlet(data=mark, subset=subsets.Range([offset])))


def split_loop(sdfg: SDFG, loop: LoopRegion, found: LastValueStore) -> None:
    """Store per iteration, scan the last writer, and read the picked value in a second loop (module docstring)."""
    start, end = loop_analysis.get_init_assignment(loop), loop_analysis.get_loop_end(loop)
    offset = symbolic.pystr_to_symbolic(loop.loop_variable) - start
    point = (offset, offset, 1)
    desc = sdfg.arrays[found.name]
    trip = [end - start + 1]
    values = sdfg.add_transient(f"{found.name}_by_iteration", trip, desc.dtype, find_new_name=True)[0]
    mark = sdfg.add_transient(f"{found.name}_written_at", trip, dtypes.int64, find_new_name=True)[0]
    last = sdfg.add_transient(f"{found.name}_last_written", trip, dtypes.int64, find_new_name=True)[0]
    held = sdfg.add_scalar(f"{found.name}_held", desc.dtype, transient=True, find_new_name=True)[0]

    # First loop: every iteration stores at its own index and marks whether it stored
    found.store.data = values
    store_edge = found.branch.in_edges(found.store)[0]
    store_edge.data.data = values
    store_edge.data.subset = subsets.Range([point])
    add_mark(found.branch, mark, point, symbolic.symstr(offset))
    add_mark(loop.add_state_before(loop.start_block, "unmark", is_start_block=True), mark, point, "-1")

    # Second loop: the body after the store, reading the picked value
    memo = {id(b): b for b in loop.nodes()}
    memo.update((id(e.data), e.data) for e in loop.edges())
    second = copy.deepcopy(loop, memo)
    # The copy keeps the loop's header only; its body is moved in below
    for block in list(second.nodes()):
        second.remove_node(block)
    second.label = f"{loop.label}_after_{found.name}"
    edges = [(a, b, loop.edges_between(a, b)[0].data) for a, b in zip(found.after, found.after[1:])]
    # The blocks after the store leave the first loop
    for block in found.after:
        loop.remove_node(block)
    head = second.add_state(f"pick_{found.name}", is_start_block=True)
    add_pick(sdfg, head, values, last, point, found.name, held)
    # Move each later block into the second loop, reading the picked value in place of the scalar
    for block in found.after:
        second.add_node(block)
        # Access nodes of the scalar now read the picked value
        for node in block.data_nodes():
            if node.data == found.name:
                node.data = held
        # Memlets of the scalar now name the picked value
        for edge in block.edges():
            if edge.data.data == found.name:
                edge.data.data = held
    second.add_edge(head, found.after[0], InterstateEdge())
    # Restore the edges between the moved blocks
    for a, b, edge_data in edges:
        second.add_edge(a, b, edge_data)

    # Between the loops the max-scan; after them the value of the last iteration that stored
    parent = loop.parent_graph
    exits = list(parent.out_edges(loop))
    between = parent.add_state(f"scan_{found.name}")
    node = scan.Scan(f"scan_{found.name}", op=scan.ScanOp.MAX)
    between.add_node(node)
    between.add_edge(
        between.add_read(mark), None, node, scan.INPUT_CONNECTOR_NAME, Memlet.from_array(mark, sdfg.arrays[mark])
    )
    between.add_edge(
        node, scan.OUTPUT_CONNECTOR_NAME, between.add_write(last), None, Memlet.from_array(last, sdfg.arrays[last])
    )
    parent.add_node(second)
    keep = ConditionalBlock(f"keep_{found.name}", sdfg=sdfg)
    branch = ControlFlowRegion(f"keep_{found.name}_body", sdfg=sdfg)
    final = end - start
    add_pick(
        sdfg,
        branch.add_state(f"keep_{found.name}", is_start_block=True),
        values,
        last,
        (final, final, 1),
        found.name,
        found.name,
    )
    keep.add_branch(properties.CodeBlock(f"{symbolic.symstr(end)} >= {symbolic.symstr(start)}"), branch)
    parent.add_node(keep)
    # The loop's exits now leave from the final pick
    for edge in exits:
        parent.add_edge(keep, edge.dst, edge.data)
        parent.remove_edge(edge)
    parent.add_edge(loop, between, InterstateEdge())
    parent.add_edge(between, second, InterstateEdge())
    parent.add_edge(second, keep, InterstateEdge())


class ConditionalLastValue(ppl.Pass):
    """Split a loop at the conditional store of a scalar its later blocks read, so both halves lift to maps."""

    CATEGORY: str = "Optimization Preparation"

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Everything

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def apply_pass(self, sdfg: SDFG, pipeline_results: dict[str, Any]) -> int | None:
        """Split every loop whose only refusal is a conditional last-value scalar; the number split, or None."""
        split = 0
        # Each SDFG, nested ones included, holds its own loops
        for owner in list(sdfg.all_sdfgs_recursive()):
            loops = [r for r in owner.all_control_flow_regions() if isinstance(r, LoopRegion) and r.loop_variable]
            # Each loop is split at most once, on its own refusal
            for loop in loops:
                found = conditional_store(owner, loop)
                if found is None:
                    continue
                accepted, refused_on = PromoteConstantIndexAccess._l2m_verdict(loop, owner)
                if accepted or refused_on != found.name:
                    continue
                if not PromoteConstantIndexAccess._l2m_verdict(loop, owner, frozenset({found.name}))[0]:
                    continue
                split_loop(owner, loop, found)
                split += 1
        return split or None
