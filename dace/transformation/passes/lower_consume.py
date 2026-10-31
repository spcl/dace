# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Lowers consume scopes to maps over processing elements whose bodies loop popping chunks."""

import copy
from typing import Any

from dace import data, dtypes, nodes, subsets
from dace.memlet import Memlet
from dace.properties import CodeBlock
from dace.sdfg import SDFG, InterstateEdge, SDFGState
from dace.sdfg.graph import MultiConnectorEdge
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion, LoopRegion
from dace.transformation import pass_pipeline as ppl
from dace.transformation import transformation
from dace.transformation.helpers import nest_state_subgraph

NUM_POPPED_CONNECTOR = data.Stream.NUM_POPPED_CONNECTOR


def chunk_name(consume: nodes.Consume) -> str:
    """The name of the register container a consume body reads its popped element(s) from."""
    suffix = "element" if consume.chunksize == 1 else "elements"
    return f"__dace_{consume.label}_{suffix}"


def count_name(consume: nodes.Consume) -> str:
    """The name of the scalar holding the number of elements of the current chunk.

    Bodies with ``chunksize > 1`` read it by this name.
    """
    return f"__dace_{consume.label}_numelems"


def pending_name(consume: nodes.Consume) -> str:
    """The name of the scalar holding the number of queued or running tasks of the stream."""
    return f"__dace_{consume.label}_pending"


@transformation.explicit_cf_compatible
class LowerConsumeScopes(ppl.Pass):
    """Rewrites every consume scope into a map over processing elements (PEs).

    ``consume(S, num_pes, condition, chunksize) { body(p, chunk) }`` becomes::

        map p in [0:num_pes]:                         # schedule of the consume
            nested SDFG:
                loop (do-while: n > 0 or pending > 0  |  while: not condition):
                    pop:  S -> chunk (dynamic copy: try-pop of up to chunksize elements)
                          S.num_popped -> n
                    if n > 0:
                        body(p, chunk[0:n])
                        S.task_done(n)
                    else:
                        pending = S.pending()         (no condition only)

    The stream is one global FIFO. A popped chunk stays pending until ``task_done``, and pushes count
    at once, so the default exit test (nothing queued and no task running) only holds at global
    quiescence and every PE leaves together. An explicit condition is a stop condition: the PE leaves
    as soon as it holds and keeps polling otherwise. Data a condition reads may be written by other
    PEs at the same time; that read is relaxed and only decides when a PE stops.
    """

    CATEGORY: str = "Code Generation"

    def modifies(self) -> ppl.Modifies:
        return (
            ppl.Modifies.Scopes
            | ppl.Modifies.Nodes
            | ppl.Modifies.Edges
            | ppl.Modifies.Descriptors
            | ppl.Modifies.CFG
            | ppl.Modifies.NestedSDFGs
        )

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return modified & (ppl.Modifies.Scopes | ppl.Modifies.Nodes)

    def apply_pass(self, sdfg: SDFG, _: dict[str, Any]) -> int | None:
        lowered = 0
        while True:
            target = _find_consume(sdfg)
            if target is None:
                break
            lower_consume(*target)
            lowered += 1
        return lowered or None


def _find_consume(sdfg: SDFG) -> tuple[SDFG, SDFGState, nodes.ConsumeEntry] | None:
    for nsdfg in sdfg.all_sdfgs_recursive():
        for state in nsdfg.states():
            for node in state.nodes():
                if isinstance(node, nodes.ConsumeEntry):
                    return nsdfg, state, node
    return None


def _subtree_edges(state: SDFGState, edge: MultiConnectorEdge) -> list[MultiConnectorEdge]:
    """The edges of the memlet tree below (and excluding) ``edge``."""
    tree = state.memlet_tree(edge)
    root = next(t for t in tree.traverse_children(include_self=True) if t.edge is edge)
    return [t.edge for t in root.traverse_children()]


def _unique_connector_pair(node: nodes.EntryNode, base: str) -> str:
    name = base
    index = 0
    while f"IN_{name}" in node.in_connectors or f"OUT_{name}" in node.out_connectors:
        index += 1
        name = f"{base}_{index}"
    return name


def block_level_pes(state: SDFGState, entry: nodes.ConsumeEntry) -> bool:
    """Whether each PE of a consume scope is a GPU thread block rather than a thread.

    That is the case for a device-level consume whose body maps work to the threads of a block
    (``GPU_ThreadBlock`` or ``GPU_ThreadBlock_Dynamic`` maps): the block pops a chunk together and
    its threads share the task.
    """
    if entry.consume.schedule != dtypes.ScheduleType.GPU_Device:
        return False
    thread_block = (dtypes.ScheduleType.GPU_ThreadBlock, dtypes.ScheduleType.GPU_ThreadBlock_Dynamic)
    for node in state.scope_subgraph(entry, include_entry=False, include_exit=False).nodes():
        inner = [node] if not isinstance(node, nodes.NestedSDFG) else [n for n, _ in node.sdfg.all_nodes_recursive()]
        if any(isinstance(n, nodes.MapEntry) and n.map.schedule in thread_block for n in inner):
            return True
    return False


def lower_consume(sdfg: SDFG, state: SDFGState, entry: nodes.ConsumeEntry) -> nodes.MapEntry:
    """Lowers one consume scope in place (see :class:`LowerConsumeScopes`).

    :param sdfg: The SDFG that contains ``state``.
    :param state: The state that contains the consume scope.
    :param entry: The entry node of the consume scope.
    :return: The entry node of the map that replaces the scope.
    """
    exit_node = state.exit_node(entry)
    consume: nodes.Consume = entry.consume

    stream_edge = next(e for e in state.in_edges(entry) if e.dst_conn == "IN_stream")
    stream_name = stream_edge.data.data
    stream_desc = sdfg.arrays[stream_name]
    if not isinstance(stream_desc, data.Stream):
        raise TypeError(f'Consume "{consume.label}" reads "{stream_name}", which is not a stream')
    if stream_desc.is_stream_array():
        raise NotImplementedError(f'Consume "{consume.label}" reads from a stream array, which is unsupported')

    # The chunk the body reads, in place of the stream the ``OUT_stream`` paths name. A thread block
    # that is one PE shares its chunk.
    block_pes = block_level_pes(state, entry)
    chunk_storage = dtypes.StorageType.GPU_Shared if block_pes else dtypes.StorageType.Register
    # A nested SDFG reads the element as data of its own: an array, so that it can index it (``S[0]``)
    nested_reader = any(
        isinstance(e.dst, nodes.NestedSDFG)
        for out_edge in state.out_edges_by_connector(entry, "OUT_stream")
        for e in state.memlet_tree(out_edge).leaves()
    )
    if consume.chunksize == 1 and not nested_reader:
        chunk, chunk_desc = sdfg.add_scalar(
            chunk_name(consume), stream_desc.dtype, transient=True, storage=chunk_storage, find_new_name=True
        )
    else:
        chunk, chunk_desc = sdfg.add_array(
            chunk_name(consume),
            [consume.chunksize],
            stream_desc.dtype,
            transient=True,
            storage=chunk_storage,
            find_new_name=True,
        )
    for out_edge in list(state.out_edges_by_connector(entry, "OUT_stream")):
        for inner in _subtree_edges(state, out_edge):
            inner.data.data = chunk
        for inner in [out_edge] + _subtree_edges(state, out_edge):
            if isinstance(inner.dst, nodes.NestedSDFG):
                inner_desc = copy.deepcopy(chunk_desc)
                inner_desc.transient = False
                inner.dst.sdfg.arrays[inner.dst_conn] = inner_desc
        chunk_node = state.add_access(chunk)
        state.add_nedge(entry, chunk_node, Memlet())  # Keeps the chunk inside the scope
        memlet = copy.deepcopy(out_edge.data)
        memlet.data = chunk
        if consume.chunksize == 1 or memlet.subset is None:
            memlet.subset = subsets.Range.from_array(chunk_desc)
        state.add_edge(chunk_node, None, out_edge.dst, out_edge.dst_conn, memlet)
        state.remove_edge(out_edge)

    # Replace the consume scope nodes with map scope nodes
    pe_map = nodes.Map(
        consume.label,
        [consume.pe_index],
        subsets.Range([(0, consume.num_pes - 1, 1)]),
        schedule=consume.schedule,
        debuginfo=consume.debuginfo,
    )
    pe_map.instrument = consume.instrument
    if consume.schedule in (dtypes.ScheduleType.Default, dtypes.ScheduleType.CPU_Multicore):
        # One PE per thread, like a thread per PE, so every PE polls the queue at once
        pe_map.omp_schedule = dtypes.OMPScheduleType.Static
        pe_map.omp_chunk_size = 1
        if consume.num_pes.is_Integer:
            pe_map.omp_num_threads = int(consume.num_pes)
    map_entry = nodes.MapEntry(pe_map)
    map_exit = nodes.MapExit(pe_map)
    state.add_nodes_from([map_entry, map_exit])
    stream_conn = _unique_connector_pair(entry, stream_name)

    for e in list(state.in_edges(entry)):
        if e is stream_edge:
            continue
        map_entry.add_in_connector(e.dst_conn, entry.in_connectors[e.dst_conn])
        state.add_edge(e.src, e.src_conn, map_entry, e.dst_conn, e.data)
    for e in list(state.out_edges(entry)):
        if e.src_conn is not None:
            map_entry.add_out_connector(e.src_conn, entry.out_connectors[e.src_conn])
        state.add_edge(map_entry, e.src_conn, e.dst, e.dst_conn, e.data)
    for e in list(state.in_edges(exit_node)):
        if e.dst_conn is not None:
            map_exit.add_in_connector(e.dst_conn, exit_node.in_connectors[e.dst_conn])
        state.add_edge(e.src, e.src_conn, map_exit, e.dst_conn, e.data)
    for e in list(state.out_edges(exit_node)):
        if e.src_conn is not None:
            map_exit.add_out_connector(e.src_conn, exit_node.out_connectors[e.src_conn])
        state.add_edge(map_exit, e.src_conn, e.dst, e.dst_conn, e.data)
    stream_src, stream_src_conn, stream_outer_memlet = stream_edge.src, stream_edge.src_conn, stream_edge.data
    # Keeps the map in the scope of the stream's source until the stream is connected (after nesting)
    placeholder = state.add_nedge(stream_src, map_entry, Memlet())
    state.remove_node(entry)
    state.remove_node(exit_node)

    # Nest the body: a map body is dataflow, and the PE loop is control flow
    body = state.scope_subgraph(map_entry, include_entry=False, include_exit=False)
    nsdfg_node = nest_state_subgraph(sdfg, state, body, name=f"consume_{consume.label}")
    nsdfg_node.label = f"consume_{consume.label}"
    nsdfg = nsdfg_node.sdfg
    if chunk in sdfg.arrays and not any(
        isinstance(n, nodes.AccessNode) and n.data == chunk for s in sdfg.states() for n in s.nodes()
    ):
        sdfg.remove_data(chunk, validate=False)

    # The nested SDFG pops from, retires into and maybe pushes to the stream: connect it both ways
    map_entry.add_in_connector(f"IN_{stream_conn}")
    map_entry.add_out_connector(f"OUT_{stream_conn}")
    state.add_edge(stream_src, stream_src_conn, map_entry, f"IN_{stream_conn}", copy.deepcopy(stream_outer_memlet))
    state.remove_edge(placeholder)
    inner_stream = next((e.src_conn for e in state.out_edges(nsdfg_node) if e.data.data == stream_name), None)
    if inner_stream is None:
        inner_stream = next((e.dst_conn for e in state.in_edges(nsdfg_node) if e.data.data == stream_name), None)
    if inner_stream is None:
        inner_desc = copy.deepcopy(stream_desc)
        inner_desc.transient = False
        inner_stream = nsdfg.add_datadesc(stream_name, inner_desc, find_new_name=True)
    stream_memlet = Memlet(data=stream_name, subset=copy.deepcopy(stream_outer_memlet.subset), dynamic=True)
    if inner_stream not in nsdfg_node.in_connectors:
        nsdfg_node.add_in_connector(inner_stream, force=True)
        state.add_edge(map_entry, f"OUT_{stream_conn}", nsdfg_node, inner_stream, copy.deepcopy(stream_memlet))
    else:
        # The body already read the stream through some connector of its own
        state.add_nedge(map_entry, nsdfg_node, Memlet())
    if inner_stream not in nsdfg_node.out_connectors:
        nsdfg_node.add_out_connector(inner_stream, force=True)
        out_conn = _unique_connector_pair(map_exit, stream_name)
        map_exit.add_in_connector(f"IN_{out_conn}")
        map_exit.add_out_connector(f"OUT_{out_conn}")
        state.add_edge(nsdfg_node, inner_stream, map_exit, f"IN_{out_conn}", copy.deepcopy(stream_memlet))
        state.add_edge(map_exit, f"OUT_{out_conn}", state.add_write(stream_name), None, copy.deepcopy(stream_memlet))
    for e in list(state.out_edges_by_connector(map_entry, f"OUT_{stream_conn}")):
        if e.dst is nsdfg_node and e.dst_conn != inner_stream:
            state.remove_edge(e)
    if not state.out_edges_by_connector(map_entry, f"OUT_{stream_conn}"):
        state.add_edge(map_entry, f"OUT_{stream_conn}", nsdfg_node, None, Memlet())

    # Data the stop condition reads must reach the nested SDFG
    if consume.condition is not None:
        for name in sorted(consume.condition.get_free_symbols() & sdfg.arrays.keys()):
            _connect_condition_input(sdfg, state, map_entry, nsdfg_node, name)

    _build_pe_loop(nsdfg, consume, inner_stream, chunk, chunk_desc, block_pes)
    return map_entry


def _connect_condition_input(
    sdfg: SDFG, state: SDFGState, map_entry: nodes.MapEntry, nsdfg_node: nodes.NestedSDFG, name: str
) -> None:
    """Passes container ``name`` into the nested SDFG as an input, for the stop condition to read."""
    nsdfg = nsdfg_node.sdfg
    inner = next((e.dst_conn for e in state.in_edges(nsdfg_node) if e.data.data == name), None)
    if inner is not None:
        return
    inner = next((e.src_conn for e in state.out_edges(nsdfg_node) if e.data.data == name), None)
    if inner is None:
        desc = copy.deepcopy(sdfg.arrays[name])
        desc.transient = False
        inner = nsdfg.add_datadesc(name, desc, find_new_name=True)
    memlet = Memlet.from_array(name, sdfg.arrays[name])
    memlet.dynamic = True
    conn = _unique_connector_pair(map_entry, name)
    map_entry.add_in_connector(f"IN_{conn}")
    map_entry.add_out_connector(f"OUT_{conn}")
    nsdfg_node.add_in_connector(inner, force=True)
    state.add_edge(state.add_read(name), None, map_entry, f"IN_{conn}", copy.deepcopy(memlet))
    state.add_edge(map_entry, f"OUT_{conn}", nsdfg_node, inner, memlet)


def _build_pe_loop(
    nsdfg: SDFG, consume: nodes.Consume, stream: str, chunk: str, chunk_desc: data.Data, block_pes: bool
) -> None:
    """Wraps the single body state of ``nsdfg`` in the loop one PE runs.

    With ``block_pes``, every thread of a block runs the loop, and the queue operations are the
    block-collective ones, so the block agrees on each count.
    """
    method = "block_" if block_pes else ""
    (body_state,) = nsdfg.nodes()
    nsdfg.remove_node(body_state)

    # A body reads the chunk length by name, which nesting took for a free symbol
    for name in (count_name(consume), pending_name(consume)):
        if name in nsdfg.symbols:
            nsdfg.remove_symbol(name)
        if nsdfg.parent_nsdfg_node is not None:
            nsdfg.parent_nsdfg_node.symbol_mapping.pop(name, None)

    count, _ = nsdfg.add_scalar(count_name(consume), dtypes.int32, transient=True, storage=dtypes.StorageType.Register)
    if consume.condition is None:
        pending, _ = nsdfg.add_scalar(
            pending_name(consume), dtypes.int32, transient=True, storage=dtypes.StorageType.Register
        )
        loop = LoopRegion(f"consume_{consume.label}", f"{count} > 0 or {pending} > 0", inverted=True)
    else:
        loop = LoopRegion(f"consume_{consume.label}", CodeBlock(f"not ({consume.condition.as_string})"))
    nsdfg.add_node(loop, is_start_block=True)

    stream_desc = nsdfg.arrays[stream]
    stream_subset = subsets.Range.from_array(stream_desc)

    # Pop: a non-blocking copy of up to ``chunksize`` elements, and their number
    pop = loop.add_state(f"pop_{consume.label}", is_start_block=True)
    stream_read = pop.add_read(stream)
    stream_read.add_out_connector(NUM_POPPED_CONNECTOR)
    pop.add_edge(
        stream_read,
        None,
        pop.add_write(chunk),
        None,
        Memlet(
            data=stream,
            subset=copy.deepcopy(stream_subset),
            other_subset=subsets.Range.from_array(chunk_desc),
            dynamic=True,
        ),
    )
    pop.add_edge(stream_read, NUM_POPPED_CONNECTOR, pop.add_write(count), None, Memlet(data=count, subset="0"))

    # Run the task, then retire its chunk
    has_work = ConditionalBlock(f"has_work_{consume.label}")
    loop.add_node(has_work)
    loop.add_edge(pop, has_work, InterstateEdge())
    task = ControlFlowRegion(f"task_{consume.label}")
    has_work.add_branch(CodeBlock(f"{count} > 0"), task)
    if consume.condition is None:
        # Only an empty pop needs the pending count (the loop condition reads it only then). It is
        # read after the pop, but either order is sound: once zero it stays zero.
        idle = ControlFlowRegion(f"idle_{consume.label}")
        has_work.add_branch(None, idle)
        poll = idle.add_state(f"poll_{consume.label}", is_start_block=True)
        tasklet = poll.add_tasklet(
            f"pending_{consume.label}",
            {"__stream"},
            {"__pending"},
            f"__pending = __stream.{method}pending();",
            language=dtypes.Language.CPP,
        )
        poll.add_edge(poll.add_read(stream), None, tasklet, "__stream", Memlet(data=stream, subset="0", dynamic=True))
        poll.add_edge(tasklet, "__pending", poll.add_write(pending), None, Memlet(data=pending, subset="0"))
    task.add_node(body_state, is_start_block=True)
    done = task.add_state(f"done_{consume.label}")
    task.add_edge(body_state, done, InterstateEdge())
    # A dynamic memlet hands the tasklet the stream object itself; retiring writes the stream
    tasklet = done.add_tasklet(
        f"task_done_{consume.label}",
        {"__count"},
        {"__stream"},
        f"__stream.{method}task_done(__count);",
        language=dtypes.Language.CPP,
    )
    done.add_edge(done.add_read(count), None, tasklet, "__count", Memlet(data=count, subset="0"))
    done.add_edge(tasklet, "__stream", done.add_write(stream), None, Memlet(data=stream, subset="0", dynamic=True))
    nsdfg.reset_cfg_list()
