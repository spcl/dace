# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``LastWriterDFSStreamScheduler`` -- v1 stream-scheduling strategy.

DFS chain inheritance within each ``SDFGState`` + ``LastWriter`` forwarding
across blocks of every ``ControlFlowRegion``. Loops are scheduled as
black boxes with a fixed-point check on the body's stream signature;
conditionals merge their branches at the join.

Sync points are materialised as the three sync libnodes
(:class:`StreamEventRecordLibraryNode`,
:class:`StreamWaitEventLibraryNode`,
:class:`StreamSynchronizeLibraryNode`) and hoisted by
:func:`hoist_redundant_syncs` to elide intra-region redundancy.

Events are first-class in the IR: a ``gpu_events`` transient is
allocated at the root SDFG (sized by the unique-producer event count),
and every ``StreamEventRecord`` / ``StreamWaitEvent`` libnode reads its
event handle from a ``gpu_events[i]`` ``AccessNode`` -- symmetric to how
streams are wired.
"""
from typing import Dict

import dace
from dace import SDFG, properties
from dace.config import Config
from dace.sdfg import nodes
from dace.transformation import transformation

from dace.transformation.passes.gpu_specialization.gpu_stream_scheduling import GPUStreamSchedulingStrategy
from dace.transformation.passes.gpu_specialization.helpers.gpu_helpers import (EVENT_CONNECTOR, STREAM_CONNECTOR,
                                                                               exit_anchor_for,
                                                                               get_gpu_event_array_name,
                                                                               get_gpu_stream_array_name)
from dace.transformation.passes.gpu_specialization.stream_lowering_helpers import (allocate_event_array,
                                                                                   insert_state_end_syncs)

from .libnodes import (StreamEventRecordLibraryNode, StreamSynchronizeLibraryNode, StreamWaitEventLibraryNode)
from .scheduler import SchedulingContext, schedule_cfr
from .hoist import hoist_redundant_syncs


@properties.make_properties
@transformation.explicit_cf_compatible
class LastWriterDFSStreamScheduler(GPUStreamSchedulingStrategy):
    """v1 stream-scheduling strategy: DFS intra-state, LastWriter inter-block.

    Inserts:
      - :class:`StreamEventRecordLibraryNode` after every producer that
        has at least one cross-stream consumer.
      - :class:`StreamWaitEventLibraryNode` before every cross-stream
        consumer.
      - :class:`StreamSynchronizeLibraryNode` before every interstate
        edge whose condition or assignment reads a GPU-resident array.

    Each unique producer gets a unique event-id slot in the SDFG-level
    ``gpu_events`` array; consumers reference the same slot via memlets.
    """

    def __init__(self):
        self._max_concurrent_streams = int(Config.get("compiler", "cuda", "max_concurrent_streams"))
        self._last_ctx: SchedulingContext = None  # type: ignore[assignment]

    def assign_streams(self, sdfg: SDFG) -> Dict[nodes.Node, int]:
        ctx = SchedulingContext(cap_K=self._max_concurrent_streams)
        schedule_cfr(sdfg, dict(), ctx)
        hoist_redundant_syncs(sdfg, ctx)
        self._last_ctx = ctx
        return ctx.assignments

    def insert_sync_tasklets(self, sdfg: SDFG, assignments: Dict[nodes.Node, int]):
        ctx = self._last_ctx
        assert ctx is not None, "insert_sync_tasklets called before assign_streams"

        # Allocate the SDFG-level ``gpu_events`` array sized by the
        # high-water mark of the event-id counter. ``producer_event``
        # is a cache (same producer's repeat calls return the same id),
        # so its dict length undercounts events allocated through a
        # scratch loop / conditional pass that committed their ids to
        # ``cross_stream_edges`` without keeping the producer key.
        num_events = ctx.next_event_id
        if num_events > 0:
            allocate_event_array(sdfg, num_events)

        for state, src, dst, event_id in ctx.cross_stream_edges:
            if state is None or src is None or dst is None:
                continue
            src_stream = assignments.get(src)
            dst_stream = assignments.get(dst)
            if src_stream is None or dst_stream is None:
                continue
            _insert_record_then_wait(state, src, dst, event_id, src_stream, dst_stream)

        for producer_state, producer, src_stream, consumer_state, consumer, dst_stream, event_id in \
                ctx.inter_state_cross_stream:
            _insert_inter_state_record_wait(producer_state, producer, src_stream, consumer_state, consumer, dst_stream,
                                            event_id)

        for ie, stream_id in ctx.interstate_host_reads:
            _insert_host_sync_at_interstate(sdfg, ie, stream_id)

        # End-of-SDFG sync: every program-sink state synchronizes all streams
        # that the scheduler bound any node to. Handles the jacobi case
        # (1 stream in a loop -> 1 sync at exit) and the "unsynchronized
        # kernels at SDFG end" requirement. cudaDeviceSynchronize in the
        # cuda exit hook only runs once at program-lifetime end -- callers
        # invoke ``__program_*`` multiple times and expect the output
        # to be ready immediately on return.
        _insert_program_sink_syncs(sdfg, assignments)


def _event_slot_memlet(event_id: int) -> dace.Memlet:
    """Memlet referencing one slot of the ``gpu_events`` array."""
    return dace.Memlet(f"{get_gpu_event_array_name()}[{event_id}]")


def _stream_slot_memlet(stream_id: int) -> dace.Memlet:
    """Memlet referencing one slot of the ``gpu_streams`` array."""
    return dace.Memlet(f"{get_gpu_stream_array_name()}[{stream_id}]")


def _insert_record_then_wait(state, src, dst, event_id: int, src_stream: int, dst_stream: int):
    """Insert a ``StreamEventRecord`` after ``src`` and a
    ``StreamWaitEvent`` before ``dst``, wiring stream + event slots
    via memlets from the matching SDFG-level arrays.

    ``src`` / ``dst`` may be MapEntries. The dependency chain must hang off
    the MapExit (for ``src``) and feed into the MapEntry (for ``dst``);
    adding an out-edge directly from a MapEntry to a peer node would be a
    scope-crossing edge and corrupts codegen ordering.

    The codegen reads the stream / event handles through the typed
    connectors -- the SDFG explicitly shows which stream-slot records
    and which stream-slot waits, plus the event slot that connects them.
    """
    src_anchor = exit_anchor_for(state, src)
    dst_anchor = dst  # MapEntry / LibraryNode / Tasklet -- peer-level

    record = StreamEventRecordLibraryNode(name=f"record_e{event_id}_after_{getattr(src, 'label', 'src')}")
    wait = StreamWaitEventLibraryNode(name=f"wait_e{event_id}_before_{getattr(dst, 'label', 'dst')}")
    state.add_node(record)
    state.add_node(wait)

    # Event slot wiring: same gpu_events[event_id] feeds both libnodes.
    events_access = state.add_access(get_gpu_event_array_name())
    state.add_edge(events_access, None, record, EVENT_CONNECTOR, _event_slot_memlet(event_id))
    state.add_edge(events_access, None, wait, EVENT_CONNECTOR, _event_slot_memlet(event_id))

    # Stream slot wiring: record runs on src's stream, wait on dst's stream.
    streams_for_record = state.add_access(get_gpu_stream_array_name())
    state.add_edge(streams_for_record, None, record, STREAM_CONNECTOR, _stream_slot_memlet(src_stream))
    streams_for_wait = state.add_access(get_gpu_stream_array_name())
    state.add_edge(streams_for_wait, None, wait, STREAM_CONNECTOR, _stream_slot_memlet(dst_stream))

    # Dependency chain: src_anchor -> record -> wait -> dst_anchor (empty memlets,
    # used only for codegen ordering).
    state.add_edge(src_anchor, None, record, None, dace.Memlet())
    state.add_edge(record, None, wait, None, dace.Memlet())
    state.add_edge(wait, None, dst_anchor, None, dace.Memlet())


def _insert_inter_state_record_wait(producer_state, producer, src_stream: int, consumer_state, consumer,
                                    dst_stream: int, event_id: int):
    """Emit a producer-side ``StreamEventRecord`` and a consumer-side ``StreamWaitEvent``
    sharing one event id, when producer + consumer live in different states.

    The record hangs off the producer's MapExit (peer-anchor) in the producer's state;
    the wait hangs off the consumer's MapEntry (peer-anchor) in the consumer's state.
    """
    src_anchor = exit_anchor_for(producer_state, producer)
    record = StreamEventRecordLibraryNode(name=f"record_e{event_id}_after_{getattr(producer, 'label', 'src')}")
    producer_state.add_node(record)

    events_for_record = producer_state.add_access(get_gpu_event_array_name())
    producer_state.add_edge(events_for_record, None, record, EVENT_CONNECTOR, _event_slot_memlet(event_id))
    streams_for_record = producer_state.add_access(get_gpu_stream_array_name())
    producer_state.add_edge(streams_for_record, None, record, STREAM_CONNECTOR, _stream_slot_memlet(src_stream))
    producer_state.add_edge(src_anchor, None, record, None, dace.Memlet())

    wait = StreamWaitEventLibraryNode(name=f"wait_e{event_id}_before_{getattr(consumer, 'label', 'dst')}")
    consumer_state.add_node(wait)

    events_for_wait = consumer_state.add_access(get_gpu_event_array_name())
    consumer_state.add_edge(events_for_wait, None, wait, EVENT_CONNECTOR, _event_slot_memlet(event_id))
    streams_for_wait = consumer_state.add_access(get_gpu_stream_array_name())
    consumer_state.add_edge(streams_for_wait, None, wait, STREAM_CONNECTOR, _stream_slot_memlet(dst_stream))
    consumer_state.add_edge(wait, None, consumer, None, dace.Memlet())


def _insert_program_sink_syncs(sdfg: SDFG, assignments: Dict[nodes.Node, int]):
    """Sync every used stream at the end of every state that uses streams.

    Reuses :func:`insert_state_end_syncs` from ``stream_lowering_helpers`` --
    one fused ``cudaStreamSynchronize`` tasklet per qualifying state with one
    in-connector per stream that the state's nodes are bound to. This both
    makes the host's read of GPU outputs safe after ``__program_*`` returns
    and covers cross-state dependencies that the data-flow analysis can miss
    (e.g. NSDFG-internal writes that propagate through opaque connectors).
    Same-stream sequencing is preserved by submit order; only host-side
    waits get materialized -- adjacent same-stream kernels do not pay a
    sync cost because the fused sync only emits ``cudaStreamSynchronize``
    for streams the state actually uses.
    """
    per_state_streams: Dict["dace.SDFGState", set] = {}
    for n, sid in assignments.items():
        st = _owning_state(sdfg, n)
        if st is None:
            continue
        per_state_streams.setdefault(st, set()).add(sid)

    if per_state_streams:
        insert_state_end_syncs(sdfg, per_state_streams, assignments)


def _owning_state(root: SDFG, node):
    """The state that directly contains ``node`` (regardless of SDFG nesting)."""
    for cur in root.all_sdfgs_recursive():
        for st in cur.states():
            if node in st.nodes():
                return st
    return None


def _insert_host_sync_at_interstate(sdfg: SDFG, interstate_edge, stream_id: int):
    """Insert a ``StreamSynchronize`` libnode at the end of the
    interstate edge's source state (host-side wait on the writer's
    stream before the condition / assignment evaluates).
    """
    if interstate_edge is None:
        return
    src_block = getattr(interstate_edge, "src", None)
    if not isinstance(src_block, dace.SDFGState):
        return
    sync = StreamSynchronizeLibraryNode(name=f"sync_stream_{stream_id}")
    src_block.add_node(sync)
    streams_access = src_block.add_access(get_gpu_stream_array_name())
    src_block.add_edge(streams_access, None, sync, STREAM_CONNECTOR, _stream_slot_memlet(stream_id))
    for sink in list(src_block.sink_nodes()):
        if sink is sync:
            continue
        if sink is streams_access:
            continue
        src_block.add_edge(sink, None, sync, None, dace.Memlet())
