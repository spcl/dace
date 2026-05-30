# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Stream-assignment walker: DFS chain inheritance per state + LastWriter
forwarding across blocks of every ``ControlFlowRegion``.

Single-file scheduler -- one function per block class (state, loop,
conditional, nested CFR) plus the per-state DFS. Nothing is wrapped in
a class; the walker threads a ``SchedulingContext`` dataclass through
the recursion. Aligns with the IEC / copy-libnode design: a small set
of focused helpers, no strategy/visitor scaffolding.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from dace import SDFG, SDFGState
from dace.sdfg import nodes
from dace.sdfg.state import ConditionalBlock, ControlFlowBlock, ControlFlowRegion, LoopRegion
from dace.sdfg.graph import Edge
from dace.transformation.passes.gpu_specialization.helpers.gpu_helpers import (is_gpu_relevant_node,
                                                                               is_gpu_stream_consumer)

from .last_writer import LastWriter, StreamEventToken, lastwriter_stream_join, stream_signatures_match

# ----------------------------------------------------------------------------
# Context object (mutable; threaded through the walk).
# ----------------------------------------------------------------------------


@dataclass
class SchedulingContext:
    """All the state the scheduler accumulates while walking the SDFG."""
    assignments: Dict[nodes.Node, int] = field(default_factory=dict)
    # Producer -> event_id. Allocated lazily: a producer gets an event
    # slot the first time its output is consumed cross-stream. Each
    # slot is a unique integer identifying a position in the SDFG-level
    # ``gpu_events`` array.
    producer_event: Dict[nodes.Node, int] = field(default_factory=dict)
    # Each entry: (state, src, dst, event_id) -- emitted as one
    # EventRecord on src's stream + one StreamWaitEvent on dst's stream.
    cross_stream_edges: List[Tuple[SDFGState, nodes.Node, nodes.Node, int]] = field(default_factory=list)
    # Each entry: (producer_state, producer, src_stream, consumer_state, consumer, dst_stream, event_id)
    # -- EventRecord in ``producer_state`` on ``src_stream`` + StreamWaitEvent in
    # ``consumer_state`` on ``dst_stream``. Closes the inter-state sync gap that
    # ``cross_stream_edges`` (intra-state only) can't express.
    inter_state_cross_stream: List[Tuple[SDFGState, nodes.Node, int, SDFGState, nodes.Node, int, int]] = field(
        default_factory=list)
    # Each entry: (interstate_edge, stream_id) -- one StreamSynchronize
    # at the interstate-edge boundary before the condition evaluates.
    interstate_host_reads: List[Tuple[Edge, int]] = field(default_factory=list)
    # Loops whose body's stream signature didn't reach a fixed point;
    # codegen emits per-iteration events for these.
    per_iteration_loops: Set[LoopRegion] = field(default_factory=set)
    cap_K: int = 0
    next_stream_id: int = 0
    next_event_id: int = 0

    def fresh_stream(self) -> int:
        sid = self.next_stream_id
        self.next_stream_id += 1
        return sid % self.cap_K if self.cap_K > 0 else sid

    def fresh_event(self) -> int:
        eid = self.next_event_id
        self.next_event_id += 1
        return eid

    def event_for_producer(self, producer: nodes.Node) -> int:
        """Lazy allocation: each producer gets a unique event id only on
        first cross-stream demand. Returned id is stable for subsequent
        consumers of the same producer."""
        eid = self.producer_event.get(producer)
        if eid is None:
            eid = self.fresh_event()
            self.producer_event[producer] = eid
        return eid


# ----------------------------------------------------------------------------
# Node-type predicates.
# ----------------------------------------------------------------------------


def _is_scheduled(node, sdfg: SDFG, state: SDFGState) -> bool:
    """A node is *scheduled* if it consumes a CUDA stream -- the scheduler
    needs to bind one. AccessNodes are data carriers and excluded.

    A ``NestedSDFG`` whose body contains any GPU-relevant work behaves as a
    consumer at this level: it inherits the parent's stream, and every
    stream-consuming node inside it shares that stream (propagated by
    :func:`_propagate_stream_into_nested`).
    """
    if is_gpu_stream_consumer(node, sdfg, state):
        return True
    if isinstance(node, nodes.NestedSDFG):
        for ninner, parent_state in node.sdfg.all_nodes_recursive():
            if is_gpu_relevant_node(ninner, parent_state.sdfg, parent_state):
                return True
    return False


def _propagate_stream_into_nested(nsdfg_node: nodes.NestedSDFG, stream_id: int, ctx: 'SchedulingContext'):
    """Every stream-consuming node inside ``nsdfg_node.sdfg`` shares the parent's stream id.

    No cross-stream events get manufactured inside the NestedSDFG body -- conservative but
    safe; matches the legacy ``NaiveGPUStreamScheduler`` ``in_nested_sdfg=True`` branch.
    """
    for inner_node, parent_state in nsdfg_node.sdfg.all_nodes_recursive():
        if isinstance(parent_state, SDFGState) and is_gpu_stream_consumer(inner_node, parent_state.sdfg, parent_state):
            ctx.assignments[inner_node] = stream_id


def _read_data_names(node, state: SDFGState) -> List[str]:
    """Data array names this node reads from, via in-edge memlets that
    originate at an AccessNode (after the optional MapEntry).

    Empty memlets and views are skipped.
    """
    result: List[str] = []
    for e in state.in_edges(node):
        if e.data.is_empty():
            continue
        # Walk back through MapEntry boundaries until an AccessNode is found.
        for path_edge in state.memlet_path(e):
            if isinstance(path_edge.src, nodes.AccessNode):
                result.append(path_edge.src.data)
                break
    return result


def _written_data_names(node, state: SDFGState) -> List[str]:
    """Data array names this node writes to, via out-edge memlets that
    terminate at an AccessNode (after the optional MapExit).

    For a ``MapEntry`` the writes flow through its matching ``MapExit``;
    walk out-edges from there. Other scheduled-relevant nodes write
    directly.
    """
    if isinstance(node, nodes.MapEntry):
        try:
            exit_node = state.exit_node(node)
        except (KeyError, StopIteration):
            return []
        src_node = exit_node
    else:
        src_node = node
    result: List[str] = []
    for e in state.out_edges(src_node):
        if e.data.is_empty():
            continue
        for path_edge in state.memlet_path(e):
            if isinstance(path_edge.dst, nodes.AccessNode):
                result.append(path_edge.dst.data)
                break
    return result


# ----------------------------------------------------------------------------
# Per-state DFS chain inheritance.
# ----------------------------------------------------------------------------


def schedule_state(state: SDFGState, last_writer: LastWriter, ctx: SchedulingContext) -> LastWriter:
    """DFS chain inheritance, seeded by the entering ``LastWriter``.

    Semantics (the legacy ``_compute_cudastreams`` model):
      - Source nodes (no scheduled predecessor in this state) inherit
        the stream of their first input via ``last_writer`` if available,
        otherwise allocate a fresh stream.
      - A non-source scheduled node visits its scheduled predecessors:
        the *first* predecessor's stream becomes its inherited stream;
        every subsequent predecessor reading from a *different* stream
        contributes a cross-stream event.
      - When a predecessor has already donated its stream to one
        successor (its ``childpath`` is used), the next successor forks
        onto a fresh stream -- this is what produces parallel sibling
        chains in the diamond / fan-out case.
    """
    sdfg = state.sdfg
    # Only schedule **top-level** scheduled nodes (immediate children of the state, not
    # nested inside a Map/Consume scope). Inner-scope nodes share their outer scope's
    # kernel stream and have no independent identity at the host-side scheduler level;
    # binding them would create scope-crossing dependency edges from the inserted
    # sync libnodes, which corrupts the scope tree (cyclic ``scope_children``).
    scope_dict = state.scope_dict()
    scheduled = [n for n in state.nodes() if _is_scheduled(n, sdfg, state) and scope_dict.get(n) is None]
    if not scheduled:
        return last_writer

    order = _topo_order_scheduled(state, scheduled)
    scheduled_set = set(scheduled)

    out = dict(last_writer)
    # Track which scheduled nodes have already chained one child.
    childpath_used: Dict[nodes.Node, bool] = {}
    # Per-data: which scheduled node in this state wrote it most recently.
    writer_node: Dict[str, nodes.Node] = {}

    for n in order:
        reads = _read_data_names(n, state)
        writes = _written_data_names(n, state)
        # Resolve each read-after-write and write-after-write to its prior
        # writer + stream + producer-token. RAW + WAW both produce real
        # dependencies; missing WAW causes overwrite races when two writers
        # of the same array end up on different streams (e.g. ``copyin``
        # writes ``gpu_A``, then a zero-fill kernel ``gpu_A[i]=0`` runs
        # concurrently). Contributions carry the full StreamEventToken (or
        # None) so cross-state syncs can demand-allocate events on the
        # original producer.
        contributions: List[Tuple[Optional[nodes.Node], int, Optional[StreamEventToken]]] = []
        seen_deps: Set[str] = set()
        for d in list(reads) + list(writes):
            if d in seen_deps:
                continue
            seen_deps.add(d)
            if d in writer_node:
                pred = writer_node[d]
                contributions.append((pred, ctx.assignments[pred], None))
            elif d in out and out[d].stream_id >= 0:
                tok = out[d]
                contributions.append((None, tok.stream_id, tok))

        if not contributions:
            # No dependency -> source node. Default to stream 0 instead of
            # allocating a fresh stream; independent sources sequentialise via
            # submit order on stream 0, which is correct and minimises the
            # stream count. Forks happen only at real diamond patterns (a
            # predecessor with ``childpath_used==True`` below).
            chosen = 0
            ctx.next_stream_id = max(ctx.next_stream_id, 1)
        else:
            # First predecessor donates its stream (forking if it has
            # already donated to a previous sibling).
            first_pred, first_stream, _ = contributions[0]
            if first_pred is not None and childpath_used.get(first_pred, False):
                chosen = ctx.fresh_stream()
            else:
                chosen = first_stream
                if first_pred is not None:
                    childpath_used[first_pred] = True

        ctx.assignments[n] = chosen

        # NestedSDFG: propagate the chosen stream to every inner consumer so
        # the codegen can wire ``__dace_current_stream`` inside the nested
        # function. The inner body shares the parent's stream end-to-end.
        if isinstance(n, nodes.NestedSDFG):
            _propagate_stream_into_nested(n, chosen, ctx)

        # Cross-stream events: one per contribution on a different stream.
        # Lazy allocation -- the producer gets a unique event slot only on
        # first cross-stream demand. ``ctx.event_for_producer`` returns
        # the same id for every subsequent consumer of the same producer.
        for pred, pred_stream, prev_tok in contributions:
            if pred_stream == chosen:
                continue
            if pred is not None:
                # Intra-state cross-stream: record + wait both in ``state``.
                event_id = ctx.event_for_producer(pred)
                ctx.cross_stream_edges.append((state, pred, n, event_id))
            elif prev_tok is not None and prev_tok.producer is not None and prev_tok.producer_state is not None:
                # Inter-state cross-stream: record runs in the producer's state
                # (after the producer), wait runs in this state (before n).
                # The pair shares one event id allocated on the producer node.
                event_id = ctx.event_for_producer(prev_tok.producer)
                ctx.inter_state_cross_stream.append((prev_tok.producer_state, prev_tok.producer, pred_stream, state, n,
                                                     chosen, event_id))

        # Update outputs. The token's event_id is left ``None`` until a
        # later consumer cross-streams from this node and asks for one
        # via ``ctx.event_for_producer``. The producer + producer_state
        # are recorded so the demand-driven allocation can pin the record
        # site in the producer's state (not the consumer's).
        for d in writes:
            out[d] = StreamEventToken(chosen, event_id=None, producer=n, producer_state=state)
            writer_node[d] = n

    return out


def _topo_order_scheduled(state: SDFGState, scheduled: List[nodes.Node]) -> List[nodes.Node]:
    """Topological order over scheduled-only nodes, derived from the
    state's actual edge graph (not data-name analysis).

    Two distinct AccessNodes can share a name -- e.g. ``x1`` as a host
    input + ``x1`` as the host destination of an output copy -- so
    data-name-driven dep inference invents a false cycle between them.
    The state's edges are already acyclic by SDFG invariant, so a plain
    topological sort gives the canonical order; we then drop everything
    that isn't scheduled to get the scheduled-only sequence.
    """
    scheduled_set = set(scheduled)
    try:
        from dace.sdfg.utils import dfs_topological_sort
        sources = state.source_nodes()
        topo_iter = dfs_topological_sort(state, sources=sources)
    except (RuntimeError, KeyError, AttributeError):
        topo_iter = state.nodes()
    return [n for n in topo_iter if n in scheduled_set]


def _find_writer_in_state(state: SDFGState,
                          data: str,
                          scheduled_set: Set[nodes.Node],
                          exclude: Optional[nodes.Node] = None) -> Optional[nodes.Node]:
    """Find a scheduled writer of ``data`` inside this state, optionally
    skipping ``exclude`` (the querying node itself in the in-place-update case)."""
    for n in scheduled_set:
        if n is exclude:
            continue
        if data in _written_data_names(n, state):
            return n
    return None


def _find_writer_node(state: SDFGState, data: str, assignments: Dict[nodes.Node, int]) -> Optional[nodes.Node]:
    """Best-effort: a writer node assigned in this state. For inter-state
    inheritance the writer lives in a prior block and is not in
    ``assignments`` keyed for this state -- we record ``None`` then."""
    for n, _ in assignments.items():
        if data in _written_data_names(state, n) if state.nodes() and n in state.nodes() else False:
            return n
    return None


# ----------------------------------------------------------------------------
# Loop with fixed-point check + widening.
# ----------------------------------------------------------------------------


def schedule_loop(loop: LoopRegion, last_writer: LastWriter, ctx: SchedulingContext) -> LastWriter:
    """Schedule a loop as a black box.

    Iteration semantics: every iteration applies the body's effect to
    ``LastWriter``. For symbolic iteration counts the only sound exit
    state is the *fixed point* of that effect.

    Strategy:
      1. Schedule the body once seeded by ``last_writer``  →  ``after_iter0``.
      2. Schedule the body a second time seeded by ``after_iter0`` on a
         scratch context (don't commit).
      3. If stream signatures match: fixed point in 1 iteration. Commit
         iter-0's assignment, return ``after_iter0``.
      4. Otherwise widen: ``join(last_writer, after_iter1)``. Re-schedule.
      5. If still no fixed point: pin iter-0's assignment and mark the
         loop for per-iteration sync.
    """
    # Iter-0, iter-1, and widened all run on scratch contexts; only the accepted
    # candidate is merged into the real ``ctx``. Without scratch isolation an
    # earlier probe's edges would stay committed even after widening replaced it,
    # leaving duplicate cross-stream syncs (and event-id leaks via
    # ``cross_stream_edges`` whose ``producer_event`` entry never made it back to
    # the real context).
    s0 = _scratch_ctx(ctx)
    after_iter0 = _schedule_cfr_with_scratch(loop, last_writer, s0)

    s1 = _scratch_ctx(ctx)
    after_iter1 = _schedule_cfr_with_scratch(loop, after_iter0, s1)

    if stream_signatures_match(after_iter0, after_iter1):
        _merge_scratch_into(ctx, s0)
        if _condition_reads_gpu(loop, after_iter0):
            for stream_id in _gpu_streams_read_by_condition(loop, after_iter0):
                ctx.interstate_host_reads.append((None, stream_id))  # type: ignore[arg-type]
        return after_iter0

    widened = lastwriter_stream_join(last_writer, after_iter1)
    sw = _scratch_ctx(ctx)
    after_widened = _schedule_cfr_with_scratch(loop, widened, sw)

    sv = _scratch_ctx(ctx)
    after_widened2 = _schedule_cfr_with_scratch(loop, after_widened, sv)
    if stream_signatures_match(after_widened, after_widened2):
        _merge_scratch_into(ctx, sw)
        return after_widened

    # Fixed point not reached: commit iter-0 and accept per-iter events inside the body.
    _merge_scratch_into(ctx, s0)
    ctx.per_iteration_loops.add(loop)
    return after_iter0


# ----------------------------------------------------------------------------
# Conditional with branch-merge at the join.
# ----------------------------------------------------------------------------


def schedule_conditional(cond: ConditionalBlock, last_writer: LastWriter, ctx: SchedulingContext) -> LastWriter:
    """Schedule every branch independently from the same entry
    ``LastWriter``; merge per-data at the join.

    Per-data merge:
      - All branches assigned the same stream → keep it.
      - Streams disagree → pick a join stream (heuristic: the entry's
        stream if present, else the most-used across branches), and
        emit a cross-stream event for every disagreeing branch.
      - Data written in only some branches → fall back to the entry
        token for the unwritten branches (the consumer waits on the
        entry stream regardless).
    """
    branch_results: List[LastWriter] = []
    for _cond_expr, branch_cfr in cond.branches:
        branch_results.append(schedule_cfr(branch_cfr, dict(last_writer), ctx))

    if not branch_results:
        return last_writer

    out = dict(last_writer)
    all_keys = set().union(*[lw.keys() for lw in branch_results])
    for d in all_keys:
        tokens = [lw.get(d, last_writer.get(d)) for lw in branch_results]
        stream_ids = {t.stream_id for t in tokens if t is not None}
        if len(stream_ids) == 1:
            sole_token = next(t for t in tokens if t is not None)
            out[d] = StreamEventToken(next(iter(stream_ids)), event_id=None, producer=sole_token.producer)
            continue
        join_stream = (last_writer[d].stream_id if d in last_writer else max(stream_ids))
        # The disagreeing branches need an event-wait on the join stream;
        # demand-allocate an event for each branch's producer.
        for t in tokens:
            if t is None or t.stream_id == join_stream or t.producer is None:
                continue
            event_id = ctx.event_for_producer(t.producer)
            # No state context at this point (cross-block); the
            # interstate / hoist passes attach the libnodes downstream.
            ctx.cross_stream_edges.append((None, t.producer, None, event_id))  # type: ignore[arg-type]
        # Pick any branch's producer as the canonical writer on the join stream.
        join_producer = next((t.producer for t in tokens if t is not None and t.stream_id == join_stream), None)
        out[d] = StreamEventToken(join_stream, event_id=None, producer=join_producer)
    return out


# ----------------------------------------------------------------------------
# CFR walker + interstate-edge analysis.
# ----------------------------------------------------------------------------


def schedule_cfr(cfr: ControlFlowRegion, last_writer: LastWriter, ctx: SchedulingContext) -> LastWriter:
    """Walk the CFR's blocks (a line graph) in topological order and
    dispatch on block type."""
    blocks = _topo_blocks(cfr)
    for block in blocks:
        # Interstate sync triggers for the edges feeding this block.
        for ie in cfr.in_edges(block) if hasattr(cfr, "in_edges") else []:
            _record_interstate_sync_triggers(ie, last_writer, ctx)
        # Block-type dispatch.
        if isinstance(block, SDFGState):
            last_writer = schedule_state(block, last_writer, ctx)
        elif isinstance(block, LoopRegion):
            last_writer = schedule_loop(block, last_writer, ctx)
        elif isinstance(block, ConditionalBlock):
            last_writer = schedule_conditional(block, last_writer, ctx)
        elif isinstance(block, ControlFlowRegion):
            last_writer = schedule_cfr(block, last_writer, ctx)
        # Other block kinds (Return, Break, Continue) are no-ops for stream state.
    return last_writer


def _topo_blocks(cfr: ControlFlowRegion) -> List[ControlFlowBlock]:
    """Topological order of CFR blocks (CFRs are line graphs of blocks)."""
    if hasattr(cfr, "topological_sort"):
        return list(cfr.topological_sort())
    if hasattr(cfr, "nodes"):
        return list(cfr.nodes())
    return []


def _record_interstate_sync_triggers(ie: Edge, last_writer: LastWriter, ctx: SchedulingContext):
    """If the interstate edge's condition or assignments read a
    GPU-resident array or symbol, record a ``StreamSynchronize`` at the
    edge boundary on the writer's stream."""
    cond = getattr(ie.data, "condition", None)
    assigns = getattr(ie.data, "assignments", {}) or {}
    syms: Set[str] = set()
    if cond is not None and not _is_unconditional(cond):
        try:
            syms |= {str(s) for s in cond.get_free_symbols()}
        except (AttributeError, RuntimeError):
            pass
    for rhs in assigns.values():
        syms |= _free_symbols_in_rhs(rhs)
    for name in syms:
        tok = last_writer.get(name)
        if tok is not None and tok.event_id is not None:
            ctx.interstate_host_reads.append((ie, tok.stream_id))


def _is_unconditional(cond) -> bool:
    """An interstate edge condition that is the constant ``True`` /
    empty -- nothing to sync for."""
    if cond is None:
        return True
    txt = getattr(cond, "as_string", None)
    if callable(txt):
        txt = txt()
    return txt is None or str(txt).strip() in ("", "1", "True", "true")


def _free_symbols_in_rhs(rhs) -> Set[str]:
    """Symbol names referenced on the RHS of an interstate assignment."""
    try:
        return {str(s) for s in rhs.get_free_symbols()}
    except (AttributeError, RuntimeError):
        try:
            return set(map(str, rhs.free_symbols))
        except AttributeError:
            return set()


# ----------------------------------------------------------------------------
# Loop helpers (sketched here, kept private).
# ----------------------------------------------------------------------------


def _scratch_ctx(ctx: SchedulingContext) -> SchedulingContext:
    """A fresh context that shares the caller's stream/event counters
    (allocations from inside a scratch pass are *not* later folded in
    unless we explicitly merge -- the scheduler uses scratch to probe
    convergence before committing)."""
    return SchedulingContext(cap_K=ctx.cap_K, next_stream_id=ctx.next_stream_id, next_event_id=ctx.next_event_id)


def _merge_scratch_into(target: SchedulingContext, scratch: SchedulingContext):
    """Commit a scratch context's results into the real context."""
    target.assignments.update(scratch.assignments)
    target.cross_stream_edges.extend(scratch.cross_stream_edges)
    target.inter_state_cross_stream.extend(scratch.inter_state_cross_stream)
    target.interstate_host_reads.extend(scratch.interstate_host_reads)
    target.per_iteration_loops |= scratch.per_iteration_loops
    # ``producer_event`` is a (producer -> id) cache. Merging it keeps the same producer
    # mapped to the same id if a future query hits it; not merging would re-allocate a
    # fresh id and orphan the scratch's id committed to ``cross_stream_edges``.
    for prod, eid in scratch.producer_event.items():
        target.producer_event.setdefault(prod, eid)
    target.next_stream_id = max(target.next_stream_id, scratch.next_stream_id)
    target.next_event_id = max(target.next_event_id, scratch.next_event_id)


def _schedule_cfr_with_scratch(loop_or_cfr, last_writer: LastWriter, ctx: SchedulingContext) -> LastWriter:
    """Schedule a CFR (or a LoopRegion's body) accumulating into
    ``ctx``; returns the final ``LastWriter``."""
    if isinstance(loop_or_cfr, LoopRegion):
        # The body of a LoopRegion is itself a CFR-like structure
        # exposed via ``.body`` (older DaCe) or by iterating its
        # blocks directly.
        body = getattr(loop_or_cfr, "body", loop_or_cfr)
        return schedule_cfr(body, last_writer, ctx)
    return schedule_cfr(loop_or_cfr, last_writer, ctx)


def _condition_reads_gpu(loop: LoopRegion, last_writer: LastWriter) -> bool:
    cond = getattr(loop, "loop_condition", None) or getattr(loop, "condition", None)
    if cond is None:
        return False
    try:
        syms = {str(s) for s in cond.get_free_symbols()}
    except (AttributeError, RuntimeError):
        return False
    return any(s in last_writer and last_writer[s].event_id is not None for s in syms)


def _gpu_streams_read_by_condition(loop: LoopRegion, last_writer: LastWriter) -> Set[int]:
    cond = getattr(loop, "loop_condition", None) or getattr(loop, "condition", None)
    if cond is None:
        return set()
    try:
        syms = {str(s) for s in cond.get_free_symbols()}
    except (AttributeError, RuntimeError):
        return set()
    streams: Set[int] = set()
    for s in syms:
        tok = last_writer.get(s)
        if tok is not None and tok.event_id is not None:
            streams.add(tok.stream_id)
    return streams
