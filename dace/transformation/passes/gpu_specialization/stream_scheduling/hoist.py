# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Bottom-up sync-hoist over the SDFG's CFR tree.

The scheduler is conservative: it inserts a ``StreamWaitEvent`` /
``StreamSynchronize`` at every cross-stream consumer or interstate
host-read it finds. After assignment, many of these are redundant --
a whole region might use a single stream, in which case no in-region
sync is needed. This pass elides them.

Single principle: if every scheduled node inside a ``ControlFlowRegion``
ended up on the same stream and no contained block carries an
interstate-edge host-read sync, then no intra-region sync is required.
Drop the corresponding sync entries from the context.
"""
from typing import Set

from dace import SDFG
from dace.sdfg.state import ControlFlowRegion, SDFGState

from .scheduler import SchedulingContext


def hoist_redundant_syncs(sdfg: SDFG, ctx: SchedulingContext):
    """Walk every CFR bottom-up; drop intra-region cross-stream events and
    interstate host-reads when the whole region uses a single stream."""
    _hoist_cfr(sdfg, ctx, _streams_in_cfr_cache={})


def _hoist_cfr(cfr: ControlFlowRegion, ctx: SchedulingContext, _streams_in_cfr_cache):
    if cfr in _streams_in_cfr_cache:
        return _streams_in_cfr_cache[cfr]

    nested_streams: Set[int] = set()
    for block in (cfr.nodes() if hasattr(cfr, "nodes") else []):
        if isinstance(block, ControlFlowRegion) and not isinstance(block, SDFGState):
            nested_streams |= _hoist_cfr(block, ctx, _streams_in_cfr_cache)
        elif isinstance(block, SDFGState):
            for n in block.nodes():
                if n in ctx.assignments:
                    nested_streams.add(ctx.assignments[n])

    _streams_in_cfr_cache[cfr] = nested_streams

    if len(nested_streams) > 1 or not nested_streams:
        return nested_streams

    # Whole CFR runs on a single stream -- drop intra-region syncs.
    only_stream = next(iter(nested_streams))
    contained_blocks = _all_blocks_recursive(cfr)
    if _has_external_host_read(cfr, ctx, contained_blocks):
        return nested_streams

    ctx.cross_stream_edges = [(s, src, dst, e) for (s, src, dst, e) in ctx.cross_stream_edges
                              if s not in contained_blocks]
    ctx.interstate_host_reads = [(ie, sid) for (ie, sid) in ctx.interstate_host_reads
                                 if not _interstate_inside(ie, contained_blocks)]
    # The CFR's net effect is "this stream produced everything"; nothing
    # else to record (the parent will see it via ``ctx.assignments``).
    _ = only_stream
    return nested_streams


def _all_blocks_recursive(cfr: ControlFlowRegion) -> Set:
    out = set()
    for block in (cfr.nodes() if hasattr(cfr, "nodes") else []):
        out.add(block)
        if isinstance(block, ControlFlowRegion) and not isinstance(block, SDFGState):
            out |= _all_blocks_recursive(block)
    return out


def _has_external_host_read(cfr, ctx: SchedulingContext, contained_blocks: Set) -> bool:
    """True if any interstate host-read sync sits inside this CFR's blocks."""
    return any(_interstate_inside(ie, contained_blocks) for (ie, _) in ctx.interstate_host_reads)


def _interstate_inside(ie, contained_blocks: Set) -> bool:
    if ie is None:
        return False
    src = getattr(ie, "src", None)
    dst = getattr(ie, "dst", None)
    return src in contained_blocks or dst in contained_blocks
