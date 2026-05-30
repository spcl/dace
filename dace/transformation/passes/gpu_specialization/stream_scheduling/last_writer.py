# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Per-data ``LastWriter`` provenance + lattice operations for the scheduler.

``LastWriter`` is a mapping from data-array name to the
``(stream_id, event_id)`` token that recorded its most-recent write at a
point in the schedule. The scheduler threads this map through the
recursive CFR walk; loops require a join (lattice widening) when the
body's effect doesn't reach a fixed point in one iteration.
"""
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class StreamEventToken:
    """A data array's most-recent writer at one point in the schedule.

    ``event_id`` is ``None`` until a cross-stream consumer demands one;
    ``producer`` and ``producer_state`` carry references to the writing
    SDFG node + its containing state, so that demand-driven allocation
    (via :meth:`SchedulingContext.event_for_producer`) plus inter-state
    sync emission have something to key off. ``producer`` is ``None`` for
    host-initialised arrays (no GPU writer yet).
    """
    stream_id: int
    event_id: Optional[int] = None
    producer: Optional[Any] = None  # nodes.Node, kept generic to avoid an import cycle
    producer_state: Optional[Any] = None  # SDFGState, kept generic to avoid an import cycle


# ``LastWriter`` is just a dict; the explicit alias keeps signatures readable.
LastWriter = Dict[str, StreamEventToken]


def stream_signatures_match(a: LastWriter, b: LastWriter) -> bool:
    """Compare two LastWriter maps on stream assignment only.

    Event ids are re-allocated every scheduler pass; they always differ
    between the two iterations of a fixed-point check. The fixed-point
    property holds on the *stream* signature: if every data array's
    writer stays on the same stream across two iterations, the loop is
    iteration-invariant and we can commit the first iteration's
    assignment.
    """
    if a.keys() != b.keys():
        return False
    return all(a[k].stream_id == b[k].stream_id for k in a)


def lastwriter_stream_join(entry: LastWriter, body_out: LastWriter) -> LastWriter:
    """Lattice join used when the body's stream assignment drifts between
    iterations.

    For each data array, if both maps agree on the stream → keep it; if
    they disagree → fall back to ``None`` for the event id (the codegen
    will treat that as a host-initialised array, which forces a
    synchronisation). The widened map is then re-scheduled; convergence
    happens because the lattice has finite height.

    Variables present in only one map carry through unchanged -- they
    were not touched by the iteration we're widening against.
    """
    out: LastWriter = {}
    for k in entry.keys() | body_out.keys():
        e = entry.get(k)
        b = body_out.get(k)
        if e is None:
            out[k] = b
        elif b is None:
            out[k] = e
        elif e.stream_id == b.stream_id:
            out[k] = StreamEventToken(e.stream_id, event_id=None)
        else:
            # Disagreement -> widen to "no committed stream". The
            # subsequent re-schedule will assign one.
            out[k] = StreamEventToken(stream_id=-1, event_id=None)
    return out
