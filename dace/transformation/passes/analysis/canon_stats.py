# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Which stage settled each loop of a canonicalization, and the time spent in the solvers.

``canonicalize(sdfg, stats=...)`` attaches a :class:`CanonStats` to the root SDFG (``SDFG._canon_stats``) for the run;
every hook here takes the SDFG, state or region it acts on and finds the recorder of that root. With none attached
every hook returns at once, so a run without ``stats`` does the same work as before, and a trial copy (whose
``_canon_stats`` the deep copy clears) records nothing.

- The stage runner snapshots the loops around every stage (:meth:`CanonStats.stage_done`): a loop that disappears
  in a stage was settled by it.
- ``LoopToMap`` reports every loop it lifts, at level 1 (subscript and range tests) or level 3 (a Z3 query proved a
  pair). A loop gone in ``WavefrontSkew`` is level 2 (isl).
- :func:`timed` adds the time of the Z3- and isl-backed analyses, measured where a pass calls them.
"""

import contextlib
import json
import time
from collections import Counter
from typing import Any

#: Stage units whose disappearing loops are guarded loops: a parallel version behind a runtime check.
GUARDED = frozenset({"ScatterToGuardedMaps", "ParallelizeUnderConstraint"})


def of(anchor: Any) -> "CanonStats | None":
    """The recorder of the root SDFG of ``anchor`` (an SDFG, state or control-flow region), or ``None``."""
    from dace.sdfg import SDFG

    sdfg = anchor if isinstance(anchor, SDFG) else anchor.sdfg
    while sdfg is not None and sdfg.parent_sdfg is not None:
        sdfg = sdfg.parent_sdfg
    return None if sdfg is None else sdfg._canon_stats


@contextlib.contextmanager
def timed(kind: str, anchor: Any):
    """Add the time of the enclosed block to ``kind`` (``'z3'`` or ``'isl'``) of ``anchor``'s recorder; nested
    blocks count once."""
    stats = of(anchor)
    if stats is None or stats.depth[kind]:
        yield
        return
    stats.depth[kind] += 1
    start = time.perf_counter()
    try:
        yield
    finally:
        stats.seconds[kind] += time.perf_counter() - start
        stats.depth[kind] -= 1


def lifted(loop: Any, used_solver: bool) -> None:
    """``LoopToMap`` lifts ``loop``; ``used_solver`` when a Z3 query proved a pair of its probe."""
    stats = of(loop)
    if stats is not None:
        # The skew's own lift of the exposed axis is level 2: the isl skew is what made it parallel.
        # A guarding pass's lift is the parallel arm of its guard, not a proof.
        unit = stats.stage[1]
        if unit == "WavefrontSkew":
            level = "level2"
        elif unit in GUARDED:
            level = "guarded"
        else:
            level = "level3" if used_solver else "level1"
        stats.record(loop, level)


def settle(loop: Any, outcome: str) -> None:
    """A pass settles ``loop`` whose object it keeps (``LoopToScan`` reuses the loop as its fill loop)."""
    stats = of(loop)
    if stats is not None:
        stats.record(loop, outcome)


def loops_of(sdfg: Any) -> dict[int, Any]:
    from dace.sdfg.state import LoopRegion

    return {
        id(region): region
        for sd in sdfg.all_sdfgs_recursive()
        for region in sd.all_control_flow_regions()
        if isinstance(region, LoopRegion)
    }


def remaining_outcome(loop: Any) -> str:
    """The class of a loop left after canonicalization, as the CPF rendering labels it."""
    from dace.sdfg.state import ConditionalBlock
    from dace.transformation.passes.canonicalize import annotate_loop_kinds as kinds

    note = (loop.specialization_hint or "").splitlines()
    if note and ("wavefront" in note[0]) and note[0].startswith("sequential"):
        return "wavefront"
    hint = kinds.loop_hint(loop).splitlines()[0]
    if hint.startswith("sequential -- carried"):
        return "sequential_witness" if kinds.witnessed_carrying_access(loop) is not None else "sequential_pattern"
    # The fallback arm of a specialization: pinned, and an arm of the conditional the guard is.
    if loop.pinned_sequential and isinstance(loop.parent_graph.parent_graph, ConditionalBlock):
        return "guarded_fallback"
    if kinds.NEVER_EXAMINED in hint:
        return "never_examined"
    if hint.startswith("unsure"):
        return "unknown"
    return "parallel_unlifted"


class CanonStats:
    """One canonicalization's record; :meth:`as_dict` is the JSON line."""

    def __init__(self, kernel: str):
        self.kernel = kernel
        self.seconds: Counter = Counter()
        self.loops_in = 0
        #: ``(stage label, unit class, loop label, outcome)`` per settled loop, in order.
        self.settled: list[tuple[str, str, str, str]] = []
        self.remaining: list[tuple[str, str]] = []
        self.stage = ("", "")
        self.depth: Counter = Counter()
        self.seen: set[int] = set()
        # Keeps every snapshot loop alive, so an ``id`` never names two loops.
        self.loops: dict[int, Any] = {}

    def record(self, loop: Any, outcome: str) -> None:
        """``loop`` settled with ``outcome`` in the current stage; a loop counts once."""
        if id(loop) not in self.seen:
            self.seen.add(id(loop))
            self.settled.append((*self.stage, loop.label, outcome))

    @contextlib.contextmanager
    def recording(self, sdfg: Any):
        """Attach this recorder to ``sdfg`` for the enclosed block."""
        sdfg._canon_stats = self
        start = time.perf_counter()
        self.loops = loops_of(sdfg)
        self.loops_in = len(self.loops)
        try:
            yield self
        finally:
            self.seconds["total"] += time.perf_counter() - start
            sdfg._canon_stats = None

    def stage_start(self, label: str, unit: Any) -> None:
        self.stage = (label, type(unit).__name__)

    def stage_done(self, sdfg: Any) -> None:
        """Attribute the loops gone since the last snapshot to the stage that just ran."""
        now = loops_of(sdfg)
        unit = self.stage[1]
        for key, loop in self.loops.items():
            if key in now:
                continue
            if unit == "WavefrontSkew":
                self.record(loop, "level2")
            elif unit in GUARDED:
                self.record(loop, "guarded")
            elif self.stage[0] == "lift_semantics":
                self.record(loop, f"lift:{unit}")
            else:
                self.record(loop, f"restructured:{unit}")
        self.loops = now

    def finish(self, sdfg: Any) -> None:
        """Classify the loops canonicalization left; not timed, it is the CPF rendering's work."""
        self.remaining = [(loop.label, remaining_outcome(loop)) for loop in loops_of(sdfg).values()]
        self.loops = {}

    def as_dict(self) -> dict[str, Any]:
        outcomes = Counter(outcome for *_, outcome in self.settled)
        outcomes.update(outcome for _, outcome in self.remaining)
        return {
            "kernel": self.kernel,
            "seconds": round(self.seconds["total"], 6),
            "z3_seconds": round(self.seconds["z3"], 6),
            "isl_seconds": round(self.seconds["isl"], 6),
            "loops_in": self.loops_in,
            "outcomes": dict(sorted(outcomes.items())),
            "settled": [
                {"stage": stage, "unit": unit, "loop": loop, "outcome": outcome}
                for stage, unit, loop, outcome in self.settled
            ],
            "remaining": [{"loop": loop, "outcome": outcome} for loop, outcome in self.remaining],
        }

    def write(self, path: str) -> None:
        with open(path, "a", encoding="utf-8") as out:
            out.write(json.dumps(self.as_dict(), sort_keys=True) + "\n")
