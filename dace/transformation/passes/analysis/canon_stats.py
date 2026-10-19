# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Which stage settled each loop of a canonicalization, and the time spent in the solvers.

``canonicalize(sdfg, stats=...)`` makes a :class:`CanonStats` the active recorder for the run. With none active
every hook here returns at once, so a run without ``stats`` does the same work as before.

- The stage runner snapshots the loops around every stage (:meth:`CanonStats.stage_done`): a loop that disappears
  in a stage was settled by it.
- ``LoopToMap`` reports every loop it lifts, at level 1 (subscript and range tests) or level 3 (a Z3 query proved a
  pair), :func:`solver_proof` marking the latter. A loop gone in ``WavefrontSkew`` is level 2 (isl).
- :func:`timed` adds the time inside Z3's ``check`` and inside the isl queries.
"""

import contextlib
import contextvars
import json
import time
from collections import Counter
from typing import Any

_ACTIVE: contextvars.ContextVar["CanonStats | None"] = contextvars.ContextVar("canon_stats", default=None)

#: Stage units whose disappearing loops are guarded loops: a parallel version behind a runtime check.
GUARDED = frozenset({"ScatterToGuardedMaps", "ParallelizeUnderConstraint"})


def active() -> "CanonStats | None":
    return _ACTIVE.get()


@contextlib.contextmanager
def timed(kind: str):
    """Add the time of the enclosed block to ``kind`` (``'z3'`` or ``'isl'``); nested blocks count once."""
    stats = _ACTIVE.get()
    if stats is None or stats._depth[kind]:
        yield
        return
    stats._depth[kind] += 1
    start = time.perf_counter()
    try:
        yield
    finally:
        stats.seconds[kind] += time.perf_counter() - start
        stats._depth[kind] -= 1


def begin_probe() -> None:
    """A ``LoopToMap`` probe starts: no solver proof yet."""
    stats = _ACTIVE.get()
    if stats is not None:
        stats._probe_used_solver = False


def solver_proof() -> None:
    """A Z3 query proved a pair independent in the current ``LoopToMap`` probe."""
    stats = _ACTIVE.get()
    if stats is not None:
        stats._probe_used_solver = True


def lifted(loop: Any) -> None:
    """``LoopToMap`` lifts ``loop``, accepted by the probe just before."""
    stats = _ACTIVE.get()
    if stats is not None:
        # The skew's own lift of the exposed axis is level 2: the isl skew is what made it parallel.
        level = "level2" if stats.stage[1] == "WavefrontSkew" else "level3" if stats._probe_used_solver else "level1"
        stats._settle(loop, level)


def settle(loop: Any, outcome: str) -> None:
    """A pass settles ``loop`` whose object it keeps (``LoopToScan`` reuses the loop as its fill loop)."""
    stats = _ACTIVE.get()
    if stats is not None:
        stats._settle(loop, outcome)


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
    from dace.transformation.passes.canonicalize import annotate_loop_kinds as kinds

    note = (loop.specialization_hint or "").splitlines()
    if note and ("wavefront" in note[0]) and note[0].startswith("sequential"):
        return "wavefront"
    hint = kinds.loop_hint(loop).splitlines()[0]
    if hint.startswith("sequential -- pinned"):
        return "guarded_fallback"
    if hint.startswith("sequential -- carried"):
        return "sequential_witness" if kinds.witnessed_carrying_access(loop) is not None else "sequential_pattern"
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
        self._depth: Counter = Counter()
        self._probe_used_solver = False
        self._seen: set[int] = set()
        # Keeps every snapshot loop alive, so an ``id`` never names two loops.
        self._loops: dict[int, Any] = {}
        self._root: Any = None

    def _settle(self, loop: Any, outcome: str) -> None:
        # A trial rewrite on a copy (LoopToEinsum's) settles nothing in the program being canonicalized.
        sdfg = loop.sdfg
        while sdfg is not None and sdfg.parent_sdfg is not None:
            sdfg = sdfg.parent_sdfg
        if sdfg is not self._root:
            return
        if id(loop) not in self._seen:
            self._seen.add(id(loop))
            self.settled.append((*self.stage, loop.label, outcome))

    @contextlib.contextmanager
    def recording(self, sdfg: Any):
        token = _ACTIVE.set(self)
        start = time.perf_counter()
        self._root = sdfg
        self._loops = loops_of(sdfg)
        self.loops_in = len(self._loops)
        try:
            yield self
        finally:
            self.seconds["total"] += time.perf_counter() - start
            _ACTIVE.reset(token)

    def stage_start(self, label: str, unit: Any) -> None:
        self.stage = (label, type(unit).__name__)

    def stage_done(self, sdfg: Any) -> None:
        """Attribute the loops gone since the last snapshot to the stage that just ran."""
        now = loops_of(sdfg)
        unit = self.stage[1]
        for key, loop in self._loops.items():
            if key in now:
                continue
            if unit == "WavefrontSkew":
                self._settle(loop, "level2")
            elif unit in GUARDED:
                self._settle(loop, "guarded")
            elif self.stage[0] == "lift_semantics":
                self._settle(loop, f"lift:{unit}")
            else:
                self._settle(loop, f"restructured:{unit}")
        self._loops = now

    def finish(self, sdfg: Any) -> None:
        """Classify the loops canonicalization left; not timed, it is the CPF rendering's work."""
        self.remaining = [(loop.label, remaining_outcome(loop)) for loop in loops_of(sdfg).values()]
        self._loops = {}

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
