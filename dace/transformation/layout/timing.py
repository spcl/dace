# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Time the COMPUTE region of a laid-out SDFG, excluding the layout relayout copies.

Applying a global layout in wrap mode brackets the compute with relayout states: a copy-IN state
(source) that reorders each input into its laid-out transient, and a copy-OUT state (sink) that
reorders results back. To attribute time to the compute -- not to the relayout -- this module:

  * :func:`add_fusion_barrier` drops an empty side-effect tasklet into a state. ``StateFusion`` treats
    a side-effecting tasklet as a fusion barrier (dace/transformation/interstate/state_fusion.py), so
    the copy-in / copy-out states can never be merged into the compute region -- the timed boundary
    (a state boundary) survives any later simplification.
  * :class:`InsertLayoutTiming` classifies each state (a source/sink whose every tasklet is a pure
    copy is a copy-in/out state; the rest is compute), barriers the copy states, and sets
    ``InstrumentationType.Timer`` on the compute states -- so a run times the compute alone (a start
    marker after copy-in, a stop marker before copy-out).
  * :func:`time_compute` runs the instrumented SDFG and reads the compute duration back from the
    instrumentation report -- the timing the brute-force sweep should compare (relayout is a
    one-time global cost, not part of the steady-state kernel).
"""
import statistics
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import dace
from dace import nodes
from dace.transformation import pass_pipeline as ppl


def add_fusion_barrier(state: dace.SDFGState) -> nodes.Tasklet:
    """Add an empty side-effect tasklet to ``state`` so ``StateFusion`` cannot merge it."""
    return state.add_tasklet(f"__layout_barrier_{state.label}", {}, {}, "", side_effects=True)


def _is_pure_copy_tasklet(t: nodes.Node) -> bool:
    """True for an ``out = in`` copy tasklet (the relayout copy), False for arithmetic / the empty
    fusion barrier."""
    if not isinstance(t, nodes.Tasklet) or len(t.out_connectors) != 1 or len(t.in_connectors) != 1:
        return False
    code = t.code.as_string.strip().rstrip(';').strip()
    if '=' not in code:
        return False
    _, _, rhs = code.partition('=')
    rhs = rhs.strip()
    while rhs.startswith('(') and rhs.endswith(')'):
        rhs = rhs[1:-1].strip()
    return rhs in t.in_connectors


def is_copy_state(state: dace.SDFGState) -> bool:
    """True iff ``state`` is a pure relayout copy: it has tasklets and every one is an ``out = in``
    copy (no arithmetic)."""
    tasklets = [n for n in state.nodes() if isinstance(n, nodes.Tasklet)]
    return len(tasklets) >= 1 and all(_is_pure_copy_tasklet(t) for t in tasklets)


@dataclass
class InsertLayoutTiming(ppl.Pass):
    """Barrier the copy-in / copy-out relayout states and Timer-instrument the compute region.

    After this pass a run of the SDFG produces an instrumentation report whose entries are the
    compute states only (the relayout copies are excluded). Returns the number of compute states
    instrumented.
    """

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Nodes

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def apply_pass(self, sdfg: dace.SDFG, pipeline_results: Dict[str, Any]) -> int:
        states = list(sdfg.states())
        copy_in = [s for s in states if sdfg.in_degree(s) == 0 and is_copy_state(s)]
        copy_out = [s for s in states if sdfg.out_degree(s) == 0 and is_copy_state(s)]
        boundary = set(copy_in) | set(copy_out)

        for state in boundary:
            add_fusion_barrier(state)

        instrumented = 0
        for state in states:
            if state in boundary:
                continue
            if any(isinstance(n, nodes.Tasklet) for n in state.nodes()):
                state.instrument = dace.InstrumentationType.Timer
                instrumented += 1
        return instrumented


def _report_total_ms(report) -> Optional[float]:
    """Sum of all instrumented state durations (ms) in one report, or ``None`` if empty."""
    if report is None:
        return None
    total = 0.0
    seen = False
    for by_state in report.durations.values():
        for by_node in by_state.values():
            for samples in by_node.values():
                if samples:
                    total += sum(samples) / len(samples)
                    seen = True
    return total if seen else None


def time_compute(sdfg: dace.SDFG, run: Callable[[dace.SDFG], Any], reps: int = 5, warmup: int = 1) -> Optional[float]:
    """Run ``run(sdfg)`` ``reps`` times and return the median compute-region time (ms) from the
    instrumentation report (``None`` if the SDFG carries no timers). ``sdfg`` must already be
    instrumented (see :class:`InsertLayoutTiming`)."""
    for _ in range(warmup):
        run(sdfg)
    samples: List[float] = []
    for _ in range(reps):
        run(sdfg)
        ms = _report_total_ms(sdfg.get_latest_report())
        if ms is not None:
            samples.append(ms)
    if not samples:
        return None
    return statistics.median(samples)


def compute_region_timer(sdfg: dace.SDFG,
                         run: Callable[[dace.SDFG], Any],
                         reps: int = 5,
                         warmup: int = 1) -> Optional[float]:
    """A ``brute_force.sweep`` ``timer``: barrier the relayout states, Timer-instrument the compute
    region, and return the median compute time (ms) -- so the sweep ranks candidates by compute
    cost, not by the one-time relayout. Mutates ``sdfg`` (instrumentation)."""
    InsertLayoutTiming().apply_pass(sdfg, {})
    return time_compute(sdfg, run, reps, warmup)
