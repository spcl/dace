# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Time the compute region of a laid-out SDFG, excluding the relayout copy-in/copy-out states.
"""
import statistics
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import dace
from dace import nodes
from dace.transformation import pass_pipeline as ppl

BARRIER_PREFIX = "__layout_barrier_"


def add_fusion_barrier(state: dace.SDFGState) -> nodes.Tasklet:
    """Add an empty side-effect tasklet to ``state`` so ``StateFusion`` cannot merge it."""
    return state.add_tasklet(f"{BARRIER_PREFIX}{state.label}", {}, {}, "", side_effects=True)


def is_fusion_barrier(node: nodes.Node) -> bool:
    """True for a barrier tasklet this module added (see :func:`add_fusion_barrier`)."""
    return isinstance(node, nodes.Tasklet) and node.label.startswith(BARRIER_PREFIX)


def has_fusion_barrier(state: dace.SDFGState) -> bool:
    """True iff ``state`` already carries a barrier, so it is not barriered twice."""
    return any(is_fusion_barrier(n) for n in state.nodes())


def _is_pure_copy_tasklet(t: nodes.Node) -> bool:
    """True for an ``out = in`` copy tasklet, False for arithmetic or the barrier tasklet."""
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


def state_runs_on_gpu(state: dace.SDFGState) -> bool:
    """True iff ``state`` carries a GPU map (recurses into nested SDFGs); needs CUDA-event timing,
    not a host ``Timer``."""
    return any(
        isinstance(node, nodes.MapEntry) and node.map.schedule in dace.dtypes.GPU_SCHEDULES
        for node, _ in state.all_nodes_recursive())


def instrumentation_for(state: dace.SDFGState) -> dace.InstrumentationType:
    """The instrument to time ``state`` with: CUDA events on the GPU, the host timer otherwise."""
    return dace.InstrumentationType.GPU_Events if state_runs_on_gpu(state) else dace.InstrumentationType.Timer


def is_copy_state(state: dace.SDFGState) -> bool:
    """True iff ``state`` has tasklets and all are ``out = in`` copies (barrier tasklets are ignored,
    so an already-barriered copy state is still recognized as one)."""
    tasklets = [n for n in state.nodes() if isinstance(n, nodes.Tasklet) and not is_fusion_barrier(n)]
    return len(tasklets) >= 1 and all(_is_pure_copy_tasklet(t) for t in tasklets)


def barrier_relayout_states(sdfg: dace.SDFG) -> int:
    """Barrier every relayout copy state so a later fusing transform can't merge it into compute.
    Call BEFORE ``SDFG.apply_gpu_transformations``, which otherwise fuses them away. Returns the count."""
    count = 0
    for state in list(sdfg.states()):
        if is_copy_state(state) and not has_fusion_barrier(state):
            add_fusion_barrier(state)
            count += 1
    return count


def state_has_tasklets(state: dace.SDFGState) -> bool:
    """True iff ``state`` has a non-barrier tasklet. dace's host copy-in/out states carry memlet
    copies but no tasklets, so they read as transparent here."""
    return any(isinstance(n, nodes.Tasklet) and not is_fusion_barrier(n) for n in state.nodes())


def reaches_tasklets(sdfg: dace.SDFG, state: dace.SDFGState, forward: bool) -> bool:
    """True iff any state reachable from ``state`` (forward or backward), looking through states
    with no tasklets, does real work. More robust than plain ``in_degree == 0``, which misclassifies
    a relayout wrapped by ``apply_gpu_transformations``'s tasklet-less host copy states as compute."""
    edges = sdfg.out_edges if forward else sdfg.in_edges
    pick = (lambda e: e.dst) if forward else (lambda e: e.src)
    seen = set()
    stack = [pick(e) for e in edges(state)]
    while stack:
        nxt = stack.pop()
        if nxt in seen:
            continue
        seen.add(nxt)
        if state_has_tasklets(nxt):
            return True
        stack.extend(pick(e) for e in edges(nxt))
    return False


@dataclass
class InsertLayoutTiming(ppl.Pass):
    """Barrier the copy-in/copy-out relayout states and instrument the compute region so a run's
    instrumentation report covers compute only. Returns the number of states instrumented."""

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Nodes

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def apply_pass(self, sdfg: dace.SDFG, pipeline_results: Dict[str, Any]) -> int:
        states = list(sdfg.states())
        copy_in = {s for s in states if is_copy_state(s) and not reaches_tasklets(sdfg, s, forward=False)}
        copy_out = {s for s in states if is_copy_state(s) and not reaches_tasklets(sdfg, s, forward=True)}
        # sym diff: a copy state on both lists has compute on neither side, so it IS the compute
        # (a lone transpose/copy nest) -- treating it as boundary would leave it untimed.
        boundary = copy_in ^ copy_out

        for state in boundary:
            if not has_fusion_barrier(state):  # idempotent: skip if already barriered
                add_fusion_barrier(state)

        instrumented = 0
        for state in states:
            if state in boundary:
                continue
            if any(isinstance(n, nodes.Tasklet) and not is_fusion_barrier(n) for n in state.nodes()):
                state.instrument = instrumentation_for(state)
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


#: Spread above this is flagged contended (kept, but marked untrusted).
SPREAD_CONTENDED_THRESHOLD = 0.10


def time_compute_stats(sdfg: dace.SDFG,
                       run: Callable[[dace.SDFG], Any],
                       reps: int = 10,
                       warmup: int = 2) -> Optional[Dict[str, Any]]:
    """Run ``run(sdfg)`` ``reps`` times and return compute-region stats from the instrumentation
    report: ``{"median": ms, "spread": (max-min)/min, "contended": bool, "samples": [...]}``, or
    ``None`` if ``sdfg`` carries no timers (see :class:`InsertLayoutTiming`). Every rep must produce
    a fresh report -- the build folder is keyed on SDFG name and shared across sweep candidates, so
    a stale or missing report is a hard error, never silently absorbed."""
    if not any(state.instrument != dace.InstrumentationType.No_Instrumentation for state in sdfg.states()):
        return None
    for _ in range(warmup):
        run(sdfg)
    samples: List[float] = []
    for rep in range(reps):
        previous_path = sdfg.get_latest_report_path()
        run(sdfg)
        path = sdfg.get_latest_report_path()
        if path is None or path == previous_path:
            raise RuntimeError(f"time_compute_stats: rep {rep} of '{sdfg.name}' produced no fresh instrumentation "
                               f"report (latest: {path}); the run did not write one (report_each_invocation off, "
                               f"or a save path failure) -- refusing to report stale timings")
        ms = _report_total_ms(sdfg.get_latest_report())
        if ms is None:
            raise RuntimeError(f"time_compute_stats: rep {rep} of '{sdfg.name}' wrote an EMPTY instrumentation "
                               f"report ({path}) despite instrumented states -- refusing to shrink the sample set "
                               f"silently")
        samples.append(ms)
    low = min(samples)
    spread = (max(samples) - low) / low if low > 0.0 else 0.0
    return {
        "median": statistics.median(samples),
        "spread": spread,
        "contended": spread > SPREAD_CONTENDED_THRESHOLD,
        "samples": samples,
    }


def time_compute(sdfg: dace.SDFG, run: Callable[[dace.SDFG], Any], reps: int = 5, warmup: int = 1) -> Optional[float]:
    """The median compute-region time (ms) of ``run(sdfg)`` -- see :func:`time_compute_stats`."""
    stats = time_compute_stats(sdfg, run, reps, warmup)
    return None if stats is None else stats["median"]


def compute_region_timer(sdfg: dace.SDFG,
                         run: Callable[[dace.SDFG], Any],
                         reps: int = 5,
                         warmup: int = 1) -> Optional[float]:
    """A ``brute_force.sweep`` ``timer``: instruments the compute region and returns its median
    time (ms), so the sweep ranks by compute cost, not the one-time relayout. Mutates ``sdfg``."""
    InsertLayoutTiming().apply_pass(sdfg, {})
    return time_compute(sdfg, run, reps, warmup)


def compute_region_stats_timer(sdfg: dace.SDFG, run: Callable[[dace.SDFG], Any], reps: int = 10, warmup: int = 2):
    """Like :func:`compute_region_timer` but returns ``(median_ms, {"spread", "contended",
    "samples"})`` so ``sweep`` records the trust signal in ``SweepResult.metadata``; ``None`` if
    nothing to instrument."""
    InsertLayoutTiming().apply_pass(sdfg, {})
    stats = time_compute_stats(sdfg, run, reps, warmup)
    if stats is None:
        return None
    return stats["median"], {k: stats[k] for k in ("spread", "contended", "samples")}
