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
    copy is a copy-in/out state; the rest is compute), barriers the copy states, and instruments the
    compute states -- so a run times the compute alone (a start marker after copy-in, a stop marker
    before copy-out). The instrument is chosen PER DEVICE, never hardcoded: a GPU-scheduled state
    gets ``InstrumentationType.GPU_Events`` (CUDA events around the kernels -- a host wall-clock
    ``Timer`` would only bracket the asynchronous launch, not the GPU work), a host state gets
    ``InstrumentationType.Timer``. Both report milliseconds, so a report mixing the two still sums.
  * :func:`time_compute` runs the instrumented SDFG and reads the compute duration back from the
    instrumentation report -- the timing the brute-force sweep should compare (relayout is a
    one-time global cost, not part of the steady-state kernel).

.. note:: Barriers only stop LATER fusion, so they must exist BEFORE anything that fuses.
   ``SDFG.apply_gpu_transformations`` fuses the relayout states into the compute state, leaving
   nothing to time apart -- call :func:`barrier_relayout_states` BEFORE it and the split survives
   (measured on an RTX 4050: the compute region is then 0.009-0.021 ms of a ~400 ms whole call, and
   it separates layouts the whole-call timer cannot). :class:`InsertLayoutTiming` is idempotent with
   respect to barriers already present, so it is safe to call afterwards.
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


def state_runs_on_gpu(state: dace.SDFGState) -> bool:
    """True iff ``state``'s compute is scheduled on the GPU, i.e. it carries a GPU map. Such a state
    must be timed with CUDA events: a host ``Timer`` around it brackets only the asynchronous kernel
    launch, not the kernel.

    Recurses into nested SDFGs: ``state.nodes()`` is flat, so a GPU map inside a NestedSDFG would
    otherwise read as host work and get the wall-clock timer."""
    return any(
        isinstance(node, nodes.MapEntry) and node.map.schedule in dace.dtypes.GPU_SCHEDULES
        for node, _ in state.all_nodes_recursive())


def instrumentation_for(state: dace.SDFGState) -> dace.InstrumentationType:
    """The instrument to time ``state`` with: CUDA events on the GPU, the host timer otherwise."""
    return dace.InstrumentationType.GPU_Events if state_runs_on_gpu(state) else dace.InstrumentationType.Timer


def is_copy_state(state: dace.SDFGState) -> bool:
    """True iff ``state`` is a pure relayout copy: it has tasklets and every one is an ``out = in``
    copy (no arithmetic). A barrier tasklet this module added is ignored -- barriering a copy state
    must not stop it from being recognized as one (it may have been barriered by an earlier pass, or
    before a GPU transform)."""
    tasklets = [n for n in state.nodes() if isinstance(n, nodes.Tasklet) and not is_fusion_barrier(n)]
    return len(tasklets) >= 1 and all(_is_pure_copy_tasklet(t) for t in tasklets)


def barrier_relayout_states(sdfg: dace.SDFG) -> int:
    """Barrier every relayout copy state, so a later fusing transform cannot merge relayout into
    compute. Call this BEFORE ``SDFG.apply_gpu_transformations`` (which otherwise fuses them into one
    state, leaving no boundary to time). Already-barriered states are left alone. Returns the count.
    """
    count = 0
    for state in list(sdfg.states()):
        if is_copy_state(state) and not has_fusion_barrier(state):
            add_fusion_barrier(state)
            count += 1
    return count


def state_has_tasklets(state: dace.SDFGState) -> bool:
    """True iff ``state`` does real work -- a tasklet that is not one of our barriers. dace's host
    copy-in/copy-out states (added by ``apply_gpu_transformations``) carry memlet copies but no
    tasklets, so they are transparent by this measure."""
    return any(isinstance(n, nodes.Tasklet) and not is_fusion_barrier(n) for n in state.nodes())


def reaches_tasklets(sdfg: dace.SDFG, state: dace.SDFGState, forward: bool) -> bool:
    """True iff any state reachable from ``state`` (forward or backward) does real work, looking
    THROUGH states that do none.

    This is the source/sink test that identifies a relayout boundary, made robust to wrapper states:
    a plain ``in_degree == 0`` breaks as soon as ``apply_gpu_transformations`` wraps the graph in
    tasklet-less host copy states, which leaves the relayout in the middle and makes it read as
    compute. Looking through those keeps a MID-GRAPH copy (a real algorithmic copy, which has a
    working state on both sides) classified as compute, as it must be.
    """
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
    """Barrier the copy-in / copy-out relayout states and instrument the compute region.

    After this pass a run of the SDFG produces an instrumentation report whose entries are the
    compute states only (the relayout copies are excluded). Returns the number of compute states
    instrumented. The instrument is picked per state by :func:`instrumentation_for` -- CUDA events for
    a GPU-scheduled state, the host timer otherwise -- so the pass never hardcodes a device-specific
    measurement.
    """

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Nodes

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def apply_pass(self, sdfg: dace.SDFG, pipeline_results: Dict[str, Any]) -> int:
        states = list(sdfg.states())
        copy_in = [s for s in states if is_copy_state(s) and not reaches_tasklets(sdfg, s, forward=False)]
        copy_out = [s for s in states if is_copy_state(s) and not reaches_tasklets(sdfg, s, forward=True)]
        boundary = set(copy_in) | set(copy_out)

        for state in boundary:
            if not has_fusion_barrier(state):  # idempotent: a state barriered earlier is left alone
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


def time_compute(sdfg: dace.SDFG, run: Callable[[dace.SDFG], Any], reps: int = 5, warmup: int = 1) -> Optional[float]:
    """Run ``run(sdfg)`` ``reps`` times and return the median compute-region time (ms) from the
    instrumentation report (``None`` if the SDFG carries no timers). ``sdfg`` must already be
    instrumented (see :class:`InsertLayoutTiming`).

    The no-timer case MUST short-circuit: ``get_latest_report`` reads the newest report in
    ``build_folder/perf``, and the build folder is keyed on the SDFG NAME -- every candidate in a
    layout sweep shares one. An uninstrumented SDFG writes no report, so falling through would read
    the PREVIOUS candidate's report and silently return its time as this one's.
    """
    if not any(state.instrument != dace.InstrumentationType.No_Instrumentation for state in sdfg.states()):
        return None
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
