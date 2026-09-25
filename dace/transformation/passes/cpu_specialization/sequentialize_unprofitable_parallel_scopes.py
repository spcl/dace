# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The CPU fork/join cost model: the one place where a parallel scope is made sequential again.

The canonical form is the maximally parallel, device-neutral one: every DOALL loop is a Map and
every bulk transfer is an element map, with no fork/join reasoning anywhere in it. Schedules there
are labels, not decisions. Turning one of those maps back into a serial loop is a CPU
specialization -- it trades parallelism for the ``#pragma omp parallel`` a region would cost -- so
it lives here, in the ``cpu_specialize`` band, and nowhere else.

Two independent reasons to take the parallelism away, in the order they are checked:

1. **Nested parallelism.** A map (or library node) that already runs inside a parallel map of the
   same device schedule would open a team per outer iteration. Always sequential, at any size.
   One OpenMP team level is what OpenMP gives (nested parallelism is off by default), so only a
   map no parallel map encloses opens a region; using the inner dimension too is what map collapse
   and fusion are for. This is the rule the old
   ``canonicalize/finalize.sequentialize_nested_parallel_scopes`` implemented, and it is preserved
   exactly, transitively across nested-SDFG boundaries.
2. **Below break-even.** The map's OWN iteration count is PROVABLY under
   ``compiler.cpu.parallel_min_work_per_region``. Constants decide statically; a count written in
   the SDFG's own PARAMETERS is assumed big enough and stays parallel unguarded (the same default
   :func:`~dace.libraries.standard.helper.is_parallel_cpu_transfer_size` applies to transfers).
3. **Unbounded at compile time.** A count naming a value the PROGRAM computes -- a scalar an
   interstate edge assigns from data, e.g. the ``ntouch`` of a stream compaction inside a sparse
   triple product -- cannot be ruled on either way, and guessing "big" costs a whole OpenMP region
   per entry for a handful of elements (measured on amg_setup: three such loops, 230 ms against
   30 ms). Those keep their Map and their parallel schedule and carry the cost model as an
   ``if(parallel: count >= threshold)`` clause instead, decided per call. The clause names
   ``parallel`` explicitly so a combined ``parallel for simd`` keeps its simd clause -- a BARE
   ``if`` there devectorizes, which is why one is never emitted.

The map's own iteration count, not the work of its whole subtree: a 16-iteration map wrapping a
1M-element inner map is better served by sequentializing the outer one and letting rule 1 release
the inner, than by forking 16 ways over 8 threads.

An enclosing loop is NOT a reason. The fork/join algebra cancels it: a region entered ``E`` times
costs ``E * (fork + work/P)`` against ``E * work`` sequential, so ``E`` multiplies both sides and
only the region itself decides. An earlier rule compared work per entry against ``E`` and pinned
npbench ``cavity_flow``'s 1,044,484-element pressure map ``Sequential`` because its ``nit*nt``
entry count evaluated 4092 higher -- a map a thousand times above break-even, made serial by a
quantity that cannot matter.

A reduction nested in a sequential loop is the same case and is decided here too. There used to be
a ``PinNestedSequentialLoops`` pass in the canonicalization band that kept such a loop a loop, so
it never reached a schedule at all; it is gone. Measured against it at 8 threads: a 4096x4096 inner
reduction ran 17.62 ms pinned against 10.98 ms as a WCR map, and on polybench ``nussinov`` -- the
kernel the pin was written for -- the generated C++ was BYTE-IDENTICAL either way, because
``WavefrontSkew`` now turns its ``(i, j)`` nest into a parallel diagonal map and rule 1 owns the
k-reduction underneath it.

Bulk transfers are decided one pass over in
:mod:`~dace.transformation.passes.cpu_specialization.specialize_cpu_transfers`: a copy / memset
library node carries no map to measure until it expands, so
:func:`~dace.libraries.standard.helper.is_reentered_cpu_transfer` still applies the ENCLOSING-SCOPE
test to it directly (that is npbench ``stockham_fft``'s 349,525-entry inner copy, a measured 44x).

Setting the threshold to 0 turns rule 2 off, leaving only the nested-parallelism rule -- the A/B
lever for measuring the cost model itself.
"""
from typing import Any, Dict, List, Optional

from dace import SDFG, dtypes, properties, symbolic
from dace.config import Config
from dace.sdfg import nodes
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion, LoopRegion, SDFGState
from dace.transformation import pass_pipeline as ppl

#: Schedules a CPU map or library node still parallelizes under. ``Default`` is included because
#: canonicalization leaves the schedule unset: codegen infers it, and a top-level ``Default`` map
#: becomes ``CPU_Multicore``. Anything else (GPU schedules, ``Unrolled``, ``CPU_Persistent``) is
#: someone else's decision and is left untouched.
CPU_PARALLEL_SCHEDULES = (dtypes.ScheduleType.Default, dtypes.ScheduleType.CPU_Multicore)


def min_work_per_region() -> int:
    """The configured break-even work of one OpenMP region, in elements.

    :returns: ``compiler.cpu.parallel_min_work_per_region``; 0 disables the size rule.
    """
    return int(Config.get('compiler', 'cpu', 'parallel_min_work_per_region'))


@properties.make_properties
class SequentializeUnprofitableParallelScopes(ppl.Pass):
    """Pin every CPU parallel scope the fork/join cost model refuses to ``Sequential``."""

    CATEGORY: str = 'Device Specialization'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Nodes

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self):
        return set()

    def apply_pass(self, sdfg: SDFG, _pipeline_results: Dict[str, Any]) -> Optional[int]:
        """Sequentialize the parallel scopes that do not pay for their fork/join.

        :param sdfg: the SDFG to specialize, in place.
        :param _pipeline_results: unused.
        :returns: how many scopes were pinned or guarded, or ``None`` if none were.
        """
        self.threshold = min_work_per_region()
        self.pinned = 0
        self.guarded = 0
        # Parameter names per SDFG: what no call can change midway, so a count written in them is
        # the one a compile-time verdict may rule on. Built once per SDFG, never rebuilt -- this
        # pass only rewrites schedules.
        self.params: Dict[int, frozenset] = {}
        # One pass's worth of enclosing-loop trip-count verdicts: this pass never mutates a
        # LoopRegion's bounds, so a verdict computed under one scope stays valid for a sibling
        # transfer under the same loop.
        self.loop_cache: Dict[int, bool] = {}
        # Trip count -> ``ask('negative', count - threshold)``. Maps share a handful of trip counts
        # (cloudsc: 2264 queries, 16 distinct), and each query is a sympy SAT problem.
        self.below_threshold: Dict[Any, Optional[bool]] = {}
        self.visit_region(sdfg, False)
        return (self.pinned + self.guarded) or None

    def visit_region(self, region: ControlFlowRegion, in_parallel: bool) -> None:
        """Walk ``region``'s blocks, threading whether a parallel map encloses them.

        :param region: the control-flow region (or nested SDFG) to walk.
        :param in_parallel: whether a device-parallel map encloses this region.
        """
        for block in region.nodes():
            if isinstance(block, SDFGState):
                self.visit_scope(block, block.scope_children(), None, in_parallel)
            elif isinstance(block, ConditionalBlock):
                # A ConditionalBlock is NOT a ControlFlowRegion, so its branches are only reachable
                # through ``branches``. Everything an ``if`` guards was invisible here otherwise --
                # amg_setup puts its whole body under one, and every map inside it kept a region it
                # could not pay for.
                for _condition, branch in block.branches:
                    self.visit_region(branch, in_parallel)
            elif isinstance(block, ControlFlowRegion):
                self.visit_region(block, in_parallel)

    def visit_scope(self, state: SDFGState, children: Dict[Any, List[nodes.Node]], entry: Optional[nodes.MapEntry],
                    in_parallel: bool) -> None:
        """Decide the nodes directly inside one scope, then descend into their scopes.

        :param state: the state holding the scope.
        :param children: ``state.scope_children()``, built once per state.
        :param entry: the scope's map entry, or ``None`` for the state's top level.
        :param in_parallel: whether a device-parallel map encloses this scope.
        """
        from dace.libraries.standard.helper import is_reentered_cpu_transfer
        for node in children[entry]:
            if isinstance(node, nodes.MapEntry):
                self.visit_scope(state, children, node, self.decide_map(node, state.sdfg, in_parallel))
            elif isinstance(node, nodes.LibraryNode):
                if node.schedule in CPU_PARALLEL_SCHEDULES and (in_parallel or is_reentered_cpu_transfer(
                        node, state, loop_cache=self.loop_cache)):
                    node.schedule = dtypes.ScheduleType.Sequential
                    self.pinned += 1
            elif isinstance(node, nodes.NestedSDFG) and node.sdfg is not None:
                self.visit_region(node.sdfg, in_parallel)

    def parameter_names(self, sdfg: SDFG) -> frozenset:
        """The names whose magnitude an extent may be read against: parameters, constants, iterators.

        Loop variables and map parameters are in here because an extent written in one -- a
        triangular sweep's ``LEN - it``, whose loop runs to a parameter -- is still an extent the
        PROGRAM TEXT bounds, so the existing "a symbol is assumed big" ruling owns it. What is left
        out is a scalar an interstate edge assigns from data.

        :param sdfg: the SDFG to read.
        :returns: names a compile-time verdict may rule on.
        """
        cached = self.params.get(sdfg.cfg_id)
        if cached is None:
            names = {str(sym) for sym in sdfg.free_symbols} | set(sdfg.constants)
            for region in sdfg.all_control_flow_regions(recursive=False):
                if isinstance(region, LoopRegion) and region.loop_variable:
                    names.add(region.loop_variable)
            for state in sdfg.states():
                for node in state.nodes():
                    if isinstance(node, nodes.MapEntry):
                        names.update(node.map.params)
            cached = frozenset(names)
            self.params[sdfg.cfg_id] = cached
        return cached

    def fewer_than_threshold(self, count) -> Optional[bool]:
        """Whether a trip count provably falls short of the break-even work (three-valued).

        :param count: the map's own iteration count, constant or symbolic.
        :returns: ``symbolic.ask('negative', count - threshold)``, computed once per count.
        """
        if count not in self.below_threshold:
            self.below_threshold[count] = symbolic.ask('negative', symbolic.simplify(count - self.threshold))
        return self.below_threshold[count]

    def worth_forking(self, count) -> bool:
        """Whether a map of ``count`` iterations earns its own OpenMP region (``True`` keeps it parallel).

        :param count: the map's own iteration count, constant or symbolic.
        """
        return self.threshold <= 0 or self.fewer_than_threshold(count) is not True

    def wants_runtime_guard(self, node: nodes.MapEntry, sdfg: SDFG) -> bool:
        """Whether ``node``'s trip count is a value the program computes rather than a parameter.

        :param node: the map entry, already decided to stay parallel.
        :param sdfg: the SDFG holding it.
        :returns: ``True`` when only a run-time test can rule on the fork.
        """
        count = node.map.range.num_elements()
        if self.threshold <= 0 or not symbolic.issymbolic(count):
            return False
        if self.fewer_than_threshold(count) is False:
            return False
        params = self.parameter_names(sdfg)
        return any(str(sym) not in params for sym in count.free_symbols)

    def decide_map(self, node: nodes.MapEntry, sdfg: SDFG, in_parallel: bool) -> bool:
        """Set ``node``'s schedule and report whether its body runs inside a parallel map.

        :param node: the map entry to decide.
        :param sdfg: the SDFG holding the map.
        :param in_parallel: whether a device-parallel map encloses this map.
        :returns: whether the map's body is enclosed by a parallel map.
        """
        if node.map.schedule not in CPU_PARALLEL_SCHEDULES:
            return in_parallel
        if in_parallel:
            if node.map.schedule == dtypes.ScheduleType.CPU_Multicore:
                node.map.schedule = dtypes.ScheduleType.Sequential
                self.pinned += 1
            return True
        if self.worth_forking(node.map.range.num_elements()):
            if self.wants_runtime_guard(node, sdfg):
                node.map.omp_min_parallel_iterations = self.threshold
                self.guarded += 1
            return True
        node.map.schedule = dtypes.ScheduleType.Sequential
        self.pinned += 1
        return False
