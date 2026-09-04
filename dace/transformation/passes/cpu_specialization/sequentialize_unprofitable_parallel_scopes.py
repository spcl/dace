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
   ``compiler.cpu.parallel_min_work_per_region``. Constants decide statically; a symbolic count is
   assumed big enough and stays parallel (the same default
   :func:`~dace.libraries.standard.helper.is_parallel_cpu_transfer_size` applies to transfers). No
   runtime guard is ever emitted -- a bare ``if`` clause on a combined ``parallel for simd``
   silently devectorizes.

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
from dace.sdfg.state import ControlFlowRegion, SDFGState
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


def worth_forking(iterations, threshold: int) -> bool:
    """Whether a map of ``iterations`` iterations earns its own OpenMP region.

    :param iterations: the map's own iteration count, constant or symbolic.
    :param threshold: break-even work per region, in elements; 0 disables the model.
    :returns: ``True`` to keep the region parallel.
    """
    if threshold <= 0:
        return True
    return symbolic.ask('negative', symbolic.simplify(iterations - threshold)) is not True


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
        :returns: how many scopes were pinned, or ``None`` if none were.
        """
        self.threshold = min_work_per_region()
        self.pinned = 0
        self.visit_region(sdfg, False)
        return self.pinned or None

    def visit_region(self, region: ControlFlowRegion, in_parallel: bool) -> None:
        """Walk ``region``'s blocks, threading whether a parallel map encloses them.

        :param region: the control-flow region (or nested SDFG) to walk.
        :param in_parallel: whether a device-parallel map encloses this region.
        """
        for block in region.nodes():
            if isinstance(block, SDFGState):
                self.visit_scope(block, block.scope_children(), None, in_parallel)
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
                self.visit_scope(state, children, node, self.decide_map(node, in_parallel))
            elif isinstance(node, nodes.LibraryNode):
                if node.schedule in CPU_PARALLEL_SCHEDULES and (in_parallel or is_reentered_cpu_transfer(node, state)):
                    node.schedule = dtypes.ScheduleType.Sequential
                    self.pinned += 1
            elif isinstance(node, nodes.NestedSDFG) and node.sdfg is not None:
                self.visit_region(node.sdfg, in_parallel)

    def decide_map(self, node: nodes.MapEntry, in_parallel: bool) -> bool:
        """Set ``node``'s schedule and report whether its body runs inside a parallel map.

        :param node: the map entry to decide.
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
        if worth_forking(node.map.range.num_elements(), self.threshold):
            return True
        node.map.schedule = dtypes.ScheduleType.Sequential
        self.pinned += 1
        return False
