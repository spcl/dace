# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The CPU fork/join cost model: the one place where a parallel scope is made sequential again.

The canonical form is the maximally parallel, device-neutral one: every DOALL loop is a Map and
every bulk transfer is an element map, with no fork/join reasoning anywhere in it. Schedules there
are labels, not decisions. Turning one of those maps back into a serial loop is a CPU
specialization -- it trades parallelism for the ``#pragma omp parallel`` a region would cost -- so
it lives here, in the ``cpu_specialize`` band, and nowhere else.

Three independent reasons to take the parallelism away, in the order they are checked:

1. **Nested parallelism.** A map (or library node) that already runs inside a parallel map of the
   same device schedule would open a team per outer iteration. Always sequential, at any size.
   This is the rule the old ``canonicalize/finalize.sequentialize_nested_parallel_scopes``
   implemented, and it is preserved exactly, transitively across nested-SDFG boundaries.
2. **Below break-even.** The region's own work is PROVABLY under
   ``compiler.cpu.parallel_min_work_per_region`` elements. Constants decide statically; a symbolic
   count is assumed big enough and stays parallel (the same default
   :func:`~dace.libraries.standard.helper.is_parallel_cpu_transfer_size` applies to transfers). No
   runtime guard is ever emitted -- a bare ``if`` clause on a combined ``parallel for simd``
   silently devectorizes.
3. **Re-entry.** An enclosing parallel map (always) or an enclosing loop that is not PROVABLY short
   re-enters the region, so its fork/join is paid once per entry. Note the asymmetry with rule 2: a
   symbolic MAP extent is big (stay parallel), a symbolic LOOP bound is long (a hazard). The region
   survives re-entry only if its work per entry outgrows the number of entries, both evaluated with
   EVERY symbol set to the same threshold value (all symbols are equally big, none outranks
   another): ``for t: map[i, j]`` over ``N x N`` keeps its map (``K*K > K``), while TSVC ``s115``'s
   ``for j: map[i in j+1:LEN_2D]`` loses it (``K - K - 1 <= K``), which is the 768-fork bucket.

The same rules decide bulk transfers, one pass over in
:mod:`~dace.transformation.passes.cpu_specialization.specialize_cpu_transfers`: a copy / memset
library node carries no map to measure until it expands, so the re-entry half is applied to it
directly (that is npbench ``stockham_fft``'s 349,525-entry inner copy).

Setting the threshold to 0 turns rules 2 and 3 off, leaving only the nested-parallelism rule -- the
A/B lever for measuring the cost model itself.
"""
from typing import Any, Dict, List, Optional

from dace import SDFG, dtypes, properties, symbolic
from dace.config import Config
from dace.sdfg import nodes
from dace.sdfg.state import ControlFlowRegion, LoopRegion, SDFGState
from dace.transformation import pass_pipeline as ppl

#: Schedules a CPU map or library node still parallelizes under. ``Default`` is included because
#: canonicalization leaves the schedule unset: codegen infers it, and a top-level ``Default`` map
#: becomes ``CPU_Multicore``. Anything else (GPU schedules, ``Unrolled``, ``CPU_Persistent``) is
#: someone else's decision and is left untouched.
CPU_PARALLEL_SCHEDULES = (dtypes.ScheduleType.Default, dtypes.ScheduleType.CPU_Multicore)

#: Stand-in for a trip count the loop analysis cannot read. It is a symbol, so the "every symbol is
#: big enough" default applies to it and the enclosing region stays parallel.
UNKNOWN_TRIPS = symbolic.symbol('__fork_join_unknown_trips')


def min_work_per_region() -> int:
    """The configured break-even work of one OpenMP region, in elements.

    :returns: ``compiler.cpu.parallel_min_work_per_region``; 0 disables the cost model.
    """
    return int(Config.get('compiler', 'cpu', 'parallel_min_work_per_region'))


def at_common_symbol(expr, value: int):
    """Evaluate ``expr`` with every free symbol set to ``value``.

    Implements "all symbols are equivalent in magnitude, and all of them are big": a comparison
    between two symbolic counts carries no information unless both are read at the same symbol
    value, so both sides are evaluated at the break-even threshold. A loop iterator inside the
    expression is read the same way -- it is bounded by a symbol of that magnitude.

    :param expr: a symbolic count.
    :param value: the value every free symbol takes.
    :returns: the numeric value, or ``None`` if it does not evaluate to a number.
    """
    try:
        substituted = symbolic.simplify(symbolic.simplify(expr).subs({s: value for s in expr.free_symbols}))
        return substituted if substituted.is_number else None
    except (TypeError, ValueError, AttributeError, ArithmeticError, RecursionError):
        return None


def loop_trip_count(loop: LoopRegion):
    """Ascending trip count of ``loop``, or :data:`UNKNOWN_TRIPS` when it cannot be read.

    :param loop: the loop region to measure.
    :returns: a symbolic trip count.
    """
    from dace.transformation.passes.analysis import loop_analysis
    start = loop_analysis.get_init_assignment(loop)
    end = loop_analysis.get_loop_end(loop)
    stride = loop_analysis.get_loop_stride(loop)
    if start is None or end is None or stride is None:
        return UNKNOWN_TRIPS
    if symbolic.ask('positive', symbolic.simplify(stride)) is not True:
        return UNKNOWN_TRIPS
    return symbolic.int_floor(end - start, stride) + 1


def scope_work(state: SDFGState, children: Dict[Any, List[nodes.Node]], entry: nodes.MapEntry):
    """Elements one entry of ``entry``'s scope processes: its own volume times its body's.

    The body factor is what separates ``for t: map[i] { for k: ... }`` (polybench ``adi``: a real
    ``N*N`` of work per entry) from a bare ``map[i]`` inside the same loop. Sibling scopes are
    MULTIPLIED rather than summed -- an over-estimate, which biases the verdict towards keeping the
    parallelism.

    :param state: the state holding the scope.
    :param children: ``state.scope_children()``, built once by the caller.
    :param entry: the map entry whose scope is measured.
    :returns: a symbolic element count.
    """
    work = entry.map.range.num_elements()
    for node in children[entry]:
        if isinstance(node, nodes.MapEntry):
            work = work * scope_work(state, children, node)
        elif isinstance(node, nodes.NestedSDFG) and node.sdfg is not None:
            work = work * region_work(node.sdfg)
    return work


def region_work(region: ControlFlowRegion):
    """Elements one entry of ``region`` processes, counting loop trips and map volumes.

    :param region: the control-flow region (or nested SDFG) to measure.
    :returns: a symbolic element count.
    """
    work = 1
    for block in region.nodes():
        if isinstance(block, SDFGState):
            children = block.scope_children()
            for node in children[None]:
                if isinstance(node, nodes.MapEntry):
                    work = work * scope_work(block, children, node)
                elif isinstance(node, nodes.NestedSDFG) and node.sdfg is not None:
                    work = work * region_work(node.sdfg)
        elif isinstance(block, LoopRegion):
            work = work * loop_trip_count(block) * region_work(block)
        elif isinstance(block, ControlFlowRegion):
            work = work * region_work(block)
    return work


def worth_forking(work, entries, reentered: bool, threshold: int) -> bool:
    """Whether a parallel region carrying ``work`` elements per entry earns its fork/join.

    :param work: elements processed by one entry of the region.
    :param entries: how many times the region is entered per call (product of enclosing trip
        counts and sequential map volumes).
    :param reentered: whether any enclosing loop / sequential map re-enters it more than a
        provably short number of times.
    :param threshold: break-even work per region, in elements; 0 disables the model.
    :returns: ``True`` to keep the region parallel.
    """
    if threshold <= 0:
        return True
    if symbolic.ask('negative', symbolic.simplify(work - threshold)) is True:
        return False
    if not reentered:
        return True
    per_entry = at_common_symbol(work, threshold)
    total_entries = at_common_symbol(entries, threshold)
    if per_entry is None or total_entries is None:
        return True
    return bool(per_entry > total_entries)


class ScopeContext:
    """What the enclosing scopes of a node imply for its fork/join cost."""

    __slots__ = ('in_parallel', 'entries', 'reentered')

    def __init__(self, in_parallel: bool = False, entries=1, reentered: bool = False):
        self.in_parallel = in_parallel  #: inside a device-parallel map (across nsdfg boundaries)
        self.entries = entries  #: symbolic number of entries per call
        self.reentered = reentered  #: entered more than a provably short number of times

    def under(self, factor, is_short: bool, parallel: bool = False) -> 'ScopeContext':
        """The context inside a scope entered ``factor`` times.

        :param factor: the enclosing scope's trip count / volume.
        :param is_short: whether that count is provably short.
        :param parallel: whether the enclosing scope is a parallel map.
        :returns: the nested context.
        """
        return ScopeContext(self.in_parallel or parallel, self.entries * factor, self.reentered or parallel
                            or not is_short)


@properties.make_properties
class SequentializeParallelScopes(ppl.Pass):
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
        self.visit_region(sdfg, ScopeContext())
        return self.pinned or None

    def visit_region(self, region: ControlFlowRegion, ctx: ScopeContext) -> None:
        """Walk ``region``'s blocks, threading the enclosing-scope context.

        :param region: the control-flow region (or nested SDFG) to walk.
        :param ctx: what the enclosing scopes imply.
        """
        from dace.libraries.standard.helper import is_short_loop
        for block in region.nodes():
            if isinstance(block, SDFGState):
                self.visit_state(block, ctx)
            elif isinstance(block, LoopRegion):
                self.visit_region(block, ctx.under(loop_trip_count(block), is_short_loop(block)))
            elif isinstance(block, ControlFlowRegion):
                self.visit_region(block, ctx)

    def visit_state(self, state: SDFGState, ctx: ScopeContext) -> None:
        """Walk one state's scope tree, deciding every map and library node in it.

        :param state: the state to walk.
        :param ctx: what the enclosing scopes imply.
        """
        self.visit_scope(state, state.scope_children(), None, ctx)

    def visit_scope(self, state: SDFGState, children: Dict[Any, List[nodes.Node]], entry: Optional[nodes.MapEntry],
                    ctx: ScopeContext) -> None:
        """Decide the nodes directly inside one scope, then descend into their scopes.

        :param state: the state holding the scope.
        :param children: ``state.scope_children()``, built once per state.
        :param entry: the scope's map entry, or ``None`` for the state's top level.
        :param ctx: what the enclosing scopes imply.
        """
        from dace.libraries.standard.helper import is_reentered_cpu_transfer
        for node in children[entry]:
            if isinstance(node, nodes.MapEntry):
                self.visit_scope(state, children, node, self.decide_map(state, children, node, ctx))
            elif isinstance(node, nodes.LibraryNode):
                if node.schedule in CPU_PARALLEL_SCHEDULES and (ctx.in_parallel
                                                                or is_reentered_cpu_transfer(node, state)):
                    node.schedule = dtypes.ScheduleType.Sequential
                    self.pinned += 1
            elif isinstance(node, nodes.NestedSDFG) and node.sdfg is not None:
                self.visit_region(node.sdfg, ctx)

    def decide_map(self, state: SDFGState, children: Dict[Any, List[nodes.Node]], node: nodes.MapEntry,
                   ctx: ScopeContext) -> ScopeContext:
        """Set ``node``'s schedule and return the context its body runs under.

        :param state: the state holding the map.
        :param children: ``state.scope_children()``.
        :param node: the map entry to decide.
        :param ctx: what the enclosing scopes imply.
        :returns: the context for the map's body.
        """
        volume = node.map.range.num_elements()
        if node.map.schedule not in CPU_PARALLEL_SCHEDULES:
            return ctx.under(volume, self.is_short_count(volume))
        if ctx.in_parallel:
            if node.map.schedule == dtypes.ScheduleType.CPU_Multicore:
                node.map.schedule = dtypes.ScheduleType.Sequential
                self.pinned += 1
            return ctx.under(volume, self.is_short_count(volume))
        if worth_forking(scope_work(state, children, node), ctx.entries, ctx.reentered, self.threshold):
            return ctx.under(volume, self.is_short_count(volume), parallel=True)
        node.map.schedule = dtypes.ScheduleType.Sequential
        self.pinned += 1
        return ctx.under(volume, self.is_short_count(volume))

    def is_short_count(self, count) -> bool:
        """Whether a sequential scope of ``count`` iterations re-enters its body few enough times
        to ignore -- the map counterpart of
        :func:`~dace.libraries.standard.helper.is_short_loop`.

        :param count: the scope's iteration count.
        :returns: ``True`` only when provably short.
        """
        from dace.libraries.standard.helper import REENTRY_SHORT_LOOP_TRIPS
        return symbolic.ask('negative', symbolic.simplify(count - REENTRY_SHORT_LOOP_TRIPS)) is True
