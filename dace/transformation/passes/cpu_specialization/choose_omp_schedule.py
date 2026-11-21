# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""CPU specialization of OpenMP loop schedules: ``guided`` for a parallel map whose iterations do uneven work.

A ``CPU_Multicore`` map with the default schedule is split into equal static chunks. That is right when every
iteration costs the same, and wrong when an iteration's work depends on the data it reads: a loop over a CSR row
(``for k in range(ptr[i], ptr[i + 1])``), a ``while`` loop, or an inner map whose range is read from memory. Then a
static chunk of heavy iterations decides the runtime (spgemm_hash's numeric phase, whose rows are ordered by size:
637 ms static, 337 ms guided). So it is when an inner trip count grows or shrinks with the iteration, a triangle
(tsvc_2_s1232's ``range(j * VLEN, n)``, 7.4 ms static against 2.9 ms before the interchange put the long rows last).
Such a map gets ``schedule(guided)``; every other map keeps its schedule.
"""

from typing import Any

from dace import SDFG, dtypes, symbolic
from dace.sdfg import nodes
from dace.sdfg.state import LoopRegion, SDFGState
from dace.transformation import pass_pipeline as ppl
from dace.transformation.passes.analysis import loop_analysis


def data_bound_names(sdfg: SDFG) -> set[str]:
    """The symbols an interstate edge of ``sdfg`` (or of an SDFG nested in it) sets from a container read."""
    names = set()
    for sd in sdfg.all_sdfgs_recursive():
        for edge in sd.all_interstate_edges():
            for name, value in edge.data.assignments.items():
                if symbolic.pystr_to_symbolic(value).atoms(symbolic.Subscript):
                    names.add(name)
    return names


def reads_data(expr: Any, data_bound: set[str]) -> bool:
    """Whether ``expr`` reads a container directly or through a data-bound symbol."""
    if expr is None:
        return True
    return bool(expr.atoms(symbolic.Subscript)) or any(str(s) in data_bound for s in expr.free_symbols)


def uneven_loop(loop: LoopRegion, data_bound: set[str], iteration: set[str] = frozenset()) -> bool:
    """Whether ``loop``'s trip count can differ between iterations of the map around it: a ``while`` loop, bounds
    that read memory, or a trip count naming one of the ``iteration`` symbols (the map's parameters, as the loop's
    SDFG names them)."""
    if not loop.loop_variable:
        return True
    start, end = loop_analysis.get_init_assignment(loop), loop_analysis.get_loop_end(loop)
    if reads_data(start, data_bound) or reads_data(end, data_bound):
        return True
    return names_any(end - start, iteration)


def names_any(expr: Any, names: set[str]) -> bool:
    """Whether ``expr`` reads a symbol of ``names``."""
    return any(str(s) in names for s in symbolic.pystr_to_symbolic(expr).free_symbols)


def uneven_body(state: SDFGState, entry: nodes.MapEntry, data_bound: set[str]) -> bool:
    """Whether the work of an iteration of ``entry``'s map depends on the iteration: an inner map with a dynamic
    range input or an extent naming the map's parameters, or a nested SDFG holding a loop whose trip count is uneven
    (:func:`uneven_loop`)."""
    params = set(entry.map.params)
    for node in state.scope_subgraph(entry, include_entry=False, include_exit=False).nodes():
        if isinstance(node, nodes.MapEntry) and (
            any(not c.startswith("IN_") for c in node.in_connectors)
            or any(names_any(extent, params) for extent in node.map.range.size())
        ):
            return True
        if isinstance(node, nodes.NestedSDFG):
            inner = data_bound | data_bound_names(node.sdfg)
            iteration = {name for name, value in node.symbol_mapping.items() if names_any(value, params)}
            for sd in node.sdfg.all_sdfgs_recursive():
                regions = [r for r in sd.all_control_flow_regions() if isinstance(r, LoopRegion)]
                if any(uneven_loop(r, inner, iteration) for r in regions):
                    return True
    return False


class ChooseOmpSchedule(ppl.Pass):
    """Give every ``CPU_Multicore`` map with the default OpenMP schedule whose iterations do uneven work
    ``schedule(guided)``."""

    CATEGORY: str = "Optimization Preparation"

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Nodes

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return modified & ppl.Modifies.Nodes

    def apply_pass(self, sdfg: SDFG, _: dict[str, Any]) -> int | None:
        """:returns: The number of maps given a guided schedule, or ``None`` if none was."""
        chosen = 0
        data_bound = data_bound_names(sdfg)
        for node, state in sdfg.all_nodes_recursive():
            if not isinstance(node, nodes.MapEntry) or not isinstance(state, SDFGState):
                continue
            if node.map.schedule != dtypes.ScheduleType.CPU_Multicore:
                continue
            if node.map.omp_schedule != dtypes.OMPScheduleType.Default:
                continue
            if uneven_body(state, node, data_bound):
                node.map.omp_schedule = dtypes.OMPScheduleType.Guided
                chosen += 1
        return chosen or None
