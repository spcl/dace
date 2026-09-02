# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
""" Marks the maps whose innermost loop can carry an OpenMP ``simd`` clause. """

from dataclasses import dataclass
from typing import Optional, Set, Tuple

import networkx as nx

from dace import SDFG, dtypes, properties
from dace.frontend import operations
from dace.sdfg import nodes
from dace.sdfg.state import LoopRegion, SDFGState
from dace.transformation import pass_pipeline as ppl, transformation
from dace.transformation.dataflow import MapExpansion

OMP_DIRECTIVE = '#pragma omp'


def map_body_is_leaf(state: SDFGState, map_entry: nodes.MapEntry) -> bool:
    """ True if the map's body holds no Map and no loop, at any depth.

        A NestedSDFG is opened rather than refused on sight: an outlined body is often just
        straight-line dataflow. Inside one a loop takes two forms, a ``LoopRegion`` and a back
        edge in a plain state machine; a ``ConditionalBlock`` lowers to an ``if`` the vectorizer
        masks, so it does not disqualify. Anything undecidable reads as not a leaf.
    """

    def loop_and_map_free(sdfg: SDFG, seen: Set[int]) -> bool:
        if id(sdfg) in seen:  # self-referential: refuse rather than recurse
            return False
        seen.add(id(sdfg))
        for cfr in sdfg.all_control_flow_regions():
            if isinstance(cfr, LoopRegion) or not nx.is_directed_acyclic_graph(cfr.nx):
                return False
        for st in sdfg.all_states():
            for n in st.nodes():
                if not node_is_loop_free(n, seen):
                    return False
        return True

    def node_is_loop_free(n: nodes.Node, seen: Set[int]) -> bool:
        if isinstance(n, nodes.MapEntry):
            return False
        # A library node still to be expanded is undecidable here: its expansion routinely
        # lowers to a Map, and an ``omp parallel for`` inside a simd region does not compile.
        if isinstance(n, nodes.LibraryNode):
            return False
        # Neither does a directive the tasklet brings itself -- the OpenMP Reduce expansion
        # writes ``#pragma omp parallel for`` straight into the tasklet body.
        if isinstance(n, nodes.Tasklet):
            return OMP_DIRECTIVE not in n.code.as_string
        if isinstance(n, nodes.NestedSDFG):
            return n.sdfg is not None and loop_and_map_free(n.sdfg, seen)
        return True

    try:
        map_exit = state.exit_node(map_entry)
    except (KeyError, StopIteration):
        return False
    return all(node_is_loop_free(n, set()) for n in state.all_nodes_between(map_entry, map_exit))


def body_wcr(state: SDFGState, map_entry: nodes.MapEntry):
    """ Every WCR the map body executes: the map exit's own in-edges, plus each edge inside a
        NestedSDFG the body holds, at any depth.

        The exit edges alone are not the whole story. A frontend-outlined body carries its
        accumulate INSIDE the nested SDFG (``oc[bsym] += w`` with the nested-SDFG-to-exit edge
        plain), and that write is a read-modify-write the clause would still let lanes
        interleave. Undecidable reads as "there is a WCR": the callers use this to WITHHOLD the
        clause, so missing one is a miscompile while an extra one only costs vectorization.
    """
    try:
        map_exit = state.exit_node(map_entry)
    except (KeyError, StopIteration):
        return []
    wcrs = [e.data.wcr for e in state.in_edges(map_exit) if e.data is not None and e.data.wcr is not None]
    for n in state.all_nodes_between(map_entry, map_exit):
        if not isinstance(n, nodes.NestedSDFG) or n.sdfg is None:
            continue
        for inner in n.sdfg.all_sdfgs_recursive():
            for ist in inner.all_states():
                wcrs += [e.data.wcr for e in ist.edges() if e.data is not None and e.data.wcr is not None]
    return wcrs


def map_has_minmax_wcr(state: SDFGState, map_entry: nodes.MapEntry) -> bool:
    """ True if a WCR the map body executes reduces with ``min``/``max``, whose NaN-preserving
        compare/branch combine vectorizers do not reliably fold.
    """
    return any(
        operations.detect_reduction_type(wcr) in (dtypes.ReductionType.Min, dtypes.ReductionType.Max)
        for wcr in body_wcr(state, map_entry))


def map_has_wcr(state: SDFGState, map_entry: nodes.MapEntry) -> bool:
    """ True if the map body executes any WCR.

        A Sequential map lowers WCR to a plain ``wcr_fixed::reduce``, a read-modify-write of the
        target in the loop body: an accumulation into a fixed location carries across iterations,
        and a scatter (``hist[bin(a[i])] += 1``) can alias across them. ``simd`` asserts neither
        happens, so a Sequential map that reduces gets no clause. CPU_Multicore goes through
        ``reduce_atomic`` instead, which composes with the clause.
    """
    return bool(body_wcr(state, map_entry))


@dataclass(unsafe_hash=True)
@properties.make_properties
@transformation.explicit_cf_compatible
class MarkSIMDMaps(ppl.Pass):
    """
    Sets ``Map.omp_simd`` on the CPU maps whose innermost loop can carry an OpenMP ``simd`` clause.
    Code generation only renders the clause; the safety analysis lives here.
    """

    CATEGORY: str = 'Optimization Preparation'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Nodes

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return modified & (ppl.Modifies.States | ppl.Modifies.Nodes | ppl.Modifies.Memlets)

    def apply_pass(self, sdfg: SDFG, _) -> Optional[Set[Tuple[int, str]]]:
        """
        Marks every qualifying map in ``sdfg`` and its nested SDFGs.

        :param sdfg: The SDFG to modify.
        :param _: Pipeline results, unused.
        :return: The marked maps as ``(state id, map label)``, or ``None`` if none qualified.
        """
        marked: Set[Tuple[int, str]] = set()
        for nsdfg in sdfg.all_sdfgs_recursive():
            for state in nsdfg.all_states():
                candidates = [
                    n for n in state.nodes() if isinstance(n, nodes.MapEntry) and self.map_takes_simd(state, n)
                ]
                for entry in candidates:
                    # The clause vectorizes the loop it precedes, so a multidimensional map has to
                    # give its innermost dimension a map of its own first. MapExpansion returns the
                    # new nest outermost-first, and propagates the scope it rewrote by itself, so
                    # ``annotate`` stays off: a whole-SDFG propagation per map is quadratic.
                    if len(entry.map.params) > 1:
                        entry = MapExpansion.apply_to(nsdfg,
                                                      options={
                                                          'expansion_limit': len(entry.map.params) - 1,
                                                          'inner_schedule': dtypes.ScheduleType.Sequential
                                                      },
                                                      map_entry=entry,
                                                      verify=False,
                                                      annotate=False,
                                                      save=False)[-1]
                    entry.map.omp_simd = True
                    marked.add((nsdfg.cfg_id, entry.map.label))
        return marked or None

    def map_takes_simd(self, state: SDFGState, map_entry: nodes.MapEntry) -> bool:
        """ Whether this map's innermost loop can carry the clause. """
        if map_entry.map.unroll:  # its own pragma, and an unrolled body is not a loop to vectorize
            return False
        if map_entry.map.collapse > 1:
            # HONOURED, not overridden. ``collapse(k)`` fuses k dimensions into one iteration space,
            # which is a request for their COMBINED trip count as thread parallelism, and it is the
            # opposite of what the clause needs -- fusing the loops leaves no inner loop to
            # vectorize. Taking the nest apart to vectorize would silently answer a different
            # question than the one asked, and it costs real parallelism when the outer dimension is
            # short: a ``[0:2, 0:1000000]`` nest goes from 2,000,000-way to 2-way. So an explicit
            # hint wins and this map keeps its collapsed, unvectorized form. Nothing in DaCe sets
            # the property (``Map.collapse`` defaults to 1), so this is the hand-written case only.
            return False
        if map_entry.map.schedule == dtypes.ScheduleType.CPU_Multicore:
            # A conflicted WCR write lowers through ``wcr_fixed::reduce_atomic``, which composes
            # with ``simd``, so only the min/max combine withholds the clause here.
            return map_body_is_leaf(state, map_entry) and not map_has_minmax_wcr(state, map_entry)
        if map_entry.map.schedule == dtypes.ScheduleType.Sequential:
            return map_body_is_leaf(state, map_entry) and not map_has_wcr(state, map_entry)
        return False
