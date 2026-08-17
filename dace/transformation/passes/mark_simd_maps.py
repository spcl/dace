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


def map_has_minmax_wcr(state: SDFGState, map_entry: nodes.MapEntry) -> bool:
    """ True if a WCR out-edge of this map's exit reduces with ``min``/``max``, whose
        NaN-preserving compare/branch combine vectorizers do not reliably fold.
    """
    try:
        map_exit = state.exit_node(map_entry)
    except (KeyError, StopIteration):
        return False
    for iedge in state.in_edges(map_exit):
        if iedge.data is None or iedge.data.wcr is None:
            continue
        if operations.detect_reduction_type(iedge.data.wcr) in (dtypes.ReductionType.Min, dtypes.ReductionType.Max):
            return True
    return False


def map_has_wcr(state: SDFGState, map_entry: nodes.MapEntry) -> bool:
    """ True if any out-edge of this map's exit carries a WCR.

        A Sequential map lowers WCR to a plain ``wcr_fixed::reduce``, a read-modify-write of the
        target in the loop body: an accumulation into a fixed location carries across iterations,
        and a scatter (``hist[bin(a[i])] += 1``) can alias across them. ``simd`` asserts neither
        happens, so a Sequential map that reduces gets no clause. CPU_Multicore goes through
        ``reduce_atomic`` instead, which composes with the clause.
    """
    try:
        map_exit = state.exit_node(map_entry)
    except (KeyError, StopIteration):
        return False
    return any(e.data is not None and e.data.wcr is not None for e in state.in_edges(map_exit))


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
                    # give its innermost dimension a map of its own first.
                    if len(entry.map.params) > 1:
                        MapExpansion.apply_to(nsdfg,
                                              options={
                                                  'expansion_limit': len(entry.map.params) - 1,
                                                  'inner_schedule': dtypes.ScheduleType.Sequential
                                              },
                                              map_entry=entry,
                                              verify=False,
                                              save=False)
                        entry = self.innermost_entry(state, entry)
                    entry.map.omp_simd = True
                    marked.add((nsdfg.cfg_id, entry.map.label))
        return marked or None

    def innermost_entry(self, state: SDFGState, entry: nodes.MapEntry) -> nodes.MapEntry:
        """ The innermost MapEntry of the nest ``entry`` now heads, after expansion. """
        scope = state.scope_children()
        while True:
            inner = [n for n in scope[entry] if isinstance(n, nodes.MapEntry)]
            if not inner:
                return entry
            entry = inner[0]

    def map_takes_simd(self, state: SDFGState, map_entry: nodes.MapEntry) -> bool:
        """ Whether this map's innermost loop can carry the clause. """
        if map_entry.map.unroll:  # its own pragma, and an unrolled body is not a loop to vectorize
            return False
        if map_entry.map.schedule == dtypes.ScheduleType.CPU_Multicore:
            # A conflicted WCR write lowers through ``wcr_fixed::reduce_atomic``, which composes
            # with ``simd``, so only the min/max combine withholds the clause here.
            return map_body_is_leaf(state, map_entry) and not map_has_minmax_wcr(state, map_entry)
        if map_entry.map.schedule == dtypes.ScheduleType.Sequential:
            return map_body_is_leaf(state, map_entry) and not map_has_wcr(state, map_entry)
        return False
