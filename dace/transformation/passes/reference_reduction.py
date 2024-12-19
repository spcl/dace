# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
from collections import defaultdict
import copy
from typing import Any, Dict, List, Optional, Set, Tuple

from dace import SDFG, SDFGState, data, properties, Memlet
from dace.sdfg import nodes
from dace.transformation import pass_pipeline as ppl, transformation
from dace.transformation.helpers import all_isedges_between
from dace.transformation.passes import analysis as ap


@properties.make_properties
@transformation.explicit_cf_compatible
class ReferenceToView(ppl.Pass):
    """
    Replaces Reference data descriptors that are only set to one source with views.
    """

    CATEGORY: str = 'Simplification'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Descriptors | ppl.Modifies.AccessNodes

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return modified & ppl.Modifies.AccessNodes

    def depends_on(self):
        return {ap.StateReachability, ap.FindAccessStates, ap.FindReferenceSources}

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> Optional[Set[str]]:
        """
        Removes redundant arrays and access nodes.

        :param sdfg: The SDFG to modify.
        :param pipeline_results: If in the context of a ``Pipeline``, a dictionary that is populated with prior Pass
                                 results as ``{Pass subclass name: returned object from pass}``. If not run in a
                                 pipeline, an empty dictionary is expected.
        :return: A set of removed data descriptor names, or None if nothing changed.
        """
        reachable: Dict[SDFGState, Set[SDFGState]] = pipeline_results[ap.StateReachability.__name__][sdfg.cfg_id]
        access_states: Dict[str, Set[SDFGState]] = pipeline_results[ap.FindAccessStates.__name__][sdfg.cfg_id]
        reference_sources: Dict[str, Set[Memlet]] = pipeline_results[ap.FindReferenceSources.__name__][sdfg.cfg_id]

        # Early exit if no references exist
        if not reference_sources:
            return None

        # Filter out multi-source references and tasklet-set references
        candidates = set(k for k, v in reference_sources.items()
                         if len(v) == 1 and not isinstance(next(iter(v)), nodes.CodeNode))

        refsets = self.find_refsets(candidates, access_states)

        result: Set[str] = self.find_candidates(sdfg, reference_sources, refsets, access_states, reachable)
        if not result:
            return None

        # Remove reference set edges and eliminate orphaned access nodes
        self.remove_refsets(result, refsets)

        # Reconnect reference uses as views
        self.reconnect_views(sdfg, result, access_states, reference_sources)

        # Modify data descriptor from Reference to View
        self.change_ref_descriptors_to_views(sdfg, result)

        return result or None

    def report(self, pass_retval: Set[str]) -> str:
        return f'Converted {len(pass_retval)} references to views: {pass_retval}.'

    def find_refsets(self, candidates: Set[str],
                     access_states: Dict[str, Set[SDFGState]]) -> Dict[str, List[Tuple[SDFGState, nodes.AccessNode]]]:
        """
        Returns a dictionary of reference name to a list of tuples of (state, access node)
        where the reference is set via a memlet.
        """
        result: Dict[str, List[Tuple[SDFGState, nodes.AccessNode]]] = defaultdict(list)
        all_states_to_consider: Set[SDFGState] = set()
        for candidate in candidates:
            all_states_to_consider.update(access_states[candidate])

        for state in all_states_to_consider:
            # Loop over all states that use the references once
            for node in state.data_nodes():
                if node.data not in candidates:
                    continue
                for _ in state.in_edges_by_connector(node, 'set'):
                    result[node.data].append((state, node))
                    break

        return result

    def find_candidates(
        self,
        sdfg: SDFG,
        reference_sources: Dict[str, Set[Memlet]],
        refsets: Dict[str, List[Tuple[SDFGState, nodes.AccessNode]]],
        access_states: Dict[str, Set[SDFGState]],
        reachable_states: Dict[SDFGState, Set[SDFGState]],
    ) -> Set[str]:
        """
        Returns a set of candidates for conversion to views.
        """
        result = set(refsets.keys())
        if not result:  # Early return
            return result

        # If memlet does not depend on any symbol, it can be kept. Otherwise,
        # it may depend on a (free) symbol. There are multiple options:
        #   * If dependent on scope symbol (e.g., map parameter) - remove from candidates
        #   * If dependent on symbol defined in inter-state edges - make sure it is not changed between set and uses
        #   * If dependent on a free symbol - also make sure it is not changed between set and uses
        for cand in list(result):  # Copying the set to a list allows us to iterate over it while removing elements
            source = next(iter(reference_sources[cand]))
            fsyms = source.subset.free_symbols
            if not fsyms:
                continue

            for state, node in refsets[cand]:
                # Check if any of the symbols is a scope symbol
                entry = state.entry_node(node)
                while entry is not None:
                    if fsyms & entry.new_symbols(sdfg, state, {}).keys():
                        result.remove(cand)
                        break
                    entry = state.entry_node(entry)
                if cand not in result:
                    break

                # Otherwise, they are only inter-state or free symbols. Test all paths to uses in different states
                # NOTE: This is an expensive check!
                for other_state in access_states[cand]:
                    # Filter self and unreachable states
                    if other_state is state or other_state not in reachable_states[state]:
                        continue
                    for e in all_isedges_between(state, other_state):
                        # The symbol was modified/reassigned in one of the paths, skip
                        if fsyms & e.data.assignments.keys():
                            result.remove(cand)
                            break
                    if cand not in result:
                        break

        return result

    def remove_refsets(
        self,
        candidates: Set[str],
        all_refsets: Dict[str, List[Tuple[SDFGState, nodes.AccessNode]]],
    ):
        for ref, refsets in all_refsets.items():
            if ref not in candidates:
                continue
            for state, node in refsets:
                # Loop over all states that use the reference and remove reference
                # set memlets, reconnecting the remaining surrounding nodes so as
                # to not break scopes
                edges_to_add = []
                edges_to_remove = set()
                nodes_to_remove = set()
                affected_nodes = set()
                for e in state.in_edges_by_connector(node, 'set'):
                    # This is a reference set edge. Consider scope and neighbors and remove set
                    if state.out_degree(e.dst) == 0:
                        edges_to_remove.add(e)
                        affected_nodes.add(e.src)
                        affected_nodes.add(e.dst)

                        # If source node does not have any other neighbors, it can be removed
                        if all(ee is e or ee.data.is_empty() for ee in state.all_edges(e.src)):
                            nodes_to_remove.add(e.src)
                        # If set reference does not have any other neighbors, it can be removed
                        if all(ee is e or ee.data.is_empty() for ee in state.all_edges(node)):
                            nodes_to_remove.add(node)

                        # If in a scope, ensure reference node will not be disconnected
                        scope = state.entry_node(node)
                        if scope is not None and node not in nodes_to_remove:
                            edges_to_add.append((scope, None, node, None, Memlet()))
                    else:  # Node has other neighbors, modify edge to become an empty memlet instead
                        e.dst_conn = None
                        e.dst.remove_in_connector('set')
                        e.data = Memlet()



                # Modify the state graph as necessary
                for e in edges_to_remove:
                    state.remove_memlet_path(e)
                for n in nodes_to_remove:
                    state.remove_node(n)
                for e in edges_to_add:
                    if len(state.edges_between(e[0], e[2])) == 0:
                        state.add_edge(*e)
                for n in affected_nodes:  # Orphaned nodes
                    if n in nodes_to_remove:
                        continue
                    if state.degree(n) == 0:
                        state.remove_node(n)

    def reconnect_views(self, sdfg: SDFG, candidates: Set[str], access_states: Dict[str, Set[SDFGState]],
                        reference_sources: Dict[str, Set[Memlet]]):
        all_states_to_consider: Set[SDFGState] = set()
        for cand in candidates:
            all_states_to_consider.update(access_states[cand])

        # For each instance of the access node, connect the original data container to the view
        for state in all_states_to_consider:
            for node in state.data_nodes():
                if node.data not in candidates:
                    continue
                refsource = next(iter(reference_sources[node.data]))

                needs_pred_view = any(not e.data.is_empty() for e in state.in_edges(node))
                needs_succ_view = any(not e.data.is_empty() for e in state.out_edges(node))
                if needs_pred_view:
                    self._create_view(refsource, state, node, predecessor=True)
                if needs_succ_view:
                    self._create_view(refsource, state, node, predecessor=False)

                # Replace node's data container with the reference source
                node.data = refsource.data

    def _create_view(self, refsource: Memlet, state: SDFGState, node: nodes.AccessNode, predecessor: bool):
        """
        Creates a view access node and redirects all the edges appropriately.
        """
        edges = state.in_edges if predecessor else state.out_edges
        view = state.add_access(node.data)
        src = (lambda e: e.src) if predecessor else (lambda _: view)
        dst = (lambda _: view) if predecessor else (lambda e: e.dst)

        # Redirect edges to view
        for e in edges(node):
            state.remove_edge(e)
            state.add_edge(src(e), e.src_conn, dst(e), e.dst_conn, e.data)

        # Use "views" connector to disambiguate potential corner cases
        if predecessor:
            view.add_out_connector('views')
            state.add_edge(view, 'views', node, None, copy.deepcopy(refsource))
        else:
            view.add_in_connector('views')
            state.add_edge(node, None, view, 'views', copy.deepcopy(refsource))

    def change_ref_descriptors_to_views(self, sdfg: SDFG, names: Set[str]):
        for name in names:
            sdfg.arrays[name] = data.View.view(sdfg.arrays[name])
