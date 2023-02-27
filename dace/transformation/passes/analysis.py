# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.

from collections import defaultdict
from dace.transformation import pass_pipeline as ppl
from dace import SDFG, SDFGState, properties, InterstateEdge
from dace.sdfg import nodes as nd
from typing import Dict, Set, Tuple, Any, Optional, Union
import networkx as nx
from networkx.algorithms import shortest_paths as nxsp

WriteScopeDict = Dict[str, Dict[Optional[Tuple[SDFGState, nd.AccessNode]],
                                Set[Tuple[SDFGState, Union[nd.AccessNode, InterstateEdge]]]]]


@properties.make_properties
class StateReachability(ppl.Pass):
    """
    Evaluates state reachability (which other states can be executed after each state).
    """

    CATEGORY: str = 'Analysis'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Nothing

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        # If anything was modified, reapply
        return modified & ppl.Modifies.States

    def apply_pass(self, top_sdfg: SDFG, _) -> Dict[int, Dict[SDFGState, Set[SDFGState]]]:
        """
        :return: A dictionary mapping each state to its other reachable states.
        """
        reachable: Dict[int, Dict[SDFGState, Set[SDFGState]]] = {}
        for sdfg in top_sdfg.all_sdfgs_recursive():
            reachable[sdfg.sdfg_id] = {}
            tc: nx.DiGraph = nx.transitive_closure(sdfg.nx)
            for state in sdfg.nodes():
                reachable[sdfg.sdfg_id][state] = set(tc.successors(state))
        return reachable


@properties.make_properties
class AccessSets(ppl.Pass):
    """
    Evaluates memory access sets (which arrays/data descriptors are read/written in each state).
    """

    CATEGORY: str = 'Analysis'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Nothing

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        # If anything was modified, reapply
        return modified & ppl.Modifies.AccessNodes

    def apply_pass(self, top_sdfg: SDFG, _) -> Dict[int, Dict[SDFGState, Tuple[Set[str], Set[str]]]]:
        """
        :return: A dictionary mapping each state to a tuple of its (read, written) data descriptors.
        """
        top_result: Dict[int, Dict[SDFGState, Tuple[Set[str], Set[str]]]] = {}
        for sdfg in top_sdfg.all_sdfgs_recursive():
            result: Dict[SDFGState, Tuple[Set[str], Set[str]]] = {}
            for state in sdfg.nodes():
                readset, writeset = set(), set()
                for anode in state.data_nodes():
                    if state.in_degree(anode) > 0:
                        writeset.add(anode.data)
                    if state.out_degree(anode) > 0:
                        readset.add(anode.data)

                result[state] = (readset, writeset)

            # Edges that read from arrays add to both ends' access sets
            anames = sdfg.arrays.keys()
            for e in sdfg.edges():
                fsyms = e.data.free_symbols & anames
                if fsyms:
                    result[e.src][0].update(fsyms)
                    result[e.dst][0].update(fsyms)

            top_result[sdfg.sdfg_id] = result
        return top_result


@properties.make_properties
class FindAccessStates(ppl.Pass):
    """
    For each data descriptor, creates a set of states in which access nodes of that data are used.
    """

    CATEGORY: str = 'Analysis'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Nothing

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        # If anything was modified, reapply
        return modified & ppl.Modifies.AccessNodes

    def apply_pass(self, top_sdfg: SDFG, _) -> Dict[int, Dict[str, Set[SDFGState]]]:
        """
        :return: A dictionary mapping each data descriptor name to states where it can be found in.
        """
        top_result: Dict[int, Dict[str, Set[SDFGState]]] = {}

        for sdfg in top_sdfg.all_sdfgs_recursive():
            result: Dict[str, Set[SDFGState]] = defaultdict(set)
            for state in sdfg.nodes():
                for anode in state.data_nodes():
                    result[anode.data].add(state)

            # Edges that read from arrays add to both ends' access sets
            anames = sdfg.arrays.keys()
            for e in sdfg.edges():
                fsyms = e.data.free_symbols & anames
                for access in fsyms:
                    result[access].update({e.src, e.dst})

            top_result[sdfg.sdfg_id] = result
        return top_result


@properties.make_properties
class FindAccessNodes(ppl.Pass):

    CATEGORY: str = 'Analysis'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Nothing

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return modified & ppl.Modifies.AccessNodes

    def apply_pass(self, top_sdfg: SDFG,
                   _) -> Dict[int, Dict[str, Dict[SDFGState, Tuple[Set[nd.AccessNode], Set[nd.AccessNode]]]]]:
        """
        :return: A dictionary mapping each data descriptor name to a dictionary keyed by states with all access nodes
                 that use that data descriptor.
        """
        top_result: Dict[int, Dict[str, Set[nd.AccessNode]]] = dict()

        for sdfg in top_sdfg.all_sdfgs_recursive():
            result: Dict[str, Dict[SDFGState, Tuple[Set[nd.AccessNode], Set[nd.AccessNode]]]] = defaultdict(
                lambda: defaultdict(lambda: [set(), set()]))
            for state in sdfg.nodes():
                for anode in state.data_nodes():
                    if state.in_degree(anode) > 0:
                        result[anode.data][state][1].add(anode)
                    if state.out_degree(anode) > 0:
                        result[anode.data][state][0].add(anode)
            top_result[sdfg.sdfg_id] = result
        return top_result


@properties.make_properties
class ScalarWriteShadowScopes(ppl.Pass):
    """
    For each scalar or array of size 1, create a dictionary mapping each write to that data container
    to the set of reads that are shadowed / dominated by that write.
    """

    CATEGORY: str = 'Analysis'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Nothing

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        # If anything was modified, reapply
        return modified & ppl.Modifies.States

    def depends_on(self):
        return {AccessSets, FindAccessNodes}

    def _find_dominating_write(
        self, desc: str, state: SDFGState, read: Union[nd.AccessNode, InterstateEdge],
        access_nodes: Dict[SDFGState, Tuple[Set[nd.AccessNode], Set[nd.AccessNode]]],
        state_idom: Dict[SDFGState, SDFGState], access_sets: Dict[SDFGState, Tuple[Set[str], Set[str]]]
    ) -> Optional[Tuple[SDFGState, nd.AccessNode]]:
        write_state = None

        if isinstance(read, nd.AccessNode):
            # If the read is also a write, it shadows itself.
            iedges = state.in_edges(read)
            if len(iedges) > 0 and any(not e.data.is_empty() for e in iedges):
                return (state, read)

            # Find a dominating write within the same state.
            # TODO: Can this be done more efficiently?
            closest_candidate = None
            write_nodes = access_nodes[desc][state][1]
            for cand in write_nodes:
                if nxsp.has_path(state._nx, cand, read):
                    if closest_candidate is None or nxsp.has_path(state._nx, closest_candidate, cand):
                        closest_candidate = cand
            if closest_candidate is not None:
                return (state, closest_candidate)

            # Find the dominating write state if the current state is not the dominating write state.
            write_state = None
            nstate = state_idom[state] if state_idom[state] != state else None
            while nstate is not None and write_state is None:
                if desc in access_sets[nstate][1]:
                    write_state = nstate
                nstate = state_idom[nstate] if state_idom[nstate] != nstate else None
        elif isinstance(read, InterstateEdge):
            # Consider the current state as the write state, since the read is happening on an outgoing interstate edge.
            write_state = state

        # Find a dominating write in the write state, i.e., the 'last' write to the data container.
        if write_state is not None:
            closest_candidate = None
            for cand in access_nodes[desc][write_state][1]:
                if write_state.out_degree(cand) == 0:
                    closest_candidate = cand
                    break
                elif closest_candidate is None or nxsp.has_path(write_state._nx, closest_candidate, cand):
                    closest_candidate = cand
            if closest_candidate is not None:
                return (write_state, closest_candidate)

        return None

    def apply_pass(self, top_sdfg: SDFG, pipeline_results: Dict[str, Any]) -> Dict[int, WriteScopeDict]:
        """
        :return: A dictionary mapping each data descriptor name to a dictionary, where writes to that data descriptor
                 and the states they are contained in are mapped to the set of reads (and their states) that are in the
                 scope of that write.
        """
        top_result: Dict[int, WriteScopeDict] = dict()

        for sdfg in top_sdfg.all_sdfgs_recursive():
            result: WriteScopeDict = defaultdict(lambda: defaultdict(lambda: set()))
            idom = nx.immediate_dominators(sdfg.nx, sdfg.start_state)
            access_sets: Dict[SDFGState, Tuple[Set[str],
                                               Set[str]]] = pipeline_results[AccessSets.__name__][sdfg.sdfg_id]
            access_nodes: Dict[str, Dict[SDFGState, Tuple[Set[nd.AccessNode], Set[nd.AccessNode]]]] = pipeline_results[
                FindAccessNodes.__name__][sdfg.sdfg_id]

            anames = sdfg.arrays.keys()
            for desc in sdfg.arrays:
                desc_states_with_nodes = set(access_nodes[desc].keys())
                for state in desc_states_with_nodes:
                    for read_node in access_nodes[desc][state][0]:
                        write = self._find_dominating_write(desc, state, read_node, access_nodes, idom, access_sets)
                        result[desc][write].add((state, read_node))
                # Ensure accesses to interstate edges are also considered.
                for state, accesses in access_sets.items():
                    if desc in accesses[0]:
                        out_edges = sdfg.out_edges(state)
                        for oedge in out_edges:
                            syms = oedge.data.free_symbols & anames
                            if desc in syms:
                                write = self._find_dominating_write(desc, state, oedge, access_nodes, idom, access_sets)
                                result[desc][write].add((state, oedge))
            top_result[sdfg.sdfg_id] = result
        return top_result
