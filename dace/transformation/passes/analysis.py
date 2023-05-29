# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.

from collections import defaultdict
from dace.transformation import pass_pipeline as ppl
from dace import SDFG, SDFGState, properties, InterstateEdge
from dace.sdfg.graph import Edge
from dace.sdfg import nodes as nd
from dace.sdfg.analysis import cfg
from typing import Dict, Set, Tuple, Any, Optional, Union
import networkx as nx
from networkx.algorithms import shortest_paths as nxsp
import dace.subsets as subsets
from dace.symbolic import issymbolic, pystr_to_symbolic, simplify
from dace.sdfg.graph import MultiConnectorEdge
import sympy
from dace.sdfg.propagation_underapproximation import UnderapproximateWrites
from dace import Memlet


WriteScopeDict = Dict[str, Dict[Optional[Tuple[SDFGState, nd.AccessNode]],
                                Set[Tuple[SDFGState, Union[nd.AccessNode, InterstateEdge]]]]]
ArrayWriteScopeDict = Dict[str, Dict[Optional[set[Tuple[SDFGState, set[nd.AccessNode]]]], 
                                Set[Tuple[SDFGState, Union[nd.AccessNode, InterstateEdge]]]]]
SymbolScopeDict = Dict[str, Dict[Edge[InterstateEdge],
                                 Set[Union[Edge[InterstateEdge], SDFGState]]]]


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
class SymbolAccessSets(ppl.Pass):
    """
    Evaluates symbol access sets (which symbols are read/written in each state or interstate edge).
    """

    CATEGORY: str = 'Analysis'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Nothing

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        # If anything was modified, reapply
        return modified & ppl.Modifies.States | ppl.Modifies.Edges | ppl.Modifies.Symbols | ppl.Modifies.Nodes

    def apply_pass(
            self, top_sdfg: SDFG, _
    ) -> Dict[int, Dict[Union[SDFGState, Edge[InterstateEdge]], Tuple[Set[str], Set[str]]]]:
        """
        :return: A dictionary mapping each state to a tuple of its (read, written) data descriptors.
        """
        top_result: Dict[int, Dict[SDFGState, Tuple[Set[str], Set[str]]]] = {}
        for sdfg in top_sdfg.all_sdfgs_recursive():
            adesc = set(sdfg.arrays.keys())
            result: Dict[SDFGState, Tuple[Set[str], Set[str]]] = {}
            for state in sdfg.nodes():
                readset = state.free_symbols
                # No symbols may be written to inside states.
                result[state] = (readset, set())
                for oedge in sdfg.out_edges(state):
                    edge_readset = oedge.data.read_symbols() - adesc
                    edge_writeset = set(oedge.data.assignments.keys())
                    result[oedge] = (edge_readset, edge_writeset)
            top_result[sdfg.sdfg_id] = result
        return top_result


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
    """
    For each data descriptor, creates a dictionary mapping states to all read and write access nodes with the given
    data descriptor.
    """

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
class SymbolWriteScopes(ppl.Pass):
    """
    For each symbol, create a dictionary mapping each writing interstate edge to that symbol to the set of interstate
    edges and states reading that symbol that are dominated by that write.
    """

    CATEGORY: str = 'Analysis'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Nothing

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        # If anything was modified, reapply
        return modified & ppl.Modifies.Symbols | ppl.Modifies.States | ppl.Modifies.Edges | ppl.Modifies.Nodes

    def depends_on(self):
        return {SymbolAccessSets, StateReachability}

    def _find_dominating_write(
            self, sym: str, read: Union[SDFGState, Edge[InterstateEdge]], state_idom: Dict[SDFGState, SDFGState]
    ) -> Optional[Edge[InterstateEdge]]:
        last_state: SDFGState = read if isinstance(
            read, SDFGState) else read.src

        in_edges = last_state.parent.in_edges(last_state)
        deg = len(in_edges)
        if deg == 0:
            return None
        elif deg == 1 and any([sym == k for k in in_edges[0].data.assignments.keys()]):
            return in_edges[0]

        write_isedge = None
        n_state = state_idom[last_state] if state_idom[last_state] != last_state else None
        while n_state is not None and write_isedge is None:
            oedges = n_state.parent.out_edges(n_state)
            odeg = len(oedges)
            if odeg == 1:
                if any([sym == k for k in oedges[0].data.assignments.keys()]):
                    write_isedge = oedges[0]
            else:
                dom_edge = None
                for cand in oedges:
                    if nxsp.has_path(n_state.parent.nx, cand.dst, last_state):
                        if dom_edge is not None:
                            dom_edge = None
                            break
                        elif any([sym == k for k in cand.data.assignments.keys()]):
                            dom_edge = cand
                write_isedge = dom_edge
            n_state = state_idom[n_state] if state_idom[n_state] != n_state else None
        return write_isedge

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> Dict[int, SymbolScopeDict]:
        top_result: Dict[int, SymbolScopeDict] = dict()

        for sdfg in sdfg.all_sdfgs_recursive():
            result: SymbolScopeDict = defaultdict(
                lambda: defaultdict(lambda: set()))

            idom = nx.immediate_dominators(sdfg.nx, sdfg.start_state)
            all_doms = cfg.all_dominators(sdfg, idom)
            symbol_access_sets: Dict[
                Union[SDFGState, Edge[InterstateEdge]
                      ], Tuple[Set[str], Set[str]]
            ] = pipeline_results[SymbolAccessSets.__name__][sdfg.sdfg_id]
            state_reach: Dict[SDFGState, Set[SDFGState]
                              ] = pipeline_results[StateReachability.__name__][sdfg.sdfg_id]

            for read_loc, (reads, _) in symbol_access_sets.items():
                for sym in reads:
                    dominating_write = self._find_dominating_write(
                        sym, read_loc, idom)
                    result[sym][dominating_write].add(
                        read_loc if isinstance(read_loc, SDFGState) else read_loc)

            # If any write A is dominated by another write B and any reads in B's scope are also reachable by A,
            # then merge A and its scope into B's scope.
            to_remove = set()
            for sym in result.keys():
                for write, accesses in result[sym].items():
                    if write is None:
                        continue
                    dominators = all_doms[write.dst]
                    reach = state_reach[write.dst]
                    for dom in dominators:
                        iedges = dom.parent.in_edges(dom)
                        if len(iedges) == 1 and iedges[0] in result[sym]:
                            other_accesses = result[sym][iedges[0]]
                            coarsen = False
                            for a_state_or_edge in other_accesses:
                                if isinstance(a_state_or_edge, SDFGState):
                                    if a_state_or_edge in reach:
                                        coarsen = True
                                        break
                                else:
                                    if a_state_or_edge.src in reach:
                                        coarsen = True
                                        break
                            if coarsen:
                                other_accesses.update(accesses)
                                other_accesses.add(write)
                                to_remove.add((sym, write))
                                result[sym][write] = set()
            for sym, write in to_remove:
                del result[sym][write]

            top_result[sdfg.sdfg_id] = result
        return top_result


@properties.make_properties
class ScalarWriteShadowScopes(ppl.Pass):
    """
    For each scalar or array of size 1, create a dictionary mapping writes to that data container to the set of reads
    and writes that are dominated by that write.
    """

    CATEGORY: str = 'Analysis'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Nothing

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        # If anything was modified, reapply
        return modified & ppl.Modifies.States

    def depends_on(self):
        return {AccessSets, FindAccessNodes, StateReachability}

    # find dominating write given a descriptor and a read access node in a state
    def _find_dominating_write(
        self, desc: str, state: SDFGState, read: Union[nd.AccessNode, InterstateEdge],
        access_nodes: Dict[SDFGState, Tuple[Set[nd.AccessNode], Set[nd.AccessNode]]],
        state_idom: Dict[SDFGState, SDFGState], access_sets: Dict[SDFGState, Tuple[Set[str], Set[str]]],
        no_self_shadowing: bool = False
    ) -> Optional[Tuple[SDFGState, nd.AccessNode]]:
        if isinstance(read, nd.AccessNode):
            # If the read is also a write, it shadows itself.
            iedges = state.in_edges(read)
            if len(iedges) > 0 and any(not e.data.is_empty() for e in iedges) and not no_self_shadowing:
                return (state, read)

            # Find a dominating write within the same state.
            # TODO: Can this be done more efficiently?
            closest_candidate = None
            write_nodes = access_nodes[desc][state][1]
            # iterate over all the candidate writes to find closest write to read
            for cand in write_nodes:
                if cand != read and nxsp.has_path(state._nx, cand, read):
                    if closest_candidate is None or nxsp.has_path(state._nx, closest_candidate, cand):
                        closest_candidate = cand
            if closest_candidate is not None:
                return (state, closest_candidate)
        elif isinstance(read, InterstateEdge):
            # Attempt to find a shadowing write in the current state.
            # TODO: Can this be done more efficiently?
            closest_candidate = None
            write_nodes = access_nodes[desc][state][1]
            for cand in write_nodes:
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
                 and the states they are contained in are mapped to the set of reads and writes (and their states) that
                 are dominated by that write.
        """
        top_result: Dict[int, WriteScopeDict] = dict()

        for sdfg in top_sdfg.all_sdfgs_recursive():
            result: WriteScopeDict = defaultdict(
                lambda: defaultdict(lambda: set()))
            idom = nx.immediate_dominators(sdfg.nx, sdfg.start_state)
            all_doms = cfg.all_dominators(sdfg, idom)
            access_sets: Dict[SDFGState, Tuple[Set[str],
                                               Set[str]]] = pipeline_results[AccessSets.__name__][sdfg.sdfg_id]
            # get mapping from data descriptor to mapping from states to all read/write access nodes with the given descriptor
            access_nodes: Dict[str, Dict[SDFGState, Tuple[Set[nd.AccessNode], Set[nd.AccessNode]]]] = pipeline_results[
                FindAccessNodes.__name__][sdfg.sdfg_id]
            # mapping from state to other states that can be executed after given state
            state_reach: Dict[SDFGState, Set[SDFGState]
                              ] = pipeline_results[StateReachability.__name__][sdfg.sdfg_id]

            anames = sdfg.arrays.keys()
            # iterate over all data descriptors
            for desc in sdfg.arrays:
                # all states that contain a read/write access node to desc
                desc_states_with_nodes = set(access_nodes[desc].keys())
                # iterate over all states that contain a read/write node to desc
                for state in desc_states_with_nodes:
                    # iterate over all the read nodes in the current state
                    for read_node in access_nodes[desc][state][0]:
                        # find the dominating write for the current read_node
                        write = self._find_dominating_write(
                            desc, state, read_node, access_nodes, idom, access_sets)
                        result[desc][write].add((state, read_node))
                # Ensure accesses to interstate edges are also considered.
                # iterate over all states and the access sets they are mapped to
                for state, accesses in access_sets.items():
                    # check if the current descriptor is in the read set of the current state
                    if desc in accesses[0]:
                        # get the outgoing CFG-edges of the current state
                        out_edges = sdfg.out_edges(state)
                        # iterate over the outgoing insterstate edges
                        for oedge in out_edges:
                            # intersect the names of the arrays with the name of the free symbols in the edge
                            syms = oedge.data.free_symbols & anames
                            # if the intersection contains the name of the descriptor find the dominating write
                            if desc in syms:
                                write = self._find_dominating_write(
                                    desc, state, oedge.data, access_nodes, idom, access_sets
                                )
                                result[desc][write].add((state, oedge.data))
                # Take care of any write nodes that have not been assigned to a scope yet, i.e., writes that are not
                # dominating any reads and are thus not part of the results yet.
                for state in desc_states_with_nodes:
                    for write_node in access_nodes[desc][state][1]:
                        if not (state, write_node) in result[desc]:
                            write = self._find_dominating_write(
                                desc, state, write_node, access_nodes, idom, access_sets, no_self_shadowing=True
                            )
                            result[desc][write].add((state, write_node))

                # If any write A is dominated by another write B and any reads in B's scope are also reachable by A,
                # then merge A and its scope into B's scope.
                to_remove = set()
                for write, accesses in result[desc].items():
                    if write is None:
                        continue
                    write_state, write_node = write
                    dominators = all_doms[write_state]
                    reach = state_reach[write_state]
                    for other_write, other_accesses in result[desc].items():
                        if other_write is not None and other_write[1] is write_node and other_write[0] is write_state:
                            continue
                        if other_write is None or other_write[0] in dominators:
                            noa = len(other_accesses)
                            if noa > 0 and (noa > 1 or list(other_accesses)[0] != other_write):
                                if any([a_state in reach for a_state, _ in other_accesses]):
                                    other_accesses.update(accesses)
                                    other_accesses.add(write)
                                    to_remove.add(write)
                                    result[desc][write] = set()
                for write in to_remove:
                    del result[desc][write]
            top_result[sdfg.sdfg_id] = result
        return top_result


@properties.make_properties
class ArrayWriteShadowScopes(ppl.Pass):
    """
    For each scalar or array of size 1, create a dictionary mapping writes to that data container to the set of reads
    and writes that are dominated by that write.
    """

    CATEGORY: str = 'Analysis'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Nothing

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        # If anything was modified, reapply
        return modified & ppl.Modifies.States

    def depends_on(self):
        return {AccessSets, FindAccessNodes, StateReachability, UnderapproximateWrites}

    # find dominating write given a descriptor and a read access node in a state
    def _find_dominating_write(
        self, desc: str, state: SDFGState, read: Union[nd.AccessNode, InterstateEdge],
        access_nodes: Dict[SDFGState, Tuple[Set[nd.AccessNode], Set[nd.AccessNode]]],
        state_idom: Dict[SDFGState, SDFGState], access_sets: Dict[SDFGState, Tuple[Set[str], Set[str]]],
        underapproximated_writes: Dict[str, Any], no_self_shadowing: bool = False
    ) -> Optional[list[Tuple[SDFGState, set[nd.AccessNode]]]]:
        sdfg = state.parent
        array_subset = subsets.Range.from_array(sdfg.arrays[desc])
        loop_writes: dict[SDFGState, dict[str, Memlet]] = underapproximated_writes["loop_approximation"]
        edge_approximation = underapproximated_writes["approximation"]
        loops = underapproximated_writes["loops"]

        if isinstance(read, nd.AccessNode):
            # If the read is also a write, it shadows itself.
            # TODO: need to check here if the write is a full write
            iedges = state.in_edges(read)
            if (len(iedges) > 0 and any(not e.data.is_empty() for e in iedges) and not no_self_shadowing and 
                any(edge_approximation[e].data.subset.covers(array_subset) for e in iedges)):
                return [(state, set(read))]

            # Find a dominating write within the same state.
            # TODO: Can this be done more efficiently?
            closest_candidate = None
            write_nodes = access_nodes[desc][state][1]
            # iterate over all the candidate writes to find closest write to read
            for cand in write_nodes:
                if cand != read and nxsp.has_path(state._nx, cand, read):
                    if (closest_candidate is None or nxsp.has_path(state._nx, closest_candidate, cand) and 
                        any(edge_approximation[e].data.subset.covers(array_subset) for e in state.in_edges(cand))):
                        closest_candidate = cand
            if closest_candidate is not None:
                return [(state, set(closest_candidate))]
        elif isinstance(read, InterstateEdge):
            # Attempt to find a shadowing write in the current state.
            # TODO: Can this be done more efficiently?
            # TODO: check if the candidate is a full write
            closest_candidate = None
            write_nodes = access_nodes[desc][state][1]
            for cand in write_nodes:
                if (closest_candidate is None or nxsp.has_path(state._nx, closest_candidate, cand) and 
                    any(edge_approximation[e].data.subset.covers(array_subset) for e in iedges)):
                    closest_candidate = cand    
            if closest_candidate is not None:
                return [(state, set(closest_candidate))]

        # Find the dominating write state if the current state is not the dominating write state.
        write_state = None
        nstate = state_idom[state] if state_idom[state] != state else None
        while nstate is not None:
            # check if the current state is a loopheader
            if nstate in loop_writes.keys():
                # check if the loop overwrites the array
                loop_memlet = loop_writes[nstate][desc]
                if not loop_memlet.subset.covers(array_subset):
                    continue

                _,_,loop_states,_,_ = loops[nstate]
                # the state is in the body of the loop of nstate so it is not dominated by the loop
                if state in loop_states:
                    continue

                # TODO: make a better check for this that verifies if every read in the loop is dominated by a write

                # check if there is any read from the array in the loop
                if any(desc in access_sets[s][0] for s in loop_states):
                    continue
                
                # whole loop overwrites the array so we declare the whole loop as a dominator
                dominators = [(s, access_nodes[desc][s][1]) for s in loop_states]

                return dominators



            
            if desc in access_sets[nstate][1]:
                write_state = nstate
            # Find a dominating write in the write state, i.e., the 'last' write to the data container.
            if write_state is not None:
                closest_candidate = None
                for cand in access_nodes[desc][write_state][1]:
                    # check if array is fully overwritten by the access node
                    in_edges = write_state.in_edges(cand)
                    # TODO: Maybe change the order of the conditionals here to check the "easy to check" conditions first
                    if any(edge_approximation[e].data.subset.covers(array_subset) for e in in_edges):
                        # if the access node has no outgoing edges we found the closest write
                        if write_state.out_degree(cand) == 0:
                            closest_candidate = cand
                            break
                        elif closest_candidate is None or nxsp.has_path(write_state._nx, closest_candidate, cand):
                            closest_candidate = cand
                if closest_candidate is not None:
                    return [(write_state, set(closest_candidate))]

            nstate = state_idom[nstate] if state_idom[nstate] != nstate else None

        return None

    def apply_pass(self, top_sdfg: SDFG, pipeline_results: Dict[str, Any]) -> Dict[int, ArrayWriteScopeDict]:
        """
        :return: A dictionary mapping each data descriptor name to a dictionary, where writes to that data descriptor
                 and the states they are contained in are mapped to the set of reads and writes (and their states) that
                 are dominated by that write.
        """
        top_result: Dict[int, ArrayWriteScopeDict] = dict()

        for sdfg in top_sdfg.all_sdfgs_recursive():
            result: ArrayWriteScopeDict = defaultdict(
                lambda: defaultdict(lambda: set()))
            idom = nx.immediate_dominators(sdfg.nx, sdfg.start_state)
            all_doms = cfg.all_dominators(sdfg, idom)
            access_sets: Dict[SDFGState, Tuple[Set[str],
                                               Set[str]]] = pipeline_results[AccessSets.__name__][sdfg.sdfg_id]
            # get mapping from data descriptor to mapping from states to all read/write access nodes with the given descriptor
            access_nodes: Dict[str, Dict[SDFGState, Tuple[Set[nd.AccessNode], Set[nd.AccessNode]]]] = pipeline_results[
                FindAccessNodes.__name__][sdfg.sdfg_id]
            # mapping from state to other states that can be executed after given state
            state_reach: Dict[SDFGState, Set[SDFGState]
                              ] = pipeline_results[StateReachability.__name__][sdfg.sdfg_id]
            # mapping from dataflow edges to dataflow edges with underapproximated subsets
            underapproximated_writes: Dict[str, Any] = pipeline_results[UnderapproximateWrites.__name__]

            anames = sdfg.arrays.keys()
            # iterate over all data descriptors
            for desc in sdfg.arrays:
                # all states that contain a read/write access node to desc
                desc_states_with_nodes = set(access_nodes[desc].keys())
                # iterate over all states that contain a read/write node to desc
                for state in desc_states_with_nodes:
                    # iterate over all the read nodes in the current state
                    for read_node in access_nodes[desc][state][0]:
                        # find the dominating write for the current read_node
                        write = self._find_dominating_write(
                            desc, state, read_node, access_nodes, idom, access_sets, underapproximated_writes)
                        result[desc][write].add((state, read_node))
                # Ensure accesses to interstate edges are also considered.
                # iterate over all states and the access sets they are mapped to
                for state, accesses in access_sets.items():
                    # check if the current descriptor is in the read set of the current state
                    if desc in accesses[0]:
                        # get the outgoing CFG-edges of the current state
                        out_edges = sdfg.out_edges(state)
                        # iterate over the outgoing insterstate edges
                        for oedge in out_edges:
                            # intersect the names of the arrays with the name of the free symbols in the edge
                            syms = oedge.data.free_symbols & anames
                            # if the intersection contains the name of the descriptor find the dominating write
                            if desc in syms:
                                write = self._find_dominating_write(
                                    desc, state, oedge.data, access_nodes, idom, access_sets, underapproximated_writes
                                )
                                result[desc][write].add((state, oedge.data))
                # Take care of any write nodes that have not been assigned to a scope yet, i.e., writes that are not
                # dominating any reads and are thus not part of the results yet.
                for state in desc_states_with_nodes:
                    for write_node in access_nodes[desc][state][1]:
                        if not (state, write_node) in result[desc]:
                            write = self._find_dominating_write(
                                desc, state, write_node, access_nodes, idom, access_sets, underapproximated_writes, no_self_shadowing=True
                            )
                            result[desc][write].add((state, write_node))
                
                # Merge two scopes if the dominators overlap

                #repeatedly find two scopes that can be merged and stop if the list doesn't change
                # TODO: find something faster. Complexity of the merging is a nightmare
                to_remove = set()
                while(True):
                    for writes in result[desc].keys():
                        for other_writes in result[desc].keys():
                            for state,_ in other_writes:
                                if any(state is s for s,_ in writes):
                                    to_remove.add(writes)
                                    to_remove.add(other_writes)
                                    scope1: Set[Tuple[SDFGState, Union[nd.AccessNode, InterstateEdge]]] = result[desc][writes]
                                    scope2: Set[Tuple[SDFGState, Union[nd.AccessNode, InterstateEdge]]] = result[desc][other_writes]
                                    # merge the scopes
                                    new_scope = scope1.union(scope2)
                                    new_writes: Optional[set[Tuple[SDFGState, set[nd.AccessNode]]]] = set()
                                    # merge the accessnodes that are in the same state
                                    for state1, nodes1 in writes:
                                        for state2, nodes2 in other_writes:
                                            if state1 is state2:
                                                new_writes.add((state1, nodes1.union(nodes2)))
                                    rest1 = [(states, nodes) for states, nodes in writes if state in [t[0] for t in writes]]
                                    rest2 = [(states, nodes) for states, nodes in writes if state in [t[0] for t in other_writes]]
                                    new_writes = new_writes.union(rest1.union(rest2))
                                    break
                            else:
                                continue  # only executed if the inner loop did NOT break
                            break  # only executed if the inner loop DID break
                        else:
                            continue  # only executed if the inner loop did NOT break
                        break  # only executed if the inner loop DID break
                    else:
                        # if the inner loop didnt merge anything it didnt break so we break out of the while loop
                        break
                    for r in to_remove:
                        del result[desc][r]
                    to_remove = set()
                    result[desc][new_writes] = new_scope

                # If any write A is dominated by another write B and any reads in B's scope are also reachable by A,
                # then merge A and its scope into B's scope.
                to_remove = set()
                for write, accesses in result[desc].items():
                    if write is None:
                        continue
                    write_state, write_node = write
                    dominators = all_doms[write_state]
                    reach = state_reach[write_state]
                    for other_write, other_accesses in result[desc].items():
                        if other_write is not None and other_write[1] is write_node and other_write[0] is write_state:
                            continue
                        if other_write is None or other_write[0] in dominators:
                            noa = len(other_accesses)
                            if noa > 0 and (noa > 1 or list(other_accesses)[0] != other_write):
                                if any([a_state in reach for a_state, _ in other_accesses]):
                                    other_accesses.update(accesses)
                                    other_accesses.add(write)
                                    to_remove.add(write)
                                    result[desc][write] = set()
                for write in to_remove:
                    del result[desc][write]
            top_result[sdfg.sdfg_id] = result
        return top_result
