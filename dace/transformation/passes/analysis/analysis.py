# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.

from collections import defaultdict
from dace.sdfg.state import ControlFlowBlock, ControlFlowRegion, LoopRegion
from dace.transformation import pass_pipeline as ppl, transformation
from dace import SDFG, SDFGState, properties, InterstateEdge, Memlet, data as dt, symbolic
from dace.sdfg.graph import Edge
from dace.sdfg import nodes as nd
from dace.sdfg.analysis import cfg
from typing import Dict, Set, Tuple, Any, Optional, Union
import networkx as nx
from networkx.algorithms import shortest_paths as nxsp

WriteScopeDict = Dict[str, Dict[Optional[Tuple[SDFGState, nd.AccessNode]],
                                Set[Tuple[SDFGState, Union[nd.AccessNode, InterstateEdge]]]]]
SymbolScopeDict = Dict[str, Dict[Edge[InterstateEdge], Set[Union[Edge[InterstateEdge], SDFGState]]]]


@properties.make_properties
@transformation.experimental_cfg_block_compatible
class StateReachability(ppl.Pass):
    """
    Evaluates state reachability (which other states can be executed after each state).
    """

    CATEGORY: str = 'Analysis'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Nothing

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        # If anything was modified, reapply
        return modified & ppl.Modifies.CFG

    def depends_on(self):
        return {ControlFlowBlockReachability}

    def _region_closure(self, region: ControlFlowRegion,
                        block_reach: Dict[int, Dict[ControlFlowBlock, Set[ControlFlowBlock]]]) -> Set[SDFGState]:
        closure: Set[SDFGState] = set()
        if isinstance(region, LoopRegion):
            # Any point inside the loop may reach any other point inside the loop again.
            # TODO(later): This is an overapproximation. A branch terminating in a break is excluded from this.
            closure.update(region.all_states())

        # Add all states that this region can reach in its parent graph to the closure.
        for reached_block in block_reach[region.parent_graph.cfg_id][region]:
            if isinstance(reached_block, ControlFlowRegion):
                closure.update(reached_block.all_states())
            elif isinstance(reached_block, SDFGState):
                closure.add(reached_block)
            
        # Walk up the parent tree.
        pivot = region.parent_graph
        while pivot and not isinstance(pivot, SDFG):
            closure.update(self._region_closure(pivot, block_reach))
            pivot = pivot.parent_graph
        return closure

    def apply_pass(self, top_sdfg: SDFG, pipeline_res: Dict) -> Dict[int, Dict[SDFGState, Set[SDFGState]]]:
        """
        :return: A dictionary mapping each state to its other reachable states.
        """
        # Ensure control flow block reachability is run if not run within a pipeline.
        if not ControlFlowBlockReachability.__name__ in pipeline_res:
            cf_block_reach_dict = ControlFlowBlockReachability().apply_pass(top_sdfg, {})
        else:
            cf_block_reach_dict = pipeline_res[ControlFlowBlockReachability.__name__]
        reachable: Dict[int, Dict[SDFGState, Set[SDFGState]]] = {}
        for sdfg in top_sdfg.all_sdfgs_recursive():
            result: Dict[SDFGState, Set[SDFGState]] = defaultdict(set)
            for state in sdfg.states():
                for reached in cf_block_reach_dict[state.parent_graph.cfg_id][state]:
                    if isinstance(reached, ControlFlowRegion):
                        result[state].update(reached.all_states())
                    elif isinstance(reached, SDFGState):
                        result[state].add(reached)
                if state.parent_graph is not sdfg:
                    result[state].update(self._region_closure(state.parent_graph, cf_block_reach_dict))
            reachable[sdfg.cfg_id] = result
        return reachable


@properties.make_properties
@transformation.experimental_cfg_block_compatible
class ControlFlowBlockReachability(ppl.Pass):
    """
    Evaluates control flow block reachability (which control flow block can be executed after each control flow block)
    """

    CATEGORY: str = 'Analysis'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Nothing

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return modified & ppl.Modifies.CFG

    def apply_pass(self, top_sdfg: SDFG, _) -> Dict[int, Dict[ControlFlowBlock, Set[ControlFlowBlock]]]:
        """
        :return: For each control flow region, a dictionary mapping each control flow block to its other reachable
                 control flow blocks in the same region.
        """
        reachable: Dict[int, Dict[ControlFlowBlock, Set[ControlFlowBlock]]] = defaultdict(lambda: defaultdict(set))
        for cfg in top_sdfg.all_control_flow_regions(recursive=True):
            # In networkx this is currently implemented naively for directed graphs.
            # The implementation below is faster
            # tc: nx.DiGraph = nx.transitive_closure(sdfg.nx)
            for n, v in reachable_nodes(cfg.nx):
                reachable[cfg.cfg_id][n] = set(v)
        return reachable


def _single_shortest_path_length_no_self(adj, source):
    """Yields (node, level) in a breadth first search, without the first level
    unless a self-edge exists.

    Adapted from Shortest Path Length helper function in NetworkX.

    Parameters
    ----------
        adj : dict
            Adjacency dict or view
        firstlevel : dict
            starting nodes, e.g. {source: 1} or {target: 1}
        cutoff : int or float
            level at which we stop the process
    """
    firstlevel = {source: 1}

    seen = {}  # level (number of hops) when seen in BFS
    level = 0  # the current level
    nextlevel = set(firstlevel)  # set of nodes to check at next level
    n = len(adj)
    while nextlevel:
        thislevel = nextlevel  # advance to next level
        nextlevel = set()  # and start a new set (fringe)
        found = []
        for v in thislevel:
            if v not in seen:
                if level == 0 and v is source:  # Skip 0-length path to self
                    found.append(v)
                    continue
                seen[v] = level  # set the level of vertex v
                found.append(v)
                yield (v, level)
        if len(seen) == n:
            return
        for v in found:
            nextlevel.update(adj[v])
        level += 1
    del seen


def reachable_nodes(G):
    """Computes the reachable nodes in G."""
    adj = G.adj
    for n in G:
        yield (n, dict(_single_shortest_path_length_no_self(adj, n)))


@properties.make_properties
@transformation.experimental_cfg_block_compatible
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

    def apply_pass(self, top_sdfg: SDFG,
                   _) -> Dict[int, Dict[Union[SDFGState, Edge[InterstateEdge]], Tuple[Set[str], Set[str]]]]:
        """
        :return: A dictionary mapping each state and interstate edge to a tuple of its (read, written) symbols.
        """
        top_result: Dict[int, Dict[Union[SDFGState, Edge[InterstateEdge]], Tuple[Set[str], Set[str]]]] = {}
        for sdfg in top_sdfg.all_sdfgs_recursive():
            for cfg in sdfg.all_control_flow_regions():
                adesc = set(sdfg.arrays.keys())
                result: Dict[SDFGState, Tuple[Set[str], Set[str]]] = {}
                for block in cfg.nodes():
                    if isinstance(block, SDFGState):
                        # No symbols may be written to inside states.
                        result[block] = (block.free_symbols, set())
                    for oedge in cfg.out_edges(block):
                        edge_readset = oedge.data.read_symbols() - adesc
                        edge_writeset = set(oedge.data.assignments.keys())
                        result[oedge] = (edge_readset, edge_writeset)
                top_result[cfg.cfg_id] = result
        return top_result


@properties.make_properties
@transformation.experimental_cfg_block_compatible
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
            for state in sdfg.states():
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

            top_result[sdfg.cfg_id] = result
        return top_result


@properties.make_properties
@transformation.experimental_cfg_block_compatible
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
            for state in sdfg.states():
                for anode in state.data_nodes():
                    result[anode.data].add(state)

            # Edges that read from arrays add to both ends' access sets
            anames = sdfg.arrays.keys()
            for e in sdfg.all_interstate_edges():
                fsyms = e.data.free_symbols & anames
                for access in fsyms:
                    result[access].update({e.src, e.dst})

            top_result[sdfg.cfg_id] = result
        return top_result


@properties.make_properties
@transformation.experimental_cfg_block_compatible
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
            for state in sdfg.states():
                for anode in state.data_nodes():
                    if state.in_degree(anode) > 0:
                        result[anode.data][state][1].add(anode)
                    if state.out_degree(anode) > 0:
                        result[anode.data][state][0].add(anode)
            top_result[sdfg.cfg_id] = result
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

    def _find_dominating_write(self, sym: str, read: Union[SDFGState, Edge[InterstateEdge]],
                               state_idom: Dict[SDFGState, SDFGState]) -> Optional[Edge[InterstateEdge]]:
        last_state: SDFGState = read if isinstance(read, SDFGState) else read.src

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
            result: SymbolScopeDict = defaultdict(lambda: defaultdict(lambda: set()))

            idom = nx.immediate_dominators(sdfg.nx, sdfg.start_state)
            all_doms = cfg.all_dominators(sdfg, idom)
            symbol_access_sets: Dict[Union[SDFGState, Edge[InterstateEdge]],
                                     Tuple[Set[str],
                                           Set[str]]] = pipeline_results[SymbolAccessSets.__name__][sdfg.cfg_id]
            state_reach: Dict[SDFGState, Set[SDFGState]] = pipeline_results[StateReachability.__name__][sdfg.cfg_id]

            for read_loc, (reads, _) in symbol_access_sets.items():
                for sym in reads:
                    dominating_write = self._find_dominating_write(sym, read_loc, idom)
                    result[sym][dominating_write].add(read_loc if isinstance(read_loc, SDFGState) else read_loc)

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

            top_result[sdfg.cfg_id] = result
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

    def _find_dominating_write(self,
                               desc: str,
                               state: SDFGState,
                               read: Union[nd.AccessNode, InterstateEdge],
                               access_nodes: Dict[SDFGState, Tuple[Set[nd.AccessNode], Set[nd.AccessNode]]],
                               state_idom: Dict[SDFGState, SDFGState],
                               access_sets: Dict[SDFGState, Tuple[Set[str], Set[str]]],
                               no_self_shadowing: bool = False) -> Optional[Tuple[SDFGState, nd.AccessNode]]:
        if isinstance(read, nd.AccessNode):
            # If the read is also a write, it shadows itself.
            iedges = state.in_edges(read)
            if len(iedges) > 0 and any(not e.data.is_empty() for e in iedges) and not no_self_shadowing:
                return (state, read)

            # Find a dominating write within the same state.
            # TODO: Can this be done more efficiently?
            closest_candidate = None
            write_nodes = access_nodes[desc][state][1]
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
            result: WriteScopeDict = defaultdict(lambda: defaultdict(lambda: set()))
            idom = nx.immediate_dominators(sdfg.nx, sdfg.start_state)
            all_doms = cfg.all_dominators(sdfg, idom)
            access_sets: Dict[SDFGState, Tuple[Set[str],
                                               Set[str]]] = pipeline_results[AccessSets.__name__][sdfg.cfg_id]
            access_nodes: Dict[str, Dict[SDFGState, Tuple[Set[nd.AccessNode], Set[nd.AccessNode]]]] = pipeline_results[
                FindAccessNodes.__name__][sdfg.cfg_id]
            state_reach: Dict[SDFGState, Set[SDFGState]] = pipeline_results[StateReachability.__name__][sdfg.cfg_id]

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
                                write = self._find_dominating_write(desc, state, oedge.data, access_nodes, idom,
                                                                    access_sets)
                                result[desc][write].add((state, oedge.data))
                # Take care of any write nodes that have not been assigned to a scope yet, i.e., writes that are not
                # dominating any reads and are thus not part of the results yet.
                for state in desc_states_with_nodes:
                    for write_node in access_nodes[desc][state][1]:
                        if not (state, write_node) in result[desc]:
                            write = self._find_dominating_write(desc,
                                                                state,
                                                                write_node,
                                                                access_nodes,
                                                                idom,
                                                                access_sets,
                                                                no_self_shadowing=True)
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
            top_result[sdfg.cfg_id] = result
        return top_result


@properties.make_properties
@transformation.experimental_cfg_block_compatible
class AccessRanges(ppl.Pass):
    """
    For each data descriptor, finds all memlets used to access it (read/write ranges).
    """

    CATEGORY: str = 'Analysis'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Nothing

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return modified & ppl.Modifies.Memlets

    def apply_pass(self, top_sdfg: SDFG, _) -> Dict[int, Dict[str, Set[Memlet]]]:
        """
        :return: A dictionary mapping each data descriptor name to a set of memlets.
        """
        top_result: Dict[int, Dict[str, Set[Memlet]]] = dict()

        for sdfg in top_sdfg.all_sdfgs_recursive():
            result: Dict[str, Set[Memlet]] = defaultdict(set)
            for state in sdfg.states():
                for anode in state.data_nodes():
                    for e in state.all_edges(anode):
                        if e.dst is anode and e.dst_conn == 'set':  # Skip reference sets
                            continue
                        if e.data.is_empty():  # Skip empty memlets
                            continue
                        # Find (hopefully propagated) root memlet
                        e = state.memlet_tree(e).root().edge
                        result[anode.data].add(e.data)
            top_result[sdfg.cfg_id] = result
        return top_result


@properties.make_properties
@transformation.experimental_cfg_block_compatible
class FindReferenceSources(ppl.Pass):
    """
    For each Reference data descriptor, finds all memlets used to set it. If a Tasklet was used
    to set the reference, the Tasklet is given as a source.
    """

    CATEGORY: str = 'Analysis'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Nothing

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return modified & ppl.Modifies.Memlets

    def apply_pass(self, top_sdfg: SDFG, _) -> Dict[int, Dict[str, Set[Union[Memlet, nd.CodeNode]]]]:
        """
        :return: A dictionary mapping each data descriptor name to a set of memlets.
        """
        top_result: Dict[int, Dict[str, Set[Union[Memlet, nd.CodeNode]]]] = dict()

        for sdfg in top_sdfg.all_sdfgs_recursive():
            result: Dict[str, Set[Memlet]] = defaultdict(set)
            reference_descs = set(k for k, v in sdfg.arrays.items() if isinstance(v, dt.Reference))
            for state in sdfg.states():
                for anode in state.data_nodes():
                    if anode.data not in reference_descs:
                        continue
                    for e in state.in_edges(anode):
                        if e.dst_conn != 'set':
                            continue
                        true_src = state.memlet_path(e)[0].src
                        if isinstance(true_src, nd.CodeNode):
                            # Code  -> Reference
                            result[anode.data].add(true_src)
                        else:
                            # Array -> Reference
                            result[anode.data].add(e.data)
            top_result[sdfg.cfg_id] = result
        return top_result


@properties.make_properties
@transformation.experimental_cfg_block_compatible
class DeriveSDFGConstraints(ppl.Pass):

    CATEGORY: str = 'Analysis'

    assume_max_data_size = properties.Property(dtype=int, default=None, allow_none=True,
                                               desc='Assume that all data containers have no dimension larger than ' +
                                               'this value. If None, no assumption is made.')

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Nothing

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        # If anything was modified, reapply
        return modified & ppl.Modifies.Everything

    def _derive_parameter_datasize_constraints(self, sdfg: SDFG, invariants: Dict[str, Set[str]]) -> None:
        handled = set()
        for arr in sdfg.arrays.values():
            for dim in arr.shape:
                if isinstance(dim, symbolic.symbol) and not dim in handled:
                    ds = str(dim)
                    if ds not in invariants:
                        invariants[ds] = set()
                    invariants[ds].add(f'{ds} > 0')
                    if self.assume_max_data_size is not None:
                        invariants[ds].add(f'{ds} <= {self.assume_max_data_size}')
                    handled.add(ds)

    def apply_pass(self, sdfg: SDFG, _) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]], Dict[str, Set[str]]]:
        invariants: Dict[str, Set[str]] = {}
        self._derive_parameter_datasize_constraints(sdfg, invariants)
        return {}, invariants, {}
