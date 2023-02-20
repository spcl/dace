# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.

from collections import defaultdict
from dace.transformation import pass_pipeline as ppl
from dace import SDFG, SDFGState, properties
from typing import Dict, Set, Tuple, Any, Optional
import networkx as nx
from networkx.algorithms import shortest_paths as nxsp


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
class FindAccessNodes(ppl.Pass):
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
class ScalarWriteShaddowScopes(ppl.Pass):
    """
    For each scalar or array of size 1, create a dictionary mapping each state containing a write to that data container
    to the set of states that are shaddowed by that write. This means all states containing reads from that data
    container that are dominated by a given write to that container.
    """

    CATEGORY: str = 'Analysis'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Nothing

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        # If anything was modified, reapply
        return modified & ppl.Modifies.States

    def depends_on(self):
        return {AccessSets}

    def apply_pass(self, top_sdfg: SDFG,
                   pipeline_results: Dict[str, Any]) -> Dict[int, Dict[str, Dict[SDFGState, Set[SDFGState]]]]:
        """
        :return: A dictionary mapping each data descriptor name to a dictionary, where states with writes to that data
                 descriptor are mapped to the set of states that are in the scope of that write.
        """
        top_result: Dict[int, Dict[str, Dict[SDFGState, Set[SDFGState]]]] = {}

        for sdfg in top_sdfg.all_sdfgs_recursive():
            result: Dict[str, Dict[SDFGState, Set[SDFGState]]] = dict()
            idom = nx.immediate_dominators(sdfg.nx, sdfg.start_state)
            access_sets: Dict[SDFGState, Tuple[Set[str], Set[str]]] = pipeline_results['AccessSets'][sdfg.sdfg_id]
            for state in sdfg.states():
                accesses = access_sets[state]
                reads = accesses[0]
                writes = accesses[1]
                for desc in reads:
                    # We only look at scalars or arrays with size 1 in this pass.
                    if desc not in sdfg.arrays or sdfg.arrays[desc].total_size != 1:
                        continue

                    if desc not in result:
                        result[desc] = dict()
                    write_state = None
                    if desc in writes:
                        # There may be a self-write, i.e. a write to the same data descriptor in the same state. In this
                        # case, we need to check if that write happens before the read.
                        # We check all reads from the given descriptor.
                        read_nodes = set()
                        write_nodes = set()
                        for dnode in state.data_nodes():
                            if dnode.data == desc:
                                iedges = state.in_edges(dnode)
                                oedges = state.in_edges(dnode)
                                for edge in iedges:
                                    if not edge.data.is_empty():
                                        write_nodes.add(dnode)
                                        break
                                for edge in oedges:
                                    if not edge.data.is_empty():
                                        read_nodes.add(dnode)
                                        break
                        all_shaddowed = True
                        for read_node in read_nodes:
                            shaddowed = False
                            if read_node in write_nodes:
                                shaddowed = True
                            else:
                                for candidate in write_nodes:
                                    if nxsp.has_path(sdfg.nx, candidate, read_node):
                                        shaddowed = True
                                        break
                            if not shaddowed:
                                all_shaddowed = False
                                break
                        if all_shaddowed:
                            write_state = state

                    # Find the dominating write state if the current state is not the shaddowing write state.
                    nstate = idom[state] if idom[state] != state else None
                    while nstate is not None and write_state is None:
                        if desc in access_sets[nstate][1]:
                            write_state = nstate
                        nstate = idom[nstate] if idom[nstate] != nstate else None

                    # Add the read to the scope of any found write state, or to the scope of 'None' if no write state
                    # was found.
                    if write_state not in result[desc]:
                        result[desc][write_state] = set()
                    result[desc][write_state].add(state)
            top_result[sdfg.sdfg_id] = result
        return top_result
