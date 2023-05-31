# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
from collections import defaultdict
from typing import Any, Dict, Optional, Set, Tuple

from dace import SDFG, InterstateEdge
from dace.memlet import Memlet
from dace.sdfg import nodes as nd
from dace.transformation import pass_pipeline as ppl
from dace.transformation.passes import analysis as ap
from dace.sdfg.propagation_underapproximation import UnderapproximateWrites
import networkx as nx
from dace.sdfg import SDFGState
import dace.subsets as subsets


class ArrayFission(ppl.Pass):
    """
    Fission transient arrays that are dominated by writes to the whole array into separate data containers.
    """

    CATEGORY: str = 'Optimization Preparation'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Descriptors | ppl.Modifies.AccessNodes

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return modified & ppl.Modifies.AccessNodes

    def depends_on(self):
        return {ap.AccessSets, ap.SymbolAccessSets, ap.FindAccessNodes, ap.StateReachability, UnderapproximateWrites}

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> Optional[Dict[str, Set[str]]]:
        """
        Rename scalars and arrays of size 1 based on dominated scopes.

        :param sdfg: The SDFG to modify.
        :param pipeline_results: If in the context of a ``Pipeline``, a dictionary that is populated with prior Pass
                                 results as ``{Pass subclass name: returned object from pass}``. If not run in a
                                 pipeline, an empty dictionary is expected.
        :return: A dictionary mapping the original name to a set of all new names created for each data container.
        """
        results: Dict[str, Set[str]] = defaultdict(lambda: set())

        immediate_dominators = nx.dominance.immediate_dominators(
            sdfg.nx, sdfg.start_state)
        dominance_frontiers = nx.dominance.dominance_frontiers(
            sdfg.nx, sdfg.start_state)
        write_approximation = pipeline_results[UnderapproximateWrites.__name__]["approximation"]
        loop_write_approximation: dict[SDFGState, dict[str, Memlet]
                                       ] = pipeline_results[UnderapproximateWrites.__name__]["loop_approximation"]
        loops: dict[SDFGState, (SDFGState, SDFGState, list[SDFGState], str, subsets.Range)
                    ] = pipeline_results[UnderapproximateWrites.__name__]["loops"]

        access_nodes: Dict[str, Dict[SDFGState, Tuple[Set[nd.AccessNode], Set[nd.AccessNode]]]
                           ] = pipeline_results[ap.FindAccessNodes.__name__][sdfg.sdfg_id]
        state_reach: Dict[SDFGState, Set[SDFGState]
                          ] = pipeline_results[ap.StateReachability.__name__][sdfg.sdfg_id]
        access_sets: Dict[SDFGState, Tuple[Set[str], Set[str]]
                          ] = pipeline_results[ap.AccessSets.__name__][sdfg.sdfg_id]

        # array that stores "virtual" phi nodes for each variable for each SDFGstate
        # phi nodes are represented by dictionaries that contain the definitions reaching the phi node
        # can be extended for later use with path constraints for example
        phi_nodes: Dict[SDFGState, Dict[str, Dict]] = defaultdict(None)
        for state in sdfg.states():
            phi_nodes[state] = {}

        # insert phi nodes

        # list of original array names in the sdfg
        anames = sdfg.arrays.copy().keys()
        # populate the phi node dictionary
        for var in anames:
            desc = sdfg.arrays[var]
            array_set = subsets.Range.from_array(desc)
            phi_states = set()
            # array of states that define/fully overwrite the array
            defining_states = set()

            for state in sdfg.states():
                # iterate over access nodes to the array in the current state and check if it fully overwrites the
                # array with the write underapproximation

                # check if there's even a write to the descriptor in the current state
                write_nodes = access_nodes[var][state][1]
                if len(write_nodes) == 0:
                    continue

                for node in write_nodes:
                    # if any of the edges fully overwrites the array add the state to the defining states
                    if any(write_approximation[e].data.subset.covers(array_set) for e in state.in_edges(node)):
                        defining_states.add(state)
                        break

            while (defining_states):
                current_state = next(iter(defining_states))
                dominance_frontier = dominance_frontiers[current_state]

                for frontier_state in dominance_frontier:
                    # check if this state was already handled
                    if frontier_state in phi_states:
                        continue
                    phi_nodes[frontier_state][var] = {
                        "name": var,
                        "variables": [],
                        "descriptor": None
                    }
                    phi_states.add(frontier_state)
                    if frontier_state not in defining_states:
                        defining_states.add(frontier_state)

                defining_states.remove(current_state)

        for loopheader, write_dict in loop_write_approximation.items():
            for var, memlet in write_dict.items():
                if var in phi_nodes[loopheader].keys():
                    continue
                if memlet.subset.covers(subsets.Range.from_array(sdfg.arrays[var])):
                    phi_nodes[loopheader][var] = {
                        "name": var,
                        "variables": [],
                        "descriptor": None
                    }

        # store all the new names for each original variable name
        definitions: Dict[str, set(str)] = defaultdict(None)

        # dictionary mapping each variable from the original SDFG to a dictionary mapping
        # each state to the last definition of that variable in that state
        last_defs: Dict[str, Dict[SDFGState, str]] = defaultdict(None)

        # aux function to find the reaching definition given an AccessNode and a state in the original SDFG
        # only works with the guarantee that predecessors of state have already been traversed/renamed
        def find_reaching_def(state: SDFGState, var: str, node: nd.AccessNode = None):
            # look for the reaching definition within the state
            if node:
                # traverse dataflow graph breadth first in reverse direction from node
                g_reversed = state.nx.reverse()
                bfs_edges = nx.bfs_edges(g_reversed, node)
                if var in definitions.keys():
                    names = definitions[var]
                    for edge in bfs_edges:
                        # check if the destination node is a renamed instance of the original variable
                        if not isinstance(edge[1], nd.AccessNode):
                            continue
                        if edge[1].data in names:
                            return edge[1].data

            # if there is a phi node for the variable in the same state, return the variable name defined by the phi node
            if phi_nodes[state].get(var):
                return phi_nodes[state][var]["name"]
            # otherwise return the last definition of the immediate dominator
            idom = immediate_dominators[state]
            # handle the case where state is the first state
            if last_defs.get(var):
                if last_defs[var].get(idom):
                    return last_defs[var][idom]

            # in case the state is the initial state and there isn't any reaching definition in the current state just return the original variable
            return var

        def rename_node(state: SDFGState, node: nd.AccessNode, new_name: str):
            old_name = node.data
            node.data = newname
            for iedge in state.in_edges(node):
                if iedge.data.data == old_name:
                    iedge.data.data = newname
            for oedge in state.out_edges(node):
                if oedge.data.data == old_name:
                    oedge.data.data = newname

        def add_definition(new_def: str, original_name: str):
            if not definitions.get(original_name):
                definitions[original_name] = set()
            definitions[original_name].add(new_def)

        def update_last_def(state: SDFGState, new_def: str, original_name: str):
            if not last_defs.get(original_name):
                last_defs[original_name] = {}
            last_defs[original_name][state] = new_def

        # create dictionary mapping each node to the nodes it immediatly dominates basically representing the dominator tree
        immediate_dominated = defaultdict(None)
        for node, dom_node in immediate_dominators.items():
            if dom_node not in immediate_dominated:
                immediate_dominated[dom_node] = set()
            immediate_dominated[dom_node].add(node)

        # traverse the dominator tree depth first and rename all variables
        # TODO: Handle occurences of the variables in insterstate edges
        stack = []
        visited = set()
        stack.append(sdfg.start_state)
        while stack:
            current_state: SDFGState = stack.pop()
            if current_state in visited:
                continue
            visited.add(current_state)

            if immediate_dominated.get(current_state):
                children = immediate_dominated[current_state]
                for child in children:
                    if child not in visited:
                        stack.append(child)

            # rename the phi nodes
            for var in anames:
                # check if there is a phi node for the current variable in the current state
                if not phi_nodes[current_state].get(var):
                    continue

                newdesc = sdfg.arrays[var].clone()
                newname = sdfg.add_datadesc(var, newdesc, find_new_name=True)
                phi_nodes[current_state][var]["descriptor"] = newdesc
                phi_nodes[current_state][var]["name"] = newname

                add_definition(newname, var)
                update_last_def(current_state, newname, var)

            # rename data nodes
            # get topological ordering of nodes in the dataflowgraph
            toposort = list(nx.topological_sort(current_state.nx))
            for node in toposort:
                if isinstance(node, nd.AccessNode):
                    var = node.data
                    array_set = subsets.Range.from_array(sdfg.arrays[var])
                    # if array is not fully overwritten at this access node treat it as a use otherwise as a def
                    # check if there is a parallel write or read to/from the same array in the same state
                    # if so we cannot introduce a new variable
                    iedges = current_state.in_edges(node)
                    if (any(write_approximation[e].data.subset.covers(array_set) for e in iedges) and
                        not any(
                            not (nx.has_path(current_state.nx, node, other_node) or
                                 nx.has_path(current_state.nx, other_node, node)) for other_node in access_nodes[var][current_state][1])):

                        # rename the variable to the reaching definition
                        newdesc = desc.clone()
                        newname = sdfg.add_datadesc(
                            var, newdesc, find_new_name=True)

                        add_definition(newname, var)
                        update_last_def(current_state, newname, var)

                    else:
                        newname = find_reaching_def(current_state, var, node)

                    rename_node(current_state, node, newname)

            # if last definition in this state has not been defined yet, define it here
            for var in anames:
                if not last_defs.get(var):
                    last_defs[var] = {}
                if not last_defs[var].get(current_state):
                    last_defs[var][current_state] = find_reaching_def(
                        current_state, var)

            # rename occurences of variables in successors of current state
            successors = [e.dst for e in sdfg.out_edges(current_state)]
            for successor in successors:
                for var in anames:
                    if phi_nodes[successor].get(var):
                        newname = last_defs[var][current_state]
                        phi_nodes[successor][var]["variables"].append(newname)

            # iterate over all the outgoing interstate edges of the current state and rename all the occurences of the original
            # variable to the last definition in the current state
            rename_dict = {}
            for var in last_defs.keys():
                rename_dict[var] = last_defs[var][current_state]
            for oedge in sdfg.out_edges(current_state):
                oedge.data.replace_dict(rename_dict)

        # iterate over the phi nodes and replace all the occurences of each parameter with the variable defined by the phi node
        for state, phi_dict in phi_nodes.items():
            for original_var, phi_node in phi_dict.items():
                newname = phi_node["name"]
                parameters = phi_node["variables"]

                candidate_states = set()
                # check if the phi node belongs to a loopheader that completely overwrites the array and the loop does not read from the array
                # if so, only rename nodes in the loop body and nodes dominated by the loopheader
                if (state in loop_write_approximation.keys() and
                    original_var in loop_write_approximation[state].keys() and
                    loop_write_approximation[state][original_var].subset.covers(subsets.Range.from_array(sdfg.arrays[original_var]))):
                    
                    _, _, loop_states, _, _ = loops[state]

                    if not any(original_var in access_sets[s][0] for s in loop_states):
                        candidate_states = state_reach[state].union(
                            set(loop_states))

                # iterate over all the states that read from the variable and rename the occurences
                for other_state, (reads, writes) in access_nodes[original_var].items():
                    if candidate_states and other_state not in candidate_states:
                        continue
                    ans = reads.union(writes)
                    for an in ans:
                        if an.data in parameters:
                            rename_node(other_state, an, newname)

                # rename all the occurences in other phi nodes
                # TODO: restrict the set of states here
                for other_state, other_phi_dict in phi_nodes.items():
                    if candidate_states and other_state not in candidate_states:
                        continue
                    if original_var in other_phi_dict.keys():
                        other_phi_node = other_phi_dict[original_var]
                        if other_phi_node["name"] in parameters:
                            other_phi_node["name"] = newname
                            phi_nodes[other_state][original_var] = other_phi_node

                rename_dict = {}
                for parameter in parameters:
                    rename_dict[parameter] = newname
                for other_state, other_accesses in access_sets.items():
                    if candidate_states and other_state not in candidate_states:
                        continue
                    if original_var in other_accesses[0]:
                        out_edges = sdfg.out_edges(other_state)
                        for oedge in out_edges:
                            oedge.data.replace_dict(rename_dict)

                definitions[original_var] -= definitions[original_var] - \
                    set(parameters)

        import pprint
        print("phi_nodes")
        pprint.pprint(phi_nodes)
        pprint.pprint(loop_write_approximation)

        results = definitions
        return results

    def report(self, pass_retval: Any) -> Optional[str]:
        return f'Renamed {len(pass_retval)} scalars: {pass_retval}.'
