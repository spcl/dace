# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
from collections import defaultdict
import functools
from typing import Any, Dict, Optional, Set, Tuple, Union

from dace import SDFG, InterstateEdge
from dace.memlet import Memlet
from dace.sdfg import nodes as nd
from dace.sdfg.graph import Edge
from dace.transformation import pass_pipeline as ppl
from dace.transformation.passes import analysis as ap
from dace.sdfg.propagation_underapproximation import UnderapproximateWrites
import networkx as nx
from dace.sdfg import SDFGState
import dace.subsets as subsets
import sympy
from dace.sdfg import utils as sdutil



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
        return {UnderapproximateWrites, ap.AccessSets, ap.SymbolAccessSets, ap.FindAccessNodes, ap.StateReachability}

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
        symbol_access_sets: Dict[Union[SDFGState, Edge[InterstateEdge]],
                                 Tuple[Set[str], Set[str]]] = pipeline_results[ap.SymbolAccessSets.__name__][sdfg.sdfg_id]
        # list of original array names in the sdfg
        anames = sdfg.arrays.copy().keys()
        #anames = [aname for aname, a in sdfg.arrays.items() if a.transient and a.total_size == 1 and aname == "_ZQE_0"]
        anames = [aname for aname, a in sdfg.arrays.items() if a.transient ]


        # store all the new names for each original variable name
        definitions: Dict[str, set(str)] = defaultdict(None)

        # dictionary mapping each variable from the original SDFG to a dictionary mapping
        # each state to the last definition of that variable in that state
        last_defs: Dict[str, Dict[SDFGState, str]] = defaultdict(None)

        # dictionary mapping each variable in the SDFG to the states that read from it. For reads in edges
        # only take outgoing edges into account
        var_reads: Dict[str, Set[SDFGState]] = defaultdict(set)


        # array that stores "virtual" phi nodes for each variable for each SDFGstate
        # phi nodes are represented by dictionaries that contain the definitions reaching the phi node
        # can be extended for later use with path constraints for example
        phi_nodes: Dict[SDFGState, Dict[str, Dict]] = defaultdict(None)

        # initialize phi_nodes and var_reads
        for state in sdfg.states():
            phi_nodes[state] = {}

        # create dictionary mapping each node to the nodes it immediatly dominates basically representing the dominator tree
        immediate_dominated = defaultdict(None)
        for node, dom_node in immediate_dominators.items():
            if dom_node not in immediate_dominated:
                immediate_dominated[dom_node] = set()
            immediate_dominated[dom_node].add(node)
        dominators: dict[SDFGState, set[SDFGState]] = {}

        for state in sdfg.states():
            if state not in immediate_dominators.keys():
                continue
            curr_state = state
            dominators[state] = set()
            while(True):
                dominators[state].add(curr_state)
                if immediate_dominators[curr_state] is curr_state:
                    break
                curr_state = immediate_dominators[curr_state]

        def dict_dfs(graph_dict: Dict[SDFGState, SDFGState]):
            stack = []
            visited = []
            stack.append(sdfg.start_state)
            while stack:
                current_state: SDFGState = stack.pop()
                if current_state in visited:
                    continue
                visited.append(current_state)

                if graph_dict.get(current_state):
                    children = graph_dict[current_state]
                    for child in children:
                        if child not in visited:
                            stack.append(child)
            return visited

        # aux function to find the reaching definition given an AccessNode and a state in the original SDFG
        # only works with the guarantee that predecessors of state have already been traversed/renamed
        def find_reaching_def(state: SDFGState, var: str, node: nd.AccessNode = None):
            if last_defs.get(var):
                if state in last_defs[var].keys():
                    return last_defs[var][state]

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
                for edge in state.memlet_path(iedge):
                    if edge.data.data == old_name:
                        edge.data.data = newname
            for oedge in state.out_edges(node):
                for edge in state.memlet_path(oedge):
                    if edge.data.data == old_name:
                        edge.data.data = newname


        def add_definition(new_def: str, original_name: str):
            if not definitions.get(original_name):
                definitions[original_name] = set()
            definitions[original_name].add(new_def)

        def update_last_def(state: SDFGState, new_def: str, original_name: str):
            if not last_defs.get(original_name):
                last_defs[original_name] = {}
            last_defs[original_name][state] = new_def

        # first condition then assignment
        # recursively traverse the state machine and collect the path constraints until the definition of the variable is reached
        def collect_path_constraints(state: SDFGState, variable: str, orig_var: str, visited_states=[]) -> sympy.And:

            if phi_nodes[state].get(orig_var) or sdfg.in_degree(state) == 0:
                return sympy.And(True)

            no_path = True
            condition = sympy.Or(False)
            for edge in sdfg.in_edges(state):
                if not (variable is last_defs[orig_var][edge.src] and edge.src not in visited_states):
                    continue

                curr_condition = sympy.And(edge.data.condition_sympy(), collect_path_constraints(
                    edge.src, variable, orig_var, visited_states + [state]))

                # there is an assignment to a symbol that is part of the incoming constraints so we omit the constraint
                if (any(s in curr_condition.free_symbols for s in symbol_access_sets[edge][1]) or
                    any(s in curr_condition.free_symbols for s in symbol_access_sets[state][1]) or
                        any(s in curr_condition.free_symbols for s in access_sets[state][1])):
                    continue

                condition = sympy.Or(condition, curr_condition)
                no_path = False
            if no_path:
                condition = sympy.And(True)

            return condition
        
        def custom_comparator(graph,immediate_dominators, a, b, start):
            a_to_b = nx.has_path(graph, a, b)
            b_to_a = nx.has_path(graph, b, a)
            if a_to_b is b_to_a:
                i = 0
                current_node = a
                while(True):
                    if current_node is start:
                        break
                    current_node = immediate_dominators[current_node]
                    i +=1
                j = 0
                current_node = b
                while(True):
                    if current_node is start:
                        break
                    current_node = immediate_dominators[current_node]
                    j +=1
                return i - j
            elif a_to_b:
                return - 1
            else:
                return 1
            
        keyfunction = functools.cmp_to_key(lambda x,y: custom_comparator(sdfg.nx, immediate_dominators, x,y, sdfg.start_state))

        ordered_states = sdfg.states()
        ordered_states.sort(key=keyfunction)

        # insert phi nodes
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
                    if any(write_approximation[e].subset.covers_precise(array_set) for e in state.in_edges(node)):
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
                        "variables": set(),
                        "descriptor": None,
                        "path_conditions": {}
                    }
                    phi_states.add(frontier_state)
                    if frontier_state not in defining_states:
                        defining_states.add(frontier_state)

                defining_states.remove(current_state)

        for loopheader, write_dict in loop_write_approximation.items():
            if loopheader not in sdfg.states():
                continue

            for var, memlet in write_dict.items():
                if loopheader in phi_nodes.keys() and var in phi_nodes[loopheader].keys():
                    continue
                if var not in anames:
                    continue
                if memlet.subset.covers_precise(subsets.Range.from_array(sdfg.arrays[var])):
                    phi_nodes[loopheader][var] = {
                        "name": var,
                        "variables": set(),
                        "descriptor": None,
                        "path_conditions": {}
                    }

        # traverse the dominator tree depth first and rename all variables
        # FIXME: rename all edges in the memlet path, right now only incoming edges are renamed
        dom_tree_dfs = dict_dfs(immediate_dominated)
        for current_state in dom_tree_dfs:
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
            renamed = defaultdict(None)
            toposort = list(nx.topological_sort(current_state.nx))
            for node in toposort:
                if not isinstance(node, nd.AccessNode):
                    continue
                var = node.data
                if var not in anames:
                    continue
                array = sdfg.arrays[var]
                array_set = subsets.Range.from_array(array)
                # if array is not fully overwritten at this access node treat it as a use otherwise as a def
                # also make sure that this is the first (and last) renaming in this state
                iedges = current_state.in_edges(node)
                if (any(write_approximation[e].subset.covers_precise(array_set) for e in iedges) and
                        not renamed.get(var)):
                    # rename the variable to the reaching definition
                    newdesc = array.clone()
                    newname = sdfg.add_datadesc(
                        var, newdesc, find_new_name=True)
                    add_definition(newname, var)
                    update_last_def(current_state, newname, var)
                    renamed[var] = True
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
                    if not phi_nodes[successor].get(var):
                        continue
                    newname = last_defs[var][current_state]
                    phi_nodes[successor][var]["variables"].add(newname)

            # iterate over all the outgoing interstate edges of the current state and rename all the occurences of the original
            # variable to the last definition in the current state
            rename_dict = {}
            for var in last_defs.keys():
                rename_dict[var] = last_defs[var][current_state]
            for oedge in sdfg.out_edges(current_state):
                oedge.data.replace_dict(rename_dict)

        


        for state in sdfg.nodes():
            for anode in state.data_nodes():
                if state.out_degree(anode)>0:
                    var_reads[anode.data].add(state)

        # Edges that read from arrays add to the source access sets
        array_names = sdfg.arrays.keys()
        for e in sdfg.edges():
            fsyms = e.data.free_symbols & array_names
            for access in fsyms:
                var_reads[access].update({e.src})

        # iterate over the phi nodes and replace all the occurences of each parameter with the variable defined by the phi node
        to_remove: Set[str] = set()
        for state in dom_tree_dfs.__reversed__():
            if not state in phi_nodes.keys():
                continue
            phi_dict = phi_nodes[state]
            for original_var, phi_node in phi_dict.items():
                #sdfg.view("cloudsc_debug")
                newname = phi_node["name"]
                parameters = phi_node["variables"]

                candidate_states = sdfg.states()
                if not any(s in state_reach[state] or s is state for s in var_reads[newname]):
                    candidate_states = []
                    dfs_generator = sdutil.dfs_conditional(sdfg,
                                       sources=[state],
                                       condition=lambda parent, child: child not in dominators[state])
                    for other_state in dfs_generator:
                        candidate_states.append(other_state)
                # check if the phi node belongs to a loopheader that completely overwrites the array and the loop does not read from the array defined by the phi node
                # if so, only rename nodes in the loop body and nodes reached by the loopheader
                elif (state in loops.keys() and
                    (state in loop_write_approximation.keys() and
                     original_var in loop_write_approximation[state].keys() and
                     loop_write_approximation[state][original_var].subset.covers_precise(subsets.Range.from_array(sdfg.arrays[original_var])))):

                    # TODO: only fission dfs_reachable states here too
                    _, _, loop_states, _, _ = loops[state]
                    # check if loop reads from outside the loop
                    # TODO: also check for interstate reads here
                    if not any(newname in [a.label for a in access_nodes[original_var][s][0]] for s in loop_states):
                        candidate_states = state_reach[state]
                    

                

                # iterate over all the states that read from the variable and rename the occurences
                for other_state in candidate_states:
                    reads, writes = access_nodes[original_var][other_state]
                    ans = reads.union(writes)
                    for an in ans:
                        if not an.data in parameters:
                            continue
                        rename_node(other_state, an, newname)

                # rename all the occurences in other phi nodes
                for other_state in phi_nodes.keys():
                    if other_state is state or not other_state in candidate_states:
                        continue
                    other_phi_dict = phi_nodes[other_state]
                    if not original_var in other_phi_dict.keys():
                        continue
                    other_phi_node = other_phi_dict[original_var]

                    new_variables = set()
                    if other_phi_node["name"] in parameters:
                        other_phi_node["name"] = newname


                    for other_param in other_phi_node["variables"]:
                        if other_param in parameters:
                            new_variables.add(newname)
                        else:
                            new_variables.add(other_param)
                    if newname in other_phi_node["variables"]:
                        new_variables.update(parameters)
                    other_phi_node["variables"] = new_variables
                    phi_nodes[other_state][original_var] = other_phi_node

                # rename occurences in the interstate edges
                rename_dict = {}
                for parameter in parameters:
                    rename_dict[parameter] = newname
                for other_state in candidate_states:
                    other_accesses = access_sets[other_state]
                    if not original_var in other_accesses[0]:
                        continue
                    out_edges = sdfg.out_edges(other_state)
                    for oedge in out_edges:
                        oedge.data.replace_dict(rename_dict)

                # update var_read
                for parameter in parameters:
                    read_remove = set()
                    for other_state in var_reads[parameter]:
                        if other_state in candidate_states:
                            var_reads[newname].update({other_state})
                            read_remove.add(other_state)
                    var_reads[parameter].difference_update(read_remove)

                
                current_remove = set(parameters)
                current_remove.discard(newname)
                to_remove = to_remove.union(current_remove)
                # TODO: In the case where the renamed states are not all states this is wrong
                definitions[original_var] = definitions[original_var] - set(parameters)

            del phi_nodes[state]
        for parameter in to_remove:
            try:
                sdfg.remove_data(parameter)
            except:
                continue

        # import pprint
        # print("phi_nodes")
        # pprint.pprint(phi_nodes)

        definitions.clear()
        for var,an_dict in access_nodes.items():
            current_defs = set()
            for _, ans in an_dict.items():
                ans = ans[0].union(ans[1])
                for an in ans:
                    current_defs.add(an.data)
            if len(current_defs) > 1:
                definitions[var] = current_defs


        results = definitions
        return results

    def report(self, pass_retval: Any) -> Optional[str]:
        return f'Renamed {len(pass_retval)} scalars: {pass_retval}.'
