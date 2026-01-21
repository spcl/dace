# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.

from collections import defaultdict
from typing import Any, Callable, Dict, Iterable, Optional, Set, Tuple, List, Union
import networkx as nx

from dace.properties import make_properties
from dace import SDFG, properties
from dace import subsets
from dace.sdfg import nodes as nd
from dace.sdfg import SDFGState
from dace.memlet import Memlet
from dace.transformation import pass_pipeline as ppl
from dace.sdfg.graph import Edge
from dace.sdfg.analysis.writeset_underapproximation import UnderapproximateWrites
from dace.transformation.passes import analysis as ap


class _PhiNode():

    def __init__(self, name: str, variables: Set[str]):
        self.name: str = name
        self.variables: Set[str] = variables

@make_properties
class VariableFission(ppl.Pass):
    """
    Fission transient arrays that are dominated by full writes 
    to the whole array into separate data containers.
    """

    CATEGORY: str = 'Optimization Preparation'

    fission_arrays = properties.Property(dtype=bool, default=True, desc='''If True, only fissions variables of size > 1.
                                                                        If false only fissions scalar variables.''')
    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Descriptors | ppl.Modifies.AccessNodes

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return modified & ppl.Modifies.AccessNodes

    def depends_on(self):
        return {
            UnderapproximateWrites, ap.AccessSets, ap.FindAccessNodes,
            ap.StateReachability
        }

    def apply_pass(
            self, sdfg: SDFG,
            pipeline_results: Dict[str, Any]) -> Optional[Dict[str, Set[str]]]:
        """
        Fission transient arrays that are dominated by full writes
        to the whole array into separate data containers.

        :param sdfg: The SDFG to modify.
        :param pipeline_results: If in the context of a ``Pipeline``, a dictionary that is
                                populated with prior Pass results as ``{Pass subclass name:
                                returned object from pass}``. If not run in a pipeline, an
                                empty dictionary is expected.
        :return: A dictionary mapping the original name to a set of all new names created
                for each data container.
        """
        results: Dict[str, Set[str]] = defaultdict(set)
        write_approximation: Dict[Edge, Memlet] = pipeline_results[
            UnderapproximateWrites.__name__]["approximation"]
        loop_write_approximation: Dict[SDFGState, Dict[ str, Memlet]] = pipeline_results[
                UnderapproximateWrites.__name__]["loop_approximation"]
        loops: Dict[SDFGState, Tuple[SDFGState, SDFGState, List[SDFGState], str, subsets.Range]
                    ] = pipeline_results[UnderapproximateWrites.__name__]["loops"]
        state_reach: Dict[SDFGState, Set[SDFGState]] = pipeline_results[
            ap.StateReachability.__name__][sdfg.sdfg_id]
        access_sets: Dict[SDFGState, Tuple[Set[str], Set[str]]] = pipeline_results[
                              ap.AccessSets.__name__][sdfg.sdfg_id]
        access_nodes: Dict[str, Dict[SDFGState, Tuple[ Set[nd.AccessNode], Set[nd.AccessNode]]]
                           ] = pipeline_results[ap.FindAccessNodes.__name__][sdfg.sdfg_id]
        # list of original variable names in the sdfg that fissioning is performed on
        variable_names: List[str] =  [
            aname for aname, a in sdfg.arrays.items()
            if a.transient and not a.total_size == 1
        ] if self.fission_arrays else [
            aname for aname, a in sdfg.arrays.items()
            if a.transient and a.total_size == 1
        ]

        # dictionary that stores "virtual" phi nodes for each variable and SDFGstate.
        # phi nodes are represented by dictionaries that contain:
        # - the variable defined by the phi-node
        # - the corresponding descriptor
        # - the definitions reaching the phi node
        # can be extended for later use with path constraints for example
        phi_nodes: Dict[SDFGState, Dict[str, _PhiNode]] = defaultdict(None)

        # maps each variable to it's defining writes that don't involve phi-nodes
        def_states: Dict[str, Set[SDFGState]] = {}

        # maps each variable to it's defining writes that involve phi-nodes
        def_states_phi: Dict[str, Set[SDFGState]] = {}

        phi_nodes, def_states, def_states_phi = _insert_phi_nodes(
            sdfg,
            variable_names,
            write_approximation,
            loop_write_approximation,
            access_nodes,
        )

        _rename(sdfg, write_approximation, phi_nodes, variable_names)

        _eliminate_phi_nodes(sdfg, phi_nodes, def_states, def_states_phi,
                             loops, loop_write_approximation, state_reach,
                             access_nodes, access_sets)

        # store all the new names for each original variable name
        definitions: Dict[str, Set[str]] = defaultdict(None)
        for var, an_dict in access_nodes.items():
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
        return f'Renamed {len(pass_retval)} arrays: {pass_retval}.'


def _insert_phi_nodes(
    sdfg: SDFG, 
    variable_names: List[str], 
    write_approximation: Dict[Edge, Memlet],
    loop_write_approximation: Dict[SDFGState, Dict[str, Memlet]],
    access_nodes: Dict[str, Dict[SDFGState, Tuple[Set[nd.AccessNode], Set[nd.AccessNode]]]]
) -> Tuple[Dict[SDFGState, Dict[str, _PhiNode]], Dict[str, Set[SDFGState]], Dict[str, Set[SDFGState]]]:
    """
    Inserts phi-nodes at loop-headers of loops that overwrite an 
    array and at merging states of states that overwrite an array.

    :param sdfg: The SDFG to perform the insertion on.
    :param variable_names: The variable names to consider.
    :param write_approximation: Underapproximation of writes in SDFG.
    :param loop_write_approximation: Underapproximation of write-sets of loops in the SDFG.
    :param access_nodes: mapping from variables to their read and write nodes for each state.
    :return: 3-tuple containing dictionary that maps each state in the SDFG to its phi-nodes, 
    all defining states in the original SDFG, defining states that define variables via a phi-node.
    """
    phi_nodes: Dict[SDFGState, Dict[str, _PhiNode]] = defaultdict(None)
    for state in sdfg.states():
        phi_nodes[state] = {}

    defining_states_phi: Dict[str,
                              Set[SDFGState]] = _insert_phi_nodes_loopheaders(
                                  sdfg, variable_names,
                                  loop_write_approximation, phi_nodes)

    # for each variable find states that overwrite it
    defining_states: Dict[str, Set[SDFGState]] = _find_defining_states(
        sdfg, variable_names, phi_nodes, access_nodes, write_approximation)

    _insert_phi_nodes_regular(sdfg, defining_states, phi_nodes,
                              defining_states_phi)

    return (phi_nodes, defining_states, defining_states_phi)


def _rename(
        sdfg: SDFG, 
        write_approximation: Dict[Edge, Memlet],
        phi_nodes: Dict[SDFGState, Dict[str, _PhiNode]],
        variable_names: List[str]
        ):
    """
    For each variable introduces new variable for each redefinition of that variable and renames 
    uses and partial writes accordingly.

    :param sdfg: The SDFG to perform the renaming on.
    :param phi_nodes: dictionary that maps each state in the SDFG to its phi-nodes.
    :param variable_names: List of variables in the SDFG to consider.
    """

    # dictionary mapping each variable from the original SDFG to a dictionary mapping
    # each state to the last definition of that variable in that state
    last_defs: Dict[str, Dict[SDFGState, str]] = defaultdict(None)
    immediate_dominators = nx.dominance.immediate_dominators( sdfg.nx, sdfg.start_state)
    dom_tree_dfs = _dominator_tree_DFS_order(sdfg.start_state, immediate_dominators)
    # traverse the dominator tree depth first and rename all variables
    for current_state in dom_tree_dfs:
        _rename_DFG_and_interstate_edges(sdfg, current_state, variable_names,
                                         phi_nodes, write_approximation,
                                         last_defs, immediate_dominators)
        _propagate_new_names_to_phi_nodes(sdfg, current_state, variable_names,
                                          phi_nodes, last_defs)


def _eliminate_phi_nodes(
        sdfg: SDFG,
        phi_nodes: Dict[SDFGState, Dict[str, _PhiNode]],
        def_states: Dict[str, Set[SDFGState]],
        def_states_phi: Dict[str, Set[SDFGState]],
        loops: Dict[SDFGState, Tuple[SDFGState, SDFGState, List[SDFGState], str, subsets.Range]],
        loop_write_approximation: Dict[SDFGState, Dict[str, Memlet]],
        state_reach: Dict[SDFGState, Set[SDFGState]],
        access_nodes: Dict[str, Dict[SDFGState, Tuple[Set[nd.AccessNode], Set[nd.AccessNode]]]],
        access_sets: Dict[SDFGState, Tuple[Set[str], Set[str]]]
        ):
    """
    Deletes phi-nodes from phi-node dictionary. If variable version introduced by phi-node is used
    anywhere in the SDFG all the phi-related variables of that phi-node are renamed to a common
    variable name. If the phi-node is at a loop-header state and the loop overwrites the variable
    renaming is only performed in the states that are reachable from the phi node. This way new
    variables are introduced after overwriting loops.

    :param sdfg: SDFG to perform the phi-node elimination on.
    :param def_states: dictionary that maps each variable to the states which overwrite that 
    variable.
    :param def_states_phi: dictionary that maps each variable to the states which overwrite that
    variable via a phi-node.
    :param loops: Dictionary containing information about for-loops in the given SDFG.
    :param loop_write_approximation: Underapproximation of write-sets of loops in the SDFG.
    :param state_reach: Maps each state to the states it can reach in the SDFG.
    :param access_nodes: Mapping from variables to their read and write nodes for each state.
    :param access_sets: Maps each state in the SDFG to the data containers it accesses.
    """

    # dictionary mapping each variable in the SDFG to the states that read from it.
    # For reads in edges only take outgoing edges into account
    var_reads: Dict[str, Set[SDFGState]] = defaultdict(set)
    for state in sdfg.states():
        for anode in state.data_nodes():
            if state.out_degree(anode) > 0:
                var_reads[anode.data].add(state)
    array_names = sdfg.arrays.keys()
    for e in sdfg.edges():
        fsyms = e.data.free_symbols & array_names
        for access in fsyms:
            var_reads[access].update({e.src})

    # iterate over the phi nodes and replace all the occurences of each parameter with the
    # variable defined by the phi node
    to_remove: Set[str] = set()
    for state in sdfg.states():
        if not state in phi_nodes:
            continue
        phi_dict = phi_nodes[state]
        for original_var, phi_node in phi_dict.items():
            newname = phi_node.name
            parameters = phi_node.variables
            # Find all states that can be reached by the variable defined by the current phi-node
            reached_by_def = _conditional_dfs(
                sdfg, lambda node:
                (node is state or (node not in def_states_phi[original_var] and
                                   node not in def_states[original_var])),
                state)

            candidate_states = sdfg.states()
            loop_states = []
            is_read = True  # is the variable defined by the phi-node read in the graph
            overwriting_loop = False  # if the state is a loop guard, does the loop overwrite the variable

            # if the phi node belongs to a loopheader that completely overwrites the array
            # and the loop does not read from the array defined by the phi node
            # only rename nodes reached by the loopheader
            if (state in loop_write_approximation
                    and original_var in loop_write_approximation[state]
                    and loop_write_approximation[state][original_var].subset
                    and loop_write_approximation[state]
                [original_var].subset.covers_precise(
                    subsets.Range.from_array(sdfg.arrays[original_var]))):
                _, _, loop_states, _, _ = loops[state]
                # check if loop reads from outside the loop
                if not any(newname in [
                        a.label
                        for a in access_nodes[original_var][other_state][0]
                ] or other_state in var_reads[newname]
                           for other_state in loop_states):

                    candidate_states = state_reach[state]
                    overwriting_loop = True
            # if the variable defined by the phi node is read by any other state we perform
            # renaming in the whole SDFG. If not we only perform phi propagation in all the
            # states that are reachable by the defined variable
            elif not any(
                    other_state in state_reach[state] or other_state is state
                    for other_state in var_reads[newname]):
                candidate_states = reached_by_def
                is_read = False

            _rename_phi_related_phi_nodes_and_propagate_parameters(
                parameters, original_var, newname, candidate_states, state,
                phi_nodes, reached_by_def, is_read)
            # rename accesses
            if is_read:
                _rename_phi_related_accesses(sdfg, parameters, original_var,
                                             newname, candidate_states,
                                             loop_states, access_nodes,
                                             access_sets, overwriting_loop)

            # update var_read if any renaming was done
            if is_read:
                # if a state read from the parameter it now reads from the variable
                # defined by the current phi node
                for parameter in parameters:
                    if parameter is newname:
                        continue
                    read_remove = set()
                    for other_state in var_reads[parameter]:
                        if other_state in candidate_states:
                            var_reads[newname].update({other_state})
                            read_remove.add(other_state)
                    var_reads[parameter].difference_update(read_remove)
                    # try to remove renamed parameters from SDFG
                    to_remove.add(parameter)

            # remove the phi-node if it is not an overwriting loop. If not, we keep it such that no
            # definition is propagated past this phi node
            if not overwriting_loop:
                def_states_phi[original_var].remove(state)

        del phi_nodes[state]
    for parameter in to_remove:
        try:
            sdfg.remove_data(parameter)
        except ValueError:
            continue


def _rename_phi_related_accesses(
        sdfg: SDFG,
        phi_node_parameters: Set[str],
        original_variable: str,
        new_name: str,
        candidate_states: Iterable[SDFGState],
        loop_states: Iterable[SDFGState],
        access_nodes: Dict[str, Dict[SDFGState, Tuple[Set[nd.AccessNode], Set[nd.AccessNode]]]],
        access_sets: Dict[SDFGState, Tuple[Set[str], Set[str]]],
        overwriting_loop: bool
        ):
    """
    Rename all accesses to phi-related variables of a phi-node P to a common variable name.

    :param sdfg: SDFG to perform the renaming on.
    :param phi_node_parameters: Parameters of P.
    :param original_variable: Name of the variable in the original SDFG.
    :param new_name: Name the phi-related variables are renamed to.
    :param candidate_states: States to perform the renaming in.
    :param loop_states: States of the loop body if P is in the loop header state of an
    overwriting loop.
    :param access_nodes: Mapping from variables to their read and write nodes for each state.
    :param access_sets: Maps each state in the SDFG to the data containers it accesses.
    :param overwriting_loop: true iff P is in the loop header state of an overwriting loop
    that can be fissioned safely.
    """
    rename_dict = {}
    for parameter in phi_node_parameters:
        rename_dict[parameter] = new_name
    for other_state in candidate_states:
        if overwriting_loop and other_state not in loop_states:
            continue
        # rename all accesses to the parameters by accessnodes
        reads, writes = access_nodes[original_variable][other_state]
        ans = reads.union(writes)
        for an in ans:
            if not an.data in phi_node_parameters:
                continue
            _rename_node(other_state, an, new_name)

        # rename all accesses to the parameters by interstate edges
        other_accesses = access_sets[other_state]
        if original_variable in other_accesses[0]:
            for oedge in sdfg.out_edges(other_state):
                oedge.data.replace_dict(rename_dict)


def _rename_phi_related_phi_nodes_and_propagate_parameters(
    phi_node_parameters: Set[str],
    original_variable: str,
    new_name: str,
    candidate_states: Iterable[SDFGState],
    phi_node_origin_state: SDFGState,
    phi_nodes: Dict[SDFGState, Dict[str, _PhiNode]],
    reached_by_def: Set[SDFGState],
    is_read: bool,
):
    """
    Renames all phi-related variables of a phi-node P that are defined by other phi-nodes.
    Propagates the parameters of P to all the phi-nodes that can be reached by P. Modifies
    the phi-node dictionary.

    :param phi_node_parameters: Parameters of P.
    :param original_variable: Name of the variable in the original SDFG.
    :param new_name: Name the phi-related variables are renamed to.
    :param candidate_states: States to perform the renaming in.
    :param phi_node_origin_state: State P is located in.
    :param phi_nodes: Dictionary that maps each state in the SDFG to its phi-nodes.
    :param reached_by_def: States that can be reached from P.
    :param is_read: True iff variable defined by P is read anywhere else in the SDFG.
    """
    for other_state in candidate_states:
        # rename all phi nodes and propagate
        if (not other_state is phi_node_origin_state
                and other_state in phi_nodes
                and original_variable in phi_nodes[other_state]):

            other_phi_node = phi_nodes[other_state][original_variable]
            new_variables = set()

            # if the variable defined by the other phi-node is in the parameters
            # rename the variable
            if other_phi_node.name in phi_node_parameters and is_read:
                other_phi_node.name = new_name

            # propagate parameter or variable defined by phi node to other phi nodes
            # that can be reached by the definition
            if other_state in reached_by_def:
                if not is_read:
                    new_variables.update(phi_node_parameters)
                else:
                    new_variables.add(new_name)
            new_variables.update(other_phi_node.variables)
            other_phi_node.variables = new_variables
            phi_nodes[other_state][original_variable] = other_phi_node


def _update_last_def(
        state: SDFGState,
        new_def: str,
        original_name: str,
        last_defs: Dict[str, Dict[SDFGState, str]]
        ):
    """
    Inserts new variable name of variable in the original SDFG to the last definition 
    dictionary.

    :param state: State the new variable was introduced in.
    :param new_def: Name of the new variable.
    :param original_name: Name of the original variable.
    :param last_defs: For each variable in the original SDFG maps states to the last 
    definition of that variable in that state.
    """
    if not last_defs.get(original_name):
        last_defs[original_name] = {}
    last_defs[original_name][state] = new_def


def _find_reaching_def(
        state: SDFGState,
        var: str,
        last_defs: Dict[str, Dict[SDFGState, str]],
        phi_nodes: Dict[SDFGState, Dict[str, _PhiNode]],
        immediate_dominators: Dict[SDFGState, SDFGState]
        ) -> str:
    """
    Finds the variable definition reaching a variable given an AccessNode and the state it
    is located in. First looks in the last definition dictionary and then in the successor
    states.

    :param state: The state the access node is located in.
    :param last_defs: For each variable in the original SDFG maps states to the last 
    definition of that variable in that state.
    :param phi_nodes: Dictionary that maps each state in the SDFG to its phi-nodes.
    :param immediate_dominators: Maps each state to its immediate dominator.
    :return: Name of the variable reaching the given AccessNode as a string.
    """

    if var in last_defs and state in last_defs[var]:
        return last_defs[var][state]
    # if there is a phi node for the variable in the same state, return the variable
    # name defined by the phi node
    if phi_nodes[state].get(var):
        return phi_nodes[state][var].name
    # otherwise return the last definition of the immediate dominator
    idom = immediate_dominators[state]
    if var in last_defs and idom in last_defs[var]:
        return last_defs[var][idom]
    # in case the state is the initial state and there isn't any reaching definition
    # in the current state just return the original variable
    return var


def _rename_DFG_and_interstate_edges(
        sdfg: SDFG,
        state: SDFGState,
        variable_names: List[str],
        phi_nodes: Dict[SDFGState, Dict[str, _PhiNode]],
        write_approximation: Dict[Edge, Memlet],
        last_defs: Dict[str, Dict[SDFGState, str]],
        immediate_dominators: Dict[SDFGState, SDFGState]
        ):
    """
    Given a state iterates over all candidate variables and introduces new variable if it is
    overwritten in that state. Occurences of that variable in the same state and in outgoing
    interstate edges are renamed and the last_defs dictionary is updted. Only introduces one 
    new variable name per candidate variable and per state. If no new variable is introduced 
    in the given state the occurences of the variable are renamed to the last definition in 
    the predecessor state.

    :param sdfg: The SDFG to perform the renaming on.
    :param variable_names: List of variables in the SDFG to consider.
    :param phi_nodes: Dictionary that maps each state in the SDFG to its phi-nodes.
    :param write_approximation: Underapproximation of writes in SDFG.
    :param last_defs: For each variable in the original SDFG maps states to the last 
    definition of that variable in that state.
    :param immediate_dominators: Maps each state to its immediate dominator.
    """
    for var in variable_names:
        # check if there is a phi node for the current variable in the current state
        if not phi_nodes[state].get(var):
            continue
        newdesc = sdfg.arrays[var].clone()
        newname = sdfg.add_datadesc(var, newdesc, find_new_name=True)
        phi_nodes[state][var].name = newname
        _update_last_def(state, newname, var, last_defs)
    # rename data nodes
    # get topological ordering of nodes in the dataflowgraph
    renamed = defaultdict(None)
    toposort = list(nx.topological_sort(state.nx))
    for node in toposort:
        if not isinstance(node, nd.AccessNode):
            continue
        var = node.data
        if var not in variable_names:
            continue
        # if array is not fully overwritten at this access node treat it as a use
        # otherwise as a def also make sure that this is the first (and last)
        # renaming in this state
        array = sdfg.arrays[var]
        array_set = subsets.Range.from_array(array)
        iedges = state.in_edges(node)
        if (any(write_approximation[edge].subset.covers_precise(array_set)
                for edge in iedges) and not renamed.get(var)):
            # rename the variable to the reaching definition
            newdesc = array.clone()
            newname = sdfg.add_datadesc(var, newdesc, find_new_name=True)
            _update_last_def(state, newname, var, last_defs)
            renamed[var] = True
        else:
            newname = _find_reaching_def(state, var, last_defs, phi_nodes,
                                         immediate_dominators)
        _rename_node(state, node, newname)
    # define last definition in this state if it has not been defined yet
    for var in variable_names:
        if not last_defs.get(var):
            last_defs[var] = {}
        if not last_defs[var].get(state):
            last_defs[var][state] = _find_reaching_def(state, var, last_defs,
                                                       phi_nodes,
                                                       immediate_dominators)
    # iterate over all the outgoing interstate edges of the current state and
    # rename all the occurences of the original variable to the last definition
    # in the current state
    rename_dict = {}
    for var in last_defs.keys():
        rename_dict[var] = last_defs[var][state]
    for oedge in sdfg.out_edges(state):
        oedge.data.replace_dict(rename_dict)


def _propagate_new_names_to_phi_nodes(
        sdfg: SDFG,
        state: SDFGState,
        variable_names: List[str],
        phi_nodes: Dict[SDFGState, Dict[str, _PhiNode]],
        last_defs: Dict[str, Dict[SDFGState, str]]
        ):
    """
    Inserts newly introduced variable names in the phi-nodes they reach. Directly modifies
    the phi-nodes in the phi_nodes dictionary.

    :param sdfg: The SDFG to perform the propagation on.
    :param variable_names: List of variables in the SDFG to consider.
    :param phi_nodes: Dictionary that maps each state in the SDFG to its phi-nodes.
    :param last_defs: For each variable in the original SDFG maps states to the last 
    definition of that variable in that state.
    """
    successors = [edge.dst for edge in sdfg.out_edges(state)]
    for successor in successors:
        for var in variable_names:
            if not phi_nodes[successor].get(var):
                continue
            newname = last_defs[var][state]
            phi_nodes[successor][var].variables.add(newname)


def _rename_node(
        state: SDFGState,
        node: nd.AccessNode,
        new_name: str
        ):
    """
    Given an AccessNode renames all the occurences of the variable it accesses in connected 
    memlets to a new name.

    :param state: State the AccessNode is located in.
    :param node: AccessNode to perform the renaming on.
    :param new_name: Name to rename the original variable to.
    """
    # helper function that traverses memlet trees of all incoming and outgoing
    # edges of an accessnode and renames it to newname
    old_name = node.data
    node.data = new_name
    for iedge in state.in_edges(node):
        for edge in state.memlet_tree(iedge):
            if edge.data.data == old_name:
                edge.data.data = new_name
    for oedge in state.out_edges(node):
        for edge in state.memlet_tree(oedge):
            if edge.data.data == old_name:
                edge.data.data = new_name


def _dominator_tree_DFS_order(
        start_state: SDFGState,
        immediate_dominators: Dict[SDFGState, SDFGState]
        ) -> List[SDFGState]:
    """
    Returns DFS traversial order of dominator tree. The dominator tree is inferred 
    from the immediate_dominator relation.

    :param start_state: Starting node of DFS traversal.
    :param immediate_dominators: Dictionary that maps each state to its immediate dominator.
    :returns: List of states in DFS traversal order of the dominator tree.
    """

    # build dominator tree
    dominator_tree = defaultdict(None)
    for node, dominator_node in immediate_dominators.items():
        if dominator_node not in dominator_tree:
            dominator_tree[dominator_node] = set()
        dominator_tree[dominator_node].add(node)

    stack = []
    visited = []
    stack.append(start_state)
    while stack:
        current_state: SDFGState = stack.pop()
        if current_state in visited:
            continue
        visited.append(current_state)
        if not current_state in dominator_tree:
            continue
        children = dominator_tree[current_state]
        for child in children:
            if child not in visited:
                stack.append(child)
    return visited


def _conditional_dfs(
        sdfg: SDFG,
        condition: Callable[[Union[SDFGState, None]], bool],
        start: Union[SDFGState, None] = None
        ) -> Set[SDFGState]:
    """
    Returns DFS traversal of given SDFG. Only traverses past a state if the condition-function
    for that state returns true.

    :param sdfg: SDFG to traverse
    :param condition: Function that takes a state and returns true iff the traversal should
    go past this state
    :param start: Starting state of the DFS traversal
    :return: Traversed states in DFS order
    """
    successors = sdfg.successors
    visited = set()
    node = start
    stack = [node]
    while stack:
        node = stack.pop()
        if node not in visited:
            visited.add(node)
            if condition(node):
                stack.extend(iter(successors(node)))
    return visited


def _insert_phi_nodes_loopheaders(
    sdfg: SDFG,
    variable_names: List[str],
    loop_write_approximation: Dict[SDFGState, Dict[str, Memlet]],
    phi_nodes: Dict[SDFGState, Dict[str, _PhiNode]]
    ) -> Dict[str, Set[SDFGState]]:
    """
    For each variable inserts phi-nodes at loop-headers of loops that overwrite that variable.
    Directly inserts phi-nodes in phi-node dictionary given as argument.

    :param sdfg: The SDFG to perform the insertion on.
    :param variable_names: List of variables in the SDFG to consider.
    :param loop_write_approximation: Underapproximation of write-sets of loops in the SDFG.
    :param phi_nodes: dictionary that maps each state in the SDFG to its phi-nodes.
    :return: dictionary mapping each variable to its defining phi-nodes in the loop-header states.
    """
    def_states_phi: Dict[str, Set[SDFGState]] = {}
    for loopheader, write_dict in loop_write_approximation.items():
        if loopheader not in sdfg.states():
            continue
        for var, memlet in write_dict.items():
            if loopheader in phi_nodes.keys(
            ) and var in phi_nodes[loopheader].keys():
                continue
            if var not in variable_names:
                continue
            if memlet.subset.covers_precise(
                    subsets.Range.from_array(sdfg.arrays[var])):
                phi_nodes[loopheader][var] = _PhiNode(var, set())
                if var not in def_states_phi:
                    def_states_phi[var] = set()
                def_states_phi[var].add(loopheader)
    return def_states_phi


def _find_defining_states(
        sdfg: SDFG,
        variable_names: List[str],
        phi_nodes: Dict[SDFGState, Dict[str, _PhiNode]],
        access_nodes: Dict[str, Dict[SDFGState, Tuple[Set[nd.AccessNode], Set[nd.AccessNode]]]],
        write_approximation: Dict[Edge, Memlet]
        ) -> Dict[str, Set[SDFGState]]:
    """
    For each variable finds states in which that variable is overwritten.

    :param sdfg: The SDFG to traverse.
    :param variable_names: List of variables in the SDFG to consider.
    :param access_nodes: Mapping from variables to their read and write nodes for each state.
    :param write_approximation: Underapproximation of writes in SDFG.
    :returns: Mapping that maps each candidate variable to its defining states in the SDFG.
    """

    def_states: Dict[str, Set[SDFGState]] = {}

    for var in variable_names:
        # iterate over access nodes to the array in the current state and check if it
        # fully overwrites the array with the write underapproximation
        desc = sdfg.arrays[var]
        array_set = subsets.Range.from_array(desc)
        defining_states = set()
        for state in sdfg.states():
            # loopheaders that have phi nodes are also defining states
            if state in phi_nodes.keys():
                if var in phi_nodes[state].keys():
                    defining_states.add(state)
            # check if there is a write to the descriptor in the current state
            write_nodes = access_nodes[var][state][1]
            if len(write_nodes) == 0:
                continue
            for node in write_nodes:
                # if any of the edges fully overwrites the array add the state to
                # the defining states
                if any(write_approximation[edge].subset.covers_precise(
                        array_set) for edge in state.in_edges(node)):
                    defining_states.add(state)
                    break
        def_states[var] = defining_states

    return def_states


def _insert_phi_nodes_regular(
        sdfg: SDFG,
        def_states: Dict[str, Set[SDFGState]],
        phi_nodes: Dict[SDFGState, Dict[str, _PhiNode]],
        def_states_phi: Dict[str, Set[SDFGState]]
        ):
    """
    Insert phi-nodes in SDFG similar to standard SSA. Non overwriting writes to arrays are treated
    as reads

    :param sdfg: The SDFG to perform the insertion on.
    :param phi_nodes: dictionary that maps each state in the SDFG to its phi-nodes.
    :param def_states_phi: dictionary that maps each variable to the states which overwrite that
    variable via a phi node.
    """
    dominance_frontiers = nx.dominance.dominance_frontiers(
        sdfg.nx, sdfg.start_state)
    for var, defining_states in def_states.items():
        phi_states = set()
        defining_states = defining_states.copy()
        # array of states that define/fully overwrite the array
        while defining_states:
            current_state = next(iter(defining_states))
            dominance_frontier = dominance_frontiers[current_state]

            for frontier_state in dominance_frontier:
                # check if this state was already handled
                if frontier_state in phi_states:
                    continue
                phi_nodes[frontier_state][var] = _PhiNode(var, set())
                phi_states.add(frontier_state)
                if frontier_state not in defining_states:
                    defining_states.add(frontier_state)

            defining_states.remove(current_state)

        for state, phi_dict in phi_nodes.items():
            if var not in phi_dict.keys():
                continue
            if var not in def_states_phi:
                def_states_phi[var] = set()
            def_states_phi[var].add(state)
