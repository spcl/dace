# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
"""
Analysis helpers for autodiff
"""
from typing import Dict, Set, Tuple, Optional
import collections

import networkx as nx

from dace.sdfg import SDFG, SDFGState, nodes, utils as sdfg_utils
from dace.transformation.passes import analysis
from dace.sdfg.state import FunctionCallRegion

AccessSets = Dict[SDFGState, Tuple[Set[str], Set[str]]]


def dependency_analysis(sdfg: SDFG) -> Dict[str, Set[str]]:
    """
    Analyze read dependencies of arrays in the SDFG.

    :param sdfg: SDFG to analyze
    :returns: A dictionary mapping array names to a list of read dependencies.
    """

    # FIXME can be made more efficient
    dependencies = nx.DiGraph()
    for sdfg_node in sdfg.nodes():
        if isinstance(sdfg_node, SDFGState):
            for node in sdfg_node.data_nodes():
                for edge in sdfg_node.edge_bfs(node, reverse=True):
                    dependencies.add_edge(node.data, edge.data.data)
        elif isinstance(sdfg_node, FunctionCallRegion):
            for state in sdfg_node.nodes():
                assert isinstance(state, SDFGState)
                for node in state.data_nodes():
                    for edge in state.edge_bfs(node, reverse=True):
                        dependencies.add_edge(node.data, edge.data.data)

    dependencies = nx.transitive_closure(dependencies)
    result = {}
    for array in dependencies:
        result[array] = {nbr for nbr in dependencies.neighbors(array)}
    return result


def inverse_reachability(sdfg: SDFG) -> Dict[SDFGState, Set[SDFGState]]:

    reachability = analysis.StateReachability().apply_pass(sdfg, {})
    inverse_reachability = collections.defaultdict(set)
    # iterate over cfg_ids
    for cfg_id in reachability.keys():
        for pred, successors in reachability[cfg_id].items():
            for successor in successors:
                inverse_reachability[successor].add(pred)

    return inverse_reachability


def is_previously_written(sdfg: SDFG,
                          state: SDFGState,
                          node: nodes.Node,
                          array_name: str,
                          access_sets: Optional[AccessSets] = None) -> bool:
    """
    Determine whether the given array name was written before the current node.

    :param sdfg: the sdfg containing the node
    :param state: the state containing the node
    :param node: the node to check
    :param array_name: the array name to check
    :returns: True if the array was written before the node, False otherwise.
    """

    if access_sets is None:
        access_sets = analysis.AccessSets().apply_pass(sdfg, {})

    reachable = inverse_reachability(sdfg)

    # Check the current state
    for subgraph in sdfg_utils.concurrent_subgraphs(state):
        if node in subgraph.nodes():
            # Get all the access nodes in the subgraph to the same data
            for other_node in subgraph.data_nodes():
                if other_node != node and other_node.data == array_name:
                    # Check if this is a write node
                    for in_edge in subgraph.in_edges(other_node):
                        if in_edge.data.data == array_name:
                            # Check if there's a path to our node,
                            # since we only care about writes that happen before the current node
                            if nx.has_path(state.nx, other_node, node):
                                return True
        else:
            # This is not our current subgraph, check the write states
            _, write_set = subgraph.read_and_write_sets()
            if array_name in write_set:
                return True

    # Check other states
    for predecessor in reachable[state]:
        _, write_set = access_sets[predecessor]
        if array_name in write_set:
            return True
    return False
