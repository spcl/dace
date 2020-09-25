# Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
""" Contains functions related to pattern matching in transformations. """

from dace.sdfg import SDFG, SDFGState
from dace.sdfg import graph as gr, nodes as nd
import networkx as nx
from networkx.algorithms import isomorphism as iso
from typing import Dict, List, Tuple, Type, Union
from dace.transformation.transformation import Transformation


def collapse_multigraph_to_nx(
        graph: Union[gr.MultiDiGraph, gr.OrderedMultiDiGraph]) -> nx.DiGraph:
    """ Collapses a directed multigraph into a networkx directed graph.

        In the output directed graph, each node is a number, which contains
        itself as node_data['node'], while each edge contains a list of the
        data from the original edges as its attribute (edge_data[0...N]).

        :param graph: Directed multigraph object to be collapsed.
        :return: Collapsed directed graph object.
  """

    # Create the digraph nodes.
    digraph_nodes: List[Tuple[int, Dict[str,
                                        nd.Node]]] = ([None] *
                                                      graph.number_of_nodes())
    node_id = {}
    for i, node in enumerate(graph.nodes()):
        digraph_nodes[i] = (i, {'node': node})
        node_id[node] = i

    # Create the digraph edges.
    digraph_edges = {}
    for edge in graph.edges():
        src = node_id[edge.src]
        dest = node_id[edge.dst]

        if (src, dest) in digraph_edges:
            edge_num = len(digraph_edges[src, dest])
            digraph_edges[src, dest].update({edge_num: edge.data})
        else:
            digraph_edges[src, dest] = {0: edge.data}

    # Create the digraph
    result = nx.DiGraph()
    result.add_nodes_from(digraph_nodes)
    result.add_edges_from(digraph_edges)

    return result


def type_match(node_a, node_b):
    """ Checks whether the node types of the inputs match.
        :param node_a: First node.
        :param node_b: Second node.
        :return: True if the object types of the nodes match, False otherwise.
        :raise TypeError: When at least one of the inputs is not a dictionary
                          or does not have a 'node' attribute.
        :raise KeyError: When at least one of the inputs is a dictionary,
                         but does not have a 'node' key.
    """
    return isinstance(node_a['node'], type(node_b['node']))


def match_pattern(state: SDFGState,
                  pattern: Type[Transformation],
                  sdfg: SDFG,
                  node_match=type_match,
                  edge_match=None,
                  strict=False):
    """ Returns a list of single-state Transformations of a certain class that
        match the input SDFG.
        :param state: An SDFGState object to match.
        :param pattern: Transformation type to match.
        :param sdfg: The SDFG to match in.
        :param node_match: Function for checking whether two nodes match.
        :param edge_match: Function for checking whether two edges match.
        :param strict: Only match transformation if strict (i.e., can only
                       improve the performance/reduce complexity of the SDFG).
        :return: A list of Transformation objects that match.
    """

    # Collapse multigraph into directed graph
    # Handling VF2 in networkx for now
    digraph = collapse_multigraph_to_nx(state)

    for idx, expression in enumerate(pattern.expressions()):
        cexpr = collapse_multigraph_to_nx(expression)
        graph_matcher = iso.DiGraphMatcher(digraph,
                                           cexpr,
                                           node_match=node_match,
                                           edge_match=edge_match)
        for subgraph in graph_matcher.subgraph_isomorphisms_iter():
            subgraph = {
                cexpr.nodes[j]['node']: state.node_id(digraph.nodes[i]['node'])
                for (i, j) in subgraph.items()
            }
            try:
                match_found = pattern.can_be_applied(state,
                                                     subgraph,
                                                     idx,
                                                     sdfg,
                                                     strict=strict)
            except Exception as e:
                print('WARNING: {p}::can_be_applied triggered a {c} exception:'
                      ' {e}'.format(p=pattern.__name__,
                                    c=e.__class__.__name__,
                                    e=e))
                match_found = False
            if match_found:
                yield pattern(sdfg.sdfg_id, sdfg.node_id(state), subgraph, idx)

    # Recursive call for nested SDFGs
    for node in state.nodes():
        if isinstance(node, nd.NestedSDFG):
            sub_sdfg = node.sdfg
            for sub_state in sub_sdfg.nodes():
                yield from match_pattern(sub_state,
                                         pattern,
                                         sub_sdfg,
                                         strict=strict)


def match_stateflow_pattern(sdfg,
                            pattern,
                            node_match=type_match,
                            edge_match=None,
                            strict=False):
    """ Returns a list of multi-state Transformations of a certain class that
        match the input SDFG.
        :param sdfg: The SDFG to match in.
        :param pattern: Transformation object to match.
        :param node_match: Function for checking whether two nodes match.
        :param edge_match: Function for checking whether two edges match.
        :param strict: Only match transformation if strict (i.e., can only
                       improve the performance/reduce complexity of the SDFG).
        :return: A list of Transformation objects that match.
    """

    # Collapse multigraph into directed graph
    # Handling VF2 in networkx for now
    digraph = collapse_multigraph_to_nx(sdfg)

    for idx, expression in enumerate(pattern.expressions()):
        cexpr = collapse_multigraph_to_nx(expression)
        graph_matcher = iso.DiGraphMatcher(digraph,
                                           cexpr,
                                           node_match=node_match,
                                           edge_match=edge_match)
        for subgraph in graph_matcher.subgraph_isomorphisms_iter():
            subgraph = {
                cexpr.nodes[j]['node']: sdfg.node_id(digraph.nodes[i]['node'])
                for (i, j) in subgraph.items()
            }
            try:
                match_found = pattern.can_be_applied(sdfg, subgraph, idx, sdfg,
                                                     strict)
            except Exception as e:
                print('WARNING: {p}::can_be_applied triggered a {c} exception:'
                      ' {e}'.format(p=pattern.__name__,
                                    c=e.__class__.__name__,
                                    e=e))
                match_found = False
            if match_found:
                yield pattern(sdfg.sdfg_id, -1, subgraph, idx)

    # Recursive call for nested SDFGs
    for state in sdfg.nodes():
        for node in state.nodes():
            if isinstance(node, nd.NestedSDFG):
                yield from match_stateflow_pattern(node.sdfg,
                                                   pattern,
                                                   strict=strict)
