# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" Contains functions related to pattern matching in transformations. """

from dace.config import Config
from dace.sdfg import SDFG, SDFGState
from dace.sdfg import graph as gr, nodes as nd
import networkx as nx
from networkx.algorithms import isomorphism as iso
from typing import (Any, Callable, Dict, Iterator, List, Optional, Tuple, Type, Union)
from dace.transformation import transformation as xf


def collapse_multigraph_to_nx(graph: Union[gr.MultiDiGraph, gr.OrderedMultiDiGraph]) -> nx.DiGraph:
    """ Collapses a directed multigraph into a networkx directed graph.

        In the output directed graph, each node is a number, which contains
        itself as node_data['node'], while each edge contains a list of the
        data from the original edges as its attribute (edge_data[0...N]).

        :param graph: Directed multigraph object to be collapsed.
        :return: Collapsed directed graph object.
  """

    # Create the digraph nodes.
    digraph_nodes: List[Tuple[int, Dict[str, nd.Node]]] = ([None] * graph.number_of_nodes())
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


def type_match(graph_node, pattern_node):
    """ Checks whether the node types of the inputs match.
        :param graph_node: First node (in matched graph).
        :param pattern_node: Second node (in pattern subgraph).
        :return: True if the object types of the nodes match, False otherwise.
        :raise TypeError: When at least one of the inputs is not a dictionary
                          or does not have a 'node' attribute.
        :raise KeyError: When at least one of the inputs is a dictionary,
                         but does not have a 'node' key.
    """
    if isinstance(pattern_node['node'], xf.PatternNode):
        return isinstance(graph_node['node'], pattern_node['node'].node)
    return isinstance(graph_node['node'], type(pattern_node['node']))


def type_or_class_match(node_a, node_b):
    """
    Checks whether `node_a` is an instance of the same type as `node_b`, or
    if either `node_a`/`node_b` is a type and the other is an instance of that
    type. This is used in subgraph matching to allow the subgraph pattern to
    be either a graph of instantiated nodes, or node types.

    :param node_a: First node.
    :param node_b: Second node.
    :return: True if the object types of the nodes match according to the
             description, False otherwise.
    :raise TypeError: When at least one of the inputs is not a dictionary
                        or does not have a 'node' attribute.
    :raise KeyError: When at least one of the inputs is a dictionary,
                        but does not have a 'node' key.
    :see: enumerate_matches
    """
    if isinstance(node_b['node'], type):
        return issubclass(type(node_a['node']), node_b['node'])
    elif isinstance(node_a['node'], type):
        return issubclass(type(node_b['node']), node_a['node'])
    elif isinstance(node_b['node'], xf.PatternNode):
        return isinstance(node_a['node'], node_b['node'].node)
    elif isinstance(node_a['node'], xf.PatternNode):
        return isinstance(node_b['node'], node_a['node'].node)
    return isinstance(node_a['node'], type(node_b['node']))


def _try_to_match_transformation(graph: Union[SDFG, SDFGState], collapsed_graph: nx.DiGraph, subgraph: Dict[int, int],
                                 sdfg: SDFG, xform: Type[xf.PatternTransformation], expr_idx: int,
                                 nxpattern: nx.DiGraph, state_id: int, permissive: bool,
                                 options: Dict[str, Any]) -> Optional[xf.PatternTransformation]:
    """ 
    Helper function that tries to instantiate a pattern match into a 
    transformation object. 
    """
    subgraph = {
        nxpattern.nodes[j]['node']: graph.node_id(collapsed_graph.nodes[i]['node'])
        for i, j in subgraph.items()
    }

    try:
        match = xform(sdfg.sdfg_id, state_id, subgraph, expr_idx, options=options)
        match_found = match.can_be_applied(graph, subgraph, expr_idx, sdfg, permissive=permissive)
    except Exception as e:
        if Config.get_bool('optimizer', 'match_exception'):
            raise
        print('WARNING: {p}::can_be_applied triggered a {c} exception:'
              ' {e}'.format(p=xform.__name__, c=e.__class__.__name__, e=e))
        return None

    if match_found:
        return match

    return None


TransformationData = List[Tuple[Type[xf.PatternTransformation], int, nx.DiGraph, Callable, Dict[str, Any]]]
PatternMetadataType = Tuple[TransformationData, TransformationData]


def get_transformation_metadata(patterns: List[Type[xf.PatternTransformation]],
                                options: Optional[List[Dict[str, Any]]] = None) -> PatternMetadataType:
    """
    Collect all transformation expressions and metadata once, for use when
    applying transformations repeatedly.
    :param patterns: PatternTransformation type (or list thereof) to compute.
    :param options: An optional list of transformation parameter dictionaries.
    :return: A tuple of inter-state and single-state pattern matching
             transformations.
    """
    if options is None:
        options = [None] * len(patterns)

    singlestate_transformations: TransformationData = []
    interstate_transformations: TransformationData = []
    for pattern, opts in zip(patterns, options):
        # Find if the transformation is inter-state
        is_interstate = issubclass(pattern, xf.MultiStateTransformation)
        for i, expr in enumerate(pattern.expressions()):
            # Make a networkx-version of the match subgraph
            nxpattern = collapse_multigraph_to_nx(expr)
            if len(nxpattern.nodes) == 1:
                matcher = _node_matcher
            elif len(nxpattern.nodes) == 2 and len(nxpattern.edges) == 1:
                matcher = _edge_matcher
            else:
                matcher = _subgraph_isomorphism_matcher

            if is_interstate:
                interstate_transformations.append((pattern, i, nxpattern, matcher, opts))
            else:
                singlestate_transformations.append((pattern, i, nxpattern, matcher, opts))

    return interstate_transformations, singlestate_transformations


def _subgraph_isomorphism_matcher(digraph, nxpattern, node_pred, edge_pred):
    """ Match based on the VF2 algorithm for general SI. """
    graph_matcher = iso.DiGraphMatcher(digraph, nxpattern, node_match=node_pred, edge_match=edge_pred)
    yield from graph_matcher.subgraph_isomorphisms_iter()


def _node_matcher(digraph, nxpattern, node_pred, edge_pred):
    """ Match individual nodes. """
    pnid = next(iter(nxpattern))
    pnode = nxpattern.nodes[pnid]

    for nid in digraph:
        if node_pred(digraph.nodes[nid], pnode):
            yield {nid: pnid}


def _edge_matcher(digraph, nxpattern, node_pred, edge_pred):
    """ Match individual edges. """
    pedge = next(iter(nxpattern.edges))
    pu = nxpattern.nodes[pedge[0]]
    pv = nxpattern.nodes[pedge[1]]

    if edge_pred is None:
        for u, v in digraph.edges:
            if (node_pred(digraph.nodes[u], pu) and node_pred(digraph.nodes[v], pv)):
                yield {u: pedge[0], v: pedge[1]}
    else:
        for u, v in digraph.edges:
            if (node_pred(digraph.nodes[u], pu) and node_pred(digraph.nodes[v], pv)
                    and edge_pred(digraph.edges[u, v], nxpattern.edges[pedge])):
                yield {u: pedge[0], v: pedge[1]}


def match_patterns(sdfg: SDFG,
                   patterns: Union[Type[xf.PatternTransformation], List[Type[xf.PatternTransformation]]],
                   node_match: Callable[[Any, Any], bool] = type_match,
                   edge_match: Optional[Callable[[Any, Any], bool]] = None,
                   permissive: bool = False,
                   metadata: Optional[PatternMetadataType] = None,
                   states: Optional[List[SDFGState]] = None,
                   options: Optional[List[Dict[str, Any]]] = None):
    """ Returns a generator of Transformations that match the input SDFG. 
        Ordered by SDFG ID.
        :param sdfg: The SDFG to match in.
        :param patterns: PatternTransformation type (or list thereof) to match.
        :param node_match: Function for checking whether two nodes match.
        :param edge_match: Function for checking whether two edges match.
        :param permissive: Match transformations in permissive mode.
        :param metadata: Transformation metadata that can be reused.
        :param states: If given, only tries to match single-state 
                       transformations on this list.
        :param options: An optional iterable of transformation parameter
                        dictionaries.
        :return: A list of PatternTransformation objects that match.
    """

    if isinstance(patterns, type):
        patterns = [patterns]
    if isinstance(options, dict):
        options = [options]

    # Collect transformation metadata
    if metadata is not None:
        # Transformation metadata can be evaluated once per apply loop
        interstate_transformations, singlestate_transformations = metadata
    else:
        # Otherwise, precompute all transformation data once
        (interstate_transformations, singlestate_transformations) = get_transformation_metadata(patterns, options)

    # Collect SDFG and nested SDFGs
    sdfgs = sdfg.all_sdfgs_recursive()

    # Try to find transformations on each SDFG
    for tsdfg in sdfgs:
        ###################################
        # Match inter-state transformations
        if len(interstate_transformations) > 0:
            # Collapse multigraph into directed graph in order to use VF2
            digraph = collapse_multigraph_to_nx(tsdfg)

        for xform, expr_idx, nxpattern, matcher, opts in interstate_transformations:
            for subgraph in matcher(digraph, nxpattern, node_match, edge_match):
                match = _try_to_match_transformation(tsdfg, digraph, subgraph, tsdfg, xform, expr_idx, nxpattern, -1,
                                                     permissive, opts)
                if match is not None:
                    yield match

        ####################################
        # Match single-state transformations
        if len(singlestate_transformations) == 0:
            continue
        for state_id, state in enumerate(tsdfg.nodes()):
            if states is not None and state not in states:
                continue

            # Collapse multigraph into directed graph in order to use VF2
            digraph = collapse_multigraph_to_nx(state)

            for xform, expr_idx, nxpattern, matcher, opts in singlestate_transformations:
                for subgraph in matcher(digraph, nxpattern, node_match, edge_match):
                    match = _try_to_match_transformation(state, digraph, subgraph, tsdfg, xform, expr_idx, nxpattern,
                                                         state_id, permissive, opts)
                    if match is not None:
                        yield match


def enumerate_matches(sdfg: SDFG,
                      pattern: gr.Graph,
                      node_match=type_or_class_match,
                      edge_match=None) -> Iterator[gr.SubgraphView]:
    """
    Returns a generator of subgraphs that match the given subgraph pattern.
    :param sdfg: The SDFG to search in.
    :param pattern: A subgraph to look for.
    :param node_match: An optional function to use for matching nodes.
    :param node_match: An optional function to use for matching edges.
    :return: Yields SDFG subgraph view objects.
    """
    if len(pattern.nodes()) == 0:
        raise ValueError('Subgraph pattern cannot be empty')

    # Find if the subgraph is within states or SDFGs
    is_interstate = (isinstance(pattern.node(0), SDFGState)
                     or (isinstance(pattern.node(0), type) and pattern.node(0) is SDFGState))

    # Collapse multigraphs into directed graphs
    pattern_digraph = collapse_multigraph_to_nx(pattern)

    # Find matches in all SDFGs and nested SDFGs
    for graph in sdfg.all_sdfgs_recursive():
        if is_interstate:
            graph_matcher = iso.DiGraphMatcher(collapse_multigraph_to_nx(graph),
                                               pattern_digraph,
                                               node_match=node_match,
                                               edge_match=edge_match)
            for subgraph in graph_matcher.subgraph_isomorphisms_iter():
                yield gr.SubgraphView(graph, [graph.node(i) for i in subgraph.keys()])
        else:
            for state in graph.nodes():
                graph_matcher = iso.DiGraphMatcher(collapse_multigraph_to_nx(state),
                                                   pattern_digraph,
                                                   node_match=node_match,
                                                   edge_match=edge_match)
                for subgraph in graph_matcher.subgraph_isomorphisms_iter():
                    yield gr.SubgraphView(state, [state.node(i) for i in subgraph.keys()])
