"""Contains classes and functions related to patterns/transformations.
"""

from __future__ import print_function
import copy
import dace
import inspect
from typing import Dict
from dace.sdfg import SDFG, SDFGState
from dace.sdfg import utils as sdutil, propagation
from dace.properties import make_properties, Property, SubgraphProperty
from dace.registry import make_registry
from dace.sdfg import graph as gr, nodes as nd
from dace.dtypes import ScheduleType
import networkx as nx
from networkx.algorithms import isomorphism as iso
from typing import Dict, List, Tuple, Type, Union


@make_registry
@make_properties
class Transformation(object):
    """ Base class for transformations, as well as a static registry of
        transformations, where new transformations can be added in a
        decentralized manner.

        New transformations are registered with ``Transformation.register``
        (or ``dace.registry.autoregister_params``) with two optional boolean
        keyword arguments: ``singlestate`` (default: False) and ``strict``
        (default: False).
        If ``singlestate`` is True, the transformation operates on a single
        state; otherwise, it will be matched over an entire SDFG.
        If ``strict`` is True, this transformation will be considered strict
        (i.e., always important to perform) and will be performed automatically
        as part of SDFG strict transformations.
    """

    # Properties
    sdfg_id = Property(dtype=int, category="(Debug)")
    state_id = Property(dtype=int, category="(Debug)")
    subgraph = SubgraphProperty(dtype=dict, category="(Debug)")
    expr_index = Property(dtype=int, category="(Debug)")

    @staticmethod
    def annotates_memlets():
        """ Indicates whether the transformation annotates the edges it creates
            or modifies with the appropriate memlets. This determines
            whether to apply memlet propagation after the transformation.
        """

        return False

    @staticmethod
    def expressions():
        """ Returns a list of Graph objects that will be matched in the
            subgraph isomorphism phase. Used as a pre-pass before calling
            `can_be_applied`.
            @see Transformation.can_be_applied
        """

        raise NotImplementedError

    @staticmethod
    def can_be_applied(graph, candidate, expr_index, sdfg, strict=False):
        """ Returns True if this transformation can be applied on the candidate
            matched subgraph.
            :param graph: SDFGState object if this Transformation is
                          single-state, or SDFG object otherwise.
            :param candidate: A mapping between node IDs returned from
                              `Transformation.expressions` and the nodes in
                              `graph`.
            :param expr_index: The list index from `Transformation.expressions`
                               that was matched.
            :param sdfg: If `graph` is an SDFGState, its parent SDFG. Otherwise
                         should be equal to `graph`.
            :param strict: Whether transformation should run in strict mode.
            :return: True if the transformation can be applied.
        """
        raise NotImplementedError

    @staticmethod
    def match_to_str(graph, candidate):
        """ Returns a string representation of the pattern match on the
            candidate subgraph. Used when identifying matches in the console
            UI.
        """
        raise NotImplementedError

    def __init__(self, sdfg_id, state_id, subgraph, expr_index):
        """ Initializes an instance of Transformation.
            :param sdfg_id: A unique ID of the SDFG.
            :param state_id: The node ID of the SDFG state, if applicable.
            :param subgraph: A mapping between node IDs returned from
                             `Transformation.expressions` and the nodes in
                             `graph`.
            :param expr_index: The list index from `Transformation.expressions`
                               that was matched.
            :raise TypeError: When transformation is not subclass of
                              Transformation.
            :raise TypeError: When state_id is not instance of int.
            :raise TypeError: When subgraph is not a dict of
                              dace.sdfg.nodes.Node : int.
        """

        self.sdfg_id = sdfg_id
        self.state_id = state_id
        for value in subgraph.values():
            if not isinstance(value, int):
                raise TypeError('All values of '
                                'subgraph'
                                ' dictionary must be '
                                'instances of int.')
        self.subgraph = subgraph
        self.expr_index = expr_index

    def __lt__(self, other):
        """ Comparing two transformations by their class name and node IDs
            in match. Used for ordering transformations consistently.
        """
        if type(self) != type(other):
            return type(self).__name__ < type(other).__name__

        self_ids = iter(self.subgraph.values())
        other_ids = iter(self.subgraph.values())

        try:
            self_id = next(self_ids)
        except StopIteration:
            return True
        try:
            other_id = next(other_ids)
        except StopIteration:
            return False

        self_end = False

        while self_id is not None and other_id is not None:
            if self_id != other_id:
                return self_id < other_id
            try:
                self_id = next(self_ids)
            except StopIteration:
                self_end = True
            try:
                other_id = next(other_ids)
            except StopIteration:
                if self_end:  # Transformations are equal
                    return False
                return False
            if self_end:
                return True

    def apply_pattern(self, sdfg):
        """ Applies this transformation on the given SDFG. """
        self.apply(sdfg)
        if not self.annotates_memlets():
            propagation.propagate_memlets_sdfg(sdfg)

    def __str__(self):
        return type(self).__name__

    def modifies_graph(self):
        return True

    def print_match(self, sdfg):
        """ Returns a string representation of the pattern match on the
            given SDFG. Used for printing matches in the console UI.
        """
        if not isinstance(sdfg, dace.SDFG):
            raise TypeError("Expected SDFG, got: {}".format(
                type(sdfg).__name__))
        if self.state_id == -1:
            graph = sdfg
        else:
            graph = sdfg.nodes()[self.state_id]
        string = type(self).__name__ + ' in '
        string += type(self).match_to_str(graph, self.subgraph)
        return string


class ExpandTransformation(Transformation):
    """Base class for transformations that simply expand a node into a
       subgraph, and thus needs only simple matching and replacement
       functionality. Subclasses only need to implement the method
       "expansion".
    """
    @classmethod
    def expressions(clc):
        return [sdutil.node_path_graph(clc._match_node)]

    @staticmethod
    def can_be_applied(graph: dace.sdfg.graph.OrderedMultiDiConnectorGraph,
                       candidate: Dict[dace.sdfg.nodes.Node, int],
                       expr_index: int,
                       sdfg,
                       strict: bool = False):
        # All we need is the correct node
        return True

    @classmethod
    def match_to_str(clc, graph: dace.sdfg.graph.OrderedMultiDiConnectorGraph,
                     candidate: Dict[dace.sdfg.nodes.Node, int]):
        node = graph.nodes()[candidate[clc._match_node]]
        return str(node)

    @staticmethod
    def expansion(node):
        raise NotImplementedError("Must be implemented by subclass")

    @staticmethod
    def postprocessing(sdfg, state, expansion):
        pass

    def apply(self, sdfg, *args, **kwargs):
        state = sdfg.nodes()[self.state_id]
        node = state.nodes()[self.subgraph[type(self)._match_node]]
        expansion = type(self).expansion(node, state, sdfg, *args, **kwargs)
        if isinstance(expansion, dace.SDFG):
            # Modify internal schedules according to node schedule
            if node.schedule != ScheduleType.Default:
                for nstate in expansion.nodes():
                    topnodes = nstate.scope_dict(node_to_children=True)[None]
                    for topnode in topnodes:
                        if isinstance(topnode, (nd.EntryNode, nd.LibraryNode)):
                            topnode.schedule = node.schedule

            expansion = state.add_nested_sdfg(expansion,
                                              sdfg,
                                              node.in_connectors,
                                              node.out_connectors,
                                              name=node.name)
        elif isinstance(expansion, dace.sdfg.nodes.CodeNode):
            pass
        else:
            raise TypeError("Node expansion must be a CodeNode or an SDFG")
        expansion.environments = copy.copy(
            set(map(lambda a: a.__name__,
                    type(self).environments)))
        sdutil.change_edge_dest(state, node, expansion)
        sdutil.change_edge_src(state, node, expansion)
        state.remove_node(node)
        type(self).postprocessing(sdfg, state, expansion)


# Module functions ############################################################


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
