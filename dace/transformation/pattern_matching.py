"""Contains classes and functions related to patterns/transformations.
"""

from __future__ import print_function
import bisect
import timeit
from types import GeneratorType
import dace
from dace import sdfg as sd
from dace.properties import make_properties, Property
from dace.graph import labeling, graph as gr
import networkx as nx
from networkx.algorithms import isomorphism as iso


@make_properties
class Transformation(object):
    """ Base class for transformations, as well as a static registry of 
        transformations, where new transformations can be added in a 
        decentralized manner.
    """

    ####################################################################
    # Transformation registry

    # Class attributes

    _patterns = set()
    _stateflow_patterns = set()

    # Static methods

    @staticmethod
    def patterns():
        """ Returns a list of single-state (dataflow) transformations 
            currently in the registry. """

        pattern_list = sorted(
            Transformation._patterns, key=lambda cls: cls.__name__)
        return pattern_list

    @staticmethod
    def stateflow_patterns():
        """ Returns a list of multiple-state (interstate) transformations 
            currently in the registry. """

        pattern_list = sorted(
            Transformation._stateflow_patterns, key=lambda cls: cls.__name__)
        return pattern_list

    @staticmethod
    def register_pattern(clazz):
        """ Registers a single-state (dataflow) transformation in the registry.
            @param clazz: The Transformation class type.
        """

        if not issubclass(clazz, Transformation):
            raise TypeError
        Transformation._patterns.add(clazz)

    @staticmethod
    def register_stateflow_pattern(clazz):
        """ Registers a multi-state transformation in the registry.
            @param clazz: The Transformation class type.
        """

        if not issubclass(clazz, Transformation):
            raise TypeError
        Transformation._stateflow_patterns.add(clazz)

    @staticmethod
    def register_pattern_file(filename):
        """ Registers all transformations in a single Python file. """

        pattern_members = {}
        with open(pattern_path) as pattern_file:
            exec(pattern_file.read(), pattern_members)
        for member in pattern_members.values():
            if inspect.isclass(member) and issubclass(member, Transformation):
                Transformation.register_pattern(member)

    @staticmethod
    def deregister_pattern(clazz):
        """ De-registers a transformation.
            @param clazz: The Transformation class type.
        """

        if not issubclass(clazz, Transformation):
            raise TypeError
        Transformation._patterns.remove(clazz)

    ####################################################################
    # Static and object methods

    # Properties
    sdfg_id = Property(dtype=int)
    state_id = Property(dtype=int)
    subgraph = Property(dtype=dict)
    expr_index = Property(dtype=int)

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
            @param graph: SDFGState object if this Transformation is 
                          single-state, or SDFG object otherwise.
            @param candidate: A mapping between node IDs returned from 
                              `Transformation.expressions` and the nodes in 
                              `graph`.
            @param expr_index: The list index from `Transformation.expressions`
                               that was matched.
            @param sdfg: If `graph` is an SDFGState, its parent SDFG. Otherwise
                         should be equal to `graph`.
            @return: True if the transformation can be applied.
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
            @param sdfg_id: A unique ID of the SDFG.
            @param state_id: The node ID of the SDFG state, if applicable.
            @param subgraph: A mapping between node IDs returned from 
                             `Transformation.expressions` and the nodes in 
                             `graph`.
            @param expr_index: The list index from `Transformation.expressions`
                               that was matched.
            @raise TypeError: When transformation is not subclass of
                              Transformation.
            @raise TypeError: When state_id is not instance of int.
            @raise TypeError: When subgraph is not a dict of 
                              dace.graph.nodes.Node : int.
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
            return type(self) < type(other)

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

        while self_id is not None and other_id is not None:
            if self_id != other_id:
                return self_id < other_id
            try:
                self_id = next(self_ids)
            except StopIteration:
                return True
            try:
                other_id = next(other_ids)
            except StopIteration:
                return False

    def apply_pattern(self, sdfg):
        """ Applies this transformation on the given SDFG. """
        self.apply(sdfg)
        if not self.annotates_memlets():
            labeling.propagate_labels_sdfg(sdfg)

    def __str__(self):
        raise NotImplementedError

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


# Module functions ############################################################


def collapse_multigraph_to_nx(graph: gr.MultiDiGraph) -> nx.DiGraph:
    """ Collapses a directed multigraph into a networkx directed graph.

        In the output directed graph, each node is a number, which contains 
        itself as node_data['node'], while each edge contains a list of the 
        data from the original edges as its attribute (edge_data[0...N]).

        @param graph: Directed multigraph object to be collapsed.
        @return: Collapsed directed graph object.
  """

    # Create the digraph nodes.
    digraph_nodes = [None] * graph.number_of_nodes()
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
        @param node_a: First node.
        @param node_b: Second node.
        @return: True if the object types of the nodes match, False otherwise.
        @raise TypeError: When at least one of the inputs is not a dictionary 
                          or does not have a 'node' attribute.
        @raise KeyError: When at least one of the inputs is a dictionary, 
                         but does not have a 'node' key.
    """
    return isinstance(node_a['node'], type(node_b['node']))


def match_expression(graph,
                     expressions,
                     node_match=type_match,
                     edge_match=None,
                     pattern_match=None,
                     strict=False):
    """ Returns a generator which yields a subgraph mapping from 
        `expression_node` to `graph_node`.
        @param graph: Directed multigraph object to be searched for subgraphs.
        @param expressions: List of directed graphs, isomorphic to any 
                            (sub)graph that potentially matches a 
                            transformation.
        @param node_match: Function for checking whether two nodes match.
        @param edge_match: Function for checking whether two edges match.
        @param pattern_match: Function for checking whether a subgraph matches
                              a transformation.
        @return: Generator of 2-tuples: (subgraph, expression index in 
                 `expressions`).
    """

    # Collapse multigraph into directed graph
    digraph = collapse_multigraph_to_nx(graph)

    # If expression is a list, try to match each one of them
    if not isinstance(expressions, list) and not isinstance(
            expressions, GeneratorType):
        expressions = [expressions]

    for expr_index, expr in enumerate(expressions):
        # Also collapse expression multigraph
        cexpr = collapse_multigraph_to_nx(expr)

        # Find candidate subgraphs (per-node / per-edge matching)
        graph_matcher = iso.DiGraphMatcher(
            digraph, cexpr, node_match=node_match, edge_match=edge_match)
        for subgraph in graph_matcher.subgraph_isomorphisms_iter():
            # Convert candidate to original graph node representation
            # The type of subgraph is {graph_node_id: subgraph_node_id}
            # We return the inverse mapping: {subgraph_node: graph_node} for
            # ease of access
            subgraph = {
                cexpr.node[j]['node']: digraph.node[i]['node']
                for (i, j) in subgraph.items()
            }

            # Match original (regular) expression on found candidate
            if pattern_match is None:
                # Yield mapping and index of expression found
                yield subgraph, expr_index
            else:
                match_found = pattern_match(graph, subgraph)
                if match_found:
                    # Yield mapping and index of expression found
                    # expr_index_list = list(range(match_num))
                    yield subgraph, expr_index  # expr_index_list


def match_pattern(state_id,
                  state,
                  pattern,
                  sdfg,
                  node_match=type_match,
                  edge_match=None,
                  strict=False):
    """ Returns a list of single-state Transformations of a certain class that
        match the input SDFG.
        @param state_id: The node ID of the state in the given SDFG.
        @param state: An SDFGState object to match.
        @param pattern: Transformation object to match.
        @param sdfg: The SDFG to match in.
        @param node_match: Function for checking whether two nodes match.
        @param edge_match: Function for checking whether two edges match.
        @param strict: Only match transformation if strict (i.e., can only
                       improve the performance/reduce complexity of the SDFG).
        @return: A list of Transformation objects that match.
    """

    # Collapse multigraph into directed graph
    # Handling VF2 in networkx for now
    digraph = collapse_multigraph_to_nx(state)

    matches = []

    for idx, expression in enumerate(pattern.expressions()):
        cexpr = collapse_multigraph_to_nx(expression)
        graph_matcher = iso.DiGraphMatcher(
            digraph, cexpr, node_match=node_match, edge_match=edge_match)
        for subgraph in graph_matcher.subgraph_isomorphisms_iter():
            subgraph = {
                cexpr.node[j]['node']: state.node_id(digraph.node[i]['node'])
                for (i, j) in subgraph.items()
            }
            match_found = pattern.can_be_applied(
                state, subgraph, idx, sdfg, strict=strict)
            if match_found:
                bisect.insort_left(
                    matches,
                    pattern(
                        sdfg.sdfg_list.index(sdfg), state_id, subgraph, idx))

    # Recursive call for nested SDFGs
    for node in state.nodes():
        if isinstance(node, dace.graph.nodes.NestedSDFG):
            sub_sdfg = node.sdfg
            for i, sub_state in enumerate(sub_sdfg.nodes()):
                matches += match_pattern(i, sub_state, pattern, sub_sdfg)

    return matches


def match_stateflow_pattern(sdfg,
                            pattern,
                            node_match=type_match,
                            edge_match=None,
                            strict=False):
    """ Returns a list of multi-state Transformations of a certain class that
        match the input SDFG.
        @param sdfg: The SDFG to match in.
        @param pattern: Transformation object to match.
        @param node_match: Function for checking whether two nodes match.
        @param edge_match: Function for checking whether two edges match.
        @param strict: Only match transformation if strict (i.e., can only
                       improve the performance/reduce complexity of the SDFG).
        @return: A list of Transformation objects that match.
    """

    # Collapse multigraph into directed graph
    # Handling VF2 in networkx for now
    digraph = collapse_multigraph_to_nx(sdfg)

    matches = []

    for idx, expression in enumerate(pattern.expressions()):
        cexpr = collapse_multigraph_to_nx(expression)
        graph_matcher = iso.DiGraphMatcher(
            digraph, cexpr, node_match=node_match, edge_match=edge_match)
        for subgraph in graph_matcher.subgraph_isomorphisms_iter():
            subgraph = {
                cexpr.node[j]['node']: sdfg.node_id(digraph.node[i]['node'])
                for (i, j) in subgraph.items()
            }
            match_found = pattern.can_be_applied(sdfg, subgraph, idx, sdfg,
                                                 strict)
            if match_found:
                bisect.insort_left(
                    matches,
                    pattern(sdfg.sdfg_list.index(sdfg), -1, subgraph, idx))
                # matches.append(
                #     pattern(pattern, state_id, subgraph, options))

    # Recursive call for nested SDFGs
    for state in sdfg.nodes():
        for node in state.nodes():
            if isinstance(node, dace.graph.nodes.NestedSDFG):
                matches += match_stateflow_pattern(node.sdfg, pattern)

    return matches
