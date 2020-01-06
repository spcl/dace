""" Contains classes that implement transformations relating to streams
    and transient nodes. """

import copy
import warnings
from abc import ABC

from dace import symbolic, subsets, sdfg as sd
from dace.properties import Property, make_properties
from dace.graph import nodes, nxutil
from dace.transformation import pattern_matching


@make_properties
class LocalStorage(pattern_matching.Transformation, ABC):
    """ Implements the Local Storage prototype transformation, which adds a
        transient data node between two nodes.
    """

    _node_a = nodes.Node()
    _node_b = nodes.Node()

    array = Property(
        dtype=str,
        desc="Array to create local storage for (if empty, first available)",
        default=None,
        allow_none=True)

    def __init__(self, sdfg_id, state_id, subgraph, expr_index):
        super().__init__(sdfg_id, state_id, subgraph, expr_index)
        self._local_name = None
        self._data_node = None

    @staticmethod
    def expressions():
        return [
            nxutil.node_path_graph(LocalStorage._node_a, LocalStorage._node_b)
        ]

    @staticmethod
    def match_to_str(graph, candidate):
        a = candidate[LocalStorage._node_a]
        b = candidate[LocalStorage._node_b]
        return '%s -> %s' % (a, b)

    def apply(self, sdfg):
        graph = sdfg.nodes()[self.state_id]
        node_a = graph.nodes()[self.subgraph[LocalStorage._node_a]]
        node_b = graph.nodes()[self.subgraph[LocalStorage._node_b]]

        # Determine direction of new memlet
        scope_dict = graph.scope_dict()
        propagate_forward = sd.scope_contains_scope(scope_dict, node_a, node_b)

        array = self.array
        if array is None or len(array) == 0:
            array = graph.edges_between(node_a, node_b)[0].data.data

        original_edge = None
        invariant_memlet = None
        for edge in graph.edges_between(node_a, node_b):
            if array == edge.data.data:
                original_edge = edge
                invariant_memlet = edge.data
                break
        if invariant_memlet is None:
            for edge in graph.edges_between(node_a, node_b):
                original_edge = edge
                invariant_memlet = edge.data
                warnings.warn('Array %s not found! Using array %s instead.' %
                              (array, invariant_memlet.data))
                array = invariant_memlet.data
                break
        if invariant_memlet is None:
            raise NameError('Array %s not found!' % array)

        # Add transient array
        new_data, _ = sdfg.add_array(
            'trans_' + invariant_memlet.data, [
                symbolic.overapproximate(r)
                for r in invariant_memlet.bounding_box_size()
            ],
            sdfg.arrays[invariant_memlet.data].dtype,
            transient=True,
            find_new_name=True)
        data_node = nodes.AccessNode(new_data)

        # Store as fields so that other transformations can use them
        self._local_name = new_data
        self._data_node = data_node

        to_data_mm = copy.deepcopy(invariant_memlet)
        from_data_mm = copy.deepcopy(invariant_memlet)
        offset = subsets.Indices([r[0] for r in invariant_memlet.subset])

        # Reconnect, assuming one edge to the access node
        graph.remove_edge(original_edge)
        if propagate_forward:
            graph.add_edge(node_a, original_edge.src_conn, data_node, None,
                           to_data_mm)
            new_edge = graph.add_edge(data_node, None, node_b,
                                      original_edge.dst_conn, from_data_mm)
        else:
            new_edge = graph.add_edge(node_a, original_edge.src_conn,
                                      data_node, None, to_data_mm)
            graph.add_edge(data_node, None, node_b, original_edge.dst_conn,
                           from_data_mm)

        # Offset all edges in the memlet tree (including the new edge)
        for edge in graph.memlet_tree(new_edge):
            edge.data.subset.offset(offset, True)
            edge.data.data = new_data


@make_properties
class InLocalStorage(LocalStorage):
    """ Implements the InLocalStorage transformation, which adds a transient
        data node between two scope entry nodes.
    """

    @staticmethod
    def can_be_applied(graph, candidate, expr_index, sdfg, strict=False):
        node_a = graph.nodes()[candidate[LocalStorage._node_a]]
        node_b = graph.nodes()[candidate[LocalStorage._node_b]]
        return (isinstance(node_a, nodes.EntryNode)
                and isinstance(node_b, nodes.EntryNode))


@make_properties
class OutLocalStorage(LocalStorage):
    """ Implements the OutLocalStorage transformation, which adds a transient
        data node between two scope exit nodes.
    """

    @staticmethod
    def can_be_applied(graph, candidate, expr_index, sdfg, strict=False):
        node_a = graph.nodes()[candidate[LocalStorage._node_a]]
        node_b = graph.nodes()[candidate[LocalStorage._node_b]]

        # WCR edges not supported (use AccumulateTransient instead
        if len(graph.edges_between(node_a, node_b)) == 1:
            edge = graph.edges_between(node_a, node_b)[0]
            if edge.data.wcr is not None:
                return False

        return (isinstance(node_a, nodes.ExitNode)
                and isinstance(node_b, nodes.ExitNode))


pattern_matching.Transformation.register_pattern(InLocalStorage)
pattern_matching.Transformation.register_pattern(OutLocalStorage)
