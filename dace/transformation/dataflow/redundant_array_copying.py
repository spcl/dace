""" Contains redundant array removal transformations. """

import copy
from dace import data as dt, types, subsets, symbolic
from dace.memlet import Memlet
from dace.graph import nodes, nxutil
from dace.sdfg import SDFGState
from dace.transformation import pattern_matching as pm
from dace.properties import ShapeProperty


class RedundantArrayCopying(pm.Transformation):
    """ Implements the redundant array removal transformation. Removes array B
        in pattern A -> B -> A.
    """

    _in_array = nodes.AccessNode('_')
    _med_array = nodes.AccessNode('_')
    _out_array = nodes.AccessNode('_')

    @staticmethod
    def expressions():
        return [
            nxutil.node_path_graph(RedundantArrayCopying._in_array,
                                   RedundantArrayCopying._med_array,
                                   RedundantArrayCopying._out_array),
        ]

    @staticmethod
    def can_be_applied(graph, candidate, expr_index, sdfg, strict=False):
        in_array = graph.nodes()[candidate[RedundantArrayCopying._in_array]]
        med_array = graph.nodes()[candidate[RedundantArrayCopying._med_array]]
        out_array = graph.nodes()[candidate[RedundantArrayCopying._out_array]]

        # Ensure out degree is one (only one target, which is out_array)
        if graph.out_degree(in_array) != 1:
            return False

        # Make sure that the candidate is a transient variable
        # if not in_array.desc.transient:
        #     return False

        # Make sure that both arrays are using the same storage location
        if in_array.desc(sdfg).storage != out_array.desc(sdfg).storage:
            return False

        # Find occurrences in this and other states
        # (This could be relaxed)
        # occurrences = []
        # for state in sdfg.nodes():
        #     occurrences.extend([
        #         n for n in state.nodes()
        #         if isinstance(n, nodes.AccessNode) and n.desc == med_array.desc
        #     ])

        # if len(occurrences) > 1:
        #     return False

        # Only apply if arrays are of same shape (no need to modify memlet subset)
        if (len(in_array.desc(sdfg).shape) != len(out_array.desc(sdfg).shape)
                or any(i != o for i, o in zip(
                    in_array.desc(sdfg).shape,
                    out_array.desc(sdfg).shape))):
            return False

        return True

    @staticmethod
    def match_to_str(graph, candidate):
        med_array = graph.nodes()[candidate[RedundantArrayCopying._med_array]]

        return 'Remove ' + str(med_array)

    def apply(self, sdfg):
        def gnode(nname):
            return graph.nodes()[self.subgraph[nname]]

        graph = sdfg.nodes()[self.state_id]
        in_array = gnode(RedundantArrayCopying._in_array)
        med_array = gnode(RedundantArrayCopying._med_array)
        out_array = gnode(RedundantArrayCopying._out_array)

        med_edges = len(graph.out_edges(med_array))
        med_out_edges = 0
        for med_e in graph.out_edges(med_array):
            if (isinstance(med_e.dst, nodes.AccessNode)
                    and med_e.dst.data == out_array.data):
                # Modify all outcoming edges to point to in_array
                for out_e in graph.out_edges(med_e.dst):
                    path = graph.memlet_path(out_e)
                    for pe in path:
                        if pe.data.data == out_array.data:
                            pe.data.data = in_array.data
                    # Redirect edge to in_array
                    graph.remove_edge(out_e)
                    graph.add_edge(in_array, out_e.src_conn, out_e.dst,
                                   out_e.dst_conn, out_e.data)
                # Remove out_array
                for e in graph.edges_between(med_e, med_e.dst):
                    graph.remove_edge(e)
                graph.remove_node(med_e.dst)
                med_out_edges += 1

        # Finally, med_array node
        if med_array.desc(sdfg).transient and med_edges == med_out_edges:
            for e in graph.edges_between(in_array, med_array):
                graph.remove_edge(e)
            graph.remove_node(med_array)

    def modifies_graph(self):
        return True


pm.Transformation.register_pattern(RedundantArrayCopying)


class RedundantArrayCopying2(pm.Transformation):
    """ Implements the redundant array removal transformation. Removes 
        multiples of array B in pattern A -> B.
    """

    _in_array = nodes.AccessNode('_')
    _out_array = nodes.AccessNode('_')

    @staticmethod
    def expressions():
        return [
            nxutil.node_path_graph(RedundantArrayCopying2._in_array,
                                   RedundantArrayCopying2._out_array),
        ]

    @staticmethod
    def can_be_applied(graph, candidate, expr_index, sdfg, strict=False):
        in_array = graph.nodes()[candidate[RedundantArrayCopying2._in_array]]
        out_array = graph.nodes()[candidate[RedundantArrayCopying2._out_array]]

        # Ensure out degree is one (only one target, which is out_array)
        found = 0
        for _, _, dst, _, _ in graph.out_edges(in_array):
            if (isinstance(dst, nodes.AccessNode) and dst != out_array
                    and dst.data == out_array.data):
                found += 1

        return found > 0

    @staticmethod
    def match_to_str(graph, candidate):
        out_array = graph.nodes()[candidate[RedundantArrayCopying2._out_array]]

        return 'Remove ' + str(out_array)

    def apply(self, sdfg):
        def gnode(nname):
            return graph.nodes()[self.subgraph[nname]]

        graph = sdfg.nodes()[self.state_id]
        in_array = gnode(RedundantArrayCopying2._in_array)
        out_array = gnode(RedundantArrayCopying2._out_array)

        for e1 in graph.out_edges(in_array):
            dst = e1.dst
            if (isinstance(dst, nodes.AccessNode) and dst != out_array
                    and dst.data == out_array.data):
                for e2 in graph.out_edges(dst):
                    graph.add_edge(out_array, None, e2.dst, e2.dst_conn,
                                   e2.data)
                    graph.remove_edge(e2)
                graph.remove_edge(e1)
                graph.remove_node(dst)

    def modifies_graph(self):
        return True


pm.Transformation.register_pattern(RedundantArrayCopying2)


class RedundantArrayCopying3(pm.Transformation):
    """ Implements the redundant array removal transformation. Removes multiples
        of array B in pattern MapEntry -> B.
    """

    _map_entry = nodes.MapEntry(nodes.Map("", [], []))
    _out_array = nodes.AccessNode('_')

    @staticmethod
    def expressions():
        return [
            nxutil.node_path_graph(RedundantArrayCopying3._map_entry,
                                   RedundantArrayCopying3._out_array),
        ]

    @staticmethod
    def can_be_applied(graph, candidate, expr_index, sdfg, strict=False):
        map_entry = graph.nodes()[candidate[RedundantArrayCopying3._map_entry]]
        out_array = graph.nodes()[candidate[RedundantArrayCopying3._out_array]]

        # Ensure out degree is one (only one target, which is out_array)
        found = 0
        for _, _, dst, _, _ in graph.out_edges(map_entry):
            if (isinstance(dst, nodes.AccessNode) and dst != out_array
                    and dst.data == out_array.data):
                found += 1

        return found > 0

    @staticmethod
    def match_to_str(graph, candidate):
        out_array = graph.nodes()[candidate[RedundantArrayCopying3._out_array]]

        return 'Remove ' + str(out_array)

    def apply(self, sdfg):
        def gnode(nname):
            return graph.nodes()[self.subgraph[nname]]

        graph = sdfg.nodes()[self.state_id]
        map_entry = gnode(RedundantArrayCopying3._map_entry)
        out_array = gnode(RedundantArrayCopying3._out_array)

        for e1 in graph.out_edges(map_entry):
            dst = e1.dst
            if (isinstance(dst, nodes.AccessNode) and dst != out_array
                    and dst.data == out_array.data):
                for e2 in graph.out_edges(dst):
                    graph.add_edge(out_array, None, e2.dst, e2.dst_conn,
                                   e2.data)
                    graph.remove_edge(e2)
                graph.remove_edge(e1)
                graph.remove_node(dst)

    def modifies_graph(self):
        return True


pm.Transformation.register_pattern(RedundantArrayCopying3)
