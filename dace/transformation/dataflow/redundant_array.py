""" Contains classes that implement a redundant array removal transformation.
"""

import copy
from dace import data as dt, types, subsets, symbolic
from dace.memlet import Memlet
from dace.graph import nodes, nxutil
from dace.sdfg import SDFGState
from dace.transformation import pattern_matching as pm
from dace.properties import ShapeProperty


class RedundantArray(pm.Transformation):
    """ Implements the redundant array removal transformation, applied
        when a transient array is copied to and from (to another array),
        but never used anywhere else. """

    _in_array = nodes.AccessNode('_')
    _out_array = nodes.AccessNode('_')

    @staticmethod
    def expressions():
        return [
            nxutil.node_path_graph(RedundantArray._in_array,
                                   RedundantArray._out_array),
        ]

    @staticmethod
    def can_be_applied(graph, candidate, expr_index, sdfg, strict=False):
        in_array = graph.nodes()[candidate[RedundantArray._in_array]]
        out_array = graph.nodes()[candidate[RedundantArray._out_array]]

        # Ensure out degree is one (only one target, which is out_array)
        if graph.out_degree(in_array) != 1:
            return False

        # Make sure that the candidate is a transient variable
        if not in_array.desc(sdfg).transient:
            return False

        # Make sure that both arrays are using the same storage location
        if in_array.desc(sdfg).storage != out_array.desc(sdfg).storage:
            return False

        # Find occurrences in this and other states
        occurrences = []
        for state in sdfg.nodes():
            occurrences.extend([
                n for n in state.nodes() if isinstance(n, nodes.AccessNode)
                and n.desc(sdfg) == in_array.desc(sdfg)
            ])

        if len(occurrences) > 1:
            return False

        # Only apply if arrays are of same shape (no need to modify memlet subset)
        if (len(in_array.desc(sdfg).shape) != len(out_array.desc(sdfg).shape)
                or any(i != o for i, o in zip(
                    in_array.desc(sdfg).shape,
                    out_array.desc(sdfg).shape))):
            return False

        return True

    @staticmethod
    def match_to_str(graph, candidate):
        in_array = graph.nodes()[candidate[RedundantArray._in_array]]

        return 'Remove ' + str(in_array)

    def apply(self, sdfg):
        def gnode(nname):
            return graph.nodes()[self.subgraph[nname]]

        graph = sdfg.nodes()[self.state_id]
        in_array = gnode(RedundantArray._in_array)
        out_array = gnode(RedundantArray._out_array)

        for e in graph.in_edges(in_array):
            # Modify all incoming edges to point to out_array
            path = graph.memlet_path(e)
            for pe in path:
                if pe.data.data == in_array.data:
                    pe.data.data = out_array.data

            # Redirect edge to out_array
            graph.remove_edge(e)
            graph.add_edge(e.src, e.src_conn, out_array, e.dst_conn, e.data)

        # Finally, remove in_array node
        graph.remove_node(in_array)

    def modifies_graph(self):
        return True


pm.Transformation.register_pattern(RedundantArray)
