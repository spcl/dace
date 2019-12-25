""" Contains classes that implement a redundant array removal transformation.
"""

import copy
from dace import data as dt, dtypes, subsets, symbolic
from dace.memlet import Memlet
from dace.graph import nodes, nxutil
from dace.sdfg import SDFGState
from dace.transformation import pattern_matching as pm
from dace.properties import ShapeProperty
from dace.config import Config


class TensorflowRedundantArray(pm.Transformation):
    """ Implements the redundant array removal transformation, applied
        to remove ReadVariableOps and control dependencies. """

    _arrays_removed = 0
    _in_array = nodes.AccessNode("_")
    _out_array = nodes.AccessNode("_")

    @staticmethod
    def expressions():
        return [
            nxutil.node_path_graph(TensorflowRedundantArray._in_array,
                                   TensorflowRedundantArray._out_array)
        ]

    @staticmethod
    def can_be_applied(graph, candidate, expr_index, sdfg, strict=False):
        in_array = graph.nodes()[candidate[TensorflowRedundantArray._in_array]]
        out_array = graph.nodes()[candidate[
            TensorflowRedundantArray._out_array]]

        # Just to be sure, check for the OP name in the out array
        if not ("ReadVariable" in out_array.data
                or "control_dependency" in out_array.data):
            return False

        # Make sure that the candidate is a transient variable
        if not in_array.desc(sdfg).transient:
            return False

        # Make sure that both arrays are using the same storage location
        if in_array.desc(sdfg).storage != out_array.desc(sdfg).storage:
            return False

        # Only apply if arrays are of same shape (no need to modify memlet subset)
        if len(in_array.desc(sdfg).shape) != len(
                out_array.desc(sdfg).shape) or any(i != o for i, o in zip(
                    in_array.desc(sdfg).shape,
                    out_array.desc(sdfg).shape)):
            return False

        return True

    @staticmethod
    def match_to_str(graph, candidate):
        out_array = graph.nodes()[candidate[
            TensorflowRedundantArray._out_array]]

        return "Remove " + str(out_array)

    def apply(self, sdfg):
        def gnode(nname):
            return graph.nodes()[self.subgraph[nname]]

        graph = sdfg.nodes()[self.state_id]
        in_array = gnode(TensorflowRedundantArray._in_array)
        out_array = gnode(TensorflowRedundantArray._out_array)

        for e in graph.out_edges(out_array):
            # Modify all outgoing edges to point to in_array
            path = graph.memlet_tree(e)
            for pe in path:
                if pe.data.data == out_array.data:
                    pe.data.data = in_array.data

            # Pre-emptively add edge from in_array to out_array's adjacent nodes.
            new_memlet = e.data
            new_memlet.data = in_array.data
            graph.add_edge(in_array, e.src_conn, e.dst, e.dst_conn, new_memlet)
            graph.remove_edge(e)

        try:
            assert len(graph.in_edges(out_array)) == 1
        except (AssertionError):
            print("Multiple in-edges for ", str(out_array))
        e = graph.in_edges(out_array)[0]
        graph.remove_edge(e)

        # Finally, remove out_array node
        #print("Removed ", str(out_array.data))
        graph.remove_node(out_array)
        if Config.get_bool("debugprint"):
            TensorflowRedundantArray._arrays_removed += 1

    def modifies_graph(self):
        return True

    @staticmethod
    def print_debuginfo():
        print(
            "Automatically removed {} tensorflow redundant arrays using TensorflowRedundantArray transform."
            .format(TensorflowRedundantArray._arrays_removed))


pm.Transformation.register_pattern(TensorflowRedundantArray)
