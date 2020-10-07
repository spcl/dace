# Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
""" Contains classes that implement a redundant array removal transformation.
"""

from dace import registry, subsets
from dace.sdfg import nodes
from dace.sdfg import utils as sdutil
from dace.transformation import transformation as pm
from dace.config import Config


@registry.autoregister_params(singlestate=True, strict=True)
class RedundantArray(pm.Transformation):
    """ Implements the redundant array removal transformation, applied
        when a transient array is copied to and from (to another array),
        but never used anywhere else. """

    _arrays_removed = 0
    _in_array = nodes.AccessNode("_")
    _out_array = nodes.AccessNode("_")

    @staticmethod
    def expressions():
        return [
            sdutil.node_path_graph(RedundantArray._in_array,
                                   RedundantArray._out_array)
        ]

    @staticmethod
    def can_be_applied(graph, candidate, expr_index, sdfg, strict=False):
        in_array = graph.nodes()[candidate[RedundantArray._in_array]]
        out_array = graph.nodes()[candidate[RedundantArray._out_array]]

        in_desc = in_array.desc(sdfg)
        out_desc = out_array.desc(sdfg)

        # Ensure out degree is one (only one target, which is out_array)
        if graph.out_degree(in_array) != 1:
            return False

        # Make sure that the candidate is a transient variable
        if not in_desc.transient:
            return False

        # Make sure that both arrays are using the same storage location
        # and are of the same type (e.g., Stream->Stream)
        if in_desc.storage != out_desc.storage:
            return False
        if type(in_desc) != type(out_desc):
            return False

        # Find occurrences in this and other states
        occurrences = []
        for state in sdfg.nodes():
            occurrences.extend([
                n for n in state.nodes()
                if isinstance(n, nodes.AccessNode) and n.desc(sdfg) == in_desc
            ])

        if len(occurrences) > 1:
            return False

        # Only apply if arrays are of same shape (no need to modify subset)
        if len(in_desc.shape) != len(out_desc.shape) or any(
                i != o for i, o in zip(in_desc.shape, out_desc.shape)):
            return False

        if strict:
            # In strict mode, make sure the memlet covers the removed array
            edge = graph.edges_between(in_array, out_array)[0]
            if any(m != a
                   for m, a in zip(edge.data.subset.size(), in_desc.shape)):
                return False

        return True

    @staticmethod
    def match_to_str(graph, candidate):
        in_array = graph.nodes()[candidate[RedundantArray._in_array]]

        return "Remove " + str(in_array)

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
        # TODO: Should the array be removed from the SDFG?
        # del sdfg.arrays[in_array]
        if Config.get_bool("debugprint"):
            RedundantArray._arrays_removed += 1


@registry.autoregister_params(singlestate=True, strict=True)
class RedundantSecondArray(pm.Transformation):
    """ Implements the redundant array removal transformation, applied
        when a transient array is copied from and to (from another array),
        but never used anywhere else. This transformation removes the second
        array. """

    _arrays_removed = 0
    _in_array = nodes.AccessNode("_")
    _out_array = nodes.AccessNode("_")

    @staticmethod
    def expressions():
        return [
            sdutil.node_path_graph(RedundantSecondArray._in_array,
                                   RedundantSecondArray._out_array)
        ]

    @staticmethod
    def can_be_applied(graph, candidate, expr_index, sdfg, strict=False):
        in_array = graph.nodes()[candidate[RedundantSecondArray._in_array]]
        out_array = graph.nodes()[candidate[RedundantSecondArray._out_array]]

        in_desc = in_array.desc(sdfg)
        out_desc = out_array.desc(sdfg)

        # Ensure in degree is one (only one source, which is in_array)
        if graph.in_degree(out_array) != 1:
            return False

        # Make sure that the candidate is a transient variable
        if not out_desc.transient:
            return False

        # Make sure that both arrays are using the same storage location
        # and are of the same type (e.g., Stream->Stream)
        if in_desc.storage != out_desc.storage:
            return False
        if type(in_desc) != type(out_desc):
            return False

        # Find occurrences in this and other states
        occurrences = []
        for state in sdfg.nodes():
            occurrences.extend([
                n for n in state.nodes()
                if isinstance(n, nodes.AccessNode) and n.desc(sdfg) == out_desc
            ])

        if len(occurrences) > 1:
            return False

        # Only apply if arrays are of same shape (no need to modify memlet subset)
        # if len(in_desc.shape) != len(
        #         out_desc.shape) or any(i != o for i, o in zip(
        #             in_desc.shape,
        #             out_desc.shape)):
        #     return False

        return True

    @staticmethod
    def match_to_str(graph, candidate):
        out_array = graph.nodes()[candidate[RedundantSecondArray._out_array]]

        return "Remove " + str(out_array)

    def apply(self, sdfg):
        def gnode(nname):
            return graph.nodes()[self.subgraph[nname]]

        graph = sdfg.nodes()[self.state_id]
        in_array = gnode(RedundantSecondArray._in_array)
        out_array = gnode(RedundantSecondArray._out_array)
        memlet = graph.edges_between(in_array, out_array)[0].data
        if memlet.data == in_array.data:
            subset = memlet.subset
        else:
            subset = memlet.other_subset

        for e in graph.out_edges(out_array):
            # Modify all outgoing edges to point to in_array
            path = graph.memlet_tree(e)
            for pe in path:
                if pe.data.data == out_array.data:
                    pe.data.data = in_array.data
                    if isinstance(subset, subsets.Indices):
                        pe.data.subset.offset(subset, False)
                    else:
                        pe.data.subset = subset.compose(pe.data.subset)
                elif pe.data.other_subset:
                    if isinstance(subset, subsets.Indices):
                        pe.data.other_subset.offset(subset, False)
                    else:
                        pe.data.other_subset = subset.compose(
                            pe.data.other_subset)

            # Redirect edge to out_array
            graph.remove_edge(e)
            graph.add_edge(in_array, e.src_conn, e.dst, e.dst_conn, e.data)

        # Finally, remove out_array node
        graph.remove_node(out_array)
        # TODO: Should the array be removed from the SDFG?
        # del sdfg.arrays[out_array]
        if Config.get_bool("debugprint"):
            RedundantSecondArray._arrays_removed += 1
