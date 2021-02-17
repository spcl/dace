# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" Contains classes that implement a redundant array removal transformation.
"""

import copy
import functools
import typing

from dace import data, registry, subsets
from dace.sdfg import nodes
from dace.sdfg import utils as sdutil
from dace.sdfg import graph
from dace.transformation import transformation as pm
from dace.config import Config


# Helper methods #############################################################

def _validate_subsets(edge: graph.MultiConnectorEdge,
                      arrays: typing.Dict[str, data.Data],
                      src_name: str = None,
                      dst_name: str = None) -> typing.Tuple[subsets.Subset]:
    """ Extracts and validates src and dst subsets from the edge. """

    # Find src and dst names
    if not src_name and isinstance(edge.src, nodes.AccessNode):
        src_name = edge.src.data
    if not dst_name and isinstance(edge.dst, nodes.AccessNode):
        dst_name = edge.dst.data
    if not src_name and not dst_name:
        raise NotImplementedError

    # Find the src and dst subsets (deep-copy to allow manipulation)
    src_subset = copy.deepcopy(edge.data.src_subset)
    dst_subset = copy.deepcopy(edge.data.dst_subset)

    # Infer missing subsets
    if not src_subset and src_name:
        src_desc = arrays[src_name]
        dst_subset_size = dst_subset.size_exact()
        # If the number of dimensions doesn't match, try squeezing
        if len(dst_subset_size) > len(src_desc.shape):
            tmp = copy.deepcopy(dst_subset)
            tmp.squeeze()
            dst_subset_size = tmp.size_exact()
        # If the number of dimensions still doesn't match, fail to apply
        if len(dst_subset_size) != len(src_desc.shape):
            raise NotImplementedError
        # If the dimension sizes don't match, fail to apply
        for a, b in zip(src_desc.shape, dst_subset_size):
            if a != b:
                raise NotImplementedError
        src_subset = subsets.Range.from_array(src_desc)
    if not dst_subset and dst_name:
        dst_desc = arrays[dst_name]
        src_subset_size = src_subset.size_exact()
        # If the number of dimensions doesn't match, try squeezing
        if len(src_subset_size) > len(dst_desc.shape):
            tmp = copy.deepcopy(src_subset)
            tmp.squeeze()
            src_subset_size = tmp.size_exact()
        # If the number of dimensions still doesn't match, fail to apply
        if len(src_subset_size) != len(dst_desc.shape):
            raise NotImplementedError
        # If the dimension sizes don't match, fail to apply
        for a, b in zip(dst_desc.shape, src_subset_size):
            if a != b:
                raise NotImplementedError
        dst_subset = subsets.Range.from_array(dst_desc)

    return src_subset, dst_subset

##############################################################################


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
        for isedge in sdfg.edges():
            if in_array.data in isedge.data.free_symbols:
                occurrences.append(isedge)

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

        # Dimensionality must be the same in strict mode
        if strict and len(in_desc.shape) != len(out_desc.shape):
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
        for isedge in sdfg.edges():
            if out_array.data in isedge.data.free_symbols:
                occurrences.append(isedge)

        if len(occurrences) > 1:
            return False

        # Check whether the data copied from the first datanode cover
        # the subsets of all the output edges of the second datanode.
        # We assume the following pattern: A -- e1 --> B -- e2 --> others

        # 1. Get edge e1 and extract/validate subsets for arrays A and B
        e1 = graph.edges_between(in_array, out_array)[0]
        try:
            _, b1_subset = _validate_subsets(e1, sdfg.arrays)
        except NotImplementedError:
            return False
        # 2. Iterate over the e2 edges
        for e2 in graph.out_edges(out_array):
            # 2-a. Extract/validate subsets for array B and others
            try:
                b2_subset, _ = _validate_subsets(e2, sdfg.arrays)
            except NotImplementedError:
                return False
            # 2-b. Check where b1_subset covers b2_subset
            if not b1_subset.covers(b2_subset):
                return False
            # 2-c. Validate subsets in memlet tree
            # (should not be needed for valid SDGs)
            path = graph.memlet_tree(e2)
            for e3 in path:
                if e3 is not e2:
                    try:
                        _validate_subsets(e3, sdfg.arrays,
                                          src_name=out_array.data)
                    except NotImplementedError:
                        return False

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

        # We assume the following pattern: A -- e1 --> B -- e2 --> others

        # 1. Get edge e1 and extract subsets for arrays A and B
        e1 = graph.edges_between(in_array, out_array)[0]
        a_subset, b1_subset = _validate_subsets(e1, sdfg.arrays)
        # 2. Iterate over the e2 edges and traverse the memlet tree
        for e2 in graph.out_edges(out_array):
            path = graph.memlet_tree(e2)
            for e3 in path:
                # 2-a. Extract subsets for array B and others
                b3_subset, other_subset = _validate_subsets(
                    e3, sdfg.arrays, src_name=out_array.data)
                # 2-b. Modify memlet to match array A. Example:
                # A -- (0, a:b)/(c:c+b) --> B -- (c+d)/None --> others
                # A -- (0, a+d)/None --> others
                e3.data.data = in_array.data
                # (c+d) - (c:c+b) = (d)
                b3_subset.offset(b1_subset, negative=True)
                # (0, a:b)(d) = (0, a+d) (or offset for indices)
                if isinstance(a_subset, subsets.Indices):
                    tmp = copy.deepcopy(a_subset)
                    tmp.offset(b3_subset, negative=False)
                    e3.data.subset = tmp
                else:
                    e3.data.subset = a_subset.compose(b3_subset)
                e3.data.other_subset = other_subset
            # 2-c. Remove edge and add new one
            graph.remove_edge(e2)
            graph.add_edge(in_array, e2.src_conn, e2.dst, e2.dst_conn, e2.data)

        # Finally, remove out_array node
        graph.remove_node(out_array)
        # TODO: Should the array be removed from the SDFG?
        # del sdfg.arrays[out_array]
        if Config.get_bool("debugprint"):
            RedundantSecondArray._arrays_removed += 1
