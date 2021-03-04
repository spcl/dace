# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" Contains classes that implement a redundant array removal transformation.
"""

import copy
import functools
import typing

from dace import data, registry, subsets, dtypes
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

    if not src_subset and not dst_subset:
        # NOTE: This should never happen
        raise NotImplementedError
    # NOTE: If any of the subsets is None, it means that we proceed in 
    # experimental mode. The base case here is that we just copy the other
    # subset. However, if we can locate the other array, we check the
    # dimensionality of the subset and we pop or pad indices/ranges accordingly.
    # In that case, we also set the subset to start from 0 in each dimension. 
    if not src_subset:
        if src_name:
            desc = arrays[src_name]
            if not isinstance(desc, data.View):
                src_subset = copy.deepcopy(dst_subset)
                padding = len(desc.shape) - len(src_subset)
                if padding != 0:
                    if padding > 0:
                        if isinstance(src_subset, subsets.Indices):
                            indices = [0] * padding + src_subset.indices
                            src_subset = subsets.Indices(indices)
                        elif isinstance(src_subset, subsets.Range):
                            ranges = [(0, 0, 1)] * padding + src_subset.ranges
                            src_subset = subsets.Range(ranges)
                    elif padding < 0:
                        if isinstance(src_subset, subsets.Indices):
                            indices = src_subset.indices[-padding:]
                            src_subset = subsets.Indices(indices)
                        elif isinstance(src_subset, subsets.Range):
                            ranges = src_subset.ranges[-padding:]
                            src_subset = subsets.Range(ranges)
                    src_subset.offset(src_subset, True)
    elif not dst_subset:
        if dst_name:
            desc = arrays[dst_name]
            if not isinstance(desc, data.View):
                dst_subset = copy.deepcopy(src_subset)
                padding = len(desc.shape) - len(dst_subset)
                if padding != 0:
                    if padding > 0:
                        if isinstance(dst_subset, subsets.Indices):
                            indices = [0] * padding + dst_subset.indices
                            dst_subset = subsets.Indices(indices)
                        elif isinstance(dst_subset, subsets.Range):
                            ranges = [(0, 0, 1)] * padding + dst_subset.ranges
                            dst_subset = subsets.Range(ranges)
                    elif padding < 0:
                        if isinstance(dst_subset, subsets.Indices):
                            indices = dst_subset.indices[-padding:]
                            dst_subset = subsets.Indices(indices)
                        elif isinstance(dst_subset, subsets.Range):
                            ranges = dst_subset.ranges[-padding:]
                            dst_subset = subsets.Range(ranges)
                    dst_subset.offset(dst_subset, True)

    return src_subset, dst_subset


##############################################################################


@registry.autoregister_params(singlestate=True, strict=True)
class RedundantArray(pm.Transformation):
    """ Implements the redundant array removal transformation, applied
        when a transient array is copied to and from (to another array),
        but never used anywhere else. """

    in_array = pm.PatternNode(nodes.AccessNode)
    out_array = pm.PatternNode(nodes.AccessNode)

    @staticmethod
    def expressions():
        return [
            sdutil.node_path_graph(RedundantArray.in_array,
                                   RedundantArray.out_array)
        ]

    @staticmethod
    def can_be_applied(graph, candidate, expr_index, sdfg, strict=False):
        in_array = graph.nodes()[candidate[RedundantArray.in_array]]
        out_array = graph.nodes()[candidate[RedundantArray.out_array]]

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
        if isinstance(in_desc, data.View):  # Two views connected to each other
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

        if strict:
            # In strict mode, make sure the memlet covers the removed array
            edge = graph.edges_between(in_array, out_array)[0]
            if any(m != a
                   for m, a in zip(edge.data.subset.size(), in_desc.shape)):
                return False

        return True

    @staticmethod
    def match_to_str(graph, candidate):
        in_array = graph.nodes()[candidate[RedundantArray.in_array]]

        return "Remove " + str(in_array)

    def apply(self, sdfg):
        def gnode(nname):
            return graph.nodes()[self.subgraph[nname]]

        graph = sdfg.nodes()[self.state_id]
        in_array = self.in_array(sdfg)
        out_array = self.out_array(sdfg)
        in_desc = sdfg.arrays[in_array.data]
        out_desc = sdfg.arrays[out_array.data]

        # 1. Get edge e1 and extract subsets for arrays A and B
        e1 = graph.edges_between(in_array, out_array)[0]
        a1_subset, b_subset = _validate_subsets(e1, sdfg.arrays)

        # If the memlet does not cover the removed array, create a view.
        if any(m != a for m, a in zip(a1_subset.size(), in_desc.shape)):
            sdfg.arrays[in_array.data] = data.View(
                in_desc.dtype, in_desc.shape, True, in_desc.allow_conflicts,
                out_desc.storage, out_desc.location, in_desc.strides,
                in_desc.offset, out_desc.may_alias,
                dtypes.AllocationLifetime.Scope, in_desc.alignment,
                in_desc.debuginfo, in_desc.total_size)
            return
        
        # 2. Iterate over the e2 edges and traverse the memlet tree
        for e2 in graph.in_edges(in_array):
            path = graph.memlet_tree(e2)
            for e3 in path:
                # 2-a. Extract subsets for array B and others
                other_subset, a3_subset = _validate_subsets(
                    e3, sdfg.arrays, dst_name=in_array.data)
                # 2-b. Modify memlet to match array B.
                e3.data.data = out_array.data
                a3_subset.offset(a1_subset, negative=True)
                if isinstance(b_subset, subsets.Indices):
                    e3.data.dst_subset = b_subset.new_offset(a3_subset, False)
                else:
                    e3.data.dst_subset = b_subset.compose(a3_subset)
                # NOTE: This fixes the following case:
                # Tasklet ----> A[subset] ----> ... -----> A
                # Tasklet is not data, so it doesn't have an other subset.
                if isinstance(e3.src, nodes.AccessNode):
                    e3.data.data = e3.src.data
                    e3.data.src_subset = other_subset
                else:
                    e3.data.src_subset = None
                    e3.data.subset = copy.deepcopy(e3.data.dst_subset)
                    e3.data.other_subset = None
                    
            # 2-c. Remove edge and add new one
            graph.remove_edge(e2)
            graph.add_edge(e2.src, e2.src_conn, out_array, e2.dst_conn, e2.data)

        # Finally, remove in_array node
        graph.remove_node(in_array)
        if in_array.data in sdfg.arrays:
            del sdfg.arrays[in_array.data]


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

        # 1. Get edge e1 and extract/validate subsets for arrays A and B
        e1 = graph.edges_between(in_array, out_array)[0]
        _, b1_subset = _validate_subsets(e1, sdfg.arrays)

        # In strict mode, make sure the memlet covers the removed array
        if strict:
            if any(m != a
                   for m, a in zip(b1_subset.size(), out_desc.shape)):
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
                        _validate_subsets(e3,
                                          sdfg.arrays,
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
                    e3.data.subset = a_subset.new_offset(b3_subset, False)
                else:
                    e3.data.subset = a_subset.compose(b3_subset)
                # NOTE: This fixes the following case:
                # A ----> A[subset] ----> ... -----> Tasklet
                # Tasklet is not data, so it doesn't have an other subset.
                if isinstance(e3.dst, nodes.AccessNode):
                    e3.data.other_subset = other_subset
                else:
                    e3.data.other_subset = None
            # 2-c. Remove edge and add new one
            graph.remove_edge(e2)
            graph.add_edge(in_array, e2.src_conn, e2.dst, e2.dst_conn, e2.data)

        # Finally, remove out_array node
        graph.remove_node(out_array)
        # TODO: Should the array be removed from the SDFG?
        # del sdfg.arrays[out_array]
        if Config.get_bool("debugprint"):
            RedundantSecondArray._arrays_removed += 1
