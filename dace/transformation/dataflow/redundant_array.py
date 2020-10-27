# Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
""" Contains classes that implement a redundant array removal transformation.
"""

import dace
import functools
import typing

from copy import deepcopy as dcpy
from dace import registry, subsets
from dace.sdfg import nodes
from dace.sdfg import utils as sdutil
from dace.sdfg import graph
from dace.transformation import transformation as pm
from dace.config import Config

DSize = typing.Union[int, dace.symbol]
DList = typing.Iterable[DSize]

# Helper methods #############################################################

# Direct-to-Data redundant copying pattern A --- e1 ---> B --- e2 ---> C

# Edge 1 (e1) cases:
# 1. array A + subset
# 2. array A + subset + other_subset (array B)
# 3. array B + subset
# 4. array B + subset + other_subset (array A)

# Edge 2 (e2) cases:
# 1. array B + subset
# 2. array B + subset + other_subset (array C)
# 3. array C + subset
# 4. array C + subset + other_subset (array C)

def _find_edge_case(e: graph.MultiConnectorEdge):
    """ Finds which case does the input edge falls into. """

    if e.data.data == e.src.data:
        # Case 1 or 2
        if e.data.other_subset:
            return 2
        else:
            return 1
    else:
        # Case 3 or 4
        if e.data.other_subset:
            return 4
        else:
            return 3


def _extract_subsets(
    e: graph.MultiConnectorEdge) -> typing.Tuple[subsets.Subset]:
    """ Extracts the src and dst subsets from the edge. """

    if e.data.data == e.src.data:
        return (e.data.subset, e.data.other_subset)
    else:
        return (e.data.other_subset, e.data.subset)


def _check_dtd_11(a_subset: subsets.Subset,
                  b_subset: subsets.Subset,
                  b_strides: DList) -> bool:
    """ Implements the matching condition of direct-to-data redundant copying
        for cases 1-1, 1-2, and 1-4.

        The direct-to-data redundant copying pattern is:
            A --- e1 ---> B --- e2 ---> C
        This method can be used to check the redundant-array transformation
        matching condition when the subsets of arrays A and B from edges e1 and
        e2 respectively are known.
    """

    # Count the number of elements to be copied (read) from array A.
    a_num_elements = a_subset.num_elements()
    # Find the 1D range of elements written from array A to array B.
    min_widx = 0
    max_widx = a_num_elements - 1
    # Find the 1D range of elements written from array B to array C.
    min_idx = [0] * b_subset.dims()
    max_idx = [s - 1 for s in b_subset.size_exact()]
    min_ridx = b_subset.at(min_idx, b_strides)
    max_ridx = b_subset.at(max_idx, b_strides)
    # Compare the ranges and return.
    return min_ridx >= min_widx and max_ridx <= max_widx


def _check_dtd_13(a_subset: subsets.Subset,
                  c_subset: subsets.Subset,
                  b_strides: DList) -> bool:
    """ Implements the matching condition of direct-to-data redundant copying
        for case 1-3.

        The direct-to-data redundant copying pattern is:
            A --- e1 ---> B --- e2 ---> C
        This method can be used to check the redundant-array transformation
        matching condition when the subsets of arrays A and C from edges e1 and
        e2 respectively are known, but not any of the subsets of array B.
    """

    # Count the number of elements to be copied (read) from array A.
    a_num_elements = a_subset.num_elements()
    # Find the 1D range of elements written from array A to array B.
    min_widx = 0
    max_widx = a_num_elements - 1
    # Count the number of elements to be written to array C.
    c_num_elements = c_subset.num_elements()
    # Find the 1D range of elements written from array B to array C.
    min_ridx = 0
    max_ridx = c_num_elements - 1
    # Compare the ranges and return.
    return min_ridx >= min_widx and max_ridx <= max_widx


def _check_dtd_21(b1_subset: subsets.Subset,
                  b2_subset: subsets.Subset) -> bool:
    """ Implements the matching condition of direct-to-data redundant copying
        for cases 2-1, 2-2, 2-4, 3-1, 3-2, 3-4, 4-1, 4-2, and 4-4.

        The direct-to-data redundant copying pattern is:
            A --- e1 ---> B --- e2 ---> C
        This method can be used to check the redundant-array transformation
        matching condition when the subsets of array B are known for both edges
        e1 and e2.
    """

    return b2_subset.covers(b1_subset)


def _check_dtd_23(b_subset: subsets.Subset,
                  c_subset: subsets.Subset,
                  b_strides: DList) -> bool:
    """ Implements the matching condition of direct-to-data redundant copying
        for case 2-3, 3-3, and 4-3.

        The direct-to-data redundant copying pattern is:
            A --- e1 ---> B --- e2 ---> C
        This method can be used to check the redundant-array transformation
        matching condition when the subsets of arrays B and C from edges e1 and
        e2 respectively are known.
    """

    # Find the 1D range of elements written from array A to array B.
    min_idx = [0] * b_subset.dims()
    max_idx = [s - 1 for s in b_subset.size_exact()]
    min_widx = b_subset.at(min_idx, b_strides)
    max_widx = b_subset.at(max_idx, b_strides)
    # Count the number of elements to be written to array C.
    c_num_elements = c_subset.num_elements()
    # Find the 1D range of elements written from array B to array C.
    min_ridx = 0
    max_ridx = c_num_elements - 1
    # Compare the ranges and return.
    return min_ridx >= min_widx and max_ridx <= max_widx


def _apply_dtd_11(e1: graph.MultiConnectorEdge,
                  e2: graph.MultiConnectorEdge) -> dace.Memlet:
    """ Creates the memlet needed to apply direct-to-data redundant copying
        remove for cases 1-1, 1-2, and 1-4.

        The direct-to-data redundant copying pattern is:
            A --- e1 ---> B --- e2 ---> C
        This method can be used to create the new memlet needed for the
        redundant-array transformation when the subsets of arrays A and B from
        edges e1 and e2 respectively are known.
    """

    new_array = e1.src.data
    if e2.src.data == e2.data.data:
        new_subset = e1.data.subset.compose(e2.data.subset)
        new_other_subset = e2.data.other_subset
    else:
        new_subset = e1.data.subset.compose(e2.data.other_subset)
        new_other_subset = e2.data.subset
    return dace.Memlet(data=new_array, subset=new_subset,
                       other_subset=new_other_subset, volume=e2.data.volume,
                       wcr=e2.data.wcr, wcr_nonatomic=e2.data.wcr_nonatomic)

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
        
        # Check whether the data copied from the first datanode cover
        # the subsets of all the output edges of the second datanode.

        # 1. Get the strides of array B
        b_strides = out_desc.strides

        # 2. Get the input edge (e1)
        e1 = graph.edges_between(in_array, out_array)[0]
        a_subset, b1_subset = _extract_subsets(e1)
        e1_case = _find_edge_case(e1)

        # 3. Iterate over the output edges (e2)
        for e2 in graph.out_edges(out_array):
            if isinstance(e2.dst, nodes.AccessNode):
                # Direct-to-Data redundant copying pattern
                b2_subset, c_subset = _extract_subsets(e2)
                e2_case = _find_edge_case(e2)
                case = (e1_case, e2_case)
                if case in {(1, 1), (1, 2), (1, 4)}:
                    if not _check_dtd_11(a_subset, b2_subset, b_strides):
                        return False
                elif case in {(1, 3)}:
                    if not _check_dtd_13(a_subset, c_subset, b_strides):
                        return False
                elif case in {(2, 3), (3, 3), (4, 3)}:
                    if not _check_dtd_23(b1_subset, c_subset, b_strides):
                        return False
                else:
                    if not _check_dtd_21(b1_subset, b2_subset):
                        return False
            else:
                raise NotImplementedError

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

        # 1. Get the input edge (e1)
        e1 = graph.edges_between(in_array, out_array)[0]
        e1_case = _find_edge_case(e1)

        # 2. Iterate over the output edges (e2)
        for e2 in graph.out_edges(out_array):
            # 2-a. Generate memlet for new edge
            if isinstance(e2.dst, nodes.AccessNode):
                # Direct-to-Data redundant copying pattern
                e2_case = _find_edge_case(e2)
                case = (e1_case, e2_case)
                if case in {(1, 1), (1, 2), (1, 4)}:
                    memlet = _apply_dtd_11(e1, e2)
                elif case in {(1, 3)}:
                    raise NotImplementedError
                elif case in {(2, 3), (3, 3), (4, 3)}:
                    raise NotImplementedError
                else:
                    raise NotImplementedError
            else:
                raise NotImplementedError

            # 2-b. Remove edge and add new one
            graph.remove_edge(e2)
            graph.add_edge(in_array, e2.src_conn, e2.dst, e2.dst_conn, memlet)

        # # # Extract the input (first) and output (second) array subsets.
        # # memlet = graph.edges_between(in_array, out_array)[0].data
        # # if memlet.data == in_array.data:
        # #     inp_subset = memlet.subset
        # #     out_subset = memlet.other_subset
        # # else:
        # #     inp_subset = memlet.other_subset
        # #     out_subset = memlet.subset
        
        # # if not inp_subset:
        # #     inp_subset = dcpy(out_subset)
        # #     inp_subset.offset(out_subset, negative=True)
        # # if not out_subset:
        # #     out_subset = dcpy(inp_subset)
        # #     out_subset.offset(inp_subset, negative=True)

        # # 1. Count the number of "input" elements and
        # # extract the "input" subset (if it exists).
        # # Also extract the subset of in_array.
        # memlet = graph.edges_between(in_array, out_array)[0].data
        # inp_num_elements = memlet.subset.num_elements()
        # if memlet.data == in_array.data:
        #     in_arr_subset = memlet.subset
        #     inp_subset = memlet.other_subset
        # else:
        #     in_arr_subset = memlet.other_subset
        #     inp_subset = memlet.subset

        # # 2. Modify all outgoing edges to point to in_array
        # for e in graph.out_edges(out_array):
        #     # 2a. Count the number of "output" elements and
        #     # extract the "output" subset (if it exists).
        #     out_num_elements = e.data.subset.num_elements()
        #     if e.data.data == out_array.data:
        #         # subset = e.data.subset
        #         out_subset = e.data.subset
        #     else:
        #         out_subset = e.data.other_subset
        #     # 2b. Walk the memlet path and update the memlets
        #     # We assume that any operations needed are possible due having
        #     # passed all the checks in can_be_applied.
        #     path = graph.memlet_tree(e)
        #     if len(path) > 1:
        #         assert in_arr_subset
        #         for pe in path:

        #             if pe.data.data == out_array.data:
        #                 # Example
        #                 # inp -- (0, a:b)/(c:c+b) --> out -- (c+d) --> other
        #                 # must become
        #                 # inp -- (0, a+d) --> other
        #                 subset = pe.data.subset
        #                 # (c+d) - (c:c+b) = (d)
        #                 subset.offset(in_arr_subset, negative=True)
        #                 # (0, a:b)(d) = (0, a+d) (or offset for indices)
        #                 if isinstance(in_arr_subset, subsets.Indices):
        #                     tmp = dcpy(in_arr_subset)
        #                     tmp.offset(subset, negative=False)
        #                     subset = tmp
        #                 else:
        #                     subset = inp_subset.compose(subset)
        #                 pe.data.subset = subset
        #             elif pe.data.other_subset:
        #                 # We do the same, but for other_subset
        #                 # We do not change the data
        #                 subset = pe.data.other_subset
        #                 subset.offset(out_subset, negative=True)
        #                 if isinstance(inp_subset, subsets.Indices):
        #                     tmp = dcpy(inp_subset)
        #                     tmp.offset(subset, negative=False)
        #                     subset = tmp
        #                 else:
        #                     subset = inp_subset.compose(subset)
        #                 pe.data.other_subset = subset
        #             else:
        #                 # The subset is the entirety of the out array
        #                 # Assuming that the input subset covers this,
        #                 # we do not need to do anything
        #                 pass
                    
        #     for pe in path:
        #         if pe.data.data == out_array.data:
        #             pe.data.data = in_array.data
        #             # Example
        #             # inp -- (0, a:b)/(c:c+b) --> out -- (c+d) --> other
        #             # must become
        #             # inp -- (0, a+d) --> other
        #             subset = pe.data.subset
        #             # (c+d) - (c:c+b) = (d)
        #             subset.offset(in_arr_subset, negative=True)
        #             # (0, a:b)(d) = (0, a+d) (or offset for indices)
        #             if isinstance(in_arr_subset, subsets.Indices):
        #                 tmp = dcpy(in_arr_subset)
        #                 tmp.offset(subset, negative=False)
        #                 subset = tmp
        #             else:
        #                 subset = inp_subset.compose(subset)
        #             pe.data.subset = subset
        #             # if isinstance(subset, subsets.Indices):
        #             #     pe.data.subset.offset(subset, False)
        #             # else:
        #             #     pe.data.subset = subset.compose(pe.data.subset)
        #         elif pe.data.other_subset:
        #             # We do the same, but for other_subset
        #             # We do not change the data
        #             subset = pe.data.other_subset
        #             subset.offset(out_subset, negative=True)
        #             if isinstance(inp_subset, subsets.Indices):
        #                 tmp = dcpy(inp_subset)
        #                 tmp.offset(subset, negative=False)
        #                 subset = tmp
        #             else:
        #                 subset = inp_subset.compose(subset)
        #             pe.data.other_subset = subset
        #         else:
        #             # The subset is the entirety of the out array
        #             # Assuming that the input subset covers this,
        #             # we do not need to do anything
        #             pass

        #     # Redirect edge to out_array
        #     graph.remove_edge(e)
        #     graph.add_edge(in_array, e.src_conn, e.dst, e.dst_conn, e.data)

        # Finally, remove out_array node
        graph.remove_node(out_array)
        # TODO: Should the array be removed from the SDFG?
        # del sdfg.arrays[out_array]
        if Config.get_bool("debugprint"):
            RedundantSecondArray._arrays_removed += 1
