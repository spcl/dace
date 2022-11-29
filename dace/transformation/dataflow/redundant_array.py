# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" Contains classes that implement a redundant array removal transformation.
"""

import copy
import warnings
from typing import Dict, List, Optional, Tuple

import networkx as nx
from networkx.exception import NetworkXError, NodeNotFound

from dace import data, dtypes
from dace import memlet as mm
from dace import subsets, symbolic
from dace.config import Config
from dace.sdfg import SDFG, SDFGState, graph, nodes
from dace.sdfg import utils as sdutil
from dace.transformation import helpers
from dace.transformation import transformation as pm

# Helper methods #############################################################


def _validate_subsets(edge: graph.MultiConnectorEdge,
                      arrays: Dict[str, data.Data],
                      src_name: str = None,
                      dst_name: str = None) -> Tuple[subsets.Subset]:
    """ Extracts and validates src and dst subsets from the edge. """

    # Find src and dst names
    if not src_name and isinstance(edge.src, nodes.AccessNode):
        src_name = edge.src.data
    if not dst_name and isinstance(edge.dst, nodes.AccessNode):
        dst_name = edge.dst.data
    if not src_name and not dst_name:
        raise NotImplementedError('No source or destination name given')

    # Find the src and dst subsets (deep-copy to allow manipulation)
    src_subset = copy.deepcopy(edge.data.src_subset)
    dst_subset = copy.deepcopy(edge.data.dst_subset)

    if not src_subset and not dst_subset:
        # NOTE: This should never happen
        raise NotImplementedError('Neither source nor destination subsets are defined')
    # NOTE: If any of the subsets is None, it means that we proceed in
    # experimental mode. The base case here is that we just copy the other
    # subset. However, if we can locate the other array, we check the
    # dimensionality of the subset and we pop or pad indices/ranges accordingly.
    # In that case, we also set the subset to start from 0 in each dimension.
    if not src_subset:
        if src_name:
            desc = arrays[src_name]
            if isinstance(desc, data.View) or edge.data.data == dst_name:
                src_subset = subsets.Range.from_array(desc)
                src_expr = src_subset.num_elements()
                src_expr_exact = src_subset.num_elements_exact()
                dst_expr = dst_subset.num_elements()
                dst_expr_exact = dst_subset.num_elements_exact()
                if (src_expr != dst_expr and symbolic.inequal_symbols(src_expr_exact, dst_expr_exact)):
                    raise ValueError("Source subset is missing (dst_subset: {}, "
                                     "src_shape: {}".format(dst_subset, desc.shape))
            else:
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
            if isinstance(desc, data.View) or edge.data.data == src_name:
                dst_subset = subsets.Range.from_array(desc)
                src_expr = src_subset.num_elements()
                src_expr_exact = src_subset.num_elements_exact()
                dst_expr = dst_subset.num_elements()
                dst_expr_exact = dst_subset.num_elements_exact()
                if (src_expr != dst_expr and symbolic.inequal_symbols(src_expr_exact, dst_expr_exact)):
                    raise ValueError("Destination subset is missing (src_subset: {}, "
                                     "dst_shape: {}".format(src_subset, desc.shape))
            else:
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


def find_dims_to_pop(a_size, b_size):
    dims_to_pop = []
    for i, sz in enumerate(reversed(a_size)):
        if sz not in b_size:
            dims_to_pop.append(len(a_size) - 1 - i)
    return dims_to_pop


def pop_dims(subset, dims):
    popped = []
    if isinstance(subset, subsets.Indices):
        indices = copy.deepcopy(subsets.Indices)
        for i in dims:
            popped.append(indices.pop(i))
        return subsets.Indices(indices)
    else:
        ranges = copy.deepcopy(subset.ranges)
        tsizes = copy.deepcopy(subset.tile_sizes)
        for i in dims:
            r = ranges.pop(i)
            t = tsizes.pop(i)
            popped.append((r, t))
        new_subset = subsets.Range(ranges)
        new_subset.tile_sizes = tsizes
        return new_subset, popped


def compose_and_push_back(first, second, dims=None, popped=None):
    if isinstance(first, subsets.Indices):
        subset = first.new_offset(second, negative=False)
    else:
        subset = first.compose(second)
    if dims and popped and len(dims) == len(popped):
        if isinstance(first, subsets.Indices):
            indices = subset.Indices
            for d, p in zip(reversed(dims), reversed(popped)):
                indices.insert(d, p)
            subset = subsets.Indices(indices)
        else:
            ranges = subset.ranges
            tsizes = subset.tile_sizes
            for d, (r, t) in zip(reversed(dims), reversed(popped)):
                ranges.insert(d, r)
                tsizes.insert(d, t)
            subset = subsets.Range(ranges)
            subset.tile_sizes = tsizes
    return subset


##############################################################################


class RedundantArray(pm.SingleStateTransformation):
    """ Implements the redundant array removal transformation, applied
        when a transient array is copied to and from (to another array),
        but never used anywhere else. """

    in_array = pm.PatternNode(nodes.AccessNode)
    out_array = pm.PatternNode(nodes.AccessNode)

    @classmethod
    def expressions(cls):
        return [sdutil.node_path_graph(cls.in_array, cls.out_array)]

    def can_be_applied(self, graph: SDFGState, expr_index, sdfg, permissive=False):
        in_array = self.in_array
        out_array = self.out_array

        in_desc = in_array.desc(sdfg)
        out_desc = out_array.desc(sdfg)

        # Ensure out degree is one (only one target, which is out_array)
        if graph.out_degree(in_array) != 1:
            return False

        # Ensure it is not an isolated copy
        if graph.in_degree(in_array) == 0:
            return False

        # Make sure that the candidate is a transient variable
        if not in_desc.transient:
            return False

        # 1. Get edge e1 and extract subsets for arrays A and B
        e1 = graph.edges_between(in_array, out_array)[0]
        try:
            a1_subset, b_subset = _validate_subsets(e1, sdfg.arrays)
        except (NotImplementedError, ValueError) as ex:
            warnings.warn(f'validate_subsets failed: {ex}')
            return False

        # Find the true in desc (in case in_array is a view).
        true_in_array = in_array
        true_in_desc = in_desc
        if isinstance(in_desc, data.View):
            true_in_array = sdutil.get_last_view_node(graph, in_array)
            if not true_in_array:
                return False
            true_in_desc = sdfg.arrays[true_in_array.data]
        # Find the true out_desc (in case out_array is a view).
        true_out_array = out_array
        true_out_desc = out_desc
        true_out_subsets = [b_subset]
        if isinstance(out_desc, data.View):
            true_out_array = sdutil.get_last_view_node(graph, out_array)
            if not true_out_array:
                return False
            true_out_desc = sdfg.arrays[true_out_array.data]
            true_out_subsets = [e.data.get_dst_subset(e, graph) for e in graph.in_edges(true_out_array)]

        # Fail in the case of A -> V(A) or V(A) -> A
        is_array_to_view = (isinstance(in_desc, data.View) ^ isinstance(out_desc, data.View))
        if true_in_array is true_out_array and is_array_to_view:
            return False

        if not permissive:
            # Make sure the memlet covers the removed array
            subset = copy.deepcopy(e1.data.subset)
            subset.squeeze()
            shape = [sz for sz in in_desc.shape if sz != 1]
            if any(m != a for m, a in zip(subset.size(), shape)):
                return False

            # NOTE: Library node check
            # The transformation must not apply in non-permissive mode if in_array is
            # not a view, is output of a library node, and an access or a view
            # of out_desc is also input to the same library node.
            # The reason is that the application of the transformation will lead
            # to out_desc being both input and output of the library node.
            # We do not know if this is safe.

            if not isinstance(in_desc, data.View):

                edges_to_check = []
                for a in graph.in_edges(in_array):
                    if isinstance(a.src, nodes.LibraryNode):
                        edges_to_check.append(a)
                    elif (isinstance(a.src, nodes.AccessNode) and isinstance(sdfg.arrays[a.src.data], data.View)):
                        for b in graph.in_edges(a.src):
                            edges_to_check.append(graph.memlet_path(b)[0])

                for a in edges_to_check:
                    if isinstance(a.src, nodes.LibraryNode):
                        for b in graph.in_edges(a.src):
                            if isinstance(b.src, nodes.AccessNode):
                                desc = sdfg.arrays[b.src.data]
                                if isinstance(desc, data.View):
                                    n = sdutil.get_last_view_node(graph, b.src)
                                    if not n:
                                        return False
                                    desc = sdfg.arrays[n.data]
                                    if desc is true_out_desc:
                                        return False

            # In non-permissive mode, check if the state has two or more access nodes
            # for the output array. Definitely one of them (out_array) is a
            # write access. Therefore, there might be a RW, WR, or WW dependency.
            accesses = [
                n for n in graph.nodes()
                if isinstance(n, nodes.AccessNode) and n.data == true_out_array.data and n is not true_out_array
            ]
            if len(accesses) > 0:
                # We need to ensure that a data race will not happen if we
                # remove in_array.
                # First, we simplify the graph
                G = helpers.simplify_state(graph, remove_views=True)
                # Loop over the accesses
                for a in accesses:
                    subsets_intersect = False
                    for e in graph.out_edges(a):
                        try:
                            subset, _ = _validate_subsets(e, sdfg.arrays, src_name=a.data)
                        except (NotImplementedError, ValueError) as ex:
                            warnings.warn(f'validate_subsets failed: {ex}')
                            return False
                        for oset in true_out_subsets:
                            res = subsets.intersects(oset, subset)
                            if res == True or res is None:
                                subsets_intersect = True
                                break
                        if subsets_intersect:
                            break
                    if not subsets_intersect:
                        continue
                    try:
                        has_bward_path = nx.has_path(G, a, true_out_array)
                    except NodeNotFound:
                        has_bward_path = nx.has_path(graph.nx, a, true_out_array)
                    try:
                        has_fward_path = nx.has_path(G, true_out_array, a)
                    except NodeNotFound:
                        has_fward_path = nx.has_path(graph.nx, true_out_array, a)
                    # If there is no path between the access nodes (disconnected
                    # components), then it is definitely possible to have data
                    # races. Abort.
                    if not (has_bward_path or has_fward_path):
                        return False
                    # If there is a backward path then the (true) in_array must
                    # not be a direct successor of a.
                    try:
                        if has_bward_path and true_in_array in G.successors(a):
                            return False
                    except NetworkXError:
                        # The exception occurs when access a is not in G.
                        # This happens when the access is inside a (Map) scope.
                        # In such a case, it is dangerous to apply the
                        # transformation.
                        return False

        # Make sure that both arrays are using the same storage location
        # and are of the same type (e.g., Stream->Stream)
        if in_desc.storage != out_desc.storage:
            return False
        if type(in_desc) != type(out_desc):
            if isinstance(in_desc, data.View):
                # Case View -> Access
                # If the View points to the Access and has the same shape,
                # it can be removed, unless there is a reduction!
                e = sdutil.get_view_edge(graph, in_array)
                if e and e.dst is out_array and in_desc.shape == out_desc.shape:
                    from dace.libraries.standard import Reduce
                    for e in graph.in_edges(in_array):
                        if isinstance(e.src, Reduce):
                            return False
                    return True
                return False
            elif isinstance(out_desc, data.View):
                # Case Access -> View
                # If the View points to the Access (and has a different shape?)
                # then we should (probably) not remove the Access.
                e = sdutil.get_view_edge(graph, out_array)
                if e and e.src is in_array and in_desc.shape != out_desc.shape:
                    return False

                # Check that the View's immediate successors are Accesses.
                # Otherwise, the application of the transformation will result
                # in an ambiguous View.
                view_successors_desc = [
                    e.dst.desc(sdfg) if isinstance(e.dst, nodes.AccessNode) else None
                    for e in graph.out_edges(out_array)
                ]
                if any([not desc or isinstance(desc, data.View) for desc in view_successors_desc]):
                    return False
            else:
                # Something else, for example, Stream
                return False
        else:
            # Two views connected to each other
            if isinstance(in_desc, data.View):
                # Merge will be ambiguous
                if 'views' in in_array.in_connectors and 'views' in out_array.out_connectors:
                    return False
                return True

        # Find occurrences in this and other states
        occurrences = []
        for state in sdfg.nodes():
            occurrences.extend(
                [n for n in state.nodes() if isinstance(n, nodes.AccessNode) and n.data == in_array.data])
        for isedge in sdfg.edges():
            if in_array.data in isedge.data.free_symbols:
                occurrences.append(isedge)

        if len(occurrences) > 1:
            return False

        # 2. Iterate over the e2 edges
        for e2 in graph.in_edges(in_array):
            # 2-a. Extract/validate subsets for array A and others
            try:
                _, a2_subset = _validate_subsets(e2, sdfg.arrays)
            except (NotImplementedError, ValueError) as ex:
                warnings.warn(f'validate_subsets failed: {ex}')
                return False
            # 2-b. Check whether a2_subset covers a1_subset
            if not a2_subset.covers(a1_subset):
                return False
            # 2-c. Validate subsets in memlet tree
            # (should not be needed for valid SDGs)
            path = graph.memlet_tree(e2)
            for e3 in path:
                if e3 is not e2:
                    try:
                        _validate_subsets(e3, sdfg.arrays, dst_name=in_array.data)
                    except (NotImplementedError, ValueError) as ex:
                        warnings.warn(f'validate_subsets failed: {ex}')
                        return False

            # 2-d. If array is connected to a nested SDFG or view and strides are unequal, skip
            if in_desc.strides != out_desc.strides:
                sources = []
                if path.downwards:
                    sources = [path.root().edge]
                else:
                    sources = [e for e in path.leaves()]
                for source_edge in sources:
                    if isinstance(source_edge.src, nodes.AccessNode):
                        if isinstance(source_edge.src.desc(sdfg), data.View):
                            if not permissive:
                                return False
                    elif isinstance(source_edge.src, nodes.NestedSDFG):
                        if not permissive:
                            return False
                        conn = source_edge.src_conn
                        inner_desc = source_edge.src.sdfg.arrays[conn]
                        if inner_desc.strides != in_desc.strides:
                            # Cannot safely remove node without modifying strides and correctness
                            return False

        return True

    def _make_view(self, sdfg: SDFG, graph: SDFGState, in_array: nodes.AccessNode, out_array: nodes.AccessNode,
                   e1: graph.MultiConnectorEdge[mm.Memlet], b_subset: subsets.Subset, b_dims_to_pop: List[int]):
        in_desc = sdfg.arrays[in_array.data]
        out_desc = sdfg.arrays[out_array.data]
        # NOTE: We do not want to create another view, if the immediate
        # ancestors of in_array are views as well. We just remove it.
        in_ancestors_desc = [
            e.src.desc(sdfg) if isinstance(e.src, nodes.AccessNode) else None for e in graph.in_edges(in_array)
        ]
        if all([desc and isinstance(desc, data.View) for desc in in_ancestors_desc]):
            for e in graph.in_edges(in_array):
                a_subset, _ = _validate_subsets(e, sdfg.arrays)
                graph.add_edge(
                    e.src, e.src_conn, out_array, None,
                    mm.Memlet(out_array.data,
                              subset=b_subset,
                              other_subset=a_subset,
                              wcr=e1.data.wcr,
                              wcr_nonatomic=e1.data.wcr_nonatomic))
                graph.remove_edge(e)
            graph.remove_edge(e1)
            graph.remove_node(in_array)
            if in_array.data in sdfg.arrays:
                del sdfg.arrays[in_array.data]
            return
        view_strides = in_desc.strides
        if (b_dims_to_pop and len(b_dims_to_pop) == len(out_desc.shape) - len(in_desc.shape)):
            view_strides = [s for i, s in enumerate(out_desc.strides) if i not in b_dims_to_pop]
        sdfg.arrays[in_array.data] = data.View(in_desc.dtype, in_desc.shape, True, in_desc.allow_conflicts,
                                               out_desc.storage, out_desc.location, view_strides, in_desc.offset,
                                               out_desc.may_alias, dtypes.AllocationLifetime.Scope, in_desc.alignment,
                                               in_desc.debuginfo, in_desc.total_size)
        in_array.add_out_connector('views', force=True)
        e1._src_conn = 'views'

    def apply(self, graph, sdfg):
        in_array = self.in_array
        out_array = self.out_array
        in_desc = sdfg.arrays[in_array.data]
        out_desc = sdfg.arrays[out_array.data]

        # 1. Get edge e1 and extract subsets for arrays A and B
        e1 = graph.edges_between(in_array, out_array)[0]
        a1_subset, b_subset = _validate_subsets(e1, sdfg.arrays)

        # View connected to a view: simple case
        if (isinstance(in_desc, data.View) and isinstance(out_desc, data.View)):
            simple_case = True
            for e in graph.in_edges(in_array):
                if e.data.dst_subset is not None and a1_subset != e.data.dst_subset:
                    simple_case = False
                    break
            if simple_case:
                for e in graph.in_edges(in_array):
                    for e2 in graph.memlet_tree(e):
                        if e2 is e:
                            continue
                        if e2.data.data == in_array.data:
                            e2.data.data = out_array.data
                    new_memlet = copy.deepcopy(e.data)
                    if new_memlet.data == in_array.data:
                        new_memlet.data = out_array.data
                    new_memlet.dst_subset = b_subset
                    graph.add_edge(e.src, e.src_conn, out_array, e.dst_conn, new_memlet)
                graph.remove_node(in_array)
                try:
                    if in_array.data in sdfg.arrays:
                        sdfg.remove_data(in_array.data)
                except ValueError:  # Used somewhere else
                    pass
                return

        # Find extraneous A or B subset dimensions
        a_dims_to_pop = []
        b_dims_to_pop = []
        bset = b_subset
        popped = []
        if a1_subset and b_subset and a1_subset.dims() != b_subset.dims():
            a_size = a1_subset.size_exact()
            b_size = b_subset.size_exact()
            if a1_subset.dims() > b_subset.dims():
                a_dims_to_pop = find_dims_to_pop(a_size, b_size)
            else:
                b_dims_to_pop = find_dims_to_pop(b_size, a_size)
                bset, popped = pop_dims(b_subset, b_dims_to_pop)

        from dace.libraries.standard import Reduce
        reduction = False
        for e in graph.in_edges(in_array):
            if isinstance(e.src, Reduce) or (isinstance(e.src, nodes.NestedSDFG)
                                             and len(in_desc.shape) != len(out_desc.shape)):
                reduction = True

        # If:
        # 1. A reduce node is involved; or
        # 2. A NestedSDFG node is involved and the arrays have different dimensionality; or
        # 3. The memlet does not cover the removed array; or
        # 4. Dimensions are mismatching (all dimensions are popped);
        # create a view.
        if reduction or len(a_dims_to_pop) == len(in_desc.shape) or any(
                m != a for m, a in zip(a1_subset.size(), in_desc.shape)):
            self._make_view(sdfg, graph, in_array, out_array, e1, b_subset, b_dims_to_pop)
            return in_array

        # Validate that subsets are composable. If not, make a view
        try:
            for e2 in graph.in_edges(in_array):
                path = graph.memlet_tree(e2)
                wcr = e1.data.wcr
                wcr_nonatomic = e1.data.wcr_nonatomic
                for e3 in path:
                    # 2-a. Extract subsets for array B and others
                    other_subset, a3_subset = _validate_subsets(e3, sdfg.arrays, dst_name=in_array.data)
                    # 2-b. Modify memlet to match array B.
                    dname = out_array.data
                    src_is_data = False
                    a3_subset.offset(a1_subset, negative=True)

                    if a3_subset and a_dims_to_pop:
                        aset, _ = pop_dims(a3_subset, a_dims_to_pop)
                    else:
                        aset = a3_subset

                    compose_and_push_back(bset, aset, b_dims_to_pop, popped)
        except (ValueError, NotImplementedError):
            self._make_view(sdfg, graph, in_array, out_array, e1, b_subset, b_dims_to_pop)
            return in_array

        # 2. Iterate over the e2 edges and traverse the memlet tree
        for e2 in graph.in_edges(in_array):
            path = graph.memlet_tree(e2)
            wcr = e1.data.wcr
            wcr_nonatomic = e1.data.wcr_nonatomic
            for e3 in path:
                # 2-a. Extract subsets for array B and others
                other_subset, a3_subset = _validate_subsets(e3, sdfg.arrays, dst_name=in_array.data)
                # 2-b. Modify memlet to match array B.
                dname = out_array.data
                src_is_data = False
                a3_subset.offset(a1_subset, negative=True)

                if a3_subset and a_dims_to_pop:
                    aset, _ = pop_dims(a3_subset, a_dims_to_pop)
                else:
                    aset = a3_subset

                dst_subset = compose_and_push_back(bset, aset, b_dims_to_pop, popped)
                # NOTE: This fixes the following case:
                # Tasklet ----> A[subset] ----> ... -----> A
                # Tasklet is not data, so it doesn't have an other subset.
                if isinstance(e3.src, nodes.AccessNode):
                    if e3.src.data == out_array.data:
                        dname = e3.src.data
                        src_is_data = True
                    src_subset = other_subset
                else:
                    src_subset = None

                subset = src_subset if src_is_data else dst_subset
                other_subset = dst_subset if src_is_data else src_subset
                e3.data.data = dname
                e3.data.subset = subset
                e3.data.other_subset = other_subset
                wcr = wcr or e3.data.wcr
                wcr_nonatomic = wcr_nonatomic or e3.data.wcr_nonatomic
                e3.data.wcr = wcr
                e3.data.wcr_nonatomic = wcr_nonatomic

            # 2-c. Remove edge and add new one
            graph.remove_edge(e2)
            e2.data.wcr = wcr
            e2.data.wcr_nonatomic = wcr_nonatomic
            graph.add_edge(e2.src, e2.src_conn, out_array, e2.dst_conn, e2.data)

            # 2-d. Fix strides in nested SDFGs
            if in_desc.strides != out_desc.strides:
                sources = []
                if path.downwards:
                    sources = [path.root().edge]
                else:
                    sources = [e for e in path.leaves()]
                for source_edge in sources:
                    if not isinstance(source_edge.src, nodes.NestedSDFG):
                        continue
                    conn = source_edge.src_conn
                    inner_desc = source_edge.src.sdfg.arrays[conn]
                    inner_desc.strides = out_desc.strides

        # Finally, remove in_array node
        graph.remove_node(in_array)
        try:
            if in_array.data in sdfg.arrays:
                sdfg.remove_data(in_array.data)
        except ValueError:  # Already in use (e.g., with Views)
            pass


class RedundantSecondArray(pm.SingleStateTransformation):
    """ Implements the redundant array removal transformation, applied
        when a transient array is copied from and to (from another array),
        but never used anywhere else. This transformation removes the second
        array. """

    in_array = pm.PatternNode(nodes.AccessNode)
    out_array = pm.PatternNode(nodes.AccessNode)

    @classmethod
    def expressions(cls):
        return [sdutil.node_path_graph(cls.in_array, cls.out_array)]

    def can_be_applied(self, graph, expr_index, sdfg, permissive=False):
        in_array = self.in_array
        out_array = self.out_array

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
        try:
            a_subset, b1_subset = _validate_subsets(e1, sdfg.arrays)
        except (NotImplementedError, ValueError) as ex:
            warnings.warn(f'validate_subsets failed: {ex}')
            return False

        # Find the true in desc (in case in_array is a view).
        true_in_array = in_array
        true_in_desc = in_desc
        true_in_subsets = [a_subset]
        if isinstance(in_desc, data.View):
            true_in_array = sdutil.get_last_view_node(graph, in_array)
            if not true_in_array:
                return False
            true_in_desc = sdfg.arrays[true_in_array.data]
            true_in_subsets = [e.data.get_src_subset(e, graph) for e in graph.out_edges(true_in_array)]
        # Find the true out_desc (in case out_array is a view).
        true_out_array = out_array
        true_out_desc = out_desc
        if isinstance(out_desc, data.View):
            true_out_array = sdutil.get_last_view_node(graph, out_array)
            if not true_out_array:
                return False
            true_out_desc = sdfg.arrays[true_out_array.data]

        # Fail in the case of A -> V(A) or V(A) -> A
        is_array_to_view = (isinstance(in_desc, data.View) ^ isinstance(out_desc, data.View))
        if true_in_array is true_out_array and is_array_to_view:
            return False

        if not permissive:
            # Make sure the memlet covers the removed array
            if not b1_subset:
                return False
            subset = copy.deepcopy(b1_subset)
            subset.squeeze()
            shape = [sz for sz in out_desc.shape if sz != 1]
            if any(m != a for m, a in zip(subset.size(), shape)):
                return False

            # NOTE: Library node check
            # The transformation must not apply if out_array is
            # not a view, is input to a library node, and an access or a view
            # of in_desc is also output to the same library node.
            # The reason is that the application of the transformation will lead
            # to in_desc being both input and output of the library node.
            # We do not know if this is safe.

            if not isinstance(out_desc, data.View):

                edges_to_check = []
                for a in graph.out_edges(out_array):
                    if isinstance(a.dst, nodes.LibraryNode):
                        edges_to_check.append(a)
                    elif (isinstance(a.dst, nodes.AccessNode) and isinstance(sdfg.arrays[a.dst.data], data.View)):
                        for b in graph.out_edges(a.dst):
                            edges_to_check.append(graph.memlet_path(b)[-1])

                for a in edges_to_check:
                    if isinstance(a.dst, nodes.LibraryNode):
                        for b in graph.out_edges(a.dst):
                            if isinstance(b.dst, nodes.AccessNode):
                                desc = sdfg.arrays[b.dst.data]
                                if isinstance(desc, data.View):
                                    n = sdutil.get_last_view_node(graph, b.dst)
                                    if not n:
                                        return False
                                    desc = sdfg.arrays[n.data]
                                    if desc is true_in_desc:
                                        return False

            # Check if the state has two or more access nodes
            # for in_array and at least one of them is a write access. There
            # might be a RW, WR, or WW dependency.
            accesses = [
                n for n in graph.nodes()
                if isinstance(n, nodes.AccessNode) and n.data == true_in_array.data and n is not true_in_array
            ]
            if len(accesses) > 0:
                if (graph.in_degree(true_in_array) > 0 or any(graph.in_degree(a) > 0 for a in accesses)):
                    # We need to ensure that a data race will not happen if we
                    # remove in_array.
                    # First, we simplify the graph
                    G = helpers.simplify_state(graph, remove_views=True)
                    # Loop over the accesses
                    for a in accesses:
                        subsets_intersect = False
                        for e in graph.in_edges(a):
                            try:
                                _, subset = _validate_subsets(e, sdfg.arrays, dst_name=a.data)
                            except (NotImplementedError, ValueError) as ex:
                                warnings.warn(f'validate_subsets failed: {ex}')
                                return False
                            for iset in true_in_subsets:
                                res = subsets.intersects(iset, subset)
                                if res == True or res is None:
                                    subsets_intersect = True
                                    break
                            if subsets_intersect:
                                break
                        if not subsets_intersect:
                            continue
                        try:
                            has_bward_path = nx.has_path(G, a, true_in_array)
                        except NodeNotFound:
                            has_bward_path = nx.has_path(graph.nx, a, true_in_array)
                        try:
                            has_fward_path = nx.has_path(G, true_in_array, a)
                        except NodeNotFound:
                            has_fward_path = nx.has_path(graph.nx, true_in_array, a)
                        # If there is no path between the access nodes
                        # (disconnected components), then it is definitely
                        # possible to have data races. Abort.
                        if not (has_bward_path or has_fward_path):
                            return False
                        # If there is a forward path then a must not be a direct
                        # successor of the (true) out_array.
                        try:
                            if has_fward_path and a in G.successors(true_out_array):
                                return False
                        except NetworkXError:
                            return False

        # Make sure that both arrays are using the same storage location
        # and are of the same type (e.g., Stream->Stream)
        if in_desc.storage != out_desc.storage:
            return False
        if type(in_desc) != type(out_desc):
            if isinstance(in_desc, data.View):
                # Case View -> Access
                # If the View points to the Access (and has a different shape?)
                # then we should (probably) not remove the Access.
                e = sdutil.get_view_edge(graph, in_array)
                if e and e.dst is out_array and in_desc.shape != out_desc.shape:
                    return False
                # Check that the View's immediate ancestors are Accesses.
                # Otherwise, the application of the transformation will result
                # in an ambiguous View.
                view_ancestors_desc = [
                    e.src.desc(sdfg) if isinstance(e.src, nodes.AccessNode) else None for e in graph.in_edges(in_array)
                ]
                if any([not desc or isinstance(desc, data.View) for desc in view_ancestors_desc]):
                    return False
            elif isinstance(out_desc, data.View):
                # Case Access -> View
                # If the View points to the Access and has the same shape,
                # it can be removed
                e = sdutil.get_view_edge(graph, out_array)
                if e and e.src is in_array and in_desc.shape == out_desc.shape:
                    return True
                return False
            else:
                # Something else, for example, Stream
                return False
        else:
            # Two views connected to each other
            if isinstance(in_desc, data.View):
                return False

        # Find occurrences in this and other states
        occurrences = []
        for state in sdfg.nodes():
            occurrences.extend(
                [n for n in state.nodes() if isinstance(n, nodes.AccessNode) and n.data == out_array.data])
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
            except (NotImplementedError, ValueError) as ex:
                warnings.warn(f'validate_subsets failed: {ex}')
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
                        _validate_subsets(e3, sdfg.arrays, src_name=out_array.data)
                    except (NotImplementedError, ValueError) as ex:
                        warnings.warn(f'validate_subsets failed: {ex}')
                        return False

            # 2-d. If array is connected to a nested SDFG or view and strides are unequal, skip
            if in_desc.strides != out_desc.strides:
                sources = []
                if not path.downwards:
                    sources = [path.root().edge]
                else:
                    sources = [e for e in path.leaves()]
                for source_edge in sources:
                    if isinstance(source_edge.dst, nodes.AccessNode):
                        if isinstance(source_edge.dst.desc(sdfg), data.View):
                            if not permissive:
                                return False
                    elif isinstance(source_edge.dst, nodes.NestedSDFG):
                        if not permissive:
                            return False
                        conn = source_edge.dst_conn
                        inner_desc = source_edge.dst.sdfg.arrays[conn]
                        if inner_desc.strides != in_desc.strides:
                            # Cannot safely remove node without modifying strides and correctness
                            return False

        return True

    def apply(self, graph: SDFGState, sdfg: SDFG):
        in_array = self.in_array
        out_array = self.out_array
        in_desc = sdfg.arrays[in_array.data]
        out_desc = sdfg.arrays[out_array.data]

        # We assume the following pattern: A -- e1 --> B -- e2 --> others

        # 1. Get edge e1 and extract subsets for arrays A and B
        e1 = graph.edges_between(in_array, out_array)[0]
        a_subset, b1_subset = _validate_subsets(e1, sdfg.arrays)

        # Find extraneous A or B subset dimensions
        a_dims_to_pop = []
        b_dims_to_pop = []
        aset = a_subset
        popped = []
        if a_subset and b1_subset and a_subset.dims() != b1_subset.dims():
            a_size = a_subset.size_exact()
            b_size = b1_subset.size_exact()
            if a_subset.dims() > b1_subset.dims():
                a_dims_to_pop = find_dims_to_pop(a_size, b_size)
                aset, popped = pop_dims(a_subset, a_dims_to_pop)
            else:
                b_dims_to_pop = find_dims_to_pop(b_size, a_size)

        # If the src subset does not cover the removed array, create a view.
        if a_subset and any(m != a for m, a in zip(a_subset.size(), out_desc.shape)):
            # NOTE: We do not want to create another view, if the immediate
            # successors of out_array are views as well. We just remove it.
            out_successors_desc = [
                e.dst.desc(sdfg) if isinstance(e.dst, nodes.AccessNode) else None for e in graph.out_edges(out_array)
            ]
            if all([desc and isinstance(desc, data.View) for desc in out_successors_desc]):
                for e in graph.out_edges(out_array):
                    _, b_subset = _validate_subsets(e, sdfg.arrays)
                    graph.add_edge(
                        in_array, None, e.dst, e.dst_conn,
                        mm.Memlet(in_array.data,
                                  subset=a_subset,
                                  other_subset=b_subset,
                                  wcr=e1.data.wcr,
                                  wcr_nonatomic=e1.data.wcr_nonatomic))
                    graph.remove_edge(e)
                graph.remove_edge(e1)
                graph.remove_node(out_array)
                if out_array.data in sdfg.arrays:
                    del sdfg.arrays[out_array.data]
                # If first node is now isolated, remove it
                if len(graph.all_edges(in_array)) == 0:
                    graph.remove_node(in_array)
                return
            view_strides = out_desc.strides
            if (a_dims_to_pop and len(a_dims_to_pop) == len(in_desc.shape) - len(out_desc.shape)):
                view_strides = [s for i, s in enumerate(in_desc.strides) if i not in a_dims_to_pop]
            sdfg.arrays[out_array.data] = data.View(out_desc.dtype, out_desc.shape, True, out_desc.allow_conflicts,
                                                    in_desc.storage, in_desc.location, view_strides, out_desc.offset,
                                                    in_desc.may_alias, dtypes.AllocationLifetime.Scope,
                                                    out_desc.alignment, out_desc.debuginfo, out_desc.total_size)
            out_array.add_in_connector('views', force=True)
            e1._dst_conn = 'views'
            return out_array

        # 2. Iterate over the e2 edges and traverse the memlet tree
        for e2 in graph.out_edges(out_array):
            path = graph.memlet_tree(e2)
            wcr = e1.data.wcr
            wcr_nonatomic = e1.data.wcr_nonatomic
            for e3 in path:
                # 2-a. Extract subsets for array B and others
                b3_subset, other_subset = _validate_subsets(e3, sdfg.arrays, src_name=out_array.data)
                # 2-b. Modify memlet to match array A. Example:
                # A -- (0, a:b)/(c:c+b) --> B -- (c+d)/None --> others
                # A -- (0, a+d)/None --> others
                e3.data.data = in_array.data
                # (c+d) - (c:c+b) = (d)
                b3_subset.offset(b1_subset, negative=True)
                # (0, a:b)(d) = (0, a+d) (or offset for indices)

                if b3_subset and b_dims_to_pop:
                    bset, _ = pop_dims(b3_subset, b_dims_to_pop)
                else:
                    bset = b3_subset

                e3.data.subset = compose_and_push_back(aset, bset, a_dims_to_pop, popped)
                # NOTE: This fixes the following case:
                # A ----> A[subset] ----> ... -----> Tasklet
                # Tasklet is not data, so it doesn't have an other subset.
                if isinstance(e3.dst, nodes.AccessNode):
                    e3.data.other_subset = other_subset
                else:
                    e3.data.other_subset = None
                wcr = wcr or e3.data.wcr
                wcr_nonatomic = wcr_nonatomic or e3.data.wcr_nonatomic
                e3.data.wcr = wcr
                e3.data.wcr_nonatomic = wcr_nonatomic

            # 2-c. Remove edge and add new one
            graph.remove_edge(e2)
            e2.data.wcr = wcr
            e2.data.wcr_nonatomic = wcr_nonatomic
            graph.add_edge(in_array, e2.src_conn, e2.dst, e2.dst_conn, e2.data)

            # 2-d. Fix strides in nested SDFGs
            if in_desc.strides != out_desc.strides:
                sources = []
                if not path.downwards:
                    sources = [path.root().edge]
                else:
                    sources = [e for e in path.leaves()]
                for source_edge in sources:
                    if not isinstance(source_edge.dst, nodes.NestedSDFG):
                        continue
                    conn = source_edge.dst_conn
                    inner_desc = source_edge.dst.sdfg.arrays[conn]
                    inner_desc.strides = out_desc.strides

        # Finally, remove out_array node
        graph.remove_node(out_array)
        if out_array.data in sdfg.arrays:
            try:
                sdfg.remove_data(out_array.data)
            except ValueError:  # Already in use (e.g., with Views)
                pass

        # If first node is now isolated, remove it
        if len(graph.all_edges(in_array)) == 0:
            graph.remove_node(in_array)


class SqueezeViewRemove(pm.SingleStateTransformation):
    in_array = pm.PatternNode(nodes.AccessNode)
    out_array = pm.PatternNode(nodes.AccessNode)

    @classmethod
    def expressions(cls):
        return [sdutil.node_path_graph(cls.in_array, cls.out_array)]

    def can_be_applied(self, state: SDFGState, expr_index: int, sdfg: SDFG, permissive: bool = False):
        in_array = self.in_array
        out_array = self.out_array

        in_desc = in_array.desc(sdfg)
        out_desc = out_array.desc(sdfg)

        if state.out_degree(out_array) != 1:
            return False

        if not isinstance(out_desc, data.View):
            return False

        vedge = state.out_edges(out_array)[0]
        if vedge.data.data != out_array.data:  # Ensures subset comes from view
            return False
        view_subset = copy.deepcopy(vedge.data.subset)

        aedge = state.edges_between(in_array, out_array)[0]
        if aedge.data.data != in_array.data:
            return False
        array_subset = copy.deepcopy(aedge.data.subset)

        vsqdims = view_subset.squeeze()

        # View may modify the behavior of a library node
        if not permissive and isinstance(vedge.dst, nodes.LibraryNode):
            return False

        # Check that subsets are equivalent
        if array_subset != view_subset:
            return False

        # Verify strides after squeeze
        astrides = tuple(in_desc.strides)  #s for i, s in enumerate(in_desc.strides) if i not in asqdims)
        vstrides = tuple(s for i, s in enumerate(out_desc.strides) if i in vsqdims)
        if astrides != vstrides:
            return False

        return True

    def apply(self, state: SDFGState, sdfg: SDFG):
        in_array = self.in_array
        out_array = self.out_array
        out_desc = out_array.desc(sdfg)

        vedge = state.out_edges(out_array)[0]
        view_subset = copy.deepcopy(vedge.data.subset)

        aedge = state.edges_between(in_array, out_array)[0]
        array_subset = copy.deepcopy(aedge.data.subset)

        vsqdims = view_subset.squeeze()

        # Modify data and subset on all outgoing edges
        for e in state.memlet_tree(vedge):
            e.data.data = in_array.data
            e.data.subset.squeeze(vsqdims)

        # Redirect original edge to point to data
        state.remove_edge(vedge)
        state.add_edge(in_array, vedge.src_conn, vedge.dst, vedge.dst_conn, vedge.data)

        # Remove node and descriptor
        state.remove_node(out_array)
        try:
            sdfg.remove_data(out_array.data)
        except ValueError:  # Already in use (e.g., with Views)
            pass


class UnsqueezeViewRemove(pm.SingleStateTransformation):
    in_array = pm.PatternNode(nodes.AccessNode)
    out_array = pm.PatternNode(nodes.AccessNode)

    @classmethod
    def expressions(cls):
        return [sdutil.node_path_graph(cls.in_array, cls.out_array)]

    def can_be_applied(self, state: SDFGState, expr_index: int, sdfg: SDFG, permissive: bool = False):
        in_array = self.in_array
        out_array = self.out_array

        in_desc = in_array.desc(sdfg)
        out_desc = out_array.desc(sdfg)

        if state.in_degree(in_array) != 1:
            return False

        if not isinstance(in_desc, data.View):
            return False

        vedge = state.in_edges(in_array)[0]
        if vedge.data.data != in_array.data:  # Ensures subset comes from view
            return False
        view_subset = copy.deepcopy(vedge.data.subset)

        aedge = state.edges_between(in_array, out_array)[0]
        if aedge.data.data != out_array.data:
            return False
        array_subset = copy.deepcopy(aedge.data.subset)

        asqdims = array_subset.squeeze()

        # View may modify the behavior of a library node
        if not permissive and isinstance(vedge.src, nodes.LibraryNode):
            return False

        # Check that subsets are equivalent
        if array_subset != view_subset:
            return False

        # Verify strides after squeeze
        vstrides = tuple(in_desc.strides)
        astrides = tuple(s for i, s in enumerate(out_desc.strides) if i in asqdims)
        if astrides != vstrides:
            return False

        return True

    def apply(self, state: SDFGState, sdfg: SDFG):
        in_array = self.in_array
        out_array = self.out_array

        vedge = state.in_edges(in_array)[0]

        aedge = state.edges_between(in_array, out_array)[0]
        array_subset = copy.deepcopy(aedge.data.subset)

        asqdims = array_subset.squeeze()
        asqdims_mirror = [i for i in range(aedge.data.subset.dims()) if i not in asqdims]

        # Modify data and subset on all outgoing edges
        for e in state.memlet_tree(vedge):
            e.data.data = out_array.data
            # Offset subset, then unsqueeze
            e.data.subset.offset(aedge.data.subset, False)
            e.data.subset.unsqueeze(asqdims_mirror)
            for i in asqdims_mirror:
                e.data.subset.ranges[i] = aedge.data.subset.ranges[i]

        # Redirect original edge to point to data
        state.remove_edge(vedge)
        state.add_edge(vedge.src, vedge.src_conn, out_array, vedge.dst_conn, vedge.data)

        # Remove node and descriptor
        state.remove_node(in_array)
        try:
            sdfg.remove_data(in_array.data)
        except ValueError:  # Already in use (e.g., with Views)
            pass


def _is_slice(adesc: data.Array, vdesc: data.View) -> bool:
    """ Checks whether a View of an Array is a slice or not. """
    # Explicitly fail in case of Views with more dimensions than the Array.
    # NOTE: We want to avoid matching slices produced with np.newaxis
    if len(vdesc.shape) > len(adesc.shape):
        return False
    try:
        # Iterate over the View's strides.
        for vi, s in enumerate(vdesc.strides):
            # All of the View's strides must exist in the Array's strides.
            # Otherwise, it is not a slice but a reintepretation.
            ai = adesc.strides.index(s)
            # If the View's length is not clearly less or equal than
            # the Array's corresponding length, then we cannot confirm that
            # the View is a slice.
            if (vdesc.shape[vi] <= adesc.shape[ai]) == True:
                continue
            else:
                return False
    except ValueError:  # list.index throws ValueError if a stride is not found
        return False

    return True


def _sliced_dims(adesc: data.Array, vdesc: data.View) -> List[int]:
    """ Returns the Array dimensions viewed by a slice-View.
        NOTE: This method assumes that `_is_slice(adesc, vdesc) == True`.
    """
    return [adesc.strides.index(s) for s in vdesc.strides]


class RedundantReadSlice(pm.SingleStateTransformation):
    """ Detects patterns of the form Array -> View(Array) and removes
    the View if it is a slice. """

    in_array = pm.PatternNode(nodes.AccessNode)
    out_array = pm.PatternNode(nodes.AccessNode)

    @classmethod
    def expressions(cls):
        return [sdutil.node_path_graph(cls.in_array, cls.out_array)]

    def can_be_applied(self, graph, expr_index, sdfg, permissive=False):
        in_array = self.in_array
        out_array = self.out_array

        in_desc = in_array.desc(sdfg)
        out_desc = out_array.desc(sdfg)

        # Make sure that both arrays are using the same storage location.
        if in_desc.storage != out_desc.storage:
            return False

        # The match must be Array -> View
        if not (isinstance(in_desc, data.Array) and not isinstance(in_desc, data.View)):
            return False
        if not isinstance(out_desc, data.View):
            return False

        # Ensure in degree is one (only one source, which is in_array)
        if graph.in_degree(out_array) != 1:
            return False

        # The match must be Array -> View(Array), i.e.,
        # the View must point to the Array. Find the true out_desc.
        true_out_array = sdutil.get_last_view_node(graph, out_array)
        if not true_out_array:
            return False
        if true_out_array is not in_array:
            return False

        # Ensure that the View is a slice of the Array.
        if not _is_slice(in_desc, out_desc):
            return False

        # Get edge e1 and extract subsets for the Array and View
        e = graph.edges_between(in_array, out_array)[0]
        a_subset = e.data.get_src_subset(e, graph)
        v_subset = e.data.get_dst_subset(e, graph)

        # Make sure the memlet covers the removed View.
        # NOTE: Since we assume that the View is a slice of the Array, the
        # following must hold (after removing unit-sized dimensions):
        # a_subset.size() == v_subset.size() == out_desc.shape
        if not (a_subset and v_subset):
            return False
        out_shape = [s for s in out_desc.shape if s != 1]
        for subset in (a_subset, v_subset):
            tmp = copy.deepcopy(subset)
            tmp.squeeze()
            if len(tmp) != len(out_shape):
                return False
            if any(m != a for m, a in zip(tmp.size(), out_shape)):
                return False

        return True

    def apply(self, graph, sdfg):
        in_array = self.in_array
        out_array = self.out_array
        in_desc = sdfg.arrays[in_array.data]
        out_desc = sdfg.arrays[out_array.data]

        # We assume the following pattern: A -- e1 --> V(A) -- e2 --> others

        # Get edge e1 and extract subsets for the Array and View
        e1 = graph.edges_between(in_array, out_array)[0]
        # a_subset, v1_subset = _validate_subsets(e1, sdfg.arrays)
        a_subset = e1.data.get_src_subset(e1, graph)
        v1_subset = e1.data.get_dst_subset(e1, graph)

        # Split the dimensions of A to sliced and non-viewed
        sliced_dims = _sliced_dims(in_desc, out_desc)
        nviewed_dims = [i for i in range(len(in_desc.shape) - 1, -1, -1) if i not in sliced_dims]
        aset, popped = pop_dims(a_subset, nviewed_dims)

        # Iterate over the e2 edges and traverse the memlet tree
        for e2 in graph.out_edges(out_array):
            path = graph.memlet_tree(e2)
            wcr = e1.data.wcr
            wcr_nonatomic = e1.data.wcr_nonatomic
            for e3 in path:
                # Extract subsets for view V and others
                v3_subset, other_subset = _validate_subsets(e3, sdfg.arrays, src_name=out_array.data)
                # Modify memlet to match array A. Example:
                # A -- (0, a:b)/(c:c+b) --> V -- (c+d)/None --> others
                # A -- (0, a+d)/None --> others
                e3.data.data = in_array.data
                e3.data._is_data_src = True
                # (c+d) - (c:c+b) = (d)
                v3_subset.offset(v1_subset, negative=True)
                # (0, a:b)(d) = (0, a+d) (or offset for indices)

                vset = v3_subset

                e3.data.src_subset = compose_and_push_back(aset, vset, nviewed_dims, popped)
                # NOTE: This fixes the following case:
                # A ----> A[subset] ----> ... -----> Tasklet
                # Tasklet is not data, so it doesn't have an other subset.
                if isinstance(e3.dst, nodes.AccessNode):
                    e3.data.dst_subset = other_subset
                else:
                    e3.data.dst_subset = None
                wcr = wcr or e3.data.wcr
                wcr_nonatomic = wcr_nonatomic or e3.data.wcr_nonatomic
                e3.data.wcr = wcr
                e3.data.wcr_nonatomic = wcr_nonatomic

            # Remove edge and add new one
            graph.remove_edge(e2)
            e2.data.wcr = wcr
            e2.data.wcr_nonatomic = wcr_nonatomic
            graph.add_edge(in_array, e2.src_conn, e2.dst, e2.dst_conn, e2.data)

        # Finally, remove out_array node
        graph.remove_node(out_array)
        if out_array.data in sdfg.arrays:
            try:
                sdfg.remove_data(out_array.data)
            except ValueError:  # Already in use (e.g., with Views)
                pass


class RedundantWriteSlice(pm.SingleStateTransformation):
    """ Detects patterns of the form View(Array) -> Array and removes
    the View if it is a slice. """

    in_array = pm.PatternNode(nodes.AccessNode)
    out_array = pm.PatternNode(nodes.AccessNode)

    @classmethod
    def expressions(cls):
        return [sdutil.node_path_graph(cls.in_array, cls.out_array)]

    def can_be_applied(self, graph, expr_index, sdfg, permissive=False):
        in_array = self.in_array
        out_array = self.out_array

        in_desc = in_array.desc(sdfg)
        out_desc = out_array.desc(sdfg)

        # Make sure that both arrays are using the same storage location.
        if in_desc.storage != out_desc.storage:
            return False

        # The match must be View -> Array
        if not (isinstance(out_desc, data.Array) and not isinstance(out_desc, data.View)):
            return False
        if not isinstance(in_desc, data.View):
            return False

        # Ensure out degree is one (only one target, which is out_array)
        if graph.out_degree(in_array) != 1:
            return False

        # The match must be View(Array) -> Array , i.e.,
        # the View must point to the Array. Find the true in_desc.
        true_in_array = sdutil.get_last_view_node(graph, in_array)
        if not true_in_array:
            return False
        if true_in_array is not out_array:
            return False

        # If the View receives data from a reduction, fail.
        from dace.libraries.standard import Reduce
        for e in graph.in_edges(in_array):
            if isinstance(e.src, Reduce):
                return False

        # Ensure that the View is a slice of the Array.
        if not _is_slice(out_desc, in_desc):
            return False

        # Get edge e1 and extract subsets for the Array and View
        e = graph.edges_between(in_array, out_array)[0]
        v_subset = e.data.get_src_subset(e, graph)
        a_subset = e.data.get_dst_subset(e, graph)

        # Make sure the memlet covers the removed View.
        # NOTE: Since we assume that the View is a slice of the Array, the
        # following must hold:
        # a_subset.size() == v_subset.size() == in_desc.shape
        if not (a_subset and v_subset):
            return False
        in_shape = [s for s in in_desc.shape if s != 1]
        for subset in (a_subset, v_subset):
            tmp = copy.deepcopy(subset)
            tmp.squeeze()
            if len(tmp) != len(in_shape):
                return False
            if any(m != a for m, a in zip(tmp.size(), in_shape)):
                return False

        return True

    def apply(self, graph, sdfg):
        in_array = self.in_array
        out_array = self.out_array
        in_desc = sdfg.arrays[in_array.data]
        out_desc = sdfg.arrays[out_array.data]

        # We assume the following pattern: others -- e2 --> V(A) -- e1 --> A

        # Get edge e1 and extract subsets for the Array and View
        e1 = graph.edges_between(in_array, out_array)[0]
        v1_subset = e1.data.get_src_subset(e1, graph)
        a_subset = e1.data.get_dst_subset(e1, graph)

        # Split the dimensions of A to sliced and non-viewed
        sliced_dims = _sliced_dims(out_desc, in_desc)
        nviewed_dims = [i for i in range(len(out_desc.shape) - 1, -1, -1) if i not in sliced_dims]
        aset, popped = pop_dims(a_subset, nviewed_dims)

        # Iterate over the e2 edges and traverse the memlet tree
        for e2 in graph.in_edges(in_array):
            path = graph.memlet_tree(e2)
            wcr = e1.data.wcr
            wcr_nonatomic = e1.data.wcr_nonatomic
            for e3 in path:
                # Extract subsets for view V and others
                other_subset, v3_subset = _validate_subsets(e3, sdfg.arrays, dst_name=in_array.data)
                # Modify memlet to match array A.
                e3.data.data = out_array.data
                e3.data._is_data_src = False
                v3_subset.offset(v1_subset, negative=True)

                vset = v3_subset

                e3.data.dst_subset = compose_and_push_back(aset, vset, nviewed_dims, popped)
                # NOTE: This fixes the following case:
                # Tasklet ----> A[subset] ----> ... -----> A
                # Tasklet is not data, so it doesn't have an other subset.
                if isinstance(e3.src, nodes.AccessNode):
                    e3.data.src_subset = other_subset
                else:
                    e3.data.src_subset = None
                wcr = wcr or e3.data.wcr
                wcr_nonatomic = wcr_nonatomic or e3.data.wcr_nonatomic
                e3.data.wcr = wcr
                e3.data.wcr_nonatomic = wcr_nonatomic

            # Remove edge and add new one
            graph.remove_edge(e2)
            e2.data.wcr = wcr
            e2.data.wcr_nonatomic = wcr_nonatomic
            graph.add_edge(e2.src, e2.src_conn, out_array, e2.dst_conn, e2.data)

        # Finally, remove in_array node
        graph.remove_node(in_array)
        if in_array.data in sdfg.arrays:
            try:
                sdfg.remove_data(in_array.data)
            except ValueError:  # Already in use (e.g., with Views)
                pass


class RemoveSliceView(pm.SingleStateTransformation):
    """ Removes views which can be represented by slicing (e.g., A[i, :, j, None]). """

    view = pm.PatternNode(nodes.AccessNode)

    @classmethod
    def expressions(cls):
        return [sdutil.node_path_graph(cls.view)]

    def can_be_applied(self, state: SDFGState, _: int, sdfg: SDFG, permissive=False):
        desc = self.view.desc(sdfg)

        # Ensure view
        if not isinstance(desc, data.View):
            return False

        # Get viewed node and non-viewed edges
        view_edge = sdutil.get_view_edge(state, self.view)
        if view_edge is None:
            return False

        # Gather metadata
        viewed: nodes.AccessNode
        non_view_edges: List[graph.MultiConnectorEdge[mm.Memlet]]
        is_src: bool
        if view_edge.dst is self.view:
            viewed = state.memlet_path(view_edge)[0].src
            non_view_edges = state.out_edges(self.view)
            subset = view_edge.data.get_src_subset(view_edge, state)
            is_src = True
        else:
            viewed = state.memlet_path(view_edge)[-1].dst
            non_view_edges = state.in_edges(self.view)
            subset = view_edge.data.get_dst_subset(view_edge, state)
            is_src = False

        if subset is None:
            # `subset = None` means the entire viewed data container is used
            subset = subsets.Range.from_array(viewed.desc(sdfg))

        ########################################################
        # Syntactic feasibility: ensure memlets reach managable node types (access nodes, tasklets, nested SDFGs if
        # strides match) rather than library nodes, which may behave in a custom manner based on the memlet shape.
        if not permissive:
            for e in non_view_edges:
                for sink in state.memlet_tree(e).leaves():
                    sink_node = sink.dst if is_src else sink.src
                    sink_conn = sink.dst_conn if is_src else sink.src_conn
                    if isinstance(sink_node, nodes.LibraryNode):
                        return False
                    if isinstance(sink_node, nodes.NestedSDFG):
                        if sink_conn in sink_node.sdfg.arrays:
                            ndesc = sink_node.sdfg.arrays[sink_conn]
                            if ndesc.strides != desc.strides or ndesc.dtype != desc.dtype:
                                return False

        ########################################################
        # Semantic feasibility: ensure memlets can be safely collapsed/modified to match the original access node.

        # If descriptor type changed, bail
        vdesc = viewed.desc(sdfg)
        if vdesc.dtype != desc.dtype:
            return False

        if sdutil.map_view_to_array(desc, vdesc, subset) is None:
            return False

        return True

    def apply(self, state: SDFGState, sdfg: SDFG):
        desc = self.view.desc(sdfg)

        # Get viewed node and non-viewed edges
        view_edge = sdutil.get_view_edge(state, self.view)

        # Gather metadata
        viewed: nodes.AccessNode
        non_view_edges: List[graph.MultiConnectorEdge[mm.Memlet]]
        subset: subsets.Range
        is_src: bool
        if view_edge.dst is self.view:
            viewed = state.memlet_path(view_edge)[0].src
            non_view_edges = state.out_edges(self.view)
            subset = view_edge.data.get_src_subset(view_edge, state)
            is_src = True
        else:
            viewed = state.memlet_path(view_edge)[-1].dst
            non_view_edges = state.in_edges(self.view)
            subset = view_edge.data.get_dst_subset(view_edge, state)
            is_src = False

        if subset is None:
            # `subset = None` means the entire viewed data container is used
            subset = subsets.Range.from_array(viewed.desc(sdfg))

        mapping, unsqueezed, squeezed = sdutil.map_view_to_array(desc, viewed.desc(sdfg), subset)

        # Update edges
        for edge in non_view_edges:
            # Update all memlets in tree
            for e in state.memlet_tree(edge):
                if e.data.data == self.view.data:
                    e.data.data = viewed.data

                    # Update subsets with instructions as follows:
                    #   * Dimensions in mapping are offset by view subset, size is taken from view subset
                    #   * Unsqueezed dimensions are ignored (should always be 0)
                    #   * Squeezed dimensions remain as they were in original subset
                    if e.data.subset is not None:
                        e.data.subset = self._offset_subset(mapping, subset, e.data.subset)
                    elif subset is not None:
                        # Fill in the subset from the original memlet
                        e.data.subset = copy.deepcopy(subset)
                        
                else:  # The memlet points to the other side, use ``other_subset``
                    if e.data.other_subset is not None:
                        e.data.other_subset = self._offset_subset(mapping, subset, e.data.other_subset)
                    elif subset is not None:
                        # Fill in the subset from the original memlet
                        e.data.other_subset = copy.deepcopy(subset)

                # NOTE: It's only necessary to modify one subset of the memlet, as the space of the other differs from
                #       the view space.


            # Remove edge directly adjacent to view and reconnect
            state.remove_edge(edge)
            if is_src:
                state.add_edge(view_edge.src, view_edge.src_conn, edge.dst, edge.dst_conn, edge.data)
            else:
                state.add_edge(edge.src, edge.src_conn, view_edge.dst, view_edge.dst_conn, edge.data)

        # Remove view node
        state.remove_node(self.view)

    def _offset_subset(self, mapping: Dict[int, int], subset: subsets.Range, edge_subset: subsets.Range):
        # Get offset and size from the space of the view to compose
        old_subset = edge_subset.min_element()
        old_size = edge_subset.size()

        # Create a new subset in the space of the data container from the offsets and sizes
        new_subset: List[Tuple[int, int, int]] = subset.ndrange()
        for vdim, adim in mapping.items():
            rb, re, rs = new_subset[adim]
            rb += old_subset[vdim]
            re = rb + old_size[vdim] - 1
            new_subset[adim] = (rb, re, rs)

        return subsets.Range(new_subset)
