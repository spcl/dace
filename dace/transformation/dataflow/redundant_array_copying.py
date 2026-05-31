# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" Contains redundant array removal transformations. """

from dace import subsets
from dace.sdfg import nodes
from dace.sdfg import utils as sdutil
from dace.sdfg.sdfg import SDFG
from dace.sdfg.state import SDFGState
from dace.transformation import transformation as pm


def _is_full_copy(graph: SDFGState, edge, src_desc, dst_desc) -> bool:
    """Whether ``edge`` copies the whole source array onto the whole destination array.

    A side whose subset is ``None`` is treated as covering the full extent of
    that side's descriptor; the edge is a full identity iff both effective
    subsets equal ``Range.from_array(<desc on that side>)``.
    """
    src_subset = edge.data.get_src_subset(edge, graph)
    dst_subset = edge.data.get_dst_subset(edge, graph)
    src_ok = src_subset is None or src_subset == subsets.Range.from_array(src_desc)
    dst_ok = dst_subset is None or dst_subset == subsets.Range.from_array(dst_desc)
    return src_ok and dst_ok


def _shapes_match(a, b) -> bool:
    """Whether two shape tuples have the same rank and equal symbolic extents."""
    return len(a) == len(b) and all(x == y for x, y in zip(a, b))


class RedundantArrayCopyingIn(pm.SingleStateTransformation):
    """Fold an ``A -> B -> C`` chain of full identity copies into writers-of-``A`` writing straight to ``C``.

    Matches three sequential AccessNodes where ``A`` and ``B`` are transient.
    ``apply`` removes ``A`` and ``B`` and redirects every writer of ``A`` onto
    ``C``, renaming the memlet data so the redirected edges describe ``C``.

    The fold is only sound when ``B`` has exactly one consumer (``C``), ``A``
    and ``C`` share rank, shape and storage, and both copy edges are full
    identity. A partial copy in the chain would be silently widened to a full
    one by the rename and corrupt the region of ``C`` the chain never wrote.
    """

    in_array = pm.PatternNode(nodes.AccessNode)
    med_array = pm.PatternNode(nodes.AccessNode)
    out_array = pm.PatternNode(nodes.AccessNode)

    @classmethod
    def expressions(cls):
        return [sdutil.node_path_graph(cls.in_array, cls.med_array, cls.out_array)]

    def can_be_applied(self, graph, expr_index, sdfg, permissive=False):
        in_array = self.in_array
        med_array = self.med_array
        out_array = self.out_array
        in_desc = in_array.desc(sdfg)
        med_desc = med_array.desc(sdfg)
        out_desc = out_array.desc(sdfg)

        # Degree gates: ``in`` and ``med`` are about to be removed, so each
        # must have a single outgoing edge into the chain. ``med`` having any
        # other consumer would leave it dangling without a source after the fold.
        if graph.out_degree(in_array) != 1:
            return False
        if graph.in_degree(med_array) != 1 or graph.out_degree(med_array) != 1:
            return False

        # ``in`` and ``med`` are the nodes the fold deletes; only transients
        # can be deleted without losing externally-visible storage.
        if not (in_desc.transient and med_desc.transient):
            return False

        # The redirected writers of ``in`` keep their original subsets and end
        # up writing to ``out``; this is only meaningful if ``in`` and ``out``
        # share storage location (so the writers can address it the same way)
        # and the three arrays share rank and shape (so the subsets remain
        # valid). ``med`` may live on a different device (this is exactly the
        # CPU-GPU-CPU staging chain the pass is designed to short-circuit).
        if in_desc.storage != out_desc.storage:
            return False
        if not (_shapes_match(in_desc.shape, med_desc.shape) and _shapes_match(in_desc.shape, out_desc.shape)):
            return False

        # The two copy edges in the chain must be full identity copies. A
        # partial copy here would be silently widened to a full one when the
        # writers of ``in`` are redirected onto ``out``, corrupting the region
        # of ``out`` the chain never actually wrote. The unique edges are
        # guaranteed by the degree gates above; fetch the first.
        in_med = graph.edges_between(in_array, med_array)[0]
        med_out = graph.edges_between(med_array, out_array)[0]
        if not _is_full_copy(graph, in_med, in_desc, med_desc):
            return False
        if not _is_full_copy(graph, med_out, med_desc, out_desc):
            return False

        # ``apply`` redirects the writers of ``in_array`` straight onto
        # ``out_array`` keeping their original subsets. That is only sound if
        # the whole array flows through the chain unchanged -- i.e. both copies
        # are full identity copies and the middle array shares the shape.
        # Otherwise a partial / offset copy would be silently widened to a full
        # one, corrupting the region the chain never actually wrote.
        in_desc = in_array.desc(sdfg)
        med_desc = med_array.desc(sdfg)
        if len(med_desc.shape) != len(in_desc.shape) or any(i != m for i, m in zip(in_desc.shape, med_desc.shape)):
            return False
        in_med = graph.edges_between(in_array, med_array)
        med_out = graph.edges_between(med_array, out_array)
        if len(in_med) != 1 or len(med_out) != 1:
            return False
        if not (_is_full_copy(graph, in_med[0], in_desc, med_desc)
                and _is_full_copy(graph, med_out[0], med_desc, out_array.desc(sdfg))):
            return False

        return True

    def apply(self, graph: SDFGState, sdfg: SDFG):
        in_array = self.in_array
        med_array = self.med_array
        out_array = self.out_array

        # Modify all edges that point to in_array to point to out_array
        for in_edge in graph.in_edges(in_array):

            # Make all memlets that write to in_array write to out_array instead
            tree = graph.memlet_tree(in_edge)
            for te in tree:
                if te.data.data == in_array.data:
                    te.data.data = out_array.data

            # Redirect edge to in_array
            graph.remove_edge(in_edge)
            graph.add_edge(in_edge.src, in_edge.src_conn, out_array, None, in_edge.data)

        graph.remove_node(med_array)
        graph.remove_node(in_array)


class RedundantArrayCopying(pm.SingleStateTransformation):
    """ Implements the redundant array removal transformation. Removes the last access node
        in pattern A -> B -> A, and the second (if possible)
    """

    in_array = pm.PatternNode(nodes.AccessNode)
    med_array = pm.PatternNode(nodes.AccessNode)
    out_array = pm.PatternNode(nodes.AccessNode)

    @classmethod
    def expressions(cls):
        return [sdutil.node_path_graph(cls.in_array, cls.med_array, cls.out_array)]

    def can_be_applied(self, graph, expr_index, sdfg, permissive=False):
        in_array = self.in_array
        med_array = self.med_array
        out_array = self.out_array

        # Ensure out degree is one (only one target, which is out_array)
        if graph.out_degree(in_array) != 1:
            return False

        # Make sure that the removal candidate is a transient variable
        if not permissive and not out_array.desc(sdfg).transient:
            return False

        # Make sure that the middle access node is not transient. We do this to ensure that everything copied from
        # B -> A is either copied in from A, or uninitialized memory.
        if not permissive and not med_array.desc(sdfg).transient:
            return False

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
        if len(in_array.desc(sdfg).shape) != len(out_array.desc(sdfg).shape) or any(
                i != o for i, o in zip(in_array.desc(sdfg).shape,
                                       out_array.desc(sdfg).shape)):
            return False

        return True

    def apply(self, graph, sdfg):
        in_array = self.in_array
        med_array = self.med_array
        out_array = self.out_array

        med_edges = len(graph.out_edges(med_array))
        med_out_edges = 0
        for med_e in graph.out_edges(med_array):
            if med_e.dst == out_array:
                # Modify all outcoming edges to point to in_array
                for out_e in graph.out_edges(med_e.dst):
                    path = graph.memlet_path(out_e)
                    for pe in path:
                        if pe.data.data == out_array.data or pe.data.data == med_array.data:
                            pe.data.data = in_array.data
                    # Redirect edge to in_array
                    graph.remove_edge(out_e)
                    graph.add_edge(in_array, out_e.src_conn, out_e.dst, out_e.dst_conn, out_e.data)
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


class RedundantArrayCopying2(pm.SingleStateTransformation):
    """ Implements the redundant array removal transformation. Removes
        multiples of array B in pattern A -> B.
    """
    in_array = pm.PatternNode(nodes.AccessNode)
    out_array = pm.PatternNode(nodes.AccessNode)

    @classmethod
    def expressions(cls):
        return [sdutil.node_path_graph(cls.in_array, cls.out_array)]

    def can_be_applied(self, graph, expr_index, sdfg, permissive=False):
        in_array = self.in_array
        out_array = self.out_array

        # Ensure out degree is one (only one target, which is out_array)
        found = 0
        for _, _, dst, _, _ in graph.out_edges(in_array):
            if (isinstance(dst, nodes.AccessNode) and dst != out_array and dst.data == out_array.data):
                found += 1

        return found > 0

    def apply(self, graph, sdfg):
        in_array = self.in_array
        out_array = self.out_array

        for e1 in graph.out_edges(in_array):
            dst = e1.dst
            if (isinstance(dst, nodes.AccessNode) and dst != out_array and dst.data == out_array.data):
                for e2 in graph.out_edges(dst):
                    graph.add_edge(out_array, None, e2.dst, e2.dst_conn, e2.data)
                    graph.remove_edge(e2)
                graph.remove_edge(e1)
                graph.remove_node(dst)


class RedundantArrayCopying3(pm.SingleStateTransformation):
    """ Implements the redundant array removal transformation. Removes multiples
        of array B in pattern MapEntry -> B.
    """

    map_entry = pm.PatternNode(nodes.MapEntry)
    out_array = pm.PatternNode(nodes.AccessNode)

    @classmethod
    def expressions(cls):
        return [sdutil.node_path_graph(cls.map_entry, cls.out_array)]

    def can_be_applied(self, graph, expr_index, sdfg, permissive=False):
        map_entry = self.map_entry
        out_array = self.out_array

        # Ensure out degree is one (only one target, which is out_array)
        found = 0
        for _, _, dst, _, _ in graph.out_edges(map_entry):
            if (isinstance(dst, nodes.AccessNode) and dst != out_array and dst.data == out_array.data):
                found += 1

        return found > 0

    def apply(self, graph, sdfg):
        map_entry = self.map_entry
        out_array = self.out_array

        for e1 in graph.out_edges(map_entry):
            dst = e1.dst
            if (isinstance(dst, nodes.AccessNode) and dst != out_array and dst.data == out_array.data):
                for e2 in graph.out_edges(dst):
                    graph.add_edge(out_array, None, e2.dst, e2.dst_conn, e2.data)
                    graph.remove_edge(e2)
                graph.remove_edge(e1)
                graph.remove_node(dst)
