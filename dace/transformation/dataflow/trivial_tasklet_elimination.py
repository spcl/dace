# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" Contains classes that implement the trivial-tasklet-elimination transformation. """

from dace import data
from dace.sdfg import nodes
from dace.sdfg import utils as sdutil
from dace.transformation import transformation
from dace.properties import make_properties


def _is_carried_reduction_accumulator(sdfg, name: str) -> bool:
    """Report whether ``name`` is a transient scalar that carries a reduction
    across states -- it appears in more than one state and is the target of a
    write-conflict-resolution (WCR) edge somewhere.

    Such a scalar is a loop-carried accumulator: it is staged from an array
    element in one state (``w = A[i, j]``), reduced via a WCR in another
    (``w (+)= ...``) and written back (``A[i, j] = w``). The staging and
    write-back copies are the boundary that sequences the cross-state carry;
    eliminating either splices the array element straight onto the WCR-written
    scalar, which drops the accumulator's initial value / reduction ordering
    and miscompiles (polybench ludcmp's LU update collapses to zero).

    :param sdfg: The SDFG owning ``name``.
    :param name: The data descriptor name to classify.
    :returns: ``True`` if ``name`` is a cross-state WCR accumulator.
    """
    desc = sdfg.arrays.get(name)
    if desc is None or not desc.transient:
        return False
    if not (isinstance(desc, data.Scalar) or (isinstance(desc, data.Array) and desc.total_size == 1)):
        return False
    seen_states = 0
    is_wcr_target = False
    for sub in sdfg.all_sdfgs_recursive():
        if name not in sub.arrays:
            continue
        for state in sub.states():
            nodes_here = [n for n in state.nodes() if isinstance(n, nodes.AccessNode) and n.data == name]
            if not nodes_here:
                continue
            seen_states += 1
            if not is_wcr_target:
                is_wcr_target = any(e.data is not None and e.data.wcr is not None for n in nodes_here
                                    for e in state.in_edges(n))
    return seen_states > 1 and is_wcr_target


@make_properties
class TrivialTaskletElimination(transformation.SingleStateTransformation):
    """ Implements the Trivial-Tasklet Elimination pattern.

        Trivial-Tasklet Elimination removes tasklets that just copy the input
        to the output without WCR.
    """

    read = transformation.PatternNode(nodes.AccessNode)
    read_map = transformation.PatternNode(nodes.MapEntry)
    tasklet = transformation.PatternNode(nodes.Tasklet)
    write = transformation.PatternNode(nodes.AccessNode)
    write_map = transformation.PatternNode(nodes.MapExit)

    @classmethod
    def expressions(cls):
        return [
            sdutil.node_path_graph(cls.read, cls.tasklet, cls.write),
            sdutil.node_path_graph(cls.read_map, cls.tasklet, cls.write),
            sdutil.node_path_graph(cls.read, cls.tasklet, cls.write_map),
        ]

    def can_be_applied(self, graph, expr_index, sdfg, permissive=False):
        read = self.read_map if expr_index == 1 else self.read
        tasklet = self.tasklet
        write = self.write_map if expr_index == 2 else self.write
        if len(tasklet.in_connectors) != 1:
            return False
        if len(graph.in_edges(tasklet)) != 1:
            return False
        if len(tasklet.out_connectors) != 1:
            return False
        if len(graph.out_edges(tasklet)) != 1:
            return False
        in_conn = list(tasklet.in_connectors.keys())[0]
        out_conn = list(tasklet.out_connectors.keys())[0]
        if tasklet.code.as_string != f'{out_conn} = {in_conn}':
            return False
        read_memlet = graph.edges_between(read, tasklet)[0].data
        read_desc = sdfg.arrays[read_memlet.data]
        write_memlet = graph.edges_between(tasklet, write)[0].data
        if write_memlet.wcr:
            return False
        write_desc = sdfg.arrays[write_memlet.data]
        # Do not apply on streams
        if isinstance(read_desc, data.Stream):
            return False
        if isinstance(write_desc, data.Stream):
            return False
        # Keep the copy tasklet when the endpoints differ in dtype: the tasklet performs an
        # implicit cast that a plain memlet copy does not, so eliminating it silently drops the
        # conversion. This holds for the direct AccessNode -> tasklet -> AccessNode form
        # (expr_index 0) too, which is why the check is not gated on expr_index.
        if read_desc.dtype != write_desc.dtype:
            return False

        # expr_index == 2 (AccessNode -> tasklet -> MapExit): removing the tasklet splices
        # the read AccessNode directly onto the MapExit connector. Such an edge is valid
        # only if its memlet names that AccessNode, but the surviving stage-out memlet names
        # the outer mapped array -- so the merge would yield an invalid ``<scalar> -> MapExit``
        # edge whose ``memlet.data`` is the outer array (rejected by the SDFG validator and
        # StateFusionExtended's post-apply check). Keep the trivial copy tasklet at the map
        # boundary; it is exactly the shape InsertAssignTaskletsAtMapBoundary re-creates.
        if expr_index == 2 and write_memlet.data != read.data:
            return False

        # A copy bridging a cross-state reduction accumulator (a transient scalar
        # staged from an array, reduced via WCR, and written back) must survive:
        # eliminating it splices the array element straight onto the WCR-written
        # scalar and drops the accumulator's carry (ludcmp's LU update -> zero).
        if _is_carried_reduction_accumulator(sdfg, read_memlet.data) or \
                _is_carried_reduction_accumulator(sdfg, write_memlet.data):
            return False

        return True

    def apply(self, graph, sdfg):
        read = self.read_map if self.expr_index == 1 else self.read
        tasklet = self.tasklet
        write = self.write_map if self.expr_index == 2 else self.write

        in_edge = graph.edges_between(read, tasklet)[0]
        out_edge = graph.edges_between(tasklet, write)[0]
        graph.remove_edge(in_edge)
        graph.remove_edge(out_edge)
        if self.expr_index == 1 or (self.expr_index == 0 and self.read.data == self.write.data):
            # expr_index == 1 -- source is a MapEntry: the surviving edge leaves the
            # map's ``OUT_<read>`` connector, so its memlet must keep the read-side
            # data and subset (e.g. an offset access ``a[i + k]``) and carry the write
            # subset in ``other_subset``. Reusing the write memlet here would strand the
            # read offset in ``other_subset`` and drop it when the map is re-lowered.
            #
            # expr_index == 0 self-copy (``read.data == write.data``, e.g. the transpose
            # ``corr[i, j] -> corr[j, i]`` in correlation's symmetrize): the merged edge's
            # data name matches BOTH endpoints, so ``Memlet.try_initialize`` defaults
            # ``_is_data_src = True`` and treats ``subset`` as the SOURCE. Reusing the
            # WRITE memlet (whose ``subset`` is the destination) would read the destination
            # subset -- flipping the copy direction and clobbering the source data. Build
            # from the READ-side edge so ``subset`` is the source, ``other_subset`` the dst.
            in_edge.data.other_subset = out_edge.data.subset
            graph.add_edge(read, in_edge.src_conn, write, out_edge.dst_conn, in_edge.data)
        else:
            out_edge.data.other_subset = in_edge.data.subset
            graph.add_edge(read, in_edge.src_conn, write, out_edge.dst_conn, out_edge.data)
        graph.remove_node(tasklet)
