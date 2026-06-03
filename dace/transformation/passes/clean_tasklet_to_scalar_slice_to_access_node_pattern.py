import dace
from dace.sdfg.state import MultiConnectorEdge
from dace.transformation import pass_pipeline as ppl
from typing import Optional
import copy
from dace.transformation.transformation import explicit_cf_compatible


@dace.properties.make_properties
@explicit_cf_compatible
class CleanTaskletToScalarSliceToAccessNodePattern(ppl.Pass):
    """Clean up the frontend ``tasklet -> A_slice -> A`` pattern (the output-side
    inverse of :class:`CleanAccessNodeToScalarSliceToTaskletPattern`).

    The Python frontend lowers a scalar-valued expression's write as
    ``tasklet -[A_slice[0]]-> A_slice -[A[slice]]-> A``, where ``A_slice`` is
    a single-element transient. The intermediate scalar is materialised
    purely as a staging point for the binop result; once
    ``TrivialTaskletElimination`` has already folded any explicit
    ``_out = _in`` copy tasklet that previously sat between ``A_slice`` and
    ``A``, the chain remains as a bare ``tasklet -> A_slice -> A`` AN-AN
    memlet copy that prevents ``AugAssignToWCR``'s
    ``input -> tasklet -> output`` (expr_index 0) shape from matching the
    accumulator chain.

    Two cases, decided by whether ``A_slice`` is reused anywhere else in
    the read/write set (other states or interstate edges):

    - **Not reused** -> remove ``A_slice`` and wire the tasklet output
      directly into ``A`` (``tasklet -[A[slice]]-> A``).
    - **Reused** -> keep ``A_slice`` but replace the AN-AN copy with an
      assignment tasklet (``tasklet -[A_slice[0]]-> assign(`_out = _in`)
      -[A[slice]]-> A``), so any dtype change becomes an implicit C++
      assignment cast in the tasklet body. No map is introduced.

    ``permissive=True`` ignores the reuse check and always removes the
    scalar (the caller asserts the scalar is dead elsewhere).
    """

    permissive = dace.properties.Property(
        dtype=bool, default=False, desc="If permissive the pass does not check if scalar is used in other states")

    def __init__(self, permissive: bool = False):
        self.permissive = permissive

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.AccessNodes | ppl.Modifies.Memlets | ppl.Modifies.Tasklets

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def _check_pattern(self, state: dace.SDFGState, in_edge: MultiConnectorEdge[dace.Memlet],
                       sink_node: dace.nodes.Node):
        """Match ``tasklet -> A_slice(scalar transient) -> sink_node``.

        ``sink_node`` may be the destination array AccessNode (the
        frontend's direct ``A_slice -> A``) or a MapExit (after
        ``LoopToMap``, the slice is staged through the map connector:
        ``A_slice -> MapExit -> A``). Both fold the same way -- the write
        subset + destination array name come from the edge memlet, not
        the sink.

        :param state: State holding the candidate subgraph.
        :param in_edge: The ``A_slice -> sink_node`` copy edge.
        :param sink_node: The destination AccessNode (or MapExit).
        :returns: ``(tasklet, an_slice, sink_node)`` on a structural match
            (reuse is decided separately in :meth:`_apply_recursive`),
            else ``(None, None, None)``.
        """
        sdfg = state.sdfg

        if not isinstance(sink_node, (dace.nodes.AccessNode, dace.nodes.MapExit)):
            return None, None, None

        e2 = in_edge
        if e2.data is None or e2.data.subset is None:
            return None, None, None

        if e2.data.wcr is not None:
            return None, None, None

        an_slice = e2.src
        if not isinstance(an_slice, dace.nodes.AccessNode):
            return None, None, None

        desc = sdfg.arrays.get(an_slice.data)
        if desc is None or not desc.transient:
            return None, None, None

        if not (isinstance(desc, dace.data.Scalar) or
                (isinstance(desc, dace.data.Array) and len(desc.shape) == 1 and desc.total_size == 1)):
            return None, None, None
        if isinstance(desc, dace.data.View):
            return None, None, None

        if state.in_degree(an_slice) != 1 or state.out_degree(an_slice) != 1:
            return None, None, None

        e1 = state.in_edges(an_slice)[0]
        tasklet = e1.src
        if not isinstance(tasklet, dace.nodes.Tasklet):
            return None, None, None

        # The tasklet -> A_slice edge must write the whole transient (subset 0).
        if e1.data.subset != dace.subsets.Range([(0, 0, 1)]):
            return None, None, None

        # Refuse if the tasklet->A_slice edge carries WCR. Folding
        # would collapse ``tasklet -> A_slice -> sink`` to a single
        # ``tasklet -> sink`` edge whose memlet comes from the
        # outgoing side, dropping any atomic / reduction-style
        # semantics on the inbound write. (TSVC s141 exposes this: a
        # triangular ``flat_2d_array[k] += bb[j, i]`` with a
        # carried-scalar update produces an SDFG where this fold can
        # otherwise misclassify a scattered update as a single-cell
        # reduction once ``LoopToReduce`` sees the collapsed shape.)
        if e1.data.wcr is not None:
            return None, None, None

        return tasklet, an_slice, sink_node

    def _scalar_reused_elsewhere(self, sdfg: dace.SDFG, scalar_name: str, this_state: dace.SDFGState) -> bool:
        """Return whether ``scalar_name`` appears in the read/write set of
        any state other than ``this_state``, or in any interstate edge.

        The structural match already guarantees the scalar's only uses in
        ``this_state`` are the single tasklet write + the copy read, so a
        reuse can only be in another state's data nodes or an interstate
        edge's assignments / condition.

        :param sdfg: SDFG to scan.
        :param scalar_name: The scalar transient's data name.
        :param this_state: The state the pattern was matched in (excluded).
        :returns: ``True`` if the scalar is read or written elsewhere.
        """
        for state in sdfg.all_states():
            if state is this_state:
                continue
            for dn in state.data_nodes():
                if dn.data == scalar_name:
                    return True
        for ise in sdfg.all_interstate_edges():
            if any(str(s) == scalar_name for s in ise.data.free_symbols):
                return True
        return False

    def _slice_write(self, oe: MultiConnectorEdge[dace.Memlet], an_slice: dace.nodes.AccessNode, sink_node):
        """Recover ``(array_name, slice_subset)`` for the write out of
        the scalar.

        The edge carries the destination array slice either as ``subset``
        (memlet data == array, the usual ``A[slice]`` / ``A[i, j]`` form,
        valid for both an AccessNode and a MapExit sink) or as
        ``other_subset`` (memlet data == scalar; only for an AccessNode
        sink, where the array name is the destination node's data).

        :param oe: The ``an_slice -> sink_node`` copy edge.
        :param an_slice: Scalar transient AccessNode.
        :param sink_node: Sink node (AccessNode or MapExit).
        :returns: ``(array_name, deep-copied slice subset)``, or
                  ``(None, None)`` when the write subset cannot be
                  recovered (memlet data is the scalar name but
                  ``other_subset`` is unset, and the sink is not a
                  plain AccessNode whose own data we can use). The
                  caller skips the fold in that case.
        """
        if oe.data.data != an_slice.data:
            return oe.data.data, copy.deepcopy(oe.data.subset)
        if not isinstance(sink_node, dace.nodes.AccessNode) or oe.data.other_subset is None:
            return None, None
        return sink_node.data, copy.deepcopy(oe.data.other_subset)

    def _safe_to_fold(self, state: dace.SDFGState, an_slice: dace.nodes.AccessNode, dest_data: str, sink_node) -> bool:
        """In-state safety check: refuse the fold when the destination's
        data is ALSO read in this state through a different AccessNode
        with ``out_degree > 0``.

        The matcher would otherwise expose ``AugAssignToWCR``'s
        ``expr_index == 0`` latent hole: that shape requires
        ``out_degree(input) == 1``, and folding the
        ``tasklet -> A_slice -> A`` chain into ``tasklet -> A`` exposes
        that ``A`` is read through a different chain (e.g. TSVC s3112's
        ``sum += a[i]; b[i] = sum`` where ``sum`` is read both by the
        RMW Add and by ``b_i_assign``). Refuse the fold here so AAW
        never sees the unsafe shape.

        The destination's own AccessNode (``sink_node`` when it's an
        AccessNode) is excluded from the check: it's the legitimate
        target of the fold, not a sibling reader.

        :param state: The state to scan.
        :param an_slice: The intermediate slice AccessNode being folded.
        :param dest_data: The destination array's data name.
        :param sink_node: The sink endpoint (AccessNode or MapExit) of
                          the fold; excluded from the scan.
        :returns: ``False`` to refuse the fold.
        """
        for n in state.nodes():
            if not isinstance(n, dace.nodes.AccessNode):
                continue
            if n.data != dest_data:
                continue
            if n is sink_node or n is an_slice:
                continue
            if state.out_degree(n) > 0:
                return False
        return True

    def _apply_recursive(self, sdfg: dace.SDFG):
        for state in list(sdfg.all_states()):
            pre_transform_state_nodes = list(state.nodes())
            for node in pre_transform_state_nodes:
                if node not in state.nodes():
                    continue
                if isinstance(node, dace.nodes.NestedSDFG):
                    self._apply_recursive(node.sdfg)
                    continue
                for e in list(state.in_edges(node)):
                    tasklet, an_slice, sink = self._check_pattern(state, e, node)
                    if tasklet is None or an_slice is None or sink is None:
                        continue

                    ie = state.in_edges(an_slice)[0]
                    oe = state.out_edges(an_slice)[0]
                    assert ie.src == tasklet
                    array_name, write_subset = self._slice_write(oe, an_slice, sink)
                    if array_name is None:
                        # _slice_write could not recover the write
                        # subset (memlet data == scalar name but
                        # ``other_subset`` is None and the sink is
                        # not a plain AccessNode); skip rather than
                        # synthesise an incorrect subscript.
                        continue

                    if not self._safe_to_fold(state, an_slice, array_name, sink):
                        continue

                    reused = (not self.permissive) and self._scalar_reused_elsewhere(sdfg, an_slice.data, state)

                    if reused:
                        # Keep the scalar; replace the copy with an assignment
                        # tasklet that casts in its body. No map is introduced.
                        state.remove_edge(oe)
                        cast = state.add_tasklet(name=f"_assign_out_{an_slice.data}_to_{array_name}",
                                                 inputs={"_in"},
                                                 outputs={"_out"},
                                                 code="_out = _in")
                        state.add_edge(an_slice, None, cast, "_in",
                                       dace.memlet.Memlet(data=an_slice.data, subset=dace.subsets.Range([(0, 0, 1)])))
                        state.add_edge(cast, "_out", oe.dst, oe.dst_conn,
                                       dace.memlet.Memlet(data=array_name, subset=write_subset))
                    else:
                        # Not reused: drop the scalar and wire the tasklet
                        # output straight into the sink (array AccessNode or
                        # MapExit connector).
                        state.remove_node(an_slice)
                        state.add_edge(ie.src, ie.src_conn, oe.dst, oe.dst_conn,
                                       dace.memlet.Memlet(data=array_name, subset=write_subset))

    def apply_pass(self, sdfg: dace.SDFG, _) -> Optional[int]:
        self._apply_recursive(sdfg)
