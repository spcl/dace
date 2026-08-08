import dace
from dace.sdfg.state import MultiConnectorEdge
from dace.transformation import pass_pipeline as ppl
from typing import Optional
import copy
from dace.transformation.transformation import explicit_cf_compatible


@dace.properties.make_properties
@explicit_cf_compatible
class CleanAccessNodeToScalarSliceToTaskletPattern(ppl.Pass):
    """Clean up the frontend ``A -> A_slice -> tasklet`` pattern.

    The Python frontend lowers a scalar element read into an
    AccessNode -> AccessNode copy: ``A -[A[slice]]-> A_slice -[A_slice[0]]->
    tasklet``, where ``A_slice`` is a single-element transient. That
    AccessNode -> AccessNode copy is emitted as a ``CopyNDDynamic`` call,
    which is templated on a single element type — so when ``A`` and
    ``A_slice`` differ in dtype (a mixed-precision read) it fails to
    compile (``cannot convert 'A_slice' (double*) to 'const float*'``).

    Two cases, decided by whether ``A_slice`` is reused anywhere else in
    the read/write set (other states or interstate edges):

    - **Not reused** -> remove ``A_slice`` and wire ``A`` straight into
      the tasklet (``A -[A[slice]]-> tasklet``). The scalar then no longer
      appears in any read/write set.
    - **Reused** -> keep ``A_slice`` but replace the AccessNode ->
      AccessNode copy with an assignment tasklet
      (``A -[A[slice]]-> assign(`_out = _in`) -[A_slice[0]]-> A_slice``).
      The dtype change becomes an implicit C++ assignment cast in the
      tasklet body, which compiles. No map is introduced.

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

    def _check_pattern(self, state: dace.SDFGState, out_edge: MultiConnectorEdge[dace.Memlet],
                       access_node: dace.nodes.AccessNode):
        """Match ``access_node -> A_slice(scalar transient) -> tasklet``.

        :param state: State holding the candidate subgraph.
        :param out_edge: An out-edge of ``access_node`` (the copy edge).
        :param access_node: The source array AccessNode.
        :returns: ``(an1, an2, tasklet)`` on a structural match (reuse is
            decided separately in :meth:`_apply_recursive`), else
            ``(None, None, None)``.
        """
        sdfg = state.sdfg

        # The source feeding the scalar may be the array AccessNode (the
        # frontend's direct ``A -> A_slice``) or a MapEntry (after
        # ``LoopToMap``, the slice is staged through the map connector:
        # ``A -> MapEntry -> A_slice``). Both fold the same way — the read
        # subset + array name come from the edge memlet, not the source.
        if not isinstance(access_node, (dace.nodes.AccessNode, dace.nodes.MapEntry)):
            return None, None, None

        e1 = out_edge
        if e1.data is None or e1.data.subset is None:
            return None, None, None

        if e1.data.wcr is not None:
            return None, None, None

        an2 = e1.dst
        if not isinstance(an2, dace.nodes.AccessNode):
            return None, None, None

        desc = sdfg.arrays.get(an2.data)
        if desc is None or not desc.transient:
            return None, None, None

        if not (isinstance(desc, dace.data.Scalar) or
                (isinstance(desc, dace.data.Array) and len(desc.shape) == 1 and desc.total_size == 1)):
            return None, None, None
        if isinstance(desc, dace.data.View):
            return None, None, None

        if state.in_degree(an2) != 1 or state.out_degree(an2) != 1:
            return None, None, None

        e2 = state.out_edges(an2)[0]
        tasklet = e2.dst
        if not isinstance(tasklet, dace.nodes.Tasklet):
            return None, None, None

        if e2.data.subset != dace.subsets.Range([(0, 0, 1)]):
            return None, None, None

        # Refuse if the A_slice->tasklet edge carries WCR. Folding
        # this chain would replace the WCR-bearing scalar-read edge
        # with a plain array memlet, dropping any atomic / reduction
        # semantics. Symmetric with the inbound-WCR refusal in the
        # inverse pass.
        if e2.data.wcr is not None:
            return None, None, None

        return access_node, an2, tasklet

    def _scalar_reused_elsewhere(self, sdfg: dace.SDFG, scalar_name: str, this_state: dace.SDFGState) -> bool:
        """Return whether ``scalar_name`` appears in the read/write set of
        any state other than ``this_state``, or in any interstate edge.

        The structural match already guarantees the scalar's only uses in
        ``this_state`` are the copy write + the single tasklet read, so a
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

    def _slice_read(self, ie: MultiConnectorEdge[dace.Memlet], an1, an2: dace.nodes.AccessNode):
        """Recover ``(array_name, slice_subset)`` for the read into the
        scalar.

        The edge carries the array slice either as ``subset`` (memlet
        data == array, the usual ``A[slice]`` / ``A[i, j]`` form, valid
        for both an AccessNode and a MapEntry source) or as
        ``other_subset`` (memlet data == scalar; only for an AccessNode
        source, where the array name is the source node's data).

        :param ie: The ``src -> an2`` copy edge.
        :param an1: Source node (AccessNode or MapEntry).
        :param an2: Scalar transient AccessNode.
        :returns: ``(array_name, deep-copied slice subset)``, or
                  ``(None, None)`` when the read subset cannot be
                  recovered (memlet data is the scalar name but
                  ``other_subset`` is unset, and the source is not a
                  plain AccessNode whose own data we can use). The
                  caller skips the fold in that case.
        """
        if ie.data.data != an2.data:
            return ie.data.data, copy.deepcopy(ie.data.subset)
        if not isinstance(an1, dace.nodes.AccessNode) or ie.data.other_subset is None:
            return None, None
        return an1.data, copy.deepcopy(ie.data.other_subset)

    def _apply_recursive(self, sdfg: dace.SDFG) -> int:
        """Fold every matching pattern in ``sdfg`` and its nested SDFGs.

        :returns: Number of patterns folded.
        """
        folded = 0
        for state in list(sdfg.all_states()):
            pre_transform_state_nodes = list(state.nodes())
            for node in pre_transform_state_nodes:
                if node not in state.nodes():
                    continue
                if isinstance(node, dace.nodes.NestedSDFG):
                    folded += self._apply_recursive(node.sdfg)
                    continue
                for e in list(state.out_edges(node)):
                    an1, an2, tasklet = self._check_pattern(state, e, node)
                    if an1 is None or an2 is None or tasklet is None:
                        continue

                    ie = state.in_edges(an2)[0]
                    oe = state.out_edges(an2)[0]
                    assert oe.dst == tasklet
                    array_name, read_subset = self._slice_read(ie, an1, an2)
                    if array_name is None:
                        # _slice_read could not recover the read subset
                        # (memlet data == scalar name but
                        # ``other_subset`` is None and the source is
                        # not a plain AccessNode); skip rather than
                        # synthesise an incorrect subscript.
                        continue

                    # Refuse if ``array_name`` is also WRITTEN anywhere
                    # in this state (any AccessNode with ``in_degree >
                    # 0``). The intermediate scalar makes a gather +
                    # update sequence explicit; once folded, the body
                    # reads as ``arr[k] = arr[k] + ...`` which
                    # ``LoopToReduce`` mis-classifies as a single-cell
                    # reduction (TSVC s141's triangular
                    # ``flat_2d_array[k] += bb[j, i]`` with carried-scalar
                    # ``k`` is exactly this trap). Same-state read+write
                    # of ``array_name`` is the signal that the scalar
                    # gather + add + write-back is load-bearing for
                    # downstream matchers and the fold must not collapse
                    # it.
                    src_array_written_here = False
                    for n in state.data_nodes():
                        if n.data == array_name and state.in_degree(n) > 0 and n is not an1:
                            src_array_written_here = True
                            break
                    if src_array_written_here:
                        continue

                    reused = (not self.permissive) and self._scalar_reused_elsewhere(sdfg, an2.data, state)

                    if reused:
                        # Keep the scalar; replace the copy with an assignment
                        # tasklet that casts in its body. No map is introduced.
                        state.remove_edge(ie)
                        cast = state.add_tasklet(name=f"_assign_in_{array_name}_to_{an2.data}",
                                                 inputs={"_in"},
                                                 outputs={"_out"},
                                                 code="_out = _in")
                        state.add_edge(ie.src, ie.src_conn, cast, "_in",
                                       dace.memlet.Memlet(data=array_name, subset=read_subset))
                        state.add_edge(cast, "_out", an2, None,
                                       dace.memlet.Memlet(data=an2.data, subset=dace.subsets.Range([(0, 0, 1)])))
                    else:
                        # Not reused: drop the scalar and wire the source
                        # (array AccessNode or MapEntry connector) straight
                        # into the tasklet.
                        state.remove_node(an2)
                        state.add_edge(ie.src, ie.src_conn, oe.dst, oe.dst_conn,
                                       dace.memlet.Memlet(data=array_name, subset=read_subset))

                    folded += 1

        return folded

    def apply_pass(self, sdfg: dace.SDFG, _) -> Optional[int]:
        """Fold every ``A -> A_slice -> tasklet`` pattern in the SDFG hierarchy.

        :returns: Number of patterns folded, or ``None`` if none matched.
        """
        folded = self._apply_recursive(sdfg)
        return folded or None
