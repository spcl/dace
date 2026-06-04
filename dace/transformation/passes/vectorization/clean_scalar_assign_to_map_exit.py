# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Pass that folds a redundant ``assign tasklet -> scalar -> MapExit`` chain.

Inside a Map scope the vectorization frontend sometimes stages a Map result
through a thread-local scalar transient: a producer writes into a trivial
``__out = __inp`` assignment tasklet, that tasklet writes a single-element
transient AccessNode, and the AccessNode flows straight into the enclosing
``MapExit``::

    ... -> assign_tasklet -> scalar_access_node -> MapExit

When ``scalar_access_node`` is a thread-local transient that is written
exactly once and read exactly once (i.e. it is not reused anywhere else in
the SDFG), the scalar and the assign tasklet are pure overhead: the producer
can write directly to the MapExit. This pass removes the scalar AccessNode and
the assign tasklet and rewires the producer's output edge straight to the
MapExit, preserving the outgoing memlet (data, subset) and the MapExit
connector.
"""
import copy
from typing import Optional, Tuple

import dace
from dace import properties
from dace.sdfg.state import MultiConnectorEdge
from dace.transformation import pass_pipeline as ppl, transformation


@properties.make_properties
@transformation.explicit_cf_compatible
class CleanScalarAssignToMapExit(ppl.Pass):
    """Remove a redundant ``producer -> assign tasklet -> scalar -> MapExit`` chain.

    The matched chain is, inside a single state and a single Map scope:

    - a producer node (Tasklet, AccessNode, MapEntry, NestedSDFG, ...) writes a
      trivial copy-assignment tasklet (``__out = __inp``);
    - that assign tasklet writes a thread-local single-element transient scalar
      AccessNode;
    - the scalar AccessNode flows straight into the enclosing ``MapExit``.

    When the scalar is a transient :class:`dace.data.Scalar` (or a length-1
    transient :class:`dace.data.Array`) written exactly once and read exactly
    once, with no other consumers anywhere in the SDFG (other states or
    interstate edges), the scalar and the assign tasklet are pure overhead. The
    pass removes both and rewires the producer's output edge straight to the
    MapExit, preserving the scalar -> MapExit memlet (data, subset) and the
    MapExit input connector.
    """

    # This pass is tested as part of the vectorization pipeline.
    CATEGORY: str = 'Vectorization'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Edges | ppl.Modifies.AccessNodes | ppl.Modifies.Tasklets | ppl.Modifies.Memlets

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def _is_assignment(self, tasklet: dace.nodes.Tasklet) -> bool:
        """Report whether ``tasklet`` is a single-input, single-output copy.

        Mirrors the assign-detection used by the sibling pass
        ``RemoveRedundantAssignments``: the body must be ``out = in`` over the
        tasklet's single in/out connectors.

        :param tasklet: Tasklet to inspect.
        :returns: ``True`` if the tasklet body is ``out = in``.
        """
        inc = tasklet.in_connectors
        outc = tasklet.out_connectors
        if len(inc) != 1 or len(outc) != 1:
            return False
        in_conn = list(inc)[0]
        out_conn = list(outc)[0]
        code = tasklet.code.as_string.strip()
        return code == f"{out_conn} = {in_conn}" or code == f"{out_conn} = {in_conn};"

    def _check_pattern(
        self, state: dace.SDFGState, tasklet: dace.nodes.Tasklet
    ) -> Optional[Tuple[MultiConnectorEdge[dace.Memlet], dace.nodes.AccessNode, MultiConnectorEdge[dace.Memlet],
                        dace.nodes.MapExit]]:
        """Match ``producer -> assign tasklet -> scalar transient -> MapExit``.

        :param state: State holding the candidate subgraph.
        :param tasklet: Candidate assignment tasklet.
        :returns: ``(producer_edge, scalar, scalar_to_exit_edge, map_exit)`` on
            a structural match, where ``producer_edge`` is the single in-edge of
            the assign tasklet; ``None`` if the pattern does not match.
        """
        sdfg = state.sdfg

        if not self._is_assignment(tasklet):
            return None

        in_edges = state.in_edges(tasklet)
        out_edges = state.out_edges(tasklet)
        if len(in_edges) != 1 or len(out_edges) != 1:
            return None

        producer_edge = in_edges[0]
        tasklet_to_scalar = out_edges[0]

        # WCR on either edge changes the semantics of the copy: refuse.
        if producer_edge.data is not None and producer_edge.data.wcr is not None:
            return None
        if tasklet_to_scalar.data is not None and tasklet_to_scalar.data.wcr is not None:
            return None

        scalar = tasklet_to_scalar.dst
        if not isinstance(scalar, dace.nodes.AccessNode):
            return None

        desc = sdfg.arrays.get(scalar.data)
        if desc is None or not desc.transient:
            return None
        if isinstance(desc, dace.data.View):
            return None
        if not (isinstance(desc, dace.data.Scalar) or
                (isinstance(desc, dace.data.Array) and len(desc.shape) == 1 and desc.total_size == 1)):
            return None

        # The scalar must be written exactly once (by this tasklet) and read
        # exactly once in this state.
        if state.in_degree(scalar) != 1 or state.out_degree(scalar) != 1:
            return None

        scalar_to_exit = state.out_edges(scalar)[0]
        map_exit = scalar_to_exit.dst
        if not isinstance(map_exit, dace.nodes.MapExit):
            return None

        if scalar_to_exit.data is None or scalar_to_exit.data.subset is None:
            return None
        if scalar_to_exit.data.wcr is not None:
            return None

        return producer_edge, scalar, scalar_to_exit, map_exit

    def _scalar_reused_elsewhere(self, sdfg: dace.SDFG, scalar_name: str, this_state: dace.SDFGState) -> bool:
        """Return whether ``scalar_name`` is read or written anywhere outside
        the matched chain.

        The structural match already guarantees the scalar's only uses in
        ``this_state`` are the single tasklet write and the single MapExit read,
        so a reuse can only show up as a data node in another state or as a
        symbol in an interstate edge's assignments / condition.

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

    def _apply_recursive(self, sdfg: dace.SDFG) -> int:
        """Fold every matching chain in ``sdfg`` and recurse into nested SDFGs.

        :param sdfg: SDFG to transform in place.
        :returns: The number of chains folded in ``sdfg`` and its descendants.
        """
        count = 0
        for state in list(sdfg.all_states()):
            for node in list(state.nodes()):
                if node not in state.nodes():
                    continue
                if isinstance(node, dace.nodes.NestedSDFG):
                    count += self._apply_recursive(node.sdfg)
                    continue
                if not isinstance(node, dace.nodes.Tasklet):
                    continue

                match = self._check_pattern(state, node)
                if match is None:
                    continue
                producer_edge, scalar, scalar_to_exit, map_exit = match

                if self._scalar_reused_elsewhere(sdfg, scalar.data, state):
                    continue

                # Preserve the scalar -> MapExit memlet and the MapExit input
                # connector before tearing the chain down.
                exit_memlet = copy.deepcopy(scalar_to_exit.data)
                exit_conn = scalar_to_exit.dst_conn

                src = producer_edge.src
                src_conn = producer_edge.src_conn

                # Removing the scalar AccessNode drops both its in-edge
                # (tasklet -> scalar) and its out-edge (scalar -> MapExit).
                state.remove_node(scalar)
                # The assign tasklet only had this one producer in-edge and the
                # now-deleted out-edge, so it is dead.
                state.remove_node(node)

                # Rewire the producer straight to the MapExit, keeping the
                # original destination memlet and connector.
                state.add_edge(src, src_conn, map_exit, exit_conn, exit_memlet)
                count += 1
        return count

    def apply_pass(self, sdfg: dace.SDFG, _) -> Optional[int]:
        """Run the pass over ``sdfg``.

        :param sdfg: SDFG to transform in place.
        :param _: Unused pipeline results.
        :returns: The number of chains folded, or ``None`` if none matched.
        """
        count = self._apply_recursive(sdfg)
        if count == 0:
            return None
        sdfg.validate()
        return count
