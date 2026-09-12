# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Revert every conflict-free WCR back to an explicit augmented assignment.

The traversal half of :class:`~dace.transformation.dataflow.wcr_conversion.WCRToAugAssign`.
All six of that transformation's patterns are anchored on a WCR-carrying edge, so the
candidates can be enumerated by walking the WCR edges of a state directly -- which is what
this pass does. Legality and the rewrite itself stay entirely on the transformation, so the
pass and the transformation can never disagree.

Why not ``PatternApplyOnceEverywhere([WCRToAugAssign()])``: that wrapper re-derives every
match from scratch after each single application -- ``collapse_multigraph_to_nx`` plus a VF2
subgraph isomorphism per state per sweep. On CloudSC's terminal ``revert_nonreduction_wcr``
stage that is 396 states and 368 applications, i.e. 165494 graph collapses to find 368 sites,
and it is 31% of the whole canonicalization. Six WCR edges are cheaper to look at than one
isomorphism.
"""
from typing import Any, Dict, Iterator, Optional, Tuple

from dace import SDFG
from dace.sdfg import nodes
from dace.sdfg.state import SDFGState
from dace.transformation import pass_pipeline as ppl, transformation
from dace.transformation.dataflow.wcr_conversion import WCRToAugAssign

#: ``{PatternNode: node}`` binding for one candidate.
Binding = Dict[Any, nodes.Node]


def _exit_outputs(state: SDFGState, map_exit: nodes.MapExit, data: str) -> Iterator[nodes.AccessNode]:
    """The access nodes ``map_exit`` writes ``data`` into.

    A map exit with several outputs has one valid ``output`` binding per array, and nothing in
    the path pattern ties it to the array the WCR edge writes -- ``WCRToAugAssign.can_be_applied``
    refuses the mismatched ones (CloudSC's flux band writes four arrays through one exit), so
    filtering here is the same set of candidates, found without building them.

    :param state: The state holding the map exit.
    :param map_exit: The map exit whose outputs to enumerate.
    :param data: The array name the WCR edge writes.
    :returns: Yields each matching access node.
    """
    for edge in state.out_edges(map_exit):
        if isinstance(edge.dst, nodes.AccessNode) and edge.dst.data == data:
            yield edge.dst


def wcr_candidates(state: SDFGState) -> Iterator[Tuple[int, Binding]]:
    """Enumerate ``(expr_index, binding)`` for every WCR edge in ``state``.

    The edge's endpoint types select the pattern: the six ``WCRToAugAssign.expressions()``
    entries are ``Tasklet -> AccessNode`` (0), ``Tasklet -> MapExit -> AccessNode`` (1),
    ``AccessNode -> AccessNode`` (2), ``AccessNode -> MapExit -> AccessNode`` (3), the same
    topology as 1 with the WCR stranded on the outer ``MapExit -> AccessNode`` edge (4), and
    that one again with a body ``NestedSDFG`` in place of the tasklet (5).

    :param state: The state to scan.
    :returns: Yields a pattern index and the node binding to match it with.
    """
    for edge in state.edges():
        if edge.data.wcr is None:
            continue
        src, dst = edge.src, edge.dst
        if isinstance(dst, nodes.AccessNode):
            if isinstance(src, nodes.Tasklet):
                yield 0, {WCRToAugAssign.tasklet: src, WCRToAugAssign.output: dst}
            elif isinstance(src, nodes.AccessNode):
                yield 2, {WCRToAugAssign.inp: src, WCRToAugAssign.output: dst}
            elif isinstance(src, nodes.MapExit):
                # expr 4: the producer inside the map is a bare tasklet, and its edge to the
                # exit is the WCR-free precise write. The transformation checks both.
                for inner in state.in_edges(src):
                    if isinstance(inner.src, nodes.Tasklet):
                        yield 4, {
                            WCRToAugAssign.tasklet: inner.src,
                            WCRToAugAssign.map_exit: src,
                            WCRToAugAssign.output: dst
                        }
        elif isinstance(dst, nodes.MapExit):
            for out in _exit_outputs(state, dst, edge.data.data):
                if isinstance(src, nodes.Tasklet):
                    yield 1, {WCRToAugAssign.tasklet: src, WCRToAugAssign.map_exit: dst, WCRToAugAssign.output: out}
                elif isinstance(src, nodes.AccessNode):
                    yield 3, {WCRToAugAssign.inp: src, WCRToAugAssign.map_exit: dst, WCRToAugAssign.output: out}
                elif isinstance(src, nodes.NestedSDFG):
                    yield 5, {WCRToAugAssign.nested: src, WCRToAugAssign.map_exit: dst, WCRToAugAssign.output: out}


@transformation.explicit_cf_compatible
class RevertNonReductionWCR(ppl.Pass):
    """Apply ``WCRToAugAssign`` at every WCR site it accepts, to a fixpoint."""
    CATEGORY: str = 'Canonicalization'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Nodes | ppl.Modifies.Edges | ppl.Modifies.Memlets | ppl.Modifies.Descriptors

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self):
        return {}

    def apply_pass(self, sdfg: SDFG, _pipeline_results: Dict[str, Any]) -> Optional[int]:
        """Revert every conflict-free WCR in ``sdfg`` and its nested SDFGs.

        Outer fixpoint over the whole SDFG, not per state: ``WCRToAugAssign`` decides expr 4 and
        expr 5 partly on ``boundary_write_index_syms``, a whole-SDFG scan, so a rewrite in one
        state can in principle change the verdict in another. A sweep that changes nothing ends it.

        :param sdfg: The SDFG to transform in place.
        :returns: Number of reverted WCRs, or ``None`` if none.
        """
        applied = 0
        changed = True
        while changed:
            changed = False
            for sd in sdfg.all_sdfgs_recursive():
                for state in sd.states():
                    count = self._revert_state(sd, state)
                    applied += count
                    changed = changed or count > 0
        return applied or None

    def _revert_state(self, sdfg: SDFG, state: SDFGState) -> int:
        """Revert every acceptable WCR site in one state, restarting after each rewrite.

        :param sdfg: The SDFG owning ``state``.
        :param state: The state to rewrite in place.
        :returns: Number of rewrites performed in ``state``.
        """
        applied = 0
        cfg_id = state.parent_graph.cfg_id
        state_id = state.block_id
        changed = True
        while changed:
            changed = False
            for expr_index, binding in wcr_candidates(state):
                xform = WCRToAugAssign()
                xform.setup_match(sdfg, cfg_id, state_id, {k: state.node_id(v) for k, v in binding.items()}, expr_index)
                if not xform.can_be_applied(state, expr_index, sdfg):
                    continue
                xform.apply(state, sdfg)
                applied += 1
                changed = True
                break
        return applied
