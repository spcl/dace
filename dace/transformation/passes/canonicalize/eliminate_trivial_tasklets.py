# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Remove every copy tasklet ``TrivialTaskletElimination`` accepts, in one traversal.

The traversal half of
:class:`~dace.transformation.dataflow.trivial_tasklet_elimination.TrivialTaskletElimination`.
All three of that transformation's patterns are a tasklet with exactly one in-edge and one
out-edge, so the candidates are the tasklets of a state and the endpoint types pick the pattern
index. Legality and the rewrite stay on the transformation.

Why not ``PatternApplyOnceEverywhere([TrivialTaskletElimination()])``: that wrapper re-derives
every match from scratch after each single application, one ``collapse_multigraph_to_nx`` plus a
VF2 subgraph isomorphism per state per sweep. Same reason as
:mod:`dace.transformation.passes.canonicalize.revert_nonreduction_wcr`, which this mirrors.
"""
from typing import Any, Dict, Iterator, Optional, Tuple

from dace import SDFG
from dace.sdfg import nodes
from dace.sdfg.state import SDFGState
from dace.transformation import pass_pipeline as ppl, transformation
from dace.transformation.dataflow.trivial_tasklet_elimination import TrivialTaskletElimination

#: ``{PatternNode: node}`` binding for one candidate.
Binding = Dict[Any, nodes.Node]


def trivial_tasklet_candidates(state: SDFGState) -> Iterator[Tuple[int, Binding]]:
    """Enumerate ``(expr_index, binding)`` for every single-in single-out tasklet in ``state``.

    The three ``TrivialTaskletElimination.expressions()`` entries are ``AccessNode -> Tasklet ->
    AccessNode`` (0), ``MapEntry -> Tasklet -> AccessNode`` (1) and ``AccessNode -> Tasklet ->
    MapExit`` (2); no other endpoint pair is a pattern. The arity gate is the transformation's
    own first check, applied here so a many-edged tasklet is never built into a candidate.

    :param state: The state to scan.
    :returns: Yields a pattern index and the node binding to match it with.
    """
    for node in state.nodes():
        if not isinstance(node, nodes.Tasklet):
            continue
        in_edges, out_edges = state.in_edges(node), state.out_edges(node)
        if len(in_edges) != 1 or len(out_edges) != 1:
            continue
        src, dst = in_edges[0].src, out_edges[0].dst
        if isinstance(dst, nodes.AccessNode):
            if isinstance(src, nodes.AccessNode):
                yield 0, {
                    TrivialTaskletElimination.read: src,
                    TrivialTaskletElimination.tasklet: node,
                    TrivialTaskletElimination.write: dst
                }
            elif isinstance(src, nodes.MapEntry):
                yield 1, {
                    TrivialTaskletElimination.read_map: src,
                    TrivialTaskletElimination.tasklet: node,
                    TrivialTaskletElimination.write: dst
                }
        elif isinstance(dst, nodes.MapExit) and isinstance(src, nodes.AccessNode):
            yield 2, {
                TrivialTaskletElimination.read: src,
                TrivialTaskletElimination.tasklet: node,
                TrivialTaskletElimination.write_map: dst
            }


@transformation.explicit_cf_compatible
class EliminateTrivialTasklets(ppl.Pass):
    """Apply ``TrivialTaskletElimination`` at every copy tasklet it accepts, to a fixpoint."""
    CATEGORY: str = 'Canonicalization'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Nodes | ppl.Modifies.Edges | ppl.Modifies.Memlets

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self):
        return {}

    def apply_pass(self, sdfg: SDFG, _pipeline_results: Dict[str, Any]) -> Optional[int]:
        """Eliminate every trivial copy tasklet in ``sdfg`` and its nested SDFGs.

        Outer fixpoint over the whole SDFG, not per state: the transformation refuses a copy that
        bridges a cross-state reduction accumulator, a whole-SDFG predicate, so a removal in one
        state can change the verdict in another. A sweep that changes nothing ends it.

        :param sdfg: The SDFG to transform in place.
        :returns: Number of eliminated tasklets, or ``None`` if none.
        """
        applied = 0
        changed = True
        while changed:
            changed = False
            for sd in sdfg.all_sdfgs_recursive():
                for state in sd.states():
                    count = self._eliminate_state(sd, state)
                    applied += count
                    changed = changed or count > 0
        return applied or None

    def _eliminate_state(self, sdfg: SDFG, state: SDFGState) -> int:
        """Eliminate every acceptable copy tasklet in one state, restarting after each removal.

        :param sdfg: The SDFG owning ``state``.
        :param state: The state to rewrite in place.
        :returns: Number of removals performed in ``state``.
        """
        applied = 0
        cfg_id = state.parent_graph.cfg_id
        state_id = state.block_id
        changed = True
        while changed:
            changed = False
            for expr_index, binding in trivial_tasklet_candidates(state):
                xform = TrivialTaskletElimination()
                xform.setup_match(sdfg, cfg_id, state_id, {k: state.node_id(v) for k, v in binding.items()}, expr_index)
                if not xform.can_be_applied(state, expr_index, sdfg):
                    continue
                xform.apply(state, sdfg)
                applied += 1
                changed = True
                break
        return applied
