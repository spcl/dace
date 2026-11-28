# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Revert every conflict-free WCR back to an explicit augmented assignment."""

from collections.abc import Callable, Iterator
from typing import Any

from dace import SDFG
from dace.sdfg import nodes
from dace.sdfg.state import SDFGState
from dace.transformation import pass_pipeline as ppl
from dace.transformation import transformation
from dace.transformation.dataflow.wcr_conversion import WCRToAugAssign

#: ``{PatternNode: node}`` binding for one candidate.
Binding = dict[Any, nodes.Node]


def _exit_outputs(state: SDFGState, map_exit: nodes.MapExit, data: str) -> Iterator[nodes.AccessNode]:
    """The access nodes ``map_exit`` writes ``data`` into."""
    for edge in state.out_edges(map_exit):
        if isinstance(edge.dst, nodes.AccessNode) and edge.dst.data == data:
            yield edge.dst


def wcr_candidates(state: SDFGState) -> Iterator[tuple[int, Binding]]:
    """Enumerate ``(expr_index, binding)`` for every WCR edge in ``state``."""
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
                        yield (
                            4,
                            {
                                WCRToAugAssign.tasklet: inner.src,
                                WCRToAugAssign.map_exit: src,
                                WCRToAugAssign.output: dst,
                            },
                        )
        elif isinstance(dst, nodes.MapExit):
            assert edge.data.data is not None, "an edge into a map exit carries a value"
            for out in _exit_outputs(state, dst, edge.data.data):
                if isinstance(src, nodes.Tasklet):
                    yield 1, {WCRToAugAssign.tasklet: src, WCRToAugAssign.map_exit: dst, WCRToAugAssign.output: out}
                elif isinstance(src, nodes.AccessNode):
                    yield 3, {WCRToAugAssign.inp: src, WCRToAugAssign.map_exit: dst, WCRToAugAssign.output: out}
                elif isinstance(src, nodes.NestedSDFG):
                    yield 5, {WCRToAugAssign.nested: src, WCRToAugAssign.map_exit: dst, WCRToAugAssign.output: out}


#: Enumerates ``(expr_index, binding)`` candidates in one state.
CandidateFn = Callable[[SDFGState], Iterator[tuple[int, Binding]]]


def apply_at_candidates(sdfg: SDFG, xform_type: type, candidates: CandidateFn) -> int:
    """Apply ``xform_type`` at every candidate it accepts, over ``sdfg`` and its nested SDFGs, to a fixpoint."""
    applied = 0
    changed = True
    while changed:
        changed = False
        for sd in sdfg.all_sdfgs_recursive():
            for state in sd.states():
                cfg_id = state.parent_graph.cfg_id
                state_id = state.block_id
                restart = True
                while restart:
                    restart = False
                    for expr_index, binding in candidates(state):
                        xform = xform_type()
                        ids = {k: state.node_id(v) for k, v in binding.items()}
                        xform.setup_match(sd, cfg_id, state_id, ids, expr_index)
                        if not xform.can_be_applied(state, expr_index, sd):
                            continue
                        xform.apply(state, sd)
                        applied += 1
                        changed = restart = True
                        break
    return applied


@transformation.explicit_cf_compatible
class RevertNonReductionWCR(ppl.Pass):
    """Apply ``WCRToAugAssign`` at every WCR site it accepts, to a fixpoint."""

    CATEGORY: str = "Canonicalization"

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Nodes | ppl.Modifies.Edges | ppl.Modifies.Memlets | ppl.Modifies.Descriptors

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self) -> list[type[ppl.Pass] | ppl.Pass]:
        return []

    def apply_pass(self, sdfg: SDFG, _pipeline_results: dict[str, Any]) -> int | None:
        """Revert every conflict-free WCR in ``sdfg`` and its nested SDFGs."""
        return apply_at_candidates(sdfg, WCRToAugAssign, wcr_candidates) or None
