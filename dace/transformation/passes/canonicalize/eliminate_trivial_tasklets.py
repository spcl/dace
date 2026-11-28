# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Remove every copy tasklet ``TrivialTaskletElimination`` accepts, in one traversal."""

from collections.abc import Iterator
from typing import Any

from dace import SDFG
from dace.sdfg import nodes
from dace.sdfg.state import SDFGState
from dace.transformation import pass_pipeline as ppl
from dace.transformation import transformation
from dace.transformation.dataflow.trivial_tasklet_elimination import TrivialTaskletElimination
from dace.transformation.passes.canonicalize.revert_nonreduction_wcr import Binding, apply_at_candidates


def trivial_tasklet_candidates(state: SDFGState) -> Iterator[tuple[int, Binding]]:
    """Enumerate ``(expr_index, binding)`` for every single-in single-out tasklet in ``state``."""
    for node in state.nodes():
        if not isinstance(node, nodes.Tasklet):
            continue
        in_edges, out_edges = state.in_edges(node), state.out_edges(node)
        if len(in_edges) != 1 or len(out_edges) != 1:
            continue
        src, dst = in_edges[0].src, out_edges[0].dst
        if isinstance(dst, nodes.AccessNode):
            if isinstance(src, nodes.AccessNode):
                yield (
                    0,
                    {
                        TrivialTaskletElimination.read: src,
                        TrivialTaskletElimination.tasklet: node,
                        TrivialTaskletElimination.write: dst,
                    },
                )
            elif isinstance(src, nodes.MapEntry):
                yield (
                    1,
                    {
                        TrivialTaskletElimination.read_map: src,
                        TrivialTaskletElimination.tasklet: node,
                        TrivialTaskletElimination.write: dst,
                    },
                )
        elif isinstance(dst, nodes.MapExit) and isinstance(src, nodes.AccessNode):
            yield (
                2,
                {
                    TrivialTaskletElimination.read: src,
                    TrivialTaskletElimination.tasklet: node,
                    TrivialTaskletElimination.write_map: dst,
                },
            )


@transformation.explicit_cf_compatible
class EliminateTrivialTasklets(ppl.Pass):
    """Apply ``TrivialTaskletElimination`` at every copy tasklet it accepts, to a fixpoint."""

    CATEGORY: str = "Canonicalization"

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Nodes | ppl.Modifies.Edges | ppl.Modifies.Memlets

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self) -> list[type[ppl.Pass] | ppl.Pass]:
        return []

    def apply_pass(self, sdfg: SDFG, _pipeline_results: dict[str, Any]) -> int | None:
        """Eliminate every trivial copy tasklet in ``sdfg`` and its nested SDFGs."""
        return apply_at_candidates(sdfg, TrivialTaskletElimination, trivial_tasklet_candidates) or None
