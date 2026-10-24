# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Expand the library nodes that only remap indices into their pure map, so map fusion can inline them.

A ``numpy.roll`` (``CShift``) or a ``broadcast_to`` (``Broadcast``) writes every element of its output from one
element of its input at a remapped index: no arithmetic, no reduction, nothing a vendor library would do faster.
Kept as a library node it reaches code generation as one full-size transient per call, because fusion cannot see
through a library node. An 8th-order Laplacian (laplacian_stencil_3d) rolls its grid 24 times into one sum: 24 live
copies of the grid, over the memory cap at the L preset. Expanded before maximal fusion, each remap is a gather map
that fuses into its consumer as an index expression, and the transient goes.

A ``numpy.where`` the frontend wires as a ``MergeLibraryNode`` is the same case: a per-element select, one map over
its output and nothing a library does faster. Kept, it is the one node of an element-wise chain fusion cannot enter,
so every operand and the result stay full-size transients and the chain is cut in two. ``velocity_tendencies``'
CFL clip ``np.where(lo_clip, -0.85 * h / dtime, np.where(hi_clip, 0.85 * h / dtime, zc))`` kept five such
``(nblks, nlev, nproma)`` temporaries, each allocated, written and re-read per call.

A ``Transpose`` of an operand with at most one non-unit dimension (``x.T`` of an ``(N, 1)`` column) only reshapes:
no locality for a library to win, and nbody's 20 per step each kept an ``N``-vector transient.
"""

from typing import Any

from dace import SDFG, properties
from dace.libraries.linalg.nodes.transpose import Transpose
from dace.libraries.standard.nodes.broadcast import Broadcast
from dace.libraries.standard.nodes.cshift import CShift
from dace.libraries.standard.nodes.merge_node import MergeLibraryNode
from dace.transformation import pass_pipeline as ppl
from dace.transformation import transformation

#: The library nodes whose only expansion is one gather map over their output.
INDEX_REMAPS: tuple[type, ...] = (CShift, Broadcast)

#: The library nodes whose only expansion is one element-wise select map over their output.
ELEMENTWISE_SELECTS: tuple[type, ...] = (MergeLibraryNode,)


def _is_reshape(state, node: Transpose) -> bool:
    """Whether the transposed operand has at most one non-unit dimension."""
    edges = state.in_edges(node)
    return len(edges) == 1 and sum(1 for s in edges[0].data.subset.size() if s != 1) <= 1


@properties.make_properties
@transformation.explicit_cf_compatible
class ExpandIndexRemaps(ppl.Pass):
    """Expand every :data:`INDEX_REMAPS` and :data:`ELEMENTWISE_SELECTS` node to its pure map (a nested SDFG the next
    inlining step dissolves)."""

    CATEGORY: str = "Canonicalization"

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Nodes | ppl.Modifies.Descriptors

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return bool(modified & ppl.Modifies.Nodes)

    def apply_pass(self, sdfg: SDFG, pipeline_results: dict[str, Any]) -> int | None:
        expanded = 0
        for sd in list(sdfg.all_sdfgs_recursive()):
            for state in sd.states():
                for node in [
                    n
                    for n in state.nodes()
                    if isinstance(n, INDEX_REMAPS + ELEMENTWISE_SELECTS)
                    or (isinstance(n, Transpose) and _is_reshape(state, n))
                ]:
                    # A CShift whose amount is still unknown has no map yet; it expands at code generation
                    if isinstance(node, CShift) and node.shift is None:
                        continue
                    node.implementation = "pure"
                    node.expand(sd, state)
                    expanded += 1
        return expanded or None

    def report(self, pass_retval: int) -> str:
        return f"Expanded {pass_retval} index-remap and element-wise select library nodes."
