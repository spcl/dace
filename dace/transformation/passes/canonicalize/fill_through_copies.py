# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A copy out of an array that a constant fill wrote in the same state is a fill of the copy's destination.

``i_new = np.zeros(n); i_old = np.copy(i_new); j_new = np.copy(i_new)`` (warpx_esirkepov_deposition) reaches the
form as one fill and copies of its result; each copy becomes its own fill, so the destination no longer reads
``i_new`` and dead-fill elimination and fusion treat every array alone.
"""

from typing import Any

from dace import SDFG, Memlet, nodes
from dace.libraries.standard.nodes.fill.node import FillLibraryNode
from dace.sdfg.state import SDFGState
from dace.transformation import pass_pipeline as ppl
from dace.transformation import transformation


def constant_fill_of(state: SDFGState, access: nodes.AccessNode) -> tuple[FillLibraryNode, Memlet] | None:
    """The constant fill that is the only writer of ``access`` in ``state``, with the memlet it writes."""
    writes = [e for e in state.in_edges(access) if not e.data.is_empty()]
    if len(writes) != 1 or not isinstance(writes[0].src, FillLibraryNode) or writes[0].data.wcr is not None:
        return None
    fill = writes[0].src
    if fill.value_edge(state) is not None:
        return None
    return fill, writes[0].data


def fill_copies(sdfg: SDFG, state: SDFGState) -> int:
    """Replace every copy out of a filled access node of ``state`` by a fill of its destination."""
    replaced = 0
    for access in state.data_nodes():
        found = constant_fill_of(state, access)
        if found is None:
            continue
        fill, filled = found
        for edge in list(state.out_edges(access)):
            dst = edge.dst
            if not isinstance(dst, nodes.AccessNode) or dst.data == access.data:
                continue
            if edge.data.is_empty() or edge.data.wcr is not None:
                continue
            # A copy into another dtype converts the constant; the fill would store it unconverted.
            if sdfg.arrays[dst.data].dtype != sdfg.arrays[access.data].dtype:
                continue
            src_subset = edge.data.get_src_subset(edge, state)
            dst_subset = edge.data.get_dst_subset(edge, state)
            # A copy memlet naming one side only reads and writes the same subset on both.
            src_subset = src_subset if src_subset is not None else dst_subset
            dst_subset = dst_subset if dst_subset is not None else src_subset
            if src_subset is None:
                continue
            if src_subset.num_elements() != dst_subset.num_elements():
                continue
            if not filled.subset.covers_precise(src_subset, sdfg.facts()):
                continue
            new_fill = FillLibraryNode(f"{fill.label}_{dst.data}", value=fill.value)
            state.add_node(new_fill)
            # Keeps the destination after whatever the copy waited for besides the fill.
            for before in state.in_edges(fill):
                if before.data.is_empty():
                    state.add_nedge(before.src, new_fill, Memlet())
            state.add_edge(
                new_fill,
                FillLibraryNode.OUTPUT_CONNECTOR_NAME,
                dst,
                edge.dst_conn,
                Memlet(data=dst.data, subset=dst_subset),
            )
            state.remove_edge(edge)
            replaced += 1
    return replaced


@transformation.explicit_cf_compatible
class FillThroughCopies(ppl.Pass):
    """Turn every copy of a constant-filled array, in the fill's state, into a fill of the copy's destination."""

    CATEGORY: str = "Canonicalization"

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Nodes | ppl.Modifies.Edges

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return bool(modified & ppl.Modifies.Nodes)

    def apply_pass(self, sdfg: SDFG, pipeline_results: dict[str, Any]) -> int | None:
        """:returns: How many copies became fills, or ``None`` if none did."""
        replaced = sum(fill_copies(sd, state) for sd in sdfg.all_sdfgs_recursive() for state in sd.states())
        return replaced or None
