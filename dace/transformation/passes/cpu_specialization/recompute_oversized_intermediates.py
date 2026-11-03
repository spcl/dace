# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Recompute a producer map inside its consumer maps (``OTFMapFusion``) wherever that is legal and deletes the
intermediate.

The CPU counterpart of :func:`~dace.transformation.passes.canonicalize.finalize.recompute_fuse_for_gpu`. A chain
``producer map -> transient -> consumer maps`` is fused when every reader of the transient is a map reading it from
that one access node: each consumer recomputes the producer in registers, which is cheaper than storing and loading
the intermediate (a 5-point stencil of a stencil becomes one map), and the transient is deleted.

A device specialization, not a canonical form: the canonical graph keeps its materialized intermediates.
"""

from typing import Any

from dace import SDFG, properties
from dace.sdfg import nodes
from dace.sdfg.sdfg import forget_facts, memoize_facts
from dace.sdfg.state import SDFGState
from dace.transformation import pass_pipeline as ppl
from dace.transformation.dataflow import OTFMapFusion


def reader_counts(sdfg: SDFG) -> dict[str, int]:
    """Number of reading access nodes per data name over every state of ``sdfg``."""
    counts: dict[str, int] = {}
    for state in sdfg.states():
        for node in state.data_nodes():
            if state.out_degree(node) > 0:
                counts[node.data] = counts.get(node.data, 0) + 1
    return counts


@properties.make_properties
class RecomputeIntermediates(ppl.Pass):
    """Fuse producer->consumer map chains with ``OTFMapFusion`` wherever the module docstring's contract holds."""

    CATEGORY: str = "CPU Specialization"

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Nodes | ppl.Modifies.Memlets | ppl.Modifies.Descriptors

    def should_reapply(self, _modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self):
        return set()

    def apply_pass(self, sdfg: SDFG, _pipeline_results: dict[str, Any]) -> int | None:
        """Apply to fixpoint; returns how many chains were collapsed, or ``None``. Does not validate."""
        xform = OTFMapFusion()
        applied = 0
        changed = True
        with memoize_facts(sdfg):
            while changed:
                changed = False
                for sd in list(sdfg.all_sdfgs_recursive()):
                    readers = reader_counts(sd)
                    for state in sd.states():
                        while self.fuse_one(xform, sd, state, readers):
                            applied += 1
                            changed = True
                            forget_facts(sd)
                            readers = reader_counts(sd)
        return applied or None

    def fuse_one(self, xform: OTFMapFusion, sdfg: SDFG, state: SDFGState, readers: dict[str, int]) -> bool:
        """Fuse the first legal chain of ``state`` into one of its consumers; whether one was fused."""
        for node in state.data_nodes():
            desc = sdfg.arrays[node.data]
            if not desc.transient or readers.get(node.data, 0) != 1 or state.in_degree(node) != 1:
                continue
            exit_node = state.in_edges(node)[0].src
            consumers = list(dict.fromkeys(e.dst for e in state.out_edges(node)))
            if not isinstance(exit_node, nodes.MapExit) or not consumers:
                continue
            if not all(isinstance(c, nodes.MapEntry) for c in consumers):
                continue
            for consumer in consumers:
                xform.setup_match(
                    sdfg,
                    state.parent_graph.cfg_id,
                    state.block_id,
                    {
                        OTFMapFusion.first_map_exit: exit_node,
                        OTFMapFusion.array: node,
                        OTFMapFusion.second_map_entry: consumer,
                    },
                    0,
                    override=True,
                )
                if not xform.can_be_applied(state, 0, sdfg):
                    continue
                name = node.data
                xform.apply(state, sdfg)
                # The intermediate's descriptor would otherwise still be allocated at codegen.
                if not any(n.data == name for st in sdfg.states() for n in st.data_nodes()):
                    sdfg.remove_data(name, validate=False)
                return True
        return False
