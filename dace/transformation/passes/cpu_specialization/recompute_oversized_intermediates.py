# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Recompute a producer map inside its consumer map (``OTFMapFusion``) when that does not cost the CPU.

The CPU counterpart of :func:`~dace.transformation.passes.canonicalize.finalize.recompute_fuse_for_gpu`, with a
CPU cost model. A chain ``producer map -> transient -> consumer map`` is fused, and the transient deleted, when the
transient has exactly one reader in its SDFG and either

* every consumer iteration reads one distinct element exactly once, so nothing is recomputed, or
* the transient is not provably below the per-core L2 cache (a numeric size above it, or a symbolic size that does not
  name ``SMALL`` symbols only), and the producer is pointwise and cheap: one element per iteration, computed by
  tasklets and scalar access nodes only.

A device specialization, not a canonical form: the canonical graph keeps its materialized intermediates.
"""

from typing import Any

from dace import SDFG, data, properties, symbolic
from dace.sdfg import nodes
from dace.sdfg.sdfg import forget_facts, memoize_facts
from dace.sdfg.state import SDFGState
from dace.transformation import pass_pipeline as ppl
from dace.transformation.dataflow import OTFMapFusion
from dace.transformation.passes.cpu_specialization.machine import topology


def intermediate_outgrows_cache(sdfg: SDFG, name: str, facts: symbolic.Facts) -> bool:
    """Whether the array ``name`` is not provably below the host's per-core L2 cache.

    :param sdfg: The SDFG owning the descriptor.
    :param name: The intermediate transient's name.
    :param facts: ``sdfg``'s symbol facts (``SMALL`` symbols).
    :returns: False for a non-array, a numeric size at most the L2 size, or a size over ``SMALL`` symbols only.
    """
    desc = sdfg.arrays[name]
    if not isinstance(desc, data.Array):
        return False
    size = desc.total_size * desc.dtype.bytes
    if not symbolic.issymbolic(size):
        return int(size) > topology().l2_bytes
    return not symbolic.is_small(size, facts)


def consumer_reads(state: SDFGState, name: str, consumer: nodes.MapEntry) -> list:
    """The memlets inside ``consumer`` that read ``name``."""
    return [e.data for e in state.out_edges(consumer) if e.data.data == name]


def reads_each_element_once(reads: list, producer: nodes.MapEntry, consumer: nodes.MapEntry) -> bool:
    """Whether the consumer's iterations read distinct single elements, one access pattern, in the producer's
    dimensionality: the fused map then computes each element once."""
    if len(consumer.map.params) != len(producer.map.params):
        return False
    patterns = {tuple(memlet.subset.ranges) for memlet in reads}
    if len(patterns) != 1 or reads[0].subset.num_elements() != 1:
        return False
    return set(consumer.map.params) <= reads[0].subset.free_symbols


def pointwise_cheap_producer(state: SDFGState, sdfg: SDFG, producer: nodes.MapEntry, exit_node: nodes.MapExit) -> bool:
    """Whether each producer iteration writes its own single element (no WCR) from tasklets and scalar access nodes
    only."""
    params = set(producer.map.params)
    for edge in state.in_edges(exit_node):
        subset = edge.data.subset
        if subset is None or edge.data.wcr is not None or subset.num_elements() != 1:
            return False
        if not params <= subset.free_symbols:
            return False
    for node in state.scope_subgraph(producer, include_entry=False, include_exit=False).nodes():
        if isinstance(node, nodes.Tasklet):
            continue
        if not isinstance(node, nodes.AccessNode) or sdfg.arrays[node.data].total_size != 1:
            return False
    return True


def reader_counts(sdfg: SDFG) -> dict[str, int]:
    """Number of reading access nodes per data name over every state of ``sdfg``."""
    counts: dict[str, int] = {}
    for state in sdfg.states():
        for node in state.data_nodes():
            if state.out_degree(node) > 0:
                counts[node.data] = counts.get(node.data, 0) + 1
    return counts


@properties.make_properties
class RecomputeOversizedIntermediates(ppl.Pass):
    """Fuse producer->consumer map chains with ``OTFMapFusion`` where the CPU gate (module docstring) admits it."""

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
        oversized: dict[SDFG, dict[str, bool]] = {}
        applied = 0
        changed = True
        with memoize_facts():
            while changed:
                changed = False
                for sd in list(sdfg.all_sdfgs_recursive()):
                    readers = reader_counts(sd)
                    for state in sd.states():
                        while self.fuse_one(xform, sd, state, readers, oversized.setdefault(sd, {})):
                            applied += 1
                            changed = True
                            forget_facts(sd)
                            readers = reader_counts(sd)
        return applied or None

    def fuse_one(
        self,
        xform: OTFMapFusion,
        sdfg: SDFG,
        state: SDFGState,
        readers: dict[str, int],
        oversized: dict[str, bool],
    ) -> bool:
        """Fuse the first admitted chain of ``state``; whether one was fused. ``oversized`` caches
        :func:`intermediate_outgrows_cache` per name of ``sdfg``."""
        for node in state.data_nodes():
            desc = sdfg.arrays[node.data]
            if not desc.transient or readers.get(node.data, 0) != 1 or state.in_degree(node) != 1:
                continue
            exit_node = state.in_edges(node)[0].src
            consumers = {e.dst for e in state.out_edges(node)}
            if not isinstance(exit_node, nodes.MapExit) or len(consumers) != 1:
                continue
            consumer = next(iter(consumers))
            if not isinstance(consumer, nodes.MapEntry):
                continue
            producer = state.entry_node(exit_node)
            if not reads_each_element_once(consumer_reads(state, node.data, consumer), producer, consumer):
                if node.data not in oversized:
                    oversized[node.data] = intermediate_outgrows_cache(sdfg, node.data, sdfg.facts())
                if not oversized[node.data] or not pointwise_cheap_producer(state, sdfg, producer, exit_node):
                    continue
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
