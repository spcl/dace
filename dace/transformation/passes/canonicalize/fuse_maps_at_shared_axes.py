# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Fuse a producer map into a consumer map that iterates over a subset of the producer's axes.

``for i, k: T[i, k] = f(i, k)`` followed by ``for i: g(T[i, 4], T[i, 7])`` becomes
``for i: { for k: t[k] = f(i, k); g(t[4], t[7]) }``: the producer splits into an outer map over the axes the
consumer shares (equal ranges) and an inner map over the rest, and the outer map fuses vertically with the consumer.
A split that does not fuse is merged back into the original map.
"""

from typing import Any

from dace import SDFG, SDFGState, properties
from dace.sdfg import nodes
from dace.sdfg import utils as sdutil
from dace.sdfg.sdfg import forget_all_facts
from dace.transformation import pass_pipeline as ppl
from dace.transformation.dataflow.map_expansion import MapExpansion
from dace.transformation.dataflow.map_fusion_vertical import MapFusionVertical
from dace.transformation.passes import analysis as ap


def shared_axes(producer: nodes.Map, consumer: nodes.Map) -> list[str] | None:
    """The producer parameters, in the consumer's parameter order, that iterate the consumer's ranges; ``None`` unless
    every consumer parameter has one and the producer keeps at least one axis of its own."""
    if len(consumer.params) >= len(producer.params):
        return None
    ranges = list(zip(producer.params, producer.range.ranges))
    chosen: list[str] = []
    for crange in consumer.range.ranges:
        match = next((p for p, r in ranges if p not in chosen and r == crange), None)
        if match is None:
            return None
        chosen.append(match)
    return chosen


def reorder(entry: nodes.MapEntry, params: list[str]) -> None:
    """Permute ``entry``'s map parameters (and their ranges) into ``params`` order."""
    order = [entry.map.params.index(p) for p in params]
    entry.map.range.ranges = [entry.map.range.ranges[i] for i in order]
    entry.map.range.tile_sizes = [entry.map.range.tile_sizes[i] for i in order]
    entry.map.params = list(params)


def candidates(state: SDFGState) -> list[tuple[nodes.MapEntry, nodes.AccessNode, nodes.MapEntry, list[str]]]:
    """Top-level ``(producer entry, transient, consumer entry, shared axes)`` triples of ``state``."""
    sdict = state.scope_dict()
    found = []
    for node in state.data_nodes():
        if sdict[node] is not None or not node.desc(state.sdfg).transient:
            continue
        for ie in state.in_edges(node):
            if not isinstance(ie.src, nodes.MapExit):
                continue
            producer = state.entry_node(ie.src)
            if sdict[producer] is not None:
                continue
            for oe in state.out_edges(node):
                if not isinstance(oe.dst, nodes.MapEntry) or sdict[oe.dst] is not None:
                    continue
                axes = shared_axes(producer.map, oe.dst.map)
                if axes is not None:
                    found.append((producer, node, oe.dst, axes))
    return found


@properties.make_properties
class FuseMapsAtSharedAxes(ppl.Pass):
    """Fuse each top-level producer map with a consumer map over a subset of its axes (module docstring); the
    vertical fusion is :class:`MapFusionVertical` with ``FuseMaps``' canonicalization settings."""

    CATEGORY: str = "Simplification"

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Scopes | ppl.Modifies.AccessNodes | ppl.Modifies.Memlets

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return bool(modified & (ppl.Modifies.Scopes | ppl.Modifies.AccessNodes | ppl.Modifies.Memlets))

    def depends_on(self):
        return [ap.FindSingleUseData]

    def apply_pass(self, sdfg: SDFG, pipeline_results: dict[str, Any]) -> int | None:
        """:returns: The number of producer maps fused, or ``None`` when none was."""
        if ap.FindSingleUseData.__name__ not in pipeline_results:
            pipeline_results = dict(pipeline_results)
            pipeline_results[ap.FindSingleUseData.__name__] = ap.FindSingleUseData().apply_pass(sdfg, {})
        fusion = MapFusionVertical(allow_intermediates_with_symbolic_size=True)
        fused = 0
        for sd in sdfg.all_sdfgs_recursive():
            for state in sd.states():
                # Keyed by transient and consumer: a merge-back rebuilds the producer's map, never these two
                tried: set[tuple[nodes.AccessNode, nodes.MapEntry]] = set()
                progress = True
                while progress:
                    progress = False
                    for producer, node, consumer, axes in candidates(state):
                        if (node, consumer) in tried:
                            continue
                        restored = self.try_fuse(sd, state, producer, node, consumer, axes, fusion, pipeline_results)
                        if restored is None:
                            fused += 1
                        else:
                            tried.add((node, consumer))
                        # Either way the producer's entry was rebuilt: the remaining candidates name the old one
                        progress = True
                        break
        return fused or None

    @staticmethod
    def try_fuse(
        sdfg: SDFG,
        state: SDFGState,
        producer: nodes.MapEntry,
        node: nodes.AccessNode,
        consumer: nodes.MapEntry,
        axes: list[str],
        fusion: MapFusionVertical,
        pipeline_results: dict[str, Any],
    ) -> nodes.MapEntry | None:
        """Split ``producer`` at ``axes`` and fuse its outer map with ``consumer`` through ``node``: ``None`` when
        fused, else the entry of the producer merged back as it was."""
        original = list(producer.map.params)
        collapse = producer.map.collapse
        rest = [p for p in original if p not in axes]
        reorder(producer, axes + rest)
        expansion = MapExpansion()
        expansion.setup_match(
            sdfg, state.parent_graph.cfg_id, state.block_id, {MapExpansion.map_entry: producer}, 0, override=True
        )
        expansion.expansion_limit = len(axes)
        entries = expansion.apply(state, sdfg)
        outer, inner = entries[0], entries[-1]
        # Expansion leaves one map per shared axis; they become one outer map again.
        for middle in entries[1 : len(axes)]:
            outer, _ = sdutil.merge_maps(state, outer, state.exit_node(outer), middle, state.exit_node(middle))
        fusion.setup_match(
            sdfg,
            state.parent_graph.cfg_id,
            state.block_id,
            {
                MapFusionVertical.first_map_exit: state.exit_node(outer),
                MapFusionVertical.array: node,
                MapFusionVertical.second_map_entry: consumer,
            },
            0,
            override=True,
        )
        fusion._pipeline_results = pipeline_results
        if fusion.can_be_applied(state, 0, sdfg):
            fusion.apply(state, sdfg)
            forget_all_facts()
            return None
        merged, _ = sdutil.merge_maps(state, outer, state.exit_node(outer), inner, state.exit_node(inner))
        reorder(merged, original)
        merged.map.collapse = collapse
        return merged
