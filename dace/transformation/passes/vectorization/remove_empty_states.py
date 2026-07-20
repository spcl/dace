# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
"""Remove empty states, splicing their interstate edges and assignments together."""
import copy
from typing import Any, Dict, Optional
from dace import SDFG, InterstateEdge, properties
from dace.sdfg import nodes
from dace.sdfg.state import AbstractControlFlowRegion, ConditionalBlock
from dace.transformation import pass_pipeline as ppl, transformation


@properties.make_properties
@transformation.explicit_cf_compatible
class RemoveEmptyStates(ppl.Pass):
    """Remove empty states and splice their incident interstate edges.

    Tested as part of the vectorization pipeline.
    """

    CATEGORY: str = 'Vectorization'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.AccessNodes | ppl.Modifies.InterstateEdges | ppl.Modifies.Tasklets | ppl.Modifies.Edges

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self):
        return {}

    def _apply(self, sdfg: SDFG) -> int:
        """Remove empty states recursively, merging assignments across the spliced edge.

        :param sdfg: The SDFG to transform in place.
        :returns: Number of empty states removed, including those in nested SDFGs.
        """
        removed = 0
        for state in sdfg.all_states():
            # If empty state
            if len(state.nodes()) != 0:
                continue
            g = state.parent_graph

            # With structured controlflow this should not happen but check in case
            if len(g.in_edges(state)) > 1:
                continue
            if len(g.out_edges(state)) > 1:
                continue

            # Case distinction depending on if it is the starting block or not
            if len(g.in_edges(state)) == 0:
                if len(g.out_edges(state)) == 1:
                    out_edge = g.out_edges(state)[0]
                    if out_edge.data.assignments == dict():
                        # Remove state, remove edge too
                        dst_node = out_edge.dst
                        dst_node_out_edges = g.out_edges(dst_node)
                        g.remove_node(state)
                        g.remove_node(dst_node)
                        g.add_node(dst_node, is_start_block=True)
                        for e in dst_node_out_edges:
                            g.add_edge(dst_node, e.dst, copy.deepcopy(e.data))
                        removed += 1
            else:
                if len(g.in_edges(state)) == 1:
                    if len(g.out_edges(state)) == 1:
                        in_edge = g.in_edges(state)[0]
                        out_edge = g.out_edges(state)[0]
                        if out_edge.data.assignments == dict():
                            joined_assignments = copy.deepcopy(in_edge.data.assignments)
                            joined_assignments.update(out_edge.data.assignments)
                            src = in_edge.src
                            dst = out_edge.dst
                            g.remove_node(state)
                            g.add_edge(src, dst, InterstateEdge(assignments=joined_assignments))
                            removed += 1

        # A region's start block is stored as an integer index (``_start_block``) with a
        # cached object (``_cached_start_block``); ``remove_node`` maintains neither. Removing
        # a state shifts the surviving node indices, and an earlier CFG rewrite (branch
        # lowering) can leave the cache pointing at a node we just removed. The ``start_block``
        # property returns the stale cache before consulting ``source_nodes()``, so
        # ``validate``'s ``dfs_edges(start_block)`` then dereferences a removed node. Re-derive
        # every region's start block from its unique source node so it stays consistent.
        for region in sdfg.all_control_flow_regions(recursive=False):
            self._repair_start_block(region)

        for state in sdfg.all_states():
            for node in state.nodes():
                if isinstance(node, nodes.NestedSDFG):
                    removed += self._apply(node.sdfg)

        return removed

    def _repair_start_block(self, region: AbstractControlFlowRegion) -> None:
        """Re-establish ``region``'s start block from its unique source node.

        No-op for a :class:`ConditionalBlock` (its children are branches, not a linear graph)
        and for an empty region. When the source node is unambiguous (the well-formed case
        after empty-state splicing) it becomes the start block; an ambiguous region keeps its
        explicit start index but drops a dangling cache.

        :param region: The control-flow region whose start block to repair.
        """
        if isinstance(region, ConditionalBlock):
            return
        block_nodes = region.nodes()
        if not block_nodes:
            return
        sources = region.source_nodes()
        if len(sources) == 1:
            region._cached_start_block = sources[0]
            region._start_block = region.node_id(sources[0])
        elif region._cached_start_block is not None and region._cached_start_block not in block_nodes:
            region._cached_start_block = None

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> Optional[int]:
        """Remove all empty states from the SDFG and validate the result.

        :param sdfg: The SDFG to transform in place.
        :param pipeline_results: Results from previously run passes (unused).
        :returns: Number of empty states removed, or ``None`` if none were.
        """
        removed = self._apply(sdfg)
        return removed or None
