# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.

from dace.sdfg import utils as sdutil
from dace.sdfg.state import AbstractControlFlowRegion, ControlFlowBlock, ControlFlowRegion, SDFGState
from dace.transformation import transformation


@transformation.explicit_cf_compatible
class BlockFusion(transformation.MultiStateTransformation):
    """ Implements the block-fusion transformation.

        Block-fusion takes two control flow blocks that are connected through a single edge, where either one or both
        blocks are 'no-op' control flow blocks, and fuses them into one.
    """

    first_block = transformation.PatternNode(ControlFlowBlock)
    second_block = transformation.PatternNode(ControlFlowBlock)

    @staticmethod
    def annotates_memlets():
        return False

    @classmethod
    def expressions(cls):
        return [sdutil.node_path_graph(cls.first_block, cls.second_block)]

    def _is_noop(self, block: ControlFlowBlock) -> bool:
        if isinstance(block, SDFGState):
            return block.is_empty()
        elif type(block) == ControlFlowBlock:
            return True
        return False

    def can_be_applied(self, graph, expr_index, sdfg, permissive=False):
        # First block must have only one unconditional output edge (with dst the second block).
        out_edges = graph.out_edges(self.first_block)
        if len(out_edges) != 1 or out_edges[0].dst is not self.second_block or not out_edges[0].data.is_unconditional():
            return False
        # Inversely, the second block may only have one input edge, with src being the first block.
        in_edges_second = graph.in_edges(self.second_block)
        if len(in_edges_second) != 1 or in_edges_second[0].src is not self.first_block:
            return False

        # Ensure that either that both blocks are fusable blocks, meaning that at least one of the two blocks must be
        # a 'no-op' block. That can be an empty SDFGState or a general control flow block without further semantics
        # (no loop, conditional, break, continue, control flow region, etc.).
        if not self._is_noop(self.first_block) and not self._is_noop(self.second_block):
            return False

        # The interstate edge may have assignments if there are input edges to the first block that can absorb them.
        in_edges = graph.in_edges(self.first_block)
        if out_edges[0].data.assignments:
            if not in_edges:
                return False
            # If the first block is a control flow region, no absorption is possible.
            if isinstance(self.first_block, AbstractControlFlowRegion):
                return False
            # Fail if symbol is set before the block to fuse
            new_assignments = set(out_edges[0].data.assignments.keys())
            if any((new_assignments & set(e.data.assignments.keys())) for e in in_edges):
                return False
            # Fail if symbol is used in the dataflow of that block
            if len(new_assignments & self.first_block.free_symbols) > 0:
                return False
            # Fail if symbols assigned on the first edge are free symbols on the second edge
            symbols_used = set(out_edges[0].data.free_symbols)
            for e in in_edges:
                if e.data.assignments.keys() & symbols_used:
                    return False
                # Also fail in the inverse; symbols assigned on the second edge are free symbols on the first edge
                if new_assignments & set(e.data.free_symbols):
                    return False

        # There can be no block that has output edges pointing to both the first and the second block. Such a case will
        # produce a multi-graph.
        for src, _, _ in in_edges:
            for _, dst, _ in graph.out_edges(src):
                if dst == self.second_block:
                    return False
        return True

    def apply(self, graph: ControlFlowRegion, sdfg):
        first_is_start = graph.start_block is self.first_block
        connecting_edge = graph.edges_between(self.first_block, self.second_block)[0]
        assignments_to_absorb = connecting_edge.data.assignments
        graph.remove_edge(connecting_edge)
        for ie in graph.in_edges(self.first_block):
            if assignments_to_absorb:
                ie.data.assignments.update(assignments_to_absorb)

        if self._is_noop(self.first_block):
            # We remove the first block and let the second one remain.
            for ie in graph.in_edges(self.first_block):
                graph.add_edge(ie.src, self.second_block, ie.data)
            if first_is_start:
                graph.start_block = self.second_block.block_id
            graph.remove_node(self.first_block)
        else:
            # We remove the second block and let the first one remain.
            for oe in graph.out_edges(self.second_block):
                graph.add_edge(self.first_block, oe.dst, oe.data)
            graph.remove_node(self.second_block)
