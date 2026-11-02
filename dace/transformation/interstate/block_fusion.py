# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.

from dace import symbolic
from dace.sdfg import InterstateEdge
from dace.sdfg import utils as sdutil
from dace.sdfg.state import AbstractControlFlowRegion, ControlFlowBlock, ControlFlowRegion, SDFGState
from dace.transformation import transformation
from dace.transformation.interstate.state_fusion import is_start_block, keep_start_block


def merge_assignments(first: dict[str, str], second: dict[str, str]) -> dict[str, str] | None:
    """Merge the assignments of two consecutive interstate edges onto one edge.

    One edge's assignments run in unspecified order (validation rejects reading a symbol the same
    edge writes), so a symbol written on either side and read on the other blocks the merge, in both
    directions. A left-hand-side collision is fine: ``second`` wins either way.

    :param first: Assignments of the edge entering the empty state.
    :param second: Assignments of the edge leaving it.
    :returns: The merged assignment dict, or ``None`` if merging is unsound.
    """
    if not second:
        return dict(first)
    if first:  # membership-only: no need to copy the keys into a set first
        # ``free_symbols_and_functions`` is the scan validation itself uses, so the guard
        # accepts exactly the merges validation accepts.
        for rhs in second.values():
            if any(s in first for s in symbolic.free_symbols_and_functions(rhs)):
                return None
        for lhs, rhs in first.items():
            if lhs in second:  # dropped by the overwrite below; never reaches the merged edge
                continue
            if any(s in second for s in symbolic.free_symbols_and_functions(rhs)):
                return None
    merged = dict(first)
    merged.update(second)
    return merged


def splice_empty_state(region: ControlFlowRegion, state: SDFGState, plain_edges_only: bool = False) -> bool:
    """Bypass the node-less ``state`` of ``region``, or drop it as a dead tail.

    Every predecessor of a single-successor state jumps straight to the successor, the assignments of both
    edges merged (see :func:`merge_assignments`). An empty sink goes with its in-edges, unless they assign:
    in a nested region an exit-edge symbol can outlive the region.

    :param region: The region holding ``state``.
    :param state: The empty state.
    :param plain_edges_only: Refuse unless every edge of ``state`` is unconditional and assignment-free.
    :returns: Whether ``state`` was removed.
    """
    in_e = region.in_edges(state)
    out_e = region.out_edges(state)
    if plain_edges_only and any(not e.data.is_unconditional() or e.data.assignments for e in in_e + out_e):
        return False
    is_start = is_start_block(region, state)
    if len(out_e) == 1 and (in_e or is_start):
        succ = out_e[0].dst
        # The successor edge's condition would have to be distributed over every predecessor edge, and nothing
        # carries its assignments out of a region entry without predecessors.
        if succ is state or not out_e[0].data.is_unconditional() or (not in_e and out_e[0].data.assignments):
            return False
        merged = [merge_assignments(e.data.assignments, out_e[0].data.assignments) for e in in_e]
        if any(m is None for m in merged):
            return False
        for e, assignments in zip(in_e, merged):
            region.add_edge(e.src, succ, InterstateEdge(condition=e.data.condition, assignments=assignments))
        for e in in_e + out_e:
            region.remove_edge(e)
        region.remove_node(state)
        if is_start:
            keep_start_block(region, succ)
        return True
    if not out_e and in_e and not is_start and not any(e.data.assignments for e in in_e):
        for e in in_e:
            region.remove_edge(e)
        region.remove_node(state)
        return True
    return False


@transformation.explicit_cf_compatible
class BlockFusion(transformation.MultiStateTransformation):
    """Implements the block-fusion transformation.

    Block-fusion takes two control flow blocks connected by a single edge, where at least one is a
    'no-op' block (an empty ``SDFGState``), and splices the no-op one out. It never merges two
    blocks that both carry semantics -- two loops or two conditionals are always refused -- so it
    reaches exactly the case ``StateFusion`` cannot: an empty state next to a LoopRegion or a
    ConditionalBlock, which are not ``SDFGState`` and so never match that transformation.
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
        """Whether ``block`` carries no semantics, so fusing it away changes nothing.

        Only an empty ``SDFGState`` qualifies. A bare ``ControlFlowBlock`` would too, but nothing
        constructs one outside the ``__DACE_dummy_sink`` scratch node ``dace.sdfg.analysis.cfg``
        adds and removes within a single call, so that case was dead and is gone. Every other
        block type -- regions, conditionals, break/continue/return -- carries semantics.
        """
        return isinstance(block, SDFGState) and block.is_empty()

    def can_be_applied(self, graph, expr_index, sdfg, permissive=False):
        # First block must have only one unconditional output edge (with dst the second block).
        out_edges = graph.out_edges(self.first_block)
        if len(out_edges) != 1 or out_edges[0].dst is not self.second_block or not out_edges[0].data.is_unconditional():
            return False
        # Inversely, the second block may only have one input edge, with src being the first block.
        in_edges_second = graph.in_edges(self.second_block)
        if len(in_edges_second) != 1 or in_edges_second[0].src is not self.first_block:
            return False

        # At least one of the two blocks must be a 'no-op' block, i.e. an empty SDFGState. A block
        # that carries semantics -- a loop, conditional, break, continue, region -- is never fused
        # away here.
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
        first_is_start = is_start_block(graph, self.first_block)
        connecting_edge = graph.edges_between(self.first_block, self.second_block)[0]
        assignments_to_absorb = connecting_edge.data.assignments
        graph.remove_edge(connecting_edge)
        for ie in graph.in_edges(self.first_block):
            if assignments_to_absorb:
                ie.data.assignments.update(assignments_to_absorb)

        if self._is_noop(self.first_block):
            # We remove the first block and let the second one remain. Resolve the pattern-node
            # handles BEFORE remove_node: they re-resolve by node id on every access, and removing
            # a node shifts the ids of everything after it.
            second_block = self.second_block
            for ie in graph.in_edges(self.first_block):
                graph.add_edge(ie.src, second_block, ie.data)
            graph.remove_node(self.first_block)
            if first_is_start:
                keep_start_block(graph, second_block)
        else:
            # We remove the second block and let the first one remain.
            for oe in graph.out_edges(self.second_block):
                graph.add_edge(self.first_block, oe.dst, oe.data)
            graph.remove_node(self.second_block)
