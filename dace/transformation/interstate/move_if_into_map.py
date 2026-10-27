# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Transformation that pushes a conditional block through one level of map nesting.

``map i: { guard; if c: { map j: body } }`` becomes ``map i: { map j: { guard; if c: body } }``, so the two maps become
neighbours that fusion and ``MapCollapse`` can combine. The guard is everything that computes ``c``: a linear chain of
states that only write transients the condition reads (``if_cond = levmask[i]``) and the interstate assignments on its
edges.
"""

import copy
from dataclasses import dataclass

from dace import memlet as mm
from dace import sdfg as sd
from dace.properties import CodeBlock
from dace.sdfg import utils as sdutil
from dace.sdfg.graph import SubgraphView
from dace.sdfg.nodes import AccessNode, MapEntry, NestedSDFG, Tasklet
from dace.sdfg.sdfg import InterstateEdge
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion, SDFGState
from dace.transformation import helpers as xfh
from dace.transformation import transformation


@dataclass(slots=True)
class GuardMatch:
    """What :class:`MoveIfIntoMap` moves: the guard chain and the names it reads, writes and assigns."""

    enclosing: sd.SDFG
    branch_state: SDFGState
    chain: list[SDFGState]
    reads: set[str]
    locals: set[str]
    assigned: set[str]
    symbols: set[str]


def single_meaningful_state(region: ControlFlowRegion) -> SDFGState | None:
    """The one non-empty state of ``region`` when every other block is an empty state joined by plain edges."""
    if any(not isinstance(b, SDFGState) for b in region.nodes()):
        return None
    if any(not e.data.is_unconditional() or e.data.assignments for e in region.edges()):
        return None
    non_empty = [s for s in region.nodes() if not s.is_empty()]
    return non_empty[0] if len(non_empty) == 1 else None


def sibling_maps(state: SDFGState) -> list[MapEntry] | None:
    """The top-level maps of ``state`` when everything else at the top level is an access node, else ``None``."""
    scope = state.scope_dict()
    top = [n for n in state.nodes() if scope[n] is None]
    entries = [n for n in top if isinstance(n, MapEntry)]
    exits = {state.exit_node(me) for me in entries}
    if not entries or any(not isinstance(n, AccessNode) for n in top if n not in exits and n not in entries):
        return None
    if any(all(isinstance(n, AccessNode) for n in state.all_nodes_between(me, state.exit_node(me))) for me in entries):
        return None
    return entries


def guard_chain(enclosing: sd.SDFG, cond_block: ConditionalBlock) -> list[SDFGState] | None:
    """The linear chain of states from the region entry to ``cond_block``, joined by unconditional edges."""
    chain: list[SDFGState] = []
    block = cond_block
    while enclosing.in_degree(block) == 1:
        edge = enclosing.in_edges(block)[0]
        pred = edge.src
        if not isinstance(pred, SDFGState) or enclosing.out_degree(pred) != 1 or not edge.data.is_unconditional():
            return None
        if any(not isinstance(n, (AccessNode, Tasklet)) for n in pred.nodes()):
            return None
        chain.insert(0, pred)
        block = pred
    return chain if enclosing.in_degree(block) == 0 else None


def single_body_nsdfg(state: SDFGState, entry: MapEntry) -> NestedSDFG:
    """The nested SDFG that forms the whole body of ``entry``; nests the body first if it is not one already."""
    body = list(state.all_nodes_between(entry, state.exit_node(entry)))
    code = [n for n in body if not isinstance(n, AccessNode)]
    if len(code) == 1 and isinstance(code[0], NestedSDFG):
        return code[0]
    return xfh.nest_state_subgraph(state.sdfg, state, SubgraphView(state, body))


@transformation.explicit_cf_compatible
class MoveIfIntoMap(transformation.MultiStateTransformation):
    """
    Pushes a single-branch conditional block, and the guard that computes its condition, into each inner map.

    Matches a map whose body is a nested SDFG holding ``guard -> ConditionalBlock(c)``, where the one branch holds one
    or more sibling top-level maps. Every inner map body becomes ``guard -> ConditionalBlock(c) -> old body``; the data
    the guard reads is piped in through the inner map, and its symbols through ``symbol_mapping``.

    Soundness: the guard sits above the inner maps, so it never reads their parameters. It is re-evaluated once per
    inner iteration and yields the value it had once per outer iteration as long as nothing in the branch writes what
    the guard reads, which is refused. The guard's transients and assigned symbols must be private to the guard, since
    they leave the outer body.

    Refused: ``else``/``elif`` branches, non-linear guard chains, guard states with anything but tasklets and access
    nodes (they would be recomputed per inner iteration), nested inner maps, and name clashes with the inner body.
    """

    cond_block = transformation.PatternNode(ConditionalBlock)

    @classmethod
    def expressions(cls):
        return [sdutil.node_path_graph(cls.cond_block)]

    def match(self) -> GuardMatch | None:
        """The guard to move, or ``None`` when the conditional block does not match."""
        cond_block: ConditionalBlock = self.cond_block
        if len(cond_block.branches) != 1 or cond_block.branches[0][0] is None:
            return None
        enclosing = cond_block.parent_graph
        if not isinstance(enclosing, sd.SDFG) or enclosing.parent_nsdfg_node is None:
            return None
        if enclosing.parent.entry_node(enclosing.parent_nsdfg_node) is None:
            return None
        branch_state = single_meaningful_state(cond_block.branches[0][1])
        if branch_state is None or sibling_maps(branch_state) is None:
            return None
        chain = guard_chain(enclosing, cond_block)
        if chain is None:
            return None
        trailing = set(enclosing.nodes()) - set(chain) - {cond_block}
        if any(not isinstance(b, SDFGState) or not b.is_empty() for b in trailing):
            return None
        match = guard_footprint(enclosing, cond_block, branch_state, chain)
        if not private_to_guard(match):
            return None
        return match if all(self.fits(match, me) for me in sibling_maps(branch_state)) else None

    @staticmethod
    def fits(match: GuardMatch, entry: MapEntry) -> bool:
        """Whether the guard's names can enter the body of ``entry`` without clashing with names already there."""
        state = match.branch_state
        scope = state.scope_subgraph(entry, include_entry=False, include_exit=False)
        names = {e.data.data for e in scope.edges() if e.data.data is not None}
        symbols = {str(s) for s in scope.free_symbols}
        for node in scope.nodes():
            if isinstance(node, NestedSDFG):
                names |= node.sdfg.arrays.keys()
                symbols |= node.sdfg.symbols.keys()
        return not (match.reads | match.locals) & names and not match.assigned & symbols

    def can_be_applied(self, graph, expr_index, sdfg, permissive=False):
        return self.match() is not None

    def apply(self, graph: ControlFlowRegion, sdfg: sd.SDFG):
        cond_block: ConditionalBlock = self.cond_block
        match = self.match()
        assert match is not None
        enclosing, state = match.enclosing, match.branch_state
        for entry in sibling_maps(state):
            nsdfg = single_body_nsdfg(state, entry)
            self.guard_inner_body(match, cond_block, nsdfg)
            for name in sorted(match.reads):
                nsdfg.add_in_connector(name)
                memlet = mm.Memlet.from_array(name, enclosing.arrays[name])
                state.add_memlet_path(state.add_read(name), entry, nsdfg, memlet=memlet, dst_conn=name)

        cond_block.branches[0][1].remove_node(state)
        enclosing.add_node(state, ensure_unique_name=True)
        for e in enclosing.out_edges(cond_block):
            enclosing.add_edge(state, e.dst, e.data)
        enclosing.remove_nodes_from([cond_block, *match.chain])
        enclosing.remove_nodes_from([b for b in enclosing.nodes() if b is not state and not enclosing.degree(b)])
        enclosing.start_block = enclosing.node_id(state)
        for name in match.locals:
            enclosing.remove_data(name, validate=False)
        for name in match.assigned & enclosing.symbols.keys():
            enclosing.remove_symbol(name)

    @staticmethod
    def guard_inner_body(match: GuardMatch, cond_block: ConditionalBlock, nsdfg: NestedSDFG) -> None:
        """Wraps the body of ``nsdfg`` in a copy of the condition, preceded by a copy of the guard chain."""
        declare_guard_names(match, nsdfg)
        inner, enclosing = nsdfg.sdfg, match.enclosing
        moved = wrap_in_condition(inner, cond_block)
        guard_edges = [e for s in match.chain for e in enclosing.out_edges(s)]
        if all(s.is_empty() for s in match.chain) and not any(e.data.assignments for e in guard_edges):
            inner.add_node(moved, is_start_block=True, ensure_unique_name=True)
            return
        copies = {s: copy.deepcopy(s) for s in match.chain}
        for i, s in enumerate(match.chain):
            inner.add_node(copies[s], is_start_block=i == 0, ensure_unique_name=True)
        inner.add_node(moved, ensure_unique_name=True)
        for e in guard_edges:
            dst = moved if e.dst is cond_block else copies[e.dst]
            inner.add_edge(copies[e.src], dst, InterstateEdge(assignments=dict(e.data.assignments)))


def guard_footprint(
    enclosing: sd.SDFG, cond_block: ConditionalBlock, branch_state: SDFGState, chain: list[SDFGState]
) -> GuardMatch:
    """What the guard chain and condition read, write as transients, assign and use as symbols."""
    cond_names = cond_block.branches[0][0].get_free_symbols()
    reads = {s for s in cond_names if s in enclosing.arrays}
    symbols = {s for s in cond_names if s not in enclosing.arrays}
    written: set[str] = set()
    for state in chain:
        state_reads, state_writes = state.read_and_write_sets()
        reads |= state_reads
        written |= state_writes
        symbols |= state.used_symbols(all_symbols=True)
    assigned: set[str] = set()
    for e in (e for b in chain for e in enclosing.out_edges(b)):
        reads |= {m.data for m in e.data.get_read_memlets(enclosing.arrays, include_scalars=True)}
        symbols |= e.data.used_symbols(all_symbols=True) - enclosing.arrays.keys()
        assigned |= e.data.assignments.keys()
    reads -= written
    for name in reads:
        symbols |= {str(s) for s in enclosing.arrays[name].free_symbols}
    return GuardMatch(enclosing, branch_state, chain, reads, written, assigned, symbols - assigned)


def private_to_guard(match: GuardMatch) -> bool:
    """Whether the guard can leave the outer body: its transients and symbols are used nowhere else, and the branch
    writes nothing it reads."""
    enclosing = match.enclosing
    body_reads, body_writes = match.branch_state.read_and_write_sets()
    outside_reads, outside_syms = outside_guard(match)
    if any(not enclosing.arrays[n].transient for n in match.locals):
        return False
    if match.locals & (body_reads | body_writes | outside_reads) or match.reads & body_writes:
        return False
    return not match.assigned & outside_syms and (match.symbols | match.assigned) <= enclosing.symbols.keys()


def outside_guard(match: GuardMatch) -> tuple[set[str], set[str]]:
    """The data read and the symbols used in the outer body outside the guard, except for the branch's data."""
    enclosing = match.enclosing
    guard_edges = [e for b in match.chain for e in enclosing.out_edges(b)]
    outside_edges = [e for e in enclosing.edges() if e not in guard_edges]
    reads = {m.data for e in outside_edges for m in e.data.get_read_memlets(enclosing.arrays, True)}
    symbols = match.branch_state.used_symbols(all_symbols=True)
    symbols |= {s for e in outside_edges for s in e.data.used_symbols(all_symbols=True)}
    symbols |= {str(s) for desc in enclosing.arrays.values() for s in desc.free_symbols}
    return reads, symbols


def declare_guard_names(match: GuardMatch, nsdfg: NestedSDFG) -> None:
    """Declares the guard's data and symbols in ``nsdfg``; the guard's own transients and symbols stay private."""
    inner, enclosing = nsdfg.sdfg, match.enclosing
    for name in match.reads | match.locals:
        if name not in inner.arrays:
            desc = copy.deepcopy(enclosing.arrays[name])
            desc.transient = name in match.locals
            inner.add_datadesc(name, desc)
    for name in match.symbols | match.assigned:
        if name not in inner.symbols:
            inner.add_symbol(name, enclosing.symbols[name])
    for name in match.symbols:
        nsdfg.symbol_mapping.setdefault(name, name)
    for name in match.assigned:
        nsdfg.symbol_mapping.pop(name, None)


def wrap_in_condition(inner: sd.SDFG, cond_block: ConditionalBlock) -> ConditionalBlock:
    """Moves every block of ``inner`` into the one branch of a copy of ``cond_block``, which is returned unattached."""
    body = ControlFlowRegion(label=f"{cond_block.label}_body")
    blocks, edges, start = list(inner.nodes()), list(inner.edges()), inner.start_block
    inner.remove_nodes_from(blocks)
    for b in blocks:
        body.add_node(b, is_start_block=b is start)
    for e in edges:
        body.add_edge(e.src, e.dst, e.data)
    moved = ConditionalBlock(label=f"{cond_block.label}_moved")
    moved.add_branch(CodeBlock(cond_block.branches[0][0].as_string), body)
    return moved
