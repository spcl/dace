# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Cascade interstate-edge symbol assignments up out of enclosing loops."""

import ast
from typing import Any

from dace import SDFG, data
from dace.frontend.python import astutils
from dace.sdfg import nodes
from dace.sdfg.sdfg import InterstateEdge
from dace.sdfg.state import ConditionalBlock, ControlFlowBlock, ControlFlowRegion, LoopRegion, SDFGState, is_written
from dace.transformation import pass_pipeline as ppl
from dace.transformation import transformation
from dace.transformation.passes.analysis.loop_analysis import loop_header_names


class HoistAnalysisCache:
    """Per-round memo of the content-derived answers ``_cascade_once`` asks repeatedly."""

    __slots__ = ("reads", "writes", "preds", "region_indices")

    def __init__(self) -> None:
        self.reads: dict[int, dict[str, None] | None] = {}
        self.writes: dict[int, tuple[dict[str, None], dict[str, None]]] = {}
        self.preds: dict[tuple[int, int], dict[ControlFlowBlock, None]] = {}
        self.region_indices: dict[int, tuple[dict[str, list[str]], dict[str, None]]] = {}

    def block_reads(self, block: ControlFlowBlock) -> dict[str, None] | None:
        block_id = id(block)
        if block_id not in self.reads:
            self.reads[block_id] = _block_reads_symbols(block)
        return self.reads[block_id]

    def block_writes(self, block: ControlFlowBlock) -> tuple[dict[str, None], dict[str, None]]:
        block_id = id(block)
        if block_id not in self.writes:
            self.writes[block_id] = _block_writes(block)
        return self.writes[block_id]

    def predecessors(self, parent: ControlFlowRegion, child: ControlFlowBlock) -> dict[ControlFlowBlock, None]:
        pred_key = (id(parent), id(child))
        if pred_key not in self.preds:
            self.preds[pred_key] = _predecessors_in(parent, child)
        return self.preds[pred_key]

    def region_index(self, region: ControlFlowRegion) -> tuple[dict[str, list[str]], dict[str, None]]:
        region_id = id(region)
        if region_id not in self.region_indices:
            self.region_indices[region_id] = build_region_index(region)
        return self.region_indices[region_id]

    def clear(self) -> None:
        self.reads.clear()
        self.writes.clear()
        self.preds.clear()
        self.region_indices.clear()


def names_read_by(expr: str) -> dict[str, None]:
    """Every name a string expression reads: symbols AND the data containers it subscripts."""
    try:
        visitor = astutils.TaskletFreeSymbolVisitor([])
        visitor.visit(ast.parse(expr.strip(), mode="eval"))
    except Exception:
        return {}
    # Consumed only by membership tests, so the visitor's unordered set cannot reach codegen.
    return dict.fromkeys(visitor.free_symbols)


def _region_writes(region: ControlFlowRegion) -> tuple[dict[str, None], dict[str, None]]:
    """All symbols (interstate assignments) and data containers written anywhere inside ``region`` and its descendants."""
    asyms: dict[str, None] = {}
    wdata: dict[str, None] = {}
    for e in region.all_interstate_edges():
        asyms.update(dict.fromkeys(e.data.assignments.keys()))
    for inner in region.all_control_flow_regions():
        if isinstance(inner, LoopRegion) and inner.loop_variable:
            asyms[str(inner.loop_variable)] = None
    for st in region.states():
        for n in st.nodes():
            if isinstance(n, nodes.AccessNode) and is_written(st, n):
                wdata[n.data] = None
    return asyms, wdata


def build_region_index(region: ControlFlowRegion) -> tuple[dict[str, list[str]], dict[str, None]]:
    """Every interstate assignment's rhs, grouped by lhs, and every data name an access node writes, anywhere inside
    ``region``."""
    by_key: dict[str, list[str]] = {}
    written: dict[str, None] = {}
    for e in region.all_interstate_edges():
        for lhs, rhs_value in e.data.assignments.items():
            by_key.setdefault(lhs, []).append(str(rhs_value))
    for st in region.states():
        for n in st.nodes():
            if isinstance(n, nodes.AccessNode) and is_written(st, n):
                written[n.data] = None
    return by_key, written


def _key_has_other_writer(cache: HoistAnalysisCache, region: ControlFlowRegion, key: str, rhs: str) -> bool:
    """True iff ``key`` is written inside ``region`` by anything other than a ``key = rhs`` interstate assignment."""
    by_key, written = cache.region_index(region)
    if key in written:
        return True
    return any(other != str(rhs) for other in by_key.get(key, []))


def _meets_binding_rule(dest: ControlFlowRegion, cfg: ControlFlowRegion, sdfg: SDFG) -> bool:
    """The all-or-nothing upward rule: ``dest`` must be outside every ``LoopRegion`` enclosing ``cfg``."""
    if isinstance(dest, LoopRegion):
        return False
    p = dest.parent_graph
    while p is not None and p is not sdfg:
        if isinstance(p, LoopRegion):
            return False
        p = p.parent_graph
    return True


def _on_conditional_branch(cfg: ControlFlowRegion, dest: ControlFlowRegion, sdfg: SDFG) -> bool:
    """True iff the chain from ``cfg`` up to (but not including) ``dest``
    passes through a ``ConditionalBlock`` branch -- the iedge would not be
    unconditionally executed under ``dest`` (L5)."""
    g = cfg
    while g is not None and g is not dest:
        p = g.parent_graph
        if isinstance(p, ConditionalBlock):
            return True
        g = p
    return False


def _predecessors_in(parent: ControlFlowRegion, child: ControlFlowBlock) -> dict[ControlFlowBlock, None]:
    """Strict predecessors of a block within its parent region."""
    out: dict[ControlFlowBlock, None] = {}
    stack = [child]
    while stack:
        cur = stack.pop()
        for e in parent.in_edges(cur):
            if e.src in out or e.src is child:
                continue
            out[e.src] = None
            stack.append(e.src)
    return out


def _read_after(parent: ControlFlowRegion, child: ControlFlowBlock, key: str, cache: HoistAnalysisCache) -> bool:
    """True iff something after ``child`` in ``parent`` reads the value of ``key`` that ``child`` leaves."""
    for e in parent.dfs_edges(child):
        reads = cache.block_reads(e.dst)
        if reads is None or key in e.data.read_symbols() or key in reads and key not in e.data.assignments:
            return True
    return False


def _block_reads_symbols(block: ControlFlowBlock) -> dict[str, None] | None:
    """The symbols a block reads from outside it (state, region, etc.)."""
    try:
        return dict.fromkeys(str(s) for s in block.free_symbols)
    except Exception:
        return None


def _block_writes(block: ControlFlowBlock) -> tuple[dict[str, None], dict[str, None]]:
    """All symbols assigned and data containers written anywhere inside a block."""
    if isinstance(block, (ControlFlowRegion, ConditionalBlock)):
        return _region_writes(block)
    asyms: dict[str, None] = {}
    wdata: dict[str, None] = {}
    if isinstance(block, SDFGState):
        for n in block.nodes():
            if isinstance(n, nodes.AccessNode) and is_written(block, n):
                wdata[n.data] = None
    return asyms, wdata


def _legal_to_hoist_into(
    parent: ControlFlowRegion,
    child: ControlFlowRegion,
    key: str,
    rhs: str,
    rhs_syms: dict[str, None],
    sdfg: SDFG,
    cache: HoistAnalysisCache,
) -> bool:
    """Decide whether an assignment inside ``child`` may move up one level."""
    # L1: any new symbol introduced by ``child`` (its loop variable, or any iedge assignment inside ``child`` other than
    # the one being moved) that the rhs would now reference is a violation.
    if isinstance(child, LoopRegion) and child.loop_variable and str(child.loop_variable) in rhs_syms:
        return False
    # A self-referential assignment (``key`` appears in its own rhs, e.g. ``j = j + 2``) is loop-carried: its value
    # changes every iteration, so it is not invariant and must not be hoisted out of an enclosing loop.
    if isinstance(child, LoopRegion) and key in rhs_syms:
        return False
    # ``key`` modified elsewhere inside a loop is loop-carried: the assignment is a per-iteration reset (e.g. the
    # ``count = 0`` seeding a stream-compaction loop whose body does ``count = count + 1``).
    if isinstance(child, LoopRegion) and _key_has_other_writer(cache, child, key, rhs):
        return False
    # The loop's own header runs ahead of the assignment on the first iteration.
    if isinstance(child, LoopRegion) and key in loop_header_names(child):
        return False
    # A loop may run no iteration: an array read in the rhs would then be an access the program never made.
    if isinstance(child, LoopRegion) and any(isinstance(sdfg.arrays.get(name), data.Array) for name in rhs_syms):
        return False
    inner_asyms, inner_wdata = cache.block_writes(child)
    inner_asyms = dict(inner_asyms)
    inner_asyms.pop(key, None)  # discount the assignment we're moving
    if any(s in inner_asyms for s in rhs_syms):
        return False
    if any(s in inner_wdata for s in rhs_syms):
        return False  # a tasklet inside child writes to a container our rhs reads

    # L2/L3/L4: examine strict predecessors of ``child`` in ``parent``.
    preds = cache.predecessors(parent, child)
    for b in preds:
        reads = cache.block_reads(b)
        if reads is None or key in reads:
            return False  # L3: a predecessor would observe the moved assignment
        b_asyms, b_wdata = cache.block_writes(b)
        if key in b_asyms or key in b_wdata:
            return False  # L4: predecessor writes key
        if any(s in b_asyms for s in rhs_syms) or any(s in b_wdata for s in rhs_syms):
            return False  # L2: predecessor writes a rhs symbol

    # Also any iedge in parent (other than the in-edges of ``child``) that writes ``key`` or a ``rhs_sym`` is a
    # violation.
    for e in parent.edges():
        if e.src in preds and key in e.data.read_symbols():
            return False  # L3: an edge ahead of ``child`` would observe the moved assignment
        if e.dst is child:
            continue
        for lhs, e_rhs in e.data.assignments.items():
            if lhs == key:
                if str(e_rhs) == str(rhs):
                    continue  # same key/rhs already hoisted by a sibling
                return False
            if lhs in rhs_syms:
                return False

    # L7: ``child`` need not run the assignment (no iteration, a break), and whatever follows it
    # then reads the value from before ``child``, not the moved one.
    return not _read_after(parent, child, key, cache)


def _find_destination(
    edge_region: ControlFlowRegion, key: str, rhs: str, sdfg: SDFG, origin: ControlFlowBlock, cache: HoistAnalysisCache
) -> ControlFlowRegion | None:
    """Walk up the ``parent_graph`` chain from ``edge_region`` to find the outermost ancestor ``D`` where the move is
    legal under L1-L6."""
    rhs_syms = names_read_by(rhs)
    # legality scan (early-exit AND): iteration order does not affect the True/False outcome
    for b in dict.fromkeys([*cache.predecessors(edge_region, origin), origin]):
        reads = cache.block_reads(b)
        if reads is None or key in reads:
            return None
        # L5, for the edges of the region itself: an assignment behind a condition does not always run.
        if any(not e.data.is_unconditional() or key in e.data.read_symbols() for e in edge_region.out_edges(b)):
            return None
    dest: ControlFlowRegion = edge_region
    walker: ControlFlowRegion = edge_region
    while True:
        parent = walker.parent_graph
        if parent is None:
            break
        if not isinstance(parent, ControlFlowRegion):
            break  # crossed out of SDFG (would need L6 NSDFG passthrough)
        if not _legal_to_hoist_into(parent, walker, key, rhs, rhs_syms, sdfg, cache):
            break
        dest = parent
        walker = parent

    if dest is edge_region:
        return None
    if _on_conditional_branch(edge_region, dest, sdfg):
        return None  # L5: at least one step crossed a ConditionalBlock branch
    if not _meets_binding_rule(dest, edge_region, sdfg):
        return None
    if not _lands_without_race(dest, _direct_child(dest, edge_region), key, rhs):
        return None
    # L7 above ``dest``: a loop below it may run no iteration, and what follows the block holding ``dest`` reads ``key``
    block: ControlFlowBlock = dest
    while block.parent_graph is not None:
        if isinstance(block.parent_graph, ControlFlowRegion) and _read_after(block.parent_graph, block, key, cache):
            return None
        block = block.parent_graph
    return dest


def _edge_landing_order(existing: dict[str, str], key: str, rhs_syms: dict[str, None]) -> str:
    """How ``key = rhs`` must be sequenced against the assignments already on a landing edge."""
    after = any(s in existing for s in rhs_syms if s != key)
    before = any(key in names_read_by(str(other_rhs)) for other_key, other_rhs in existing.items() if other_key != key)
    if after and before:
        return "conflict"
    if after:
        return "after"
    if before:
        return "before"
    return "direct"


def _lands_without_race(dest: ControlFlowRegion, child: ControlFlowRegion, key: str, rhs: str) -> bool:
    """Whether ``key = rhs`` can join the edges it would land on, directly or via a split."""
    if child is None:
        return True
    rhs_syms = names_read_by(rhs)
    for e in dest.in_edges(child):
        order = _edge_landing_order(e.data.assignments, key, rhs_syms)
        if order == "direct":
            continue
        if order == "after" and e.data.is_unconditional():
            continue
        return False
    return True


def _place_assignment_at(dest: ControlFlowRegion, child: ControlFlowRegion, key: str, rhs: str) -> None:
    """Add ``key = rhs`` so it dominates ``child`` inside ``dest``."""
    in_edges = list(dest.in_edges(child))
    # One suffix: a hoist state fused into ``child`` comes back as its label, which would grow by one each round
    label = f"{child.label.split('_iedge_hoist')[0]}_iedge_hoist"
    if not in_edges:
        pre = dest.add_state(label, is_start_block=dest.start_block is child)
        dest.add_edge(pre, child, InterstateEdge(assignments={key: rhs}))
        return
    rhs_syms = names_read_by(rhs)
    needs_split = any(_edge_landing_order(e.data.assignments, key, rhs_syms) == "after" for e in in_edges)
    if needs_split:
        dest.add_state_before(child, label, assignments={key: rhs})
        return
    for e in in_edges:
        e.data.assignments[key] = rhs


def _drop_inner_symbol_declarations(sdfg: SDFG, key: str, dest: ControlFlowRegion) -> None:
    """Placeholder for the cross-NSDFG extension (currently a no-op)."""
    return


def _direct_child(dest: ControlFlowRegion, edge_region: ControlFlowRegion) -> ControlFlowRegion:
    """Walk up from ``edge_region`` to find the immediate child of ``dest``."""
    g = edge_region
    while g.parent_graph is not dest:
        g = g.parent_graph
        if g is None:
            raise RuntimeError("dest is not an ancestor of edge_region")
    return g


def _cascade_once(sdfg: SDFG) -> int:
    """One sweep across the SDFG; returns the number of assignments moved."""
    moved = 0
    # Built lazily and reused across every candidate in this sweep
    cache = HoistAnalysisCache()
    for cfg in list(sdfg.all_control_flow_regions(recursive=True)):
        for edge in list(cfg.edges()):
            if not edge.data.assignments:
                continue
            # Snapshot keys -- we mutate the dict as we go.
            for key, rhs in list(edge.data.assignments.items()):
                # ``cfg.sdfg``: the names ``rhs`` reads are declared by the SDFG owning the edge
                dest = _find_destination(cfg, key, rhs, cfg.sdfg, edge.src, cache)
                if dest is None:
                    continue
                child = _direct_child(dest, cfg)
                # Atomic move: only mutate once the destination is fixed.
                del edge.data.assignments[key]
                _place_assignment_at(dest, child, key, rhs)
                _drop_inner_symbol_declarations(sdfg, key, dest)
                moved += 1
                cache.clear()
    return moved


def _hoist_to_body_head(sdfg: SDFG) -> int:
    """Move ``key = rhs`` off the edge leaving a loop body's first state onto the body's entry, when that state neither
    reads ``key`` nor writes anything ``rhs`` reads."""
    moved = 0
    for loop in list(sdfg.all_control_flow_regions(recursive=True)):
        if not isinstance(loop, LoopRegion):
            continue
        first = loop.start_block
        # An empty ``first`` is the head this hoist left: nothing is kept apart, and moving on would never settle
        if (
            not isinstance(first, SDFGState)
            or first.is_empty()
            or loop.in_degree(first) != 0
            or loop.out_degree(first) != 1
        ):
            continue
        edge = loop.out_edges(first)[0]
        if not edge.data.assignments or not edge.data.is_unconditional():
            continue
        reads = _block_reads_symbols(first)
        if reads is None:
            continue
        written = _block_writes(first)[1]
        for key, rhs in list(edge.data.assignments.items()):
            rhs_names = names_read_by(str(rhs))
            others = [str(v) for k, v in edge.data.assignments.items() if k != key]
            if (
                key in reads
                or key in rhs_names
                or any(name in written or name in edge.data.assignments for name in rhs_names)
                or any(key in names_read_by(other) for other in others)
            ):
                continue
            del edge.data.assignments[key]
            _place_assignment_at(loop, first, key, rhs)
            first = loop.start_block if loop.start_block is first else first
            moved += 1
    return moved


@transformation.explicit_cf_compatible
class CascadeInterstateEdgeAssignmentsUp(ppl.Pass):
    """Cascade invariant interstate-edge symbol assignments past every enclosing ``LoopRegion`` (all-or-nothing upward;
    fixpoint)."""

    CATEGORY: str = "Canonicalization"

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.InterstateEdges | ppl.Modifies.CFG

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self) -> list[type[ppl.Pass] | ppl.Pass]:
        return []

    def apply_pass(self, sdfg: SDFG, _pipeline_results: dict[str, Any]) -> int | None:
        """Repeatedly hoist legal interstate-edge assignments until a fixpoint."""
        total = 0
        while True:
            n = _cascade_once(sdfg)
            if not n:
                break
            total += n
        return total or None


@transformation.explicit_cf_compatible
class HoistAssignmentsToLoopBodyHead(ppl.Pass):
    """Move a loop body's per-iteration interstate assignment above the body's first state when that state neither
    reads its symbol nor writes its inputs, so the states it separated can fuse (see :func:`_hoist_to_body_head`)."""

    CATEGORY: str = "Canonicalization"

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.InterstateEdges | ppl.Modifies.CFG

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self) -> list[type[ppl.Pass] | ppl.Pass]:
        return []

    def apply_pass(self, sdfg: SDFG, _pipeline_results: dict[str, Any]) -> int | None:
        """:returns: The number of assignments moved, or ``None`` if none."""
        return _hoist_to_body_head(sdfg) or None
