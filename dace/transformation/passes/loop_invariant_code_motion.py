# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Loop Invariant Code Motion (LICM).

Hoists invariant computations out of ``LoopRegion`` bodies to a preheader state,
and out of ``Map`` scopes to the enclosing ``SDFGState``.

A tasklet is invariant w.r.t. a loop iff
  1. its code has no free symbol that is a loop induction variable or a symbol
     written on an interstate edge inside the loop,
  2. every in-edge memlet subset has no such free symbol, and every in-edge
     source is an ``AccessNode`` whose data container is not written anywhere
     inside the loop (conservative alias check),
  3. it is unconditionally executed each iteration, has no side effects, no
     WCR output, and is not an integer div/mod by a possibly-zero invariant
     divisor.

Loops with symbolic bounds are assumed to execute at least one iteration and
floating-point operations are assumed non-trapping, so speculative execution is
not a concern.
"""
import ast
import copy
from typing import Any, Dict, List, Optional, Set, Tuple

from dace import SDFG, SDFGState, properties, symbolic
from dace.sdfg import nodes, InterstateEdge
from dace.sdfg.state import ControlFlowRegion, LoopRegion
from dace.transformation import pass_pipeline as ppl
from dace.transformation import transformation as xf


@properties.make_properties
@xf.explicit_cf_compatible
class LoopInvariantCodeMotion(ppl.Pass):
    """Hoist loop-invariant computations out of ``LoopRegion`` bodies and ``Map`` scopes."""

    CATEGORY: str = "Optimization Preparation"

    def modifies(self) -> ppl.Modifies:
        return (ppl.Modifies.States | ppl.Modifies.Nodes | ppl.Modifies.Memlets
                | ppl.Modifies.Descriptors | ppl.Modifies.InterstateEdges)

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return bool(modified & (ppl.Modifies.States | ppl.Modifies.Nodes
                                | ppl.Modifies.Memlets | ppl.Modifies.InterstateEdges))

    def apply_pass(self, sdfg: SDFG, _: Dict[str, Any]) -> Optional[int]:
        total = 0
        while True:
            made = 0
            # LoopRegions, innermost first.
            loops = [n for n, _p in sdfg.all_nodes_recursive() if isinstance(n, LoopRegion)]
            loops.sort(key=_loop_nesting_depth, reverse=True)
            for loop in loops:
                made += _hoist_loop_region(loop)

            # Maps, innermost first per state.
            for sd in sdfg.all_sdfgs_recursive():
                for state in list(sd.states()):
                    map_entries = [n for n in state.nodes() if isinstance(n, nodes.MapEntry)]
                    map_entries.sort(key=lambda me: _map_scope_depth(state, me), reverse=True)
                    for me in map_entries:
                        if me not in state.nodes():  # may have been removed
                            continue
                        made += _hoist_map_scope(state, me)

            total += made
            if made == 0:
                break
        return total or None


# ---------------------------------------------------------------------------
# Depth helpers (innermost-first ordering)
# ---------------------------------------------------------------------------


def _loop_nesting_depth(loop: LoopRegion) -> int:
    depth = 0
    p = loop.parent_graph
    while p is not None:
        if isinstance(p, LoopRegion):
            depth += 1
        p = getattr(p, "parent_graph", None)
    return depth


def _map_scope_depth(state: SDFGState, me: nodes.MapEntry) -> int:
    depth = 0
    sdict = state.scope_dict()
    p = sdict.get(me)
    while p is not None:
        depth += 1
        p = sdict.get(p)
    return depth


# ---------------------------------------------------------------------------
# LoopRegion path
# ---------------------------------------------------------------------------


def _hoist_loop_region(loop: LoopRegion) -> int:
    """Hoist invariant tasklets and whole sub-regions out of one LoopRegion."""
    count = 0

    # (1) Sub-region hoist: a nested LoopRegion or SDFGState inside the body
    # that never references any variant symbol can be lifted to the outer
    # region's parent graph, executed exactly once before the outer loop.
    count += _hoist_invariant_child_regions(loop)

    # (2) Tasklet hoist in the body's start state. ConditionalBlocks and
    # deeper-nested states are skipped (unconditional-execution gate).
    variant_syms = _variant_symbols_of_loop(loop)
    variant_data = _written_data_in_region(loop)
    region_writers = _region_writer_counts(loop)
    start = loop.start_block
    if isinstance(start, SDFGState):
        while True:
            tasklet = _find_one_invariant_tasklet(start, variant_syms, variant_data, region_writers)
            if tasklet is None:
                break
            preheader = _get_or_create_preheader(loop)
            if not _hoist_tasklet_to_preheader(start, tasklet, preheader):
                break
            count += 1

    # (3) If the body is empty after hoisting, leave an empty "hull" state so
    # the CFG remains well-formed.
    if len(loop.nodes()) == 0:
        loop.add_state(f"{loop.label}_licm_hull", is_start_block=True)

    return count


def _hoist_invariant_child_regions(loop: LoopRegion) -> int:
    """Move direct children of ``loop`` that do not reference any variant
    symbol of the loop and whose reads are alias-free against other siblings'
    writes to the parent region, placing them before ``loop``.
    """
    parent = loop.parent_graph
    if parent is None:
        return 0

    variant_syms = _variant_symbols_of_loop(loop)

    children = list(loop.nodes())
    if not children:
        return 0

    # Build a per-sibling write-set so we can alias-check one child against
    # the others that will remain in the body.
    child_reads: Dict[Any, Set[str]] = {}
    child_writes: Dict[Any, Set[str]] = {}
    for ch in children:
        r, w = _region_rw_sets(ch)
        child_reads[ch] = r
        child_writes[ch] = w

    # A child is hoistable if (a) it has no free symbol in variant_syms, and
    # (b) its read-set does not intersect any OTHER child's write-set, and
    # (c) its write-set does not intersect any OTHER child's read-or-write
    # (so order w.r.t. other children is irrelevant).
    hoistable: List[Any] = []
    for ch in children:
        fs = _region_free_symbols(ch)
        if fs & variant_syms:
            continue
        others = [o for o in children if o is not ch]
        r, w = child_reads[ch], child_writes[ch]
        conflict = False
        for o in others:
            if r & child_writes[o]:
                conflict = True
                break
            if w & (child_reads[o] | child_writes[o]):
                conflict = True
                break
        if conflict:
            continue
        # A region that reads from a container it also writes to is not
        # idempotent across outer iterations — running it K times is not
        # equivalent to running it once (each pass uses the previous result).
        # This mirrors LLVM's "in-scope write must not alias an in-scope read"
        # clause extended over the whole region.
        if r & w:
            continue
        # Reject regions with side effects (non-idempotent); also reject
        # empty / hull regions that do no useful work (prevents an infinite
        # hoist-then-add-hull loop).
        if _region_has_side_effect(ch):
            continue
        if not _region_has_work(ch):
            continue
        hoistable.append(ch)

    if not hoistable:
        return 0

    # Determine the insertion order: preserve the order in which the children
    # appear along the start-to-end path of ``loop``.
    order_map = {
        n: i
        for i, n in enumerate(loop.bfs_nodes(loop.start_block) if hasattr(loop, "bfs_nodes") else loop.nodes())
    }
    hoistable.sort(key=lambda n: order_map.get(n, 0))

    # Surgery: remove each hoistable child from ``loop`` and insert it as a
    # new node in ``parent`` before ``loop``. The chain becomes:
    #   ... -> h1 -> h2 -> ... -> loop -> ...
    for child in hoistable:
        _move_region_before(parent, loop, child)

    return len(hoistable)


def _move_region_before(parent: ControlFlowRegion, loop: Any, child: Any) -> None:
    """Relocate ``child`` from inside ``loop`` to ``parent``, chained right
    before ``loop``. Incoming edges to ``loop`` now go to ``child`` first,
    and a new unconditional edge ``child -> loop`` is added.
    """
    # Detach internal edges touching child inside loop.
    for e in list(loop.in_edges(child)) + list(loop.out_edges(child)):
        loop.remove_edge(e)
    was_start = (loop.start_block is child)
    loop.remove_node(child)
    if was_start and loop.nodes():
        # Pick any remaining node as the new start_block; if none remains,
        # the caller will add an empty hull state. The setter takes a node ID.
        loop.start_block = loop.node_id(loop.nodes()[0])

    parent.add_node(child, ensure_unique_name=True)  # reparented into shared CFG; wired by object ref
    # Reroute parent's incoming edges into loop through child.
    for e in list(parent.in_edges(loop)):
        parent.remove_edge(e)
        parent.add_edge(e.src, child, e.data)
    parent.add_edge(child, loop, InterstateEdge())


def _region_free_symbols(region: Any) -> Set[str]:
    try:
        return {str(s) for s in region.free_symbols}
    except Exception:
        return set()


def _region_rw_sets(region: Any) -> Tuple[Set[str], Set[str]]:
    reads: Set[str] = set()
    writes: Set[str] = set()
    if isinstance(region, SDFGState):
        for n in region.data_nodes():
            if region.in_degree(n) > 0:
                writes.add(n.data)
            if region.out_degree(n) > 0:
                reads.add(n.data)
        return reads, writes
    if hasattr(region, "all_states"):
        for s in region.all_states():
            for n in s.data_nodes():
                if s.in_degree(n) > 0:
                    writes.add(n.data)
                if s.out_degree(n) > 0:
                    reads.add(n.data)
    return reads, writes


def _region_has_work(region: Any) -> bool:
    """True iff ``region`` contains at least one compute / dataflow node."""
    if isinstance(region, SDFGState):
        for n in region.nodes():
            if isinstance(n, (nodes.Tasklet, nodes.NestedSDFG, nodes.LibraryNode, nodes.MapEntry)):
                return True
        return False
    if hasattr(region, "all_states"):
        for s in region.all_states():
            if _region_has_work(s):
                return True
    return False


def _region_has_side_effect(region: Any) -> bool:
    if isinstance(region, SDFGState):
        for n in region.nodes():
            if isinstance(n, nodes.Tasklet):
                if getattr(n, "side_effects", False):
                    return True
            if isinstance(n, nodes.LibraryNode):
                if getattr(n, "has_side_effects", False):
                    return True
            for oe in region.out_edges(n):
                if oe.data is not None and oe.data.wcr is not None:
                    return True
        return False
    if hasattr(region, "all_states"):
        for s in region.all_states():
            if _region_has_side_effect(s):
                return True
    return False


def _variant_symbols_of_loop(loop: LoopRegion) -> Set[str]:
    syms: Set[str] = set()
    if loop.loop_variable:
        syms.add(loop.loop_variable)
    for e in loop.all_interstate_edges():
        syms.update(e.data.assignments.keys())
    return syms


def _written_data_in_region(region: ControlFlowRegion) -> Set[str]:
    """Names of data containers written in ``region`` (any state, any depth)."""
    written: Set[str] = set()
    for state in region.all_states():
        for n in state.data_nodes():
            if state.in_degree(n) > 0:
                written.add(n.data)
    return written


def _region_writer_counts(region: ControlFlowRegion) -> Dict[str, int]:
    """Per-data count of writer AccessNodes across every state of ``region``.

    Used to reject hoisting an invariant assignment (e.g. ``s = 0.0``) whose
    target is *also* written elsewhere in the loop (e.g. an inner-loop
    accumulation ``s = s + a[i, j]``): moving the init to the preheader would
    stop it re-running per iteration, so later iterations would see the carried
    value instead of the constant.
    """
    counts: Dict[str, int] = {}
    for state in region.all_states():
        for n in state.data_nodes():
            if state.in_degree(n) > 0:
                counts[n.data] = counts.get(n.data, 0) + 1
    return counts


def _get_or_create_preheader(loop: LoopRegion) -> SDFGState:
    parent = loop.parent_graph
    if parent is None:
        raise RuntimeError("LoopRegion has no parent graph")
    # Reuse an existing dedicated preheader from a prior LICM invocation.
    for e in parent.in_edges(loop):
        if isinstance(e.src, SDFGState) and e.src.label.startswith(f"{loop.label}_licm_preheader"):
            return e.src
    return parent.add_state_before(loop, label=f"{loop.label}_licm_preheader")


def _find_one_invariant_tasklet(
    state: SDFGState,
    variant_syms: Set[str],
    variant_data: Set[str],
    region_writers: Dict[str, int],
) -> Optional[nodes.Tasklet]:
    for n in state.nodes():
        if not isinstance(n, nodes.Tasklet):
            continue
        if _is_tasklet_invariant(state, n, variant_syms, variant_data, region_writers):
            return n
    return None


def _is_tasklet_invariant(
    state: SDFGState,
    tasklet: nodes.Tasklet,
    variant_syms: Set[str],
    variant_data: Set[str],
    region_writers: Dict[str, int],
) -> bool:
    # Side effects / WCR
    try:
        if tasklet.side_effects:
            return False
    except AttributeError:
        pass
    for oe in state.out_edges(tasklet):
        if oe.data is not None and oe.data.wcr is not None:
            return False

    # Code symbols must not touch any variant symbol.
    code_syms = _code_free_symbols(tasklet)
    if code_syms & variant_syms:
        return False

    # Must have inputs (else there is nothing to analyze — and a no-input
    # tasklet producing a constant is already loop-invariant trivially, so
    # we still hoist it if it satisfies the output shape).
    in_edges = list(state.in_edges(tasklet))

    # Every in-edge must come from an AccessNode reading non-variant data,
    # with a subset whose free symbols do not touch variant_syms.
    for ie in in_edges:
        if not isinstance(ie.src, nodes.AccessNode):
            return False
        if ie.src.data in variant_data:
            return False
        if ie.data is not None and ie.data.subset is not None:
            if _subset_uses_any(ie.data.subset, variant_syms):
                return False
        # The AccessNode itself must be a pure read — no in-edges inside the state.
        if state.in_degree(ie.src) > 0:
            return False

    # Output must go to a single AccessNode write, whose data is written only
    # by this tasklet in the state (so we can pre-populate it safely).
    out_edges = list(state.out_edges(tasklet))
    if len(out_edges) != 1:
        return False
    oe = out_edges[0]
    if not isinstance(oe.dst, nodes.AccessNode):
        return False
    # Output memlet subset must be loop-invariant too (otherwise writing in
    # the preheader to a loop-index-dependent slot would be wrong).
    if oe.data is not None and oe.data.subset is not None:
        if _subset_uses_any(oe.data.subset, variant_syms):
            return False

    out_data = oe.dst.data
    # Count writers to this data in the whole state.
    writers = 0
    for n in state.data_nodes():
        if n.data == out_data and state.in_degree(n) > 0:
            writers += 1
    if writers != 1:
        return False
    # ...and across the whole loop region: if ``out_data`` is written anywhere
    # else in the loop (e.g. an inner-loop accumulation into the same scalar),
    # hoisting this assignment to the preheader would stop it re-running each
    # iteration -- later iterations would observe the carried value, not the
    # constant. Only this tasklet's single write may exist region-wide.
    if region_writers.get(out_data, 0) != 1:
        return False

    # Integer div / mod by a possibly-zero invariant divisor.
    if _has_integer_div_mod_by_possibly_zero(tasklet):
        return False

    return True


def _code_free_symbols(tasklet: nodes.Tasklet) -> Set[str]:
    """Free symbols that appear in the tasklet code AST, minus its connectors."""
    syms: Set[str] = set()
    code = tasklet.code
    if code is None:
        return syms
    stmts = code.code
    if isinstance(stmts, str):
        try:
            stmts = [ast.parse(stmts)]
        except SyntaxError:
            return syms
    if not isinstance(stmts, list):
        stmts = [stmts]
    names: Set[str] = set()
    for s in stmts:
        if not isinstance(s, ast.AST):
            continue
        for node in ast.walk(s):
            if isinstance(node, ast.Name):
                names.add(node.id)
    # Exclude tasklet connectors and Python builtins we do not care about.
    connectors = set(tasklet.in_connectors.keys()) | set(tasklet.out_connectors.keys())
    return (names - connectors) - {"True", "False", "None"}


def _subset_uses_any(subset, syms: Set[str]) -> bool:
    try:
        fs = subset.free_symbols
    except AttributeError:
        return False
    if not syms:
        return False
    sym_strs = {str(s) for s in fs}
    return bool(sym_strs & syms)


def _has_integer_div_mod_by_possibly_zero(tasklet: nodes.Tasklet) -> bool:
    """Heuristic: block hoist if the tasklet AST contains FloorDiv/Mod with a
    non-literal, non-positive-symbol RHS. True Div is assumed fp and safe."""
    code = tasklet.code
    if code is None:
        return False
    stmts = code.code if not isinstance(code.code, str) else None
    if stmts is None:
        try:
            stmts = [ast.parse(code.code)]
        except SyntaxError:
            return False
    if not isinstance(stmts, list):
        stmts = [stmts]
    for s in stmts:
        if not isinstance(s, ast.AST):
            continue
        for node in ast.walk(s):
            if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.FloorDiv, ast.Mod)):
                rhs = node.right
                if isinstance(rhs, ast.Constant) and isinstance(rhs.value, (int, float)) and rhs.value != 0:
                    continue
                if isinstance(rhs, ast.Name):
                    sym = symbolic.symbol(rhs.id)
                    if getattr(sym, "is_positive", False):
                        continue
                return True
    return False


def _written_access_in_state(state: SDFGState, data: str) -> Optional[nodes.AccessNode]:
    """An AccessNode for ``data`` in ``state`` that is written (has an in-edge).

    Used to find an already-hoisted producer's output node so a subsequently
    hoisted consumer can connect to it, rather than reading a fresh, unwritten
    node for the same data.
    """
    for n in state.nodes():
        if isinstance(n, nodes.AccessNode) and n.data == data and state.in_degree(n) > 0:
            return n
    return None


def _hoist_tasklet_to_preheader(
    body: SDFGState,
    tasklet: nodes.Tasklet,
    preheader: SDFGState,
) -> bool:
    """Move ``tasklet`` and its input read chain from ``body`` into ``preheader``.

    The output AccessNode stays in ``body`` as a pure read feeding the existing
    consumers; its in-edges are removed since the value is now written in
    ``preheader``.
    """
    in_edges = list(body.in_edges(tasklet))
    out_edges = list(body.out_edges(tasklet))
    if len(out_edges) != 1:
        return False
    out_edge = out_edges[0]
    out_access = out_edge.dst
    if not isinstance(out_access, nodes.AccessNode):
        return False

    # Clone source reads in preheader (one per unique data name). Reuse an
    # already-hoisted producer's output node for the same data if one exists,
    # so a hoisted consumer connects to that producer (W -> D -> T) instead of
    # reading a fresh, unwritten D node. A disconnected read node would leave D
    # writerless in the preheader once dead-code elimination drops the
    # producer's now-orphaned write -- an uninitialized read (npbench
    # channel_flow hoists a split-tasklet chain ``__t0_split_N`` piecemeal, the
    # producer landing in the preheader on an earlier LICM pass than the
    # consumer).
    new_reads: Dict[str, nodes.AccessNode] = {}
    for ie in in_edges:
        d = ie.src.data
        if d not in new_reads:
            existing = _written_access_in_state(preheader, d)
            new_reads[d] = existing if existing is not None else preheader.add_access(d)

    # Clone the tasklet in preheader.
    new_tasklet = preheader.add_tasklet(
        name=tasklet.label + "_licm",
        inputs=set(tasklet.in_connectors.keys()),
        outputs=set(tasklet.out_connectors.keys()),
        code=tasklet.code.as_string if tasklet.code is not None else "",
        language=tasklet.language,
    )
    # Preserve connector dtypes.
    for c, t in tasklet.in_connectors.items():
        new_tasklet.in_connectors[c] = t
    for c, t in tasklet.out_connectors.items():
        new_tasklet.out_connectors[c] = t

    for ie in in_edges:
        preheader.add_edge(
            new_reads[ie.src.data],
            None,
            new_tasklet,
            ie.dst_conn,
            copy.deepcopy(ie.data),
        )

    new_out_access = preheader.add_access(out_access.data)
    preheader.add_edge(
        new_tasklet,
        out_edge.src_conn,
        new_out_access,
        None,
        copy.deepcopy(out_edge.data),
    )

    # Surgically remove the old tasklet + its edges from the body.
    for ie in in_edges:
        body.remove_edge(ie)
    body.remove_edge(out_edge)
    # Remove orphaned source reads.
    orphans = [ie.src for ie in in_edges]
    body.remove_node(tasklet)
    for src in orphans:
        if src in body.nodes() and body.degree(src) == 0:
            body.remove_node(src)
    # The output access node stays only if something in ``body`` still reads
    # it; with no remaining consumers it is now fully produced in the
    # preheader and would otherwise be left isolated (invalid SDFG).
    if out_access in body.nodes() and body.degree(out_access) == 0:
        body.remove_node(out_access)

    return True


# ---------------------------------------------------------------------------
# Map scope path
# ---------------------------------------------------------------------------


def _hoist_map_scope(state: SDFGState, me: nodes.MapEntry) -> int:
    """Hoist invariant tasklets out of a Map scope to the enclosing state."""
    sdfg: SDFG = state.sdfg
    variant_syms: Set[str] = set(me.map.params)

    mx = state.exit_node(me)
    inside_nodes = state.all_nodes_between(me, mx)
    if inside_nodes is None:
        return 0

    # Conservative alias seed: any data written anywhere in this scope, PLUS the
    # arrays the map itself writes. The map's outputs (e.g. ``a[i]``) flow out
    # through the map exit to AccessNodes OUTSIDE ``[me, mx]``, so
    # ``all_nodes_between`` does not see them; without this, a read of an array
    # the map also writes (e.g. the invariant ``a[0]`` in ``a[i] = a[0] + b[i]``)
    # would be wrongly treated as invariant and hoisted -- producing a malformed
    # whole-array-to-scalar copy. Matches the loop-region criterion (a written
    # container makes its reads variant).
    variant_data: Set[str] = set()
    for n in inside_nodes:
        if isinstance(n, nodes.AccessNode) and state.in_degree(n) > 0:
            variant_data.add(n.data)
    for e in state.out_edges(mx):
        if e.data is not None and e.data.data is not None:
            variant_data.add(e.data.data)

    count = 0
    while True:
        tasklet = _find_one_map_invariant_tasklet(state, me, variant_syms, variant_data)
        if tasklet is None:
            break
        if not _hoist_tasklet_out_of_map(state, me, tasklet, sdfg):
            break
        count += 1
    return count


def _find_one_map_invariant_tasklet(
    state: SDFGState,
    me: nodes.MapEntry,
    variant_syms: Set[str],
    variant_data: Set[str],
) -> Optional[nodes.Tasklet]:
    mx = state.exit_node(me)
    inside = state.all_nodes_between(me, mx) or set()
    scope_dict = state.scope_dict()
    for n in inside:
        if not isinstance(n, nodes.Tasklet):
            continue
        # Must be directly in this scope, not in a nested sub-scope.
        if scope_dict.get(n) is not me:
            continue
        if _is_map_tasklet_invariant(state, me, n, variant_syms, variant_data):
            return n
    return None


def _is_map_tasklet_invariant(
    state: SDFGState,
    me: nodes.MapEntry,
    tasklet: nodes.Tasklet,
    variant_syms: Set[str],
    variant_data: Set[str],
) -> bool:
    try:
        if tasklet.side_effects:
            return False
    except AttributeError:
        pass
    for oe in state.out_edges(tasklet):
        if oe.data is not None and oe.data.wcr is not None:
            return False

    # Code
    if _code_free_symbols(tasklet) & variant_syms:
        return False

    # Inputs must all arrive through MapEntry (connectors IN_x / OUT_x pairs).
    # Each in-edge to the tasklet must come from MapEntry, with a memlet whose
    # subset has no variant symbol; that memlet's outer subset (on the in-edge
    # to MapEntry) must also read only non-variant data.
    in_edges = list(state.in_edges(tasklet))
    for ie in in_edges:
        if ie.src is not me:
            return False
        if ie.data is not None and ie.data.subset is not None:
            if _subset_uses_any(ie.data.subset, variant_syms):
                return False
        # Find the matching outer edge to MapEntry (IN_x → OUT_x).
        if not ie.dst_conn:
            return False
        outer_conn = _matching_outer_conn(ie.src_conn)
        if outer_conn is None:
            return False
        outer = [e for e in state.in_edges_by_connector(me, outer_conn)]
        if not outer:
            return False
        outer_edge = outer[0]
        if not isinstance(outer_edge.src, nodes.AccessNode):
            return False
        if outer_edge.src.data in variant_data:
            return False

    # Output must go to a single AccessNode inside the scope, and that
    # AccessNode must be written only by this tasklet (so we can rewire).
    out_edges = list(state.out_edges(tasklet))
    if len(out_edges) != 1:
        return False
    oe = out_edges[0]
    if not isinstance(oe.dst, nodes.AccessNode):
        return False
    if oe.data is not None and oe.data.subset is not None:
        if _subset_uses_any(oe.data.subset, variant_syms):
            return False
    writers = sum(1 for n in state.data_nodes() if n.data == oe.dst.data and state.in_degree(n) > 0)
    if writers != 1:
        return False

    if _has_integer_div_mod_by_possibly_zero(tasklet):
        return False

    return True


def _matching_outer_conn(src_conn: Optional[str]) -> Optional[str]:
    """MapEntry's internal OUT_x connector pairs with the external IN_x."""
    if not src_conn or not src_conn.startswith("OUT_"):
        return None
    return "IN_" + src_conn[len("OUT_"):]


def _hoist_tasklet_out_of_map(
    state: SDFGState,
    me: nodes.MapEntry,
    tasklet: nodes.Tasklet,
    sdfg: SDFG,
) -> bool:
    in_edges = list(state.in_edges(tasklet))
    out_edges = list(state.out_edges(tasklet))
    if len(out_edges) != 1:
        return False
    oe = out_edges[0]
    out_access = oe.dst
    if not isinstance(out_access, nodes.AccessNode):
        return False

    # The inner out_access's data descriptor is shared with the SDFG-level
    # arrays dict, so just reading from it outside works once we pre-populate.
    # Build the hoisted subgraph in the enclosing state (same state, outside
    # the map scope).

    # Create new outer read nodes for each distinct upstream data container.
    outer_reads: Dict[str, nodes.AccessNode] = {}
    for ie in in_edges:
        outer_conn = _matching_outer_conn(ie.src_conn)
        if outer_conn is None:
            return False
        outer_edges = list(state.in_edges_by_connector(me, outer_conn))
        if not outer_edges:
            return False
        src_access = outer_edges[0].src
        if not isinstance(src_access, nodes.AccessNode):
            return False
        if src_access.data not in outer_reads:
            outer_reads[src_access.data] = state.add_access(src_access.data)

    # Clone the tasklet outside the scope.
    new_tasklet = state.add_tasklet(
        name=tasklet.label + "_licm",
        inputs=set(tasklet.in_connectors.keys()),
        outputs=set(tasklet.out_connectors.keys()),
        code=tasklet.code.as_string if tasklet.code is not None else "",
        language=tasklet.language,
    )
    for c, t in tasklet.in_connectors.items():
        new_tasklet.in_connectors[c] = t
    for c, t in tasklet.out_connectors.items():
        new_tasklet.out_connectors[c] = t

    for ie in in_edges:
        outer_conn = _matching_outer_conn(ie.src_conn)
        outer_edge = list(state.in_edges_by_connector(me, outer_conn))[0]
        src_access = outer_edge.src
        # Use the inner edge's memlet -- the exact element the tasklet reads
        # (e.g. ``c[j]``). It is the right subset because the tasklet is
        # map-invariant: the subset carries no map parameter, so reading that
        # same element directly from the outer AccessNode (outside the scope)
        # is valid. The outer edge's memlet is MapEntry's over-approximated bulk
        # movement into the whole map (e.g. ``c[0:LEN_1D]``); feeding that
        # whole-array subset to the scalar-copy tasklet produced a malformed
        # pointer-to-scalar assignment (``_out = _in`` with ``_in`` a pointer).
        state.add_edge(
            outer_reads[src_access.data],
            None,
            new_tasklet,
            ie.dst_conn,
            copy.deepcopy(ie.data),
        )

    # Output: write to an outer AccessNode of the same transient, then feed
    # it back into the map scope through a new connector pair.
    outer_out_access = state.add_access(out_access.data)
    state.add_edge(
        new_tasklet,
        oe.src_conn,
        outer_out_access,
        None,
        copy.deepcopy(oe.data),
    )

    # Surgically remove the old tasklet and its input fan-in from the scope,
    # then route the hoisted value back in through MapEntry BEFORE we prune
    # unused connector pairs. Order matters: while the inner consumer
    # (``out_access``) has no producer we'd have an inconsistent scope, and
    # any scope-dict recomputation would fail.
    for ie in in_edges:
        state.remove_edge(ie)
    state.remove_edge(oe)
    state.remove_node(tasklet)

    _route_into_scope(state, me, outer_out_access, out_access, sdfg)
    _prune_unused_map_connectors(state, me)

    # Invalidate any cached scope-dict since we mutated node/edge structure.
    state._scope_dict_toparent_cached = None

    return True


def _prune_unused_map_connectors(state: SDFGState, me: nodes.MapEntry) -> None:
    # Find IN_x connectors whose OUT_x has no outgoing edges to scope; drop
    # the pair + the outer feeding edge. Any AccessNodes that become isolated
    # after the cut are removed too.
    freed_sources: List[nodes.Node] = []
    for in_conn in list(me.in_connectors.keys()):
        if not in_conn.startswith("IN_"):
            continue
        out_conn = "OUT_" + in_conn[len("IN_"):]
        internal_edges = list(state.out_edges_by_connector(me, out_conn))
        if internal_edges:
            continue
        # No internal consumer — remove outer feed and connector pair.
        for oe in list(state.in_edges_by_connector(me, in_conn)):
            freed_sources.append(oe.src)
            state.remove_edge(oe)
        me.remove_in_connector(in_conn)
        if out_conn in me.out_connectors:
            me.remove_out_connector(out_conn)

    # Remove any now-isolated outer AccessNodes that were feeding the pruned
    # connector. AccessNodes with other uses stay.
    for src in freed_sources:
        if src in state.nodes() and state.degree(src) == 0 and isinstance(src, nodes.AccessNode):
            state.remove_node(src)


def _route_into_scope(
    state: SDFGState,
    me: nodes.MapEntry,
    outer_access: nodes.AccessNode,
    inner_access: nodes.AccessNode,
    sdfg: SDFG,
) -> None:
    """Add an outer→MapEntry→inner edge carrying ``outer_access``'s data.

    ``inner_access`` stays in place so existing consumers keep their edges.
    """
    data_name = outer_access.data
    # Pick a fresh connector name.
    base = data_name
    idx = 0
    in_conn = f"IN_{base}"
    while in_conn in me.in_connectors:
        idx += 1
        in_conn = f"IN_{base}_{idx}"
    out_conn = "OUT_" + in_conn[len("IN_"):]
    me.add_in_connector(in_conn)
    me.add_out_connector(out_conn)

    from dace import Memlet
    desc = sdfg.arrays[data_name]
    state.add_edge(outer_access, None, me, in_conn, Memlet.from_array(data_name, desc))
    state.add_edge(me, out_conn, inner_access, None, Memlet.from_array(data_name, desc))
