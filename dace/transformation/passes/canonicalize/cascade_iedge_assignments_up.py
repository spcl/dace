# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Cascade interstate-edge symbol assignments up out of enclosing loops.

The LICM analogue for interstate-edge symbol assignments: if an interstate
edge inside a loop body carries ``key = rhs`` where ``rhs`` is invariant
w.r.t. every enclosing ``LoopRegion`` (and the move would be observationally
neutral on every block between the new location and the old one), move the
assignment up to the outermost legal scope. This complements
:class:`~dace.transformation.passes.loop_invariant_code_motion.LoopInvariantCodeMotion`
(which hoists tasklet / map-scope invariant code) and
:class:`~dace.transformation.interstate.move_loop_invariant_if_up.MoveLoopInvariantIfUp`
(which hoists invariant guards).

The motivating shape is the Python-frontend symbol promotion ``kp1 = K + 1``
that the frontend coins for non-symbol bound expressions (the
``get_target_name`` heuristic in ``newast.py``). After canonicalize stages
remix the CFG, the promoted assignment can end up *inside* the loop whose
range reads it -- producing the body-assigns-loop-range-symbol shape that
``LoopToMap``'s ``can_be_applied`` refuses ("Loop range references symbol(s)
assigned by the loop body's interstate edges"). Cascading the invariant
assignment up restores ``LoopToMap`` eligibility and stops the per-iteration
re-assignment of values that never change.

Binding rule (set by the user, see ``CASCADE_UP_DESIGN.md``): *all-or-nothing
upward*. A one-level partial hoist that leaves the assignment inside a
different enclosing loop is forbidden -- the same scope-mismatch family of
bugs reappears one level higher. The pass therefore either moves an
assignment past every enclosing ``LoopRegion`` (to the first non-loop
ancestor CFG) or leaves it where it is.

Legality predicates (per assignment ``key = rhs`` on edge ``e`` in region
``cfg``):

* **L1 RHS invariance at D** -- every free symbol of ``rhs`` is defined
  *above* ``D`` (not introduced by any loop variable / interstate-edge
  assignment / tasklet write between ``D`` and ``e``).
* **L2** -- no write to a free symbol of ``rhs`` between ``D``'s entry and
  ``e`` (otherwise the move would change which ``rhs`` value materialises
  ``key``).
* **L3** -- no read of ``key`` between ``D``'s entry and ``e`` (otherwise
  the moved assignment would shadow a previously-undefined / outer value
  observed by those reads).
* **L4** -- no other write to ``key`` between ``D``'s entry and ``e``.
* **L5** -- ``e`` is unconditionally executed by ``D``: refuse if ``e``
  lives inside a ``ConditionalBlock`` branch within ``D`` (conservative;
  the user-permitted relaxation needs whole-program dataflow on ``key``
  and is left as a future refinement, ``CASCADE_UP_DESIGN.md``).
* **L6** -- if the move would cross one or more ``NestedSDFG`` boundaries,
  each crossing needs a ``symbol_mapping`` passthrough. v1 refuses to
  cross NSDFG boundaries (conservative); the next iteration will route
  through ``symbol_mapping`` and drop the now-shadowed inner declaration.

The pass is idempotent (no-op once nothing legal remains to hoist) and is
designed to be invoked at *multiple* pipeline positions -- after the move-
inward passes that may have buried invariant assignments inside loops, and
again before the parallelization stage so the ``LoopToMap`` refuse-check
sees a clean shape.
"""
from typing import Any, Dict, List, Optional, Set, Tuple

from dace import SDFG, symbolic
from dace.sdfg import nodes
from dace.sdfg.sdfg import InterstateEdge
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion, LoopRegion, SDFGState
from dace.sdfg.utils import set_nested_sdfg_parent_references
from dace.transformation import pass_pipeline as ppl, transformation


def _free(expr: str) -> Set[str]:
    """Free symbols of a string expression."""
    try:
        return {str(s) for s in symbolic.pystr_to_symbolic(expr).free_symbols}
    except Exception:
        return set()


def _region_writes(region: ControlFlowRegion) -> Tuple[Set[str], Set[str]]:
    """All symbols (interstate assignments) and data containers written
    anywhere inside ``region`` and its descendants.

    :returns: ``(assigned_symbols, written_data)``.
    """
    asyms: Set[str] = set()
    wdata: Set[str] = set()
    for e in region.all_interstate_edges():
        asyms.update(e.data.assignments.keys())
    for st in region.all_states():
        for n in st.nodes():
            if isinstance(n, nodes.AccessNode) and st.in_degree(n) > 0:
                wdata.add(n.data)
    return asyms, wdata


def _region_reads(region: ControlFlowRegion) -> Set[str]:
    """Symbols read by any interstate edge (assignment RHS or condition)
    anywhere inside ``region`` and its descendants. Tasklet reads of
    symbols are also included via ``free_symbols`` on each state.
    """
    syms: Set[str] = set()
    for e in region.all_interstate_edges():
        for rhs in e.data.assignments.values():
            syms |= _free(rhs)
        if not e.data.is_unconditional():
            try:
                syms |= {str(s) for s in e.data.condition.get_free_symbols()}
            except Exception:
                pass
    for st in region.all_states():
        try:
            syms |= {str(s) for s in st.free_symbols}
        except Exception:
            pass
    return syms


def _loop_variables(cfg: ControlFlowRegion, sdfg: SDFG) -> Set[str]:
    """Every loop variable of every ``LoopRegion`` strictly enclosing
    ``cfg`` up to (but not including) ``sdfg``."""
    out: Set[str] = set()
    g = cfg
    while g is not None and g is not sdfg:
        if isinstance(g, LoopRegion) and g.loop_variable:
            out.add(str(g.loop_variable))
        g = getattr(g, 'parent_graph', None)
    return out


def _enclosing_loops(cfg: ControlFlowRegion, sdfg: SDFG) -> List[LoopRegion]:
    """LoopRegions strictly enclosing ``cfg`` (innermost first)."""
    out: List[LoopRegion] = []
    g = cfg
    while g is not None and g is not sdfg:
        if isinstance(g, LoopRegion):
            out.append(g)
        g = getattr(g, 'parent_graph', None)
    return out


def _ancestor_chain(cfg: ControlFlowRegion, sdfg: SDFG) -> List[ControlFlowRegion]:
    """Chain ``[cfg, cfg.parent_graph, ..., sdfg]`` (inclusive at both ends)
    or empty if ``cfg`` is not transitively contained in ``sdfg``.
    """
    out: List[ControlFlowRegion] = []
    g = cfg
    while g is not None:
        out.append(g)
        if g is sdfg:
            return out
        g = getattr(g, 'parent_graph', None)
    return []


def _meets_binding_rule(dest: ControlFlowRegion, cfg: ControlFlowRegion, sdfg: SDFG) -> bool:
    """The all-or-nothing upward rule: ``dest`` must be outside every
    ``LoopRegion`` enclosing ``cfg``. Equivalently: ``dest`` is not a
    ``LoopRegion`` and no ``LoopRegion`` strictly contains ``dest``."""
    if isinstance(dest, LoopRegion):
        return False
    p = getattr(dest, 'parent_graph', None)
    while p is not None and p is not sdfg:
        if isinstance(p, LoopRegion):
            return False
        p = getattr(p, 'parent_graph', None)
    return True


def _between_regions(parent: ControlFlowRegion, child: ControlFlowRegion) -> List[ControlFlowRegion]:
    """Regions strictly between ``parent`` and ``child`` in the parent-graph
    chain (exclusive at both ends). For an immediate child, returns ``[]``.
    """
    out: List[ControlFlowRegion] = []
    g = getattr(child, 'parent_graph', None)
    while g is not None and g is not parent:
        out.append(g)
        g = getattr(g, 'parent_graph', None)
    return out


def _on_conditional_branch(cfg: ControlFlowRegion, dest: ControlFlowRegion, sdfg: SDFG) -> bool:
    """True iff the chain from ``cfg`` up to (but not including) ``dest``
    passes through a ``ConditionalBlock`` branch -- the iedge would not be
    unconditionally executed under ``dest`` (L5)."""
    g = cfg
    while g is not None and g is not dest:
        p = getattr(g, 'parent_graph', None)
        if isinstance(p, ConditionalBlock):
            return True
        g = p
    return False


def _predecessors_in(parent: ControlFlowRegion, child) -> Set:
    """Strict predecessors of ``child`` in ``parent`` (blocks reachable
    backwards from ``child`` via ``parent``'s edges, excluding ``child``).
    """
    out: Set = set()
    stack = [child]
    while stack:
        cur = stack.pop()
        for e in parent.in_edges(cur):
            if e.src in out or e.src is child:
                continue
            out.add(e.src)
            stack.append(e.src)
    return out


def _block_reads_symbols(block) -> Set[str]:
    """All symbols read anywhere inside ``block`` (state, CFR, etc.).
    Uses ``free_symbols`` where available; falls back to a manual scan of
    interstate-edge conditions and assignment RHSes for CFRs.
    """
    syms: Set[str] = set()
    if isinstance(block, SDFGState):
        try:
            syms |= {str(s) for s in block.free_symbols}
        except Exception:
            pass
        return syms
    if isinstance(block, ControlFlowRegion):
        for st in block.all_states():
            try:
                syms |= {str(s) for s in st.free_symbols}
            except Exception:
                pass
        for e in block.all_interstate_edges():
            for rhs in e.data.assignments.values():
                syms |= _free(rhs)
            if not e.data.is_unconditional():
                try:
                    syms |= {str(s) for s in e.data.condition.get_free_symbols()}
                except Exception:
                    pass
    return syms


def _block_writes(block) -> Tuple[Set[str], Set[str]]:
    """All ``(symbols-assigned, data-containers-written)`` anywhere inside
    ``block``. State-only block: tasklet writes to AccessNodes; CFR block:
    delegate to :func:`_region_writes`.
    """
    if isinstance(block, ControlFlowRegion):
        return _region_writes(block)
    asyms: Set[str] = set()
    wdata: Set[str] = set()
    if isinstance(block, SDFGState):
        for n in block.nodes():
            if isinstance(n, nodes.AccessNode) and block.in_degree(n) > 0:
                wdata.add(n.data)
    return asyms, wdata


def _legal_to_hoist_into(parent: ControlFlowRegion, child: ControlFlowRegion, edge, key: str, rhs: str,
                         rhs_syms: Set[str], sdfg: SDFG) -> bool:
    """Predicate: is it legal to take an assignment currently on ``edge``
    inside ``child`` and place it at the entry of ``parent``? (One step.)

    Checks L1 RHS-invariance and L2/L3/L4 (no intervening reads/writes of
    ``key`` or ``rhs_syms`` between ``parent``'s entry and ``edge``). L5 is
    handled by the caller's chain check. L6 is handled by refusing to cross
    SDFG boundaries (``parent_graph`` chain stops at the owning SDFG).
    """
    # L1: any new symbol introduced by ``child`` (its loop variable, or any
    # iedge assignment inside ``child`` other than the one being moved) that
    # the rhs would now reference is a violation.
    if isinstance(child, LoopRegion) and child.loop_variable and str(child.loop_variable) in rhs_syms:
        return False
    inner_asyms, inner_wdata = _region_writes(child)
    inner_asyms.discard(key)  # discount the assignment we're moving
    if rhs_syms & inner_asyms:
        return False
    if rhs_syms & inner_wdata:
        return False  # a tasklet inside child writes to a container our rhs reads

    # L2/L3/L4: examine strict predecessors of ``child`` in ``parent``.
    # The new iedge will sit on ``child``'s in-edges; only blocks that can
    # execute BEFORE ``child`` could observe a difference. Blocks after
    # ``child`` saw the assignment via the original edge anyway.
    preds = _predecessors_in(parent, child)
    for b in preds:
        if key in _block_reads_symbols(b):
            return False  # L3: a predecessor would observe the moved assignment
        b_asyms, b_wdata = _block_writes(b)
        if key in b_asyms or key in b_wdata:
            return False  # L4: predecessor writes key
        if (rhs_syms & b_asyms) or (rhs_syms & b_wdata):
            return False  # L2: predecessor writes a rhs symbol

    # Also any iedge in parent (other than the in-edges of ``child``) that
    # writes ``key`` or a ``rhs_sym`` is a violation. ``child``'s own in-
    # edges are fine -- they are where the hoist lands. An existing iedge
    # that assigns ``key = rhs`` (the exact same rhs) is OK: it just
    # means an earlier hoist of a sibling assignment already placed the
    # invariant value; the new one redundant-but-harmless co-exists, and
    # subsequent cleanup (or another canonicalize iteration) can dedupe.
    for e in parent.edges():
        if e.dst is child:
            continue
        for lhs, e_rhs in e.data.assignments.items():
            if lhs == key:
                if str(e_rhs) == str(rhs):
                    continue  # same key/rhs already hoisted by a sibling
                return False
            if lhs in rhs_syms:
                return False
    return True


def _find_destination(edge_region: ControlFlowRegion, key: str, rhs: str, sdfg: SDFG) -> Optional[ControlFlowRegion]:
    """Walk up the ``parent_graph`` chain from ``edge_region`` to find the
    outermost ancestor ``D`` where the move is legal under L1-L6. Returns
    ``None`` if the binding all-or-nothing rule is not met or if no move
    is legal (D == ``edge_region``).
    """
    rhs_syms = _free(rhs)
    dest: ControlFlowRegion = edge_region
    walker: ControlFlowRegion = edge_region
    while True:
        parent = getattr(walker, 'parent_graph', None)
        if parent is None:
            break
        if not isinstance(parent, ControlFlowRegion):
            break  # crossed out of SDFG (would need L6 NSDFG passthrough)
        if not _legal_to_hoist_into(parent, walker, None, key, rhs, rhs_syms, sdfg):
            break
        dest = parent
        walker = parent

    if dest is edge_region:
        return None
    if _on_conditional_branch(edge_region, dest, sdfg):
        return None  # L5: at least one step crossed a ConditionalBlock branch
    if not _meets_binding_rule(dest, edge_region, sdfg):
        return None
    return dest


def _place_assignment_at(dest: ControlFlowRegion, child: ControlFlowRegion, key: str, rhs: str) -> None:
    """Add ``key = rhs`` so it dominates ``child`` inside ``dest``.

    Mirrors the placement strategy of
    :class:`~dace.transformation.interstate.move_loop_invariant_if_up.MoveLoopInvariantIfUp._move`:
    put the assignment on every in-edge of ``child`` within ``dest``;
    if ``child`` has no in-edges (it is ``dest``'s start block), prepend a
    fresh hoist state with the assignment on its outgoing edge.
    """
    in_edges = list(dest.in_edges(child))
    if in_edges:
        for e in in_edges:
            e.data.assignments[key] = rhs
        return
    is_start = dest.start_block is child
    pre = dest.add_state(f'{child.label}_iedge_hoist')
    dest.add_edge(pre, child, InterstateEdge(assignments={key: rhs}))
    if is_start:
        dest.start_block = dest.node_id(pre)


def _drop_inner_symbol_declarations(sdfg: SDFG, key: str, dest: ControlFlowRegion) -> None:
    """If ``key`` is also declared in an inner NestedSDFG inside ``dest``,
    leave the declaration (we do not cross NSDFG boundaries in v1, so this
    is a no-op). Kept as a placeholder for the cross-NSDFG extension."""
    return


def _direct_child(dest: ControlFlowRegion, edge_region: ControlFlowRegion) -> ControlFlowRegion:
    """Walk up from ``edge_region`` to find the immediate child of ``dest``."""
    g = edge_region
    while getattr(g, 'parent_graph', None) is not dest:
        g = getattr(g, 'parent_graph', None)
        if g is None:
            raise RuntimeError('dest is not an ancestor of edge_region')
    return g


def _cascade_once(sdfg: SDFG) -> int:
    """One sweep across the SDFG; returns the number of assignments moved."""
    moved = 0
    for cfg in list(sdfg.all_control_flow_regions(recursive=True)):
        for edge in list(cfg.edges()):
            if not edge.data.assignments:
                continue
            # Snapshot keys -- we mutate the dict as we go.
            for key, rhs in list(edge.data.assignments.items()):
                dest = _find_destination(cfg, key, rhs, sdfg)
                if dest is None:
                    continue
                child = _direct_child(dest, cfg)
                # Atomic move: only mutate once the destination is fixed.
                del edge.data.assignments[key]
                _place_assignment_at(dest, child, key, rhs)
                _drop_inner_symbol_declarations(sdfg, key, dest)
                moved += 1
    return moved


@transformation.explicit_cf_compatible
class CascadeInterstateEdgeAssignmentsUp(ppl.Pass):
    """Cascade invariant interstate-edge symbol assignments past every
    enclosing ``LoopRegion`` (all-or-nothing upward; fixpoint).

    Standalone, idempotent, can be invoked at multiple pipeline positions.
    """
    CATEGORY: str = 'Canonicalization'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.InterstateEdges | ppl.Modifies.CFG

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self):
        return set()

    def apply_pass(self, sdfg: SDFG, _pipeline_results: Dict[str, Any]) -> Optional[int]:
        """Repeatedly hoist legal interstate-edge assignments until a fixpoint.

        :param sdfg: The SDFG to transform in place.
        :returns: Total number of assignments moved, or ``None`` if none.
        """
        total = 0
        while True:
            n = _cascade_once(sdfg)
            if not n:
                break
            total += n
        if total:
            set_nested_sdfg_parent_references(sdfg)
        return total or None
