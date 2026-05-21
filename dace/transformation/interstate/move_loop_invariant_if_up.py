# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Hoist a loop-invariant guarding conditional out of its enclosing loop.

The inverse of ``MoveIfIntoLoop``: ``for k: { prep*; if c: body }`` becomes
``prep*; if c: { for k: body }`` when ``c`` (and the interstate-edge symbol
assignments it depends on) is provably invariant w.r.t. the loop. Applied to
a fixpoint, so an innermost invariant guard sifts all the way up through a
stack of nested loops (a mix of loops and maps -- maps are lowered to
``LoopRegion`` s by the canonicalize pipeline before this runs, so operating
on ``LoopRegion`` covers both).

Conservative -- only fires when:

* the loop body is a linear chain whose only non-empty block is the
  single-branch (no-else) ``ConditionalBlock``; empty boundary states are
  allowed and are dropped (they carry no effect, so the loop is a genuine
  no-op when ``c`` is false -> hoisting is value-preserving: ``c`` true =>
  the loop runs the body every iteration as before; ``c`` false => the loop
  did nothing observable before and is simply not entered after);
* the condition does not reference the loop variable nor any data/symbol
  written inside the loop, *except* symbols an interstate edge of the chain
  assigns from a provably-invariant expression -- those assignments are
  hoisted out *with* the guard so it still sees them;
* there are no other (non-invariant) interstate-edge assignments in the
  chain.

Anything else is a no-op. Linear-CFG assumption: the loop body is a plain
linear chain; blocks may have parents (the pass recurses into all regions).
"""
import copy
from typing import Any, Dict, List, Optional, Set, Tuple

from dace import SDFG
from dace import properties, symbolic
from dace.properties import CodeBlock
from dace.sdfg.sdfg import InterstateEdge
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion, LoopRegion, SDFGState
from dace.sdfg import nodes
from dace.sdfg.utils import set_nested_sdfg_parent_references
from dace.transformation import pass_pipeline as ppl, transformation


def _free(expr: str) -> Set[str]:
    """Free symbol names of a condition / assignment expression string.

    :param expr: A condition or interstate-assignment RHS expression.
    :returns: The set of free symbol names, empty if ``expr`` cannot be parsed.
    """
    try:
        return {str(s) for s in symbolic.pystr_to_symbolic(expr).free_symbols}
    except Exception:
        return set()


def _written(region: ControlFlowRegion) -> Set[str]:
    """Data containers and interstate-assigned symbols written inside ``region``.

    :param region: The region to scan (recursively).
    :returns: The set of written data-container and symbol names.
    """
    written: Set[str] = set()
    for e in region.all_interstate_edges():
        written |= set(e.data.assignments.keys())
    for st in region.all_states():
        for n in st.nodes():
            if isinstance(n, nodes.AccessNode) and st.in_degree(n) > 0:
                written.add(n.data)
    return written


def _linear_order(region: ControlFlowRegion) -> Optional[List]:
    """Order the blocks of ``region`` iff it is a plain linear chain.

    Edges must be unconditional (assignments are allowed -- they carry the
    condition prep).

    :param region: The region to linearize.
    :returns: The blocks in execution order, or ``None`` if ``region`` is not
        a plain linear chain of unconditional edges.
    """
    blocks = list(region.nodes())
    edges = list(region.edges())
    if not blocks or len(edges) != len(blocks) - 1:
        return None
    for e in edges:
        if e.data.condition.as_string not in ('1', 'True', '(1)'):
            return None
    succ = {e.src: e.dst for e in edges}
    order = [region.start_block]
    while order[-1] in succ:
        order.append(succ[order[-1]])
    return order if len(order) == len(blocks) else None


def _is_empty_state(b: ControlFlowRegion) -> bool:
    """Whether ``b`` is an ``SDFGState`` carrying no nodes.

    :param b: A block of a control-flow region.
    :returns: ``True`` iff ``b`` is an empty ``SDFGState``.
    """
    return isinstance(b, SDFGState) and b.number_of_nodes() == 0


def _enclosing_loops(loop: LoopRegion, sdfg: SDFG) -> List[LoopRegion]:
    """Collect every ``LoopRegion`` strictly enclosing ``loop``.

    :param loop: The inner loop whose ancestors are wanted.
    :param sdfg: The root SDFG (chain walk stops here).
    :returns: The enclosing loops, innermost first, up to the root SDFG.
    """
    out: List[LoopRegion] = []
    g = loop.parent_graph
    while g is not None and g is not sdfg:
        if isinstance(g, LoopRegion):
            out.append(g)
        g = getattr(g, 'parent_graph', None)
    return out


def _reads_outside(loop: LoopRegion, cb: ConditionalBlock) -> Set[str]:
    """Symbols read anywhere in ``loop`` outside the conditional block ``cb``.

    Covers other body blocks, iedge conditions and RHSes, and the loop's own
    init/condition/update. Used to decide which non-invariant iedge
    assignments may stay in the loop body when the guard ``cb`` is hoisted
    out: an assignment ``key = rhs`` whose ``key`` is NOT in this set is dead
    outside ``cb`` s branch, so its per-iteration value would be unused under
    ``not cond`` and keeping it in the loop body is safe.

    :param loop: The loop being analyzed.
    :param cb: The guarding conditional block being considered for hoisting.
    :returns: The set of symbol names read outside ``cb``.
    """
    reads: Set[str] = set()
    for blk in loop.nodes():
        if blk is cb:
            continue
        if isinstance(blk, SDFGState):
            try:
                reads |= {str(s) for s in blk.free_symbols}
            except Exception:
                pass
    for e in loop.edges():
        if not e.data.is_unconditional():
            try:
                reads |= {str(s) for s in e.data.condition.get_free_symbols()}
            except Exception:
                pass
        for rhs_val in e.data.assignments.values():
            reads |= _free(rhs_val)
    try:
        reads |= {str(s) for s in loop.loop_condition.get_free_symbols()}
    except Exception:
        pass
    if loop.init_statement is not None:
        try:
            reads |= {str(s) for s in loop.init_statement.get_free_symbols()}
        except Exception:
            pass
    if loop.update_statement is not None:
        try:
            reads |= {str(s) for s in loop.update_statement.get_free_symbols()}
        except Exception:
            pass
    return reads


def _match(
    sdfg: SDFG,
    require_full_hoist: bool = False
) -> Optional[Tuple[LoopRegion, ConditionalBlock, CodeBlock, List[Tuple[str, str]]]]:
    """Find a ``LoopRegion`` whose body is ``[empty*; if c; empty*]`` with a
    loop-invariant condition (plus a hoistable invariant assignment chain
    and any per-iteration assignments that are dead outside the branch).

    :param require_full_hoist: If set, only accept the candidate when the
        guard can be sifted out past *every* enclosing loop (the whole parent
        chain) -- if it would get stuck at some ancestor loop (its condition
        references that loop's variable or data it writes), do nothing at
        all. Used by the canonicalize pipeline, where a guard left stranded
        between not-perfectly-collapsed map levels is worse than not moving.
    :returns: ``(loop, cond_block, cond, hoist_assignments)`` or ``None``.
        ``hoist_assignments`` is the list of invariant ``(lhs, rhs)`` pairs
        that must move to the outer-guard's incoming edge; per-iteration
        assignments stay where they are inside the new (now-guarded) loop
        body and are NOT in this list -- the splice in ``_move`` keeps them
        in place automatically.
    """
    for loop in sdfg.all_control_flow_regions(recursive=True):
        if not isinstance(loop, LoopRegion):
            continue
        order = _linear_order(loop)
        if order is None:
            continue
        cbs = [b for b in order if isinstance(b, ConditionalBlock)]
        non_empty = [b for b in order if not _is_empty_state(b)]
        if len(cbs) != 1 or non_empty != cbs:
            # Safety: hoisting the conditional would also sweep any
            # non-empty sibling body block into the new outer-guarded
            # scope, dropping its execution under the not-taken path --
            # value-changing. Per-iteration iedge assignments on the
            # body chain ARE allowed (they live on EDGES, not blocks,
            # and have no observable side effect when their lhs is dead
            # outside the branch); see the iedge classification below.
            continue
        cb = cbs[0]
        if len(cb.branches) != 1:
            continue
        cond, branch = cb.branches[0]
        if cond is None or not isinstance(branch, ControlFlowRegion):
            continue

        lvar = str(loop.loop_variable)
        loop_w = _written(loop)
        outside_reads = _reads_outside(loop, cb)
        # Interstate-edge assignments on the loop's own chain. Split into:
        #   - invariant: hoist with the guard;
        #   - per-iteration: must be DEAD outside the branch (lhs not read
        #     anywhere except inside cb), otherwise refuse.
        chain_assignments: List[Tuple[str, str]] = []
        bad = False
        for e in loop.edges():
            for lhs, rhs in e.data.assignments.items():
                rf = _free(rhs)
                is_variant = (lvar in rf or bool(rf & loop_w) or lhs in loop_w - set(e.data.assignments))
                if not is_variant:
                    chain_assignments.append((lhs, rhs))
                    continue
                # Variant assignment: only legal to leave inside the loop
                # body if its lhs is not observed outside the branch.
                if lhs in outside_reads:
                    bad = True
                    break
            if bad:
                break
        if bad:
            continue

        assigned = {lhs for lhs, _ in chain_assignments}
        cfree = {str(s) for s in cond.get_free_symbols()}
        # The guard must be invariant: no loop variable, and no dependence on
        # loop-written data/symbols other than the hoistable chain symbols.
        if lvar in cfree:
            continue
        if (cfree & loop_w) - assigned:
            continue

        # All-or-nothing mode: only hoist if the guard can clear *every*
        # enclosing loop. If some ancestor loop's variable or written data
        # appears in the condition, the guard would stall there -- skip it.
        if require_full_hoist:
            stuck = False
            for anc in _enclosing_loops(loop, sdfg):
                avar = str(anc.loop_variable)
                aw = _written(anc)
                # The guard clears ``anc`` only if both the condition AND
                # every chain assignment it carries are invariant w.r.t.
                # ``anc`` -- a chain assignment whose RHS reads ``anc``'s
                # variable / written data (e.g. ``g_index = g[i]``) pins the
                # guard to ``anc`` even though the *condition* alone looks
                # invariant.
                if avar in cfree or ((cfree & aw) - assigned):
                    stuck = True
                    break
                if any(avar in _free(rhs) or (_free(rhs) & aw) for _, rhs in chain_assignments):
                    stuck = True
                    break
            if stuck:
                continue

        return loop, cb, cond, chain_assignments
    return None


@properties.make_properties
@transformation.explicit_cf_compatible
class MoveLoopInvariantIfUp(ppl.Pass):
    """Hoist a loop-invariant guarding conditional out of its loop (fixpoint).

    The inverse of ``MoveIfIntoLoop``. Repeatedly applied so an innermost
    invariant guard sifts all the way up through nested loops. The
    interstate-edge symbol-assignment chain the condition depends on is
    hoisted with it; emptied boundary states are dropped.

    :param require_full_hoist: All-or-nothing mode (for canonicalization):
        only hoist a guard when it can be sifted out past *every* enclosing
        loop. A guard that would stall between not-perfectly-collapsed
        scopes is left where it is.
    """
    CATEGORY: str = 'Canonicalization'

    require_full_hoist = properties.Property(
        dtype=bool, default=False, desc="Only hoist if the guard can clear every enclosing loop; else do nothing.")

    def __init__(self, require_full_hoist: bool = False):
        super().__init__()
        self.require_full_hoist = require_full_hoist

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.CFG

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self):
        return set()

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> Optional[int]:
        """Hoist invariant guards out of their loops until none remain.

        :param sdfg: The SDFG to transform in place.
        :returns: Number of guards hoisted, or ``None`` if none.
        """
        count = 0
        while True:
            m = _match(sdfg, self.require_full_hoist)
            if m is None:
                break
            self._move(*m)
            count += 1
        if count:
            set_nested_sdfg_parent_references(sdfg)
        return count or None

    @staticmethod
    def _move(loop: LoopRegion, cb: ConditionalBlock, cond: CodeBlock, hoist_assignments: List[Tuple[str, str]]):
        """Splice ``loop`` into ``[assign chain]; if cond: { loop' }`` in place.

        ``loop'`` is ``loop`` with its conditional block ``cb`` spliced out and
        replaced by ``cb`` s branch body. All other body blocks (and the
        per-iteration iedge assignments on them) are preserved.

        :param loop: The loop whose invariant guard is being hoisted.
        :param cb: The guarding conditional block inside ``loop``.
        :param cond: The (invariant) branch condition of ``cb``.
        :param hoist_assignments: Invariant iedge ``(lhs, rhs)`` assignments to
            move onto the outer guard's incoming edge.
        """
        parent = loop.parent_graph
        in_edges = list(parent.in_edges(loop))
        out_edges = list(parent.out_edges(loop))
        is_start = parent.start_block is loop

        cond, branch = copy.deepcopy(cb).branches[0]

        # loop' = a deepcopy of the loop with cb spliced out in place.
        new_loop = copy.deepcopy(loop)
        # Find the corresponding cb in the deepcopy by label match.
        new_cb_candidates = [b for b in new_loop.nodes() if isinstance(b, ConditionalBlock) and b.label == cb.label]
        assert len(new_cb_candidates) == 1, f'expected one ConditionalBlock named {cb.label!r} in deepcopy'
        new_cb = new_cb_candidates[0]
        cb_pred_edges = list(new_loop.in_edges(new_cb))
        cb_succ_edges = list(new_loop.out_edges(new_cb))
        cb_was_start = new_loop.start_block is new_cb
        # Drop edges to/from cb, then drop cb itself; preserve every other
        # block (in particular: per-iteration iedge assignments on the body
        # chain stay intact because they live on the OTHER edges).
        for e in cb_pred_edges + cb_succ_edges:
            new_loop.remove_edge(e)
        new_loop.remove_node(new_cb)
        # Splice branch body into new_loop.
        bb = list(branch.nodes())
        be = list(branch.edges())
        bstart = branch.start_block
        branch_sinks = [b for b in bb if branch.out_degree(b) == 0]
        for b in bb:
            new_loop.add_node(b, ensure_unique_name=True)
        for e in be:
            new_loop.add_edge(e.src, e.dst, copy.deepcopy(e.data))
        # Wire cb's predecessors to branch.start_block.
        for e in cb_pred_edges:
            new_loop.add_edge(e.src, bstart, copy.deepcopy(e.data))
        # Wire branch sinks to cb's successors.
        for sink in branch_sinks:
            for e in cb_succ_edges:
                new_loop.add_edge(sink, e.dst, copy.deepcopy(e.data))
        if cb_was_start:
            new_loop.start_block = new_loop.node_id(bstart)

        # Splice out empty boundary states left in ``new_loop`` (the
        # original loop may have had empty pre/post-cb states; with cb
        # gone they are now empty linear-chain hops contributing nothing).
        # Mirrors EmptyStateElimination's local-fixpoint splice.
        changed = True
        while changed:
            changed = False
            for st in list(new_loop.nodes()):
                if not isinstance(st, SDFGState) or st.number_of_nodes() != 0:
                    continue
                in_e = list(new_loop.in_edges(st))
                out_e = list(new_loop.out_edges(st))
                if any(not (e.data.is_unconditional() and not e.data.assignments) for e in in_e + out_e):
                    continue
                if len(out_e) == 1 and (in_e or new_loop.start_block is st):
                    succ = out_e[0].dst
                    for e in in_e:
                        new_loop.add_edge(e.src, succ, e.data)
                    for e in in_e + out_e:
                        new_loop.remove_edge(e)
                    was_start = new_loop.start_block is st
                    new_loop.remove_node(st)
                    if was_start:
                        new_loop.start_block = new_loop.node_id(succ)
                    changed = True
                    break
                if not out_e and in_e and new_loop.start_block is not st:
                    for e in in_e:
                        new_loop.remove_edge(e)
                    new_loop.remove_node(st)
                    changed = True
                    break

        # if cond: { loop' }
        inner = ControlFlowRegion(label=f"{loop.label}_body")
        inner.add_node(new_loop, is_start_block=True, ensure_unique_name=True)
        outer_cb = ConditionalBlock(label=f"{loop.label}_guard")
        outer_cb.add_branch(CodeBlock(cond.as_string), inner)

        parent.add_node(outer_cb, ensure_unique_name=True)
        # Hoisted invariant assignments go on the edge entering the guard.
        hoisted = dict(hoist_assignments)
        for e in in_edges:
            data = copy.deepcopy(e.data)
            data.assignments.update(hoisted)
            parent.add_edge(e.src, outer_cb, data)
        if not in_edges and hoisted:
            pre = parent.add_state(f"{loop.label}_hoist")
            parent.add_edge(pre, outer_cb, InterstateEdge(assignments=hoisted))
            if is_start:
                parent.start_block = parent.node_id(pre)
                is_start = False
        for e in out_edges:
            parent.add_edge(outer_cb, e.dst, copy.deepcopy(e.data))
        for e in in_edges + out_edges:
            parent.remove_edge(e)
        parent.remove_node(loop)
        if is_start:
            parent.start_block = parent.node_id(outer_cb)
