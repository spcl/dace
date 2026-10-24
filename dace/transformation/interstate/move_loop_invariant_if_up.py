# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Hoist a loop-invariant guarding conditional out of its enclosing loop.

The inverse of ``MoveIfIntoLoop`` -- loop unswitching: ``for k: { prep*; if c: A else: B }`` becomes
``prep*; if c: { for k: A } else: { for k: B }`` when every branch condition (and the interstate-edge symbol
assignments it depends on) is provably invariant w.r.t. the loop, so every iteration takes the same branch. Any
number of branches, with or without an ``else``; a branch with no work becomes an empty state, not a loop.
Applied to a fixpoint, so an innermost invariant guard sifts all the way up through a stack of nested loops.
The canonical form keeps guards innermost; this runs at specialization (``hoist_invariant_guards``).

Conservative -- only fires when:

* the loop body is a linear chain whose only non-empty block is the
  ``ConditionalBlock``; empty boundary states are allowed and are dropped
  (they carry no effect, so each branch's copy of the loop runs exactly the
  work that branch did in every iteration; without an ``else`` the not-taken
  case did nothing observable and the loop is simply not entered). A guard
  that shares its body with siblings it is data-independent of is first
  distributed into a loop of its own (``for k: { A; if c: B }`` ->
  ``for k: { A }; for k: { if c: B }``) using ``LoopFission``'s independence
  criterion, then hoisted -- otherwise a loop-fusion pass that welds a
  guard-only loop back onto a sibling would put the guard permanently out
  of reach;
* the condition does not reference the loop variable nor any data/symbol
  written inside the loop, *except* symbols an interstate edge of the chain
  assigns from a provably-invariant expression -- those assignments are
  hoisted out *with* the guard so it still sees them;
* there are no other (non-invariant) interstate-edge assignments in the
  chain.

Equal guards around unguarded work unswitch together: ``for k: { if c: A; B; if c: C }`` becomes
``if c: { for k: { A; B; C } } else: { for k: { B } }`` when every guard is a single ``if`` branch, the conditions
are provably equal (:func:`~dace.symbolic.guards_equal`) and each is invariant by the rules above; an ``else`` loop
left without work is dropped.

Anything else is a no-op. Linear-CFG assumption: the loop body is a plain
linear chain; blocks may have parents (the pass recurses into all regions).
"""

import copy
import re
from typing import Any

from dace import SDFG, properties, symbolic
from dace.ordered import OrderedSet
from dace.properties import CodeBlock
from dace.sdfg import nodes
from dace.sdfg.sdfg import InterstateEdge
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion, LoopRegion, SDFGState, rehome_claimed_block
from dace.transformation import pass_pipeline as ppl
from dace.transformation import transformation

# Shared soundness kernel for the loop distribution below -- the same import route
# ``fuse_loops.py`` uses, so fission, fusion and this hoist can never disagree.
from dace.transformation.passes.loop_fission import LoopFission, _independent_block_groups


def _free(expr: str) -> set[str]:
    """Free symbol names of a condition / assignment expression string.

    :param expr: A condition or interstate-assignment RHS expression.
    :returns: The set of free symbol names, empty if ``expr`` cannot be parsed.
    """
    try:
        return {str(s) for s in symbolic.pystr_to_symbolic(expr).free_symbols}
    except Exception:
        return set()


def _written(region: ControlFlowRegion) -> set[str]:
    """Data containers and interstate-assigned symbols written inside ``region``.

    :param region: The region to scan (recursively).
    :returns: The set of written data-container and symbol names.
    """
    written: set[str] = set()
    for e in region.all_interstate_edges():
        written |= set(e.data.assignments.keys())
    for st in region.states():
        for n in st.nodes():
            if isinstance(n, nodes.AccessNode) and st.in_degree(n) > 0:
                written.add(n.data)
    return written


def _linear_order(region: ControlFlowRegion) -> list | None:
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
        if e.data.condition.as_string not in ("1", "True", "(1)"):
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


def _enclosing_loops(loop: LoopRegion, sdfg: SDFG) -> list[LoopRegion]:
    """Collect every ``LoopRegion`` strictly enclosing ``loop``.

    :param loop: The inner loop whose ancestors are wanted.
    :param sdfg: The root SDFG (chain walk stops here).
    :returns: The enclosing loops, innermost first, up to the root SDFG.
    """
    out: list[LoopRegion] = []
    g = loop.parent_graph
    while g is not None and g is not sdfg:
        if isinstance(g, LoopRegion):
            out.append(g)
        g = getattr(g, "parent_graph", None)
    return out


def _reads_outside(loop: LoopRegion, cb: ConditionalBlock) -> set[str]:
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
    reads: set[str] = set()
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


def strippable_prep(loop: LoopRegion, hoisted: list[tuple[str, str]]) -> dict[str, str]:
    """The hoisted prep assignments that may be deleted from the loop body.

    Deleting the body copy is value-preserving exactly when the name has no OTHER definition
    inside the loop -- no second interstate assignment with a different right-hand side, and no
    data write to a container of that name. Then every read in the loop observed this one
    invariant expression before and observes the identical outer binding after, because the
    guard's entry edge dominates the whole loop.

    :param loop: The loop the guard is leaving (still intact).
    :param hoisted: The invariant ``(lhs, rhs)`` pairs moved onto the guard's entry edge.
    :returns: The subset of ``hoisted``, as a ``lhs -> rhs`` map, safe to delete from the body.
    """
    defined: dict[str, str] = {}
    conflicting: OrderedSet[str] = OrderedSet()
    for e in loop.all_interstate_edges():
        for lhs, rhs in e.data.assignments.items():
            if defined.setdefault(lhs, rhs) != rhs:
                conflicting.add(lhs)
    for st in loop.states():
        for n in st.nodes():
            if isinstance(n, nodes.AccessNode) and st.in_degree(n) > 0:
                conflicting.add(n.data)
    return {lhs: rhs for lhs, rhs in hoisted if lhs not in conflicting}


def _hoistable(
    sdfg: SDFG, loop: LoopRegion, cb: ConditionalBlock, require_full_hoist: bool, loop_w: set[str] | None = None
) -> tuple[CodeBlock, list[tuple[str, str]]] | None:
    """Whether the guard ``cb`` is loop-invariant enough to leave ``loop``.

    Judged on the WHOLE loop (every block, every iedge), so the answer stays
    valid -- and only gets more permissive -- if ``loop`` is later distributed
    down to just ``cb`` by :func:`_split_guard_loop`.

    :param sdfg: The root SDFG (bounds the ancestor walk).
    :param loop: The loop the guard sits in.
    :param cb: The guarding conditional block; every branch condition must be invariant (an ``else`` has none).
    :param require_full_hoist: If set, only accept the candidate when the
        guard can be sifted out past *every* enclosing loop (the whole parent
        chain) -- if it would get stuck at some ancestor loop (its condition
        references that loop's variable or data it writes), do nothing at
        all. Used by the canonicalize pipeline, where a guard left stranded
        between not-perfectly-collapsed map levels is worse than not moving.
    :returns: ``(cond, hoist_assignments)`` or ``None``. ``hoist_assignments``
        is the list of invariant ``(lhs, rhs)`` pairs that must move to the
        outer-guard's incoming edge; per-iteration assignments stay where they
        are inside the new (now-guarded) loop body and are NOT in this list --
        the splice in ``_move`` keeps them in place automatically.
    """
    conds = [cond for cond, _ in cb.branches if cond is not None]
    if not conds or any(not isinstance(branch, ControlFlowRegion) for _, branch in cb.branches):
        return None

    lvar = str(loop.loop_variable)
    if loop_w is None:
        loop_w = _written(loop)
    outside_reads = _reads_outside(loop, cb)
    # Interstate-edge assignments on the loop's own chain. Split into:
    #   - invariant: hoist with the guard;
    #   - per-iteration: must be DEAD outside the branch (lhs not read
    #     anywhere except inside cb), otherwise refuse.
    chain_assignments: list[tuple[str, str]] = []
    for e in loop.edges():
        for lhs, rhs in e.data.assignments.items():
            rf = _free(rhs)
            is_variant = lvar in rf or bool(rf & loop_w) or lhs in loop_w - set(e.data.assignments)
            if not is_variant:
                chain_assignments.append((lhs, rhs))
                continue
            # Variant assignment: only legal to leave inside the loop
            # body if its lhs is not observed outside the branch.
            if lhs in outside_reads:
                return None

    assigned = {lhs for lhs, _ in chain_assignments}
    cfree = {str(s) for cond in conds for s in cond.get_free_symbols()}
    # The guard must be invariant: no loop variable, and no dependence on
    # loop-written data/symbols other than the hoistable chain symbols.
    if lvar in cfree:
        return None
    if (cfree & loop_w) - assigned:
        return None

    # All-or-nothing mode: only hoist if the guard can clear *every*
    # enclosing loop. If some ancestor loop's variable or written data
    # appears in the condition, the guard would stall there -- skip it.
    if require_full_hoist:
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
                return None
            if any(avar in _free(rhs) or (_free(rhs) & aw) for _, rhs in chain_assignments):
                return None

    return conds[0], chain_assignments


def match_loop(
    sdfg: SDFG, loop: LoopRegion, require_full_hoist: bool = False
) -> tuple[LoopRegion, ConditionalBlock, CodeBlock, list[tuple[str, str]]] | None:
    """Match ``loop`` as a body ``[empty*; if c; empty*]`` with a loop-invariant
    condition (plus a hoistable invariant assignment chain and any
    per-iteration assignments that are dead outside the branch).

    :param sdfg: The root SDFG (bounds the ancestor walk).
    :param loop: The loop to match.
    :param require_full_hoist: See :func:`_hoistable`.
    :returns: ``(loop, cond_block, cond, hoist_assignments)`` or ``None``.
    """
    order = _linear_order(loop)
    if order is None:
        return None
    cbs = [b for b in order if isinstance(b, ConditionalBlock)]
    non_empty = [b for b in order if not _is_empty_state(b)]
    if len(cbs) != 1 or non_empty != cbs:
        # Safety: hoisting the conditional would also sweep any
        # non-empty sibling body block into the new outer-guarded
        # scope, dropping its execution under the not-taken path --
        # value-changing. Such a body is handled by distributing the
        # sibling out first (:func:`_split_guard_loop`). Per-iteration
        # iedge assignments on the body chain ARE allowed (they live on
        # EDGES, not blocks, and have no observable side effect when
        # their lhs is dead outside the branch).
        return None
    h = _hoistable(sdfg, loop, cbs[0], require_full_hoist)
    return None if h is None else (loop, cbs[0], h[0], h[1])


def match_equal_guards(
    sdfg: SDFG, loop: LoopRegion, require_full_hoist: bool = False
) -> tuple[LoopRegion, list[ConditionalBlock], list[tuple[str, str]]] | None:
    """``(loop, guards, hoist_assignments)`` when the body chain of ``loop`` holds two or more single-branch guards
    with provably equal conditions, each invariant by :func:`_hoistable`; else ``None``."""
    order = _linear_order(loop)
    if order is None:
        return None
    guards = [b for b in order if isinstance(b, ConditionalBlock)]
    if len(guards) < 2 or any(len(g.branches) != 1 or g.branches[0][0] is None for g in guards):
        return None
    if len({g.label for g in guards}) != len(guards):
        return None
    written = _written(loop)
    hoisted: list[tuple[str, str]] = []
    for guard in guards:
        verdict = _hoistable(sdfg, loop, guard, require_full_hoist, written)
        if verdict is None:
            return None
        hoisted = verdict[1]
    # Compared with the hoisted prep substituted: ``g_index > 0`` and ``g_index_0 > 0`` both read ``g[0]``.
    conditions = [expand_prep(g.branches[0][0].as_string, hoisted) for g in guards]
    if not all(symbolic.guards_equal(conditions[0], c) for c in conditions[1:]):
        return None
    return loop, guards, hoisted


def expand_prep(condition: str, hoisted: list[tuple[str, str]]) -> str:
    """``condition`` with every hoisted prep symbol replaced by its defining expression (one level)."""
    for lhs, rhs in hoisted:
        condition = re.sub(rf"\b{re.escape(lhs)}\b", f"({rhs})", condition)
    return condition


def find_match(
    sdfg: SDFG, require_full_hoist: bool = False
) -> tuple[LoopRegion, ConditionalBlock, CodeBlock, list[tuple[str, str]]] | None:
    """Find a loop that :func:`match_loop` matches."""
    for loop in sdfg.all_control_flow_regions(recursive=True):
        if isinstance(loop, LoopRegion):
            m = match_loop(sdfg, loop, require_full_hoist)
            if m is not None:
                return m
    return None


def _split_guard_loop(sdfg: SDFG, require_full_hoist: bool) -> bool:
    """Distribute one loop so its invariant guard ends up alone in a loop of its own.

    ``for k: { A; if c: B }`` -> ``for k: { A }; for k: { if c: B }``, legal exactly when
    ``A`` and the guard share no written data container -- ``LoopFission``'s own
    independence criterion, reused rather than re-derived. Needed because the guard is
    otherwise unreachable: a fusion pass that welds a guard-only loop back onto a sibling
    (canonicalize's ``loop_fuse`` stage does, for locality) leaves the guard with a
    non-empty sibling block, which :func:`find_match` must refuse forever.

    Only fires when the guard would then actually hoist -- the split is pure churn
    otherwise.

    :param sdfg: The SDFG to search and transform in place.
    :param require_full_hoist: See :func:`_hoistable`.
    :returns: ``True`` if a loop was distributed.
    """
    for loop in sdfg.all_control_flow_regions(recursive=True):
        if not isinstance(loop, LoopRegion):
            continue
        cbs = [b for b in loop.nodes() if isinstance(b, ConditionalBlock)]
        if len(cbs) != 1 or _linear_order(loop) is None:
            continue
        if _hoistable(sdfg, loop, cbs[0], require_full_hoist) is None:
            continue
        groups = _independent_block_groups(loop)
        if groups is None or not any(len(g) == 1 and g[0] is cbs[0] for g in groups):
            continue
        # The split reorders execution -- all of ``A``, then all of ``B`` -- and the
        # independence criterion only sees data containers, so refuse around anything
        # whose order relative to other side effects is observable.
        if any(
            isinstance(n, nodes.CodeNode) and n.has_ordered_side_effects(st.sdfg)
            for st in loop.states()
            for n in st.nodes()
        ):
            continue
        # Each per-group clone is re-homed, nested SDFGs included, as ``add_node`` claims it.
        LoopFission._fission_blocks(loop, groups)
        return True
    return False


def splice_guard(region: ControlFlowRegion, guard: ConditionalBlock, take: bool) -> None:
    """Replace the single-branch ``guard`` in ``region`` by its branch body (``take``) or by an empty state."""
    in_edges = list(region.in_edges(guard))
    out_edges = list(region.out_edges(guard))
    was_start = region.start_block is guard
    for e in in_edges + out_edges:
        region.remove_edge(e)
    region.remove_node(guard)
    if take:
        body = guard.branches[0][1]
        blocks, edges, start = list(body.nodes()), list(body.edges()), body.start_block
        sinks = [b for b in blocks if body.out_degree(b) == 0]
        for b in blocks:
            region.add_node(b, ensure_unique_name=True)
        for e in edges:
            region.add_edge(e.src, e.dst, e.data)
    else:
        start = region.add_state(f"{guard.label}_skip")
        sinks = [start]
    for e in in_edges:
        region.add_edge(e.src, start, e.data)
    for sink in sinks:
        for e in out_edges:
            region.add_edge(sink, e.dst, copy.deepcopy(e.data))
    if was_start:
        region.start_block = region.node_id(start)


def drop_empty_states(region: ControlFlowRegion) -> None:
    """Splice out empty states on plain (unconditional, assignment-free) edges of ``region``."""
    changed = True
    while changed:
        changed = False
        for st in list(region.nodes()):
            if not isinstance(st, SDFGState) or st.number_of_nodes() != 0:
                continue
            in_e = list(region.in_edges(st))
            out_e = list(region.out_edges(st))
            if any(not (e.data.is_unconditional() and not e.data.assignments) for e in in_e + out_e):
                continue
            if len(out_e) == 1 and (in_e or region.start_block is st):
                succ = out_e[0].dst
                for e in in_e:
                    region.add_edge(e.src, succ, e.data)
                for e in in_e + out_e:
                    region.remove_edge(e)
                was_start = region.start_block is st
                region.remove_node(st)
                if was_start:
                    region.start_block = region.node_id(succ)
                changed = True
                break
            if not out_e and in_e and region.start_block is not st:
                for e in in_e:
                    region.remove_edge(e)
                region.remove_node(st)
                changed = True
                break


@properties.make_properties
@transformation.explicit_cf_compatible
class MoveLoopInvariantIfUp(ppl.Pass):
    """Hoist a loop-invariant guarding conditional out of its loop (fixpoint).

    The inverse of ``MoveIfIntoLoop``. Repeatedly applied so an innermost
    invariant guard sifts all the way up through nested loops. The
    interstate-edge symbol-assignment chain the condition depends on is
    hoisted with it; emptied boundary states are dropped. A guard sharing its
    loop body with data-independent siblings is distributed into its own loop
    first (:func:`_split_guard_loop`).

    :param require_full_hoist: All-or-nothing mode (for canonicalization):
        only hoist a guard when it can be sifted out past *every* enclosing
        loop. A guard that would stall between not-perfectly-collapsed
        scopes is left where it is.
    """

    CATEGORY: str = "Canonicalization"

    require_full_hoist = properties.Property(
        dtype=bool, default=False, desc="Only hoist if the guard can clear every enclosing loop; else do nothing."
    )

    def __init__(self, require_full_hoist: bool = False):
        super().__init__()
        self.require_full_hoist = require_full_hoist

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.CFG

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self):
        return set()

    def apply_pass(self, sdfg: SDFG, pipeline_results: dict[str, Any]) -> int | None:
        """Hoist invariant guards out of their loops until none remain.

        :param sdfg: The SDFG to transform in place.
        :returns: Number of guards hoisted, or ``None`` if none.
        """
        count = 0
        while True:
            m = find_match(sdfg, self.require_full_hoist)
            if m is None:
                equal = next(
                    (
                        e
                        for loop in sdfg.all_control_flow_regions(recursive=True)
                        if isinstance(loop, LoopRegion)
                        and (e := match_equal_guards(sdfg, loop, self.require_full_hoist)) is not None
                    ),
                    None,
                )
                if equal is not None:
                    self._move_equal_guards(*equal)
                    count += 1
                    continue
                # The guard may be sharing its body with independent siblings: split
                # them apart and re-match. Terminates -- a split leaves the guard alone
                # in its own single-block clone, which is no longer splittable.
                if not _split_guard_loop(sdfg, self.require_full_hoist):
                    break
                continue
            self._move(*m)
            count += 1
        return count or None

    @staticmethod
    def hoist(loop: LoopRegion) -> bool:
        """Hoist the loop-invariant guard that is ``loop``'s only statement out of it.

        :param loop: The loop whose guard to hoist.
        :returns: Whether it moved; if not, the graph is unchanged.
        """
        m = match_loop(loop.sdfg, loop)
        if m is not None:
            MoveLoopInvariantIfUp._move(*m)
        return m is not None

    @staticmethod
    def _move(loop: LoopRegion, cb: ConditionalBlock, cond: CodeBlock, hoist_assignments: list[tuple[str, str]]):
        """Splice ``loop`` into ``[assign chain]; if c1: { loop_1 } elif c2: { loop_2 } ... else: { loop_n }`` in place.

        ``loop_k`` is ``loop`` with its conditional block ``cb`` replaced by branch ``k``'s body -- loop
        unswitching: the invariant condition picks the same branch in every iteration, so each branch runs as a loop
        of its own. A branch with no work becomes an empty state rather than a loop that does nothing.

        :param loop: The loop whose invariant guard is being hoisted.
        :param cb: The guarding conditional block inside ``loop``.
        :param cond: The first branch condition (kept for the call signature; every branch is used).
        :param hoist_assignments: Invariant iedge ``(lhs, rhs)`` assignments to move onto the outer guard's incoming
            edge.
        """
        del cond
        parent = loop.parent_graph
        in_edges = list(parent.in_edges(loop))
        out_edges = list(parent.out_edges(loop))
        is_start = parent.start_block is loop

        outer_cb = ConditionalBlock(label=f"{loop.label}_guard")
        for index, (branch_cond, branch) in enumerate(cb.branches):
            inner = ControlFlowRegion(label=f"{loop.label}_body_{index}" if index else f"{loop.label}_body")
            if all(_is_empty_state(b) for b in branch.nodes()):
                inner.add_state(f"{loop.label}_skip_{index}", is_start_block=True)
            else:
                unswitched = MoveLoopInvariantIfUp._loop_with_branch(loop, cb, index, hoist_assignments)
                inner.add_node(unswitched, is_start_block=True, ensure_unique_name=True)
            outer_cb.add_branch(None if branch_cond is None else CodeBlock(branch_cond.as_string), inner)

        for e in in_edges + out_edges:
            parent.remove_edge(e)
        parent.remove_node(loop)
        MoveLoopInvariantIfUp._install_guard(
            parent, loop.label, outer_cb, hoist_assignments, in_edges, out_edges, is_start
        )

    @staticmethod
    def _install_guard(
        parent: ControlFlowRegion,
        label: str,
        outer_cb: ConditionalBlock,
        hoist_assignments: list[tuple[str, str]],
        in_edges: list,
        out_edges: list,
        is_start: bool,
    ) -> None:
        """Wire ``outer_cb`` into ``parent`` along the removed loop's ``in_edges`` / ``out_edges``, the hoisted
        assignments on its entry edge."""
        parent.add_node(outer_cb, ensure_unique_name=True)
        # Hoisted invariant assignments go on the edge entering the guard.
        hoisted = dict(hoist_assignments)
        for e in in_edges:
            data = copy.deepcopy(e.data)
            data.assignments.update(hoisted)
            parent.add_edge(e.src, outer_cb, data)
        if not in_edges and hoisted:
            pre = parent.add_state(f"{label}_hoist")
            parent.add_edge(pre, outer_cb, InterstateEdge(assignments=hoisted))
            if is_start:
                parent.start_block = parent.node_id(pre)
                is_start = False
        for e in out_edges:
            parent.add_edge(outer_cb, e.dst, copy.deepcopy(e.data))
        if is_start:
            parent.start_block = parent.node_id(outer_cb)

    @staticmethod
    def _move_equal_guards(loop: LoopRegion, guards: list[ConditionalBlock], hoist_assignments: list[tuple[str, str]]):
        """Replace ``loop`` with ``if c: { loop, every guard inlined } [else: { loop, every guard removed }]``."""
        parent = loop.parent_graph
        in_edges = list(parent.in_edges(loop))
        out_edges = list(parent.out_edges(loop))
        is_start = parent.start_block is loop
        strip = strippable_prep(loop, hoist_assignments)
        labels = [g.label for g in guards]
        cond = guards[0].branches[0][0]
        skipped = copy.deepcopy(loop)
        for guard in guards:
            splice_guard(loop, guard, take=True)
        for label in labels:
            splice_guard(
                skipped,
                next(b for b in skipped.nodes() if isinstance(b, ConditionalBlock) and b.label == label),
                take=False,
            )

        outer_cb = ConditionalBlock(label=f"{loop.label}_guard")
        for e in in_edges + out_edges:
            parent.remove_edge(e)
        parent.remove_node(loop)
        arms = [(CodeBlock(cond.as_string), loop)]
        if any(not _is_empty_state(b) for b in skipped.all_control_flow_blocks()):
            arms.append((None, skipped))
        for index, (arm_cond, arm) in enumerate(arms):
            for e in arm.edges():
                for lhs, rhs in strip.items():
                    if e.data.assignments.get(lhs) == rhs:
                        del e.data.assignments[lhs]
            drop_empty_states(arm)
            inner = ControlFlowRegion(label=f"{loop.label}_body_{index}" if index else f"{loop.label}_body")
            inner.add_node(arm, is_start_block=True, ensure_unique_name=True)
            outer_cb.add_branch(arm_cond, inner)
        MoveLoopInvariantIfUp._install_guard(
            parent, loop.label, outer_cb, hoist_assignments, in_edges, out_edges, is_start
        )
        rehome_claimed_block(outer_cb, parent.sdfg)

    @staticmethod
    def _loop_with_branch(
        loop: LoopRegion, cb: ConditionalBlock, index: int, hoist_assignments: list[tuple[str, str]]
    ) -> LoopRegion:
        """A copy of ``loop`` with ``cb`` spliced out and replaced by the body of its branch ``index``.

        Every other body block, and the per-iteration iedge assignments on them, are preserved; the invariant prep
        that moves out with the guard is removed from the copy, and emptied boundary states are spliced out.
        """
        branch = copy.deepcopy(cb).branches[index][1]
        new_loop = copy.deepcopy(loop)
        new_cb_candidates = [b for b in new_loop.nodes() if isinstance(b, ConditionalBlock) and b.label == cb.label]
        assert len(new_cb_candidates) == 1, f"expected one ConditionalBlock named {cb.label!r} in deepcopy"
        new_cb = new_cb_candidates[0]
        cb_pred_edges = list(new_loop.in_edges(new_cb))
        cb_succ_edges = list(new_loop.out_edges(new_cb))
        cb_was_start = new_loop.start_block is new_cb
        for e in cb_pred_edges + cb_succ_edges:
            new_loop.remove_edge(e)
        new_loop.remove_node(new_cb)
        bb = list(branch.nodes())
        be = list(branch.edges())
        bstart = branch.start_block
        branch_sinks = [b for b in bb if branch.out_degree(b) == 0]
        for b in bb:
            new_loop.add_node(b, ensure_unique_name=True)
        for e in be:
            new_loop.add_edge(e.src, e.dst, copy.deepcopy(e.data))
        for e in cb_pred_edges:
            new_loop.add_edge(e.src, bstart, copy.deepcopy(e.data))
        for sink in branch_sinks:
            for e in cb_succ_edges:
                new_loop.add_edge(sink, e.dst, copy.deepcopy(e.data))
        if cb_was_start:
            new_loop.start_block = new_loop.node_id(bstart)

        # The prep that moved out with the guard must not stay behind: a body copy would make
        # the loop define a name the hoisted condition reads.
        for lhs, rhs in strippable_prep(loop, hoist_assignments).items():
            for e in new_loop.edges():
                if e.data.assignments.get(lhs) == rhs:
                    del e.data.assignments[lhs]

        # Splice out empty boundary states the removed conditional leaves (EmptyStateElimination's local splice).
        drop_empty_states(new_loop)
        return new_loop
