# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Privatize a constant-index access on a shared array inside a sequential loop.

A loop body that writes and reads a single fixed slot of a *non*-loop-local array, e.g.
``arr[1] = ...; ... = arr[1] * ...``, reads as a loop-carried write/write conflict on that
slot and blocks :class:`~dace.transformation.interstate.loop_to_map.LoopToMap` -- even
though every iteration sees an independent value at the slot. Buffering the slot in a
per-iteration scalar removes the false dependence so the loop parallelizes::

    for i in range(N):                for i in range(N):       # now a Map
        if cond(i): arr[1] = f(i) ->      t = arr[1]           # load live-in
        out[i] = arr[1] * g(i)            if cond(i): t = f(i)
                                          out[i] = t * g(i)

Unlike :class:`~dace.transformation.passes.buffer_expansion.BufferExpansion`, the array
``arr`` is *not* loop-local (it's read or written outside the loop too), so the slot must be
treated as a value, not a buffer with private layout. The conditional-write case is the
realistic one (cloudsc's ``zvqx[1] = ...`` inside an ``if yrecldp_laericesed`` guard): the
prologue load captures the external "live-in" value the unconditional read would observe
when the guard does not fire.

Safety. The pass refuses unless ``arr[c]``:

- is the **only** subset of ``arr`` read or written inside the loop body (no mixed
  ``arr[c]`` / ``arr[i]`` accesses),
- has no WCR / reduction edge,
- is **not live-out at that slot** -- the loop's writes to ``arr[c]`` are unobservable
  outside the loop. The check is *slot-precise*: a post-loop read of a different element
  of ``arr`` (``arr[k]`` for some ``k != c``) does not block promotion, because the
  in-loop writes to ``arr[c]`` don't affect that other element. The cloudsc pattern
  where a species loop writes ``zvqx[1]`` and a downstream consumer reads ``zvqx[2..4]``
  fits naturally under this gate.

The pass is intentionally narrow: when the loop's writes to ``arr[c]`` *are* observed
outside the loop (live-out at slot ``c``), we refuse rather than try to preserve the
last-iteration value via a writeback buffer -- that would require a ``lastprivate``-style
mechanism DaCe doesn't have yet, and the heavy buffer-plus-writeback form changes the
SDFG more than the value it adds back justifies in the cases we've seen.

The pass only mutates a loop that ``LoopToMap`` currently refuses *and* would accept after
the privatization (verified by re-running the match); a promotion that doesn't help is
reverted so the SDFG does not grow needlessly.

:note: Scope today:

    - **Fully-constant slots only.** ``arr[c]`` (1-D) and ``arr[c1, c2, ...]``
      (multi-dim, every axis a literal integer) are accepted. Mixed
      constant + loop-variable axes (``arr[jl, 3]``) are refused -- the
      loop-variable axis varies per iteration, so the access isn't a single
      privatisable point.
    - **Live-out at the same slot triggers a refusal.** Preserving the last-iteration
      value would require a ``lastprivate``-style writeback buffer; DaCe does not
      have that machinery yet, and adding it would change the SDFG more than the
      parallelism gain justifies. The slot-precise live-out check only allows
      promotion when no post-loop read targets *this* specific slot.

The pass assumes the loop body is flat dataflow: tasklets + AccessNodes connected by
plain memlets. NestedSDFG-mediated accesses to the same array are not considered.
"""
import contextlib
import copy
import io
from typing import Any, Dict, List, Optional, Set, Tuple

from dace import SDFG, data, dtypes, properties, subsets, symbolic
from dace.memlet import Memlet
from dace.sdfg import nodes
from dace.sdfg.state import ControlFlowRegion, LoopRegion, SDFGState
from dace.transformation import pass_pipeline as ppl

#: Array lifetimes we are allowed to attach a privatized scalar to (the scalar follows
#: the same scoping rules; persistent / global lifetimes are out of scope for v1).
_PRIVATIZABLE_LIFETIMES = (dtypes.AllocationLifetime.Scope, dtypes.AllocationLifetime.SDFG)


class _Promotion:
    """An applied constant-index promotion, with the information needed to undo it.

    :param sdfg: The owning SDFG (its descriptor store gets the temporary scalar added).
    :param arr_name: The shared array name we privatized at one constant index.
    :param scalar_name: The new transient scalar's name.
    :param prologue: The state inserted in front of the loop body to load ``arr[c]``.
    :param edits: The memlet edits as ``(memlet, data_before, subset_before)`` triples.
    :param node_edits: The access-node ``(node, data_before)`` pairs for the wholesale
                       rename path (single-slot). Empty for the slot-precise path.
    :param introduced_scalar_nodes: AccessNodes for the new scalar inserted during the
                                    slot-precise path (multi-slot). Tuples of ``(node, state)``.
    :param removed_arr_nodes: Original ``arr`` AccessNodes removed during slot-precise
                              orphan cleanup. Tuples of ``(node, state)``.
    """

    def __init__(self, sdfg: SDFG, arr_name: str, scalar_name: str, prologue: SDFGState,
                 edits: List[Tuple[Memlet, Optional[str], Any]],
                 node_edits: List[Tuple[nodes.AccessNode, str]],
                 introduced_scalar_nodes: Optional[List[Tuple[nodes.AccessNode, Any]]] = None,
                 removed_arr_nodes: Optional[List[Tuple[nodes.AccessNode, Any]]] = None):
        self.sdfg = sdfg
        self.arr_name = arr_name
        self.scalar_name = scalar_name
        self.prologue = prologue
        self._edits = edits
        self._node_edits = node_edits
        self._introduced_scalar_nodes = introduced_scalar_nodes or []
        self._removed_arr_nodes = removed_arr_nodes or []

    def undo(self):
        """Restore every rewritten access node and memlet, drop the prologue, drop the scalar."""
        for memlet, data_before, subset_before in self._edits:
            memlet.data = data_before
            memlet.subset = subset_before
        for node, data_before in self._node_edits:
            node.data = data_before
        # Re-add ``arr`` AccessNodes we removed (slot-precise orphan cleanup) so the
        # endpoint-swap reversal below has somewhere to reattach.
        for arr_node, state in self._removed_arr_nodes:
            if arr_node not in state.nodes():
                state.add_node(arr_node)
        # Drop the new scalar AccessNodes inserted by the multi-slot path. Each was
        # connected to the arr AccessNode via swapped edges; the memlet-edit undo above
        # restored the memlet contents, but the scalar AccessNodes themselves are still
        # in the state and reference the descriptor we are about to drop. Detach them.
        for scalar_node, state in self._introduced_scalar_nodes:
            if scalar_node in state.nodes():
                # Re-attach any still-connected edges to the original arr AccessNode.
                # By the time undo runs, the matching memlets have data=arr_name again
                # (via the memlet-edit revert above), so the swap target is unambiguous.
                arr_node = next((n for n in state.nodes()
                                 if isinstance(n, nodes.AccessNode) and n.data == self.arr_name),
                                None)
                if arr_node is None:
                    arr_node = state.add_access(self.arr_name)
                for e in list(state.in_edges(scalar_node)) + list(state.out_edges(scalar_node)):
                    is_in = (e.dst is scalar_node)
                    new_src = arr_node if not is_in else e.src
                    new_dst = arr_node if is_in else e.dst
                    state.remove_edge(e)
                    state.add_edge(new_src, e.src_conn, new_dst, e.dst_conn, e.data)
                state.remove_node(scalar_node)
        # Detach the prologue from the loop body. Its single out-edge re-enters the body's
        # original start block (or a stand-in), which becomes the start block again.
        region = self.prologue.parent_graph
        if region is not None:
            successors = [e.dst for e in region.out_edges(self.prologue)]
            new_start = successors[0] if successors else None
            for e in list(region.in_edges(self.prologue)):
                region.remove_edge(e)
                if new_start is not None:
                    region.add_edge(e.src, new_start, e.data)
            for e in list(region.out_edges(self.prologue)):
                region.remove_edge(e)
            region.remove_node(self.prologue)
            # Drop the stale start-block cache; the next ``start_block`` access recomputes.
            region._cached_start_block = None
            region._start_block = None
        # The scalar is no longer referenced anywhere; drop the descriptor.
        self.sdfg.remove_data(self.scalar_name, validate=False)


@properties.make_properties
class PromoteConstantIndexAccess(ppl.Pass):
    """Privatize a constant-index slot of a shared array inside a loop to unblock LoopToMap.

    See the module docstring. For every loop the SDFG (and its nested SDFGs) contains that
    ``LoopToMap`` currently refuses, the pass picks the candidate ``(arr, c)`` pairs that
    pass the safety check, replaces every ``arr[c]`` access in the body with a fresh
    per-iteration scalar (prologue-loaded from ``arr[c]`` so the conditional-write case
    still observes the external live-in value), and keeps the rewrite only if the loop
    then becomes parallelizable.
    """

    CATEGORY: str = 'Optimization Preparation'

    def modifies(self) -> ppl.Modifies:
        return (ppl.Modifies.Descriptors | ppl.Modifies.Memlets | ppl.Modifies.CFG | ppl.Modifies.AccessNodes)

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        # Single-shot: a successful promotion turns its loop into a map; a refused loop
        # stays refused on the same shape. Re-runs only repeat the speculative work.
        return False

    def depends_on(self) -> Set:
        return set()

    def apply_pass(self, sdfg: SDFG, _pipeline_results: Dict[str, Any]) -> Optional[Dict[str, List[str]]]:
        """Promote ``(arr, c)`` slots for loops that ``LoopToMap`` refuses, in ``sdfg`` and nested.

        :param sdfg: The SDFG to transform in place.
        :returns: A dict mapping each parallelized loop's label to the ``arr@c`` labels
                  that were promoted in it, or ``None`` if nothing was promoted.
        """
        promoted: Dict[str, List[str]] = {}
        for sd in sdfg.all_sdfgs_recursive():
            promoted.update(self._promote_sdfg(sd))
        return promoted or None

    def report(self, pass_retval: Any) -> Optional[str]:
        if not pass_retval:
            return None
        slots = sum(len(v) for v in pass_retval.values())
        return (f'PromoteConstantIndexAccess: promoted {slots} constant-index slot(s) to '
                f'unblock {len(pass_retval)} loop(s)')

    # -- core ---------------------------------------------------------------------------

    def _promote_sdfg(self, sdfg: SDFG) -> Dict[str, List[str]]:
        """Promote beneficial slots for the loops of a single SDFG (speculate, then verify)."""
        loops = [r for r in sdfg.all_control_flow_regions() if isinstance(r, LoopRegion) and r.loop_variable]

        # Cheap structural pre-filter first: collect the loops that actually have a
        # privatizable ``(arr, c)`` pair. Only if some do is the (expensive) LoopToMap
        # match worth running.
        candidates: Dict[LoopRegion, List[Tuple[str, Any]]] = {}
        for loop in loops:
            pairs = self._privatizable_slots(sdfg, loop)
            if pairs:
                candidates[loop] = pairs
        if not candidates:
            return {}

        # Speculatively privatize every currently-unmappable loop's safe slots, then
        # re-check each loop individually (per-loop direct ``LoopToMap.can_be_applied``
        # -- see ``_l2m_accepts``). A whole-SDFG ``match_patterns`` walk would miss
        # cloudsc-style deeply-nested inner-region loops.
        plan: List[Tuple[LoopRegion, List[_Promotion], List[str]]] = []
        for loop, pairs in candidates.items():
            if self._l2m_accepts(loop, sdfg):
                continue
            applied: List[_Promotion] = []
            labels: List[str] = []
            for arr_name, c_subset in pairs:
                promo = self._promote(sdfg, loop, arr_name, c_subset)
                applied.append(promo)
                labels.append(f'{arr_name}@{c_subset}')
            if applied:
                plan.append((loop, applied, labels))

        if not plan:
            return {}

        kept: Dict[str, List[str]] = {}
        for loop, applied, labels in plan:
            if self._l2m_accepts(loop, sdfg):
                kept[loop.label] = labels
            else:
                # Undo in reverse: prologues + edits + descriptors come off cleanly.
                for promo in reversed(applied):
                    promo.undo()
        return kept

    @staticmethod
    def _l2m_accepts(loop: LoopRegion, sdfg: SDFG) -> bool:
        """Whether ``LoopToMap`` would accept ``loop`` right now (refusals silenced).

        Asks ``LoopToMap.can_be_applied`` directly on the single loop. Avoids the
        whole-SDFG ``match_patterns(sdfg, LoopToMap)`` walk -- which, on
        deeply-nested CFGs like cloudsc, fails to surface inner-region loops that
        a direct check accepts (a pattern-matcher coverage gap, not a real L2M
        refusal). Per-loop checks are also faster: ``O(candidates)`` instead of
        ``O(all loops)`` of the SDFG.

        :param loop: The loop region to test.
        :param sdfg: The owning SDFG.
        :returns: ``True`` if L2M's strict-mode ``can_be_applied`` returns True.
        """
        from dace.transformation.interstate.loop_to_map import LoopToMap  # avoid an import cycle
        xform = LoopToMap()
        xform.loop = loop
        xform.expr_index = 0
        with contextlib.redirect_stdout(io.StringIO()):
            try:
                return bool(xform.can_be_applied(loop.parent_graph, 0, sdfg, permissive=False))
            except Exception:
                return False

    # -- slot detection -----------------------------------------------------------------

    def _privatizable_slots(self, sdfg: SDFG, loop: LoopRegion) -> List[Tuple[str, subsets.Range]]:
        """The ``(arr, constant_subset)`` pairs that are safe to privatize in ``loop``.

        At most one constant index ``c`` per array; an array with mixed access modes
        (constant index plus a loop-var-indexed access) is refused.
        """
        loop_states = list(loop.all_states())
        # Group the constant-index accesses by array name; reject early if a non-constant
        # access to the same array exists in the loop, or if the array appears in any
        # memlet whose primary side is a different array (v1 only handles the simple
        # tasklet<->AccessNode shape where ``memlet.data`` is the array itself).
        per_arr: Dict[str, List[subsets.Range]] = {}
        per_arr_has_wcr: Dict[str, bool] = {}
        per_arr_mixed: Set[str] = set()
        for state in loop_states:
            for node in state.nodes():
                if not isinstance(node, nodes.AccessNode):
                    continue
                name = node.data
                desc = sdfg.arrays.get(name)
                if not isinstance(desc, data.Array):
                    continue
                if desc.lifetime not in _PRIVATIZABLE_LIFETIMES:
                    continue
                for edge in list(state.in_edges(node)) + list(state.out_edges(node)):
                    memlet = edge.data
                    if memlet is None:
                        continue
                    if memlet.data != name:
                        # Cross-array memlet (e.g. AccessNode-to-AccessNode copy); refuse.
                        per_arr_mixed.add(name)
                        continue
                    if memlet.wcr is not None:
                        per_arr_has_wcr[name] = True
                    sub = memlet.subset
                    if sub is None:
                        per_arr_mixed.add(name)
                        continue
                    if self._is_constant_point_subset(sub):
                        per_arr.setdefault(name, []).append(sub)
                    else:
                        per_arr_mixed.add(name)

        if not per_arr:
            return []

        results: List[Tuple[str, subsets.Range]] = []
        for name, subs in per_arr.items():
            if name in per_arr_mixed:
                continue
            if per_arr_has_wcr.get(name):
                continue
            # One candidate per *distinct* constant point. Distinct slots are independent
            # (the per-array ``mixed`` gate above already refused promotion if a wildcard
            # access to the same array exists); each slot's live-out check is per-point.
            unique_points: List[subsets.Range] = []
            for s in subs:
                if not any(self._point_subsets_equal(s, p) for p in unique_points):
                    unique_points.append(s)
            for point in unique_points:
                if not self._not_live_out(loop, name, slot=point):
                    continue
                results.append((name, point))
        return results

    @staticmethod
    def _is_constant_point_subset(sub) -> bool:
        """True if ``sub`` is a single point (size 1 on every axis) with no free symbols.

        Accepts 1-D ``arr[3]`` and multi-dim ``arr[3, 5]`` -- both name a single element
        whose location does not depend on any symbol. Multi-dim slots with mixed
        constant + symbolic axes (``arr[i, 3]``) are NOT accepted; they require the
        non-constant axis to be promoted separately.
        """
        if not isinstance(sub, subsets.Range):
            return False
        if not sub.ranges:
            return False
        for axis, (lo, hi, st) in enumerate(sub.ranges):
            if lo != hi:
                return False
            if symbolic.pystr_to_symbolic(st) != 1:
                return False
            # Reject tiled axes (tile_size > 1 covers multiple elements).
            if sub.tile_sizes and symbolic.pystr_to_symbolic(sub.tile_sizes[axis]) != 1:
                return False
        # No free symbols across any axis -> truly constant.
        return len(sub.free_symbols) == 0

    @staticmethod
    def _point_subsets_equal(a: subsets.Range, b: subsets.Range) -> bool:
        """Equality on two single-point Ranges -- compare the lower bound on each axis."""
        if len(a.ranges) != len(b.ranges):
            return False
        return all(symbolic.pystr_to_symbolic(ar[0]) == symbolic.pystr_to_symbolic(br[0])
                   for ar, br in zip(a.ranges, b.ranges))

    def _not_live_out(self, loop: LoopRegion, name: str, slot: Optional[subsets.Range] = None) -> bool:
        """Live-out check for ``name`` (whole array) or ``name[slot]`` (one element).

        Walks the parent CFG forward from ``loop``; if the loop is nested inside another
        region (e.g. a conditional branch), walks the enclosing region forward from the
        loop's ancestor too -- a read of ``arr`` after the enclosing block would still
        observe an iteration's value. Any access *inside* ``loop`` itself is permitted
        (those are what we're privatizing). Any access in a predecessor of the loop
        (live-in) is also permitted -- the prologue load handles it.

        :param slot: When given, the check is *slot-precise*: a post-loop access to a
            *different* element of ``arr`` does not count as a live-out conflict, because
            the in-loop writes to ``arr[slot]`` don't affect those other elements. This
            lets the cheap scalar-form promotion fire for cases where the user reads
            ``arr[k != slot]`` after the loop -- the common cloudsc-style pattern where
            ``zvqx[1]`` is written by the species loop and ``zvqx[2..4]`` are read later.
            When ``None``, the legacy whole-array check applies.
        """
        loop_states = set(loop.all_states())
        cur = loop
        while True:
            parent = cur.parent_graph
            if parent is None:
                return True
            reachable: Set[Any] = set()
            frontier = [cur]
            while frontier:
                node = frontier.pop()
                for e in parent.out_edges(node):
                    if e.dst in reachable:
                        continue
                    reachable.add(e.dst)
                    frontier.append(e.dst)
            for block in reachable:
                for state in self._states_of(block):
                    if state in loop_states:
                        continue
                    if self._state_touches_slot(state, name, slot):
                        return False
            # Walk up to the next enclosing CFG and check its forward-reachable region.
            cur = parent

    @staticmethod
    def _state_touches_slot(state: SDFGState, name: str, slot: Optional[subsets.Range]) -> bool:
        """``True`` if ``state`` has any incident memlet on ``name`` that might overlap ``slot``.

        With ``slot=None``, any access-node of ``name`` qualifies (the legacy whole-array
        check). With a concrete ``slot``, only memlets whose subset overlaps the slot
        count -- delegated to :meth:`dace.subsets.Range.intersects`, treating its
        ``None`` (indeterminate) result as conservative overlap.
        """
        if slot is None:
            return any(isinstance(n, nodes.AccessNode) and n.data == name for n in state.nodes())
        for node in state.nodes():
            if not (isinstance(node, nodes.AccessNode) and node.data == name):
                continue
            for edge in list(state.in_edges(node)) + list(state.out_edges(node)):
                memlet = edge.data
                if memlet is None or memlet.data != name or memlet.subset is None:
                    continue
                if not isinstance(memlet.subset, subsets.Range):
                    return True
                if len(memlet.subset.ranges) != len(slot.ranges):
                    return True
                # ``Range.intersects`` returns True/False/None (indeterminate). Conservative
                # default: any non-False answer means we cannot rule out overlap.
                result = memlet.subset.intersects(slot)
                if result is not False:
                    return True
        return False

    @staticmethod
    def _states_of(block) -> List[SDFGState]:
        """Flatten ``block`` to a list of its constituent SDFG states."""
        if isinstance(block, SDFGState):
            return [block]
        if isinstance(block, ControlFlowRegion):
            return list(block.all_states())
        return []

    # -- rewrite ------------------------------------------------------------------------

    def _arr_accesses_only_at_slot(self, loop: LoopRegion, arr_name: str,
                                    c_subset: subsets.Range) -> bool:
        """True if every ``arr`` access inside ``loop`` matches ``c_subset``.

        Used to decide between the cheap wholesale-rename path (every access agrees
        on one slot) and the slot-precise rewrite (a sibling slot also lives in the
        same loop body, so a wholesale rename would corrupt the other slot).
        """
        for state in loop.all_states():
            for edge in state.edges():
                memlet = edge.data
                if memlet is None or memlet.data != arr_name or memlet.subset is None:
                    continue
                if not self._is_constant_point_subset(memlet.subset):
                    continue
                if not self._point_subsets_equal(memlet.subset, c_subset):
                    return False
        return True

    def _promote(self, sdfg: SDFG, loop: LoopRegion, arr_name: str, c_subset: subsets.Range) -> _Promotion:
        """Privatize ``arr[c]`` to a per-iteration scalar.

        The slot-precise live-out gate in :meth:`_privatizable_slots` has already
        established that no post-loop state reads this specific element, so the scalar
        can be ``Scope`` lifetime (per-thread under ``LoopToMap``, dying with the loop)
        and no writeback is needed.

        Two paths, chosen by :meth:`_arr_accesses_only_at_slot`:

        * **Wholesale rename** -- every ``arr`` access in the loop matches ``c_subset``;
          rename the AccessNode in place and rewrite each memlet's subset to ``[0]``.
        * **Slot-precise rewrite** -- a sibling constant slot of ``arr`` also lives in
          the loop; leave the original AccessNode for those, introduce a per-state
          scalar AccessNode for this slot, and reroute only the matching edges to it.
        """
        desc = sdfg.arrays[arr_name]
        # Use every axis of the constant point in the scalar's label so multi-dim slots
        # (``arr[3, 5]``) stay distinguishable from sibling slots (``arr[3, 6]``).
        c_str = '_'.join(str(symbolic.pystr_to_symbolic(rng[0])) for rng in c_subset.ranges)
        base = f'{arr_name}_at_{c_str}_promoted'
        scalar_name, _ = sdfg.add_scalar(base,
                                         desc.dtype,
                                         transient=True,
                                         lifetime=dtypes.AllocationLifetime.Scope,
                                         find_new_name=True)

        # Prologue: insert a state at the head of the loop body that copies arr[c] -> t.
        # This load gives the unconditional read the external "live-in" value even when
        # the in-loop write is conditional and doesn't fire on this iteration.
        old_start = loop.start_block
        prologue = loop.add_state_before(old_start, label=f'{arr_name}_at_{c_str}_load', is_start_block=True)
        src = prologue.add_access(arr_name)
        dst = prologue.add_access(scalar_name)
        prologue.add_nedge(
            src, dst, Memlet(data=arr_name, subset=copy.deepcopy(c_subset), other_subset=subsets.Range([(0, 0, 1)])))

        edits: List[Tuple[Memlet, Optional[str], Any]] = []
        node_edits: List[Tuple[nodes.AccessNode, str]] = []
        introduced_scalar_nodes: List[Tuple[nodes.AccessNode, Any]] = []
        removed_arr_nodes: List[Tuple[nodes.AccessNode, Any]] = []
        scalar_subset = subsets.Range([(0, 0, 1)])

        single_slot = self._arr_accesses_only_at_slot(loop, arr_name, c_subset)

        for state in loop.all_states():
            if state is prologue:
                continue
            # Per-state introduced scalar AccessNode (slot-precise path only).
            per_state_scalar: Optional[nodes.AccessNode] = None
            for edge in list(state.edges()):
                memlet = edge.data
                if memlet is None or memlet.data != arr_name:
                    continue
                if not single_slot and not self._point_subsets_equal(memlet.subset, c_subset):
                    # Sibling slot; left for its own _promote call.
                    continue
                edits.append((memlet, memlet.data, memlet.subset))
                memlet.data = scalar_name
                memlet.subset = copy.deepcopy(scalar_subset)
                if not single_slot:
                    # Slot-precise: swap the arr-AccessNode endpoint to a fresh per-state
                    # scalar AccessNode (one per state; reused across this slot's edges).
                    for endpoint in ('src', 'dst'):
                        end_node = getattr(edge, endpoint)
                        if isinstance(end_node, nodes.AccessNode) and end_node.data == arr_name:
                            if per_state_scalar is None:
                                per_state_scalar = state.add_access(scalar_name)
                                introduced_scalar_nodes.append((per_state_scalar, state))
                            new_src = per_state_scalar if endpoint == 'src' else edge.src
                            new_dst = per_state_scalar if endpoint == 'dst' else edge.dst
                            state.remove_edge(edge)
                            edge = state.add_edge(new_src, edge.src_conn, new_dst, edge.dst_conn, edge.data)
            if single_slot:
                # Wholesale rename the arr AccessNode itself; safe because every
                # memlet was rewritten.
                for node in list(state.nodes()):
                    if isinstance(node, nodes.AccessNode) and node.data == arr_name:
                        node_edits.append((node, node.data))
                        node.data = scalar_name
            else:
                # Slot-precise path: the original ``arr`` AccessNode may now be
                # orphaned (all its incident edges were rerouted to a per-slot
                # scalar AccessNode in this state). Remove orphaned arr nodes here
                # so the SDFG validates; tracked separately from node_edits so undo
                # can distinguish rename-restore from re-add.
                for node in list(state.nodes()):
                    if (isinstance(node, nodes.AccessNode) and node.data == arr_name
                            and state.degree(node) == 0):
                        removed_arr_nodes.append((node, state))
                        state.remove_node(node)
        return _Promotion(sdfg, arr_name, scalar_name, prologue, edits, node_edits,
                          introduced_scalar_nodes, removed_arr_nodes)
