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
- is **not live-out** of the loop -- conservatively, no state in any forward successor of
  the loop's parent CFG accesses ``arr``.

The pass only mutates a loop that ``LoopToMap`` currently refuses *and* would accept after
the privatization (verified by re-running the match); a promotion that doesn't help is
reverted so the SDFG does not grow needlessly.
"""
import contextlib
import copy
import io
from typing import Any, Dict, List, Optional, Set, Tuple

from dace import SDFG, data, dtypes, properties, subsets, symbolic
from dace.memlet import Memlet
from dace.sdfg import nodes
from dace.sdfg.state import LoopRegion, SDFGState
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
    :param node_edits: The access-node renames as ``(access_node, data_before)`` pairs.
    :param epilogue: The state inserted after the loop to write ``scalar -> arr[c]``,
                     present iff the array was live-out and ``allow_live_out`` was set.
                     ``None`` for the in-loop-only privatization case.
    """

    def __init__(self, sdfg: SDFG, arr_name: str, scalar_name: str, prologue: SDFGState,
                 edits: List[Tuple[Memlet, Optional[str], Any]], node_edits: List[Tuple[nodes.AccessNode, str]],
                 epilogue: Optional[SDFGState] = None):
        self.sdfg = sdfg
        self.arr_name = arr_name
        self.scalar_name = scalar_name
        self.prologue = prologue
        self.epilogue = epilogue
        self._edits = edits
        self._node_edits = node_edits

    def undo(self):
        """Restore every rewritten access node and memlet, drop the prologue+epilogue, drop the scalar."""
        for memlet, data_before, subset_before in self._edits:
            memlet.data = data_before
            memlet.subset = subset_before
        for node, data_before in self._node_edits:
            node.data = data_before
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
        # Detach the epilogue (if any) from the parent CFG. Predecessors flow straight to the
        # epilogue's successor again, restoring the original linear out-edge chain.
        if self.epilogue is not None:
            ep_parent = self.epilogue.parent_graph
            if ep_parent is not None:
                ep_out = list(ep_parent.out_edges(self.epilogue))
                ep_in = list(ep_parent.in_edges(self.epilogue))
                for ie in ep_in:
                    ep_parent.remove_edge(ie)
                for oe in ep_out:
                    ep_parent.remove_edge(oe)
                # Re-link predecessors to the epilogue's original successor(s) verbatim.
                for ie in ep_in:
                    for oe in ep_out:
                        ep_parent.add_edge(ie.src, oe.dst, ie.data)
                ep_parent.remove_node(self.epilogue)
                ep_parent._cached_start_block = None
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

    allow_live_out = properties.Property(
        dtype=bool,
        default=False,
        desc="Allow privatizing an ``arr[c]`` slot even when ``arr`` is read or written outside "
        "the loop. When enabled, an epilogue state is inserted after the loop that writes the "
        "scalar's final value back to ``arr[c]``. For sequential loops this preserves "
        "last-iteration semantics exactly. For loops that subsequently lift to a parallel Map "
        "(the intended use), the scalar becomes a shared race-write target inside the Map and "
        "the epilogue reads ``some`` iteration's value -- correct under L2M-permissive semantics "
        "where the caller asserts the writes are conflict-free / idempotent / order-insensitive. "
        "Off by default; opt in explicitly per the same risk model as ``LoopToMap(permissive=True)``.",
    )

    def __init__(self, allow_live_out: bool = False):
        super().__init__()
        self.allow_live_out = allow_live_out

    def modifies(self) -> ppl.Modifies:
        return (ppl.Modifies.Descriptors | ppl.Modifies.Memlets | ppl.Modifies.CFG | ppl.Modifies.AccessNodes)

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        # Single-shot: a successful promotion turns its loop into a map; a refused loop
        # stays refused on the same shape. Re-runs only repeat the speculative work.
        return False

    def depends_on(self):
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

        before = self._mappable_loops(sdfg)

        # Speculatively privatize every currently-unmappable loop's safe slots. The slots
        # are array-local to the loop (no other state of the SDFG accesses ``arr``
        # post-loop, by the live-out guard); a single LoopToMap re-match then decides
        # each loop.
        plan: List[Tuple[LoopRegion, List[_Promotion], List[str]]] = []
        for loop, pairs in candidates.items():
            if loop in before:
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
        after = self._mappable_loops(sdfg)

        kept: Dict[str, List[str]] = {}
        for loop, applied, labels in plan:
            if loop in after:
                kept[loop.label] = labels
            else:
                # Undo in reverse: prologues + edits + descriptors come off cleanly.
                for promo in reversed(applied):
                    promo.undo()
        return kept

    @staticmethod
    def _mappable_loops(sdfg: SDFG) -> Set[LoopRegion]:
        """The loop regions ``LoopToMap`` would parallelize right now (refusals silenced)."""
        from dace.transformation.interstate.loop_to_map import LoopToMap  # avoid an import cycle
        from dace.transformation.passes.pattern_matching import match_patterns

        with contextlib.redirect_stdout(io.StringIO()):
            return {m.loop for m in match_patterns(sdfg, LoopToMap)}

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
            # All constant-index accesses must agree on the same single point.
            first = subs[0]
            if not all(self._point_subsets_equal(first, s) for s in subs[1:]):
                continue
            # Live-out gate. By default refuse, since the in-body scalar holds only an
            # iteration-local value -- a state after the loop that reads ``arr`` would
            # observe the original (pre-loop) value instead of the iteration's update.
            # With ``allow_live_out`` the epilogue writes the scalar back, so the
            # external read sees an iteration's value (the *last* sequentially, or
            # *some* under L2M-permissive lifting).
            if not self.allow_live_out and not self._not_live_out(loop, name):
                continue
            results.append((name, first))
        return results

    @staticmethod
    def _is_constant_point_subset(sub) -> bool:
        """True if ``sub`` is a 1-D single point with no free symbols (a pure integer constant)."""
        if not isinstance(sub, subsets.Range):
            return False
        if len(sub.ranges) != 1:
            return False
        lo, hi, st = sub.ranges[0]
        if lo != hi:
            return False
        if symbolic.pystr_to_symbolic(st) != 1:
            return False
        # Reject tiled subsets (tile_size > 1 covers multiple elements).
        if sub.tile_sizes and symbolic.pystr_to_symbolic(sub.tile_sizes[0]) != 1:
            return False
        # No free symbols -> truly constant (independent of the loop variable).
        return len(sub.free_symbols) == 0

    @staticmethod
    def _point_subsets_equal(a: subsets.Range, b: subsets.Range) -> bool:
        """Equality on two single-point Ranges by lower bound."""
        return (symbolic.pystr_to_symbolic(a.ranges[0][0]) == symbolic.pystr_to_symbolic(b.ranges[0][0]))

    def _not_live_out(self, loop: LoopRegion, name: str) -> bool:
        """Conservative live-out check: no state forward-reachable from ``loop`` (in any
        enclosing CFG level) accesses ``name``.

        Walks the parent CFG forward from ``loop``; if the loop is nested inside another
        region (e.g. a conditional branch), walks the enclosing region forward from the
        loop's ancestor too -- a read of ``arr`` after the enclosing block would still
        observe an iteration's value. Any access *inside* ``loop`` itself is permitted
        (those are what we're privatizing). Any access in a predecessor of the loop
        (live-in) is also permitted -- the prologue load handles it.
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
                    if any(isinstance(n, nodes.AccessNode) and n.data == name for n in state.nodes()):
                        return False
            # Walk up to the next enclosing CFG and check its forward-reachable region.
            cur = parent

    @staticmethod
    def _states_of(block) -> List[SDFGState]:
        """Flatten ``block`` to a list of its constituent SDFG states."""
        if isinstance(block, SDFGState):
            return [block]
        try:
            return list(block.all_states())
        except AttributeError:
            return []

    # -- rewrite ------------------------------------------------------------------------

    def _promote(self, sdfg: SDFG, loop: LoopRegion, arr_name: str, c_subset: subsets.Range) -> _Promotion:
        """Insert a per-iteration privatized buffer for ``arr[c]`` and rewire every access in ``loop``.

        Two flavours:

        - **Non-live-out** (default): privatize to a per-iteration scalar (Scope lifetime).
          Each iteration sees its own copy; the slot is gone after the loop. No epilogue.
        - **Live-out** (``allow_live_out=True``): privatize to a 1-D buffer indexed by the
          loop variable (SDFG lifetime so the buffer survives past the loop's exit). Each
          iteration writes its own slot, so the writes are uniquely indexed -- ``LoopToMap``
          accepts the loop. An epilogue state in the parent CFG copies the last slot
          (``buf[end - start]``) back to ``arr[c]``, preserving last-iteration semantics
          exactly under sequential execution; under L2M-permissive parallel execution the
          last slot still holds whichever value the ``loop_var == end`` thread wrote, so
          the writeback observes the iteration's value the caller asserts as authoritative.
        """
        from dace.transformation.passes.analysis import loop_analysis

        desc = sdfg.arrays[arr_name]
        c_str = str(symbolic.pystr_to_symbolic(c_subset.ranges[0][0]))
        base = f'{arr_name}_at_{c_str}_promoted'
        live_out = self.allow_live_out and not self._not_live_out(loop, arr_name)

        if live_out:
            return self._promote_buffer(sdfg, loop, arr_name, desc, c_subset, c_str, base, loop_analysis)
        return self._promote_scalar(sdfg, loop, arr_name, desc, c_subset, c_str, base)

    @staticmethod
    def _promote_scalar(sdfg: SDFG, loop: LoopRegion, arr_name: str, desc: data.Array,
                        c_subset: subsets.Range, c_str: str, base: str) -> _Promotion:
        """Original scalar privatization (Scope lifetime; no epilogue)."""
        scalar_name, _ = sdfg.add_scalar(base,
                                         desc.dtype,
                                         transient=True,
                                         lifetime=dtypes.AllocationLifetime.Scope,
                                         find_new_name=True)
        old_start = loop.start_block
        prologue = loop.add_state_before(old_start, label=f'{arr_name}_at_{c_str}_load', is_start_block=True)
        src = prologue.add_access(arr_name)
        dst = prologue.add_access(scalar_name)
        prologue.add_nedge(
            src, dst, Memlet(data=arr_name, subset=copy.deepcopy(c_subset), other_subset=subsets.Range([(0, 0, 1)])))
        edits, node_edits = PromoteConstantIndexAccess._rewrite_body_to(loop, prologue, arr_name, scalar_name,
                                                                       subsets.Range([(0, 0, 1)]))
        return _Promotion(sdfg, arr_name, scalar_name, prologue, edits, node_edits, epilogue=None)

    @staticmethod
    def _promote_buffer(sdfg: SDFG, loop: LoopRegion, arr_name: str, desc: data.Array, c_subset: subsets.Range,
                        c_str: str, base: str, loop_analysis) -> _Promotion:
        """Per-iteration buffer privatization (SDFG lifetime; with writeback epilogue)."""
        start = loop_analysis.get_init_assignment(loop)
        end = loop_analysis.get_loop_end(loop)
        if start is None or end is None:
            # Can't size the buffer -- fall back to the scalar form (which will be refused
            # by ``_privatizable_slots`` next time round because the live-out check fires).
            return PromoteConstantIndexAccess._promote_scalar(sdfg, loop, arr_name, desc, c_subset, c_str, base)

        trip = symbolic.simplify(end - start + 1)
        loop_var = symbolic.pystr_to_symbolic(loop.loop_variable)
        # Per-iteration index into the buffer; ``loop_var - start`` keeps it 0-based even
        # when the loop starts at a non-zero offset, so the buffer's slot count is exactly ``trip``.
        idx_expr = symbolic.simplify(loop_var - start)
        body_subset = subsets.Range([(idx_expr, idx_expr, 1)])

        buf_name, _ = sdfg.add_array(base, [trip], desc.dtype, transient=True,
                                     lifetime=dtypes.AllocationLifetime.SDFG, find_new_name=True)

        old_start = loop.start_block
        prologue = loop.add_state_before(old_start, label=f'{arr_name}_at_{c_str}_load', is_start_block=True)
        src = prologue.add_access(arr_name)
        dst = prologue.add_access(buf_name)
        # The per-iteration load gives this iteration's slot the external live-in value, so
        # a conditional in-body write that doesn't fire still leaves a sensible value in the
        # slot (matching the un-promoted body's read-through-from-arr behavior).
        #
        # Memlet ``data`` MUST be the destination buffer here -- ``LoopToMap`` reads the
        # write-side subset from the memlet's primary ``subset`` field to check write
        # uniqueness across iterations. If we left ``data=arr`` the analysis would see
        # the constant ``arr[1]`` source subset as the buf-side write and refuse the lift.
        prologue.add_nedge(
            src, dst, Memlet(data=buf_name, subset=copy.deepcopy(body_subset), other_subset=copy.deepcopy(c_subset)))

        edits, node_edits = PromoteConstantIndexAccess._rewrite_body_to(loop, prologue, arr_name, buf_name, body_subset)

        epilogue = PromoteConstantIndexAccess._add_buffer_epilogue(loop, arr_name, c_subset, buf_name, c_str,
                                                                   end_offset=symbolic.simplify(end - start))
        return _Promotion(sdfg, arr_name, buf_name, prologue, edits, node_edits, epilogue=epilogue)

    @staticmethod
    def _rewrite_body_to(loop: LoopRegion, prologue: SDFGState, arr_name: str, new_name: str,
                         body_subset: subsets.Range
                         ) -> Tuple[List[Tuple[Memlet, Optional[str], Any]], List[Tuple[nodes.AccessNode, str]]]:
        """Retarget every ``arr`` memlet and access node in the loop body to ``new_name``
        with subset ``body_subset``. Skips the prologue state (its memlet already references
        the new descriptor correctly via ``other_subset``).
        """
        edits: List[Tuple[Memlet, Optional[str], Any]] = []
        node_edits: List[Tuple[nodes.AccessNode, str]] = []
        for state in loop.all_states():
            if state is prologue:
                continue
            for edge in list(state.edges()):
                memlet = edge.data
                if memlet is None or memlet.data != arr_name:
                    continue
                edits.append((memlet, memlet.data, memlet.subset))
                memlet.data = new_name
                memlet.subset = copy.deepcopy(body_subset)
            for node in list(state.nodes()):
                if isinstance(node, nodes.AccessNode) and node.data == arr_name:
                    node_edits.append((node, node.data))
                    node.data = new_name
        return edits, node_edits

    @staticmethod
    def _add_buffer_epilogue(loop: LoopRegion, arr_name: str, c_subset: subsets.Range, buf_name: str, c_str: str,
                             end_offset: Any) -> SDFGState:
        """Splice a writeback state after ``loop`` that copies ``buf[end_offset] -> arr[c]``.

        Inserted in the loop's parent CFG. Every existing out-edge of ``loop`` is re-routed
        to leave the epilogue instead, so the new state sits exactly between ``loop`` and
        whatever previously followed it (loop -> epilogue -> [original successors]).
        """
        import dace
        parent = loop.parent_graph
        epilogue = parent.add_state(f'{arr_name}_at_{c_str}_writeback')
        out_edges = list(parent.out_edges(loop))
        for e in out_edges:
            parent.remove_edge(e)
            parent.add_edge(epilogue, e.dst, e.data)
        parent.add_edge(loop, epilogue, dace.InterstateEdge())
        src = epilogue.add_access(buf_name)
        dst = epilogue.add_access(arr_name)
        last_subset = subsets.Range([(end_offset, end_offset, 1)])
        # ``data=arr`` because the write target is ``arr``; the buf-side ``other_subset``
        # picks the last iteration's slot. (Mirror of the prologue's data-side convention.)
        epilogue.add_nedge(
            src, dst, Memlet(data=arr_name, subset=copy.deepcopy(c_subset), other_subset=last_subset))
        return epilogue
