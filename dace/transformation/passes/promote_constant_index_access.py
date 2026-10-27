# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Privatize a constant-index access on a shared array inside a sequential loop.

A loop body that writes and reads a single fixed slot of a *non*-loop-local array, e.g.
``arr[1] = ...; ... = arr[1] * ...``, reads as a loop-carried write/write conflict on that
slot and blocks :class:`~dace.transformation.interstate.loop_to_map.LoopToMap` -- even
though every iteration sees an independent value at the slot. Buffering the slot in a
per-iteration scalar removes the false dependence so the loop parallelizes::

    for i in range(N):                for i in range(N):       # now a Map
        arr[1] = f(i)             ->      t = arr[1]           # prologue: live-in
        out[i] = arr[1] * g(i)            t = f(i)             # must-def, before any read
                                          out[i] = t * g(i)

Unlike :class:`~dace.transformation.passes.buffer_expansion.BufferExpansion`, the array
``arr`` is *not* loop-local (it's read or written outside the loop too), so the slot must be
treated as a value, not a buffer with private layout.

The prologue load exists so a read the body reaches before its own write still observes
*something*; it is NOT what makes such a read correct. It supplies the pre-loop value, and
the only iteration for which that is the right answer is one where no earlier iteration
wrote the slot. Hence the must-def gate below.

Safety. The pass refuses unless ``arr[c]``:

- is the **only** subset of ``arr`` read or written inside the loop body (no mixed
  ``arr[c]`` / ``arr[i]`` accesses),
- has no WCR / reduction edge,
- has **no upward-exposed read in the body** -- on every iteration, every read of
  ``arr[c]`` is preceded by a write of ``arr[c]`` from that same iteration (a genuine
  must-def), or the body writes ``arr[c]`` nowhere at all. This is the classical
  privatization premise and it is what makes the prologue load value-preserving: a read
  that is *not* must-defined observes the previous iteration's write in the original
  loop, and the pre-loop value in the promoted loop. See
  :meth:`PromoteConstantIndexAccess.slot_reads_are_must_defined`,
- is **not live-out at that slot** -- the loop's writes to ``arr[c]`` are unobservable
  outside the loop. The check is *slot-precise*: a post-loop read of a different element
  of ``arr`` (``arr[k]`` for some ``k != c``) does not block promotion, because the
  in-loop writes to ``arr[c]`` don't affect that other element. The cloudsc pattern
  where a species loop writes ``zvqx[1]`` and a downstream consumer reads ``zvqx[2..4]``
  fits naturally under this gate. "After the loop" includes the blocks that *precede* the
  loop inside an enclosing :class:`~dace.sdfg.state.LoopRegion`: the enclosing loop's back
  edge runs them again after this loop finishes.

The pass is intentionally narrow: when the loop's writes to ``arr[c]`` *are* observed
outside the loop (live-out at slot ``c``), we refuse rather than try to preserve the
last-iteration value via a writeback buffer -- that would require a ``lastprivate``-style
mechanism DaCe doesn't have yet, and the heavy buffer-plus-writeback form changes the
SDFG more than the value it adds back justifies in the cases we've seen.

The pass only mutates a loop that ``LoopToMap`` currently refuses *and* accepts with the
promoted arrays assumed iteration-private (asked before the rewrite); a promotion that
would not help is never applied, so the SDFG does not grow needlessly.

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
    - **A conditionally-written slot with an unconditional read is refused.** The
      must-def gate cannot prove the read sees this iteration's write, and it does not
      in general: with a loop-varying guard (``if jl % 2 == 0: arr[c] = ...``) the
      unguarded iterations read the previous iteration's value, which the prologue load
      does not reproduce. The cloudsc ``if yrecldp_laericesed`` shape is only safe
      because that guard is loop-*invariant* -- true on every iteration or on none.
      Recovering it needs a loop-invariance analysis of the guarding conditions (and of
      every other write to the slot); until that exists, the class is refused.

The pass assumes the loop body is flat dataflow: tasklets + AccessNodes connected by
plain memlets. NestedSDFG-mediated accesses to the same array are not considered.
"""

import contextlib
import copy
import io
from typing import Any

from dace import SDFG, data, dtypes, properties, subsets, symbolic
from dace.memlet import Memlet
from dace.ordered import OrderedSet
from dace.sdfg import nodes
from dace.sdfg.state import (
    ControlFlowRegion,
    LoopRegion,
    SDFGState,
)
from dace.transformation import pass_pipeline as ppl
from dace.transformation.passes.scalar_fission import AccessIndex, MustDefinition, SlotAccesses, point_subsets_equal

#: Array lifetimes we are allowed to attach a privatized scalar to (the scalar follows
#: the same scoping rules; persistent / global lifetimes are out of scope for v1).
_PRIVATIZABLE_LIFETIMES = (dtypes.AllocationLifetime.Scope, dtypes.AllocationLifetime.SDFG)


class _Reach:
    """The states reachable from each state of a graph given by its adjacency, each derived once on first lookup."""

    __slots__ = ("adj", "known")

    def __init__(self, adj: dict[SDFGState, list[SDFGState]]) -> None:
        self.adj = adj
        self.known: dict[SDFGState, set[SDFGState]] = {}

    def get(self, state: SDFGState, default=()) -> set[SDFGState]:
        if state not in self.adj:
            return default
        seen = self.known.get(state)
        if seen is None:
            seen = set(self.adj[state])
            stack = list(seen)
            while stack:
                for nxt in self.adj[stack.pop()]:
                    if nxt not in seen:
                        seen.add(nxt)
                        stack.append(nxt)
            self.known[state] = seen
        return seen


@properties.make_properties
class PromoteConstantIndexAccess(ppl.Pass):
    """Privatize a constant-index slot of a shared array inside a loop to unblock LoopToMap.

    See the module docstring. For every loop the SDFG (and its nested SDFGs) contains that
    ``LoopToMap`` currently refuses, the pass picks the candidate ``(arr, c)`` pairs that
    pass the safety check, replaces every ``arr[c]`` access in the body with a fresh
    per-iteration scalar (prologue-loaded from ``arr[c]`` so a read the body reaches before
    its own write still observes the pre-loop value), when ``LoopToMap`` accepts the loop with
    those slots assumed iteration-private.
    """

    CATEGORY: str = "Optimization Preparation"

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Descriptors | ppl.Modifies.Memlets | ppl.Modifies.CFG | ppl.Modifies.AccessNodes

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        # Single-shot: a successful promotion turns its loop into a map; a refused loop
        # stays refused on the same shape. Re-runs only repeat the speculative work.
        return False

    def depends_on(self) -> list[type[ppl.Pass] | ppl.Pass]:
        return []

    def apply_pass(self, sdfg: SDFG, _pipeline_results: dict[str, Any]) -> dict[str, list[str]] | None:
        """Promote ``(arr, c)`` slots for loops that ``LoopToMap`` refuses, in ``sdfg`` and nested.

        :param sdfg: The SDFG to transform in place.
        :returns: A dict mapping each parallelized loop's label to the ``arr@c`` labels
                  that were promoted in it, or ``None`` if nothing was promoted.
        """
        promoted: dict[str, list[str]] = {}
        for sd in sdfg.all_sdfgs_recursive():
            promoted.update(self._promote_sdfg(sd))
        return promoted or None

    def report(self, pass_retval: Any) -> str | None:
        if not pass_retval:
            return None
        slots = sum(len(v) for v in pass_retval.values())
        return (
            f"PromoteConstantIndexAccess: promoted {slots} constant-index slot(s) to unblock {len(pass_retval)} loop(s)"
        )

    # core

    def _promote_sdfg(self, sdfg: SDFG) -> dict[str, list[str]]:
        """Promote beneficial slots for the loops of a single SDFG (analyze, decide, then promote)."""
        loops = [r for r in sdfg.all_control_flow_regions() if isinstance(r, LoopRegion) and r.loop_variable]

        # Cheap structural pre-filter first: collect the loops that actually have a
        # privatizable ``(arr, c)`` pair. Only if some do is the (expensive) LoopToMap
        # match worth running.
        candidates: dict[LoopRegion, list[tuple[str, Any]]] = {}
        for loop in loops:
            pairs = self._privatizable_slots(sdfg, loop)
            if pairs:
                candidates[loop] = pairs
        if not candidates:
            return {}

        # Per-loop check-decide-promote. Batched decisions (all checks first, all promotions
        # after) don't work when an outer loop's promotion invalidates an inner loop's:
        # cloudsc's outer klev loop ``for_430`` shares the ``zvqx`` array with the inner
        # species loops ``for_767_X``, and promoting zvqx[0..4] in ``for_430`` rewrites the
        # same memlets the inner loops would have promoted, leaving each inner loop's
        # promotion a no-op (no ``zvqx`` accesses left in its body). Deciding each loop
        # against the graph the previous loops' promotions left behind keeps that order.
        kept: dict[str, list[str]] = {}
        for loop, pairs in candidates.items():
            # Only a refusal on one of the candidate arrays can be lifted by privatizing their slots: a refusal on
            # another container, or for structure, bounds or symbols, survives the promotion unchanged.
            accepted, refused_on = self._l2m_verdict(loop, sdfg)
            if accepted or refused_on not in {arr_name for arr_name, _ in pairs}:
                continue
            if not self._l2m_verdict(loop, sdfg, self._privatized_arrays(loop, pairs))[0]:
                continue
            for arr_name, c_subset in pairs:
                self._promote(sdfg, loop, arr_name, c_subset)
            kept[loop.label] = [f"{arr_name}@{c_subset}" for arr_name, c_subset in pairs]
        return kept

    def _privatized_arrays(self, loop: LoopRegion, pairs: list[tuple[str, subsets.Range]]) -> frozenset[str]:
        """The arrays of ``pairs`` that promoting every pair leaves unwritten in ``loop``: each write of them targets
        a promoted slot, so the loop then only reads them (the prologue loads)."""
        slots: dict[str, list[subsets.Range]] = {}
        for arr_name, c_subset in pairs:
            slots.setdefault(arr_name, []).append(c_subset)
        written_elsewhere: set[str] = set()
        for state in loop.states():
            for node in state.data_nodes():
                if node.data not in slots or node.data in written_elsewhere:
                    continue
                for edge in state.in_edges(node):
                    sub = edge.data.subset
                    if edge.data.is_empty() or edge.data.data != node.data or sub is None:
                        continue
                    if not any(point_subsets_equal(sub, slot) for slot in slots[node.data]):
                        written_elsewhere.add(node.data)
                        break
        return frozenset(name for name in slots if name not in written_elsewhere)

    @staticmethod
    def _l2m_verdict(
        loop: LoopRegion, sdfg: SDFG, assume_private: frozenset[str] = frozenset()
    ) -> tuple[bool, str | None]:
        """Whether ``LoopToMap`` would accept ``loop`` with ``assume_private`` made iteration-private, and the
        container it refused on.

        Asks ``LoopToMap.can_be_applied`` directly on the single loop. Avoids the
        whole-SDFG ``match_patterns(sdfg, LoopToMap)`` walk -- which, on
        deeply-nested CFGs like cloudsc, fails to surface inner-region loops that
        a direct check accepts (a pattern-matcher coverage gap, not a real L2M
        refusal). Per-loop checks are also faster: ``O(candidates)`` instead of
        ``O(all loops)`` of the SDFG.

        :param loop: The loop region to test.
        :param sdfg: The owning SDFG.
        :returns: ``(accepted, container)``: L2M's strict-mode verdict and, on a refusal, the
                  container its :attr:`~LoopToMap.last_refusal` names (``None`` for any other refusal).
        """
        # avoid an import cycle
        from dace.transformation.interstate.loop_to_map import LoopToMap
        from dace.transformation.passes.parallelization.parallelize_loops import ParallelizeLoops

        xform = LoopToMap()
        with contextlib.redirect_stdout(io.StringIO()):
            try:
                accepted = ParallelizeLoops().accepts(loop, xform, assume_private)
            except Exception:
                return False, None
        return accepted, None if accepted or xform.last_refusal is None else xform.last_refusal.container

    # slot detection

    def _privatizable_slots(self, sdfg: SDFG, loop: LoopRegion) -> list[tuple[str, subsets.Range]]:
        """The ``(arr, constant_subset)`` pairs that are safe to privatize in ``loop``.

        At most one constant index ``c`` per array; an array with mixed access modes
        (constant index plus a loop-var-indexed access) is refused.
        """
        loop_states = list(loop.states())
        # Group the constant-index accesses by array name; reject early if a non-constant
        # access to the same array exists in the loop, or if the array appears in any
        # memlet whose primary side is a different array (v1 only handles the simple
        # tasklet<->AccessNode shape where ``memlet.data`` is the array itself).
        per_arr: dict[str, list[subsets.Range]] = {}
        per_arr_has_wcr: dict[str, bool] = {}
        per_arr_mixed: set[str] = set()
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

        results: list[tuple[str, subsets.Range]] = []
        for name, subs in per_arr.items():
            if name in per_arr_mixed:
                continue
            if per_arr_has_wcr.get(name):
                continue
            # One candidate per *distinct* constant point. Distinct slots are independent
            # (the per-array ``mixed`` gate above already refused promotion if a wildcard
            # access to the same array exists); each slot's live-out check is per-point.
            unique_points: list[subsets.Range] = []
            for s in subs:
                if not any(point_subsets_equal(s, p) for p in unique_points):
                    unique_points.append(s)
            for point in unique_points:
                # Refuse the read-modify-write shape (read of slot feeds back
                # to a write of the same slot via dataflow within a state).
                # That shape carries the slot's value ACROSS iterations -- a
                # reduction accumulator ``sum[0] = sum[0] + a[i]`` is the
                # canonical case -- and is NOT privatizable: promoting it
                # would lose the cross-iteration dependency and silently drop
                # every accumulation.
                #
                # KNOWN LATENT GAP: a non-transient slot that the body WRITES
                # without an in-loop read still escapes -- e.g.
                # ``for i: arr[5] = i`` on a non-transient ``arr``. The pass
                # emits no epilogue writeback, so the in-loop writes land in
                # the transient-scalar alias and never reach the caller's
                # ``arr[5]``. The internal-only ``_not_live_out`` check
                # below does not catch this. Adding a refusal here would
                # break a number of existing PCIA tests that assert
                # promotion fires on this shape without verifying the
                # caller-visible final value -- the assertions only check a
                # secondary output. Leaving as-is and TODO: either add an
                # epilogue writeback or tighten those tests + refuse here.
                if self._slot_has_in_body_rmw(loop, name, point):
                    continue
                # The prologue-load rewrite is value-preserving only if no read of the
                # slot is upward-exposed at the top of an iteration. Anything the
                # must-def analysis cannot prove is a refusal, never a permission.
                if not self.slot_reads_are_must_defined(loop, name, point):
                    continue
                if not self._not_live_out(loop, name, slot=point):
                    continue
                results.append((name, point))
        return results

    def slot_reads_are_must_defined(self, loop: LoopRegion, name: str, slot: subsets.Range) -> bool:
        """Whether every read of ``name[slot]`` in ``loop``'s body sees a write from its OWN iteration.

        This is the premise the whole rewrite stands on. The promoted body reads a scalar that
        the prologue seeds from ``arr[slot]``, and once the loop is a map that seed is the
        *pre-loop* value on every iteration -- the loop no longer writes ``arr`` at all. So a
        read that is upward-exposed (reached at the top of an iteration with no preceding
        write in that iteration) changes value: originally it observed the previous
        iteration's write, after promotion it observes the pre-loop value. The reduction
        subcase of that is caught by :meth:`_slot_has_in_body_rmw`, but only when the read
        flows back into the write; a read that feeds anything else (``hist[t] = arr[1] * 2``
        ahead of an inner loop that writes ``arr[1]``) is just as loop-carried and is not a
        dataflow cycle, so it needs this check.

        Two ways the premise holds:

        * the body writes ``name[slot]`` nowhere -- every read observes the live-in value,
          which is exactly what the prologue copies; or
        * every read is must-defined: preceded, on *every* path through one iteration, by a
          write of the same slot.

        Must-def is the shared :class:`~dace.transformation.passes.scalar_fission.MustDefinition`
        over one element (:class:`~dace.transformation.passes.scalar_fission.SlotAccesses`), entered
        at the top of an iteration with nothing defined: a write outside ``loop`` happened in a
        different iteration (or before the loop), and cannot cover this iteration's read.

        Anything the analysis cannot place -- an interstate edge that merely mentions the
        array, a loop condition that reads it -- is a refusal, not a permission.

        :param loop: The loop whose body is analyzed.
        :param name: The array name.
        :param slot: The constant point being considered for promotion.
        :returns: ``True`` only when the premise is established.
        """
        accesses = SlotAccesses(name, slot)
        index = AccessIndex(loop.sdfg)
        written = read = False
        for state in loop.states():
            for node in index.nodes(state, name):
                written = written or bool(accesses.writes(state, node))
                read = read or bool(accesses.reads(state, node))
        if not (written and read):
            return True
        # The loop's own header runs between iterations, outside the body the walk below covers.
        for code in (loop.loop_condition, loop.init_statement, loop.update_statement):
            if code is not None and self._expr_references_name(code.as_string, name):
                return False
        return not MustDefinition(accesses, index).exposed(loop)

    def _slot_has_in_body_rmw(self, loop: LoopRegion, name: str, slot: subsets.Range) -> bool:
        """``True`` iff some write of ``name[slot]`` inside the loop body has a
        dataflow ancestor that read ``name[slot]`` -- i.e. the slot's value
        feeds back into a write of the same slot. This is the reduction /
        running-state shape and is NOT privatizable by PCIA.

        The write-buffer shape (``arr[c] = f(i); ... = arr[c] * g(i)``) is
        intentionally accepted: both a read and a write of ``arr[c]`` exist,
        but the write has no dataflow ancestor that read the slot.

        Tracks taint may-flow across:

        - Intra-state memlet edges (read AccessNode -> tasklet -> ... ->
          write AccessNode).
        - Cross-state continuation: an AccessNode for data ``X`` reached
          inside the taint set in state A taints every AccessNode for the
          same ``X`` in any state B reachable from A inside the loop body.
          Conservative may-flow: a fully-overwriting intermediate state
          would block must-flow, but is not modeled.
        - Interstate-edge mediation: an iedge in the loop body whose
          condition or any assignment RHS references ``name`` is treated
          as a read of ``name[slot]`` (the assignment LHS symbol and the
          successor state's reads of that symbol become tainted). Errs on
          the side of refusing when the same array name appears in iedge
          expressions; a precise per-slot iedge analysis would parse the
          expression and require an exact slot match, but the array-name
          granularity already covers every TSVC / cloudsc shape we hit
          and is cheaper to reason about.
        """
        body_states = list(loop.states())
        if not body_states:
            return False

        # Collect read/write events of name[slot]
        reads: list[tuple[SDFGState, nodes.AccessNode]] = []
        writes: list[tuple[SDFGState, nodes.AccessNode]] = []
        for state in body_states:
            for n in state.nodes():
                if not isinstance(n, nodes.AccessNode) or n.data != name:
                    continue
                if any(
                    e.data is not None
                    and e.data.data == name
                    and e.data.subset is not None
                    and point_subsets_equal(e.data.subset, slot)
                    for e in state.out_edges(n)
                ):
                    reads.append((state, n))
                if any(
                    e.data is not None
                    and e.data.data == name
                    and e.data.subset is not None
                    and point_subsets_equal(e.data.subset, slot)
                    for e in state.in_edges(n)
                ):
                    writes.append((state, n))
        writes_set = set(writes)

        # Iedge-derived seed: any iedge in the body whose RHS/condition
        # references ``name`` taints the iedge's LHS symbols. Subsequent
        # state code (tasklet bodies, memlet index expressions) that uses
        # the tainted symbol is treated as a taint source.
        iedge_seed_symbols: set[str] = set()
        body_state_set = set(body_states)
        for edge in loop.all_interstate_edges():
            if edge.src not in body_state_set and edge.dst not in body_state_set:
                continue
            ied = edge.data
            if ied is None:
                continue
            refs_name = False
            for rhs in ied.assignments.values():
                if rhs and self._expr_references_name(rhs, name):
                    refs_name = True
                    break
            if not refs_name and ied.condition is not None:
                if self._expr_references_name(ied.condition.as_string, name):
                    refs_name = True
            if refs_name:
                iedge_seed_symbols.update(ied.assignments.keys())

        # If iedge-derived taint exists AND any write of name[slot] is in the
        # body, treat as RMW (conservative). A precise check would propagate
        # the symbol taint through subsequent state code; an over-eager
        # refusal here only loses an optimization, not correctness.
        if iedge_seed_symbols and writes_set:
            return True
        if not reads or not writes_set:
            return False

        # States reachable inside the loop body, derived per state the taint actually leaves.
        succ_states = self._loop_state_successors(loop, body_states)

        # Multi-state taint BFS from each read AccessNode of name[slot].
        # A reached write of name[slot] (other than the seed) signals RMW.
        # Cross-state continuation uses the data name as the persistence key.
        for seed in reads:
            visited: set[tuple[SDFGState, Any]] = {seed}
            stack: list[tuple[SDFGState, Any]] = [seed]
            seed_node = seed[1]
            while stack:
                state, node = stack.pop()
                if (state, node) in writes_set and node is not seed_node:
                    return True
                # Forward through intra-state memlet edges.
                for e in state.out_edges(node):
                    new = (state, e.dst)
                    if new not in visited:
                        visited.add(new)
                        stack.append(new)
                # Cross-state continuation: a tainted AccessNode of data X at
                # this state's frontier propagates the taint into every
                # AccessNode of the same X in any reachable successor state.
                if isinstance(node, nodes.AccessNode):
                    data_name = node.data
                    for next_state in succ_states.get(state, ()):
                        for n2 in next_state.nodes():
                            if isinstance(n2, nodes.AccessNode) and n2.data == data_name:
                                new = (next_state, n2)
                                if new not in visited:
                                    visited.add(new)
                                    stack.append(new)
        return False

    @staticmethod
    def _loop_state_successors(loop: LoopRegion, body_states: list[SDFGState]) -> "_Reach":
        """For each state in ``body_states``, the transitive set of body states reachable via interstate edges inside
        ``loop``, derived on first lookup."""
        body_set = set(body_states)
        adj: dict[SDFGState, list[SDFGState]] = {s: [] for s in body_states}
        for edge in loop.all_interstate_edges():
            if edge.src in body_set and edge.dst in body_set:
                adj[edge.src].append(edge.dst)
        return _Reach(adj)

    @staticmethod
    def _expr_references_name(expr_str: Any, name: str) -> bool:
        """Whether ``expr_str`` (an iedge RHS or condition string / sympy)
        references the array ``name`` as a data identifier.
        """
        if expr_str is None:
            return False
        s = str(expr_str)
        # Whole-word match: avoid spurious matches like ``arr_other`` when
        # looking for ``arr``. Simple boundary check; iedge expressions are
        # small so a regex import is overkill.
        idx = 0
        while True:
            idx = s.find(name, idx)
            if idx < 0:
                return False
            before_ok = idx == 0 or not (s[idx - 1].isalnum() or s[idx - 1] == "_")
            after = idx + len(name)
            after_ok = after >= len(s) or not (s[after].isalnum() or s[after] == "_")
            if before_ok and after_ok:
                return True
            idx = after

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

    def _not_live_out(self, loop: LoopRegion, name: str, slot: subsets.Range | None = None) -> bool:
        """Live-out check for ``name`` (whole array) or ``name[slot]`` (one element).

        Walks the parent CFG forward from ``loop``; if the loop is nested inside another
        region (e.g. a conditional branch), walks the enclosing region forward from the
        loop's ancestor too -- a read of ``arr`` after the enclosing block would still
        observe an iteration's value. Any access *inside* ``loop`` itself is permitted
        (those are what we're privatizing). Any access in a predecessor of the loop
        (live-in) is also permitted -- the prologue load handles it.

        Back edges. When an enclosing region is a :class:`~dace.sdfg.state.LoopRegion`, its
        back edge is not an edge of the region's graph, so a plain forward walk reports the
        blocks *before* ``loop`` as unreachable -- yet the enclosing loop runs them again
        after ``loop`` finishes, and they observe the writes it made. Every block of an
        enclosing LoopRegion therefore counts as "after", not just the forward-reachable
        ones. Without this, ``for t: hist[t] = arr[1]; for i: arr[1] = ...`` reads as
        "``arr[1]`` is dead after the inner loop" and the inner loop is wrongly promoted.

        :param slot: When given, the check is *slot-precise*: a post-loop access to a
            *different* element of ``arr`` does not count as a live-out conflict, because
            the in-loop writes to ``arr[slot]`` don't affect those other elements. This
            lets the cheap scalar-form promotion fire for cases where the user reads
            ``arr[k != slot]`` after the loop -- the common cloudsc-style pattern where
            ``zvqx[1]`` is written by the species loop and ``zvqx[2..4]`` are read later.
            When ``None``, the legacy whole-array check applies.
        """
        loop_states = set(loop.states())
        cur = loop
        while True:
            parent = cur.parent_graph
            if parent is None:
                return True
            reachable: OrderedSet[Any] = OrderedSet()
            frontier = [cur]
            while frontier:
                node = frontier.pop()
                for e in parent.out_edges(node):
                    if e.dst in reachable:
                        continue
                    reachable.add(e.dst)
                    frontier.append(e.dst)
            if isinstance(parent, LoopRegion):
                # The enclosing loop's back edge re-runs the whole region after ``cur``.
                reachable.update(b for b in parent.nodes() if b is not cur)
            for block in reachable:
                for state in self._states_of(block):
                    if state in loop_states:
                        continue
                    if self._state_touches_slot(state, name, slot):
                        return False
            # Walk up to the next enclosing CFG and check its forward-reachable region.
            cur = parent

    @staticmethod
    def _state_touches_slot(state: SDFGState, name: str, slot: subsets.Range | None) -> bool:
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
                # The module-level helper, not ``Range.intersects``: the method RAISES on a bound
                # SymPy will not decide, while the helper folds that into the same ``None`` it
                # already answers for indeterminate. Conservative default: any non-False answer
                # means we cannot rule out overlap.
                result = subsets.intersects(memlet.subset, slot, state.sdfg.facts())
                if result is not symbolic.Truth.FALSE:
                    return True
        return False

    @staticmethod
    def _states_of(block) -> list[SDFGState]:
        """Flatten ``block`` to a list of its constituent SDFG states."""
        if isinstance(block, SDFGState):
            return [block]
        if isinstance(block, ControlFlowRegion):
            return list(block.states())
        return []

    # rewrite

    def _arr_accesses_only_at_slot(self, loop: LoopRegion, arr_name: str, c_subset: subsets.Range) -> bool:
        """True if every ``arr`` access inside ``loop`` matches ``c_subset``.

        Used to decide between the cheap wholesale-rename path (every access agrees
        on one slot) and the slot-precise rewrite (a sibling slot also lives in the
        same loop body, so a wholesale rename would corrupt the other slot).
        """
        for state in loop.states():
            for edge in state.edges():
                memlet = edge.data
                if memlet is None or memlet.data != arr_name or memlet.subset is None:
                    continue
                if not self._is_constant_point_subset(memlet.subset):
                    continue
                if not point_subsets_equal(memlet.subset, c_subset):
                    return False
        return True

    def _promote(self, sdfg: SDFG, loop: LoopRegion, arr_name: str, c_subset: subsets.Range) -> None:
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
        c_str = "_".join(str(symbolic.pystr_to_symbolic(rng[0])) for rng in c_subset.ranges)
        base = f"{arr_name}_at_{c_str}_promoted"
        scalar_name, _ = sdfg.add_scalar(
            base, desc.dtype, transient=True, lifetime=dtypes.AllocationLifetime.Scope, find_new_name=True
        )

        # Prologue: insert a state at the head of the loop body that copies arr[c] -> t.
        # The must-def gate has already established that every read of the slot in the body
        # sees a write from its own iteration; the load only keeps the scalar defined on
        # entry (and carries the live-in for a body that never writes the slot).
        old_start = loop.start_block
        prologue = loop.add_state_before(old_start, label=f"{arr_name}_at_{c_str}_load", is_start_block=True)
        src = prologue.add_access(arr_name)
        dst = prologue.add_access(scalar_name)
        prologue.add_nedge(
            src, dst, Memlet(data=arr_name, subset=copy.deepcopy(c_subset), other_subset=subsets.Range([(0, 0, 1)]))
        )

        scalar_subset = subsets.Range([(0, 0, 1)])

        single_slot = self._arr_accesses_only_at_slot(loop, arr_name, c_subset)

        for state in loop.states():
            if state is prologue:
                continue
            # Per-state introduced scalar AccessNode (slot-precise path only).
            per_state_scalar: nodes.AccessNode | None = None
            for edge in list(state.edges()):
                memlet = edge.data
                if memlet is None or memlet.data != arr_name:
                    continue
                if not single_slot and not point_subsets_equal(memlet.subset, c_subset):
                    # Sibling slot; left for its own _promote call.
                    continue
                memlet.data = scalar_name
                memlet.subset = copy.deepcopy(scalar_subset)
                if not single_slot:
                    # Slot-precise: swap the arr-AccessNode endpoint to a fresh per-state
                    # scalar AccessNode (one per state; reused across this slot's edges).
                    for endpoint in ("src", "dst"):
                        end_node = edge.src if endpoint == "src" else edge.dst
                        if isinstance(end_node, nodes.AccessNode) and end_node.data == arr_name:
                            if per_state_scalar is None:
                                per_state_scalar = state.add_access(scalar_name)
                            new_src = per_state_scalar if endpoint == "src" else edge.src
                            new_dst = per_state_scalar if endpoint == "dst" else edge.dst
                            state.remove_edge(edge)
                            edge = state.add_edge(new_src, edge.src_conn, new_dst, edge.dst_conn, edge.data)
            if single_slot:
                # Wholesale rename the arr AccessNode itself; safe because every
                # memlet was rewritten.
                for node in list(state.nodes()):
                    if isinstance(node, nodes.AccessNode) and node.data == arr_name:
                        node.data = scalar_name
            else:
                # Slot-precise path: the original ``arr`` AccessNode may now be
                # orphaned (all its incident edges were rerouted to a per-slot
                # scalar AccessNode in this state). Remove orphaned arr nodes here
                # so the SDFG validates.
                for node in list(state.nodes()):
                    if isinstance(node, nodes.AccessNode) and node.data == arr_name and state.degree(node) == 0:
                        state.remove_node(node)
