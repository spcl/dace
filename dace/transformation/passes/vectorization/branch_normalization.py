# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Flatten residual ``ConditionalBlock`` s into ``ITE``-tasklet form.

Runs after ``SameWriteSetIfElseToITECFG`` (which handles identical-write
arms). Single-arm ``if`` becomes ``arr = ITE(cond, expr, arr)``;
disjoint two-arm ``if/else`` is split into two sequential single-arm
conditionals and re-normalized; overlapping-but-not-identical write sets
are unsupported (``NotImplementedError``). No ``ConditionalBlock`` remains
afterwards.
"""
import copy
from typing import Dict, Optional, Set

import dace
from dace import properties, symbolic
from dace.properties import CodeBlock
from dace.sdfg.construction_utils import (
    assert_connector_role_matches_edges,
    move_branch_cfg_up_discard_conditions,
)
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion
from dace.transformation import pass_pipeline as ppl
from dace.transformation.passes.vectorization.utils.symbolic_polymorphism import free_symbol_names


def compute_arm_escape_writes(sdfg: dace.SDFG, cb: ConditionalBlock) -> Dict[int, Set[str]]:
    """Per-arm array writes that must be rerouted to a private transient.

    A write of ``arr`` in arm ``i`` escapes iff ``arr`` is non-transient,
    or read outside ``cb`` (sibling states plus interstate condition /
    assignment text), or read by another arm (all arms run unconditionally
    after the rewrite).

    :param sdfg: SDFG used for name resolution; the owning SDFG of ``cb``
        is used internally as ``cb`` may be nested.
    :param cb: the conditional block being normalized.
    :returns: ``{arm_index: {escaping_arr_name, ...}}`` (empty set per arm
        with no escaping writes).
    """
    # Resolve to the SDFG that physically owns ``cb`` (it may be inside a
    # NestedSDFG). Arrays referenced from cb's arms live in this SDFG.
    local_sdfg: dace.SDFG = cb.sdfg

    # Collect arm bodies and the SDFGStates inside each, in order.
    arm_bodies = [body for _, body in cb.branches]
    arm_states: Dict[int, Set[dace.SDFGState]] = {}
    for i, body in enumerate(arm_bodies):
        if not isinstance(body, ControlFlowRegion):
            arm_states[i] = set()
            continue
        states_in_arm = {n for n in body.all_control_flow_blocks() if isinstance(n, dace.SDFGState)}
        arm_states[i] = states_in_arm

    inside_states: Set[dace.SDFGState] = set()
    for s in arm_states.values():
        inside_states |= s

    # ---- Outside-read set (rule 2). ----
    outside_reads: Set[str] = set()
    for state in local_sdfg.all_states():
        if state in inside_states:
            continue
        read_set, _ = state.read_and_write_sets()
        outside_reads |= read_set

    # Walk every interstate edge whose endpoints both lie outside cb's
    # subtree. ``read_and_write_sets`` does not cover conditions or
    # assignment RHS expressions, so we tokenise them and intersect
    # against the SDFG's array names.
    array_names = set(local_sdfg.arrays.keys())
    for cfg in local_sdfg.all_control_flow_regions(recursive=True):
        for e in cfg.edges():
            src_in = e.src in inside_states
            dst_in = e.dst in inside_states
            if src_in and dst_in:
                continue
            assigns = e.data.assignments
            for v in assigns.values():
                outside_reads |= symbolic.symbols_in_code(str(v), potential_symbols=array_names)
            cond = e.data.condition.as_string if e.data.condition is not None else ""
            outside_reads |= symbolic.symbols_in_code(cond, potential_symbols=array_names)

    # ConditionalBlock branch conditions live on the block itself, not on
    # interstate edges. The condition of ``cb`` and its sibling cond
    # blocks may reference transients too; rule 2 only cares about
    # conditions *outside* cb.
    from dace.sdfg.state import ConditionalBlock as _CB  # local alias
    for region in local_sdfg.all_control_flow_blocks():
        if not isinstance(region, _CB) or region is cb:
            continue
        for c, _ in region.branches:
            if c is None:
                continue
            text = c.as_string if isinstance(c, CodeBlock) else str(c)
            outside_reads |= symbolic.symbols_in_code(text, potential_symbols=array_names)

    # ---- Per-arm read sets for rule 3. ----
    arm_reads: Dict[int, Set[str]] = {}
    for i, body in enumerate(arm_bodies):
        reads: Set[str] = set()
        if isinstance(body, ControlFlowRegion):
            for state in arm_states[i]:
                r, _ = state.read_and_write_sets()
                reads |= r
        arm_reads[i] = reads

    # ---- Classify per-arm writes. ----
    result: Dict[int, Set[str]] = {}
    for i, body in enumerate(arm_bodies):
        escaping: Set[str] = set()
        if not isinstance(body, ControlFlowRegion):
            result[i] = escaping
            continue
        # Collect this arm's writes via state.read_and_write_sets.
        writes_in_arm: Set[str] = set()
        for state in arm_states[i]:
            _, w = state.read_and_write_sets()
            writes_in_arm |= w

        other_arms_reads = set()
        for j in range(len(arm_bodies)):
            if j == i:
                continue
            other_arms_reads |= arm_reads[j]

        for arr in writes_in_arm:
            if arr not in local_sdfg.arrays:
                continue
            desc = local_sdfg.arrays[arr]
            non_transient = not desc.transient
            if non_transient or arr in outside_reads or arr in other_arms_reads:
                escaping.add(arr)
        result[i] = escaping
    return result


@properties.make_properties
class BranchNormalization(ppl.Pass):
    """Flatten residual ``ConditionalBlock``s into ``ITE``-tasklet form.

    See module docstring for the contract.
    """

    CATEGORY: str = "Vectorization Preparation"

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.CFG | ppl.Modifies.States | ppl.Modifies.AccessNodes

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def apply_pass(self, sdfg: dace.SDFG, _) -> Optional[int]:
        """Flatten every ``ConditionalBlock`` to fixed point.

        :param sdfg: SDFG to transform in place.
        :returns: number of rewrites, or ``None`` if none.
        """
        rewritten = 0
        # Repeat-until-fixed-point. Each pass may split disjoint two-arm
        # conditionals into pairs that need another rewrite cycle.
        progress = True
        while progress:
            progress = False
            for cfg in list(sdfg.all_control_flow_regions(recursive=True)):
                for block in list(cfg.nodes()):
                    if not isinstance(block, ConditionalBlock):
                        continue
                    if self._try_rewrite(sdfg, block):
                        rewritten += 1
                        progress = True

        # Audit each state we may have touched.
        for state in sdfg.all_states():
            assert_connector_role_matches_edges(state)

        return rewritten or None

    def _try_rewrite(self, sdfg: dace.SDFG, cb: ConditionalBlock) -> bool:
        """Dispatch ``cb`` to the single-arm or disjoint two-arm rewrite.

        :param sdfg: SDFG used for name resolution.
        :param cb: the conditional block to attempt.
        :returns: ``True`` if a rewrite was applied.
        """
        # Hoist branch-invariant interstate symbol bindings (e.g. the
        # python-frontend ``__sym_z1 = z1`` alias state) out of the arms
        # first, so an arm that is only "empty-assign state -> compute
        # state" reduces to its single substantive state and the existing
        # single-state ITE path applies. Genuinely branch-variant
        # assignments are left in place and refused loudly downstream.
        self._hoist_branch_invariant_assignments(cb)

        branches = cb.branches
        if len(branches) == 1:
            cond, body = branches[0]
            if cond is None:
                # Bare ``else`` with no condition is nonsensical for this pass.
                return False
            return self._normalize_single_arm(sdfg, cb, cond, body)

        if len(branches) == 2:
            (cond0, body0), (cond1, body1) = branches
            if cond0 is not None and cond1 is None:
                # Asymmetric arms (different state counts, or not both a
                # single substantive SDFGState) cannot go through the
                # symmetric same-write-set / disjoint single-state path.
                # Serialize ``if c: A else: B`` into ``if c: A`` then
                # ``if not c: B`` (exactly one fires; ``c`` is arm-invariant
                # here so this is value-preserving even for a shared write
                # set), and let later cycles normalize each single arm.
                if self._arms_are_asymmetric(body0, body1):
                    return self._serialize_two_arm(cb, cond0, body0, body1)
                # Two-arm if/else: split disjoint writes into two single-arm
                # conditionals first, then let the next pass cycle handle each.
                return self._split_two_arm_disjoint(sdfg, cb, cond0, body0, body1)

        return False

    def _hoist_branch_invariant_assignments(self, cb: ConditionalBlock) -> bool:
        """Hoist branch-invariant interstate symbol bindings out of each arm.

        The python frontend often begins an arm with an empty
        ``SDFGState`` whose only effect is an interstate ``assignments``
        binding (e.g. ``__sym_z1 = z1``, aliasing a kernel symbol used by
        the arm's array accesses). Such a binding is *branch-invariant*
        when its RHS is computable before the ``ConditionalBlock`` — its
        free symbols are not produced by any arm — and the bound symbol
        is neither a branch-predicate symbol nor already bound to a
        different expression on an edge entering ``cb``. It then has the
        same value whichever arm runs, so moving it onto the edges
        entering ``cb`` is value-preserving and reduces the arm to its
        substantive compute (which the single-state ITE normalization
        handles). Genuinely branch-variant assignments are left in place
        and refused loudly by :meth:`_normalize_single_arm`.

        :param cb: the conditional block whose arms are simplified.
        :returns: ``True`` if any assignment was hoisted.
        """
        parent = cb.parent_graph
        in_edges = list(parent.in_edges(cb))
        if not in_edges:
            # ``cb`` is its parent region's entry; no edge to hoist onto
            # without restructuring the start block. Leave as-is (rare).
            return False
        # Symbols produced inside any arm are NOT available before ``cb``.
        arm_assigned: Set[str] = set()
        for _c, br in cb.branches:
            for e in br.edges():
                arm_assigned |= set(e.data.assignments.keys())
        # Branch-predicate symbols must keep their pre-``cb`` value.
        pred_syms: Set[str] = set()
        for c, _br in cb.branches:
            if c is not None:
                pred_syms |= symbolic.symbols_in_code(c.as_string if isinstance(c, CodeBlock) else str(c))
        hoisted = False
        for _c, br in cb.branches:
            sb = br.start_block
            if not (isinstance(sb, dace.SDFGState) and sb.is_empty()):
                continue
            oes = br.out_edges(sb)
            if len(oes) != 1 or not oes[0].data.assignments:
                continue
            e = oes[0]
            assigns = dict(e.data.assignments)
            hoistable = True
            for sym, expr in assigns.items():
                if symbolic.symbols_in_code(str(expr)) & arm_assigned:
                    hoistable = False  # RHS depends on an arm-produced symbol
                    break
                if sym in pred_syms:
                    hoistable = False  # would change which branch is taken
                    break
                for ie in in_edges:
                    if sym in ie.data.assignments and str(ie.data.assignments[sym]) != str(expr):
                        hoistable = False  # conflicting pre-``cb`` binding
                        break
                if not hoistable:
                    break
            if not hoistable:
                continue
            for ie in in_edges:
                ie.data.assignments.update(assigns)
            # Drop the now-purposeless empty pass-through entry state and
            # promote its single successor as the new arm start_block so
            # M3.1b's single-state guard accepts the reduced arm. Leaving
            # the empty state in place was preventing
            # ``SameWriteSetIfElseToITECFG._matches`` from recognising
            # two-arm same-write-set kernels whose Python-frontend SDFG
            # begins each arm with a symbol-binding empty state (the
            # cloudsc-snippet-one ``__sym_z1 = z1`` pattern).
            successor = e.dst
            br.remove_edge(e)
            br.remove_node(sb)
            br.start_block = br.node_id(successor)
            hoisted = True
        return hoisted

    @staticmethod
    def _substantive_states(body: ControlFlowRegion):
        """SDFGStates in ``body`` that hold compute (non-empty)."""
        return [n for n in body.nodes() if isinstance(n, dace.SDFGState) and not n.is_empty()]

    def _arms_are_asymmetric(self, body0: ControlFlowRegion, body1: ControlFlowRegion) -> bool:
        """Whether the two arms cannot use the symmetric single-state path.

        True when the arms differ in state count, or either arm is not a
        single substantive ``SDFGState`` (e.g. an empty symbol-assignment
        entry state preceding the body, as the python frontend emits).
        """
        if not (isinstance(body0, ControlFlowRegion) and isinstance(body1, ControlFlowRegion)):
            return False
        if len(body0.nodes()) != len(body1.nodes()):
            return True
        return len(self._substantive_states(body0)) != 1 or len(self._substantive_states(body1)) != 1

    def _serialize_two_arm(self, cb: ConditionalBlock, cond0: CodeBlock, body0: ControlFlowRegion,
                           body1: ControlFlowRegion) -> bool:
        """Serialize ``if c: A else: B`` into ``if c: A`` then ``if not c: B``.

        Pure CFG rewrite (no disjoint-write or single-state requirement):
        ``cb`` keeps only the if-arm; a new negated single-arm block holds
        the else-arm and is stitched sequentially after ``cb``. Subsequent
        normalization cycles handle each single-arm form. Valid for a
        shared write set because the condition is not mutated by the arms,
        so exactly one arm's writes take effect — identical to the
        original if/else.

        :param cb: the two-arm conditional (becomes the if-arm only).
        :param cond0: the if condition.
        :param body0: the if-arm body (kept on ``cb``).
        :param body1: the else-arm body (moved to the negated block).
        :returns: ``True`` (always serializes).
        """
        parent = cb.parent_graph
        cond_text = cond0.as_string if isinstance(cond0, CodeBlock) else str(cond0)
        cb.remove_branch(body1)
        neg_block = ConditionalBlock(label=f"{cb.label}_negated", sdfg=parent.sdfg, parent=parent)
        neg_block.add_branch(CodeBlock(f"not ({cond_text})"), body1)
        parent.add_node(neg_block)
        out_edges = list(parent.out_edges(cb))
        for oe in out_edges:
            parent.remove_edge(oe)
            parent.add_edge(neg_block, oe.dst, copy.deepcopy(oe.data))
        parent.add_edge(cb, neg_block, dace.InterstateEdge())
        parent.reset_cfg_list()
        return True

    def _normalize_single_arm(self, sdfg: dace.SDFG, cb: ConditionalBlock, cond: CodeBlock,
                              body: ControlFlowRegion) -> bool:
        """Lower ``if cond: body`` to ``arr = ITE(cond, expr, arr)`` writes.

        :param sdfg: SDFG used for name resolution.
        :param cb: the single-arm conditional block (removed in place).
        :param cond: the arm condition.
        :param body: the arm body (one tasklet/access-node state).
        :returns: ``True`` if normalized, ``False`` if the shape is unsupported.
        :raises NotImplementedError: if the arm region carries an interstate
            assignment. Lifting the region wholesale makes that binding
            unconditional; that is value-preserving only if every assigned
            symbol is consumed solely within the arm — which we do not
            prove. Refuse loudly rather than silently rebind a symbol that
            another block may read (conservative, like MapFission /
            StateFusionExtended refusing unprovable shapes).
        """
        # The arm may be a linear chain of states: empty entry states (the
        # python frontend emits an ``if_<n>`` state) followed by one
        # substantive compute state. The wholesale lift preserves the
        # region's structure and gates only the escaping writes via
        # ``ITE``; the side-effect-free compute runs unconditionally and
        # the ITE selects. But an interstate *assignment* inside the arm
        # is a conditional symbol binding — lifting it makes it
        # unconditional, which silently corrupts any out-of-arm consumer of
        # that symbol. We don't prove arm-locality, so refuse.
        states = [n for n in body.nodes() if isinstance(n, dace.SDFGState)]
        if len(states) != len(body.nodes()):
            return False
        local_sdfg_for_arm: dace.SDFG = cb.sdfg
        for e in body.edges():
            if not e.data.assignments:
                continue
            # Each assigned symbol must be *arm-local*: every read of the
            # symbol lives inside ``body``. Otherwise the lift breaks
            # downstream consumers.
            non_local = []
            for sym in e.data.assignments.keys():
                if not self._symbol_is_arm_local(local_sdfg_for_arm, body, sym):
                    non_local.append(sym)
            if non_local:
                raise NotImplementedError(
                    f"BranchNormalization: IF arm {body.label!r} carries interstate assignment(s) "
                    f"for symbol(s) {non_local} that are read outside the arm; lifting would make "
                    f"this conditional symbol binding unconditional (unsafe). Unsupported branch shape.")
        substantive = [s for s in states if not s.is_empty()]
        if not substantive:
            return False
        for s in substantive:
            for n in s.nodes():
                if not isinstance(n, (dace.nodes.AccessNode, dace.nodes.Tasklet)):
                    return False

        # The arm may be a *linear chain* of substantive states (the
        # frontend serialises a multi-statement read-modify-write body —
        # e.g. the cloudsc "tidy up" arm `ptend_q += a; ...; ptend_q +=
        # b` — into several sequential compute states; StateFusion-
        # Extended correctly refuses to fuse them since they carry a
        # genuine WAR/RAW hazard). Lifting the whole chain and applying
        # the per-state ITE rewrite to *every* substantive state is
        # value-preserving: when ``cond`` is false each state's ITE
        # picks its running input, so the original propagates unchanged
        # through the chain; when true each increment is applied. Only a
        # straight-line chain is supported (no branching inside the arm).
        ordered = self._linear_state_order(body)
        if ordered is None:
            return False
        ordered_subst = [s for s in ordered if s in substantive]
        if len(ordered_subst) != len(substantive):
            return False

        cond_text = cond.as_string if isinstance(cond, CodeBlock) else str(cond)
        local_sdfg: dace.SDFG = cb.sdfg
        escaping = compute_arm_escape_writes(local_sdfg, cb).get(0, set())

        # Per-state escaping-write subsets, in execution order. Only
        # escaping writes get the ITE gate; arm-internal scratch stays
        # inline (nothing outside the arm observes it).
        per_state = []
        for s in ordered_subst:
            ws = self._collect_write_subsets(s)
            if ws is None:
                return False
            per_state.append((s, {arr: sub for arr, sub in ws.items() if arr in escaping}))

        # Resolve the cond ONCE, on the first substantive state that has
        # an escaping write (its producer is sequenced before every
        # consumer). Non-producer states read the cond array fresh.
        preresolved = None
        resolver_state = None
        for s, ms in per_state:
            if ms:
                any_sub = str(next(iter(ms.values())))
                preresolved = self._resolve_arm_cond(local_sdfg, s, cond_text, any_sub, skip_cb=cb)
                resolver_state = s
                break

        for s, ms in per_state:
            if not ms:
                continue
            if preresolved is not None:
                cname, cprod = preresolved
                pr = (cname, cprod if s is resolver_state else None)
            else:
                pr = None
            self._rewrite_writes_to_ite(local_sdfg, s, ms, cond_text, skip_cb=cb, preresolved=pr)

        move_branch_cfg_up_discard_conditions(if_block=cb, body_to_take=body)
        return True

    @staticmethod
    def _symbol_is_arm_local(sdfg: dace.SDFG, body: ControlFlowRegion, sym: str) -> bool:
        """Whether ``sym``'s reads are confined to ``body`` (the arm region).

        An arm-local symbol binding can be safely lifted out of the
        conditional: the assignment becomes unconditional, but no consumer
        outside the arm reads it, so the lift is value-preserving. A
        symbol referenced anywhere else (sibling state, interstate edge,
        other conditional arm) is NOT arm-local.

        :param sdfg: The SDFG that owns ``body`` and ``sym``.
        :param body: The arm region whose locality is being checked.
        :param sym: The symbol name.
        :returns: ``True`` iff every read of ``sym`` lives inside ``body``.
        """
        arm_blocks = set(body.all_control_flow_blocks(
            recursive=True)) if isinstance(body, ControlFlowRegion) else set(body.nodes())
        # Walk every block in the SDFG; any reference to ``sym`` outside the
        # arm disqualifies the lift.
        for cfg in sdfg.all_control_flow_regions(recursive=True):
            for blk in cfg.nodes():
                if blk in arm_blocks:
                    continue
                # Interstate-edge references (assignments + condition).
                in_edges = cfg.in_edges(blk) + cfg.out_edges(blk)
                for ie in in_edges:
                    for v in ie.data.assignments.values():
                        if sym in symbolic.symbols_in_code(str(v)):
                            return False
                    if ie.data.condition is not None:
                        cond_str = ie.data.condition.as_string if isinstance(ie.data.condition, CodeBlock) else str(
                            ie.data.condition)
                        if sym in symbolic.symbols_in_code(cond_str):
                            return False
                # Branch-block conditions on conditional blocks.
                if isinstance(blk, ConditionalBlock):
                    for cnd, _br in blk.branches:
                        if cnd is None:
                            continue
                        cond_str = cnd.as_string if isinstance(cnd, CodeBlock) else str(cnd)
                        if sym in symbolic.symbols_in_code(cond_str):
                            return False
                # State-level references: tasklet bodies + memlet subsets.
                if isinstance(blk, dace.SDFGState):
                    for n in blk.nodes():
                        if isinstance(n, dace.nodes.Tasklet):
                            if sym in symbolic.symbols_in_code(n.code.as_string):
                                return False
                    for ed in blk.edges():
                        if ed.data is None:
                            continue
                        for s in (ed.data.subset, ed.data.other_subset):
                            if s is None:
                                continue
                            for r in s.ranges:
                                for elem in r:
                                    if elem is None:
                                        continue
                                    fs = free_symbol_names(elem)
                                    if sym in fs:
                                        return False
        return True

    @staticmethod
    def _linear_state_order(body: ControlFlowRegion):
        """Execution-order block list iff ``body`` is a straight-line chain.

        :param body: The arm region.
        :returns: Blocks in execution order, or ``None`` if ``body``
            branches, cycles, or has unreachable blocks (only a linear
            chain is liftable by the per-state ITE composition).
        """
        start = body.start_block
        if start is None:
            return None
        order, seen, cur = [], set(), start
        while cur is not None:
            if cur in seen:
                return None
            seen.add(cur)
            order.append(cur)
            outs = list(body.out_edges(cur))
            if len(outs) == 0:
                cur = None
            elif len(outs) == 1:
                cur = outs[0].dst
            else:
                return None
        if len(order) != len(body.nodes()):
            return None
        return order

    def _split_two_arm_disjoint(self, sdfg: dace.SDFG, cb: ConditionalBlock, cond0: CodeBlock, body0: ControlFlowRegion,
                                body1: ControlFlowRegion) -> bool:
        """Split a disjoint-write ``if/else`` into two sequential single-arm ``if`` s.

        :param sdfg: SDFG used for name resolution.
        :param cb: the two-arm conditional block.
        :param cond0: the ``if`` condition.
        :param body0: the ``if`` arm body.
        :param body1: the ``else`` arm body.
        :returns: ``True`` if split (next sweep normalizes each half),
            ``False`` if the shape is unsupported.
        :raises NotImplementedError: if the arms have overlapping but
            non-identical write subsets.
        """
        if len(body0.nodes()) != 1 or len(body1.nodes()) != 1:
            return False
        s0, s1 = body0.nodes()[0], body1.nodes()[0]
        if not (isinstance(s0, dace.SDFGState) and isinstance(s1, dace.SDFGState)):
            return False
        w0 = self._collect_write_subsets(s0)
        w1 = self._collect_write_subsets(s1)
        if w0 is None or w1 is None:
            return False
        # Same-array writes are only a real conflict when the element
        # subsets actually intersect. Element-disjoint writes (typical
        # cloudsc shape, e.g. ``zsolqa[i,a]`` vs ``zsolqa[i,b]``) split
        # cleanly into ``if c: ...`` and ``if not c: ...``; each arm
        # gates its own subset via the next normalization cycle.
        # ``dace.subsets.intersects`` returns True / False / None; we
        # treat the indeterminate ``None`` as a conservative conflict.
        truly_overlapping = []
        for name in set(w0) & set(w1):
            if dace.subsets.intersects(w0[name], w1[name]) is not False:
                truly_overlapping.append(name)
        if truly_overlapping:
            # Same-element-write case is M3.1b's job; if it reached here,
            # M3.1b didn't match.
            raise NotImplementedError(
                f"BranchNormalization: two-arm ConditionalBlock {cb.label!r} has overlapping "
                f"write subsets {sorted(truly_overlapping)} that M3.1b did not normalize; this pass "
                f"cannot flatten it without dropping or duplicating writes")

        # Split into two single-arm conditionals. The else-body becomes a new
        # ``if not cond0: body1`` block sequentially after ``cb`` (now holding
        # only the if-arm). Subsequent fixed-point iterations of apply_pass
        # will then rewrite each single-arm form.
        parent = cb.parent_graph
        cond_text = cond0.as_string if isinstance(cond0, CodeBlock) else str(cond0)
        cb.remove_branch(body1)
        # New ConditionalBlock for the else-arm with the negated condition.
        neg_block = ConditionalBlock(label=f"{cb.label}_negated", sdfg=parent.sdfg, parent=parent)
        neg_block.add_branch(CodeBlock(f"not ({cond_text})"), body1)
        parent.add_node(neg_block)

        # Stitch: rewire cb's out-edges to flow through neg_block.
        out_edges = list(parent.out_edges(cb))
        for oe in out_edges:
            parent.remove_edge(oe)
            parent.add_edge(neg_block, oe.dst, copy.deepcopy(oe.data))
        parent.add_edge(cb, neg_block, dace.InterstateEdge())
        parent.reset_cfg_list()
        return True

    def _collect_write_subsets(self, state: dace.SDFGState):
        from dace.transformation.passes.vectorization.utils.queries import collect_element_write_subsets
        return collect_element_write_subsets(state)

    def _resolve_arm_cond(self,
                          sdfg: dace.SDFG,
                          state: dace.SDFGState,
                          cond_text: str,
                          any_subset_str: str,
                          skip_cb=None):
        """Resolve the arm condition to ``(cond_array_name, cond_producer)``.

        Materialises the boolean arm condition into an array (the
        ``SameWriteSetIfElseToITECFG`` resolver) and returns the array
        name + the producing access node, or ``(None, None)`` if the
        cond stays an inline expression. Extracted so a multi-state arm
        can resolve **once** (the resolver has a one-shot symbol-lift
        side effect — re-resolving per state would bake stale cond text
        into later ITE tasklets).

        :param sdfg: SDFG for name resolution.
        :param state: State whose scope the cond is resolved against
            (for a multi-state chain: the first substantive state, so
            the producer is sequenced before every consumer).
        :param cond_text: The arm condition expression.
        :param any_subset_str: A representative write subset string.
        :param skip_cb: Conditional block whose conditions to exclude.
        :returns: ``(cond_array_name, cond_producer)``.
        """
        from dace.transformation.passes.vectorization.same_write_set_if_else_to_ite_cfg import (
            SameWriteSetIfElseToITECFG, )  # noqa: avoid import cycle at module load
        resolved = SameWriteSetIfElseToITECFG()._resolve_cond_to_array(sdfg,
                                                                       state,
                                                                       cond_text,
                                                                       any_subset_str,
                                                                       skip_cb=skip_cb)
        return (None, None) if resolved is None else resolved

    def _rewrite_writes_to_ite(self,
                               sdfg: dace.SDFG,
                               state: dace.SDFGState,
                               write_subsets: dict,
                               cond_text: str,
                               *,
                               skip_cb=None,
                               preresolved=None):
        """Redirect each write in ``state`` through ``arr = ITE(cond, expr, arr)``.

        :param sdfg: SDFG used for name resolution.
        :param state: the lifted arm-body state.
        :param write_subsets: ``{arr_name: subset}`` of escaping writes to gate.
        :param cond_text: the arm condition expression.
        :param skip_cb: conditional block whose conditions to exclude when
            resolving the cond symbol (forwarded to the cond resolver).
        """
        # Resolve cond once. The symbol-lifting side effect (deleting the
        # upstream assignment for the cond symbol) must fire at most once
        # across all writes that share the same cond; resolving inside the
        # per-write loop would re-trigger the lift, find the assignment
        # gone after the first iteration, and silently bake the cond text
        # into later ITE tasklets. ``preresolved`` lets a multi-state
        # caller resolve once (on the first substantive state) and feed
        # the same ``(cond_array, producer)`` to every state's rewrite —
        # passing ``producer=None`` for non-producer states forces a
        # fresh in-state read of the (already-computed, earlier-state)
        # cond array, which is the only valid cross-state form.
        if preresolved is not None:
            cond_array_name, cond_producer = preresolved
        else:
            any_subset_str = str(next(iter(write_subsets.values())))
            cond_array_name, cond_producer = self._resolve_arm_cond(sdfg,
                                                                    state,
                                                                    cond_text,
                                                                    any_subset_str,
                                                                    skip_cb=skip_cb)

        for arr_name in list(write_subsets.keys()):
            # Find every write access node for this array in this state.
            # Each write may target a different element subset (cloudsc-style
            # chained writes like ``arr[0,3,it]`` then ``arr[3,0,it]``); the
            # captured ``write_subsets`` dict only carries one entry per array
            # name, so per-write we read the actual subset from each write's
            # own in-edge memlet.
            writes = [n for n in state.nodes() if isinstance(n, dace.nodes.AccessNode) and n.data == arr_name]
            for write_an in writes:
                in_edges = list(state.in_edges(write_an))
                if not in_edges:
                    continue
                if len(in_edges) != 1:
                    raise NotImplementedError(
                        f"BranchNormalization: write to {arr_name!r} has {len(in_edges)} in-edges; "
                        f"only single-edge writes are supported in this slice")
                in_edge = in_edges[0]
                write_subset = in_edge.data.subset

                # Build a 1-element scratch transient ``__bn_<arr>_new`` to
                # hold the computed value for this single element write.
                tmp_name, _ = sdfg.add_array(name=f"__bn_{arr_name}_new",
                                             shape=(1, ),
                                             dtype=sdfg.arrays[arr_name].dtype,
                                             storage=dace.dtypes.StorageType.Register,
                                             transient=True,
                                             find_new_name=True)
                tmp_an = state.add_access(tmp_name)

                # The "old value" for the ITE must be the value ``arr_name``
                # held BEFORE the tasklet that wrote ``write_an`` ran. For
                # cloudsc-style chained RMW patterns (``arr[s] = expr + arr[s]``)
                # that is exactly the access node the tasklet was reading from
                # via its own in-edge at the same subset; for non-RMW writes
                # (no read of ``arr`` at this subset by the same tasklet) a
                # fresh access node falls back to pre-state. Locate the chained
                # source BEFORE redirecting the tasklet's out-edge so we can
                # inspect the tasklet's read pattern.
                writer_tasklet = in_edge.src
                old_an = None
                if isinstance(writer_tasklet, dace.nodes.Tasklet):
                    for re_ in state.in_edges(writer_tasklet):
                        if (isinstance(re_.src, dace.nodes.AccessNode) and re_.src.data == arr_name
                                and re_.data.subset is not None and str(re_.data.subset) == str(in_edge.data.subset)):
                            old_an = re_.src
                            break
                if old_an is None:
                    old_an = state.add_access(arr_name)

                # Redirect the existing in-edge to write to the temp instead.
                state.remove_edge(in_edge)
                state.add_edge(in_edge.src, in_edge.src_conn, tmp_an, None, dace.Memlet(expr=f"{tmp_name}[0]"))
                if cond_array_name is not None:
                    # Reuse the producing access node (see
                    # ``_resolve_cond_to_array``): a fresh read node would
                    # disconnect the lift from this ITE and let codegen
                    # emit the ITE before the cond is computed.
                    cond_access = cond_producer if cond_producer is not None else state.add_access(cond_array_name)
                    ite_t = state.add_tasklet(
                        name=f"bn_ite_{arr_name}",
                        inputs={"_c", "_new", "_old"},
                        outputs={"_o"},
                        code="_o = ITE(_c, _new, _old)",
                    )
                    cond_subset = "0" if sdfg.arrays[cond_array_name].total_size == 1 else write_subset
                    state.add_edge(cond_access, None, ite_t, "_c",
                                   dace.Memlet(expr=f"{cond_array_name}[{cond_subset}]"))
                else:
                    ite_t = state.add_tasklet(
                        name=f"bn_ite_{arr_name}",
                        inputs={"_new", "_old"},
                        outputs={"_o"},
                        code=f"_o = ITE({cond_text}, _new, _old)",
                    )
                state.add_edge(tmp_an, None, ite_t, "_new", dace.Memlet(expr=f"{tmp_name}[0]"))
                state.add_edge(old_an, None, ite_t, "_old", dace.Memlet(expr=f"{arr_name}[{write_subset}]"))
                state.add_edge(ite_t, "_o", write_an, None, dace.Memlet(expr=f"{arr_name}[{write_subset}]"))


def _count_conditional_blocks(sdfg: dace.SDFG) -> int:
    """Count every :class:`ConditionalBlock` anywhere in ``sdfg``.

    :param sdfg: SDFG to scan (recurses through nested SDFGs and regions).
    :returns: number of remaining conditional blocks.
    """
    return sum(1 for cfg in sdfg.all_control_flow_regions(recursive=True) for b in cfg.nodes()
               if isinstance(b, ConditionalBlock))


@properties.make_properties
class BranchNormalizationPipeline(ppl.Pass):
    """Drive M3.1b + M3.2 to a fixed point, fusing branch-arm states with
    :class:`StateFusionExtended` between cycles.

    A ``ConditionalBlock`` whose arm body itself holds a
    ``ConditionalBlock`` (TSVC s2710) is *multi-state*, so the single-state
    guards in :meth:`SameWriteSetIfElseToITECFG._matches` and
    :class:`BranchNormalization` bail and the outer block never normalises.
    Flattening the inner block first explodes the arm into a 4-5 state
    sequence (the 3-CFG ``compute_then``/``compute_else``/``apply_ITE``
    split); ``StateFusionExtended`` collapses that sequence back into one
    state *without* dropping tasklets, so the next iteration's single-state
    path normalises the now-shrunk outer block.

    ``StateFusionExtended`` is required here: plain
    :class:`StateFusion` / ``fuse_states`` discards the arm tasklets on
    these states (it bails on the data-hazard reconstruction), producing an
    empty body and silently wrong results. The extended variant inserts the
    happens-before edges needed to fuse them safely.
    """

    CATEGORY: str = "Vectorization Preparation"

    MAX_ITERS = 8

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.CFG | ppl.Modifies.States | ppl.Modifies.AccessNodes | ppl.Modifies.Tasklets

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self):
        return {}

    def apply_pass(self, sdfg: dace.SDFG, pipeline_results) -> Optional[int]:
        from dace.transformation.passes.vectorization.same_write_set_if_else_to_ite_cfg import (  # avoid import cycle
            SameWriteSetIfElseToITECFG, )
        from dace.transformation.interstate import StateFusionExtended

        results = pipeline_results if pipeline_results is not None else {}
        rewritten = 0
        for _ in range(self.MAX_ITERS):
            before = _count_conditional_blocks(sdfg)
            if before == 0:
                break
            SameWriteSetIfElseToITECFG().apply_pass(sdfg, results)
            BranchNormalization().apply_pass(sdfg, results)
            # Collapse the multi-state arms the 3-CFG split produced so the
            # next cycle's single-state guards can match the outer block.
            sdfg.apply_transformations_repeated(StateFusionExtended)
            after = _count_conditional_blocks(sdfg)
            rewritten += max(0, before - after)
            if after >= before:  # no progress: unsupported shape remains
                break

        remaining = _count_conditional_blocks(sdfg)
        if remaining:
            raise NotImplementedError(f"BranchNormalizationPipeline: {remaining} ConditionalBlock(s) remain after "
                                      f"{self.MAX_ITERS} fixed-point iterations; unsupported branch shape")
        return rewritten or None
