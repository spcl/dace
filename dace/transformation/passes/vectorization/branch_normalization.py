# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Flatten residual ``ConditionalBlock`` s into ``ITE``-tasklet form.

Runs after ``SameWriteSetIfElseToITECFG`` (handles identical-write arms).
Single-arm ``if`` -> ``arr = ITE(cond, expr, arr)``; disjoint two-arm ``if/else``
-> split into two sequential single-arm conditionals + re-normalize;
``>=3`` arms or ``if/elif`` without ``else`` -> a chain of single-arm blocks, arm ``k``
guarded by ``not c0 and ... and ck`` (first-match semantics);
overlapping-but-not-identical write sets unsupported (``NotImplementedError``).
No ``ConditionalBlock`` remains afterwards.
"""
import ast
import copy

from collections.abc import Callable
from typing import Any

import dace
from dace import properties, subsets, symbolic
from dace.memlet import Memlet
from dace.sdfg.graph import MultiConnectorEdge
from dace.sdfg.state import ControlFlowBlock
from dace.properties import CodeBlock
from dace.sdfg.construction_utils import (
    assert_connector_role_matches_edges,
    move_branch_cfg_up_discard_conditions,
)
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion, LoopRegion
from dace.transformation import pass_pipeline as ppl
from dace.transformation.passes.vectorization.same_write_set_if_else_to_ite_cfg import (
    SameWriteSetIfElseToITECFG, arm_accesses_are_in_range_unguarded, condition_guards_iteration_symbol)
from dace.transformation.passes.vectorization.utils.symbolic_polymorphism import free_symbol_names
from dace.ordered import OrderedSet


def rewrite_blocks_to_fixpoint(sdfg: dace.SDFG, rewrite: Callable[[ConditionalBlock], bool]) -> int:
    """Apply ``rewrite`` to every ``ConditionalBlock`` until a sweep changes nothing; returns the rewrite count."""
    rewritten = 0
    progress = True
    while progress:
        progress = False
        for cfg in list(sdfg.all_control_flow_regions(recursive=True)):
            for block in list(cfg.nodes()):
                if isinstance(block, ConditionalBlock) and rewrite(block):
                    rewritten += 1
                    progress = True
    return rewritten


def upward_exposed_reads(state: dace.SDFGState) -> OrderedSet[str]:
    """Containers ``state`` reads at a value it did not produce itself.

    A read through an access node that the same state writes whole first (one element, no WCR) sees that
    write, never a value from before the state, so it is not counted.

    :param state: the state to inspect.
    :returns: the names of the containers read before any covering write in ``state``.
    """
    read_set, _ = state.read_and_write_sets()
    exposed: OrderedSet[str] = OrderedSet()
    for node in state.data_nodes():
        if node.data not in read_set or node.data in exposed:
            continue
        if all(edge.data.is_empty() for edge in state.out_edges(node)):
            continue
        writes = [edge for edge in state.in_edges(node) if not edge.data.is_empty()]
        if writes and all(edge.data.wcr is None for edge in writes) and node.desc(state.sdfg).total_size == 1:
            continue
        exposed.add(node.data)
    return exposed


def compute_arm_escape_writes(sdfg: dace.SDFG, cb: ConditionalBlock) -> dict[int, set[str]]:
    """Per-arm array writes that must be rerouted to a private transient.

    A write of ``arr`` in arm ``i`` escapes iff ``arr`` non-transient, or read
    outside ``cb`` (sibling states + interstate condition / assignment text), or
    read by another arm (all arms run unconditionally after the rewrite).

    :param sdfg: SDFG for name resolution; ``cb``'s owning SDFG is used
        internally as ``cb`` may be nested.
    :param cb: conditional block being normalized.
    :returns: ``{arm_index: {escaping_arr_name, ...}}`` (empty set per arm with
        no escaping writes).
    """
    # SDFG physically owning cb (may be nested); cb's arms' arrays live here.
    local_sdfg: dace.SDFG = cb.sdfg

    arm_bodies = [body for _, body in cb.branches]
    arm_states: dict[int, set[dace.SDFGState]] = {}
    for i, body in enumerate(arm_bodies):
        if not isinstance(body, ControlFlowRegion):
            arm_states[i] = set()
            continue
        states_in_arm = {n for n in body.all_control_flow_blocks() if isinstance(n, dace.SDFGState)}
        arm_states[i] = states_in_arm

    inside_states: set[dace.SDFGState] = set()
    for s in arm_states.values():
        inside_states |= s

    # Outside-read set (rule 2). Only a read that can see a value from before its own state counts: a copy of an
    # arm elsewhere that recomputes a temporary before reading it cannot observe this arm's write.
    outside_reads: set[str] = set()
    for state in local_sdfg.all_states():
        if state in inside_states:
            continue
        outside_reads.update(upward_exposed_reads(state))

    # Interstate edges (>=1 endpoint outside cb): read_and_write_sets misses
    # conditions / assignment RHS, so tokenise them against array names.
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

    # Branch conditions live on the ConditionalBlock, not interstate edges;
    # collect sibling cond-block conditions (rule 2 excludes cb's own).
    from dace.sdfg.state import ConditionalBlock
    for region in local_sdfg.all_control_flow_blocks():
        if not isinstance(region, ConditionalBlock) or region is cb:
            continue
        for c, _ in region.branches:
            if c is None:
                continue
            text = c.as_string if isinstance(c, CodeBlock) else str(c)
            outside_reads |= symbolic.symbols_in_code(text, potential_symbols=array_names)

    # Per-arm read sets for rule 3.
    arm_reads: dict[int, set[str]] = {}
    for i, body in enumerate(arm_bodies):
        reads: set[str] = set()
        if isinstance(body, ControlFlowRegion):
            for state in arm_states[i]:
                reads.update(upward_exposed_reads(state))
        arm_reads[i] = reads

    # Classify per-arm writes.
    result: dict[int, set[str]] = {}
    for i, body in enumerate(arm_bodies):
        escaping: set[str] = set()
        if not isinstance(body, ControlFlowRegion):
            result[i] = escaping
            continue
        writes_in_arm: set[str] = set()
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

    def apply_pass(self, sdfg: dace.SDFG, _: dict[str, Any]) -> int | None:
        """Flatten every ``ConditionalBlock`` to fixed point.

        :param sdfg: SDFG to transform in place.
        :returns: number of rewrites, or ``None`` if none.
        """
        rewritten = rewrite_blocks_to_fixpoint(sdfg, lambda block: self._try_rewrite(sdfg, block))

        # Audit touched states.
        for state in sdfg.all_states():
            assert_connector_role_matches_edges(state)

        return rewritten or None

    def _try_rewrite(self, sdfg: dace.SDFG, cb: ConditionalBlock) -> bool:
        # Dispatch ``cb`` to the single-arm or disjoint two-arm rewrite.
        if self.is_multi_arm(cb) and self.flatten_multi_arm_block(cb):
            return True
        # Hoist branch-invariant bindings (frontend ``__sym_z1 = z1`` alias state) out of arms so an arm
        # reduces to its one substantive state. The hoist stands even if the rewrite is then refused.
        hoisted = self._hoist_branch_invariant_assignments(cb)

        branches = cb.branches
        if len(branches) == 1:
            cond, body = branches[0]
            if cond is None:
                # Bare ``else`` with no condition is nonsensical for this pass.
                return hoisted
            return self._normalize_single_arm(sdfg, cb, cond, body) or hoisted

        if len(branches) == 2:
            (cond0, body0), (cond1, body1) = branches
            if cond0 is not None and cond1 is None:
                # Asymmetric arms (differing state counts, or not both single
                # substantive states) can't use the symmetric single-state path.
                # Serialize via ``serialize_two_arm``; later cycles normalize each.
                if self._arms_are_asymmetric(body0, body1):
                    return self.serialize_two_arm(cb, cond0, body0, body1) or hoisted
                # Disjoint two-arm: split into two single-arm conditionals; next
                # cycle handles each.
                return self._split_two_arm_disjoint(sdfg, cb, cond0, body0, body1) or hoisted

        return hoisted

    def _hoist_branch_invariant_assignments(self, cb: ConditionalBlock) -> bool:
        # Hoist branch-invariant interstate symbol bindings out of each arm.
        parent = cb.parent_graph
        in_edges = list(parent.in_edges(cb))
        if not in_edges:
            # cb is region entry: no in-edge to hoist onto. Leave as-is (rare).
            return False
        # Symbols produced inside any arm are NOT available before ``cb``.
        arm_assigned: set[str] = set()
        for _c, br in cb.branches:
            for e in br.edges():
                arm_assigned |= set(e.data.assignments.keys())
        # Branch-predicate symbols must keep their pre-``cb`` value.
        pred_syms: set[str] = set()
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
                if self._symbol_read_outside_arm(sym, br):
                    # ``sym`` is live after ``cb``: hoisting also sets it on the bypass path (``if a[i] < 0: j = i``
                    # read later by ``b[0] = j``). Refusing leaves it to ``_normalize_single_arm``.
                    hoistable = False
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
            # Drop the empty pass-through entry state so the single-state guard accepts the arm
            # (cloudsc-snippet-one ``__sym_z1 = z1``).
            successor = e.dst
            br.remove_edge(e)
            br.remove_node(sb)
            br.start_block = br.node_id(successor)
            hoisted = True
        # A binding with one value for the whole SDFG moves from ANY edge of an arm: CloudSC's fused
        # riming + melting map binds ``imelt_index = imelt[0]`` mid-arm under a per-column guard, in two
        # arms that each read it, and left there the tiler refuses the map.
        for _c, br in cb.branches:
            for e in list(br.edges()):
                constant = {
                    sym: expr
                    for sym, expr in e.data.assignments.items()
                    if sym not in pred_syms and self._sdfg_constant_binding(cb.sdfg, sym, str(expr)) and all(
                        str(ie.data.assignments.get(sym, expr)) == str(expr) for ie in in_edges)
                }
                if not constant:
                    continue
                for ie in in_edges:
                    ie.data.assignments.update(constant)
                for sym in constant:
                    del e.data.assignments[sym]
                hoisted = True
        return hoisted

    @staticmethod
    def _sdfg_constant_binding(sdfg: dace.SDFG, sym: str, expr: str) -> bool:
        # Whether ``sym = expr`` gives ``sym`` one value for the whole run of ``sdfg``.
        if sdfg.parent_nsdfg_node is not None and sym in sdfg.parent_nsdfg_node.symbol_mapping:
            return False
        bound = {r.loop_variable for r in sdfg.all_control_flow_regions() if isinstance(r, LoopRegion)}
        for edge in sdfg.all_interstate_edges():
            for lhs, rhs in edge.data.assignments.items():
                if lhs == sym and str(rhs) != expr:
                    return False
                bound.add(lhs)
        try:
            tree = ast.parse(expr, mode='eval')
        except SyntaxError:
            return False
        names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
        if names & bound:
            return False
        for sub in (n for n in ast.walk(tree) if isinstance(n, ast.Subscript)):
            if not (isinstance(sub.value, ast.Name) and sub.value.id in sdfg.arrays):
                return False
            index = sub.slice.elts if isinstance(sub.slice, ast.Tuple) else [sub.slice]
            shape = sdfg.arrays[sub.value.id].shape
            if len(index) != len(shape) or not all(
                    isinstance(i, ast.Constant) and isinstance(i.value, int) and not symbolic.issymbolic(d)
                    and 0 <= i.value < int(d) for i, d in zip(index, shape)):
                return False
        read = names & set(sdfg.arrays)
        return not any(n.data in read and state.in_degree(n) > 0 for state in sdfg.all_states()
                       for n in state.data_nodes())

    @staticmethod
    def _symbol_read_outside_arm(sym: str, arm_body: ControlFlowRegion) -> bool:
        # Whether ``sym`` is read anywhere outside ``arm_body``.
        sdfg = arm_body.sdfg
        only = {sym}
        inside_regions = set(arm_body.all_control_flow_regions(recursive=True))
        inside_states = set(arm_body.all_states())
        for cfg in sdfg.all_control_flow_regions(recursive=True):
            if cfg in inside_regions:
                continue
            for e in cfg.edges():
                for _lhs, rhs in (e.data.assignments or {}).items():
                    if symbolic.symbols_in_code(str(rhs), potential_symbols=only):
                        return True
                if e.data.condition is not None and symbolic.symbols_in_code(e.data.condition.as_string,
                                                                             potential_symbols=only):
                    return True
            if isinstance(cfg, ConditionalBlock):
                for c, _br in cfg.branches:
                    if c is not None and symbolic.symbols_in_code(c.as_string if isinstance(c, CodeBlock) else str(c),
                                                                  potential_symbols=only):
                        return True
            if isinstance(cfg, LoopRegion):
                for code in (cfg.loop_condition, cfg.update_statement, cfg.init_statement):
                    if code is not None and symbolic.symbols_in_code(
                            code.as_string if isinstance(code, CodeBlock) else str(code), potential_symbols=only):
                        return True
        for state in sdfg.all_states():
            if state in inside_states:
                continue
            for n in state.nodes():
                if isinstance(n, dace.nodes.Tasklet):
                    code = n.code.as_string if isinstance(n.code, CodeBlock) else str(n.code)
                    if symbolic.symbols_in_code(code, potential_symbols=only):
                        return True
        return False

    @staticmethod
    def _substantive_states(body: ControlFlowRegion) -> list[dace.SDFGState]:
        # SDFGStates in ``body`` that hold compute (non-empty).
        return [n for n in body.nodes() if isinstance(n, dace.SDFGState) and not n.is_empty()]

    def _arms_are_asymmetric(self, body0: ControlFlowRegion, body1: ControlFlowRegion) -> bool:
        # Whether the two arms cannot use the symmetric single-state path.
        if not (isinstance(body0, ControlFlowRegion) and isinstance(body1, ControlFlowRegion)):
            return False
        if len(body0.nodes()) != len(body1.nodes()):
            return True
        return len(self._substantive_states(body0)) != 1 or len(self._substantive_states(body1)) != 1

    @staticmethod
    def arm_written_arrays(cb: ConditionalBlock) -> set[str]:
        """Array names written anywhere inside any arm of ``cb``.

        :param cb: conditional block whose arms are scanned.
        :returns: set of written data names (transient and non-transient alike).
        """
        written: set[str] = set()
        for _cond, body in cb.branches:
            if not isinstance(body, ControlFlowRegion):
                continue
            for blk in body.all_control_flow_blocks():
                if isinstance(blk, dace.SDFGState):
                    written |= blk.read_and_write_sets()[1]
        return written

    @staticmethod
    def arm_assigned_symbols(cb: ConditionalBlock) -> OrderedSet[str]:
        """Symbols rebound anywhere inside an arm of ``cb``: interstate assignments and loop variables.

        :param cb: conditional block whose arms are scanned.
        :returns: the rebound symbol names.
        """
        assigned: OrderedSet[str] = OrderedSet()
        for _cond, body in cb.branches:
            for region in body.all_control_flow_regions():
                if isinstance(region, LoopRegion) and region.loop_variable:
                    assigned.add(region.loop_variable)
            for edge in body.all_interstate_edges():
                assigned.update(edge.data.assignments.keys())
        return assigned

    def representative_write_subset(self, cb: ConditionalBlock) -> str | None:
        """First element-write subset found in ``cb``'s arms, or ``None`` if there is none.

        Only used to SIZE a lifted per-lane transient (the guard's own reads carry their own
        captured subsets), so any one arm write is representative.

        :param cb: conditional block whose arms are scanned.
        :returns: printed subset string, or ``None``.
        """
        for _cond, body in cb.branches:
            if not isinstance(body, ControlFlowRegion):
                continue
            for blk in body.all_control_flow_blocks():
                if not isinstance(blk, dace.SDFGState):
                    continue
                write_subsets = self._collect_write_subsets(blk)
                if write_subsets:
                    return str(next(iter(write_subsets.values())))
        return None

    def freeze_guard_for_serialization(self, cb: ConditionalBlock, cond_text: str) -> str | None:
        """Guard expression that still holds when re-tested AFTER one arm has run.

        Serializing ``if c: A else: B`` into ``if c: A`` then ``if not c: B`` re-evaluates
        ``c`` once ``A`` has already executed. That is value-preserving only while no arm
        writes data ``c`` reads. TSVC s2710's ``if a[i] > b[i]: a[i] = a[i] + b[i]*d[i]``
        violates it: the second half re-reads the just-updated ``a``, so every lane whose
        update flipped the comparison takes BOTH arms and the else-arm stores land on
        if-arm lanes. Snapshot the guard instead -- evaluate it once into a per-lane bool
        transient in a state inserted before ``cb`` (dominating both halves, and written
        nowhere else), and hand both halves a read of that transient.

        :param cb: two-arm conditional about to be serialized.
        :param cond_text: the if-arm condition as written.
        :returns: guard text for both halves (``cond_text`` unchanged when no arm writes
            guard-read data), or ``None`` when the guard cannot be snapshotted and
            serializing would therefore be unsound.
        """
        lifter = SameWriteSetIfElseToITECFG()
        verdict = self.guard_snapshot_verdict(cb, cond_text, lifter)
        if verdict is None:
            return None
        return self.snapshot_guard(cb, cond_text, lifter) if verdict else cond_text

    def guard_snapshot_verdict(self, cb: ConditionalBlock, cond_text: str,
                               lifter: SameWriteSetIfElseToITECFG) -> bool | None:
        """Whether one guard of ``cb`` needs a snapshot before the arms run; mutates nothing.

        :param cb: conditional block about to be serialized.
        :param cond_text: one of its guards, as written.
        :param lifter: the instance whose lifts would snapshot the guard.
        :returns: ``False`` when no arm writes data the guard reads, ``True`` when one does and the
            guard can be snapshotted, ``None`` when one does and it cannot, or when an arm rebinds a
            symbol the guard reads (a snapshot would itself read the rebound symbol).
        """
        local_sdfg: dace.SDFG = cb.sdfg
        # The guard usually names interstate symbols staging element reads (``a_index = a[i]``);
        # expand them so the array dependence is visible. Read-only -- nothing is pruned here.
        expanded = lifter._inline_interstate_scalar_symbols(local_sdfg, cond_text, exclude=set())[0]
        guard_names = symbolic.symbols_in_code(cond_text) | symbolic.symbols_in_code(expanded)
        if not self.arm_assigned_symbols(cb).isdisjoint(guard_names):
            return None
        try:
            names = set(symbolic.arrays(expanded)) | set(symbolic.free_symbols_and_functions(expanded))
        except Exception:  # noqa: BLE001 -- unparsable guard: no provable dependence, leave as-is
            return False
        if not ({n for n in names if n in local_sdfg.arrays} & self.arm_written_arrays(cb)):
            return False
        # A gather guard (``w[idx[i]] > 0``) has no memlet form, so it cannot be snapshotted;
        # refuse the serialization rather than emit the re-read that miscompiles.
        if lifter._has_nested_subscript(local_sdfg, expanded) or self.representative_write_subset(cb) is None:
            return None
        return True

    def snapshot_guard(self, cb: ConditionalBlock, cond_text: str, lifter: SameWriteSetIfElseToITECFG) -> str:
        """Evaluate a guard once into a per-lane bool transient, in a new state before ``cb``.

        :param cb: conditional block about to be serialized.
        :param cond_text: the guard, as written.
        :param lifter: the instance that lifts the guard (and may defer its symbol deletions).
        :returns: a read of the snapshot, to test instead of ``cond_text``.
        :raises NotImplementedError: the guard has no array form to snapshot into.
        """
        local_sdfg: dace.SDFG = cb.sdfg
        subset_str = self.representative_write_subset(cb)
        parent = cb.parent_graph
        guard_state = parent.add_state_before(cb, label=f"{cb.label}_guard", is_start_block=parent.start_block is cb)
        resolved = lifter._resolve_cond_to_array(local_sdfg, guard_state, cond_text, subset_str, skip_cb=cb)
        if resolved is None:
            raise NotImplementedError(f"BranchNormalization: cannot snapshot the guard of {cb.label!r} "
                                      f"({cond_text!r}) although its arms write data it reads; "
                                      f"serializing the arms would re-test a mutated guard")
        parent.reset_cfg_list()
        cond_name = resolved[0]
        snapshot_subset = "0" if local_sdfg.arrays[cond_name].total_size == 1 else subset_str
        return f"{cond_name}[{snapshot_subset}]"

    def freeze_guards(self, cb: ConditionalBlock, lifter: SameWriteSetIfElseToITECFG) -> list[str] | None:
        """Guard texts for running ``cb``'s arms as a chain of single-arm blocks, one per conditioned arm.

        Arm ``k`` of the chain re-tests guards ``0..k``, so every guard whose data an arm writes is
        snapshotted before ``cb``. The symbol definitions a snapshot consumes may still feed another guard:
        ``lifter`` defers their deletion until :meth:`release_guard_symbols`, called once the chain has
        replaced ``cb``.

        :param cb: the multi-arm conditional block.
        :param lifter: snapshots the guards and holds the deferred deletions.
        :returns: one guard text per conditioned arm, or ``None`` with nothing mutated when a guard cannot be
            snapshotted, or when a later guard (evaluated only after an earlier one failed) would be
            evaluated unconditionally beside a guard that keeps an iteration symbol in range.
        """
        texts = [
            cond.as_string if isinstance(cond, CodeBlock) else str(cond) for cond, body in cb.branches
            if cond is not None
        ]
        verdicts = [self.guard_snapshot_verdict(cb, text, lifter) for text in texts]
        if None in verdicts or (any(verdicts[1:]) and condition_guards_iteration_symbol(cb)):
            return None
        lifter._deferred_drops = []
        return [self.snapshot_guard(cb, text, lifter) if verdict else text for text, verdict in zip(texts, verdicts)]

    @staticmethod
    def release_guard_symbols(lifter: SameWriteSetIfElseToITECFG) -> None:
        """Apply the symbol deletions :meth:`freeze_guards` deferred, each re-checked against the current graph."""
        drops = lifter._deferred_drops
        lifter._deferred_drops = None
        for drop in drops:
            lifter._drop_interstate_symbol(drop[0], drop[1], drop[2])

    @staticmethod
    def is_multi_arm(cb: ConditionalBlock) -> bool:
        """``>=3`` branches, or ``if/elif`` with no ``else``: the shapes the one- and two-arm rewrites refuse."""
        branches = cb.branches
        if len(branches) >= 3:
            return True
        return len(branches) == 2 and branches[0][0] is not None and branches[1][0] is not None

    def flatten_multi_arm_blocks(self, sdfg: dace.SDFG) -> int:
        """Flatten every multi-arm ``ConditionalBlock`` into single-arm blocks; returns the count."""
        return rewrite_blocks_to_fixpoint(sdfg, lambda cb: self.is_multi_arm(cb) and self.flatten_multi_arm_block(cb))

    def flatten_multi_arm_block(self, cb: ConditionalBlock) -> bool:
        """Replace ``cb`` with single-arm blocks guarded by ``not c0 and ... and ck``, snapshotting guards an arm
        writes; ``False`` with nothing changed when a guard cannot be snapshotted."""
        lifter = SameWriteSetIfElseToITECFG()
        guards = self.freeze_guards(cb, lifter)
        if guards is None:
            return False
        parent = cb.parent_graph
        arms: list[tuple[CodeBlock | None, ControlFlowRegion]] = list(cb.branches)
        for arm in arms:
            cb.remove_branch(arm[1])

        prior_negations: list[str] = []
        chain: list[ConditionalBlock] = []
        guard_texts = iter(guards)
        for index, (cond, body) in enumerate(arms):
            guard = None if cond is None else next(guard_texts)
            terms = list(prior_negations)
            if guard is not None:
                terms.append(f"({guard})")
            link = ConditionalBlock(label=f"{cb.label}_flat{index}", sdfg=parent.sdfg, parent=parent)
            link.add_branch(CodeBlock(" and ".join(terms) if terms else "True"), body)
            parent.add_node(link)
            chain.append(link)
            if guard is not None:
                prior_negations.append(f"(not ({guard}))")
        self.splice_chain(parent, cb, chain)
        self.release_guard_symbols(lifter)
        return True

    @staticmethod
    def splice_chain(parent: ControlFlowRegion, cb: ConditionalBlock, chain: list[ConditionalBlock]) -> None:
        """Put ``chain`` in place of ``cb``: ``cb``'s in-edges enter the first block, its out-edges leave the last."""
        for edge in list(parent.in_edges(cb)):
            parent.add_edge(edge.src, chain[0], edge.data)
        for first, second in zip(chain, chain[1:]):
            parent.add_edge(first, second, dace.InterstateEdge())
        for edge in list(parent.out_edges(cb)):
            parent.add_edge(chain[-1], edge.dst, edge.data)
        parent.remove_node(cb)
        parent.reset_cfg_list()

    def serialize_two_arm(self, cb: ConditionalBlock, cond0: CodeBlock, body0: ControlFlowRegion,
                          body1: ControlFlowRegion) -> bool:
        """Serialize ``if c: A else: B`` into ``if c: A`` then ``if not c: B``.

        Mostly a CFG rewrite: ``cb`` keeps the if-arm; a new negated single-arm
        block holds the else-arm, stitched sequentially after ``cb``. Later
        cycles normalize each single-arm form. Exactly one arm's writes take
        effect -- identical to the original if/else -- PROVIDED the guard still
        reads what it read before ``cb``, which
        :meth:`freeze_guard_for_serialization` guarantees.

        :param cb: two-arm conditional (becomes the if-arm only).
        :param cond0: if condition.
        :param body0: if-arm body (kept on ``cb``).
        :param body1: else-arm body (moved to the negated block).
        :returns: ``True`` if serialized, ``False`` if the guard is not snapshottable.
        """
        parent = cb.parent_graph
        cond_text = cond0.as_string if isinstance(cond0, CodeBlock) else str(cond0)
        frozen = self.freeze_guard_for_serialization(cb, cond_text)
        if frozen is None:
            return False
        cb.remove_branch(body1)
        if frozen != cond_text:
            # The snapshot consumed the guard's interstate symbols, so cb's own condition
            # must move to the transient too or it would name symbols that no longer exist.
            cb.remove_branch(body0)
            cb.add_branch(CodeBlock(frozen), body0)
        neg_block = ConditionalBlock(label=f"{cb.label}_negated", sdfg=parent.sdfg, parent=parent)
        neg_block.add_branch(CodeBlock(f"not ({frozen})"), body1)
        parent.add_node(neg_block, ensure_unique_name=True)
        out_edges = list(parent.out_edges(cb))
        for oe in out_edges:
            parent.remove_edge(oe)
            parent.add_edge(neg_block, oe.dst, copy.deepcopy(oe.data))
        parent.add_edge(cb, neg_block, dace.InterstateEdge())
        parent.reset_cfg_list()
        return True

    def _normalize_single_arm(self, sdfg: dace.SDFG, cb: ConditionalBlock, cond: CodeBlock,
                              body: ControlFlowRegion) -> bool:
        # Lower ``if cond: body`` to ``arr = ITE(cond, expr, arr)`` writes. An index guard (``if i < N - 1``)
        # keeps the arm's own accesses in range, so leave it for the masking path (shared detector).
        if condition_guards_iteration_symbol(cb) and not arm_accesses_are_in_range_unguarded(cb):
            return False
        # The arm may be a linear chain of empty states + one compute state; lift it whole and gate only
        # escaping writes. An interstate assignment is a conditional binding: unproven arm-locality refuses.
        states = [n for n in body.nodes() if isinstance(n, dace.SDFGState)]
        if len(states) != len(body.nodes()):
            return False
        local_sdfg_for_arm: dace.SDFG = cb.sdfg
        for e in body.edges():
            if not e.data.assignments:
                continue
            # Each assigned symbol must be arm-local (all reads inside body);
            # else the lift breaks downstream consumers.
            non_local = []
            for sym, rhs in e.data.assignments.items():
                # A self-referential binding (``k = k + 1``) crosses the loop back edge even when every read is
                # in-arm (TSVC s343 compaction counter); lifting it would make it a dense counter.
                if sym in symbolic.symbols_in_code(str(rhs)):
                    non_local.append(sym)
                elif not self._symbol_is_arm_local(local_sdfg_for_arm, body, sym):
                    non_local.append(sym)
            if non_local:
                # The arm binds a symbol read outside it (TSVC s123 counter, s331 find-last, s318 argmax). Such a
                # loop is sequential and never tiled, so leave the control flow scalar instead of raising.
                return False
        substantive = [s for s in states if not s.is_empty()]
        if not substantive:
            return False
        for s in substantive:
            for n in s.nodes():
                if not isinstance(n, (dace.nodes.AccessNode, dace.nodes.Tasklet)):
                    return False

        # A chain of substantive states (cloudsc ``ptend_q += a; ...; ptend_q += b`` that state fusion
        # refuses) takes a per-state ITE, which is value-preserving. Straight-line arms only.
        ordered = self._linear_state_order(body)
        if ordered is None:
            return False
        ordered_subst = [s for s in ordered if s in substantive]
        if len(ordered_subst) != len(substantive):
            return False

        cond_text = cond.as_string if isinstance(cond, CodeBlock) else str(cond)
        local_sdfg: dace.SDFG = cb.sdfg
        escaping = compute_arm_escape_writes(local_sdfg, cb).get(0, set())

        # Per-state escaping-write subsets in execution order. Only escaping
        # writes get the ITE gate; arm-internal scratch stays inline.
        per_state = []
        for s in ordered_subst:
            ws = self._collect_write_subsets(s)
            if ws is None:
                return False
            per_state.append((s, {arr: sub for arr, sub in ws.items() if arr in escaping}))

        # Resolve cond ONCE, on the first substantive state with an escaping
        # write (its producer precedes every consumer). Non-producer states
        # read the cond array fresh.
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
        # Whether ``sym``'s reads are confined to ``body`` (the arm region).
        arm_blocks = set(body.all_control_flow_blocks(
            recursive=True)) if isinstance(body, ControlFlowRegion) else set(body.nodes())
        # Any reference to sym outside the arm disqualifies the lift.
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
    def _linear_state_order(body: ControlFlowRegion) -> list[ControlFlowBlock] | None:
        # Execution-order block list iff ``body`` is a straight-line chain.
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
        # Split a disjoint-write ``if/else`` into two sequential single-arm ``if`` s.
        # Refuse to even split an index-guarded ``if/else`` (see _normalize_single_arm): the
        # negated-else halves this produces would be flattened just the same, fabricating the
        # out-of-range read. Leave it whole for the masking path.
        if condition_guards_iteration_symbol(cb) and not arm_accesses_are_in_range_unguarded(cb):
            return False
        if len(body0.nodes()) != 1 or len(body1.nodes()) != 1:
            return False
        s0, s1 = body0.nodes()[0], body1.nodes()[0]
        if not (isinstance(s0, dace.SDFGState) and isinstance(s1, dace.SDFGState)):
            return False
        w0 = self._collect_write_subsets(s0)
        w1 = self._collect_write_subsets(s1)
        if w0 is None or w1 is None:
            return False
        # Same-array writes conflict only when element subsets intersect.
        # Element-disjoint (cloudsc ``zsolqa[i,a]`` vs ``zsolqa[i,b]``) split
        # cleanly; each arm gates its own subset next cycle. intersects()
        # returns True/False/None; treat None as a conservative conflict.
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

        # Split into two single-arm conditionals: else-body -> new ``if not cond0: body1``
        # block after ``cb`` (now if-arm only); later cycles rewrite each single-arm form.
        # Same serialization the asymmetric-arm path uses, guard snapshot included.
        return self.serialize_two_arm(cb, cond0, body0, body1)

    def _collect_write_subsets(self, state: dace.SDFGState) -> dict[str, subsets.Range] | None:
        from dace.transformation.passes.vectorization.utils.queries import collect_element_write_subsets
        return collect_element_write_subsets(state)

    def _resolve_arm_cond(self,
                          sdfg: dace.SDFG,
                          state: dace.SDFGState,
                          cond_text: str,
                          any_subset_str: str,
                          skip_cb: ConditionalBlock | None = None) -> tuple[str | None, dace.nodes.AccessNode | None]:
        # Resolve the arm condition to ``(cond_array_name, cond_producer)``.
        from dace.transformation.passes.vectorization.same_write_set_if_else_to_ite_cfg import (
            SameWriteSetIfElseToITECFG, )  # local import: avoids an import cycle at module load
        resolved = SameWriteSetIfElseToITECFG()._resolve_cond_to_array(sdfg,
                                                                       state,
                                                                       cond_text,
                                                                       any_subset_str,
                                                                       skip_cb=skip_cb)
        return (None, None) if resolved is None else resolved

    def _rewrite_writes_to_ite(self,
                               sdfg: dace.SDFG,
                               state: dace.SDFGState,
                               write_subsets: dict[str, subsets.Range],
                               cond_text: str,
                               *,
                               skip_cb: ConditionalBlock | None = None,
                               preresolved: tuple[str | None, dace.nodes.AccessNode | None] | None = None) -> None:
        # Redirect each write in ``state`` through ``arr = ITE(cond, expr, arr)``. Resolve cond once: the
        # symbol lift deletes the upstream assignment. ``preresolved`` lets a multi-state caller share it;
        # ``producer=None`` forces a fresh in-state read of the cond array.
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
            # Every write AN for this array. Each may target a different subset
            # (cloudsc chained ``arr[0,3,it]`` then ``arr[3,0,it]``); write_subsets
            # holds one entry per array name, so read each write's actual subset
            # from its own in-edge memlet.
            writes = [n for n in state.nodes() if isinstance(n, dace.nodes.AccessNode) and n.data == arr_name]
            for write_an in writes:
                # An empty in-edge only sequences a second read; it is not a write.
                in_edges = [e for e in state.in_edges(write_an) if not e.data.is_empty()]
                if not in_edges:
                    continue
                # One AN can carry several element writes (cloudsc ``zsolqa``); each in-edge gets its own gate.
                for in_edge in in_edges:
                    self._gate_one_write(sdfg, state, arr_name, write_an, in_edge, cond_text, cond_array_name,
                                         cond_producer)

    def _gate_one_write(self, sdfg: dace.SDFG, state: dace.SDFGState, arr_name: str, write_an: dace.nodes.AccessNode,
                        in_edge: MultiConnectorEdge[Memlet], cond_text: str, cond_array_name: str | None,
                        cond_producer: dace.nodes.AccessNode | None) -> None:
        # Redirect ONE write edge through ``arr = ITE(cond, expr, arr)``.
        write_subset = in_edge.data.subset

        # 1-element scratch ``__bn_<arr>_new`` holds this element's value.
        tmp_name, _ = sdfg.add_array(name=f"__bn_{arr_name}_new",
                                     shape=(1, ),
                                     dtype=sdfg.arrays[arr_name].dtype,
                                     storage=dace.dtypes.StorageType.Register,
                                     transient=True,
                                     find_new_name=True)
        tmp_an = state.add_access(tmp_name)

        # ITE old value = ``arr_name`` before the writing tasklet ran: for chained RMW the AN the tasklet
        # read at the same subset, else a fresh pre-state AN. Locate it before redirecting the out-edge.
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
            # Reuse the producing AN (see ``_resolve_cond_to_array``): a
            # fresh read node disconnects the lift and lets codegen emit
            # the ITE before the cond is computed.
            cond_access = cond_producer if cond_producer is not None else state.add_access(cond_array_name)
            ite_t = state.add_tasklet(
                name=f"bn_ite_{arr_name}",
                inputs=OrderedSet(('_c', '_new', '_old')),
                outputs={"_o"},
                code="_o = ITE(_c, _new, _old)",
            )
            cond_subset = "0" if sdfg.arrays[cond_array_name].total_size == 1 else write_subset
            state.add_edge(cond_access, None, ite_t, "_c", dace.Memlet(expr=f"{cond_array_name}[{cond_subset}]"))
        else:
            ite_t = state.add_tasklet(
                name=f"bn_ite_{arr_name}",
                inputs=OrderedSet(('_new', '_old')),
                outputs={"_o"},
                code=f"_o = ITE({cond_text}, _new, _old)",
            )
        state.add_edge(tmp_an, None, ite_t, "_new", dace.Memlet(expr=f"{tmp_name}[0]"))
        state.add_edge(old_an, None, ite_t, "_old", dace.Memlet(expr=f"{arr_name}[{write_subset}]"))
        state.add_edge(ite_t, "_o", write_an, None, dace.Memlet(expr=f"{arr_name}[{write_subset}]"))
