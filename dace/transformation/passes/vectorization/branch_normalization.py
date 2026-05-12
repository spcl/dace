# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
``BranchNormalization`` flattens every remaining ``ConditionalBlock`` in
the SDFG into a linear sequence of plain tasklets, so the vectorizer can
treat them as masked or unconditional SIMD ops.

It runs after ``SameWriteSetIfElseToMergeCFG`` (M3.1b), which has already
handled the case where both arms of an ``if/else`` write to identical
locations. What remains here:

- **Single-arm conditional** (``if cond: body``, no else): the body is
  lifted into a sibling state and every write ``arr[subset] = expr`` is
  rewritten as ``arr[subset] = merge(cond_mask, expr, arr[subset])``.
- **Two-arm with disjoint write sets**: split into two sequential
  single-arm conditionals via ``BranchElimination._split_branches``,
  then normalize each.
- **Two-arm with overlapping (but not identical) write sets**: raise
  ``NotImplementedError``. M3.1b is supposed to catch identical-write
  sets, so anything reaching this case is genuinely unsupported by the
  current normalization model.

After the pass, no ``ConditionalBlock`` remains in any region that was
processed, and ``assert_connector_role_matches_edges`` passes on every
emitted state.
"""
import copy
import re
from typing import Dict, Optional, Set

import dace
from dace import properties
from dace.properties import CodeBlock
from dace.sdfg.construction_utils import (
    assert_connector_role_matches_edges,
    move_branch_cfg_up_discard_conditions,
)
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion
from dace.transformation import pass_pipeline as ppl


# Same token splitter shape used by ``construction_utils._tokens`` for
# the equivalent walk over interstate-edge expressions. Kept module-local
# so the analysis is self-contained.
_INTERSTATE_TOKEN_RE = re.compile(r"[()\[\]\s,+\-*/%<>!=&|^~?:]+")


def _tokenize_interstate_expr(text: str) -> Set[str]:
    return {s.strip() for s in _INTERSTATE_TOKEN_RE.split(text) if s.strip()}


def compute_arm_escape_writes(sdfg: dace.SDFG, cb: ConditionalBlock) -> Dict[int, Set[str]]:
    """Returns the per-arm set of array names whose write must be rerouted
    to a private arm-local transient before the arm bodies can run
    unconditionally.

    A write to ``arr`` inside arm ``i`` escapes (so it needs rerouting)
    iff any of:

    1. ``arr`` is non-transient (visible to callers of the SDFG that owns
       ``cb``).
    2. ``arr`` is read somewhere outside the subtree rooted at ``cb``,
       computed by skipping every state inside ``cb`` while walking
       ``state.read_and_write_sets()`` over the rest of the SDFG, plus
       tokenizing interstate-edge conditions and assignment RHSs whose
       endpoints lie outside ``cb``.
    3. ``arr`` is read by *another* arm of ``cb``, since every arm runs
       unconditionally post-rewrite and one arm's read would otherwise
       see another arm's just-written value.

    Returns ``{arm_index: {arr_name, ...}}``. Arms with no escaping
    writes get an empty set entry. The redirect plan that drives the
    actual rewrite is one private transient per ``(arm_index, arr)``
    pair.
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
            assigns = getattr(e.data, "assignments", None) or {}
            for v in assigns.values():
                outside_reads |= (_tokenize_interstate_expr(str(v)) & array_names)
            cond = e.data.condition.as_string if e.data.condition is not None else ""
            if cond:
                outside_reads |= (_tokenize_interstate_expr(cond) & array_names)

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
            outside_reads |= (_tokenize_interstate_expr(text) & array_names)

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
            non_transient = not getattr(desc, "transient", False)
            if non_transient or arr in outside_reads or arr in other_arms_reads:
                escaping.add(arr)
        result[i] = escaping
    return result


@properties.make_properties
class BranchNormalization(ppl.Pass):
    """Flatten residual ``ConditionalBlock``s into ``merge``-tasklet form.

    See module docstring for the contract.
    """

    CATEGORY: str = "Vectorization Preparation"

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.CFG | ppl.Modifies.States | ppl.Modifies.AccessNodes

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def apply_pass(self, sdfg: dace.SDFG, _) -> Optional[int]:
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
                # Two-arm if/else: split disjoint writes into two single-arm
                # conditionals first, then let the next pass cycle handle each.
                return self._split_two_arm_disjoint(sdfg, cb, cond0, body0, body1)

        return False

    def _normalize_single_arm(self, sdfg: dace.SDFG, cb: ConditionalBlock, cond: CodeBlock,
                              body: ControlFlowRegion) -> bool:
        if len(body.nodes()) != 1 or not isinstance(body.nodes()[0], dace.SDFGState):
            return False
        body_state: dace.SDFGState = body.nodes()[0]
        for n in body_state.nodes():
            if not isinstance(n, (dace.nodes.AccessNode, dace.nodes.Tasklet)):
                return False

        write_subsets = self._collect_write_subsets(body_state)
        if write_subsets is None:
            return False

        # Only writes flagged by the escape analysis get the merge rewrite;
        # arm-internal scratch stays inline because nothing outside sees it.
        cond_text = cond.as_string if isinstance(cond, CodeBlock) else str(cond)
        local_sdfg: dace.SDFG = cb.sdfg
        escaping = compute_arm_escape_writes(local_sdfg, cb).get(0, set())
        merge_subsets = {arr: sub for arr, sub in write_subsets.items() if arr in escaping}
        if merge_subsets:
            self._rewrite_writes_to_merge(local_sdfg, body_state, merge_subsets, cond_text)

        move_branch_cfg_up_discard_conditions(if_block=cb, body_to_take=body)
        return True

    def _split_two_arm_disjoint(self, sdfg: dace.SDFG, cb: ConditionalBlock, cond0: CodeBlock,
                                body0: ControlFlowRegion, body1: ControlFlowRegion) -> bool:
        if len(body0.nodes()) != 1 or len(body1.nodes()) != 1:
            return False
        s0, s1 = body0.nodes()[0], body1.nodes()[0]
        if not (isinstance(s0, dace.SDFGState) and isinstance(s1, dace.SDFGState)):
            return False
        w0 = self._collect_write_subsets(s0)
        w1 = self._collect_write_subsets(s1)
        if w0 is None or w1 is None:
            return False
        shared = set(w0) & set(w1)
        if shared:
            # Same-write-set case is M3.1b's job; if it reached here, M3.1b
            # didn't match (different subsets on the same array, etc.).
            raise NotImplementedError(
                f"BranchNormalization: two-arm ConditionalBlock {cb.label!r} has overlapping "
                f"write sets {sorted(shared)} that M3.1b did not normalize; this pass cannot "
                f"flatten it without dropping or duplicating writes")

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
        """Return ``{arr_name: subset}`` for every element-write in ``state``.
        Returns ``None`` if any write violates the element-wise restriction."""
        out = {}
        for n in state.nodes():
            if not isinstance(n, dace.nodes.AccessNode):
                continue
            for e in state.in_edges(n):
                if e.data.data is None:
                    continue
                try:
                    if e.data.subset.num_elements_exact() != 1:
                        return None
                except Exception:
                    return None
                out[n.data] = e.data.subset
        return out

    def _rewrite_writes_to_merge(self, sdfg: dace.SDFG, state: dace.SDFGState, write_subsets: dict, cond_text: str):
        """For each access-node write in ``state``, redirect through a merge
        tasklet so the post-condition write becomes
        ``arr = merge(cond, expr, arr)``."""
        for arr_name, subset in list(write_subsets.items()):
            # Find the unique write access node for this array in this state.
            writes = [n for n in state.nodes() if isinstance(n, dace.nodes.AccessNode) and n.data == arr_name]
            for write_an in writes:
                in_edges = list(state.in_edges(write_an))
                if not in_edges:
                    continue
                # We expect exactly one incoming edge for element writes.
                if len(in_edges) != 1:
                    raise NotImplementedError(
                        f"BranchNormalization: write to {arr_name!r} has {len(in_edges)} in-edges; "
                        f"only single-edge writes are supported in this slice")
                in_edge = in_edges[0]

                # Build a transient ``__bn_<arr>_new`` to hold the computed value.
                tmp_name, _ = sdfg.add_array(name=f"__bn_{arr_name}_new",
                                             shape=sdfg.arrays[arr_name].shape,
                                             dtype=sdfg.arrays[arr_name].dtype,
                                             storage=dace.dtypes.StorageType.Register,
                                             transient=True,
                                             find_new_name=True)
                tmp_an = state.add_access(tmp_name)

                # Redirect the existing in-edge to write to the temp instead.
                state.remove_edge(in_edge)
                state.add_edge(in_edge.src, in_edge.src_conn, tmp_an, None,
                               dace.Memlet(expr=f"{tmp_name}[{subset}]"))

                # Read the old value of ``arr`` so we can pass it as the
                # else-branch of the merge. Reuse M3.1b's cond resolver so
                # the same three-stage recipe (direct array / single
                # interstate symbol / compound expression over multiple
                # symbols) drives the ``_c`` connector wiring here too.
                from dace.transformation.passes.vectorization.same_write_set_if_else_to_merge_cfg import (
                    SameWriteSetIfElseToMergeCFG, )  # noqa: avoid import cycle at module load
                old_an = state.add_access(arr_name)
                cond_resolved = SameWriteSetIfElseToMergeCFG()._resolve_cond_to_array(
                    sdfg, state, cond_text, str(subset))
                if cond_resolved is not None:
                    cond_array_name, cond_access = cond_resolved
                    merge_t = state.add_tasklet(
                        name=f"bn_merge_{arr_name}",
                        inputs={"_c", "_new", "_old"},
                        outputs={"_o"},
                        code="_o = merge(_c, _new, _old)",
                    )
                    cond_subset = "0" if sdfg.arrays[cond_array_name].total_size == 1 else subset
                    state.add_edge(cond_access, None, merge_t, "_c",
                                   dace.Memlet(expr=f"{cond_array_name}[{cond_subset}]"))
                else:
                    merge_t = state.add_tasklet(
                        name=f"bn_merge_{arr_name}",
                        inputs={"_new", "_old"},
                        outputs={"_o"},
                        code=f"_o = merge({cond_text}, _new, _old)",
                    )
                state.add_edge(tmp_an, None, merge_t, "_new", dace.Memlet(expr=f"{tmp_name}[{subset}]"))
                state.add_edge(old_an, None, merge_t, "_old", dace.Memlet(expr=f"{arr_name}[{subset}]"))
                state.add_edge(merge_t, "_o", write_an, None, dace.Memlet(expr=f"{arr_name}[{subset}]"))
