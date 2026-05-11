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
from typing import Optional

import dace
from dace import properties
from dace.properties import CodeBlock
from dace.sdfg.construction_utils import (
    assert_connector_role_matches_edges,
    move_branch_cfg_up_discard_conditions,
)
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion
from dace.transformation import pass_pipeline as ppl


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

        # Lift the body up, replacing the conditional with the body's contents.
        # The condition is dropped at this point, every write that used to be
        # guarded by it is rewritten in-place to ``merge(cond, new, old)`` so
        # the gating moves from control flow into dataflow.
        # Arrays live on the local SDFG, not the outermost one passed to
        # ``apply_pass``, when the ConditionalBlock is inside a NestedSDFG.
        cond_text = cond.as_string if isinstance(cond, CodeBlock) else str(cond)
        local_sdfg: dace.SDFG = cb.sdfg
        self._rewrite_writes_to_merge(local_sdfg, body_state, write_subsets, cond_text)

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
                # else-branch of the merge. Wire cond as a 3rd in-connector
                # when it names an array in the SDFG so the vectorizer can
                # consume it as a per-lane vector, fall back to a free
                # symbol reference for symbol-based conditions.
                old_an = state.add_access(arr_name)
                cond_is_array = cond_text in sdfg.arrays
                if cond_is_array:
                    merge_t = state.add_tasklet(
                        name=f"bn_merge_{arr_name}",
                        inputs={"_c", "_new", "_old"},
                        outputs={"_o"},
                        code="_o = merge(_c, _new, _old)",
                    )
                    cond_an = state.add_access(cond_text)
                    cond_subset = "0" if sdfg.arrays[cond_text].total_size == 1 else subset
                    state.add_edge(cond_an, None, merge_t, "_c", dace.Memlet(expr=f"{cond_text}[{cond_subset}]"))
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
