# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
``SameWriteSetIfElseToMergeCFG`` — rewrites a same-write-set ``if/else`` into
three sequential CFGs (``compute_then``, ``compute_else``, ``apply_merge``),
where ``apply_merge`` emits one ``merge(cond, _then_<arr>, _else_<arr>)``
tasklet per shared write target.

After this pass runs, the original ``ConditionalBlock`` is gone for any pair
of arms that write to *identical* element subsets; the two arms become
sequentially-executed CFGs that produce temporaries, and a third state folds
those temporaries via the symbolic ``merge`` (see
:mod:`dace.runtime.include.dace.merge`) which the vectorizer later lowers
to a SIMD blend.

Restrictions in this slice (raise :class:`NotImplementedError` otherwise):
- the ``ConditionalBlock`` has exactly two branches, the first with a
  condition and the second with ``None`` (the ``else``);
- each branch is a :class:`ControlFlowRegion` containing exactly one state;
- writes are element subsets (``num_elements_exact() == 1``) with the
  *same* subset in both arms for every shared write target;
- arm bodies contain only ``Tasklet`` and ``AccessNode`` nodes (no
  nested SDFGs, no maps inside the arm; tile-level prep happens elsewhere).
"""
from typing import Optional

import dace
from dace import properties
from dace.properties import CodeBlock
from dace.sdfg.construction_utils import (
    assert_connector_role_matches_edges,
    copy_state_contents,
)
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion
from dace.transformation import pass_pipeline as ppl


@properties.make_properties
class SameWriteSetIfElseToMergeCFG(ppl.Pass):
    """Rewrite same-write-set ``if/else`` blocks into 3-CFG merge form.

    See module docstring for the pass contract.
    """

    CATEGORY: str = "Vectorization Preparation"

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.CFG | ppl.Modifies.States | ppl.Modifies.AccessNodes

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def apply_pass(self, sdfg: dace.SDFG, _) -> Optional[int]:
        rewritten = 0
        for cfg in list(sdfg.all_control_flow_regions(recursive=True)):
            for block in list(cfg.nodes()):
                if not isinstance(block, ConditionalBlock):
                    continue
                if not self._matches(block):
                    continue
                self._rewrite(sdfg, block)
                rewritten += 1
        return rewritten or None

    def _matches(self, cb: ConditionalBlock) -> bool:
        if len(cb.branches) != 2:
            return False
        (cond0, body0), (cond1, body1) = cb.branches
        if cond0 is None or cond1 is not None:
            return False
        if not (isinstance(body0, ControlFlowRegion) and isinstance(body1, ControlFlowRegion)):
            return False
        # Each branch must hold exactly one SDFGState.
        if len(body0.nodes()) != 1 or len(body1.nodes()) != 1:
            return False
        s0, s1 = body0.nodes()[0], body1.nodes()[0]
        if not (isinstance(s0, dace.SDFGState) and isinstance(s1, dace.SDFGState)):
            return False
        # No maps / nested SDFGs / non-tasklet compute in the arms.
        for s in (s0, s1):
            for n in s.nodes():
                if isinstance(n, (dace.nodes.AccessNode, dace.nodes.Tasklet)):
                    continue
                return False
        # Identical write sets, identical subsets.
        return self._writes_match(s0, s1)

    def _collect_write_subsets(self, state: dace.SDFGState):
        """Return ``{arr_name: subset}`` for every write. Each write must be a
        single element (matches the restriction documented in the module
        docstring)."""
        out = {}
        for n in state.nodes():
            if not isinstance(n, dace.nodes.AccessNode):
                continue
            for e in state.in_edges(n):
                if e.data.data is None:
                    continue
                if e.data.subset.num_elements_exact() != 1:
                    raise NotImplementedError(
                        f"SameWriteSetIfElseToMergeCFG: non-element write subset {e.data.subset} on {n.data}")
                out[n.data] = e.data.subset
        return out

    def _writes_match(self, s0: dace.SDFGState, s1: dace.SDFGState) -> bool:
        try:
            w0 = self._collect_write_subsets(s0)
            w1 = self._collect_write_subsets(s1)
        except NotImplementedError:
            return False
        if set(w0.keys()) != set(w1.keys()):
            return False
        for k in w0:
            if str(w0[k]) != str(w1[k]):
                return False
        return True

    def _rewrite(self, sdfg: dace.SDFG, cb: ConditionalBlock):
        parent = cb.parent_graph
        (cond_block, then_body), (_, else_body) = cb.branches
        then_state: dace.SDFGState = then_body.nodes()[0]
        else_state: dace.SDFGState = else_body.nodes()[0]
        write_subsets = self._collect_write_subsets(then_state)

        # Allocate per-arm temporaries (one per shared write target), sized
        # to the parent array. Element-write semantics make this safe.
        temp_then = {}
        temp_else = {}
        for arr_name in write_subsets:
            base = sdfg.arrays[arr_name]
            n_then, _ = sdfg.add_array(name=f"_then_{arr_name}",
                                       shape=base.shape,
                                       dtype=base.dtype,
                                       storage=dace.dtypes.StorageType.Register,
                                       transient=True,
                                       find_new_name=True)
            n_else, _ = sdfg.add_array(name=f"_else_{arr_name}",
                                       shape=base.shape,
                                       dtype=base.dtype,
                                       storage=dace.dtypes.StorageType.Register,
                                       transient=True,
                                       find_new_name=True)
            temp_then[arr_name] = n_then
            temp_else[arr_name] = n_else

        # New 3-CFG states in the parent graph.
        ct_state = parent.add_state(f"compute_then_{cb.label}")
        ce_state = parent.add_state(f"compute_else_{cb.label}")
        am_state = parent.add_state(f"apply_merge_{cb.label}")

        # Clone bodies and redirect writes.
        self._clone_with_redirect(then_state, ct_state, temp_then)
        self._clone_with_redirect(else_state, ce_state, temp_else)

        # Emit per-write merge tasklets in apply_merge state.
        cond_text = cond_block.as_string if isinstance(cond_block, CodeBlock) else str(cond_block)
        for arr_name, subset in write_subsets.items():
            self._emit_merge_tasklet(sdfg, am_state, arr_name, subset, temp_then[arr_name], temp_else[arr_name],
                                     cond_text)

        # Stitch in/out edges of the ConditionalBlock onto ct_state -> ce_state
        # -> am_state, then drop the original block.
        in_edges = list(parent.in_edges(cb))
        out_edges = list(parent.out_edges(cb))
        was_start = (parent.start_block is cb)
        for e in in_edges + out_edges:
            parent.remove_edge(e)
        parent.remove_node(cb)
        for e in in_edges:
            parent.add_edge(e.src, ct_state, e.data)
        parent.add_edge(ct_state, ce_state, dace.InterstateEdge())
        parent.add_edge(ce_state, am_state, dace.InterstateEdge())
        for e in out_edges:
            parent.add_edge(am_state, e.dst, e.data)
        if was_start:
            parent.start_block = parent.node_id(ct_state)

        # End-of-pass invariant: every emitted state has well-formed connectors.
        for s in (ct_state, ce_state, am_state):
            assert_connector_role_matches_edges(s)

    def _clone_with_redirect(self, src: dace.SDFGState, dst: dace.SDFGState, rename: dict):
        """Deep-copy ``src`` into ``dst``; access nodes whose ``.data`` is a key
        in ``rename`` get retargeted to ``rename[old]``, and every memlet that
        names the same array follows."""
        node_map = copy_state_contents(src, dst)
        for old, new in node_map.items():
            if not isinstance(new, dace.nodes.AccessNode):
                continue
            if new.data in rename:
                new.data = rename[new.data]
        for e in dst.edges():
            if e.data.data in rename:
                # Rebind memlet to the new array name; keep subset (writes are
                # element-wise per the slice's restriction).
                e.data.data = rename[e.data.data]

    def _emit_merge_tasklet(self, sdfg: dace.SDFG, state: dace.SDFGState, arr_name: str, subset, then_name: str,
                            else_name: str, cond_text: str):
        """Emit ``arr[subset] = merge(cond, _then_arr[subset], _else_arr[subset])``.

        The cond expression is *not* read from any AccessNode in this state —
        the symbolic ``merge`` resolves it through the SDFG's symbol scope,
        matching how IfExpr / interstate conditions reference symbols today.
        """
        access_then = state.add_access(then_name)
        access_else = state.add_access(else_name)
        access_out = state.add_access(arr_name)
        subset_str = str(subset)
        t = state.add_tasklet(
            name=f"merge_{arr_name}",
            inputs={"_t", "_e"},
            outputs={"_o"},
            code=f"_o = merge({cond_text}, _t, _e)",
        )
        state.add_edge(access_then, None, t, "_t", dace.Memlet(expr=f"{then_name}[{subset_str}]"))
        state.add_edge(access_else, None, t, "_e", dace.Memlet(expr=f"{else_name}[{subset_str}]"))
        state.add_edge(t, "_o", access_out, None, dace.Memlet(expr=f"{arr_name}[{subset_str}]"))
