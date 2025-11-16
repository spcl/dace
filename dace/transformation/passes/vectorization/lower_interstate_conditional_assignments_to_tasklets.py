# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
from typing import Any, Dict

import sympy

from dace import SDFG, properties, SDFGState, symbolic
from dace.sdfg import ControlFlowRegion, nodes
from dace.sdfg.state import ConditionalBlock, LoopRegion
from dace.transformation import pass_pipeline as ppl, transformation
import dace.sdfg.construction_utils as cutil
import dace.sdfg.utils as sdutil
from dace.transformation.interstate.state_fusion_with_happens_before import StateFusionExtended
from dace.transformation.passes import FuseStates


@properties.make_properties
@transformation.explicit_cf_compatible
class LowerInterstateConditionalAssignmentsToTasklets(ppl.Pass):
    conditional_assignment_tasklet_prefix = properties.Property(dtype=str,
                                                                default="condition_symbol_to_scalar",
                                                                allow_none=False)
    also_demote = properties.ListProperty(element_type=str, default=list())

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.AccessNodes | ppl.Modifies.InterstateEdges | ppl.Modifies.Tasklets | ppl.Modifies.Edges

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self):
        return {}

    def _apply(self, cfg: ControlFlowRegion):
        if all({isinstance(n, SDFGState) for n in cfg.nodes()}):
            tasklets = set()
            free_conditional_symbols = set()
            for state in cfg.nodes():
                for node in state.nodes():
                    if isinstance(node, nodes.Tasklet) and node.label.startswith(
                            self.conditional_assignment_tasklet_prefix):
                        tasklets.add((node, state))
                        expr = symbolic.SymExpr(node.code.as_string.split(" = ")[-1])
                        syms = expr.free_symbols
                        # If not in inconnectors then it is a symbol
                        all_free_syms = {str(s) for s in syms if str(s) not in node.in_connectors}
                        # Should be empty
                        func_calls = {str(f) for f in expr.atoms(sympy.Function)}
                        assert len(func_calls) == 0
                        #print(all_free_syms)
                        #print(func_calls)
                        no_access_fre_syms = all_free_syms - func_calls
                        free_conditional_symbols = free_conditional_symbols.union(no_access_fre_syms)

            for additional_demote_sym in self.also_demote:
                if additional_demote_sym in cfg.sdfg.symbols:
                    free_conditional_symbols.add(additional_demote_sym)

            # We should demote all the free conditional symbols
            if len(free_conditional_symbols) != 0:
                print(f"Demote symbols: {free_conditional_symbols}")
            for conditional_sym in free_conditional_symbols:
                sdfg = cfg.sdfg if not isinstance(cfg, SDFG) else cfg
                sdutil.demote_symbol_to_scalar(sdfg, conditional_sym, None, None)
                assert conditional_sym not in sdfg.symbols

        for n in cfg.nodes():
            if isinstance(n, SDFGState):
                for sn in n.nodes():
                    if isinstance(sn, nodes.NestedSDFG):
                        self._apply(sn.sdfg)
            elif isinstance(n, ConditionalBlock):
                for _, branch in n.branches:
                    self._apply(branch)
            elif isinstance(n, LoopRegion):
                for ln in n.nodes():
                    if not isinstance(ln, SDFGState):
                        self._apply(ln)
                    else:
                        for sn in ln.nodes():
                            if isinstance(sn, nodes.NestedSDFG):
                                self._apply(sn.sdfg)
            elif isinstance(n, ControlFlowRegion):
                for ln in n.nodes():
                    if not isinstance(ln, SDFGState):
                        self._apply(ln)
                    else:
                        for sn in ln.nodes():
                            if isinstance(sn, nodes.NestedSDFG):
                                self._apply(sn.sdfg)
            else:
                raise Exception(f"Unsupported node type for pass node {n} type {type(n)}")

    def _apply_extended_state_fusion(self, sdfg: SDFG):
        FuseStates().apply_pass(sdfg, {})
        applied = True
        while applied:
            applied = False
            for s1 in sdfg.bfs_nodes(sdfg.start_block):
                if s1 not in sdfg.nodes():
                    continue
                out_edges = sdfg.out_edges(s1)
                assert len(out_edges) <= 1
                if len(out_edges) == 1:
                    s2 = out_edges[0].dst
                    if s2 not in sdfg.nodes():
                        continue
                    if isinstance(s1, SDFGState) and isinstance(s2, SDFGState):
                        t = StateFusionExtended()
                        t.first_state = s1
                        t.second_state = s2
                        if t.can_be_applied(sdfg, 0, sdfg, False):
                            applied = True
                            t.apply(sdfg, sdfg)

        for state in sdfg.all_states():
            for node in state.nodes():
                if isinstance(node, nodes.NestedSDFG):
                    self._apply_extended_state_fusion(node.sdfg)

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> None:
        self._apply(sdfg)
        self._apply_extended_state_fusion(sdfg)
        return
