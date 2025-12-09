# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
from typing import Any, Dict
import sympy
import dace
from dace import SDFG, data, properties, SDFGState, symbolic
from dace.sdfg import ControlFlowRegion, nodes
from dace.sdfg.state import BreakBlock, ConditionalBlock, LoopRegion
from dace.transformation import pass_pipeline as ppl, transformation
import dace.sdfg.construction_utils as cutil
import dace.sdfg.utils as sdutil
from dace.transformation.interstate.state_fusion_with_happens_before import StateFusionExtended
from dace.transformation.passes import FuseStates
from dace import dtypes

import ast

# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------


# ------------------------------------------------------------
# HELPER FUNCTIONS
# ------------------------------------------------------------


@properties.make_properties
@transformation.explicit_cf_compatible
class LowerInterstateConditionalAssignmentsToTasklets(ppl.Pass):
    # This pass is testes as part of the vectorization pipeline
    CATEGORY: str = 'Vectorization'

    conditional_assignment_tasklet_prefix = properties.Property(dtype=str,
                                                                default="condition_symbol_to_scalar",
                                                                allow_none=False)
    also_demote = properties.ListProperty(element_type=str, default=list())
    apply_once = properties.Property(dtype=bool, default=False)

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.AccessNodes | ppl.Modifies.InterstateEdges | ppl.Modifies.Tasklets | ppl.Modifies.Edges

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self):
        return {}

    def _apply(self, cfg: ControlFlowRegion):
        if self._applied > 0 and self.apply_once:
            return False

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
                        # Remove python boolean operators
                        # Remove array names
                        # Remove symbols coming from parent sdfg can't be demoted
                        # => Exclude them
                        func_calls = {str(f.func) for f in expr.atoms(sympy.Function)}
                        boolean_func_calls = {
                            "OR", "Or", "or", "AND", "And", "and", "not", "Not", "NOT", "False", "True", "false",
                            "true", "FALSE", "TRUE"
                        }
                        arr_names = {str(k) for k in cfg.sdfg.arrays.keys()}
                        parent_symbol_name = {str(k)
                                              for k in cfg.sdfg.parent_nsdfg_node.symbol_mapping.keys()
                                              } if cfg.sdfg.parent_nsdfg_node is not None else {}
                        no_access_free_syms = all_free_syms - func_calls.union(boolean_func_calls).union(
                            arr_names).union(parent_symbol_name)
                        free_conditional_symbols = free_conditional_symbols.union(no_access_free_syms)

            for additional_demote_sym in self.also_demote:
                if additional_demote_sym in cfg.sdfg.symbols:
                    free_conditional_symbols.add(additional_demote_sym)

            # We should demote all the free conditional symbols
            for conditional_sym in free_conditional_symbols:
                sdfg = cfg.sdfg if not isinstance(cfg, SDFG) else cfg
                # Cast all symbols to fp64
                sdfg.symbols[conditional_sym] = dace.float64
                sdutil.demote_symbol_to_scalar(sdfg, conditional_sym, dace.float64, None)
                # Set-zero all of them
                assert conditional_sym not in sdfg.symbols
                self._applied += 1
                if self._applied > 0 and self.apply_once:
                    return True

        for n in cfg.nodes():
            if isinstance(n, SDFGState):
                for sn in n.nodes():
                    if isinstance(sn, nodes.NestedSDFG):
                        if self._apply(sn.sdfg) and self.apply_once:
                            return True
            elif isinstance(n, ConditionalBlock):
                for _, branch in n.branches:
                    if self._apply(branch) and self.apply_once:
                        return True
            elif isinstance(n, LoopRegion):
                for ln in n.nodes():
                    if not isinstance(ln, SDFGState):
                        if self._apply(ln) and self.apply_once:
                            return True
                    else:
                        for sn in ln.nodes():
                            if isinstance(sn, nodes.NestedSDFG):
                                if self._apply(sn.sdfg) and self.apply_once:
                                    return True
            elif isinstance(n, ControlFlowRegion):
                for ln in n.nodes():
                    if not isinstance(ln, SDFGState):
                        if self._apply(ln) and self.apply_once:
                            return True
                    else:
                        for sn in ln.nodes():
                            if isinstance(sn, nodes.NestedSDFG):
                                if self._apply(sn.sdfg) and self.apply_pass:
                                    return True
            else:
                # Ok if a break block just connintue
                if isinstance(n, BreakBlock):
                    continue
                else:
                    raise Exception(f"Unsupported node type for pass node {n} type {type(n)}")

        return False

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

    def _setzero_true_for_all_transient_scalars(self, sdfg: SDFG):
        for state in sdfg.all_states():
            for node in state.data_nodes():
                if (isinstance(state.sdfg.arrays[node.data], data.Scalar)
                        and state.sdfg.arrays[node.data].transient is True
                        and (state.sdfg.arrays[node.data].lifetime == dtypes.AllocationLifetime.Scope)):
                    node.setzero = True
            for node in state.nodes():
                if isinstance(node, nodes.NestedSDFG):
                    self._setzero_true_for_all_transient_scalars(node.sdfg)

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> bool:
        self._applied = 0
        #self._setzero_true_for_all_transient_scalars(sdfg)
        has_applied = self._apply(sdfg)
        # self._apply_extended_state_fusion(sdfg)
        sdfg.validate()

        return has_applied
