# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import dace

from dace.sdfg.state import LoopRegion
from dace.transformation import pass_pipeline as ppl
from dace.sdfg import utils as sdutil
from typing import Optional
import copy
from dace.sdfg.state import ControlFlowRegion
from dace.transformation.passes.analysis import loop_analysis
from dace.transformation.transformation import explicit_cf_compatible

import sympy as sp
from typing import Union


def replace_symbol_by_name(expr: sp.Basic, old_name: str, new: Union[str, sp.Basic]) -> sp.Basic:
    """
    Replace all symbols in `expr` whose .name matches `old_name`,
    regardless of assumptions, with `new`.
    """
    if isinstance(new, str):
        new = sp.Symbol(new)
    repl = {s: new for s in expr.free_symbols if s.name == old_name}
    if not repl:
        return expr
    return expr.subs(repl)

@dace.properties.make_properties
@explicit_cf_compatible
class SSALoopIterators(ppl.Pass):
    loop_var_counter = 0
    FOR_IT_NAME = "_it"

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.AccessNodes | ppl.Modifies.Memlets

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def _repl_recursive(self, cfg: ControlFlowRegion | dace.SDFG, loop_var: str, next_ssa_loop_var: str):
        # What about Nested SDFGs? Do we need to update symbol mapping?
        cfg.replace_meta_accesses({loop_var: next_ssa_loop_var})
        cfg.replace_dict({loop_var: next_ssa_loop_var})

        for state in cfg.all_states():
            for node in state.nodes():
                # Update symbol mapping

                if isinstance(node, dace.nodes.NestedSDFG):
                    inner_sdfg = node.sdfg
                    to_repl = str(loop_var) in node.symbol_mapping
                    if to_repl:
                        v = node.symbol_mapping.pop(str(loop_var))
                        v_symexpr = dace.symbolic.SymExpr(v)
                        node.symbol_mapping[str(next_ssa_loop_var)] = replace_symbol_by_name(v_symexpr, loop_var, next_ssa_loop_var)
                    
                    # Now we can replace what is inside
                    to_repl |= str(loop_var) in inner_sdfg.symbols
                    if to_repl:
                        self._repl_recursive(inner_sdfg, loop_var, next_ssa_loop_var)


    def _apply_recursive(self, sdfg: dace.SDFG):
        for cfg in sdfg.all_control_flow_regions():
            if isinstance(cfg, LoopRegion):
                loop_var = cfg.loop_variable
                loop_end = f"({loop_analysis.get_loop_end(cfg)})" # Inclusive
                next_ssa_loop_var = f"{SSALoopIterators.FOR_IT_NAME}_{SSALoopIterators.loop_var_counter}"
                # Replace loop variable with next_ssa_loop_var in the loop body,
                # and assign loop_var = loop_end at the end of the loop
                self._repl_recursive(cfg, loop_var, next_ssa_loop_var)

                # Assign to the variable after the loop end
                parent_graph = cfg.parent_graph
                parent_graph.add_state_after(cfg, f"SSA_loop_var_reconstruction_{SSALoopIterators.loop_var_counter}",
                                            assignments={loop_var: loop_end})

                SSALoopIterators.loop_var_counter += 1


        for state in sdfg.all_states():
            for node in state.nodes():
                if isinstance(node, dace.nodes.NestedSDFG):
                    self._apply_recursive(node.sdfg)

    def apply_pass(self, sdfg: dace.SDFG, _) -> Optional[int]:
        self._apply_recursive(sdfg)
