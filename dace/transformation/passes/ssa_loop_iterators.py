# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import dace

from dace.sdfg.state import LoopRegion
from dace.transformation import pass_pipeline as ppl
from typing import Optional
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
                        node.symbol_mapping[str(next_ssa_loop_var)] = replace_symbol_by_name(
                            v_symexpr, loop_var, next_ssa_loop_var)

                    # Now we can replace what is inside
                    to_repl |= str(loop_var) in inner_sdfg.symbols
                    if to_repl:
                        self._repl_recursive(inner_sdfg, loop_var, next_ssa_loop_var)

    def _apply_recursive(self, sdfg: dace.SDFG):
        # Names DaCe knows are arrays -- ``symstr`` uses this set to
        # render array subscripts as ``arr[idx]`` (Python syntax for
        # interstate-edge assignments) rather than ``arr(idx)`` (sympy's
        # default function-call form, which the C++ codegen later
        # rejects since ``arr`` is a pointer, not callable).
        array_names = frozenset(sdfg.arrays.keys())
        for cfg in sdfg.all_control_flow_regions():
            if isinstance(cfg, LoopRegion):
                loop_var = cfg.loop_variable
                if not loop_var:
                    # ``while``/``do-while`` loops with no explicit
                    # induction variable have nothing to SSA-rename.
                    continue
                loop_end_raw = loop_analysis.get_loop_end(cfg)
                next_ssa_loop_var = f"{SSALoopIterators.FOR_IT_NAME}_{SSALoopIterators.loop_var_counter}"
                # Replace loop variable with next_ssa_loop_var in the loop body,
                # and assign loop_var = loop_end at the end of the loop
                self._repl_recursive(cfg, loop_var, next_ssa_loop_var)

                # Assign to the variable after the loop end so any reads
                # AFTER the loop see the canonical post-loop value.
                #
                # ``get_loop_end`` returns the symbolic *cond-failing
                # boundary* (e.g. ``N`` for ``i <= N`` or equivalently
                # ``N`` for ``i < N + 1``) -- coincides with the last
                # attained value only when ``step == 1``.  Fortran /
                # gfortran convention leaves the iterator at the value
                # that **failed** the loop condition, i.e. the first
                # ``init + k*step`` past the bound.
                #
                # Use a ``Mod``-based formulation to dodge a sympy /
                # codegen interaction: ``sp.floor((loop_end - init) /
                # step)`` simplifies to ``floor(n/step - C)`` for any
                # non-zero integer ``C``, which the C++ codegen lowers
                # via ``int_floor(int_floor(n, step) - C, 1)`` -- and
                # the inner ``C = m / step`` then collapses to ``0``
                # under C's integer division when step != 1.  The
                # equivalent ``last + step - ((last + step) % step)``
                # stays integer-typed end-to-end and renders cleanly.
                #
                # Formula: ``post = init + diff - (diff mod step)`` where
                # ``diff = loop_end - init + step``.  Verified:
                #   DO i = 1, N           (step= 1):  diff=N,    mod=0,    post=N+1
                #   DO i = N, 1, -1       (step=-1):  diff=-N,   mod=0,    post=0
                #   DO i = 1, 10, 2       (step= 2):  diff=11,   mod=1,    post=11
                #   DO i = 10, 2, -2      (step=-2):  diff=-10,  mod=0,    post=0
                if loop_end_raw is not None:
                    stride = loop_analysis.get_loop_stride(cfg)
                    init = loop_analysis.get_init_assignment(cfg)
                    if stride is not None and init is not None:
                        diff = loop_end_raw - init + stride
                        exit_value_raw = init + diff - sp.Mod(diff, stride)
                    elif stride is not None:
                        # Init unknown -- fall back to last-attained + step
                        # (correct when ``step == 1`` since loop_end already
                        # equals the last attained value).
                        exit_value_raw = loop_end_raw + stride
                    else:
                        # Stride unknown -- fall back to last-attained
                        # value, preserving prior bridge behavior.
                        exit_value_raw = loop_end_raw
                    exit_value_str = dace.symbolic.symstr(exit_value_raw, arrayexprs=array_names).strip()
                    if exit_value_str:
                        parent_graph = cfg.parent_graph
                        parent_graph.add_state_after(cfg,
                                                     f"SSA_loop_var_reconstruction_{SSALoopIterators.loop_var_counter}",
                                                     assignments={loop_var: f"({exit_value_str})"})

                SSALoopIterators.loop_var_counter += 1

        for state in sdfg.all_states():
            for node in state.nodes():
                if isinstance(node, dace.nodes.NestedSDFG):
                    self._apply_recursive(node.sdfg)

    def apply_pass(self, sdfg: dace.SDFG, _) -> Optional[int]:
        self._apply_recursive(sdfg)
