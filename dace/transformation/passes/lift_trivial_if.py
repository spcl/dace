# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.

import re
import dace
from typing import Any, Dict, Optional, Set, Union
from dace import SDFG, ControlFlowRegion
from dace import symbolic
from dace.properties import CodeBlock
from dace.sdfg.sdfg import ConditionalBlock
from dace.sdfg.state import ControlFlowBlock
from dace.sdfg.construction_utils import move_branch_cfg_up_discard_conditions
from dace.transformation import pass_pipeline as ppl, transformation
import dace.sdfg.utils as sdutil
import sympy
from sympy import pycode


@transformation.explicit_cf_compatible
class LiftTrivialIf(ppl.Pass):

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.CFG | ppl.Modifies.States

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return modified & ppl.Modifies.CFG

    def depends_on(self):
        return {}

    def _make_unique_names(self, sdfg: dace.SDFG):
        all_blocks = {
            n
            for n, _ in sdfg.all_nodes_recursive()
            if isinstance(n, dace.SDFGState) or isinstance(n, ControlFlowRegion) or isinstance(n, ControlFlowBlock)
        }
        all_labels: Set[str] = set()
        for n in all_blocks:
            new_label = dace.utils.find_new_name(n.label, all_labels)
            all_labels.add(new_label)
            n.label = new_label

    def _trivial_cond_check(self, code: CodeBlock, val: bool):
        if code.language != dace.dtypes.Language.Python:
            return False

        # Primary: pystr_to_symbolic already handles Python and/or/not and
        # comparison operators, matching how DeadStateElimination evaluates
        # branch conditions. We require a concrete literal back -- bool() of
        # an unevaluated sympy expression (e.g. `A[0]` -> Function(0)) is
        # truthy, which would mis-classify dynamic conditions as trivial.
        try:
            expr = symbolic.pystr_to_symbolic(code.as_string)
            result = symbolic.evaluate(expr, symbols={})
            if isinstance(result, (bool, int, sympy.Integer)) or result in (sympy.true, sympy.false):
                return bool(result) is val
        except Exception:
            pass

        # Fallback: Fortran-frontend SDFGs produce nested comparisons like
        # `(a == 1) == 0` that sympy refuses to compare against an int.
        # Rewrite boolean ops/literals to arithmetic over 0/1 and let
        # SymExpr.simplify reduce it.
        try:
            tokens = re.split(r'(\s+|[()\[\]])', code.as_string)
            replacements = {"True": "1", "False": "0", "and": "*", "or": "+"}
            rewritten = " ".join(replacements.get(t.strip(), t.strip()) for t in tokens).strip()
            simplified = dace.symbolic.SymExpr(rewritten).simplify()
            result = symbolic.evaluate(dace.symbolic.SymExpr(pycode(simplified)), symbols={})
            if isinstance(result, (bool, int, sympy.Integer)) or result in (sympy.true, sympy.false):
                return bool(result) is val
        except Exception:
            pass
        return False

    def _trivially_true(self, code: CodeBlock):
        return self._trivial_cond_check(code, True)

    def _trivially_false(self, code: CodeBlock):
        return self._trivial_cond_check(code, False)

    def _detect_and_remove_top_level_trivial_ifs(self, graph: Union[ControlFlowRegion, SDFG]):
        cfb_to_rm_cfg_to_keep = set()
        rmed_count = 0
        for cfb in graph.nodes():
            if isinstance(cfb, ConditionalBlock):
                # Supported variants:
                # 1. if (cond) where cond is always true
                # 2. if (cond) else
                # 2.1 where cond is always true
                # 2.2 cond is always false
                conditions_and_cfgs = cfb.branches
                if len(conditions_and_cfgs) == 1:
                    cond, cfg = conditions_and_cfgs[0]
                    if self._trivially_true(cond):
                        cfb_to_rm_cfg_to_keep.add((cfb, cfg))
                    elif self._trivially_false(cond):
                        _cfg = ControlFlowRegion(label=f"empty_cfg_of_{cfb.label}", sdfg=cfb.sdfg, parent=cfb)
                        _cfg.add_state(label="empty_placholder", is_start_block=True)
                        cfb.add_branch(condition=None, branch=_cfg)
                        cfb_to_rm_cfg_to_keep.add((cfb, _cfg))
                elif len(conditions_and_cfgs) == 2:
                    cond1, cfg1 = conditions_and_cfgs[0]
                    cond2, cfg2 = conditions_and_cfgs[1]
                    # Either one of them must be none
                    if cond1 is not None and cond2 is not None:
                        continue
                    (not_none_cond, not_none_cfg), (none_cond, none_cfg) = (((cond1, cfg1),
                                                                             (cond2, cfg2)) if cond1 is not None else
                                                                            ((cond2, cfg2), (cond1, cfg1)))

                    if self._trivially_true(not_none_cond):  #2.1
                        cfb_to_rm_cfg_to_keep.add((cfb, not_none_cfg))
                    elif self._trivially_false(not_none_cond):  #2.2
                        cfb_to_rm_cfg_to_keep.add((cfb, none_cfg))

        # Remove trivial Ifs
        for cfb, cfg in cfb_to_rm_cfg_to_keep:
            move_branch_cfg_up_discard_conditions(cfb, cfg)
            assert cfb not in graph.nodes()
            rmed_count += 1

        sdutil.set_nested_sdfg_parent_references(graph.sdfg)
        graph.sdfg.reset_cfg_list()

        return rmed_count

    def _detect_trivial_ifs_and_rm_cfg(self, graph: Union[ControlFlowRegion, SDFG]):
        # We might now have trivial control flow blocks at top level, apply in fixpoint
        rmed_count = self._detect_and_remove_top_level_trivial_ifs(graph)
        local_rmed_count = rmed_count
        while local_rmed_count > 0:
            local_rmed_count = self._detect_and_remove_top_level_trivial_ifs(graph)
            rmed_count += local_rmed_count

        # Now go one one more level in the node list
        for node in graph.all_control_flow_blocks():
            local_rmed_count = self._detect_and_remove_top_level_trivial_ifs(node)
            rmed_count += local_rmed_count

        # Recurse in to nSDFGs
        for state in graph.all_states():
            for node in state.nodes():
                if isinstance(node, dace.nodes.NestedSDFG):
                    rmed_count += self._detect_trivial_ifs_and_rm_cfg(node.sdfg)

        return rmed_count

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> Optional[Dict[str, Set[str]]]:
        # Start with top level nodes and continue further to ensure a trivial if within another trivial if
        # can be processed correctly
        self._make_unique_names(sdfg)
        sdfg.reset_cfg_list()
        self._detect_trivial_ifs_and_rm_cfg(sdfg)
        sdfg.reset_cfg_list()
        return None
