# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.

import re

import sympy
import dace
import copy
from typing import Any, Dict, Optional, Set, Union
from dace import SDFG, ControlFlowRegion
from dace import symbolic
from dace.properties import CodeBlock
from dace.sdfg.sdfg import ConditionalBlock
from dace.sdfg.state import ControlFlowBlock, LoopRegion
from dace.transformation import pass_pipeline as ppl, transformation
import dace.sdfg.utils as sdutil
from sympy import pycode
from collections import Counter


@transformation.explicit_cf_compatible
class NormalizeBranchConditions(ppl.Pass):

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.CFG | ppl.Modifies.States | ppl.Modifies.InterstateEdges

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return modified & ppl.Modifies.CFG

    def depends_on(self):
        return {}

    _cond_assignment_state_id = 0

    def _get_in_edge(self, cb: ConditionalBlock, always_create_new_state: bool):
        g = cb.parent_graph
        if g.in_degree(cb) > 1 or g.in_degree(cb) == 0 or always_create_new_state:
            is_start_block = (g.start_block == cb)
            g.add_state_before(cb,
                               label=f"cond_assignment_{self._cond_assignment_state_id}",
                               is_start_block=is_start_block)
            self._cond_assignment_state_id += 1

            assert g.in_degree(cb) == 1

        ie = g.in_edges(cb)[0]
        return ie

    def analyze_condition(self, cond_str: str, sdfg: dace.SDFG):
        """
        Analyze a condition string from an if statement.
        Return None if analysis failes.
        Else returns the new condition as a string and required interstate assignment
        Returns None, None if nothing new is necessray
        """

        def _is_just_a_variable(symexpr: dace.symbolic.SymExpr, cond_lhs: str) -> bool:
            free_symbols = symexpr.free_symbols
            funcs = list(symexpr.atoms(sympy.Function))
            print(symexpr, free_symbols, funcs)
            return (len(free_symbols) == 1 and len(funcs) == 0
                    and cond_lhs.strip().replace("(", "").replace(")", "") == str(next(iter(free_symbols))).strip())

        cond_str = cond_str.strip()

        # Step 1: Check for equality
        equality_count = cond_str.count(" == ")

        if equality_count > 1:
            return None
        else:
            if equality_count == 1:
                lhs, rhs = cond_str.split(" == ")
                lhs = lhs.strip()
                if lhs.count("(") > lhs.count(")"):
                    assert lhs.count("(") == lhs.count(")") + 1
                    lhs = lhs[1:]
                rhs = rhs.strip()
                if rhs.count(")") > rhs.count(")"):
                    assert rhs.count(")") == rhs.count("(") + 1
                    rhs = rhs[:-1]
                # Parse lhs and see if it consist only of a single parameter
                symexpr = dace.symbolic.pystr_to_symbolic(cond_str)
                if _is_just_a_variable(symexpr, lhs):
                    # Found format `if(var == 1)`
                    # Is Ok.
                    return (None, None)
                else:
                    # Found format `if((expr) == 1`
                    new_cond_name = "normalized_cond"
                    new_cond_name = sdfg.add_symbol(name=new_cond_name, stype=dace.int32, find_new_name=True)
                    new_cond_str = f"{new_cond_name} == 1"
                    necessary_assignment = f"{new_cond_name}: {lhs.strip()}"
                    return (new_cond_str, necessary_assignment)
            else:
                symexpr = dace.symbolic.pystr_to_symbolic(cond_str)
                if _is_just_a_variable(symexpr, cond_str):
                    # Found format `if(var)`
                    new_cond_str = f"{pycode(symexpr).strip()} == 1"
                    return (new_cond_str, None)
                else:
                    # Found format `if(expr)`
                    new_cond_name = "normalized_cond"
                    new_cond_name = sdfg.add_symbol(name=new_cond_name, stype=dace.int32, find_new_name=True)
                    new_cond_str = f"{new_cond_name} == 1"
                    necessary_assignment = f"{new_cond_name}: {cond_str}"
                    return (new_cond_str, necessary_assignment)

    def _apply(self, sdfg: dace.SDFG):
        """
        Normalized branch conditions are of form:
        `if (cond1 == 1)`
        Where `cond1` is a symbol / variable

        Supported patterns to change are:
        `if(cond1) -> if(cond1 == 1)`
        `if((expr) == 1) -> cond1 = expr; if(cond1 == 1)`
        `if((expr)) -> cond1 = expr; if(cond1 == 1)`
        Where `expr` does not contain a `==`
        """
        for cb in sdfg.all_control_flow_blocks():
            if not isinstance(cb, ConditionalBlock):
                continue
            for i, (cond, body) in enumerate(cb.branches):
                if cond is None or cond.language != dace.dtypes.Language.Python:
                    continue

                new_cond_name = "normalized_condition"
                analysis = self.analyze_condition(cond.as_string, sdfg)
                if analysis is None:
                    # Analysis failed (multiple `==` operators, or not Python)
                    continue
                elif analysis == (None, None):
                    # Nothing needs to be done
                    continue
                else:
                    new_cond: str = analysis[0]
                    new_assignment: str = analysis[1]
                    cb.branches[i] = (CodeBlock(new_cond), body)
                    if new_assignment is not None:
                        assignment_lhs, assignment_rhs = new_assignment.split(":")
                        assignment_lhs = assignment_lhs.strip()
                        assignment_rhs = assignment_rhs.strip()
                        in_edge = self._get_in_edge(cb, True)
                        # A new name
                        assert assignment_lhs not in in_edge.data.assignments
                        in_edge.data.assignments[assignment_lhs] = assignment_rhs

        for state in sdfg.all_states():
            for node in state.nodes():
                if isinstance(node, dace.nodes.NestedSDFG):
                    self._apply(node.sdfg)

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> Optional[Dict[str, Set[str]]]:
        """
        Normalized branch conditions are of form:
        `if (cond1 == 1)`
        Where `cond1` is a symbol / variable

        Supported patterns to change are:
        `if(cond1) -> if(cond1 == 1)`
        `if((expr) == 1) -> cond1 = expr; if(cond1 == 1)`
        `if((expr)) -> cond1 = expr; if(cond1 == 1)`
        """
        self._apply(sdfg)
        return None
