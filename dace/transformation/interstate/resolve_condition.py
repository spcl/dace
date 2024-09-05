# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" Loop unroll transformation """

import sympy as sy
from typing import List
from typing import Optional

from dace.properties import CodeBlock
from dace import sdfg as sd, symbolic
import copy
from dace.properties import Property, make_properties, CodeBlock
from dace.frontend.python.astutils import ASTFindReplace
from dace.sdfg import graph as gr
from dace.sdfg import utils as sdutil
from dace.symbolic import pystr_to_symbolic
from dace.transformation.interstate.loop_detection import (DetectLoop, find_for_loop)
from dace.transformation import transformation as xf
from dace.transformation.interstate import ConditionalElimination
from sympy.parsing.sympy_parser import parse_expr
import networkx as nx
from dace import symbolic
import re
from dace.frontend.python.astutils import unparse
import ast
from dace.transformation.pass_pipeline import Pipeline


@make_properties
class ResolveCondition(xf.MultiStateTransformation):
    """
    Given a condition (e.g. var = 1) assumes the condition always holds true, and removes unreachable states
    Such variables in velocity_advection are:
        lvert_nest(true/false), simpler if true
        lextra_diffu(true/false) simpler if false
        lvn_only(true/false) simpler if true
        istep(1/2) simpler if 2
        l_vert_nested(true/false) simpler if true
        And(Eq(lvert_nest, True), Eq(lextra_diffu, False), Eq(lvn_only, True), Eq(istep, 2), Eq(l_vert_nested, True))
    """
    
    condition = Property(dtype=str, default="lvert_nest and lextra_diffu and lvn_only and istep == 2 and l_vert_nested and ptr_patch.nshift > 0", desc="Condition that always holds")

    @classmethod
    def expressions(cls):
        # Match anything
        return [nx.DiGraph()]

    @staticmethod
    def _eliminate_branch(sdfg: sd.SDFG, initial_edge: gr.Edge):
        sdfg.remove_edge(initial_edge)
        if sdfg.in_degree(initial_edge.dst) > 0:
            return
        state_list = [initial_edge.dst]
        while len(state_list) > 0:
            new_state_list = []
            for s in state_list:
                for e in sdfg.out_edges(s):
                    if len(sdfg.in_edges(e.dst)) == 1:
                        new_state_list.append(e.dst)
                        sdfg.remove_edge(e)
                sdfg.remove_node(s)
            state_list = new_state_list

    def can_be_applied(self, graph, expr_index, sdfg, permissive=False):
        return True

    def apply(self, _, sdfg: sd.SDFG):
        ####################################################################
        # Obtain loop information
        parsed_condition = parse_expr(unparse(symbolic.PythonOpToSympyConverter().visit(ast.parse(self.condition).body[0]))).simplify()
        found = True
        seen = set()
        while found:
            found = False
            for e in sdfg.edges():
                if e.data.is_unconditional() or e in seen:
                    continue
                seen.add(e)
                try:
                    cond = unparse(symbolic.PythonOpToSympyConverter().visit(ast.parse(e.data.condition.as_string).body[0]))
                    cur_parsed_condition = parse_expr(cond)
                    if sy.And(cur_parsed_condition, parsed_condition).simplify() == False:
                        found = True
                        self._eliminate_branch(sdfg, e)
                        break
                except:
                    if e.data.condition.as_string == '(p_diag.ddt_vn_adv_is_associated or p_diag.ddt_vn_cor_is_associated)':
                        found = True
                        self._eliminate_branch(sdfg, e)
                        break
        
        from dace.transformation.passes import DeadStateElimination
        
        pipeline = Pipeline([DeadStateElimination()])
        pipeline.apply_pass(sdfg, {})
        
        # make all lone edges unconditional
        for n in sdfg.nodes():
            out_edges = sdfg.out_edges(n)
            if len(out_edges) == 1 and not out_edges[0].data.is_unconditional():
                out_edges[0].data.condition = CodeBlock("1")