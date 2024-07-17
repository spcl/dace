# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" Loop unroll transformation """

import sympy as sp
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
from dace.transformation import transformation as xf
import sympy


@make_properties
class ConditionalElimination(xf.MultiStateTransformation):
    """
    Given an input conditional that is known to be always true, performs dead code elimination
    by checking the satisfiability of interstate edges
    """

    conditional = Property(
        default=True,
        desc='',
    )
    
    @classmethod
    def expressions(cls):
        return [sd.SDFG('_')]

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
                sdfg.remove_node(s)
            state_list = new_state_list

    def can_be_applied(self, graph, expr_index, sdfg, permissive=False):
        return True

    def apply(self, _, sdfg: sd.SDFG):
        found = True
        while found:
            found = False
            for e in sdfg.edges():
                if e.data.is_unconditional():
                    continue
                try:
                    if isinstance(sympy.And(e.data.condition_sympy(), self.conditional), sympy.logic.boolalg.BooleanFalse):
                        self._eliminate_branch(sdfg, e)
                        found = True
                        break
                    elif isinstance(sympy.Equivalent(e.data.condition_sympy(), self.conditional), sympy.logic.boolalg.BooleanTrue):
                        sdfg.remove_edge(e)
                        sdfg.add_edge(e.src, e.dst, sd.InterstateEdge(assignments=e.data.assignments))
                        found = True
                        break
                except TypeError:
                    continue
