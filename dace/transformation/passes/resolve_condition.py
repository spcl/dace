# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.
""" Resolve condition transformation """

from typing import Any, Dict

from dace.transformation import pass_pipeline as ppl
import sympy as sy
from dace.properties import CodeBlock
from dace import sdfg as sd, symbolic
from dace.properties import Property, make_properties, CodeBlock
from sympy.parsing.sympy_parser import parse_expr
from dace import symbolic
from dace.frontend.python.astutils import unparse
from dace.transformation.pass_pipeline import Pipeline
import ast


def eliminate_branch(sdfg: sd.SDFG, initial_edge: sd.graph.Edge):
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


@make_properties
class ResolveCondition(ppl.Pass):
    """
    Given a condition (e.g. var == 1) assumes the condition always holds true, and removes unreachable states
    """

    CATEGORY: str = 'Simplification'

    # Properties
    condition = Property(dtype=str, default="", desc="condition to be parsed")

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return modified & (ppl.Modifies.States | ppl.Modifies.InterstateEdges)

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.States | ppl.Modifies.InterstateEdges

    def apply_pass(self, sdfg: sd.SDFG, _: Dict[str, Any]) -> None:
        # obtain loop information
        try:
            parsed_condition = parse_expr(unparse(symbolic.PythonOpToSympyConverter().visit(
                    ast.parse(self.condition).body[0]))).simplify()
        except:
            return

        seen = set()
        found = True
        while found:
            found = False
            for e in sdfg.edges():
                if e.data.is_unconditional() or e in seen:
                    continue
                # cache seen edges for performance
                seen.add(e)
                try:
                    cond = unparse(symbolic.PythonOpToSympyConverter().visit(
                            ast.parse(e.data.condition.as_string).body[0]))
                    cur_parsed_condition = parse_expr(cond)
                    if sy.And(cur_parsed_condition, parsed_condition).simplify() == False:
                        found = True
                        eliminate_branch(sdfg, e)
                        break
                except:
                    pass

        from dace.transformation.passes import DeadStateElimination

        pipeline = Pipeline([DeadStateElimination()])
        pipeline.apply_pass(sdfg, {})

        # make all lone edges unconditional
        for n in sdfg.nodes():
            out_edges = sdfg.out_edges(n)
            if len(out_edges) == 1 and not out_edges[0].data.is_unconditional():
                out_edges[0].data.condition = CodeBlock("1")
