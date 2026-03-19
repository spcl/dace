# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
import copy
import re
import dace
from typing import Dict, Optional, Set
import sympy
from sympy.printing.pycode import pycode
from dace import SDFG
from dace import properties
from dace import Union
from dace import ControlFlowRegion
from dace.properties import Property
from dace.sdfg.state import ConditionalBlock, LoopRegion
from dace.transformation import pass_pipeline as ppl, transformation
from dace.sdfg.nodes import CodeBlock
import ast


def _get_expr_from_str(expr: str) -> dace.symbolic.SymExpr:
    try:
        parsed_expr = sympy.sympify(expr, evaluate=False)
    except Exception as e:
        parsed_expr = dace.symbolic.SymExpr(expr)
    return parsed_expr


@properties.make_properties
@transformation.explicit_cf_compatible
class RemoveLoopsWithEmptyBodies(ppl.Pass):
    CATEGORY: str = 'Optimization Preparation'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.CFG | ppl.Modifies.Edges

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self):
        return {}

    def _remove_node_connect_src_and_dst(self, node: LoopRegion, parent_graph: dace.ControlFlowRegion):
        ies = parent_graph.in_edges(node)
        oes = parent_graph.out_edges(node)
        assert len(ies) <= 1 and len(oes) <= 1

        if len(ies) == 1 and len(oes) == 1:
            ie = ies[0]
            oe = oes[0]

            new_assignments = dict()
            new_assignments.update(ie.data.assignments)
            new_assignments.update(oe.data.assignments)

            parent_graph.remove_node(node)
            parent_graph.add_edge(ie.src, oe.dst, dace.InterstateEdge(condition="1", assignments=new_assignments))
        elif len(ies) == 1 and len(oes) == 0:
            ie = ies[0]

            new_assignments = dict()
            new_assignments.update(ie.data.assignments)

            parent_graph.remove_node(node)

            if new_assignments != dict():
                nstate = parent_graph.add_state(label=f"{node.label}_in_assignments", is_start_block=False)
                parent_graph.add_edge(ie.src, nstate, dace.InterstateEdge(condition="1", assignments=new_assignments))
        elif len(ies) == 0 and len(oes) == 1:
            oe = oes[0]

            new_assignments = dict()
            new_assignments.update(oe.data.assignments)

            parent_graph.remove_node(node)

            if new_assignments != dict():
                nstate = parent_graph.add_state(label=f"{node.label}_out_assignments", is_start_block=True)
                parent_graph.add_edge(nstate, oe.dst, dace.InterstateEdge(condition="1", assignments=new_assignments))
        else:
            assert len(ies) == 0 and len(oes) == 0
            # Weird prob. should not happen (empty SDFG)?
            parent_graph.remove_node(node)
            parent_graph.add_state(f"{node.label}_empty_replacement", is_start_block=True)

    def _apply(self, cfg: dace.ControlFlowRegion):
        set_to_check = set(cfg.all_control_flow_regions())
        while set_to_check:
            cfgs_to_rm: Set[LoopRegion] = set()

            for node in set_to_check:
                parent_graph = node.parent_graph
                if isinstance(node, LoopRegion):
                    if len(node.nodes()) == 1 and parent_graph.in_degree(node) <= 1 and parent_graph.out_degree(
                            node) <= 1:
                        in_conds = [ie.data.condition.as_string.strip() for ie in parent_graph.in_edges(node)]
                        out_conds = [oe.data.condition.as_string.strip() for oe in parent_graph.out_edges(node)]

                        in_all_true = all(c in ("1", "(1)", "") for c in in_conds)
                        out_all_true = all(c in ("1", "(1)", "") for c in out_conds)

                        if in_all_true and out_all_true:
                            inner_nodes = node.nodes()
                            if len(inner_nodes) == 1:
                                inner_node = list(inner_nodes)[0]
                                if isinstance(inner_node, dace.SDFGState) and len(inner_node.nodes()) == 0:
                                    cfgs_to_rm.add((node, parent_graph))

            set_to_check = {g for n, g in cfgs_to_rm}

            for node, parent_graph in cfgs_to_rm:
                self._remove_node_connect_src_and_dst(node, parent_graph)

    def apply_pass(self, sdfg: SDFG, pipeline_results) -> Optional[Dict[str, Set[str]]]:
        self._apply(sdfg)

        for s in sdfg.all_states():
            for n in s.nodes():
                if isinstance(n, dace.nodes.NestedSDFG):
                    self._apply(n.sdfg)
