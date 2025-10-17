# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.

import dace
import copy
from typing import Any, Dict, Optional, Set, Union
from dace import SDFG, ControlFlowRegion
from dace import symbolic
from dace.properties import CodeBlock
from dace.sdfg.sdfg import ConditionalBlock
from dace.transformation import pass_pipeline as ppl, transformation
import dace.sdfg.utils as sdutil
from sympy import pycode

@transformation.explicit_cf_compatible
class LiftTriviallyTrueIf(ppl.Pass):

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.CFG | ppl.Modifies.States

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return modified & ppl.Modifies.CFG

    def depends_on(self):
        return {}

    def _trivial_cond_check(self, code: CodeBlock, val: bool):
        if code.language != dace.dtypes.Language.Python:
            return False
        try:
            symbolic_expr = dace.symbolic.SymExpr(code.as_string.replace(" and ", " * ").replace(" or ", " + "))
            symbolic_expr = symbolic_expr.simplify()
            pystring = pycode(symbolic_expr)
            result = symbolic.evaluate(expr=dace.symbolic.SymExpr(pystring), symbols=dict())
            return bool(result) is val
        except Exception as e:
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
                        state = dace.SDFGState(label="empty_placeholder")
                        cfg.parent_graph.add_node(node=state, is_start_block=cfg.parent_graph.start_block == cfb, ensure_unique_name=True)
                        cfb_to_rm_cfg_to_keep.add((cfb, state))
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
            self._remove_if_cfb_keep_body(cfb, cfg)
            assert cfb not in graph.nodes()
            rmed_count += 1

        return rmed_count

    def _detect_trivial_ifs_and_rm_cfg(self, graph: Union[ControlFlowRegion, SDFG]):
        # We might now have trivial control flow blocks at top level, apply in fixpoint
        rmed_count = self._detect_and_remove_top_level_trivial_ifs(graph)
        local_rmed_count = rmed_count
        while local_rmed_count > 0:
            local_rmed_count = self._detect_and_remove_top_level_trivial_ifs(graph)
            rmed_count += local_rmed_count

        # Now go one one more level in the node list
        for node in graph.nodes():
            if isinstance(node, ControlFlowRegion):
                rmed_count += self._detect_trivial_ifs_and_rm_cfg(node)
            if isinstance(node, ConditionalBlock):
                for branch, body in node.branches:
                    rmed_count += self._detect_trivial_ifs_and_rm_cfg(body)

        # Recurse in to nSDFGs
        for state in graph.all_states():
            for node in state.nodes():
                if isinstance(node, dace.nodes.NestedSDFG):
                    rmed_count += self._detect_trivial_ifs_and_rm_cfg(node.sdfg)

        return rmed_count

    def _find_new_name(self, n, all_labels):
        i = 0
        label = n.label
        while label in all_labels:
            label = n.label + "_" + str(i)
            i += 1
        return label

    def _remove_if_cfb_keep_body(self, cfb: ConditionalBlock, cfg: Union[ControlFlowRegion, dace.SDFGState], added_labels):
        if not isinstance(cfg, dace.SDFGState):
            parent_graph = cfb.parent_graph
            cfb_in_edges = parent_graph.in_edges(cfb)
            cfb_out_edges = parent_graph.out_edges(cfb)
            parent_graph.remove_node(cfb)

            node_map = dict()
            start_block = cfg.start_block
            end_blocks = [node for node in cfg.nodes() if cfg.out_degree(node) == 0]
            assert len(end_blocks) == 1
            end_block = end_blocks[0]
            added_labels = set()
            for node in cfg.nodes():
                cpnode = copy.deepcopy(node)
                node_map[node] = cpnode
                is_start_block = False
                if len(cfb_in_edges) == 0 and cfg.start_block == node:
                    is_start_block = True
                # Find new name, ensure unique name fails
                label = self._find_new_name(cpnode, {_n.label for _n in parent_graph.all_control_flow_regions()}.union(added_labels))
                cpnode.label = label
                added_labels.add(label)
                #Need to do it for children too...
                print(f"Insert {cpnode} existing labels: {set(_n.label for _n in parent_graph.all_control_flow_regions()).union(added_labels)}")
                parent_graph.add_node(cpnode, is_start_block=is_start_block)

                if isinstance(cpnode, ControlFlowRegion):
                    for n in cpnode.all_control_flow_regions():
                        nn = self._find_new_name(n, {_n.label for _n in parent_graph.all_control_flow_regions()}.union(added_labels))
                        print(nn, n.label)
                        n.label = nn
                        added_labels.add(nn)

            for edge in cfg.edges():
                assert node_map[edge.src] in parent_graph.nodes()
                assert node_map[edge.dst] in parent_graph.nodes()
                parent_graph.add_edge(node_map[edge.src], node_map[edge.dst], copy.deepcopy(edge.data))

            for ie in cfb_in_edges:
                parent_graph.add_edge(ie.src, start_block, copy.deepcopy(ie.data))
            for oe in cfb_out_edges:
                parent_graph.add_edge(end_block, oe.dst, copy.deepcopy(oe.data))

            sdutil.set_nested_sdfg_parent_references(cfg.sdfg)
            cfg.sdfg.reset_cfg_list()
        else:
            parent_graph = cfb.parent_graph
            cfb_in_edges = parent_graph.in_edges(cfb)
            cfb_out_edges = parent_graph.out_edges(cfb)
            parent_graph.remove_node(cfb)
            for ie in cfb_in_edges:
                parent_graph.add_edge(ie.src, cfg, copy.deepcopy(ie.data))
            for oe in cfb_out_edges:
                parent_graph.add_edge(cfg, oe.dst, copy.deepcopy(oe.data))

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> Optional[Dict[str, Set[str]]]:
        # Start with top level nodes and continue further to ensure a trivial if within another trivial if
        # can be processed correctly
        self._detect_trivial_ifs_and_rm_cfg(sdfg)

        return None
