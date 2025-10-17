# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.

import re
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
        all_labels = set()

        def _find_new_name(cfg: ControlFlowRegion) -> str:
            candidate_label = cfg.label
            i = 0
            while candidate_label in all_labels:
                candidate_label = cfg.label + "_" + str(i)
                i += 1
            if candidate_label in all_labels:
                assert False
            all_labels.add(candidate_label)
            return candidate_label

        for n in all_blocks:
            new_label = _find_new_name(n)
            n.label = new_label

    def _trivial_cond_check(self, code: CodeBlock, val: bool):
        if code.language != dace.dtypes.Language.Python:
            return False
        try:
            def _token_replace_dict(string_to_check: str, dict) -> str:
                # Split while keeping delimiters
                tokens = re.split(r'(\s+|[()\[\]])', string_to_check)

                # Replace tokens that exactly match src
                tokens = [dict[token.strip()] if token.strip() in dict else token.strip() for token in tokens]

                return " ".join(tokens).strip()

            symbolic_expr = dace.symbolic.SymExpr(
                _token_replace_dict(code.as_string, {
                    "True": "1",
                    "and": " * ",
                    "or": "+",
                    "False": "0"
                }))
            symbolic_expr = symbolic_expr.simplify()
            pystring = pycode(symbolic_expr)
            result = symbolic.evaluate(expr=dace.symbolic.SymExpr(pystring), symbols=dict())
            return bool(result) is val
        except Exception as e:
            print(e)
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
            self._remove_if_cfb_keep_body(cfb, cfg)
            assert cfb not in graph.nodes()
            rmed_count += 1

        sdutil.set_nested_sdfg_parent_references(graph.sdfg)
        graph.sdfg.reset_cfg_list()

        return rmed_count

    def _detect_trivial_ifs_and_rm_cfg(self, graph: Union[ControlFlowRegion, SDFG]):
        # We might now have trivial control flow blocks at top level, apply in fixpoint
        rmed_count = self._detect_and_remove_top_level_trivial_ifs(graph)
        local_rmed_count = rmed_count
        graph.sdfg.validate()
        while local_rmed_count > 0:
            local_rmed_count = self._detect_and_remove_top_level_trivial_ifs(graph)
            rmed_count += local_rmed_count

        graph.sdfg.validate()

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

    def _remove_if_cfb_keep_body(self, cfb: ConditionalBlock, cfg: ControlFlowRegion):
        parent_graph = cfb.parent_graph
        cfb_in_edges = parent_graph.in_edges(cfb)
        cfb_out_edges = parent_graph.out_edges(cfb)
        parent_graph.remove_node(cfb)

        node_map = dict()
        start_block = cfg.start_block
        start_blocks = [node for node in cfg.nodes() if cfg.in_degree(node) == 0]
        assert [start_block] == start_blocks
        end_blocks = [node for node in cfg.nodes() if cfg.out_degree(node) == 0]
        assert len(end_blocks) == 1
        end_block = end_blocks[0]
        for node in cfg.nodes():
            if node not in node_map:
                cpnode = copy.deepcopy(node)
            else:
                cpnode = node_map[node]
            node_map[node] = cpnode
            is_start_block = False
            if len(cfb_in_edges) == 0 and cfg.start_block == node:
                is_start_block = True
            parent_graph.add_node(cpnode, is_start_block=is_start_block)

        for edge in cfg.edges():
            assert node_map[edge.src] in parent_graph.nodes()
            assert node_map[edge.dst] in parent_graph.nodes()
            parent_graph.add_edge(node_map[edge.src], node_map[edge.dst], copy.deepcopy(edge.data))

        for ie in cfb_in_edges:
            assert ie.src not in node_map
            parent_graph.add_edge(ie.src, node_map[start_block], copy.deepcopy(ie.data))
            assert ie.dst not in parent_graph
        for oe in cfb_out_edges:
            assert oe.dst not in node_map
            parent_graph.add_edge(node_map[end_block], oe.dst, copy.deepcopy(oe.data))
            assert oe.src not in parent_graph

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> Optional[Dict[str, Set[str]]]:
        # Start with top level nodes and continue further to ensure a trivial if within another trivial if
        # can be processed correctly
        self._make_unique_names(sdfg)
        sdfg.reset_cfg_list()
        self._detect_trivial_ifs_and_rm_cfg(sdfg)
        sdfg.reset_cfg_list()
        return None
