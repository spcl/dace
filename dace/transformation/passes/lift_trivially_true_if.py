# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.

import ast
import dace
import copy
from collections import defaultdict
from typing import Any, Dict, Optional, Set

from dace import SDFG, ControlFlowRegion, InterstateEdge
from dace.properties import CodeBlock
from dace.sdfg import nodes as nd
from dace.sdfg.sdfg import ConditionalBlock
from dace.transformation import pass_pipeline as ppl, transformation
from dace.transformation.passes import analysis as ap
import dace.sdfg.utils as sdutil

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
            node = ast.parse(code.as_string, mode='eval')
            result = eval(compile(node, '<string>', 'eval'))
            return bool(result) is val
        except Exception as e:
            return False

    def _trivially_true(self, code: CodeBlock):
        return self._trivial_cond_check(code, True)

    def _trivially_false(self, code: CodeBlock):
        return self._trivial_cond_check(code, False)


    def _detect_trivial_ifs_and_rm_cfg(self, graph: ControlFlowRegion | SDFG, depth=0):
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
                elif len(conditions_and_cfgs) == 2:
                    cond1, cfg1 = conditions_and_cfgs[0]
                    cond2, cfg2 = conditions_and_cfgs[1]
                    # Either one of them must be none
                    if cond1 is not None and cond2 is not None:
                        continue
                    (not_none_cond, not_none_cfg), (none_cond, none_cfg) = (
                        ((cond1, cfg1), (cond2, cfg2)) if cond1 is not None else ((cond2, cfg2), (cond1, cfg1))
                    )

                    if self._trivially_true(not_none_cond): #2.1
                        cfb_to_rm_cfg_to_keep.add((cfb, not_none_cfg))
                    elif self._trivially_false(not_none_cond): #2.2
                        cfb_to_rm_cfg_to_keep.add((cfb, none_cfg))

        # Remove trivial Ifs
        for cfb, cfg in cfb_to_rm_cfg_to_keep:
            self._remove_if_cfb_keep_body(cfb, cfg)
            assert cfb not in graph.nodes()
            rmed_count += 1

        # We might now have trivial control flow blocks at top level, apply in fixpoint
        local_rmed_count = rmed_count
        while local_rmed_count > 0:
            local_rmed_count = self._detect_trivial_ifs_and_rm_cfg(graph, depth)
            rmed_count += local_rmed_count

        # Now go one one more level recursive
        for node in graph.nodes():
            if isinstance(node, ControlFlowRegion):
                rmed_count += self._detect_trivial_ifs_and_rm_cfg(node, depth+1)

        return rmed_count

    def _remove_if_cfb_keep_body(self, cfb: ConditionalBlock, cfg: ControlFlowRegion):
        parent_graph = cfb.parent_graph
        cfb_in_edges = parent_graph.in_edges(cfb)
        cfb_out_edges = parent_graph.out_edges(cfb)
        parent_graph.remove_node(cfb)

        node_map = dict()
        start_block = cfg.start_block
        end_blocks = [node for node in cfg.nodes() if cfg.out_degree(node) == 0]
        assert len(end_blocks) == 1
        end_block = end_blocks[0]
        for node in cfg.nodes():
            cpnode = copy.deepcopy(node)
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
            parent_graph.add_edge(ie.src, start_block, copy.deepcopy(ie.data))
        for oe in cfb_out_edges:
            parent_graph.add_edge(end_block, oe.dst, copy.deepcopy(oe.data))
        
        sdutil.set_nested_sdfg_parent_references(cfg.sdfg)
        cfg.sdfg.reset_cfg_list()


    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> Optional[Dict[str, Set[str]]]:
        # Start with top level nodes and continue further to ensure a trivial if within another trivial if
        # can be processed correctly
        self._detect_trivial_ifs_and_rm_cfg(sdfg)

        return None

