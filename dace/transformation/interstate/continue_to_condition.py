# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.
import copy

from dace import sdfg as sd, properties
from dace.sdfg import utils as sdutil
from dace.sdfg.state import ContinueBlock, ControlFlowRegion, ConditionalBlock
from dace.transformation import transformation as xf


@properties.make_properties
@xf.explicit_cf_compatible
class ContinueToCondition(xf.MultiStateTransformation):
    """
    Converts a continue statement in a loop to a condition in the loop.
    """

    cb = xf.PatternNode(ContinueBlock)

    @classmethod
    def expressions(cls):
        return [sdutil.node_path_graph(cls.cb)]

    def can_be_applied(self, graph, expr_index, sdfg, permissive=False):
        pg = self.cb.parent_graph.parent_graph

        if not isinstance(pg, ConditionalBlock):
            return False
        if len(pg.branches) != 1:
            return False

        code, cfg = pg.branches[0]
        if len(cfg.nodes()) != 1:
            return False
        if cfg.nodes()[0] is not self.cb:
            return False

        outer_cfg = pg.parent_graph
        if len(outer_cfg.successors(pg)) != 1:
            return False

        return True

    def apply(self, graph: ControlFlowRegion, sdfg: sd.SDFG):
        # Eliminate the continue node
        pg: ConditionalBlock = self.cb.parent_graph.parent_graph
        cfg = pg.branches[0][1]
        cfg.remove_node(self.cb)
        # Flip the condition
        pg.branches[0][0].as_string = f"not({pg.branches[0][0].as_string})"
        # Insert all the nodes after the conditional block into the conditional block
        outer_cfg = pg.parent_graph
        to_process = list(outer_cfg.successors(pg))
        old_new_mapping = {}
        to_remove = []
        while to_process:
            node = to_process.pop(0)
            to_process.extend(outer_cfg.successors(node))
            new_node = copy.deepcopy(node)
            old_new_mapping[node] = new_node
            cfg.add_node(new_node)

            for edge in outer_cfg.in_edges(node):
                if edge.src is pg:
                    continue
                cfg.add_edge(
                    old_new_mapping[edge.src],
                    new_node,
                    copy.deepcopy(edge.data),
                )
            to_remove.append(node)

        for node in to_remove:
            outer_cfg.remove_node(node)

        # Fix sdfg parents
        for _, cfg in pg.branches:
            for st in cfg.all_states():
                for node in st.nodes():
                    if isinstance(node, sd.nodes.NestedSDFG):
                        node.sdfg.parent_sdfg = cfg.sdfg
                st.sdfg = cfg.sdfg
            cfg.reset_cfg_list()
