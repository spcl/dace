# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.
""" Moves a loop around a map into the map """

import copy
import dace
from dace.sdfg.state import ControlFlowRegion, LoopRegion, SDFGState
import dace.transformation.helpers as helpers
import networkx as nx
from dace.sdfg.scope import ScopeTree
from dace import Memlet, nodes, sdfg as sd, subsets as sbs, symbolic, symbol
from dace.sdfg import nodes, propagation, utils as sdutil
from dace.transformation import transformation
from sympy import diff
from typing import List, Set, Tuple


@transformation.explicit_cf_compatible
class MoveIntoMapInto(transformation.MultiStateTransformation):
    """
    Moves a loop around a map into the map
    """

    map_entry = transformation.PatternNode(nodes.MapEntry)

    @classmethod
    def expressions(cls):
        return [sdutil.node_path_graph(cls.map_entry)]

    def can_be_applied(self, graph, expr_index, sdfg, permissive=False):
        # Any map can become a loop
        return True

    def apply(self, graph: SDFGState, sdfg: sd.SDFG):
        # 1. Add a NestedSDFG node to be replaced with the map
        # 2. Redirect all data containers to the NestedSDFG (memlet is the same as before)
        # 3. Add a for-cfg for each dimension of the map (left->right)
        # 4. Add access nodes for all data containers in the innermost for-CFG with the same memlets
        # 5. Connect the addedd access nodes to whatever was within the map scope

        inner_sdfg = dace.SDFG(
            name = f"map_{self.map_entry.map.label}_loop",
            parent = graph,
        )
        nsdfg = graph.add_nested_sdfg(
            sdfg = inner_sdfg,
            parent = graph,
            label = f"map_{self.map_entry.map.label}_loop",
            inputs={in_conn[3:] if "IN_" in in_conn else in_conn for in_conn in self.map_entry.map.in_connectors},
            outputs={out_conn[4:] if "OUT_" in out_conn else out_conn for out_conn in self.map_entry.map.out_connectors}
        )

        cfgs = [graph]
        for d, (param, (beg, end, step)) in enumerate(zip(self.map_entry.map.params, self.map_entry.map.range)):
            loop_cfg = LoopRegion(label=f"for_{self.map_entry.map.label}_dim{d}",
                                    condition_expr=f"{param} < {end - 1}",
                                    loop_var=param,
                                    initialize_expr=f"{param} = {beg}",
                                    update_expr=f"{param} = {param} + {step}",)

            loop_body_cfg = ControlFlowRegion(label=f"body_{self.map_entry.map.label}_dim{d}",
                                            sdfg=cfgs[-1].sdfg, graph=cfgs[-1])

            loop_cfg.add_node(loop_body_cfg, is_start_block=True)
            cfgs.apppend(loop_body_cfg)



        propagation.propagate_memlets_scope(sdfg, body, scope_tree)
