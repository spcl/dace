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
import dace.sdfg.utils as sdutil


@transformation.explicit_cf_compatible
class MoveMapIntoLoop(transformation.MultiStateTransformation):
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
            name=f"map_{self.map_entry.map.label}_loop",
            parent=graph,
        )
        nsdfg = graph.add_nested_sdfg(
            sdfg=inner_sdfg,
            parent=graph,
            name=f"{self.map_entry.map.label}_loop",
            inputs={in_conn[3:] if "IN_" in in_conn else in_conn
                    for in_conn in self.map_entry.in_connectors},
            outputs={
                out_conn[4:] if "OUT_" in out_conn else out_conn
                for out_conn in graph.exit_node(self.map_entry).out_connectors
            })
        sdfg.save("a.sdfg")

        parents = [nsdfg.sdfg]
        for d, (param, (beg, end, step)) in enumerate(zip(self.map_entry.map.params, self.map_entry.map.range)):
            loop_cfg = LoopRegion(
                label=f"for_{self.map_entry.map.label}_dim{d}",
                condition_expr=f"{param} < {dace.symbolic.symstr(end - 1)}",
                loop_var=param,
                initialize_expr=f"{param} = {dace.symbolic.symstr(beg)}",
                update_expr=f"{param} = {param} + {dace.symbolic.symstr(step)}",
            )

            loop_body_cfg = ControlFlowRegion(label=f"body_{self.map_entry.map.label}_dim{d}",
                                              sdfg=parents[-1].sdfg,
                                              parent=parents[-1])

            loop_cfg.add_node(loop_body_cfg, is_start_block=True)

            parents[-1].add_node(loop_cfg, is_start_block=True)
            #print(loop_cfg.sdfg.name)
            parents.append(loop_body_cfg)
            sdfg.save("b.sdfg")

        inner_state = parents[-1].add_state(
            label=f"body",
            is_start_block=True,
        )
        node_map = dict()
        map_scope_nodes = graph.all_nodes_between(self.map_entry, graph.exit_node(self.map_entry))

        for node in map_scope_nodes:
            copy_node = copy.deepcopy(node)
            inner_state.add_node(copy_node)
            node_map[node] = copy_node
        sdutil.set_nested_sdfg_parent_references(nsdfg.sdfg)
        for edge in graph.all_edges(*map_scope_nodes):
            if edge.src in node_map and edge.dst in node_map:
                src_node = node_map[edge.src]
                dst_node = node_map[edge.dst]
                inner_state.add_edge(src_node, edge.src_conn, dst_node, edge.dst_conn, copy.deepcopy(edge.data))
        for ie in graph.in_edges(self.map_entry):
            if ie.data is not None and ie.data.data is not None:
                nsdfg.sdfg.add_datadesc(ie.data.data, copy.deepcopy(sdfg.arrays[ie.data.data]))
        for oe in graph.out_edges(graph.exit_node(self.map_entry)):
            if oe.data is not None and oe.data.data is not None:
                nsdfg.sdfg.add_datadesc(oe.data.data, copy.deepcopy(sdfg.arrays[oe.data.data]))
        for node in inner_state.nodes():
            if isinstance(node, nodes.AccessNode):
                if node.data not in nsdfg.sdfg.arrays:
                    nsdfg.sdfg.add_datadesc(node.data, copy.deepcopy(sdfg.arrays[node.data]))

        for oe in graph.out_edges(self.map_entry):
            if oe.data is not None and oe.data.data is not None:
                assert oe.dst in node_map
                an = inner_state.add_access(oe.data.data)
                inner_state.add_edge(an, None, node_map[oe.dst], oe.dst_conn, copy.deepcopy(oe.data))
        for ie in graph.in_edges(self.map_entry):
            graph.add_edge(ie.src, ie.src_conn, nsdfg, ie.data.data, copy.deepcopy(ie.data))

        for ie in graph.in_edges(graph.exit_node(self.map_entry)):
            if ie.data is not None and ie.data.data is not None:
                assert ie.src in node_map
                an = inner_state.add_access(ie.data.data)
                inner_state.add_edge(node_map[ie.src], ie.src_conn, an, None, copy.deepcopy(ie.data))
            elif ie.data.data is None:
                # If the edge is empty, we can just connect the nodes directly
                graph.add_edge(ie.src, ie.src_conn, nsdfg, None, Memlet())
        for oe in graph.out_edges(graph.exit_node(self.map_entry)):
            if oe.data is not None and oe.data.data is not None:
                graph.add_edge(nsdfg, oe.data.data, oe.dst, oe.dst_conn, copy.deepcopy(oe.data))
            elif oe.data.data is None:
                # If the edge is empty, we can just connect the nodes directly
                graph.add_edge(nsdfg, None, oe.dst, oe.dst_conn, Memlet())

        print(map_scope_nodes)
        for node in [self.map_entry, graph.exit_node(self.map_entry)]:
            graph.remove_node(node)
        for node in map_scope_nodes:
            graph.remove_node(node)
        sdfg.save("d.sdfg")
        nsdfg.sdfg.save("c.sdfg")

        sdutil.add_missing_symbols_to_nsdfg(graph.sdfg, nsdfg, graph)

        #propagation.propagate_memlets_scope(sdfg, body, scope_tree)
