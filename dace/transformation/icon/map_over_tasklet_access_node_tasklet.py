# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
""" This module contains classes and functions that implement the grid-strided map tiling
    transformation."""


import copy
from dace import subsets
import dace
from dace.sdfg import SDFG, SDFGState
from dace.properties import make_properties, SymbolicProperty
from dace.sdfg import nodes
from dace.sdfg import utils as sdutil
from dace.transformation import transformation


@make_properties
class MapOverTaskletAccessNodeTaskelet(transformation.SingleStateTransformation):
    i = 0
    first_tasklet = transformation.PatternNode(nodes.Tasklet)

    @classmethod
    def expressions(cls):
        return [sdutil.node_path_graph(cls.first_tasklet)]

    def find_end_node(self, state, start_node):
        node_gen = sdutil.dfs_topological_sort(G=state, sources=start_node)
        end_nodes = set()
        for node in node_gen:
            if len(state.out_edges(node)) == 0:
                end_nodes.add(node)

        return list(end_nodes)

    def can_be_applied(self, state, expr_index, sdfg, permissive=False):
        if (
            len(state.in_edges(self.first_tasklet)) != 0
            or len(state.out_edges(self.first_tasklet)) != 1
        ):
            return False
        oe = state.out_edges(self.first_tasklet)[0]
        _, _, an, _, _ = oe
        if (
            (not isinstance(an, nodes.AccessNode))
            or len(state.in_edges(an)) != 1
            or len(state.out_edges(an)) != 1
        ):
            return False
        end_nodes = self.find_end_node(state, self.first_tasklet)
        if len(end_nodes) != 1 or (not isinstance(end_nodes[0], nodes.AccessNode)) or \
            len(state.in_edges(end_nodes[0])) != 1:
            return False
        return True

    def apply(self, state: SDFGState, sdfg: SDFG):
        # Tasklet (No In Edge) -> Access Node not mapped properly
        # If pattern is found create a single-iteration map over it
        start_node = self.first_tasklet

        node_gen = sdutil.dfs_topological_sort(G=state, sources=start_node)
        end_nodes = set()
        for node in node_gen:
            if len(state.out_edges(node)) == 0:
                end_nodes.add(node)

        counter = MapOverTaskletAccessNodeTaskelet.i
        map_entry, map_exit = state.add_map(
            name=f"tasklet_wrapper_{counter}", ndrange={"__":subsets.Range([(0, 0, 1)])}
        )
        MapOverTaskletAccessNodeTaskelet.i += 1

        end_access_node = self.find_end_node(state, self.first_tasklet)[0]
        in_edge = state.in_edges(end_access_node)[0]

        map_to_tasklet = state.add_edge(u=map_entry,
                                        u_connector=None,
                                        v=start_node,
                                        v_connector=None,
                                        memlet=dace.memlet.Memlet(data=None))

        u, uc, v, vc, memlet = in_edge
        map_exit.add_in_connector("IN_" + v.data)
        map_exit.add_out_connector("OUT_" + v.data)
        state.add_edge(
            u=end_access_node,
            u_connector=None,
            v=map_exit,
            v_connector="IN_" + v.data,
            memlet=copy.deepcopy(memlet)
        )
        second_an = nodes.AccessNode(data=v.data)
        state.add_node(second_an)
        state.add_edge(
            u=map_exit,
            u_connector="OUT_" + v.data,
            v=second_an,
            v_connector="IN_" + v.data,
            memlet=copy.deepcopy(memlet)
        )

    def annotates_memlets():
        return False
