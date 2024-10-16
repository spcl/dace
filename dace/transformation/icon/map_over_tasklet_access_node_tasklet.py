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

    def get_tasklet_acces_node_chain(self, state, start_node):
        chain = []
        current_node = start_node
        while current_node is not None:
            out_edges = state.out_edges(current_node)
            if len(out_edges) > 1:
                return None
            elif len(out_edges) == 0:
                chain.append(current_node)
                current_node = None
            else:
                chain.append(current_node)
                e = out_edges[0]
                _, _, v, _, _ = e
                current_node = v

        if not isinstance(chain[-1], nodes.AccessNode):
            return None

        return chain

    def can_be_applied(self, state, expr_index, sdfg, permissive=False):
        # The pattern to look for is [Tasklet-AccesNode] n-times ending with an access node
        if (
            len(state.in_edges(self.first_tasklet)) != 0
            or len(state.out_edges(self.first_tasklet)) != 1
        ):
            return False

        chain = self.get_tasklet_acces_node_chain(state, self.first_tasklet)
        if chain is None or len(chain) == 0:
            return False

        # If there are any access nodes without in edges, we need to put them before the map too
        # But to be able to do that, out degree needs to be 1
        for node in chain:
            if state.in_degree(node) > 1:
                for chain_in_node in [u for u,_,_,_,_ in state.in_edges(node) if not (u in chain)]:
                    if state.in_degree(chain_in_node) != 0 or \
                        state.out_degree(chain_in_node) != 1:
                        return False
        return True

    def apply(self, state: SDFGState, sdfg: SDFG):
        # Tasklet (No In Edge) -> Access Node not mapped properly
        # If pattern is found create a single-iteration map over it
        start_node = self.first_tasklet

        # Can be applied ensures chain is not None and has length > 0
        chain = self.get_tasklet_acces_node_chain(state, start_node)

        counter = MapOverTaskletAccessNodeTaskelet.i
        map_entry, map_exit = state.add_map(
            name=f"tasklet_wrapper_{counter}",
            ndrange={"__": subsets.Range([(0, 0, 1)])},
        )
        MapOverTaskletAccessNodeTaskelet.i += 1

        end_access_node = chain[-1]
        in_edge = state.in_edges(end_access_node)[0]

        state.add_edge(
            u=map_entry,
            u_connector=None,
            v=start_node,
            v_connector=None,
            memlet=dace.memlet.Memlet(data=None),
        )

        _, _, v, _, memlet = in_edge
        map_exit.add_in_connector("IN_" + v.data)
        map_exit.add_out_connector("OUT_" + v.data)
        state.add_edge(
            u=end_access_node,
            u_connector=None,
            v=map_exit,
            v_connector="IN_" + v.data,
            memlet=copy.deepcopy(memlet),
        )
        second_an = nodes.AccessNode(data=v.data)
        state.add_node(second_an)
        state.add_edge(
            u=map_exit,
            u_connector="OUT_" + v.data,
            v=second_an,
            v_connector="IN_" + v.data,
            memlet=copy.deepcopy(memlet),
        )

        # If there are any access nodes without in edges, we need to put them before the map too
        nodes_to_remove = []
        for node in chain:
            if state.in_degree(node) > 1:
                for chain_in_node in [u for u,_,_,_,_ in state.in_edges(node) if not (u in chain)]:
                    in_edge_to_copy = state.out_edges(chain_in_node)[0]
                    _, _, _, v_conn, memlet = in_edge_to_copy
                    pre_map_an = nodes.AccessNode(data=chain_in_node.data)
                    state.add_node(pre_map_an)
                    map_entry.add_in_connector("IN_" + chain_in_node.data)
                    map_entry.add_out_connector("OUT_" + chain_in_node.data)
                    state.add_edge(
                        u=pre_map_an,
                        u_connector=None,
                        v=map_entry,
                        v_connector="IN_" + chain_in_node.data,
                        memlet=copy.deepcopy(memlet),
                    )
                    state.add_edge(
                        u=map_entry,
                        u_connector="OUT_" + chain_in_node.data,
                        v=node,
                        v_connector=v_conn,
                        memlet=copy.deepcopy(memlet),
                    )
                    nodes_to_remove.append(chain_in_node)
        for n in nodes_to_remove:
            state.remove_node(n)

    def annotates_memlets():
        return False
