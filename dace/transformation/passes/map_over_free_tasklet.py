# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.
import copy
from typing import Any, Dict, List

import dace
from dace import subsets
from dace.sdfg import SDFG, SDFGState
from dace.properties import make_properties
from dace.sdfg import utils as sdutil
from dace.transformation import pass_pipeline as ppl


@make_properties
class MapOverFreeTasklet(ppl.Pass):
    @classmethod
    def expressions(cls):
        return [sdutil.node_path_graph(cls.first_tasklet)]

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.States

    def _get_component(self, state, first_node):
        nodes_to_check = set([first_node])
        start_nodes = set()
        end_nodes = set()
        checked_nodes = set()
        assert len(state.in_edges(first_node)) == 0
        while nodes_to_check:
            node = nodes_to_check.pop()
            checked_nodes.add(node)
            if len(state.in_edges(node)) == 0:
                start_nodes.add(node)
            if len(state.out_edges(node)) == 0:
                end_nodes.add(node)
            nodes_to_check = nodes_to_check.union(
                [e.src for e in state.in_edges(node) if e.src not in checked_nodes]
            )
            nodes_to_check = nodes_to_check.union(
                [e.dst for e in state.out_edges(node) if e.dst not in checked_nodes]
            )
        return checked_nodes, start_nodes, end_nodes

    def apply_pass(self, sdfg: SDFG, _: Dict[str, Any]):
        counter = 0
        for state in sdfg.states():
            sd = state.scope_dict()
            nodes = state.nodes()
            for node in nodes:
                if isinstance(node, dace.nodes.NestedSDFG):
                    inner_sdfg = node.sdfg
                    self.apply_pass(inner_sdfg, {})
                elif (len(state.in_edges(node)) == 0
                      and sd[node] is None
                      and not isinstance(node, dace.nodes.EntryNode)
                      and not isinstance(node, dace.nodes.ExitNode)
                    ):
                    component, start_nodes, end_nodes = self._get_component(state, node)
                    # Only apply if there are no entry nodes in the component
                    has_entry_node = any([isinstance(v, dace.nodes.EntryNode) for v in component])
                    if not has_entry_node:
                        self._apply(state, start_nodes, end_nodes, counter)
                        counter += 1

    def _apply(self, state: SDFGState,
               start_nodes: List[dace.nodes.Node],
               end_nodes: List[dace.nodes.Node],
               counter: int):
        # Tasklet (No In Edge) -> Access Node not mapped properly
        # If pattern is found create a single-iteration map over it
        # Can be applied ensures chain is not None and has length > 0

        map_entry, map_exit = state.add_map(
            name=f"tasklet_wrapper_{counter}",
            ndrange={f"__tasklet_wrapper_{counter}_it": subsets.Range([(0, 0, 1)])},
        )

        for start_node in start_nodes:
            if isinstance(start_node, dace.nodes.Tasklet):
                state.add_edge(
                    u=map_entry,
                    u_connector=None,
                    v=start_node,
                    v_connector=None,
                    memlet=dace.memlet.Memlet(data=None),
                )
            elif isinstance(start_node, dace.nodes.AccessNode):
                assert len(state.out_edges(start_node)) == 1
                in_conn = f"IN_{start_node.data}"
                out_conn = f"OUT_{start_node.data}"
                pre_map_access = state.add_access(start_node.data)
                edge = state.out_edges(start_node)[0]
                memlet = edge.data
                state.add_edge(
                    u=pre_map_access,
                    u_connector=None,
                    v=map_entry,
                    v_connector=in_conn,
                    memlet=copy.deepcopy(memlet),
                )
                state.add_edge(
                    u=map_entry,
                    u_connector=out_conn,
                    v=start_node,
                    v_connector=None,
                    memlet=copy.deepcopy(memlet),
                )
                map_entry.add_in_connector(in_conn)
                map_entry.add_out_connector(out_conn)
            else:
                raise Exception("MapOverTasklet encountered a free node that is not a tasklet or access node")

        for end_node in end_nodes:
            if not isinstance(end_node, dace.nodes.AccessNode):
                raise Exception("End node is not an AccessNode")
            assert len(state.in_edges(end_node)) == 1
            assert len(state.out_edges(end_node)) == 0
            in_conn = f"IN_{end_node.data}"
            out_conn = f"OUT_{end_node.data}"
            post_map_access = state.add_access(end_node.data)
            edge = state.in_edges(end_node)[0]
            memlet = edge.data
            state.add_edge(
                u=end_node,
                u_connector=None,
                v=map_exit,
                v_connector=in_conn,
                memlet=copy.deepcopy(memlet),
            )
            state.add_edge(
                u=map_exit,
                u_connector=out_conn,
                v=post_map_access,
                v_connector=None,
                memlet=copy.deepcopy(memlet),
            )
            map_exit.add_in_connector(in_conn)
            map_exit.add_out_connector(out_conn)

    def annotates_memlets():
        return False
