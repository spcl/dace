# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
"""Interchange conditional blocks with nested map regions."""

from dace import sdfg as sd
from dace.sdfg import utils as sdutil
from dace.sdfg.state import ControlFlowRegion, ConditionalBlock
from dace.sdfg.nodes import MapEntry, MapExit, NestedSDFG
from dace.memlet import Memlet
from dace.transformation import transformation
from dace.sdfg.sdfg import InterstateEdge
from dace.sdfg.utils import set_nested_sdfg_parent_references, propagation
import copy


@transformation.explicit_cf_compatible
class ConditionMapInterchange(transformation.MultiStateTransformation):
    """
    If one or multiple maps are surrounded by a conditional block, moves the conditional block into the nested SDFG of maps.
    """

    cond_block = transformation.PatternNode(ConditionalBlock)

    @classmethod
    def expressions(cls):
        return [sdutil.node_path_graph(cls.cond_block)]

    def can_be_applied(self, graph, expr_index, sdfg, permissive=False):
        # The conditional block should only contain one condition
        # Empty else branches can be preprocessed away
        if len(self.cond_block.branches) != 1:
            return False
        branch: ControlFlowRegion = self.cond_block.branches[0][1]

        # Each state in the branch is either empty or only contains maps
        for state in branch.all_states():
            for node in state.nodes():
                if (
                    not isinstance(node, (MapEntry, MapExit))
                    and state.entry_node(node) is None
                    and any(
                        [
                            not isinstance(n, (MapEntry, MapExit))
                            for n in set(state.successors(node))
                            | set(state.predecessors(node))
                        ]
                    )
                ):
                    return False

        return True

    def apply(self, graph: ControlFlowRegion, sdfg: sd.SDFG):
        branch: ControlFlowRegion = self.cond_block.branches[0][1]
        branch_cond = self.cond_block.branches[0][0]
        cond_syms = set(branch_cond.get_free_symbols())
        all_states = list(branch.all_states())

        # Wrap any map where the body does not contain a single nested SDFG
        for state in all_states:
            for node in state.nodes():
                if not isinstance(node, MapEntry):
                    continue

                body = list(state.all_nodes_between(node, state.exit_node(node)))
                if len(body) == 1 and isinstance(body[0], NestedSDFG):
                    continue

                # Get inputs and outputs of the nested SDFG
                map_exit = state.exit_node(node)
                inputs = set()
                outputs = set()
                for conn in node.out_connectors.keys():
                    edge = list(state.out_edges_by_connector(node, conn))[0]
                    inputs.add(edge.data.data)
                for conn in map_exit.in_connectors.keys():
                    edge = list(state.in_edges_by_connector(node, conn))[0]
                    outputs.add(edge.data.data)

                # Create the nested SDFG and add all symbols
                sym_mapping = {
                    s: s for s in list(graph.sdfg.symbols.keys()) + node.map.params
                }
                nsdfg = state.add_nested_sdfg(
                    sd.SDFG("map_body", parent=state),
                    inputs=inputs,
                    outputs=outputs,
                    parent=state,
                    symbol_mapping=sym_mapping,
                )
                for a, desc in graph.sdfg.arrays.items():
                    if a in inputs or a in outputs or desc.transient:
                        nsdfg.sdfg.add_datadesc(a, desc)

                start_state = nsdfg.sdfg.add_state(is_start_block=True)
                copy_mapping = {}
                for n in body:
                    new_n = copy.deepcopy(n)
                    start_state.add_node(new_n)
                    copy_mapping[n] = new_n
                for n in body + [map_exit]:
                    for edge in state.in_edges(n):
                        src = None
                        src_conn = edge.src_conn
                        dst = None
                        dst_conn = edge.dst_conn
                        if edge.src in copy_mapping:
                            src = copy_mapping[edge.src]
                        elif edge.src is node:
                            src = start_state.add_access(edge.data.data)
                            src_conn = None
                        if edge.dst in copy_mapping:
                            dst = copy_mapping[edge.dst]
                        elif edge.dst is map_exit:
                            dst = start_state.add_access(edge.data.data)
                            dst_conn = None
                        start_state.add_edge(
                            src,
                            src_conn,
                            dst,
                            dst_conn,
                            copy.deepcopy(edge.data),
                        )

                for edge in state.out_edges(node):
                    state.add_edge(
                        edge.src,
                        edge.src_conn,
                        nsdfg,
                        edge.data.data,
                        Memlet.from_array(
                            edge.data.data, graph.sdfg.arrays[edge.data.data]
                        ),
                    )
                for edge in state.in_edges(state.exit_node(node)):
                    state.add_edge(
                        nsdfg,
                        edge.data.data,
                        edge.dst,
                        edge.dst_conn,
                        Memlet(edge.data.data, graph.sdfg.arrays[edge.data.data]),
                    )

                state.remove_nodes_from(body)

        # Wrap all states in the nested SDFGs with the conditional block
        for state in all_states:
            for node in state.nodes():
                if not isinstance(node, MapEntry):
                    continue
                nsdfg: NestedSDFG = list(
                    state.all_nodes_between(node, state.exit_node(node))
                )[0]
                new_cond_branch = ControlFlowRegion()
                new_cond_branch.add_nodes_from(nsdfg.sdfg.all_control_flow_blocks())

                new_cond_block = ConditionalBlock()
                new_cond_block.add_branch(branch_cond, new_cond_branch)

                nsdfg.sdfg.remove_nodes_from(nsdfg.sdfg.all_control_flow_blocks())
                nsdfg.sdfg.add_node(new_cond_block, ensure_unique_name=True)

                # Pass the symbols used in the condition
                for sym in cond_syms:
                    nsdfg.symbol_mapping[sym] = sym

        # Move all states in the branch before the conditional block
        src_state = graph.add_state_before(self.cond_block)
        dst_state = graph.add_state_after(self.cond_block)
        copy_mapping = {}
        for state in all_states:
            new_state = copy.deepcopy(state)
            graph.add_node(new_state, ensure_unique_name=True)
            copy_mapping[state] = new_state

        for state in all_states:
            for edge in branch.in_edges(state):
                graph.add_edge(
                    copy_mapping[edge.src],
                    copy_mapping[state],
                    copy.deepcopy(edge.data),
                )

        graph.add_edge(src_state, copy_mapping[branch.start_block], InterstateEdge())
        for sink in branch.sink_nodes():
            graph.add_edge(copy_mapping[sink], dst_state, InterstateEdge())

        # Remove the conditional block
        graph.remove_node(self.cond_block)

        # Set the parent references of nested SDFGs
        set_nested_sdfg_parent_references(graph)
