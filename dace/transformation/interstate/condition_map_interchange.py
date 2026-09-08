# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
"""Interchange conditional blocks with nested map regions."""

from dace import sdfg as sd
from dace import dtypes
from dace.sdfg import utils as sdutil
from dace.sdfg.state import ControlFlowRegion, ConditionalBlock, StateSubgraphView
from dace.properties import CodeBlock
from dace.sdfg.nodes import MapEntry, MapExit, NestedSDFG
from dace.memlet import Memlet
from dace.transformation import transformation
from dace.sdfg.sdfg import InterstateEdge
from dace.sdfg.utils import set_nested_sdfg_parent_references
from dace.transformation.helpers import nest_state_subgraph
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
                if (not isinstance(node, (MapEntry, MapExit)) and state.entry_node(node) is None and any([
                        not isinstance(n, (MapEntry, MapExit)) for n in set(state.successors(node))
                        | set(state.predecessors(node))
                ])):
                    return False

        return True

    def apply(self, graph: ControlFlowRegion, sdfg: sd.SDFG):
        branch: ControlFlowRegion = self.cond_block.branches[0][1]
        branch_cond = self.cond_block.branches[0][0]
        cond_syms = set(branch_cond.get_free_symbols())
        all_states = list(branch.all_states())

        # Prepend the condition computation
        cond_sym = graph.sdfg.add_symbol(f"{self.cond_block.label}_cond", dtypes.bool, find_new_name=True)
        graph.sdfg.add_state_before(self.cond_block, assignments={cond_sym: branch_cond.as_string})
        cond_syms = [cond_sym]
        branch_cond = CodeBlock(cond_sym)

        # Wrap any map where the body does not contain a single nested SDFG
        for state in all_states:
            for node in state.nodes():
                if not isinstance(node, MapEntry):
                    continue

                body = list(state.all_nodes_between(node, state.exit_node(node)))
                if len(body) == 1 and isinstance(body[0], NestedSDFG):
                    continue

                # The general subgraph-nesting utility, rather than assembling
                # the body by hand: a hand-rolled version has to re-derive the
                # body's own index space, and gets it wrong as soon as the
                # boundary memlets are indexed by anything but this map's
                # parameters -- an enclosing map's parameter (``A[i, j]`` in the
                # body of a ``j`` map nested in an ``i`` map), or two accesses
                # to one container through different windows.
                nest_state_subgraph(state.sdfg, state, StateSubgraphView(state, body), name='map_body')

        # Wrap all states in the nested SDFGs with the conditional block
        for state in all_states:
            for node in state.nodes():
                if not isinstance(node, MapEntry):
                    continue
                # The body nested above, which is the only computation in the
                # scope. Whatever else the nesting left behind is pure data
                # movement (access nodes routing a container in and back out),
                # which is unaffected by the condition.
                bodies = [n for n in state.all_nodes_between(node, state.exit_node(node)) if isinstance(n, NestedSDFG)]
                assert len(bodies) == 1
                nsdfg: NestedSDFG = bodies[0]
                new_cond_branch = ControlFlowRegion()
                body = list(nsdfg.sdfg.nodes())

                copy_mapping = {}
                for b in body:
                    new_b = copy.deepcopy(b)
                    new_cond_branch.add_node(new_b)
                    copy_mapping[b] = new_b
                for edge in nsdfg.sdfg.edges():
                    new_cond_branch.add_edge(
                        copy_mapping[edge.src],
                        copy_mapping[edge.dst],
                        copy.deepcopy(edge.data),
                    )

                new_cond_block = ConditionalBlock()
                new_cond_block.add_branch(branch_cond, new_cond_branch)

                nsdfg.sdfg.remove_nodes_from(body)
                nsdfg.sdfg.add_node(new_cond_block, ensure_unique_name=True)

                # Pass the symbols used in the condition
                for sym in cond_syms:
                    if sym in state.sdfg.arrays:
                        if sym in nsdfg.sdfg.arrays:  # Already added
                            continue
                        nsdfg.sdfg.add_datadesc(sym, state.sdfg.arrays[sym])
                        nsdfg.add_in_connector(sym)
                        sym_access = state.add_access(sym)
                        conn_name = node.next_connector(sym)
                        node.add_in_connector(f"IN_{conn_name}")
                        node.add_out_connector(f"OUT_{conn_name}")
                        state.add_edge(
                            sym_access,
                            None,
                            node,
                            f"IN_{conn_name}",
                            Memlet.from_array(sym, state.sdfg.arrays[sym]),
                        )
                        state.add_edge(
                            node,
                            f"OUT_{conn_name}",
                            nsdfg,
                            sym,
                            Memlet.from_array(sym, state.sdfg.arrays[sym]),
                        )

                    else:
                        nsdfg.symbol_mapping[sym] = sym
                        if sym not in nsdfg.sdfg.symbols:
                            nsdfg.sdfg.add_symbol(sym, state.sdfg.symbols[sym])

        # Move all states in the branch before the conditional block
        src_state = graph.add_state_before(self.cond_block)
        dst_state = graph.add_state_after(self.cond_block)
        copy_mapping = {}
        for state in all_states:
            new_state = copy.deepcopy(state)
            graph.add_node(new_state, ensure_unique_name=True)
            copy_mapping[state] = new_state

        for edge in branch.edges():
            graph.add_edge(
                copy_mapping[edge.src],
                copy_mapping[edge.dst],
                copy.deepcopy(edge.data),
            )

        graph.add_edge(src_state, copy_mapping[branch.start_block], InterstateEdge())
        for sink in branch.sink_nodes():
            graph.add_edge(copy_mapping[sink], dst_state, InterstateEdge())

        # Remove the conditional block
        graph.remove_node(self.cond_block)

        # Set the parent references of nested SDFGs
        set_nested_sdfg_parent_references(graph)
