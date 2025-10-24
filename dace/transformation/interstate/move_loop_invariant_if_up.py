import dace

import copy
from dace.properties import CodeBlock
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion, LoopRegion, SDFGState
import dace.transformation.helpers as helpers
import networkx as nx
from dace.sdfg.scope import ScopeTree
from dace import Memlet, nodes, sdfg as sd, subsets as sbs, symbolic, symbol
from dace.sdfg import nodes, propagation, utils as sdutil
from dace.transformation import transformation
from sympy import diff
from typing import List, Set, Tuple
import dace.sdfg.construction_utils as cutil
from dace.transformation.passes.analysis import loop_analysis


def fold(memlet_subset_ranges, itervar, lower, upper):
    return [(r[0].replace(symbol(itervar), lower), r[1].replace(symbol(itervar), upper), r[2])
            for r in memlet_subset_ranges]


def offset(memlet_subset_ranges, value):
    return (memlet_subset_ranges[0] + value, memlet_subset_ranges[1] + value, memlet_subset_ranges[2])


@transformation.explicit_cf_compatible
class MoveLoopInvariantIfUp(transformation.MultiStateTransformation):
    """
    Moves a loop around a map into the map
    """

    map_state = transformation.PatternNode(dace.SDFGState)
    map_entry = transformation.PatternNode(dace.nodes.MapEntry)
    if_block = transformation.PatternNode(ConditionalBlock)

    @classmethod
    def expressions(cls):
        return [
            sdutil.node_path_graph(cls.map_state),
            sdutil.node_path_graph(cls.map_entry),
            sdutil.node_path_graph(cls.loop)
        ]

    def can_be_applied(self, graph, expr_index, sdfg, permissive=False):
        # If condition needs to be really invariant the map_entry (and all maps between map entry and the if)
        # All tasklets of the nestedSDFG needs to be inside this if condition
        # Map body needs to consist only of a nested SDFG (+ nested maps)
        # If-block is in top level nodes
        return

    def apply(self, graph: ControlFlowRegion, sdfg: sd.SDFG):
        # We have
        # MapEntry -> NestedSDFG
        # Inside the NestedSDFG we have the IfCondition that is loop invariant
        # We want to move it upwards

        # To do this:
        # 1. Create new NestedSDFG Node
        # 2. Put the if Inside the nestedSDFG
        # 2.1 Copy all interstate edges - and prune them later
        # 3. Add state inside the new If (for each body)
        # 4. Copy map conents inside the new state
        # 5. Add data for map's input-outputs
        # 6. Reconnect previous maps input-outputs to the nestedSDFG
        # 7. Remove the map nodes and edges
        # 8. Clean degree-1 nodes
        # - 2.2 Prune unneeded interstate edges in the new SDFG

        #1.
        new_sdfg = dace.SDFG(name=f"{self.if_block.label}_nsdfg", parent=self.map_state)

        cb = ConditionalBlock(label=f"{self.if_block.label}", sdfg=new_sdfg, parent=new_sdfg)

        new_sdfg.add_node(cb, is_start_block=True)

        new_branches = [(CodeBlock(cond), ControlFlowRegion(label=f"{body.label}", sdfg=new_sdfg, parent=cb))
                        for cond, body in self.if_block.branches]
        # Need to generate else branch if the if-block is not the only node
        if_block_sdfg = self.if_block.sdfg
        assert self.if_block in if_block_sdfg.nodes()
        if len(if_block_sdfg.nodes()) != 1:
            assert len(self.if_block.branches) == 1
            body_label = self.if_block.branches[0][1].label
            new_branches.append((None, ControlFlowRegion(label=f"{body_label}_else", sdfg=new_sdfg, parent=cb)))
            # Add else body the current if-block
            cfg = ControlFlowRegion(label=f"{body_label}_else", sdfg=new_sdfg, parent=cb)
            self.if_block.add_branch(None, cfg)
            # Add an empty state as placeholder
            cfg.add_state("empty_s", is_start_block=True)

        branch_and_state_sdfg_map = dict()
        branch_and_state_sdfg_map_w_labels = dict()
        for cond, body in new_branches:
            # 3
            s1 = body.add_state(f"main_{body.label}", is_start_block=True)
            cb.add_branch(cond, body)
            new_map_content_sdfg = dace.SDFG(name=f"{body.label}_nsdfg", parent=self.map_state)
            branch_and_state_sdfg_map[body] = (s1, new_map_content_sdfg)
            branch_and_state_sdfg_map_w_labels[body.label] = (s1, new_map_content_sdfg)

        # 2
        node_maps = dict()

        for i, (body_label, (state, new_map_content_sdfg)) in enumerate(branch_and_state_sdfg_map_w_labels.items()):
            # Copy everything
            if_block_sdfg = self.if_block.sdfg
            assert self.if_block in if_block_sdfg.nodes()
            node_map = cutil.copy_graph_contents(if_block_sdfg, new_map_content_sdfg)

            # Map the old if block to the new if block
            new_if_block = node_map[self.if_block]
            body_to_take = {body for c, body in new_if_block.branches if body.label == body_label}.pop()
            branches = {b for _, b in new_if_block.branches}
            assert body_to_take in branches, f"{body_to_take} not in {branches}"

            cutil.move_branch_cfg_up_discard_conditions(new_if_block, body_to_take)
            node_maps[body] = node_map

        # Both nested SDFGs can reuse the previous nested SDFGs inputs and outputs
        old_nsdfg_node = self.if_block.sdfg.parent_nsdfg_node

        input_to_arr_name = {
            ie.dst_conn: ie.data.data
            for ie in self.map_state.in_edges(old_nsdfg_node) if ie.data.data is not None
        }
        output_to_arr_name = {
            oe.src_conn: oe.data.data
            for oe in self.map_state.out_edges(old_nsdfg_node) if oe.data.data is not None
        }

        # Collect map params
        parent_maps = list()
        sdict = self.map_state.scope_dict()
        cur_parent = sdict[old_nsdfg_node]
        while cur_parent != self.map_entry:
            parent_maps.append(cur_parent)
            cur_parent = sdict[cur_parent]
        parent_maps.append(self.map_entry)

        map_symbols = set()
        map_params = set()
        for map_entry in parent_maps:
            map_params = map_params.union(map_entry.map.params)
            # dynamic in conncetors and free symbols in the ranges
            in_conns = {in_conn for in_conn in map_entry.in_connectors if not in_conn.startswith("IN_")}
            map_symbols = map_symbols.union(in_conns)

        # Delete map parameters from the new symbol mapping
        new_symbol_mapping = copy.deepcopy(old_nsdfg_node.symbol_mapping)
        syms_defined_at = self.map_state.symbols_defined_at(old_nsdfg_node)
        for k, t in syms_defined_at.items():
            #new_sdfg.add_symbol(k, syms_defined_at[k])
            if k not in new_symbol_mapping:
                new_symbol_mapping[k] = k
        new_inner_symbol_mapping = copy.deepcopy(new_symbol_mapping)

        for k in map_params:
            #new_sdfg.add_symbol(k, syms_defined_at[k])
            del new_symbol_mapping[k]
        for s in map_symbols:
            # Parent map gets them as scalars
            if s not in new_sdfg.arrays:
                new_sdfg.add_scalar(s, syms_defined_at[s])
                print(f"Add {s}")
                del new_symbol_mapping[s]

        new_outer_symbol_mapping = copy.deepcopy(new_symbol_mapping)

        for i, (body, (state, new_map_content_sdfg)) in enumerate(branch_and_state_sdfg_map.items()):
            for arr_names in [input_to_arr_name, output_to_arr_name]:
                for conn_name in arr_names:
                    if conn_name not in new_map_content_sdfg.arrays:
                        copydesc = copy.deepcopy(self.if_block.sdfg.arrays[conn_name])
                        copydesc.transient = False
                        new_map_content_sdfg.add_datadesc(conn_name, copydesc)
            for arr_name, arr in self.if_block.sdfg.arrays.items():
                if arr_name not in new_map_content_sdfg.arrays:
                    copydesc = copy.deepcopy(arr)
                    new_map_content_sdfg.add_datadesc(arr_name, copydesc)
            nsdfg = state.add_nested_sdfg(sdfg=new_map_content_sdfg,
                                          inputs=set(input_to_arr_name.keys()),
                                          outputs=set(output_to_arr_name.keys()),
                                          symbol_mapping=new_inner_symbol_mapping)
            new_map_content_sdfg.parent_nsdfg_node = nsdfg

        # Old arrays used by the map will be registered to the outside map
        new_inputs = set(input_to_arr_name.values())
        new_outputs = set(output_to_arr_name.values())
        for arr_names in [new_inputs, new_outputs]:
            for arr_name in arr_names:
                if arr_name not in new_sdfg.arrays:
                    copydesc = copy.deepcopy(self.map_state.sdfg.arrays[arr_name])
                    copydesc.transient = False
                    new_sdfg.add_datadesc(arr_name, copydesc)

        # Copy over map entry
        # If edge.dst is not in the node map then to the nested SDFG
        # If edge.src is not in the node map then need to add access node

        # If you find dynamic in connectors add to dynamic inputs of the new sdfg
        # ======================================================================================
        for i, (body, (state, new_map_content_sdfg)) in enumerate(branch_and_state_sdfg_map.items()):
            assert isinstance(state, dace.SDFGState)
            node_map = dict()

            # Copy and add map entries
            print(parent_maps)
            for j, map_entry in enumerate(reversed(parent_maps)):
                node_map[map_entry] = copy.deepcopy(map_entry)
                map_exit = self.map_state.exit_node(map_entry)
                node_map[map_exit] = copy.deepcopy(map_exit)
                state.add_node(node_map[map_entry])
                state.add_node(node_map[map_exit])

            # Handle incoming edges to the map entries
            for j, map_entry in enumerate(reversed(parent_maps)):
                for ie in self.map_state.in_edges(map_entry):
                    # If it is a dynamic in connector we should also skip
                    if j == 0 and (not ie.dst_conn.startswith("IN_")):
                        new_inputs.add(ie.dst_conn)
                        #continue

                    if ie.src not in node_map and ie.data.data is not None:
                        node_map[ie.src] = state.add_access(ie.data.data)

                    if ie.data.data is None:
                        continue

                    state.add_edge(node_map[ie.src], ie.src_conn, node_map[ie.dst], ie.dst_conn, copy.deepcopy(ie.data))

                    if ie.dst_conn is not None and ie.dst_conn not in node_map[ie.dst].in_connectors:
                        node_map[ie.dst].add_in_connector(ie.dst_conn)
                    if ie.src_conn is not None and ie.src_conn not in node_map[ie.src].out_connectors:
                        node_map[ie.src].add_out_connector(ie.src_conn)

                # Handle outgoing edges — connect to the parent NSDFG node of the nested SDFG
                for oe in self.map_state.out_edges(map_entry):
                    if (j == len(parent_maps) - 1) and not oe.src_conn.startswith("OUT_"):
                        # This should be now a symbol
                        if oe.src_conn in node_map[oe.src].in_connectors:
                            node_map[oe.src].remove_out_connector(oe.src_conn)
                        continue

                    if oe.data.data is None:
                        continue

                    if oe.dst not in node_map:
                        node_map[oe.dst] = new_map_content_sdfg.parent_nsdfg_node

                    # Ensure parent NSDFG node connectors exist
                    if oe.src_conn is not None and oe.src_conn not in node_map[oe.src].out_connectors:
                        node_map[oe.src].add_out_connector(oe.src_conn)
                    if oe.dst_conn is not None and oe.dst_conn not in node_map[oe.dst].in_connectors:
                        node_map[oe.dst].add_in_connector(oe.dst_conn)

                    # Add the edge to connect map entry → nested SDFG parent
                    state.add_edge(node_map[oe.src], oe.src_conn, node_map[oe.dst], oe.dst_conn, copy.deepcopy(oe.data))

            for j, map_entry in enumerate(reversed(parent_maps)):
                map_exit = self.map_state.exit_node(map_entry)
                for oe in self.map_state.out_edges(map_exit):
                    # Skip dynamic out connectors for the outermost map
                    if j == len(parent_maps) - 1 and (not oe.src_conn.startswith("OUT_")):
                        new_outputs.add(oe.src_conn)
                        continue

                    if oe.data.data is None:
                        continue

                    # Add missing destination access node if needed
                    if oe.dst not in node_map and oe.data.data is not None:
                        node_map[oe.dst] = state.add_access(oe.data.data)

                    # Ensure connectors exist
                    if oe.src_conn is not None and oe.src_conn not in node_map[oe.src].out_connectors:
                        node_map[oe.src].add_out_connector(oe.src_conn)
                    if oe.dst_conn is not None and oe.dst_conn not in node_map[oe.dst].in_connectors:
                        node_map[oe.dst].add_in_connector(oe.dst_conn)

                    # Add the edge (copy the data descriptor to preserve memlet)
                    state.add_edge(node_map[oe.src], oe.src_conn, node_map[oe.dst], oe.dst_conn, copy.deepcopy(oe.data))

                # Handle incoming edges — connect nested SDFG parent node → map exit
                for ie in self.map_state.in_edges(map_exit):
                    if (j == 0) and not ie.dst_conn.startswith("IN_"):
                        # This should now be a symbol
                        if ie.dst_conn in node_map[ie.dst].in_connectors:
                            node_map[ie.dst].remove_in_connector(ie.dst_conn)
                        continue

                    if ie.data.data is None:
                        continue

                    if ie.src not in node_map:
                        node_map[ie.src] = new_map_content_sdfg.parent_nsdfg_node

                    # Ensure connectors exist
                    if ie.src_conn is not None and ie.src_conn not in node_map[ie.src].out_connectors:
                        node_map[ie.src].add_out_connector(ie.src_conn)
                    if ie.dst_conn is not None and ie.dst_conn not in node_map[ie.dst].in_connectors:
                        node_map[ie.dst].add_in_connector(ie.dst_conn)

                    state.add_edge(node_map[ie.src], ie.src_conn, node_map[ie.dst], ie.dst_conn, copy.deepcopy(ie.data))
        # ======================================================================================

        # Dynamic inputs
        for arr_names in [
                new_inputs,
        ]:
            for arr_name in arr_names:
                if arr_name not in new_sdfg.arrays:
                    copydesc = copy.deepcopy(self.map_state.sdfg.arrays[arr_name])
                    copydesc.transient = False
                    assert str(copydesc.dtype) != "void"
                    new_sdfg.add_datadesc(arr_name, copydesc)

        connectors = set(new_inputs).union(set(new_outputs))
        symbols = set(k for k in new_sdfg.free_symbols if k not in connectors)
        missing_symbols = [s for s in symbols if s not in new_outer_symbol_mapping]
        if missing_symbols:
            for ms in missing_symbols:
                new_outer_symbol_mapping[ms] = ms

        new_sdfg.save("ns.sdfg")
        nsdfg2 = self.map_state.add_nested_sdfg(sdfg=new_sdfg,
                                                inputs=set(new_inputs),
                                                outputs=set(new_outputs),
                                                symbol_mapping=new_outer_symbol_mapping)
        new_sdfg.parent_nsdfg_node = nsdfg2

        # Now connect the access nodes of parent map to the nsdfg - copy full subsets
        for ie in self.map_state.in_edges(self.map_entry):
            self.map_state.add_edge(
                ie.src, ie.src_conn, nsdfg2, ie.data.data,
                dace.memlet.Memlet.from_array(ie.data.data, self.map_state.sdfg.arrays[ie.data.data]))

        for oe in self.map_state.out_edges(self.map_state.exit_node(self.map_entry)):
            self.map_state.add_edge(
                nsdfg2, oe.data.data, oe.dst, oe.dst_conn,
                dace.memlet.Memlet.from_array(oe.data.data, self.map_state.sdfg.arrays[oe.data.data]))

        # Remove all map nodes
        nodes = set(self.map_state.all_nodes_between(self.map_entry, self.map_state.exit_node(self.map_entry))).union(
            {self.map_entry, self.map_state.exit_node(self.map_entry)})

        src_nodes = {ie.src for ie in self.map_state.in_edges(self.map_entry)}
        dst_nodes = {oe.dst for oe in self.map_state.out_edges(self.map_state.exit_node(self.map_entry))}

        for arr_name in new_inputs:
            if new_sdfg.arrays[arr_name].dtype != self.map_state.sdfg.arrays[arr_name].dtype:
                new_sdfg.arrays[arr_name].dtype = self.map_state.sdfg.arrays[arr_name].dtype

        if "kfdia" in new_inputs:
            print(new_sdfg.arrays["kfdia"])
            assert str(new_sdfg.arrays["kfdia"].dtype) != "void"
            assert "kfdia" in new_sdfg.arrays

        for node in nodes:
            self.map_state.remove_node(node)

        for src_node in src_nodes:
            assert self.map_state.degree(src_node) != 0
        for dst_node in dst_nodes:
            assert self.map_state.degree(dst_node) != 0

        for i, (body, (state, new_map_content_sdfg)) in enumerate(branch_and_state_sdfg_map.items()):
            nsdfg_node = new_map_content_sdfg.parent_nsdfg_node
            connectors = nsdfg_node.in_connectors | nsdfg_node.out_connectors
            symbols = set(k for k in new_map_content_sdfg.free_symbols if k not in connectors)
            missing_symbols = [s for s in symbols if s not in nsdfg_node.symbol_mapping]
            if missing_symbols:
                raise Exception("uwu")

        nsdfg_node = new_sdfg.parent_nsdfg_node
        connectors = nsdfg_node.in_connectors | nsdfg_node.out_connectors
        symbols = set(k for k in new_sdfg.free_symbols if k not in connectors)
        missing_symbols = [s for s in symbols if s not in nsdfg_node.symbol_mapping]
        if missing_symbols:
            raise Exception("uwu2")

        self.map_state.sdfg.save("applied.sdfgz", compress=True)
