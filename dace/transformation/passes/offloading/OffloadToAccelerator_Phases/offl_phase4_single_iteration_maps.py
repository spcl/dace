

from collections import deque
from copy import deepcopy
from dace import dtypes, Memlet
from dace.sdfg import nodes, SDFG
from dace.sdfg.state import SDFGState
from typing import Dict, Tuple

import dace.transformation.passes.offloading.OffloadToAccelerator_Phases.offloading_helpers as helpers

class SingleIterationMapPhase():

    def apply(self, sdfg:SDFG, hybrid_states:set, verbose=False):
        if verbose: print("hybrid:", hybrid_states)
        for state in hybrid_states:
            self.make_size1_map_wrappers(sdfg, state)



    ########################
    ### Wrapping Helpers ###
    ########################

    def _get_root_nodes(self, state:SDFGState, bounded_set:set):
        return {node for node in bounded_set if state.in_degree(node) == 0}

    def _get_leaf_nodes(self, state:SDFGState, bounded_set:set):
        return {node for node in bounded_set if state.out_degree(node) == 0}

    def _get_boundary_in_edges(self, state:SDFGState, node, bounded_set:set):
        return {e for e in state.in_edges(node) if e.src not in bounded_set}

    def _get_boundary_out_edges(self, state:SDFGState, node, bounded_set:set):
        return {e for e in state.out_edges(node) if e.dst not in bounded_set}
    
    def _get_entry_nodes(self, state:SDFGState, bounded_set:set):
        return {node for node in bounded_set
        if all(e.src not in bounded_set for e in state.in_edges(node))
    }

    def _get_exit_nodes(self, state:SDFGState, bounded_set:set):
        return {node for node in bounded_set
        if all(e.dst not in bounded_set for e in state.out_edges(node))
    }


    ######################
    ### Wrapping Logic ###
    ######################

    def make_size1_map_wrappers(self, sdfg:SDFG, state:SDFGState):
        # top level GPU nodes partition the graph
        lib_nodes = { node for node in state.scope_children()[None] if isinstance(node, (nodes.LibraryNode)) and helpers.has_GPU_schedule(node)}
        map_entries = { node for node in state.scope_children()[None] if isinstance(node, (nodes.MapEntry)) and helpers.has_GPU_schedule(node)}
        map_exits = {state.exit_node(node) for node in map_entries}

        partition_nodes = lib_nodes | map_entries | map_exits
        partitions = self.subgraphs_after_removing_partition_nodes(state, partition_nodes)

        # each partition is wrapped into a map
        for partition in partitions:

            # if only scalars are accessed, then no wrap is needed
            array_access = False
            for node in partition:
                if isinstance(node, nodes.AccessNode) and node.data:
                    if not helpers.is_scalar(node.data, sdfg):
                        array_access = True
                        break
            if not array_access:
                continue

            # reduce partition to nodes which need to go into wrap
            self.remove_all_outer_access_nodes_from_group(state, partition)

            # if anything is left, wrap it
            if partition:
                map_entry, map_exit = self.wrap_region_in_size1_map(state, partition)

                # Avoid illegal direct map-to-map connections by routing through an access node.
                self.insert_access_between_adjacent_maps(state, map_exit)

                # Ensure all map inputs are also outputs to avoid dace erroneusly labeling them as constants
                self.forward_input_only_map_data(state, map_entry, map_exit)

    
    def wrap_region_in_size1_map(self, state:SDFGState, region_nodes:set) -> Tuple[nodes.MapEntry, nodes.MapExit]:
        if not region_nodes: return
        map_label, map_param = helpers.get_new_map_identifiers(state, "size1_wrap_region", "__wrap_i")
        map_entry, map_exit = state.add_map(name=map_label, ndrange={map_param: '0:1'}, schedule = dtypes.ScheduleType.GPU_Device)

        # make MAP ENTRY
        boundary_in_edges = set()
        for node in region_nodes:
            boundary_in_edges |= self._get_boundary_in_edges(state, node, region_nodes)

        idx = 0
        if not boundary_in_edges: # if there are no boundary in edges, add new dependcy edges
            root_nodes = self._get_root_nodes(state, region_nodes)
            assert root_nodes, f"region: {region_nodes}"
            for root in root_nodes:
                state.add_nedge(map_entry, root, Memlet())
        
        else: # if there are, rewire them through the map
            for idx, edge in enumerate(boundary_in_edges):
                src, src_conn, dst, dst_conn = edge.src, edge.src_conn, edge.dst, edge.dst_conn
                ext_memlet = deepcopy(edge.data)
                int_memlet = deepcopy(edge.data)
                state.remove_edge(edge)

                in_conn = f"IN_REGION_IN_{idx}"
                out_conn = f"OUT_REGION_IN_{idx}"
                map_entry.add_in_connector(in_conn)
                map_entry.add_out_connector(out_conn)

                state.add_edge(src, src_conn, map_entry, in_conn, ext_memlet)
                state.add_edge(map_entry, out_conn, dst, dst_conn, int_memlet)

        # make MAP EXIT
        boundary_out_edges = set()
        for node in region_nodes:
            boundary_out_edges |= self._get_boundary_out_edges(state, node, region_nodes)
        
        if not boundary_out_edges: # add new dependency edges
            leaf_nodes = self._get_leaf_nodes(state, region_nodes)
            assert leaf_nodes
            for leaf in leaf_nodes:
                state.add_nedge(leaf, map_exit, Memlet())
            
        else: # rewire out edges
            for idx, edge in enumerate(boundary_out_edges):
                src, src_conn, dst, dst_conn = edge.src, edge.src_conn, edge.dst, edge.dst_conn
                int_memlet = deepcopy(edge.data)
                ext_memlet = deepcopy(edge.data)
                state.remove_edge(edge)

                in_conn = f"IN_REGION_OUT_{idx}"
                out_conn = f"OUT_REGION_OUT_{idx}"
                map_exit.add_in_connector(in_conn)
                map_exit.add_out_connector(out_conn)

                state.add_edge(src, src_conn, map_exit, in_conn, int_memlet)
                state.add_edge(map_exit, out_conn, dst, dst_conn, ext_memlet)

        return map_entry, map_exit
    

    ######################
    ### Parition Logic ###
    ######################

    def subgraphs_after_removing_partition_nodes(self, state: SDFGState, partition_nodes: set) -> list[set[nodes.Node]]:
        """
        Returns connected components (as sets of nodes) when treating partition_nodes as deleted from state.
        Connectivity is treated as undirected (uses both in/out edges).
        """
        visited = set()
        components = []
        remaining_nodes = [n for n in state.scope_children()[None] if n not in partition_nodes] # top level nodes only

        for start in remaining_nodes:
            if start in visited:
                continue

            comp = set()
            queue = deque([start])
            visited.add(start)

            while queue:
                u = queue.popleft()
                comp.add(u)

                neighbors = {e.dst for e in state.out_edges(u)} | {e.src for e in state.in_edges(u)}
                for v in neighbors:
                    if v in partition_nodes or v in visited:
                        continue
                    visited.add(v)
                    queue.append(v)

            components.append(comp)

        return components

    def remove_all_outer_access_nodes_from_group(self, state:SDFGState, group:set):
        outer_nodes = self._get_entry_nodes(state, group) | self._get_exit_nodes(state, group)
        nodes_to_remove = {node for node in outer_nodes if isinstance(node, nodes.AccessNode)}

        while nodes_to_remove:
            group -= nodes_to_remove
            outer_nodes = self._get_entry_nodes(state, group) | self._get_exit_nodes(state, group)
            nodes_to_remove = {node for node in outer_nodes if isinstance(node, nodes.AccessNode)}


    ###############################
    ### Clean Up After Wrapping ###
    ###############################

    def insert_access_between_adjacent_maps(self, state: SDFGState, map_exit: nodes.MapExit) -> None:
        # avoid illegal direct map-to-map connections by routing through an access node.
        for edge in list(state.out_edges(map_exit)):
            if not isinstance(edge.dst, nodes.MapEntry):
                continue
            if edge.data is None or edge.data.is_empty() or edge.data.data is None:
                continue

            src, src_conn, dst, dst_conn = edge.src, edge.src_conn, edge.dst, edge.dst_conn
            access = state.add_access(edge.data.data)
            out_memlet = deepcopy(edge.data)
            in_memlet = deepcopy(edge.data)

            state.remove_edge(edge)
            state.add_edge(src, src_conn, access, None, out_memlet)
            state.add_edge(access, None, dst, dst_conn, in_memlet)

    def forward_input_only_map_data(self, state: SDFGState, map_entry: nodes.MapEntry, map_exit: nodes.MapExit) -> None:
        # For map inputs that are not map outputs, route final in-map access through map_exit
        # -> Ensure all map inputs are also outputs to avoid dace erroneusly labeling them as constants
        
        # get inputs & isolate those without corresponding outputs
        input_memlets = [
            edge.data for edge in state.in_edges(map_entry)
            if edge.data is not None and not edge.data.is_empty() and edge.data.data is not None
        ]
        input_only_data = [memlet.data for memlet in input_memlets
            if all(
                edge.data is None or edge.data.is_empty() or edge.data.data != memlet.data
                for edge in state.out_edges(map_exit)
            )
        ]
        # find last accesses(ignore data without accesses)
        # INV: dictionary holds ONLY data which goes into the map, is accessed within but does not exit -> if left unchanged this would be detected as a constant and lead to errors
        last_accesses : Dict = self._find_last_access_nodes_in_map_bfs(state, map_entry, map_exit, input_only_data)

        # wire the last access through map_exit to a new outside access node
        for input_memlet in input_memlets:
            data_name = input_memlet.data
            if not data_name in last_accesses:
                continue
            last_access = last_accesses[data_name]
            
            # create unique connectors
            connector_index = 0
            while (f"IN_INPUT_ONLY_{connector_index}" in map_exit.in_connectors
                    or f"OUT_INPUT_ONLY_{connector_index}" in map_exit.out_connectors):
                connector_index += 1
            in_conn = f"IN_INPUT_ONLY_{connector_index}"
            out_conn = f"OUT_INPUT_ONLY_{connector_index}"

            # add new external access node & edges to it
            map_exit.add_in_connector(in_conn)
            map_exit.add_out_connector(out_conn)
            outside_access = state.add_access(data_name)
            internal_memlet = deepcopy(input_memlet)
            external_memlet = deepcopy(input_memlet)
            state.add_edge(last_access, None, map_exit, in_conn, internal_memlet)
            state.add_edge(map_exit, out_conn, outside_access, None, external_memlet)

    def _find_last_access_nodes_in_map_bfs(self, state: SDFGState, map_entry: nodes.MapEntry, map_exit: nodes.MapExit, data_names: set[str]) -> dict[str, nodes.AccessNode]:
        if not data_names: return {}
        last_access: dict[str, nodes.AccessNode] = {}
        queue = deque([map_entry])
        visited = {map_entry}

        while queue:
            node = queue.popleft()

            if isinstance(node, nodes.AccessNode) and node.data in data_names:
                last_access[node.data] = node

            if node is map_exit:
                continue

            for edge in state.out_edges(node):
                child = edge.dst
                if not child or child in visited:
                    continue
                visited.add(child)
                queue.append(child)

        return last_access