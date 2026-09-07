# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.

from ordered_set import OrderedSet

from collections import deque
from copy import deepcopy
from dace import dtypes, Memlet
from dace.sdfg import nodes, SDFG
from dace.sdfg.state import SDFGState
from typing import Dict, Tuple

import dace.transformation.passes.offloading.offloading_helpers as helpers


class SingleIterationMapPhase():

    def apply(self, sdfg: SDFG, hybrid_states: OrderedSet, verbose=False):
        self.verbose = verbose
        if verbose: print("hybrid:", hybrid_states)
        for state in hybrid_states:
            self.make_size1_map_wrappers(sdfg, state)

    ########################
    ### Wrapping Helpers ###
    ########################

    def _get_root_nodes(self, state: SDFGState, bounded_set: OrderedSet):
        return OrderedSet(node for node in bounded_set if state.in_degree(node) == 0)

    def _get_leaf_nodes(self, state: SDFGState, bounded_set: OrderedSet):
        return OrderedSet(node for node in bounded_set if state.out_degree(node) == 0)

    def _get_boundary_in_edges(self, state: SDFGState, node, bounded_set: OrderedSet):
        return OrderedSet(e for e in state.in_edges(node) if e.src not in bounded_set)

    def _get_boundary_out_edges(self, state: SDFGState, node, bounded_set: OrderedSet):
        return OrderedSet(e for e in state.out_edges(node) if e.dst not in bounded_set)

    def _get_entry_nodes(self, state: SDFGState, bounded_set: OrderedSet):
        return OrderedSet(node for node in bounded_set if all(e.src not in bounded_set for e in state.in_edges(node)))

    def _get_exit_nodes(self, state: SDFGState, bounded_set: OrderedSet):
        return OrderedSet(node for node in bounded_set if all(e.dst not in bounded_set for e in state.out_edges(node)))

    ######################
    ### Wrapping Logic ###
    ######################

    def make_size1_map_wrappers(self, sdfg: SDFG, state: SDFGState):
        # top level GPU nodes partition the graph
        lib_nodes = OrderedSet(node for node in state.scope_children()[None]
                               if isinstance(node, (nodes.LibraryNode)) and helpers.has_GPU_schedule(node))
        map_entries = OrderedSet(node for node in state.scope_children()[None]
                                 if isinstance(node, (nodes.MapEntry)) and helpers.has_GPU_schedule(node))
        map_exits = OrderedSet(state.exit_node(node) for node in map_entries)

        # A callback can only run on the host, so it bounds a partition the same way a kernel does
        # rather than being swept into one.
        callbacks = OrderedSet(node for node in state.scope_children()[None] if helpers.is_callback_tasklet(node, sdfg))

        partition_nodes = lib_nodes | map_entries | map_exits | callbacks
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

            # A partition is a dataflow component, and a map scope spans one, so the partitioning
            # can hand this a lone MapEntry whose body and exit went to another component.
            partition = self.scope_closed_partition(state, partition, partition_nodes)
            if partition is None:
                continue

            # if anything is left, wrap it
            if partition:
                map_entry, map_exit = self.wrap_region_in_size1_map(state, partition)

                # Avoid illegal direct map-to-map connections by routing through an access node.
                self.insert_access_between_adjacent_maps(state, map_exit)

                # Ensure all map inputs are also outputs to avoid dace erroneusly labeling them as constants
                self.forward_input_only_map_data(state, map_entry, map_exit)

    def scope_closed_partition(self, state: SDFGState, region: OrderedSet, boundary: OrderedSet) -> OrderedSet:
        """``region`` grown until every map scope it touches lies wholly inside it, or None.

        A size-1 map around HALF a scope puts a map entry inside the new map and its own exit
        outside it, which is not a scope at all: ``entry_node`` then answers for the wrapping map,
        and validation reports the pair as Map objects that were copied separately.

        None when closing would have to swallow one of the nodes the partitioning deliberately kept
        out -- a GPU map or a device-wide library call. Those are the boundaries the partition
        exists to respect, so the answer there is to leave this group alone rather than to wrap a
        region that reaches across one.
        """
        closed: OrderedSet = OrderedSet(region)
        scope_children = state.scope_children()
        queue = list(region)
        while queue:
            node = queue.pop()
            if isinstance(node, nodes.MapEntry):
                entry = node
            elif isinstance(node, nodes.MapExit):
                entry = state.entry_node(node)
            else:
                continue
            if entry is None:
                continue
            for extra in [entry, state.exit_node(entry), *scope_children[entry]]:
                if extra is None or extra in closed:
                    continue
                if extra in boundary:
                    return None
                closed.add(extra)
                queue.append(extra)
        return closed

    def wrap_region_in_size1_map(self, state: SDFGState,
                                 region_nodes: OrderedSet) -> Tuple[nodes.MapEntry, nodes.MapExit]:
        if not region_nodes: return
        map_label, map_param = helpers.get_new_map_identifiers(state, "size1_wrap_region", "__wrap_i")
        map_entry, map_exit = state.add_map(name=map_label,
                                            ndrange={map_param: '0:1'},
                                            schedule=dtypes.ScheduleType.GPU_Device)

        # make MAP ENTRY
        # A list, not a set: the connector numbering below follows this order, so a set would make
        # the emitted names depend on PYTHONHASHSEED.
        boundary_in_edges = []
        for node in region_nodes:
            boundary_in_edges += list(self._get_boundary_in_edges(state, node, region_nodes))

        idx = 0
        for edge in boundary_in_edges:
            src, src_conn, dst, dst_conn = edge.src, edge.src_conn, edge.dst, edge.dst_conn
            ext_memlet = deepcopy(edge.data)
            int_memlet = deepcopy(edge.data)
            state.remove_edge(edge)

            # An empty memlet ORDERS; it carries no data and so may not carry a connector.
            # infer_connector_types otherwise reaches for out_connectors[None] and raises KeyError.
            if ext_memlet.is_empty():
                state.add_nedge(src, map_entry, ext_memlet)
                state.add_nedge(map_entry, dst, int_memlet)
                continue

            in_conn = f"IN_REGION_IN_{idx}"
            out_conn = f"OUT_REGION_IN_{idx}"
            map_entry.add_in_connector(in_conn)
            map_entry.add_out_connector(out_conn)
            idx += 1

            state.add_edge(src, src_conn, map_entry, in_conn, ext_memlet)
            state.add_edge(map_entry, out_conn, dst, dst_conn, int_memlet)

        # A region root the rewiring did not reach reads nothing, so no edge put it under the entry
        # and the scope does not contain it -- while whatever it feeds does, which is the invalid
        # inside-to-outside path. Order it after the entry instead. Rewiring a boundary edge does
        # not make the OTHER roots any less dangling, so this cannot be an else-branch of it.
        for node in region_nodes:
            if state.in_degree(node) == 0:
                state.add_nedge(map_entry, node, Memlet())

        # make MAP EXIT
        boundary_out_edges = []
        for node in region_nodes:
            boundary_out_edges += list(self._get_boundary_out_edges(state, node, region_nodes))

        idx = 0
        for edge in boundary_out_edges:
            src, src_conn, dst, dst_conn = edge.src, edge.src_conn, edge.dst, edge.dst_conn
            int_memlet = deepcopy(edge.data)
            ext_memlet = deepcopy(edge.data)
            state.remove_edge(edge)

            if int_memlet.is_empty():  # ordering edge, see above
                state.add_nedge(src, map_exit, int_memlet)
                state.add_nedge(map_exit, dst, ext_memlet)
                continue

            in_conn = f"IN_REGION_OUT_{idx}"
            out_conn = f"OUT_REGION_OUT_{idx}"
            map_exit.add_in_connector(in_conn)
            map_exit.add_out_connector(out_conn)
            idx += 1

            state.add_edge(src, src_conn, map_exit, in_conn, int_memlet)
            state.add_edge(map_exit, out_conn, dst, dst_conn, ext_memlet)

        # The leaf half of the same rule.
        for node in region_nodes:
            if state.out_degree(node) == 0:
                state.add_nedge(node, map_exit, Memlet())

        return map_entry, map_exit

    ######################
    ### Parition Logic ###
    ######################

    def subgraphs_after_removing_partition_nodes(self, state: SDFGState,
                                                 partition_nodes: OrderedSet) -> list[OrderedSet[nodes.Node]]:
        """
        Returns connected components (as sets of nodes) when treating partition_nodes as deleted from state.
        Connectivity is treated as undirected (uses both in/out edges).
        """
        visited = OrderedSet()
        components = []
        remaining_nodes = [n for n in state.scope_children()[None] if n not in partition_nodes]  # top level nodes only

        for start in remaining_nodes:
            if start in visited:
                continue

            comp = OrderedSet()
            queue = deque([start])
            visited.add(start)

            while queue:
                u = queue.popleft()
                comp.add(u)

                neighbors = OrderedSet(e.dst for e in state.out_edges(u)) | OrderedSet(e.src for e in state.in_edges(u))
                for v in neighbors:
                    if v in partition_nodes or v in visited:
                        continue
                    visited.add(v)
                    queue.append(v)

            components.append(comp)

        return components

    def remove_all_outer_access_nodes_from_group(self, state: SDFGState, group: OrderedSet):
        outer_nodes = self._get_entry_nodes(state, group) | self._get_exit_nodes(state, group)
        nodes_to_remove = OrderedSet(node for node in outer_nodes if isinstance(node, nodes.AccessNode))

        while nodes_to_remove:
            group -= nodes_to_remove
            outer_nodes = self._get_entry_nodes(state, group) | self._get_exit_nodes(state, group)
            nodes_to_remove = OrderedSet(node for node in outer_nodes if isinstance(node, nodes.AccessNode))

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
        input_only_data = [
            memlet.data for memlet in input_memlets
            if all(edge.data is None or edge.data.is_empty() or edge.data.data != memlet.data
                   for edge in state.out_edges(map_exit))
        ]
        # find last accesses(ignore data without accesses)
        # INV: dictionary holds ONLY data which goes into the map, is accessed within but does not exit -> if left unchanged this would be detected as a constant and lead to errors
        last_accesses: Dict = self._find_last_access_nodes_in_map_bfs(state, map_entry, map_exit, input_only_data)

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

    def _find_last_access_nodes_in_map_bfs(self, state: SDFGState, map_entry: nodes.MapEntry, map_exit: nodes.MapExit,
                                           data_names: OrderedSet[str]) -> dict[str, nodes.AccessNode]:
        if not data_names: return {}
        last_access: dict[str, nodes.AccessNode] = {}
        queue = deque([map_entry])
        visited = OrderedSet([map_entry])

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
