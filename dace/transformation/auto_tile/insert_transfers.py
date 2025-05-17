# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.
import ast
import copy
import re
from typing import Callable
import typing
import dace
from dace.data import EnumProperty, ListProperty
from dace.properties import DictProperty, Property, SetProperty, make_properties
from dace.sdfg import SDFG, SDFGState
from dace.sdfg import nodes
from dace.sdfg import utils as sdutil
from dace.sdfg.graph import MultiConnectorEdge
from dace.transformation import transformation
import json
import networkx as nx
import operator

from dace.transformation.auto_tile.explicit_memory_move import ExplicitMemoryMove
from dace.transformation import pass_pipeline as ppl, transformation


@make_properties
class InsertTransfers(ppl.Pass):
    device_map_entry = transformation.PatternNode(nodes.MapEntry)

    str_entry_location_requirements = Property(dtype=str, default="", allow_none=False)
    str_exit_location_requirements = Property(dtype=str, default="", allow_none=False)
    str_movement_graph = Property(dtype=str, default="", allow_none=False)
    dev_entry_type = EnumProperty(
        dtype=dace.dtypes.ScheduleType,
        default=dace.dtypes.ScheduleType.Default,
        allow_none=False,
    )
    str_computational_unit_register_locations = Property(
        dtype=str, default="", allow_none=False
    )
    str_input_output_types = Property(dtype=str, default="", allow_none=False)

    unspecialized_locations = ListProperty(element_type=str, default=[], allow_none=False)

    def __init__(
        self,
        movement_graph,
        entry_location_requirements,
        exit_location_requirements,
        computational_unit_register_locations,
        input_output_types,
        unspecialized_locations,
        dev_entry_type,
    ):
        super().__init__()
        self.str_movement_graph = movement_graph
        self.str_entry_location_requirements = entry_location_requirements
        self.str_exit_location_requirements = exit_location_requirements
        self.dev_entry_type = dev_entry_type
        self.str_computational_unit_register_locations = (
            computational_unit_register_locations
        )
        self.str_input_output_types = input_output_types
        self.unspecialized_locations = unspecialized_locations
        assert (
            entry_location_requirements is not None
            and entry_location_requirements != ""
        )
        assert self.str_movement_graph is not None and self.str_movement_graph != ""

    def deserialize_inputs(self):
        adj_G = json.loads(self.str_movement_graph)
        entry_location_requirements = json.loads(self.str_entry_location_requirements)
        exit_location_requirements = json.loads(self.str_exit_location_requirements)

        G = nx.DiGraph()
        nodes = adj_G.keys()

        finf = float("inf")
        for u in nodes:
            for v in nodes:
                if adj_G[u][v] != finf and u != v:
                    G.add_edge(u, v, weight=adj_G[u][v])

        # Compute shortest paths and distances
        predecessors, distances = nx.floyd_warshall_predecessor_and_distance(G)

        # Print shortest distances
        print("\nShortest distances:")
        for u in nodes:
            for v in nodes:
                print(f"{distances.get(u, {}).get(v, 'inf'):7}", end=" ")
            print()

        # Print all shortest paths
        print("\nAll shortest paths:")
        _paths = []
        for u in nodes:
            for v in nodes:
                if u != v:
                    try:
                        path = nx.reconstruct_path(u, v, predecessors)
                        print(f"Path from {u} to {v}: {path}")
                        _paths.append(path)
                    except nx.NetworkXNoPath:
                        print(f"Path from {u} to {v}: No path")

        self._G = G
        self._entry_location_requirements = entry_location_requirements
        self._exit_location_requirements = exit_location_requirements

        self._paths = _paths

        self._computational_unit_register_locations = json.loads(
            self.str_computational_unit_register_locations
        )
        self._computational_units = list(
            self._computational_unit_register_locations.keys()
        )

        __input_output_types = json.loads(self.str_input_output_types)
        self._input_output_types = {}
        for k, v in __input_output_types.items():
            kk = k.split("_AND_")
            kk.sort()
            self._input_output_types[tuple(kk)] = v

    def apply_pass(self, sdfg: SDFG, pipeline_results: typing.List[typing.Any]):
        self.deserialize_inputs()

        for state in sdfg.states():
            for node in state.nodes():
                if (
                    isinstance(node, nodes.MapEntry)
                    and node.map.schedule == self.dev_entry_type
                ):
                    self._apply(state, sdfg, node)

    def _get_first_required_computational_unit(
        self, sdfg: SDFG, state: SDFGState, node: nodes.AccessNode
    ):
        scope_dict = state.scope_dict()
        for oe in state.out_edges(node):
            print(node, state.memlet_path(oe))
            for _e in state.memlet_path(oe):
                # print(scope_dict[_e.dst].computational_units)
                for unit, dist in scope_dict[_e.dst].computational_units.items():
                    if dist == 0:
                        return unit
        raise Exception(
            f"No computational units found for scope node: {node}. Register storage without using it, I dont know how to handle it"
        )

    def _specialize_register_storage(
        self, sdfg: SDFG, state: SDFGState, device_map_entry: nodes.MapEntry
    ):
        all_nodes = state.bfs_nodes(device_map_entry)

        scope_dict = state.scope_dict()
        for node in all_nodes:
            if isinstance(node, dace.nodes.AccessNode):
                arr = sdfg.arrays[node.data]
                if arr.storage == dace.dtypes.StorageType.Register or arr.storage == dace.dtypes.StorageType.Default:
                    if scope_dict[node] is not None and hasattr(
                        scope_dict[node], "computational_units"
                    ):
                        first_used_comp_unit = (
                            self._get_first_required_computational_unit(
                                sdfg, state, node
                            )
                        )
                        storage = self._computational_unit_register_locations[
                            first_used_comp_unit
                        ]
                        arr.storage = self._get_enum_from_ser_string(storage)
                        if arr.storage == dace.dtypes.StorageType.Ascend_VECIN:
                            raise Exception(f"Waht {node} to {storage}")
                    else:
                        raise Exception(
                            f"No computational units found for scope node: {node}. Register storage without a scope is not supported"
                        )

    def _get_enum_from_ser_string(self, storage_str):
        return dace.dtypes.StorageType[storage_str.split("@")[-1].split(".")[-1]]

    def _construct_str_from_storage(self, storage):
        s = str(storage)
        prefix = ""
        if s.endswith("2"):
            prefix = str(dace.dtypes.StorageType.Ascend_L2)
        if s.endswith("1"):
            prefix = str(dace.dtypes.StorageType.Ascend_L1)
        name = prefix + "@" + s if prefix != "" else s
        if prefix in self.unspecialized_locations and s == prefix:
            return s
        return name

    def _copy_arr_to_loc(
        self,
        src_arr_name: str,
        src_arr: dace.data.Data,
        dst_loc: dace.dtypes.StorageType,
        dst_loc_prefix: str,
        sdfg: dace.SDFG,
    ):
        dst_arr = copy.deepcopy(src_arr)
        prefixes = [str(st).split(".")[-1] for st in list(dace.dtypes.StorageType)]
        dst_name = None
        for prefix in prefixes:
            if src_arr_name.startswith(prefix):
                dst_name = src_arr_name.replace(prefix, dst_loc_prefix)
                break
        if dst_name is None:
            dst_name = dst_loc_prefix + "_" + src_arr_name
        if dst_name not in sdfg.arrays:
            dst_arr.name = dst_name
            dst_arr.storage = dst_loc
            dst_arr.transient = src_arr.transient
            dst_arr.shape = src_arr.shape
            print("DD", dst_name, type(dst_name))
            sdfg.add_datadesc(dst_name, dst_arr, find_new_name=False)

        return dst_name, dst_arr

    def _insert_access_node(
        self,
        shortest_path,
        edge: MultiConnectorEdge[dace.memlet.Memlet],
        sdfg: dace.SDFG,
        state: dace.SDFGState,
    ):
        edges_to_add = set()
        edges_to_rm = set()

        def _insert(src_edge):
            (
                src_edge_src,
                src_edge_src_conn,
                src_edge_dst,
                src_edge_dst_conn,
                src_edge_data,
            ) = src_edge
            src_loc_str = shortest_path[i - 1]
            dst_loc_str = shortest_path[i]
            dst_loc_str = dst_loc_str.split(".")[-1]
            src_loc_str = src_loc_str.split(".")[-1]
            src_loc = dace.dtypes.StorageType[src_loc_str]
            dst_loc = dace.dtypes.StorageType[dst_loc_str]
            src_arr = sdfg.arrays[src_edge_data.data]
            memlet = src_edge_data
            memlet.subset = copy.deepcopy(memlet.subset)
            dst_arr = copy.deepcopy(src_arr)
            prefixes = [str(st).split(".")[-1] for st in list(dace.dtypes.StorageType)]
            dst_name = None
            for prefix in prefixes:
                if src_edge_data.data.startswith(prefix):
                    dst_name = src_edge_data.data.replace(prefix, dst_loc_str)
                    break
            if dst_name is None:
                dst_name = dst_loc_str + "_" + src_edge_data.data
            # if dst_name is None:
            #    raise Exception("Could not find a suitable prefix for the new array")
            if dst_name not in sdfg.arrays:
                dst_arr.name = dst_name
                dst_arr.storage = dst_loc
                dst_arr.transient = True
                dst_arr.shape = (
                    [(end + 1 - beg) // step for beg, end, step in memlet.subset]
                    if memlet.subset is not None
                    else src_arr.shape
                )
                # print("DD", dst_name, type(dst_name))
                sdfg.add_datadesc(dst_name, dst_arr, find_new_name=False)

            dst_arr_node = state.add_access(dst_name)
            if i > 1:
                edges_to_add.remove(
                    (
                        src_edge_src,
                        src_edge_src_conn,
                        src_edge_dst,
                        src_edge_dst_conn,
                        src_edge_data,
                    )
                )
            else:
                pass

            edges_to_add.add(
                (
                    src_edge_src,
                    src_edge_src_conn,
                    dst_arr_node,
                    None,
                    copy.deepcopy(memlet),
                )
            )
            m = dace.memlet.Memlet(expr=dst_name)
            edges_to_add.add((dst_arr_node, None, src_edge_dst, src_edge_dst_conn, m))
            return (dst_arr_node, None, src_edge_dst, src_edge_dst_conn, m)

        src_edge = (edge.src, edge.src_conn, edge.dst, edge.dst_conn, edge.data)
        for i in range(1, len(shortest_path)):
            e = _insert(src_edge)
            # print(shortest_path[i-1], shortest_path[i], e)
            src_edge = e

        # if len(edges_to_add) > 2:
        #    raise Exception(edge)

        # state.remove_edge(edge)

        src_memlet = edge.data
        new_memlet = src_edge[-1]
        offset = [beg for beg, end, step in src_memlet.subset]
        # We need to replace all occurences within the map
        assert isinstance(edge.dst, dace.nodes.MapEntry)
        # do not include map entry and exit
        all_nodes = list(state.all_nodes_between(edge.dst, state.exit_node(edge.dst)))
        print(edge.dst, state.exit_node(edge.dst))
        print(all_nodes)
        for e in state.all_edges(*all_nodes):
            # print(e)
            # If data is what we replaced, replace
            # print(e.data.data, src_memlet.data)
            if e.data.data == src_memlet.data:
                _tmp_memlet = dace.subsets.Range(
                    [
                        (beg - offset, end - offset, step)
                        for ((beg, end, step), offset) in zip(e.data.subset, offset)
                    ]
                )
                e.data = dace.memlet.Memlet(
                    data=new_memlet.data, subset=_tmp_memlet, wcr=e.data.wcr
                )

        for n in list(state.all_nodes_between(edge.dst, state.exit_node(edge.dst))) + [
            edge.dst,
            state.exit_node(edge.dst),
        ]:
            print(n.in_connectors, "IN_" + edge.data.data)
            print(n.out_connectors, "OUT_" + edge.data.data)

            for ie in state.in_edges_by_connector(n, "IN_" + edge.data.data):
                ie.dst_conn = "IN_" + new_memlet.data
            for oe in state.out_edges_by_connector(n, "OUT_" + edge.data.data):
                oe.src_conn = "OUT_" + new_memlet.data

            for ea in list(
                edges_to_add
            ):  # Convert to list to avoid modifying the set during iteration
                print("EA", ea[3], edge.data.data, new_memlet.data)
                if ea[2] == n or ea[0] == n:
                    _ea = None
                    if ea[3] == "IN_" + edge.data.data:
                        _ea = (ea[0], ea[1], ea[2], "IN_" + new_memlet.data, ea[4])
                    if ea[1] == "OUT_" + edge.data.data:
                        _ea = (ea[0], "OUT_" + new_memlet.data, ea[2], ea[3], ea[4])
                    if _ea is not None:
                        edges_to_add.discard(ea)  # Remove old entry
                        edges_to_add.add(_ea)  # Add updated entry

            if "IN_" + edge.data.data in n.in_connectors:
                v = n.in_connectors["IN_" + edge.data.data]
                n.remove_in_connector("IN_" + edge.data.data)
                n.add_in_connector("IN_" + new_memlet.data, v)
            if "OUT_" + edge.data.data in n.out_connectors:
                v = n.out_connectors["OUT_" + edge.data.data]
                n.remove_out_connector("OUT_" + edge.data.data)
                n.add_out_connector("OUT_" + new_memlet.data, v)

        # raise Exception("Not implemented")

        # edges_to_rm.add(edge)
        # print(edge.src, edge.dst)
        # for e in state.edges():
        #    if e.src == edge.src and edge.dst == e.dst and edge.data == e.data and edge.src_conn == e.src_conn and edge.dst_conn == e.dst_conn:
        #        edges_to_rm.add(e)
        edges_to_rm.add(edge)
        return edges_to_add, edges_to_rm

    def _replace_tasklet_outputs(
        self, state: SDFGState, sdfg: SDFG, device_map_entry: nodes.MapEntry
    ):
        edges_to_add = set()
        edges_to_rm = set()
        for n in list(
            state.all_nodes_between(device_map_entry, state.exit_node(device_map_entry))
        ) + [
            device_map_entry,
            state.exit_node(device_map_entry),
        ]:
            if isinstance(n, dace.nodes.Tasklet):
                itypes = set()
                for ie in state.in_edges(n):
                    print(ie.data.data)
                    arr = sdfg.arrays[ie.data.data]
                    storage_str = self._construct_str_from_storage(arr.storage)
                    itypes.add(storage_str)
                itypes = list(itypes)
                itypes.sort()
                print(itypes, self._input_output_types)
                # assert tuple(itypes) in self._input_output_types
                if tuple(itypes) in self._input_output_types:
                    otype = self._input_output_types[tuple(itypes)]
                    for oe in state.out_edges(n):
                        arr = sdfg.arrays[oe.data.data]
                        in_storage_str = self._construct_str_from_storage(arr.storage)
                        in_storage_type = self._get_enum_from_ser_string(otype)
                        out_storage_str = self._input_output_types[tuple(itypes)].split(
                            "."
                        )[-1]
                        out_storage_type = self._get_enum_from_ser_string(
                            out_storage_str
                        )
                        if arr.storage != out_storage_type:
                            dst_arr_name, dst_arr = self._copy_arr_to_loc(
                                oe.data.data,
                                arr,
                                out_storage_type,
                                out_storage_str,
                                sdfg,
                            )
                            old_data_name = oe.data.data
                            new_data_name = dst_arr_name
                            oe.data.data = dst_arr_name
                            # need to propagate until the next map exit
                            self._replace_in_connectors_and_memlet_data(
                                n, old_data_name, new_data_name, state
                            )

                            # We need to add a copy to the out edge
                            map_exit = state.exit_node(state.entry_node(n))
                            for map_out in state.out_edges(map_exit):
                                print("AB", map_out.data.data,
                                      old_data_name, new_data_name,
                                      oe.data.data, map_out.src_conn[4:])
                                # RM OUT_
                                if map_out.src_conn[4:] == new_data_name:
                                    #raise Exception("Not implemented")
                                    new_memlet = dace.memlet.Memlet(expr=new_data_name)
                                    an = state.add_access(new_data_name)
                                    #an2 = state.add_access(map_out.data.data)
                                    edges_to_add.add((map_out.src, map_out.src_conn,
                                                     an, None, new_memlet))
                                    edges_to_add.add((an, None,
                                                     map_out.dst, map_out.dst_conn,
                                                     copy.deepcopy(map_out.data)))
                                    #edges_to_add.add((an2, None,
                                    #                 map_out.dst, map_out.dst_conn,
                                    #                 copy.deepcopy(map_out.data)))
                                    edges_to_rm.add(map_out)


        for e in edges_to_add:
            state.add_edge(*e)
        for e in edges_to_rm:
            state.remove_edge(e)

    def _replace_in_connectors_and_memlet_data(
        self,
        node: dace.nodes.Node,
        old_data_name: str,
        new_data_name: str,
        state: dace.SDFGState,
    ):

        nchecks = list(state.all_nodes_between(
            node, state.exit_node(state.entry_node(node))
        )) + [state.exit_node(state.entry_node(node))]
        for _n in nchecks:
            if "IN_" + old_data_name in _n.in_connectors:
                v = _n.in_connectors["IN_" + old_data_name]
                _n.remove_in_connector("IN_" + old_data_name)
                _n.add_in_connector("IN_" + new_data_name, v)
            if "OUT_" + old_data_name in _n.out_connectors:
                v = _n.out_connectors["OUT_" + old_data_name]
                _n.remove_out_connector("OUT_" + old_data_name)
                _n.add_out_connector("OUT_" + new_data_name, v)

        for _e in state.all_edges(*nchecks):
            if _e.data.data == old_data_name:
                _e.data.data = new_data_name
            if _e.src_conn == "OUT_" + old_data_name:
                _e.src_conn = "OUT_" + new_data_name
            if _e.dst_conn == "IN_" + old_data_name:
                _e.dst_conn = "IN_" + new_data_name

    def _specialize_purpose(self, sdfg: SDFG, state: SDFGState, device_map_entry: nodes.MapEntry):
        all_nodes = state.all_nodes_between(
            device_map_entry, state.exit_node(device_map_entry)
        )
        if hasattr(device_map_entry, "purpose_dict"):
            purpose_dict = device_map_entry.purpose_dict
            for node in all_nodes:
                if isinstance(node, dace.nodes.AccessNode):
                    arr = sdfg.arrays[node.data]
                    storage_str = self._construct_str_from_storage(arr.storage)
                    print(storage_str, self.unspecialized_locations)
                    if storage_str in self.unspecialized_locations:
                        print(storage_str)
                        #raise Exception(purpose_dict)
                        #specialized_storage =
                        def get_ending_number(s):
                            match = re.search(r'\d+$', s)
                            return int(match.group()) if match else None
                        level = get_ending_number(storage_str)
                        root_data_name = node.data.split("_")[-1]
                        purpose = purpose_dict[root_data_name]
                        if purpose == "acc":
                            purpose = "CO"
                        purpose += str(level)
                        # Remove the _L2 for example
                        no_loc_stroge_str = storage_str[0:-(1+len(storage_str.split("_")[-1]))]
                        new_storage = dace.dtypes.StorageType[(no_loc_stroge_str + "_" + purpose).split(".")[-1]]

                        new_prefix = purpose
                        old_prefix = storage_str.split("_")[-1]
                        #raise Exception(storage_str, new_storage)
                        arr.storage = new_storage
                        old_name = storage_str.split("_")[-1] + "_" + root_data_name
                        new_name = new_prefix + "_" + root_data_name
                        sdfg.add_datadesc(new_name, copy.deepcopy(arr), find_new_name=False)
                        sdfg.replace(old_name, new_name)
                        #raise Exception(old_name, new_name)
                        for _n in state.nodes():
                            if isinstance(_n, dace.nodes.AccessNode):
                                if _n.data == old_name:
                                    _n.data = new_name
                        for _e in state.edges():
                            if _e.data.data == old_name:
                                _e.data.data = new_name
                        sdfg.remove_data(old_name, validate=False)

                    if arr.storage == dace.dtypes.StorageType.Register:
                        raise Exception(arr)

    def _apply(self, state: SDFGState, sdfg: SDFG, device_map_entry: nodes.MapEntry):
        print(self._G)
        self._specialize_register_storage(sdfg, state, device_map_entry)
        sdfg.save("gemm_register_specialized.sdfg")
        self._specialize_purpose(sdfg, state, device_map_entry)
        sdfg.save("gemm_locations_specialized.sdfg")
        all_nodes = state.all_nodes_between(
            device_map_entry, state.exit_node(device_map_entry)
        )
        edges_to_add = set()
        nodes_to_add = set()
        edges_to_rm = set()
        edges_to_rm_type2 = set()
        for node in all_nodes:
            if isinstance(node, dace.nodes.MapEntry):
                map_entry = node
                # print(map_entry, map_entry.computational_units)
                # If we have memlets with data types that are higher than the suggested distance, we need to insert transfers

                # If we are on a map that has 2 comp units we cant distinguish properly where to move, ti should
                # stay as global
                if len(map_entry.computational_units) > 1:
                    continue

                for compute_unit, distance in map_entry.computational_units.items():

                    if compute_unit == "VECTOR":
                        edges = state.in_edges(map_entry)
                        for e in edges:
                            if e.data.data is not None:
                                in_arr = sdfg.arrays[e.data.data]
                                print(e.data.data, in_arr, in_arr.storage)
                                print(map_entry.computational_units)
                                print(
                                    str(in_arr.storage),
                                    self._entry_location_requirements[compute_unit],
                                    (
                                        str(in_arr.storage)
                                        not in self._entry_location_requirements[
                                            compute_unit
                                        ]
                                    ),
                                )
                                if (
                                    str(in_arr.storage)
                                    not in self._entry_location_requirements[
                                        compute_unit
                                    ]
                                ):
                                    print(
                                        "AAAA",
                                        in_arr.storage,
                                        self._entry_location_requirements[compute_unit],
                                    )
                                    # Check distance
                                    # Get paths starting with the current computational unit and ending with the compute unit
                                    paths = [
                                        p
                                        for p in self._paths
                                        if p[0]
                                        == self._construct_str_from_storage(
                                            in_arr.storage
                                        )
                                        and p[-1]
                                        in self._entry_location_requirements[
                                            compute_unit
                                        ]
                                    ]
                                    print(paths)
                                    # Take the first shortest path
                                    shortest_path = min(paths, key=len)
                                    assert shortest_path is not None
                                    print(
                                        "S",
                                        shortest_path,
                                        self._construct_str_from_storage(
                                            in_arr.storage
                                        ),
                                        self._entry_location_requirements[compute_unit],
                                    )
                                    # if transfer_length > distance:
                                    #    raise Exception("Must move")
                                    # Start moving according to the shortest path
                                    # 1. Get next location
                                    # 2. Add access node with the next type
                                    # 2.1 If not in SDFG.arrays add
                                    # 3. Redirect the edge to go through the new access node
                                    src_edge = e
                                    _e_add, _e_rm = self._insert_access_node(
                                        shortest_path, src_edge, sdfg, state
                                    )
                                    print("EA", _e_add)
                                    edges_to_add = edges_to_add.union(_e_add)
                                    edges_to_rm = edges_to_rm.union(_e_rm)

        for e in edges_to_add:
            state.add_edge(*e)
        for e in edges_to_rm:
            state.remove_edge(e)
        for n in nodes_to_add:
            state.add_node(n)

        edges_to_add = set()
        edges_to_rm = set()
        nodes_to_add = set()

        # Replace outputs of tasklets
        self._replace_tasklet_outputs(state, sdfg, device_map_entry)

        # Then insert movement nodes towards the exit
        """
        for node in all_nodes:
            if isinstance(node, dace.nodes.MapExit):
                map_exit = node

                if len(map_entry.computational_units) > 1:
                    continue

                for compute_unit, distance in map_entry.computational_units.items():

                    if compute_unit == "VECTOR":
                        edges = state.out_edges(map_exit)
                        for e in edges:
                            if e.data.data is not None:
                                out_arr = sdfg.arrays[e.data.data]
                                print(e.data.data, out_arr, out_arr.storage)
                                print(map_entry.computational_units)
                                print(
                                    str(out_arr.storage),
                                    self.exit_location_requirements[compute_unit],
                                    (
                                        str(out_arr.storage)
                                        not in self.exit_location_requirements[
                                            compute_unit
                                        ]
                                    ),
                                )
                                if (
                                    str(out_arr.storage)
                                    not in self.exit_location_requirements[compute_unit]
                                ):
                                    print(
                                        "BBB",
                                        out_arr.storage,
                                        self.exit_location_requirements[compute_unit],
                                    )
                                    # Check distance
                                    # Get paths starting with the current computational unit and ending with the compute unit
                                    paths = [
                                        p
                                        for p in self._paths
                                        if p[0]
                                        == self._construct_str_from_storage(
                                            out_arr.storage
                                        )
                                        and p[-1]
                                        in self.exit_location_requirements[compute_unit]
                                    ]
                                    # Take the first shortest path
                                    shortest_path = min(paths, key=len)
                                    assert shortest_path is not None
                                    print(
                                        "S",
                                        shortest_path,
                                        self._construct_str_from_storage(
                                            in_arr.storage
                                        ),
                                        self.exit_location_requirements[compute_unit],
                                    )
                                    # Start moving according to the shortest path
                                    # 1. Get next location
                                    # 2. Add access node with the next type
                                    # 2.1 If not in SDFG.arrays add
                                    # 3. Redirect the edge to go through the new access node
                                    src_edge = e
                                    _e_add, _e_rm = self._insert_access_node_at_exit(
                                        shortest_path, src_edge, sdfg, state
                                    )
                                    print("EA", _e_add)
                                    edges_to_add = edges_to_add.union(_e_add)
                                    edges_to_rm = edges_to_rm.union(_e_rm)
        """
        # raise Exception("Not implemented")

    @staticmethod
    def annotates_memlets():
        return True
