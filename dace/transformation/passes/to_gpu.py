# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
"""This module contains classes and functions that implement the grid-strided map tiling
transformation."""

import ast
import copy

from sympy import Set
import dace
import typing
from dace.codegen.control_flow import ConditionalBlock, ControlFlowBlock
from dace.data import Property, make_properties
from dace.sdfg import is_devicelevel_gpu
from dace.sdfg.state import ControlFlowRegion
from dace.transformation import pass_pipeline as ppl
from dace.transformation.passes.duplicate_const_arrays import DuplicateConstArrays
from dace.sdfg import utils as sdutil


@make_properties
class ToGPU(ppl.Pass):
    verbose: bool = Property(dtype=bool, default=False, desc="Print debug information")
    # duplicated_const_arrays: typing.Dict[str, str] = Property(
    #    dtype=dict,
    #    default=None,
    #    desc="Dictionary of duplicated constant arrays (key: old name, value: new name)"
    # )

    def __init__(
        self,
        verbose: bool = False,
        # duplicated_const_arrays: typing.Dict[str, str] = None,
    ):
        self.verbose = verbose
        super().__init__()

    def modifies(self) -> ppl.Modifies:
        return (
            ppl.Modifies.Nodes
            | ppl.Modifies.Edges
            | ppl.Modifies.AccessNodes
            | ppl.Modifies.Memlets
            | ppl.Modifies.Descriptors
        )

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def move_transients_to_top_level(self, root: dace.SDFG):
        # If we have a transient array, make it live on the top level SDFG
        # For this, collect all transients that do not exist on top level SDFG
        # Add them to top level SDFG, if they are arrays and have storage location Default
        #
        # for s in sdfg.states():
        #    for n in s.nodes():
        #        if isinstance(n, dace.nodes.NestedSDFG):
        #            self.move_transients_to_top_level(roots + [sdfg], n.sdfg)

        # Add arrays from bottom to up
        arrays_added = dict()
        for sdfg, arr_name, arr in root.arrays_recursive():
            if sdfg != root:
                if (
                    arr.transient
                    and isinstance(arr, dace.data.Array)
                    and arr.shape != (1,)
                ):
                    if (
                        arr.storage == dace.dtypes.StorageType.Default
                        or arr.storage == dace.dtypes.CPU_Heap
                        or arr.storage == dace.dtypes.GPU_Global
                    ):
                        arr.transient = False
                        _sdfg = sdfg
                        while _sdfg is not None:
                            if _sdfg == root:
                                if arr_name not in _sdfg.arrays:
                                    arr2 = copy.deepcopy(arr)
                                    if sdfg == root:
                                        arr2.transient = True
                                    else:
                                        arr2.transient = False
                                    _sdfg.add_datadesc(arr_name, arr2)
                            else:
                                if arr_name not in sdfg.arrays:
                                    arr2 = copy.deepcopy(arr)
                                    if sdfg == root:
                                        arr2.transient = True
                                    else:
                                        arr2.transient = False
                                    _sdfg.add_datadesc(arr_name, arr2)
                            if _sdfg not in arrays_added:
                                arrays_added[_sdfg] = set([arr_name])
                            else:
                                arrays_added[_sdfg].add(arr_name)

                            _sdfg = _sdfg.parent_sdfg

        # Now go through all nodes and pass arguments needed to nested SDFGs
        # Recursively
        def pass_args(root: dace.SDFG, sdfg: dace.SDFG):
            for state in sdfg.states():
                # edges_to_add = []
                for node in state.nodes():
                    if isinstance(node, dace.nodes.NestedSDFG):
                        if node.sdfg not in arrays_added:
                            continue
                        arrays_to_pass = arrays_added[node.sdfg]
                        ies = state.in_edges(node)
                        srcs = set([e.src for e in ies])
                        # Assume Map -> Nested SDFG
                        assert len(srcs) == 1
                        src_map_entry = srcs.pop()
                        for arr_name in arrays_to_pass:
                            arr = sdfg.arrays[arr_name]
                            a0 = state.add_access(arr_name)
                            a1 = state.add_access(arr_name)
                            src_map_entry.add_in_connector("IN_" + arr_name)
                            src_map_entry.add_out_connector("OUT_" + arr_name)
                            src_map_exit = state.exit_node(src_map_entry)
                            src_map_exit.add_in_connector("IN_" + arr_name)
                            src_map_exit.add_out_connector("OUT_" + arr_name)
                            node.add_in_connector(arr_name)
                            node.add_out_connector(arr_name, force=True)
                            state.add_edge(
                                a0,
                                None,
                                src_map_entry,
                                "IN_" + arr_name,
                                dace.memlet.Memlet.from_array(arr_name, arr),
                            )
                            state.add_edge(
                                src_map_entry,
                                "OUT_" + arr_name,
                                node,
                                arr_name,
                                dace.memlet.Memlet.from_array(arr_name, arr),
                            )
                            state.add_edge(
                                node,
                                arr_name,
                                src_map_exit,
                                "IN_" + arr_name,
                                dace.memlet.Memlet.from_array(arr_name, arr),
                            )
                            state.add_edge(
                                src_map_exit,
                                "OUT_" + arr_name,
                                a1,
                                None,
                                dace.memlet.Memlet.from_array(arr_name, arr),
                            )
                            pass_args(root, node.sdfg)

        pass_args(root, root)
        root.validate()
        return arrays_added

    def get_const_arrays(self, sdfg: dace.SDFG, skip_first_and_last=True):
        arrays_written_to = {
            k: 0
            for k, v in sdfg.arrays.items()
            if isinstance(v, dace.data.Array)
            if not k.startswith("gpu_")
        }

        def _collect_writes(root: dace.SDFG, sdfg: dace.SDFG, arrays_written_to, dtype):
            for state in sdfg.all_states():
                if state == sdfg.start_block:
                    continue
                end_block = [v for v in sdfg.nodes() if sdfg.out_degree(v) == 0][0]
                if state == end_block:
                    continue
                if (
                    skip_first_and_last
                    and sdfg == root
                    and (sdfg.out_degree(state) == 0 or sdfg.in_degree(state) == 0)
                ):
                    continue
                for node in state.nodes():
                    if isinstance(node, dace.nodes.AccessNode):
                        if len(state.in_edges(node)) > 0:
                            for ie in state.in_edges(node):
                                if (
                                    ie.data is not None
                                    and ie.data.data is not None
                                    and ie.dst_conn != "views"
                                ):
                                    if isinstance(sdfg.arrays[node.data], dtype):
                                        if node.data.startswith("gpu_"):
                                            arrays_written_to[node.data[4:]] += 1
                                        else:
                                            arrays_written_to[node.data] += 1
                    # No need to be recursive, data needs to be passed and retaken from NestedSDFG
                    if isinstance(node, dace.nodes.NestedSDFG):
                        _collect_writes(sdfg, node.sdfg, arrays_written_to, dtype)
            for inter_edge in sdfg.edges():
                if isinstance(inter_edge.data, dace.InterstateEdge):
                    for assignment in inter_edge.data.assignments:
                        if assignment in arrays_written_to:
                            arrays_written_to[assignment] += 1

        _collect_writes(sdfg, sdfg, arrays_written_to, dace.data.Array)

        non_const_arrays = set(
            [
                k
                for k, v in arrays_written_to.items()
                if (
                    (
                        (v > 0 and sdfg.arrays[k].transient)
                        or (not sdfg.arrays[k].transient)
                    )
                    and isinstance(sdfg.arrays[k], dace.data.Array)
                )
            ]
        )
        arrays = set(
            [
                arr_name
                for arr_name, arr in sdfg.arrays.items()
                if isinstance(arr, dace.data.Array)
            ]
        )
        const_arrays = arrays - non_const_arrays

        return const_arrays

    def apply_pass(
        self, sdfg: dace.SDFG, pipeline_results: typing.Dict[str, typing.Any]
    ) -> int:
        # 1. Make CPU and GPU copies for all top level transients (both Y and gpu_Y should exist)
        # 2. Copy all CPU non-transients to their GPU copies (X -> gpu_X)
        # 3. Analyize which arrays are constant (Non-transients and the containger groups initialized in the first state)
        # 4. Change all top-level maps to GPU schedule (and replace the access nodes)
        # 5. Analyze what each CFG require the data to be where
        # 5.1 If "CPU" is necessary, copy the data from GPU to CPU if on GPU otherwise do nothing
        # 5.2 If "GPU" is necessary, copy the data from CPU to GPU if on CPU otherwise do nothing
        # 5.3 If the copy needs to be done from "Initial" then set then assign the initial location as that device
        # 5.4 If "BOTH" locations are necessary refined the granularity to state-level
        # 5.5 If the data is read from a constant array, do not copy it but replace the access
        # 6. Update the copies on the last and first state to the flattener / deflattener library nodes according to the initial
        # and final locations

        # 1. and 2. partially done in the flattening pass
        # Add GPU clones for all arays, copy all non-transients to GPU if already not on GPU
        transients_moved_to_top_level = self.move_transients_to_top_level(sdfg)
        print(f"Move these transients to top-level SDFG: {transients_moved_to_top_level}")

        replace_names = dict()
        descs = set()
        for name, arr in sdfg.arrays.items():
            if isinstance(arr, dace.data.Array):
                if arr.transient:
                    if (
                        arr.storage == dace.dtypes.StorageType.GPU_Global
                        and not name.startswith("gpu_")
                    ):
                        raise Exception("GPU array without gpu_ prefix")
                    if (
                        arr.storage == dace.dtypes.StorageType.Default
                        or arr.storage == dace.dtypes.StorageType.CPU_Heap
                    ):
                        if "gpu_" + name not in sdfg.arrays:
                            gpu_arr = copy.deepcopy(arr)
                            gpu_arr.storage = dace.dtypes.StorageType.GPU_Global
                            gpu_arr.transient = True
                            descs.add(("gpu_" + name, gpu_arr))
                else:
                    # If not transient and GPU, change name
                    # If not transient and CPU, copy to GPU
                    if (
                        arr.storage == dace.dtypes.StorageType.GPU_Global
                        and not name.startswith("gpu_")
                    ):
                        replace_names(name, "gpu_" + name)
                    if (
                        arr.storage == dace.dtypes.StorageType.Default
                        or arr.storage == dace.dtypes.StorageType.CPU_Heap
                    ):
                        if "gpu_" + name not in sdfg.arrays:
                            gpu_arr = copy.deepcopy(arr)
                            gpu_arr.storage = dace.dtypes.StorageType.GPU_Global
                            gpu_arr.transient = True
                            descs.add(("gpu_" + name, gpu_arr))
                            a0 = sdfg.start_state.add_access(name)
                            a1 = sdfg.start_state.add_access("gpu_" + name)
                            sdfg.start_state.add_edge(
                                a0,
                                None,
                                a1,
                                None,
                                dace.memlet.Memlet.from_array(name, arr),
                            )
                            # Do the reverse at the end
                            # last_states = [node for node in sdfg.nodes() if sdfg.out_degree(node) == 0]
                            # assert len(last_states) == 1
                            # last_state = last_states[0]
                            # a2 = last_state.add_access("gpu_" + name)
                            # a3 = last_state.add_access(name)
                            # last_state.add_edge(a2, None, a3, None, dace.memlet.Memlet.from_array("gpu_" + name, sdfg.arrays["gpu_" + name]))
        sdfg.replace_dict(replace_names)
        for name, arr in descs:
            sdfg.add_datadesc(name, arr)

        last_states = [node for node in sdfg.nodes() if sdfg.out_degree(node) == 0]
        assert len(last_states) == 1
        last_state = last_states[0]

        # App copy-outs for non-transients
        for name, arr in sdfg.arrays.items():
            if arr.transient is False and "gpu_" + name in sdfg.arrays:
                a2 = last_state.add_access("gpu_" + name)
                a3 = last_state.add_access(name)
                last_state.add_edge(
                    a2,
                    None,
                    a3,
                    None,
                    dace.memlet.Memlet.from_array(
                        "gpu_" + name, sdfg.arrays["gpu_" + name]
                    ),
                )


        # 3. Analyze which arrays are constant - Other then gpu arrays generated there should be not GPU arrays used
        # Do not check the last and initial state

        const_arrays = [
            v for v in self.get_const_arrays(sdfg, True) if not v.startswith("gpu_")
        ]
        const_arrays = []
        print(f"Constant arrays: {const_arrays}")

        # 4. Change all top-level maps to GPU schedule
        # Library nodes should know what they are doing
        gpu_nodes: Set[typing.Tuple[dace.SDFGState, dace.nodes.Node]] = set()
        for state in sdfg.all_states():
            sdict = state.scope_dict()
            for node in state.nodes():
                if sdict[node] is None:
                    if isinstance(
                        node, (dace.nodes.LibraryNode, dace.nodes.NestedSDFG)
                    ):
                        if node.guid:
                            node.schedule = dace.dtypes.ScheduleType.GPU_Default
                            gpu_nodes.add((state, node))
                    elif isinstance(node, dace.nodes.EntryNode):
                        node.schedule = dace.dtypes.ScheduleType.GPU_Device
                        gpu_nodes.add((state, node))
                else:
                    if isinstance(node, (dace.nodes.EntryNode, dace.nodes.LibraryNode)):
                        node.schedule = dace.dtypes.ScheduleType.Sequential
                    elif isinstance(node, dace.nodes.NestedSDFG):
                        for nnode, _ in node.sdfg.all_nodes_recursive():
                            if isinstance(
                                nnode, (dace.nodes.EntryNode, dace.nodes.LibraryNode)
                            ):
                                nnode.schedule = dace.dtypes.ScheduleType.Sequential

        for state in sdfg.all_states():
            for node in state.nodes():
                if isinstance(node, dace.nodes.AccessNode):
                    is_gpu = is_devicelevel_gpu(sdfg, state, node)
                    for e in state.in_edges(node):
                        if isinstance(e.src, dace.nodes.MapExit):
                            if (
                                e.src.map.schedule
                                == dace.dtypes.ScheduleType.GPU_Device
                            ):
                                is_gpu = True
                    for e in state.out_edges(node):
                        if isinstance(e.dst, dace.nodes.MapEntry):
                            if (
                                e.dst.map.schedule
                                == dace.dtypes.ScheduleType.GPU_Device
                            ):
                                is_gpu = True
                    oldname = node.data

                    if node.data.startswith("gpu_") and not is_gpu:
                        arr = sdfg.arrays[node.data]
                        if isinstance(arr, dace.data.Array):
                            if isinstance(e.dst, dace.nodes.MapEntry):
                                node.data = node.data[4:]

                                for _e in state.all_edges(
                                    *state.all_nodes_between(
                                        node, state.exit_node(e.dst)
                                    )
                                ):
                                    if _e.data.data == oldname:
                                        _e.data.data = oldname[4:]
                                e.data.data = oldname[4:]
                            if isinstance(e.src, dace.nodes.MapExit):
                                node.data = node.data[4:]

                                entry_node = [
                                    n
                                    for n in state.nodes()
                                    if isinstance(n, dace.nodes.MapEntry)
                                    and n.map == e.src.map
                                ][0]
                                for _e in state.all_edges(
                                    *state.all_nodes_between(entry_node, node)
                                ):
                                    if _e.data.data == oldname:
                                        _e.data.data = oldname[4:]
                                e.data.data = oldname[4:]
                    if not node.data.startswith("gpu_") and is_gpu:
                        arr = sdfg.arrays[node.data]

                        if isinstance(arr, dace.data.Array):
                            if isinstance(e.dst, dace.nodes.MapEntry):
                                node.data = "gpu_" + node.data

                                for _e in state.all_edges(
                                    *state.all_nodes_between(
                                        node, state.exit_node(e.dst)
                                    )
                                ):
                                    if _e.data.data == oldname:
                                        _e.data.data = "gpu_" + oldname
                                e.data.data = "gpu_" + oldname
                            if isinstance(e.src, dace.nodes.MapExit):
                                node.data = "gpu_" + node.data

                                entry_node = [
                                    n
                                    for n in state.nodes()
                                    if isinstance(n, dace.nodes.MapEntry)
                                    and n.map == e.src.map
                                ][0]
                                for _e in state.all_edges(
                                    *state.all_nodes_between(entry_node, node)
                                ):
                                    if _e.data.data == oldname:
                                        _e.data.data = "gpu_" + oldname
                                e.data.data = "gpu_" + oldname

        # Map input and outputs are on GPU but access nodes have not been changed, change
        for state in sdfg.all_states():
            for node in state.nodes():
                if isinstance(node, dace.nodes.AccessNode):
                    arr = sdfg.arrays[node.data]
                    if isinstance(arr, dace.data.Array):
                        if (
                            arr.storage == dace.dtypes.StorageType.GPU_Global
                            and node.data.startswith("gpu_")
                        ):
                            for e in state.in_edges(node):
                                if isinstance(e.src, dace.nodes.MapExit):
                                    entry_node = [
                                        n
                                        for n in state.nodes()
                                        if isinstance(n, dace.nodes.MapEntry)
                                        and n.map == e.src.map
                                    ][0]
                                    for _e in state.all_edges(
                                        *state.all_nodes_between(entry_node, e.src)
                                    ):
                                        if _e.data.data == node.data[4:]:
                                            _e.data.data = node.data
                                e.data.data = node.data

                            for e in state.out_edges(node):
                                if isinstance(e.dst, dace.nodes.MapEntry):
                                    for _e in state.all_edges(
                                        *state.all_nodes_between(
                                            node, state.exit_node(e.dst)
                                        )
                                    ):
                                        if _e.data.data == node.data[4:]:
                                            _e.data.data = node.data
                                e.data.data = node.data

        arrays_and_locations = dict()
        for name, arr in sdfg.arrays.items():
            if isinstance(arr, dace.data.Array):
                if not name.startswith("gpu_"):
                    arrays_and_locations[name] = "Unknown"

        # 5.

        # Do first touch
        location_history = [{k: "Unknown" for k in arrays_and_locations.keys()}]

        # Thanks to CFG, right now all top level nodes should have out degree 1, except last block
        # We can now track what data is needed where for each node
        def insert_state_level_transfers(
            cfg: ControlFlowBlock, prev_locs, local_location_history, depth
        ):
            assert location_history != [{}]
            states_to_add = []
            for node in sdutil.dfs_topological_sort(cfg):
                used_data = get_used_data(sdfg, node)
                print(f"Data Used By: {node}, are: {used_data}")
                _arrays_and_locations = dict()
                for dataname, location in used_data:
                    pruned_dataname = (
                        dataname[4:] if dataname.startswith("gpu_") else dataname
                    )
                    if dataname in const_arrays:
                        _arrays_and_locations[pruned_dataname] = "CONST"
                    if pruned_dataname in const_arrays:
                        _arrays_and_locations[pruned_dataname] = "CONST"
                    else:
                        if pruned_dataname in _arrays_and_locations:
                            _arrays_and_locations[pruned_dataname] = "BOTH"
                        else:
                            _arrays_and_locations[pruned_dataname] = location
                print(f"Locations for: {node}, {node.guid}, are: {_arrays_and_locations}")
                #local_location_history.append(_arrays_and_locations)
                # Update history
                _a = copy.deepcopy(local_location_history[-1])
                for k,v in _arrays_and_locations.items():
                    _a[k] = v
                local_location_history.append(_a)
                print(f"Previous Locations for: {node}, {node.guid}, are: {local_location_history[-2]}")

                # 5.4 Decrease granularity to the state of a CFG

                moves = []
                for d, loc in _arrays_and_locations.items():
                    if d in local_location_history[-2]:
                        if local_location_history[-2][d] != loc:
                            if (
                                local_location_history[-2][d] != "Unknown"
                                and loc != "CONST"
                                and loc != "BOTH"
                            ):
                                if local_location_history[-2][d] == "BOTH":
                                    assert prev_locs[d] != "BOTH"
                                    if prev_locs[d] != loc and prev_locs[d] != "Unknown":
                                        moves.append((d, prev_locs[d], loc))
                                else:
                                    moves.append((d, local_location_history[-2][d], loc))

                both_locs = [k for k, v in _arrays_and_locations.items() if v == "BOTH"]
                prev_locs = dict()
                for loc_hist in location_history[0:-1]:
                    for k, v in loc_hist.items():
                        prev_locs[k] = v
                for k, v in prev_locs.items():
                    assert v != "BOTH"

                print(f"Moves for {node}: {moves}")
                if len(moves) > 0:
                    states_to_add.append((moves, (node, cfg)))

                if len(both_locs) > 0:
                    current_locs = dict()
                    for loc_hist in local_location_history:
                        for k, v in loc_hist.items():
                            current_locs[k] = v
                    print(current_locs)
                    _states_to_add, last_locations = insert_state_level_transfers(
                        node, prev_locs, [copy.deepcopy(current_locs)], depth + 1
                    )
                    states_to_add += _states_to_add
                    local_location_history[-1].update(last_locations)
                    #for k,v in local_location_history[-2].items():
                    #    if v == "BOTH":
                    #        local_location_history[-2][k] = initial_locations[k]
                    #    else:
                    #        assert v == initial_locations[k]

                initial_locations = dict()
                #for locations in local_location_history:
                #    for d, loc in locations.items():
                #        if loc != "CONST" and loc != "BOTH" and loc != "Unknown":
                #            if d not in initial_locations:
                #                initial_locations[d] = loc

            return states_to_add, local_location_history[-1]

        def insert_transfers(sdfg: dace.SDFG):
            cfg = sdfg.start_block
            end_block = [b for b in sdfg.nodes() if sdfg.out_degree(b) == 0][0]
            visited = set()
            stack = set([cfg])
            i = 0
            current_locs = {k: "Unknown" for k in arrays_and_locations.keys()}
            states_to_add = []
            #while len(stack) > 0:
            for current in sdutil.dfs_topological_sort(sdfg):
                if current == sdfg.start_block or current == end_block:
                    continue

                print(current, type(current))
                used_data = get_used_data(sdfg, current)
                print(f"Data Used By: {current}, are: {used_data}")

                _arrays_and_locations = dict()

                for dataname, location in used_data:
                    pruned_dataname = (
                        dataname[4:] if dataname.startswith("gpu_") else dataname
                    )
                    if dataname in const_arrays:
                        _arrays_and_locations[pruned_dataname] = "CONST"
                    if pruned_dataname in const_arrays:
                        _arrays_and_locations[pruned_dataname] = "CONST"
                    else:
                        if pruned_dataname in _arrays_and_locations:
                            _arrays_and_locations[pruned_dataname] = "BOTH"
                        else:
                            _arrays_and_locations[pruned_dataname] = location
                print(f"Locations for: {current}, {current.guid}, are: {_arrays_and_locations}")

                # Update history
                _a = copy.deepcopy(location_history[-1])
                for k,v in _arrays_and_locations.items():
                    _a[k] = v
                location_history.append(_a)


                # 5.4 Decrease granularity to the state of a CFG
                moves = []
                for d, loc in _arrays_and_locations.items():
                    if d in location_history[-2]:
                        if location_history[-2][d] != loc:
                            if (
                                location_history[-2][d] != "Unknown"
                                and loc != "CONST"
                                and loc != "BOTH"
                            ):
                                if (location_history[-2][d] != "BOTH" and
                                    loc != location_history[-2][d]):
                                    moves.append((d, location_history[-2][d], loc))

                # Join all previous location histories

                both_locs = [k for k, v in _arrays_and_locations.items() if v == "BOTH"]
                prev_locs = dict()
                for loc_hist in location_history[0:-1]:
                    for k, v in loc_hist.items():
                        prev_locs[k] = v
                for k, v in prev_locs.items():
                    assert v != "BOTH"

                if len(both_locs) > 0:
                    # l = [copy.deepcopy(_arrays_and_locations)]
                    current_locs = dict()
                    for loc_hist in location_history:
                        for k, v in loc_hist.items():
                            current_locs[k] = v
                    print("Deepen:", current_locs)
                    _states_to_add, last_location_history = insert_state_level_transfers(
                        current, prev_locs, [copy.deepcopy(current_locs)], 1
                    )
                    states_to_add += _states_to_add
                    location_history[-1].update(last_location_history)
                    #for k,v in location_history[-2].items():
                    #    if v == "BOTH":
                    #        location_history[-2][k] = initial_locations[k]
                    #    else:
                    #        #assert v == initial_locations[k]
                    #        pass

                if len(moves) > 0:
                    states_to_add.append((moves, (current, None)))

                i += 1

                initial_locations = dict()
                #for locations in location_history:
                #    for d, loc in locations.items():
                #        if loc != "CONST" and loc != "BOTH" and loc != "Unknown":
                #            if d not in initial_locations:
                #                initial_locations[d] = loc

            return states_to_add, location_history[-1]

        states_to_add, last_location_history = insert_transfers(sdfg)
        for moves, (post, parent) in states_to_add:
            assert len(moves) > 0, f"{moves}, ({post}, {parent})"
            if parent is None:
                parent = sdfg
            s = parent.add_state_before(post, "pred")
            for name, src_loc, dst_loc in moves:
                if src_loc == "GPU" and dst_loc == "CPU":
                    a0 = s.add_access("gpu_" + name)
                    a1 = s.add_access(name)
                    s.add_edge(
                        a0,
                        None,
                        a1,
                        None,
                        dace.memlet.Memlet.from_array(
                            "gpu_" + name, sdfg.arrays["gpu_" + name]
                        ),
                    )
                elif src_loc == "CPU" and dst_loc == "GPU":
                    # Copy from CPU to GPU
                    a0 = s.add_access(name)
                    a1 = s.add_access("gpu_" + name)
                    s.add_edge(
                        a0,
                        None,
                        a1,
                        None,
                        dace.memlet.Memlet.from_array(name, sdfg.arrays[name]),
                    )
                else:
                    # if src_loc is BOTH
                    assert False, f"{src_loc} -> {dst_loc} not supported"

        sdfg.validate()

        # 6. Decrease number of copy-in and copy-outs
        # If first location is CPU then do not copy to GPU in the first state
        initial_locations = dict()
        for locations in location_history:
            for d, loc in locations.items():
                if loc != "CONST" and loc != "BOTH" and loc != "Unknown":
                    if d not in initial_locations:
                        initial_locations[d] = loc


        start_state = sdfg.start_state
        end_state = [s for s in sdfg.states() if sdfg.out_degree(s) == 0][0]
        nodes_to_rm = set()
        edges_to_rm = set()
        # Do not rm any initial node for now
        """
        for name, initial_loc in initial_locations.items():
            if initial_loc == "CPU":
                for edge in start_state.edges():
                    if isinstance(edge.src, dace.nodes.AccessNode) and isinstance(
                        edge.dst, dace.nodes.AccessNode
                    ):
                        if edge.src.data == name and edge.dst.data == "gpu_" + name:
                            # We can remove this edge and dst node
                            assert start_state.out_degree(edge.dst) == 0
                            nodes_to_rm.add(edge.dst)
                            edges_to_rm.add(edge)
        for e in edges_to_rm:
            start_state.remove_edge(e)
        for n in nodes_to_rm:
            start_state.remove_node(n)"
        """
        nodes_to_rm = set()
        edges_to_rm = set()


        current_locs = dict()
        for loc_hist in location_history:
            for k, v in loc_hist.items():
                current_locs[k] = v
        final_locations = copy.deepcopy(current_locs)
        print("Final Locations:", final_locations)
        end_state = [s for s in sdfg.states() if sdfg.out_degree(s) == 0][0]
        for name, final_loc in final_locations.items():
            if final_loc == "CPU" or name in const_arrays or final_loc == "Unknown" or final_loc == "CONST":
                for edge in end_state.edges():
                    if isinstance(edge.src, dace.nodes.AccessNode) and isinstance(
                        edge.dst, dace.nodes.AccessNode
                    ):
                        if edge.src.data == "gpu_" + name and edge.dst.data == name:
                            # We can remove this edge and dst node
                            assert end_state.in_degree(edge.src) == 0
                            nodes_to_rm.add(edge.src)
                            edges_to_rm.add(edge)
        for e in edges_to_rm:
            end_state.remove_edge(e)
        for n in nodes_to_rm:
            end_state.remove_node(n)

        # GO through all nodes, if NestedSDFG within GPU Device scope
        # Move all transients to GPU Global
        # Allocate all transients on GPU Global

        def move_to_gpu(
            node: dace.nodes.NestedSDFG, sdfg: dace.SDFG, state: dace.SDFGState
        ):
            _sdfg = node.sdfg
            for name, arr in _sdfg.arrays.items():
                if isinstance(arr, dace.data.Array):
                    if arr.transient is False:
                        arr.storage = dace.dtypes.StorageType.GPU_Global
                    if arr.transient is True:
                        if (
                            arr.storage == dace.dtypes.StorageType.Default
                            or arr.storage == dace.dtypes.StorageType.CPU_Heap
                        ):
                            arr.storage = dace.dtypes.StorageType.Register
            for _state in _sdfg.states():
                for _node in sdutil.dfs_topological_sort(_state):
                    if isinstance(_node, dace.nodes.NestedSDFG):
                        move_to_gpu(_node, _sdfg, _state)

        for state in sdfg.states():
            for node in sdutil.dfs_topological_sort(state):
                if (
                    isinstance(node, dace.nodes.MapEntry)
                    and node.map.schedule == dace.dtypes.ScheduleType.GPU_Device
                ):

                    for inner_node in state.all_nodes_between(
                        node, state.exit_node(node)
                    ):
                        if isinstance(inner_node, dace.nodes.NestedSDFG):
                            move_to_gpu(inner_node, sdfg, state)

        sdfg.validate()

def all_states(cfg: ControlFlowRegion):
    states = set()
    for node in cfg.nodes():
        if isinstance(node, dace.SDFGState):
            states.add(node)
        if isinstance(node, ControlFlowRegion):
            for n in node.nodes():
                states = states.union(all_states(n))
    return states


def all_states(sdfg: dace.SDFGState, cfg: ControlFlowBlock):
    states = set()
    for node, _ in cfg.all_nodes_recursive():
        if (not hasattr(node, "sdfg")) or (node.sdfg != sdfg):
            continue
        if isinstance(node, dace.SDFGState):
            states.add(node)
    return states


def get_used_data(sdfg: dace.SDFG, cfg: ControlFlowBlock | dace.SDFGState):
    if isinstance(cfg, dace.SDFGState):
        data_used = set()
        if isinstance(cfg, dace.SDFGState):
            slist = [cfg]
        else:
            slist = cfg.all_states()
        for s in slist:
            for node in s.nodes():
                if isinstance(node, dace.nodes.AccessNode):
                    is_gpu = is_devicelevel_gpu(sdfg, cfg, node)
                    for e in s.in_edges(node):
                        if isinstance(e.src, dace.nodes.MapExit):
                            if e.src.map.schedule == dace.dtypes.ScheduleType.GPU_Device:
                                is_gpu = True
                    for e in s.out_edges(node):
                        if isinstance(e.dst, dace.nodes.MapEntry):
                            if e.dst.map.schedule == dace.dtypes.ScheduleType.GPU_Device:
                                is_gpu = True
                    data_used.add((node.data, "GPU" if is_gpu else "CPU"))

        return data_used

    if isinstance(cfg, ControlFlowBlock) and not isinstance(cfg, dace.SDFGState):
        return set().union(*[get_used_data(sdfg, node) for node in cfg.nodes()])

    raise Exception("Should not reach here")
