import copy
import itertools
import os
from pathlib import Path
import shutil
from typing import Dict, List, Tuple, Type, Any, Union

import numpy as np
import dace
from dace.transformation.auto_tile import auto_tile_util
from dace.transformation.auto_tile.add_compute_element_map import AddComputeElementBlockMap
from dace.transformation.auto_tile.add_thread_block_map import AddThreadBlockMap
from dace.transformation.auto_tile.thread_coarsening import ThreadCoarsening
from dace.transformation.auto_tile.block_tiling import BlockTiling
from dace.transformation.auto_tile.remainder_loop import RemainderLoop
from dace.sdfg.analysis.cutout import SDFGCutout
from dace.transformation.dataflow import MapInterchange
from dace.transformation.dataflow.tiling import MapTiling


def clean_cache():
    script_directory = os.getcwd()
    cache_dir = Path(f"{script_directory}/.dacecache")
    print(f"Clean {script_directory}/.dacecache")
    if cache_dir.exists() and cache_dir.is_dir():
        shutil.rmtree(cache_dir)


def copy_sub_scope(state: dace.sdfg.SDFGState, scope_entry: dace.nodes.MapEntry):

    nn = []
    for n in state.bfs_nodes(scope_entry):
        if n == state.exit_node(scope_entry):
            nn.append(n)
            break
        nn.append(n)

    cut_sdfg = SDFGCutout.singlestate_cutout(state, *nn)
    return cut_sdfg


def find_node_by_cond(state, start_map_entry, cond):
    s = set([start_map_entry])
    while s:
        n = s.pop()
        if n != start_map_entry and cond(n, state):
            return n
        if n != state.exit_node(start_map_entry):
            s = s.union([v for _, _, v, _, _ in state.out_edges(n)])
    return None


def find_node_in_state_by_cond(state, cond):
    for n in state.nodes():
        if cond(n, state):
            return n
    return None


def find_nodes_by_cond(state, start_map_entry, cond):
    s = set([start_map_entry])
    ret = set()
    while s:
        n = s.pop()
        if n != start_map_entry and cond(n, state):
            ret.add(n)
        if n != state.exit_node(start_map_entry):
            s = s.union([v for _, _, v, _, _ in state.out_edges(n)])
    return list(ret)


def find_state_by_cond(sdfg, cond):
    for n in sdfg.states():
        if cond(n):
            return n
    return None


def get_ref_kernel_nodes_and_edges(state, kernel_entry):
    kernel_nodes = set()
    kernel_nodes_to_visit = [kernel_entry]
    kernel_edges = set()
    visited_node_guids = set()

    while kernel_nodes_to_visit:
        n = kernel_nodes_to_visit.pop(0)
        if n.guid in visited_node_guids:
            continue
        visited_node_guids.add(n.guid)
        kernel_nodes.add(n)

        kernel_edges = kernel_edges.union(state.out_edges(n))
        kernel_edges = kernel_edges.union(state.in_edges(n))

        if n != state.exit_node(kernel_entry):
            for _, _, v, _, _ in state.out_edges(n) + state.in_edges(n):
                if not v.guid in visited_node_guids:
                    kernel_nodes_to_visit.append(v)

    return (kernel_nodes, kernel_edges)


def validate_and_pad_params_to_three(params):
    validated_params = []
    for param in params:
        if len(param) < 3:
            padded_param = param + (1,) * (3 - len(param))
            validated_params.append(padded_param)
        elif len(param) == 3:
            validated_params.append(param)
        else:
            raise ValueError(
                f"Tuple {param} has length greater than 3, which is not allowed."
            )
    return validated_params


def _tile(
    sdfg: dace.SDFG,
    state: dace.SDFGState,
    entry: dace.nodes.EntryNode,
    work_on_copy: bool,
    memory_tiling_parameters: List[Tuple[int]],
    thread_coarsening_parameters: List[Tuple[int]],
    thread_block_parameters: List[Tuple[int]],
    apply_remainder_loop: List[bool],
    combinations,
    inputs: Dict[Type[str], Any],
    re_apply: bool,
    verbose: bool,
    verify: bool,
    call_id: int,
    num_cores: int,
):
    if not re_apply:
        raise NotImplementedError("Not re-applying is not implemeneted for tiling yet")


    # Copy kernel as a single state SDFG if we are working on the copy
    if work_on_copy:
        _kernel_sdfg = copy_sub_scope(state, entry)
        _kernel_sdfg.name = f"{sdfg.name}_auto_tiled_{call_id}"
        auto_tile_util.set_transient(_kernel_sdfg)
        _kernel_state = _kernel_sdfg.states()[0]
        _kernel_entry = find_node_in_state_by_cond(
            _kernel_state,
            lambda n, kernel_state: isinstance(n, dace.nodes.MapEntry)
            and n.map.schedule == dace.dtypes.ScheduleType.Default
            and kernel_state.entry_node(n) is None
            and n.guid == entry.guid,
        )
        _kernel_exit = _kernel_state.exit_node(_kernel_entry)
        output_name = None
        for oe in _kernel_state.out_edges(_kernel_exit):
            if isinstance(oe.dst, dace.nodes.AccessNode):
                output_name = oe.dst.data
                break

        if output_name is None:
            raise Exception("The output name could not be deduced")

        copy_inputs = copy.deepcopy(inputs)
        # print("SDFG:", _kernel_sdfg, "INPUTS:", copy_inputs)
        if _kernel_sdfg is None:
            raise Exception("_kernel_sdfg should not be None")

        """
        for cfg in _kernel_sdfg.states():
            for n in cfg.nodes():
                if (isinstance(n, dace.nodes.MapEntry)
                    and n.map.schedule == dace.dtypes.ScheduleType.Default
                    and cfg.entry_node(n) is None):
                    n.map.schedule = dace.dtypes.ScheduleType.CPU_Multicore
        """

        non_transformed_time = auto_tile_util.run_and_measure_time(
            kernel_sdfg=_kernel_sdfg,
            inputs=copy_inputs,
            repeats=2,
            warmups=1,
            dev_type=dace.dtypes.ScheduleType.Default,
            instr_type=dace.dtypes.InstrumentationType.Timer,
        )

        """
        for cfg in _kernel_sdfg.states():
            for n in cfg.nodes():
                if (isinstance(n, dace.nodes.MapEntry)
                    and n.map.schedule == dace.dtypes.ScheduleType.CPU_Multicore
                    and cfg.entry_node(n) is None):
                    n.map.schedule = dace.dtypes.ScheduleType.Default
        """

        output_from_non_transformed = copy_inputs[output_name]
        # Clean memory we do not need anymore
        for key in list(copy_inputs.keys()):
            if key != output_name:
                del copy_inputs[key]

        # Unset GPU events
        for node in _kernel_state.nodes():
            if isinstance(node, dace.nodes.MapEntry):
                node.instrument = dace.dtypes.InstrumentationType.No_Instrumentation

        kernel_work_maps = find_nodes_by_cond(
            _kernel_state,
            _kernel_entry,
            lambda n, kernel_state: isinstance(n, dace.nodes.MapEntry)
            and n.map.schedule == dace.dtypes.ScheduleType.Sequential,
        )
    else:
        _kernel_sdfg = sdfg
        _kernel_state = state
        _kernel_entry = entry

    if combinations is None:
        combinations = list(
            itertools.product(
                memory_tiling_parameters,
                validate_and_pad_params_to_three(thread_coarsening_parameters),
                validate_and_pad_params_to_three(thread_block_parameters),
                apply_remainder_loop,
            )
        )
    if not work_on_copy:
        if len(combinations) != 1:
            raise Exception(
                f"If applying to the original sdfg (work_on_copy=False) then only one combination must be provided {combinations}"
            )

    best_config = None
    best_time = None
    for i, current_config in enumerate(combinations):
        # We need to copy this sdfg if we are working in the copy as we apply transformations
        (
            memory_tiling_params,
            thread_coarsening_param,
            thread_block_param,
            apply_remainder_loop_param,
        ) = current_config
        if verbose:
            print("Current config:", current_config)

        if work_on_copy:
            kernel_sdfg = copy.deepcopy(_kernel_sdfg)
            kernel_sdfg.name = f"{kernel_sdfg.name}_c{i}"
            kernel_sdfg_nodes = kernel_sdfg.states()
            if len(kernel_sdfg_nodes) != 1:
                raise Exception("Extracted kernel should have only one state")
            kernel_state = kernel_sdfg_nodes[0]
            kernel_entry = find_node_in_state_by_cond(
                kernel_state,
                lambda n, kernel_state: isinstance(n, dace.nodes.MapEntry)
                and n.map.schedule == dace.dtypes.ScheduleType.Default
                and kernel_state.entry_node(n) is None
                and n.guid == _kernel_entry.guid,
            )
        else:
            kernel_sdfg = _kernel_sdfg
            kernel_state = _kernel_state
            kernel_entry = _kernel_entry
        # else: we do not need to do anything

        has_work_maps = False
        kernel_maps = 0
        if work_on_copy:
            if  len(kernel_sdfg.states()) != 1:
                raise Exception("If working on copy then the kernel should have only one state")

            for state in kernel_sdfg.states():
                for node in state.nodes():
                    if isinstance(node, dace.nodes.MapEntry):
                        if node.map.schedule == dace.dtypes.ScheduleType.Default:
                            kernel_maps += 1
                            nodes = state.all_nodes_between(node, state.exit_node(node))
                            has_work_maps = any([n for n in nodes if isinstance(n, dace.nodes.MapEntry) and n.map.schedule == dace.dtypes.ScheduleType.Sequential])
                            if has_work_maps is True:
                                break

            if kernel_maps != 1:
                raise Exception("Kernel should have only one kernel map")

        is_assign_kernel = len(kernel_state.in_edges(kernel_entry)) == 0
        AddComputeElementBlockMap.apply_to(
            sdfg=kernel_sdfg,
            verify=True,
            map_entry=kernel_entry,
            options={
                "compute_element_group_dims":thread_block_param,
                "map_schedule":dace.dtypes.ScheduleType.Default,
                "schedule_to_add":dace.dtypes.ScheduleType.CPU_Persistent,
            },
        )
        # Need to restore maps after each time
        kernel_entry = kernel_state.entry_node(kernel_entry)
        thread_block_map_entry = find_node_by_cond(
            kernel_state,
            kernel_entry,
            lambda n, kernel_state: isinstance(n, dace.nodes.MapEntry)
            and n.map.label == dace.dtypes.ScheduleType.CPU_Persistent.name + "Map",
        )
        if thread_block_map_entry is None:
            raise Exception(
                "ThreadBlock Map could not be found after applying threadblock map transformation"
            )

        ThreadCoarsening.apply_to(
            sdfg=kernel_sdfg,
            options={
                "tile_sizes": thread_coarsening_param,
            },
            verify=True,
            device_map_entry=kernel_entry,
            thread_group_map_entry=thread_block_map_entry,
        )
        work_maps = find_nodes_by_cond(
            kernel_state,
            kernel_entry,
            lambda n, kernel_state: isinstance(n, dace.nodes.MapEntry)
            and n.label != "KernelEntryMap"
            and n.label != "ThreadCoarsenedMap"
            and n.label != dace.dtypes.ScheduleType.CPU_Persistent.name + "Map",
        )
        thread_block_map_entry = find_node_by_cond(
            kernel_state,
            kernel_entry,
            lambda n, kernel_state: isinstance(n, dace.nodes.MapEntry)
            and n.map.label == dace.dtypes.ScheduleType.CPU_Persistent.name + "Map",
        )
        if len(work_maps) > 1:
            raise NotImplementedError(
                "Memory tiling (Tiling of Work Maps) more than once is currently not implemented"
            )

        # Need to check the "assign-kernel" variant
        # Having input should work
        if not is_assign_kernel:
            for i in range(len(work_maps)):
                work_map_entry: dace.nodes.MapEntry = work_maps[i]
                work_map_tile = memory_tiling_params[0]
                #print("Work map tile:", work_map_tile)

                # If the passed memory tiling parameter is less than the map dimension, pad
                # If it longer, then take the first elements
                tuple_size_needed = len(work_map_entry.map.range)
                work_map_tile = work_map_tile[:tuple_size_needed] + (1,) * (
                    tuple_size_needed - len(work_map_tile)
                )

                BlockTiling.apply_to(
                    sdfg=kernel_sdfg,
                    options={"block_tile_sizes": work_map_tile},
                    verify=True,
                    thread_block_map_entry=thread_block_map_entry,
                    work_map_entry=work_map_entry,
                )
            thread_block_map_entry = find_node_by_cond(
                kernel_state,
                kernel_entry,
                lambda n, kernel_staet: isinstance(n, dace.nodes.MapEntry)
                and n.map.label == dace.dtypes.ScheduleType.CPU_Persistent.name +"Map",
            )

        thread_block_map_entry = find_node_by_cond(
            kernel_state,
            kernel_entry,
            lambda n, kernel_state: isinstance(n, dace.nodes.MapEntry)
            and n.map.label == dace.dtypes.ScheduleType.CPU_Persistent.name +"Map",
        )
        if apply_remainder_loop_param:
            first_inner_work_map = find_node_by_cond(
                kernel_state,
                thread_block_map_entry,
                lambda n, kernel_state: isinstance(n, dace.nodes.MapEntry)
                and n.map.label.startswith("InnerWorkMap"),
            )
            if len(work_maps) > 0:
                RemainderLoop.apply_to(
                    sdfg=kernel_sdfg,
                    verify=True,
                    inner_work_map_entry=first_inner_work_map,
                )
            else:
                thread_coarsened_map = find_node_by_cond(
                    kernel_state,
                    thread_block_map_entry,
                    lambda n, kernel_state: isinstance(n, dace.nodes.MapEntry)
                    and n.map.label.startswith("ThreadCoarsenedMap"),
                )
                RemainderLoop.apply_to(
                    sdfg=kernel_sdfg,
                    verify=True,
                    inner_work_map_entry=thread_coarsened_map,
                    options={
                        "tblock_type":dace.dtypes.ScheduleType.CPU_Persistent,
                    }
                )

        time = None

        if work_on_copy:
            for cfg in kernel_sdfg.states():
                for node in cfg.nodes():
                    if isinstance(node, dace.nodes.MapEntry):
                        if node.map.schedule == dace.dtypes.ScheduleType.Default:
                            node.map.schedule = dace.dtypes.ScheduleType.Sequential

            def find_next_map(state, n):
                found_n = False
                for node in state.nodes():
                    if found_n and isinstance(node, dace.nodes.MapEntry):
                        return node
                    if node == n:
                        found_n = True
                return None

            interchanges = []
            for s in kernel_sdfg.states():
                for n in s.nodes():
                    if isinstance(n, dace.nodes.MapEntry):
                        if (n.map.schedule == dace.dtypes.ScheduleType.Sequential and
                            s.entry_node(n) is None):
                            # Find next map:
                            next_map = find_next_map(s, n)
                            interchanges.append((n, next_map))

            if len(interchanges) != 1:
                raise Exception("There should be only one CPU_Persitent and sequential kernel map to interchange if working ona  copy")

            # Need to apply map-interchange to make CPU_persistent appear before
            for outer, inner in interchanges:
                MapInterchange.apply_to(
                    sdfg=kernel_sdfg,
                    outer_map_entry=outer,
                    inner_map_entry=inner,
                )

        if work_on_copy and verify:
            copy_inputs_2 = copy.deepcopy(inputs)
            time = auto_tile_util.run_and_measure_time(
            kernel_sdfg=kernel_sdfg,
            inputs=copy_inputs_2,
            repeats=2,
            warmups=1,
            dev_type=dace.dtypes.ScheduleType.CPU_Persistent,
            instr_type=dace.dtypes.InstrumentationType.Timer,
        )
            output_from_transformed = copy_inputs_2[output_name]

            are_close = np.allclose(
                output_from_transformed,
                output_from_non_transformed,
                rtol=1e-3,
                atol=1e-5,
            )

            # Clean memory we do not need anymore
            for key in list(copy_inputs_2.keys()):
                del copy_inputs_2[key]

            if not are_close:
                raise Exception("Numerical verification failed.")

            if best_time is None or time < best_time:
                best_config = current_config
                best_time = time

            print(f"Transformed SDFG: {time:.10f} ms")
            print(f"Current config: {current_config}, best config: {best_config}")
            print(f"Non-transformed SDFG: {non_transformed_time:.10f} ms")
            print(f"Speed-up: {non_transformed_time / time:.2f}")
    return best_config, best_time


# Possible future parameters:
# sdfg_peak_flops_and_mem_access: Union[Tuple[int, int], None] = None,
# machine_peak_flops_and_bandwidth: Union[Tuple[int, int], None] = None,


def auto_tile_cpu(
    sdfg: dace.SDFG,
    exhaustive_search: bool,
    memory_tiling_parameters: List[Tuple[int]],
    thread_coarsening_parameters: List[Tuple[int]],
    thread_block_parameters: List[Tuple[int]],
    apply_remainder_loop: List[bool],
    combinations,
    inputs: Dict[Type[str], Any],
    re_apply: bool = False,
    verbose: bool = False,
    num_cores: int = 32,
):
    device_schedule: dace.dtypes.ScheduleType = dace.dtypes.ScheduleType.Default
    sdfg_name = sdfg.name
    sym_dict = sdfg.symbols

    # Create report folder and file
    folder = Path(f"{sdfg_name}_report")
    filename = Path.joinpath(folder, Path(f"{sdfg_name}.report"))
    folder.mkdir(parents=True, exist_ok=True)
    tiled_sdfg_path = Path.joinpath(folder, Path(f"{sdfg_name}_auto_tiled.sdfgz"))

    # If this SDFG was tiled before, just return
    if filename.exists() and tiled_sdfg_path.exists() and not re_apply:
        return dace.SDFG.from_file(str(tiled_sdfg_path)), None

    # filename.open('w').close() if filename.exists() else filename.touch()

    # Collect Device kernels
    kernel_guids: List[Tuple[dace.sdfg.SDFGState, str]] = []
    for state in sdfg.states():
        for node in state.nodes():
            if (
                isinstance(node, dace.nodes.MapEntry)
                and node.map.schedule == device_schedule
                and (state, node.guid) not in kernel_guids
            ):
                kernel_guids.append((state, node))

    # Apply tiling one-by-one to kernels
    found_tilings = dict()
    for ii, (state, kernel_entry) in enumerate(kernel_guids):
        if exhaustive_search:
            best_config, best_time = _tile(
                sdfg=sdfg,
                state=state,
                entry=kernel_entry,
                work_on_copy=True,
                memory_tiling_parameters=memory_tiling_parameters,
                thread_coarsening_parameters=thread_coarsening_parameters,
                thread_block_parameters=thread_block_parameters,
                apply_remainder_loop=apply_remainder_loop,
                combinations=combinations,
                inputs=inputs,
                re_apply=re_apply,
                verbose=verbose,
                verify=True,
                call_id=ii,
                num_cores=num_cores,
            )
            found_tilings[(state.guid, kernel_entry.guid)] = tuple([(state.label, kernel_entry.label), best_config, best_time])
        else:
            print("TODO")

        if verbose:
            print(f"Best Tiling Configuration for {kernel_entry.label}: {best_config}")

    for (state_guid, kernel_entry_guid), (labels, best_config, best_time) in found_tilings.items():
        state = find_state_by_cond(sdfg, lambda n: n.guid == state_guid)
        if state is None:
            raise Exception("After auto-tiling, the state is none")
        kernel_entry = find_node_in_state_by_cond(
            state, lambda n, kernel_state: n.guid == kernel_entry_guid
        )
        if kernel_entry is None:
            raise Exception("After auto-tiling the kernel entry is none")
        # Create a single element list for applying the transformations

        memory_tiling = [best_config[0]]
        thread_coarsening = [best_config[1]]
        thread_block_coarsening = [best_config[2]]
        remainder_loop = [best_config[3]]
        """
            memory_tiling_params,
            thread_coarsening_param,
            thread_block_param,
            apply_remainder_loop_param,
        """

        if exhaustive_search:
            _tile(
                sdfg=sdfg,
                state=state,
                entry=kernel_entry,
                work_on_copy=False,
                memory_tiling_parameters=memory_tiling,
                thread_coarsening_parameters=thread_coarsening,
                thread_block_parameters=thread_block_coarsening,
                apply_remainder_loop=remainder_loop,
                combinations=[best_config],
                inputs=inputs,
                re_apply=True,
                verbose=verbose,
                verify=False,
                call_id=len(kernel_guids),
                num_cores=num_cores,
            )
        else:
            raise Exception("TODO")

    print("Internal search completed")
    # Add missing symbols
    for input_sym, sym in inputs.items():
        if input_sym not in sdfg.symbols and input_sym not in sdfg.arrays:
            if isinstance(sym, dace.symbolic.symbol):
                sdfg.add_symbol(input_sym, sym.dtype)
            else:
                sdfg.add_symbol(input_sym, dace.dtypes.typeclass(type(sym)))

    # Post processing, COPIED FROM _tile, IMPROVE CODE QUALITY
    """
    for cfg in sdfg.states():
        for node in cfg.nodes():
            if isinstance(node, dace.nodes.MapEntry):
                if node.map.schedule == dace.dtypes.ScheduleType.Default:
                    node.map.schedule = dace.dtypes.ScheduleType.Sequential

    def find_next_map(state, n):
        found_n = False
        for node in state.nodes():
            if found_n and isinstance(node, dace.nodes.MapEntry):
                return node
            if node == n:
                found_n = True
        return None

    interchanges = []
    for s in sdfg.states():
        for n in s.nodes():
            if isinstance(n, dace.nodes.MapEntry):
                if (n.map.schedule == dace.dtypes.ScheduleType.Sequential and
                    s.entry_node(n) is None):
                    # Find next map:
                    next_map = find_next_map(s, n)
                    interchanges.append((n, next_map))

    for outer, inner in interchanges:
        MapInterchange.apply_to(
            sdfg=sdfg,
            outer_map_entry=outer,
            inner_map_entry=inner,
        )
    """
    return sdfg, found_tilings
