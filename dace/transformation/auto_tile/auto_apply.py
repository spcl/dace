import copy
from ctypes import Union
import random
import statistics
from typing import Dict, List, Tuple, Type, Any
import dace
from dace.dtypes import ScheduleType
from dace.dtypes import StorageType
from dace.sdfg import nodes
from dace.sdfg.sdfg import SDFG
from dace.symbolic import SymExpr
from dace.transformation.auto_tile.add_thread_block_map import AddThreadBlockMap
from dace.transformation.auto_tile.thread_coarsening import ThreadCoarsening
from dace.transformation.auto_tile.explicit_memory_move import ExplicitMemoryMove
from dace.transformation.auto_tile.block_tiling import BlockTiling
from dace.transformation.auto_tile.remainder_loop import RemainderLoop
from dace import nodes
from dace import config
from dace import dtypes
import itertools
import numpy
import cupy
from pathlib import Path
import json
from dace.sdfg.analysis.cutout import SDFGCutout
import sympy
from dace.transformation.auto_tile import auto_tile_util
from multiprocessing import Process, Queue
import os


def copy_sub_scope(
    parent: dace.sdfg.SDFG, state: dace.sdfg.SDFGState, scope_entry: nodes.MapEntry
):
    nn = []
    for n in state.bfs_nodes(scope_entry):
        if n == state.exit_node(scope_entry):
            break
        nn.append(n)

    cut_sdfg = SDFGCutout.singlestate_cutout(state, *nn)
    return cut_sdfg


def find_node_by_cond(state, start_map_entry, cond):
    s = set([start_map_entry])
    while s:
        n = s.pop()
        if n != start_map_entry and cond(n):
            return n
        if n != state.exit_node(start_map_entry):
            s = s.union([v for _, _, v, _, _ in state.out_edges(n)])
    return None


def find_node_in_state_by_cond(state, cond):
    for n in state.nodes():
        if cond(n):
            return n
    return None


def find_nodes_by_cond(state, start_map_entry, cond):
    s = set([start_map_entry])
    ret = set()
    while s:
        n = s.pop()
        if n != start_map_entry and cond(n):
            ret.add(n)
        if n != state.exit_node(start_map_entry):
            s = s.union([v for _, _, v, _, _ in state.out_edges(n)])
    return list(ret)


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


def apply_using_params(
    sdfg: SDFG,
    sdfg_name,
    state,
    _entry,
    peak_flops,
    peak_bandwidth,
    flops,
    mem_access,
    threshold,
    work_map_tiling_params: List[Tuple],
    thread_coarsening_params: List[Tuple],
    thread_block_params: List[Tuple],
    apply_explicit_memory_transfers: List[bool],
    apply_remainder_loop: List[bool],
    inputs: Dict[Type[str], Any] = dict(),
    output_name: str = "",
    verbose=False,
    work_on_copy=True,
    save_steps=False,
    save_individual_kernels=False,
    write_kernel_report_to_file=True,
    compare_runtime=True,
):
    randomly_generated_data = dict()
    best_params = None
    b = False
    untransformed_time = -1.0

    if work_on_copy and write_kernel_report_to_file:
        filename = f"{sdfg_name}_report/{_entry.guid}.report"
        i = 1

        while os.path.exists(filename):
            filename = f"{sdfg_name}_report/{_entry.guid}_{i}.report"
            i += 1

        file = open(filename, "w")

    for work_map_tiles in work_map_tiling_params:
        if b:
            break
        for thread_tile in thread_coarsening_params:
            if b:
                break
            for thread_block_size in thread_block_params:
                if b:
                    break
                for explicit_mem_move in apply_explicit_memory_transfers:
                    if b:
                        break
                    for remainder_loop in apply_remainder_loop:
                        if b:
                            break

                        if work_on_copy:
                            kernel_sdfg = copy_sub_scope(sdfg, state, _entry)
                            kernel_state = kernel_sdfg.nodes()[0]
                            kernel_entry = find_node_in_state_by_cond(
                                kernel_state,
                                lambda n: isinstance(n, nodes.MapEntry)
                                and n.map.schedule
                                == dace.dtypes.ScheduleType.GPU_Device
                                and n.guid == _entry.guid,
                            )
                            print(
                                f"Start working on the copy of {kernel_entry} using parameters {(work_map_tiles, thread_tile, thread_block_size, explicit_mem_move, remainder_loop)}"
                            )
                            for symbol, typeclass in sdfg.symbols.items():
                                if not symbol in kernel_sdfg.symbols:
                                    kernel_sdfg.add_symbol(symbol, typeclass)
                        else:
                            kernel_sdfg = sdfg
                            kernel_state = state
                            kernel_entry = _entry
                            print(
                                f"Apply transformations to {kernel_entry} using the parameters {(work_map_tiles, thread_tile, thread_block_size, explicit_mem_move, remainder_loop)}"
                            )

                        guid = kernel_entry.guid

                        is_assign_kernel = flops == 0  # Assign or copy map

                        if work_on_copy:
                            kernel_sdfg_copy = copy.deepcopy(kernel_sdfg)
                            auto_tile_util.convert_inputs_to_gpu_storage(kernel_sdfg)
                            auto_tile_util.set_transient(kernel_sdfg)
                            auto_tile_util.convert_inputs_to_gpu_storage(
                                kernel_sdfg_copy
                            )
                            auto_tile_util.set_transient(kernel_sdfg_copy)

                        placeholder = 1
                        thread_block_size = list(
                            itertools.islice(
                                itertools.chain(
                                    thread_block_size, itertools.cycle([placeholder])
                                ),
                                3,
                            )
                        )

                        AddThreadBlockMap.apply_to(
                            sdfg=kernel_sdfg,
                            verify=True,
                            map_entry=kernel_entry,
                            options={
                                "thread_block_size_x": thread_block_size[0],
                                "thread_block_size_y": thread_block_size[1],
                                "thread_block_size_z": thread_block_size[2],
                            },
                        )

                        # Need to restore maps after each time
                        kernel_entry = kernel_state.entry_node(kernel_entry)

                        thread_block_map_entry = find_node_by_cond(
                            kernel_state,
                            kernel_entry,
                            lambda n: isinstance(n, nodes.MapEntry)
                            and n.map.label == "ThreadBlockMap",
                        )
                        assert thread_block_map_entry

                        placeholder = 1
                        thread_tile = list(
                            itertools.islice(
                                itertools.chain(
                                    thread_tile, itertools.cycle([placeholder])
                                ),
                                3,
                            )
                        )

                        ThreadCoarsening.apply_to(
                            sdfg=kernel_sdfg,
                            options={
                                "tile_size_x": thread_tile[0],
                                "tile_size_y": thread_tile[1],
                                "tile_size_z": thread_tile[2],
                            },
                            verify=True,
                            device_map_entry=kernel_entry,
                            thread_block_map_entry=thread_block_map_entry,
                        )

                        if work_on_copy and save_steps:
                            kernel_sdfg.save(f"{guid}_thread_coarsened.sdfg")

                        work_maps = find_nodes_by_cond(
                            kernel_state,
                            kernel_entry,
                            lambda n: isinstance(n, nodes.MapEntry)
                            and n.label != "KernelEntryMap"
                            and n.label != "ThreadCoarsenedMap"
                            and n.label != "ThreadBlockMap",
                        )
                        thread_block_map_entry = find_node_by_cond(
                            kernel_state,
                            kernel_entry,
                            lambda n: isinstance(n, nodes.MapEntry)
                            and n.map.label == "ThreadBlockMap",
                        )
                        if len(work_maps) > 1:
                            raise Exception("TODO")

                        # work_map_tiles is an array of arrays
                        # N-repeated param for each work loop (if cyclic)
                        # Each param group has M elements for  work maps (again cyclic)
                        if not is_assign_kernel:
                            for i in range(len(work_maps)):
                                work_map_entry = work_maps[i]
                                work_map_tile = work_map_tiles[i % len(work_map_tiles)]
                                work_map_tile = tuple(
                                    list(
                                        itertools.islice(
                                            itertools.chain(
                                                work_map_tile,
                                                itertools.cycle([placeholder]),
                                            ),
                                            3,
                                        )
                                    )
                                )
                                BlockTiling.apply_to(
                                    sdfg=kernel_sdfg,
                                    options={"block_tile_sizes": work_map_tile},
                                    verify=True,
                                    thread_block_map_entry=thread_block_map_entry,
                                    sequential_map_entry=work_map_entry,
                                )

                        if work_on_copy and save_steps and not is_assign_kernel:
                            kernel_sdfg.save(f"{guid}_block_tiled.sdfg")

                        work_maps = find_nodes_by_cond(
                            kernel_state,
                            kernel_entry,
                            lambda n: isinstance(n, nodes.MapEntry)
                            and n.label != "KernelEntryMap"
                            and n.label != "ThreadCoarsenedMap"
                            and n.label != "ThreadBlockMap",
                        )

                        thread_block_map_entry = find_node_by_cond(
                            kernel_state,
                            kernel_entry,
                            lambda n: isinstance(n, nodes.MapEntry)
                            and n.map.label == "ThreadBlockMap",
                        )

                        if len(work_maps) == 0:
                            first_map_to_apply_mem_move = thread_block_map_entry
                        else:
                            first_map_to_apply_mem_move = find_node_by_cond(
                                kernel_state,
                                thread_block_map_entry,
                                lambda n: isinstance(n, nodes.MapEntry),
                            )

                        if explicit_mem_move[0] and not is_assign_kernel:
                            ExplicitMemoryMove.apply_to(
                                sdfg=kernel_sdfg,
                                verify=True,
                                device_map_entry=kernel_entry,
                                thread_block_map_entry=thread_block_map_entry,
                                map_entry=first_map_to_apply_mem_move,
                                options={
                                    "memory_location": StorageType.GPU_Shared,
                                    "tiles_evenly": explicit_mem_move[1],
                                },
                            )

                        if save_steps and explicit_mem_move[0] and not is_assign_kernel:
                            kernel_sdfg.save(f"{guid}_mem_moved.sdfg")

                        first_inner_work_map = find_node_by_cond(
                            kernel_state,
                            thread_block_map_entry,
                            lambda n: isinstance(n, nodes.MapEntry)
                            and n.map.label.startswith("InnerWorkMap"),
                        )

                        if remainder_loop:
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
                                    lambda n: isinstance(n, nodes.MapEntry)
                                    and n.map.label.startswith("ThreadCoarsenedMap"),
                                )
                                RemainderLoop.apply_to(
                                    sdfg=kernel_sdfg,
                                    verify=True,
                                    inner_work_map_entry=thread_coarsened_map,
                                )

                        if not work_on_copy:
                            return None

                        if save_individual_kernels:
                            kernel_sdfg.save(f"{guid}_auto_tiled.sdfg")

                        inputs = auto_tile_util.generate_random_data(
                            kernel_sdfg, inputs
                        )

                        compiled: dace.CompiledSDFG = kernel_sdfg.compile(validate=True)

                        try:
                            compiled(**inputs)
                            output_from_non_transformed = None

                            if work_on_copy and (
                                (
                                    (output_name != None and output_name != "")
                                    or compare_runtime
                                )
                            ):
                                # If output is set, check the result from the non-transformed and transformed result
                                if output_name == "auto_deduce":
                                    kernel_exit = kernel_state.exit_node(kernel_entry)
                                    for oe in kernel_state.out_edges(kernel_exit):
                                        u, uc, v, vc, memlet = oe
                                        if isinstance(v, nodes.AccessNode):
                                            _output_name = v.data
                                            if verbose:
                                                s1 = f"Auto-deduced the type of kernel {kernel_entry} to be {_output_name}"
                                                print(s1)
                                                if (
                                                    work_on_copy
                                                    and write_kernel_report_to_file
                                                ):
                                                    file.write(s1 + "\n")
                                            break
                                else:
                                    _output_name = output_name
                                    if verbose:
                                        print(f"Outputname is provided: {_output_name}")

                                copy_inputs = copy.deepcopy(inputs)
                                kernel_sdfg_copy(**copy_inputs)
                                output_from_non_transformed = copy_inputs[_output_name]
                                copy_inputs_2 = copy.deepcopy(inputs)
                                compiled(**copy_inputs_2)
                                # auto_tile_util.run_sdfg_safe(compiled, copy_inputs_2)

                                output_from_transformed = copy_inputs_2[_output_name]

                                if not isinstance(
                                    output_from_transformed, cupy.ndarray
                                ):
                                    raise Exception(
                                        f"The type of {output_from_transformed} is {type(output_from_transformed)}, should be of {cupy.ndarray}"
                                    )
                                if not isinstance(
                                    output_from_non_transformed, cupy.ndarray
                                ):
                                    raise Exception(
                                        f"The type of {output_from_non_transformed} is {type(output_from_non_transformed)}, should be of {cupy.ndarray}"
                                    )

                                are_close = cupy.allclose(
                                    output_from_transformed,
                                    output_from_non_transformed,
                                    rtol=1e-3,
                                    atol=1e-5,
                                )
                                if are_close:
                                    s = f"For config {work_map_tiles}, {thread_tile}, {thread_block_size}, {explicit_mem_move}, {remainder_loop}: the transformations numerically verify"
                                    print(s)
                                    if work_on_copy and write_kernel_report_to_file:
                                        file.write(s + "\n")
                                else:
                                    s = f"For config {work_map_tiles}, {thread_tile}, {thread_block_size}, {explicit_mem_move}, {remainder_loop}: the transformations do not numerically verify"
                                    print(s)
                                    assert (
                                        output_from_transformed
                                        is not output_from_non_transformed
                                    )
                                    if work_on_copy and write_kernel_report_to_file:
                                        file.write(s + "\n")

                                if untransformed_time == -1.0:
                                    copy_inputs = copy.deepcopy(inputs)
                                    untransformed_time = (
                                        auto_tile_util.run_and_measure_time(
                                            kernel_sdfg_copy, copy_inputs
                                        )
                                    )

                            time = auto_tile_util.run_and_measure_time(
                                kernel_sdfg, inputs
                            )
                            print(f"Transformed SDFG: {time} ms")
                            if compare_runtime and work_on_copy:
                                s1 = (
                                    f"Old Time | {untransformed_time} | New Time | {time} |"
                                )
                                s2 = f"Non-Transformed SDFG: {untransformed_time}"
                                s3 = f"Transformed SDFG: {time}"
                                print(s1)
                                print(s2)
                                print(s3)
                                if work_on_copy and write_kernel_report_to_file:
                                    file.write(s1 + "\n")
                                    file.write(s2 + "\n")
                                    file.write(s3 + "\n")
                            solved_flops = auto_tile_util.solve(flops, inputs)
                            solved_mem_access = auto_tile_util.solve(mem_access, inputs)
                            if verbose:
                                s1 = f"Solved flops are | {solved_flops} |, solved mem access are | {solved_mem_access} |."
                                print(s1)
                                if work_on_copy and write_kernel_report_to_file:
                                    file.write(s1 + "\n")

                            if flops == 0:
                                percentage_of_peak = (
                                    auto_tile_util.percentage_bandwidth(
                                        time, solved_mem_access, peak_bandwidth
                                    )
                                )
                            else:
                                percentage_of_peak = auto_tile_util.percentage_peak(
                                    time,
                                    solved_flops,
                                    solved_mem_access,
                                    peak_flops,
                                    peak_bandwidth,
                                )

                            if best_params == None or time < best_params[-2]:
                                best_params = (
                                    (
                                        work_map_tiles,
                                        thread_tile,
                                        thread_block_size,
                                        explicit_mem_move,
                                        remainder_loop,
                                    ),
                                    time,
                                    percentage_of_peak,
                                )
                            if percentage_of_peak > threshold and threshold >= 0.0:
                                b = True

                            if work_on_copy:
                                s1 = f"{kernel_entry} | {guid} | achieves | {percentage_of_peak:.2f}% | of the peak wrt. roofline model after tiling"
                                t = (
                                    (
                                        work_map_tiles,
                                        thread_tile,
                                        thread_block_size,
                                        explicit_mem_move,
                                        remainder_loop,
                                    ),
                                    time,
                                    percentage_of_peak,
                                )
                                s2 = f"{kernel_entry} | {guid} | was transformed using parameters | {t}"
                                print(s1)
                                print(s2)
                                if work_on_copy and write_kernel_report_to_file:
                                    file.write(s1 + "\n")
                                    file.write(s2 + "\n")
                        except Exception as ex:
                            print(
                                f"Transformations fail for config {work_map_tiles}, {thread_tile}, {thread_block_size}, {explicit_mem_move}, {remainder_loop}"
                            )
                            print("Exception (gen.):", ex)

    if best_params:
        apply_using_params(
            sdfg=sdfg,
            sdfg_name=sdfg_name,
            state=state,
            _entry=_entry,
            peak_flops=peak_flops,
            peak_bandwidth=peak_bandwidth,
            flops=flops,
            mem_access=mem_access,
            threshold=threshold,
            work_map_tiling_params=[best_params[0][0]],
            thread_coarsening_params=[best_params[0][1]],
            thread_block_params=[best_params[0][2]],
            apply_explicit_memory_transfers=[best_params[0][3]],
            apply_remainder_loop=[best_params[0][4]],
            inputs=inputs,
            output_name=output_name,
            verbose=verbose,
            work_on_copy=False,
            save_steps=False,
            save_individual_kernels=False,
        )

    return best_params


f = open("o.txt", "w")

kid = 0


def auto_apply(
    sdfg: SDFG,
    work_map_tiling_params: List[Tuple],
    thread_coarsening_params: List[Tuple],
    thread_block_params: List[Tuple],
    apply_explicit_memory_transfers: List[bool],
    apply_remainder_loop: List[bool],
    inputs: Dict[Type[str], Any] = dict(),
    output_name: str = None,
    verbose=False,
    save_steps=False,
    save_individual_kernels=False,
    re_apply=False,
    theo_flops_and_mem_access=None,
    write_kernel_report_to_file=True,
    compare_runtime=True,
    _threshold=None,
    machine_peak_flops_and_mem_bandwidth=None,
):
    # Any map that is GPU_Device is transformed applying the
    # AMM-guided transformations.
    # Anything inside the map is treated as work to be tiled
    # If reference result is not None then the transformed SDFG
    # is called and checked with the reference result (must be array)
    if _threshold is None or _threshold == 0.0:
        threshold = 85.0
    else:
        threshold = _threshold

    kperf = dict()

    sdfg_name = sdfg._name

    if write_kernel_report_to_file:
        os.makedirs(f"{sdfg_name}_report", exist_ok=True)

    filename = f"{sdfg_name}_report/{sdfg_name}.report"
    ii = 1

    while os.path.exists(filename):
        filename = f"{sdfg_name}_report/{sdfg_name}_{ii}.report"
        ii += 1

    if write_kernel_report_to_file:
        file = open(filename, "w")

    """
    file_name = f"{sdfg_name}_auto_tiled_perf_results.json"
    if not re_apply and Path(file_name).is_file():
        with open(file_name, "r") as json_file:
            data_dict = json.load(json_file)
            for k, v in data_dict.items():
                filename, params, flop_str,  mem_str, time, perc = v
                nv = (filename, params,
                      sympy.sympify(flop_str), sympy.sympify(mem_str),
                       time, perc)
                data_dict[k] = nv
            return data_dict
    """
    file_name = f"{sdfg_name}_auto_tiled_perf_results.json"
    if (not re_apply) and Path(file_name).is_file():
        return None

    for state in sdfg.states():
        kernel_entry_guids = []
        for node in state.nodes():
            if (
                isinstance(node, nodes.MapEntry)
                and node.map.schedule == ScheduleType.GPU_Device
            ):
                kernel_entry_guids.append(node.guid)

        for kernel_entry_guid in kernel_entry_guids:
            kernel_entry = find_node_in_state_by_cond(
                state, lambda n: n.guid == kernel_entry_guid
            )
            if theo_flops_and_mem_access is None:
                flops, mem_access = auto_tile_util.get_flops_and_mem_access(
                    sdfg, state, kernel_entry
                )
            else:
                flops, mem_access = theo_flops_and_mem_access
            if verbose:
                s1 = f"FLOPs as a symbolic expression: {flops}"
                s2 = f"Bytes accessed as a symbolic expression: {mem_access}"
                print(s1)
                print(s2)
                if write_kernel_report_to_file:
                    file.write(s1 + "\n")
                    file.write(s2 + "\n")

            if machine_peak_flops_and_mem_bandwidth is None:
                from dace.transformation.auto_tile import peak_flops_and_badwidth_nvidia
                peak_flops, mem_bandwidth = (
                    peak_flops_and_badwidth_nvidia.get_peak_flops_and_mem_bandwidth(0)
                )
            else:
                peak_flops, mem_bandwidth = machine_peak_flops_and_mem_bandwidth
            peak_flops *= 1e9
            mem_bandwidth *= 1e9
            s1 = f"Peak FLOPs FLOPs/s: {peak_flops}"
            s2 = f"Peak Mem Bandwidth B/s: {mem_bandwidth}"
            print(s1)
            print(s2)
            if write_kernel_report_to_file:
                file.write(s1 + "\n")
                file.write(s2 + "\n")

            if not kernel_entry.guid in kperf:
                    best_config = apply_using_params(
                        sdfg=sdfg,
                        sdfg_name=sdfg_name,
                        state=state,
                        _entry=kernel_entry,
                        peak_flops=peak_flops,
                        peak_bandwidth=mem_bandwidth,
                        flops=flops,
                        mem_access=mem_access,
                        threshold=threshold,
                        work_map_tiling_params=work_map_tiling_params,
                        thread_coarsening_params=thread_coarsening_params,
                        thread_block_params=thread_block_params,
                        apply_explicit_memory_transfers=apply_explicit_memory_transfers,
                        apply_remainder_loop=apply_remainder_loop,
                        inputs=inputs,
                        output_name=output_name,
                        verbose=verbose,
                        work_on_copy=True,
                        save_steps=save_steps,
                        save_individual_kernels=save_individual_kernels,
                        write_kernel_report_to_file=write_kernel_report_to_file,
                        compare_runtime=compare_runtime,
                    )
                    kperf[kernel_entry.guid] = (
                        f"{kernel_entry.guid}_auto_tiled.sdfg",
                        best_config[0],
                        str(flops),
                        str(mem_access),
                        float(best_config[1]),
                        float(best_config[2]),
                    )
                    """
                    except Exception as e:
                        best_config = None
                        s1 = f"Exception: on transforming {kernel_entry}:"
                        s2 = str(e)
                        print(s1)
                        print(s2)
                        kperf[kernel_entry.guid] = (
                            f"{kernel_entry.guid}_auto_tiled.sdfg",
                            None,
                            str(flops),
                            str(mem_access),
                            float(-1.0),
                            float("nan"),
                        )
                        if write_kernel_report_to_file:
                            file.write(s1 + "\n")
                            file.write(s2 + "\n")
                    """

            if verbose:
                s1 = f"Best config: {best_config[:-1]}" if best_config else "none"
                s2 = f"Percentage of the peak: {best_config[-1]}" if best_config else "none"
                print(s1)
                print(s2)
                if write_kernel_report_to_file:
                    file.write(s1 + "\n")
                    file.write(s2 + "\n")

    if verbose:
        print("Auto-tiled performance results:")
        print("{")
        for k, v in kperf.items():
            print(f"\t{k}: {v}")
        print("}")

    sdfg.save(f"{sdfg_name}_auto_tiled.sdfg")

    with open(f"{sdfg_name}_auto_tiled_perf_results.json", "w") as json_file:
        json.dump(kperf, json_file, indent=2)

    return sdfg
