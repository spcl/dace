import random
import statistics
from typing import Dict, List, Tuple, Type, Any
import dace
from dace.dtypes import ScheduleType
from dace.dtypes import StorageType
from dace.sdfg import nodes
from dace.sdfg.sdfg import SDFG
from dace.transformation.auto_tile.add_thread_block_map import AddThreadBlockMap
from dace.transformation.auto_tile.thread_coarsening import ThreadCoarsening
from dace.transformation.auto_tile.explicit_memory_move import ExplicitMemoryMove
from dace.transformation.auto_tile.block_tiling import BlockTiling
from dace.transformation.auto_tile.remainder_loop import RemainderLoop
from dace import nodes
from dace import config
from dace import dtypes
import itertools
import numpy as np
import cupy as cp
from pathlib import Path
import json
import pycuda.driver as cuda
import pycuda.autoinit  # Actually used
from dace.sdfg.analysis.cutout import SDFGCutout
import sympy
from dace.transformation.auto_tile import peak_flops_and_badwidth_nvidia
from dace.transformation.auto_tile import auto_tile_util


def copy_sub_scope(parent: dace.sdfg.SDFG, state: dace.sdfg.SDFGState, scope_entry: nodes.MapEntry):
    nn = []
    for n in state.bfs_nodes(scope_entry):
        if n == state.exit_node(scope_entry):
            break
        nn.append(n)

    cut_sdfg = SDFGCutout.singlestate_cutout(state, *nn)
    return cut_sdfg


arch_str = peak_flops_and_badwidth_nvidia.get_arch(0)

config.Config.set("compiler", "cuda", "cuda_arch", value=arch_str)
config.Config.set("compiler", "cuda", "args",
                  value=" -DNDEBUG -std=c++17 -Xcompiler -march=native --use_fast_math -Xcompiler -Wno-unused-parameter")
config.Config.set("compiler", "cpu", "args",
                  value=" -DNDEBUG -std=c++17 -fPIC -Wall -Wextra -O3 -march=native -ffast-math -Wno-unused-parameter")


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
    reference_result=None,
    inputs: Dict[Type[str], Any] = dict(),
    output: cp.array = None,
    verbose=False,
    work_on_copy=True,
    save_steps=False,
    save_individual_kernels=False
):
    randomly_generated_data = dict()
    best_params = None
    b = False
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
                            kernel_entry = find_node_in_state_by_cond(kernel_state, lambda n: isinstance(n, nodes.MapEntry)
                                                                      and n.map.schedule == dace.dtypes.ScheduleType.GPU_Device
                                                                      and n.guid == _entry.guid)
                            print(f"Start working on the copy of {kernel_entry} using parameters {
                                  (work_map_tiles, thread_tile, thread_block_size, explicit_mem_move, remainder_loop)}")
                            for symbol, typeclass in sdfg.symbols.items():
                                if not symbol in kernel_sdfg.symbols:
                                    kernel_sdfg.add_symbol(symbol, typeclass)
                        else:
                            kernel_sdfg = sdfg
                            kernel_state = state
                            kernel_entry = _entry
                            print(f"Apply transformations to {kernel_entry} using the parameters {
                                  (work_map_tiles, thread_tile, thread_block_size, explicit_mem_move, remainder_loop)}")

                        guid = kernel_entry.guid


                        is_assign_kernel = flops == 0  # Assign or copy map

                        if work_on_copy:
                            auto_tile_util.convert_inputs_to_gpu_storage(kernel_sdfg)
                            auto_tile_util.set_transient(kernel_sdfg)

                        placeholder = 1
                        thread_block_size = list(
                            itertools.islice(
                                itertools.chain(
                                    thread_block_size,
                                    itertools.cycle([placeholder])),
                                3))

                        AddThreadBlockMap.apply_to(sdfg=kernel_sdfg,
                                                   verify=True,
                                                   map_entry=kernel_entry,
                                                   options={
                                                       "thread_block_size_x": thread_block_size[0],
                                                       "thread_block_size_y": thread_block_size[1],
                                                       "thread_block_size_z": thread_block_size[2]
                                                   })

                        # Need to restore maps after each time
                        kernel_entry = kernel_state.entry_node(kernel_entry)

                        thread_block_map_entry = find_node_by_cond(kernel_state, kernel_entry, lambda n: isinstance(
                            n, nodes.MapEntry) and n.map.label == "ThreadBlockMap")
                        assert (thread_block_map_entry)

                        placeholder = 1
                        thread_tile = list(
                            itertools.islice(
                                itertools.chain(
                                    thread_tile,
                                    itertools.cycle([placeholder])),
                                3))

                        ThreadCoarsening.apply_to(sdfg=kernel_sdfg,
                                                  options={
                                                      "tile_size_x": thread_tile[0],
                                                      "tile_size_y": thread_tile[1],
                                                      "tile_size_z": thread_tile[2],
                                                  },
                                                  verify=True,
                                                  device_map_entry=kernel_entry,
                                                  thread_block_map_entry=thread_block_map_entry)

                        if work_on_copy and save_steps:
                            kernel_sdfg.save(f"{guid}_thread_coarsened.sdfg")

                        work_maps = find_nodes_by_cond(kernel_state, kernel_entry, lambda n: isinstance(
                            n, nodes.MapEntry) and n.label != "KernelEntryMap" and n.label != "ThreadCoarsenedMap" and n.label != "ThreadBlockMap")
                        thread_block_map_entry = find_node_by_cond(kernel_state, kernel_entry,
                                                                   lambda n: isinstance(n, nodes.MapEntry) and n.map.label == "ThreadBlockMap")
                        if len(work_maps) > 1:
                            raise Exception("TODO")

                        # work_map_tiles is an array of arrays
                        # N-repeated param for each work loop (if cyclic)
                        # Each param group has M elements for  work maps (again cyclic)
                        if not is_assign_kernel:
                            for i in range(len(work_maps)):
                                work_map_entry = work_maps[i]
                                work_map_tile = work_map_tiles[i % len(
                                    work_map_tiles)]
                                work_map_tile = tuple(list(
                                    itertools.islice(
                                        itertools.chain(
                                            work_map_tile,
                                            itertools.cycle([placeholder])),
                                        3)))
                                BlockTiling.apply_to(sdfg=kernel_sdfg,
                                                     options={
                                                         "block_tile_sizes": work_map_tile
                                                     },
                                                     verify=True,
                                                     thread_block_map_entry=thread_block_map_entry,
                                                     sequential_map_entry=work_map_entry)

                        if work_on_copy and save_steps and not is_assign_kernel:
                            kernel_sdfg.save(f"{guid}_block_tiled.sdfg")

                        work_maps = find_nodes_by_cond(kernel_state, kernel_entry, lambda n: isinstance(
                            n, nodes.MapEntry) and n.label != "KernelEntryMap" and n.label != "ThreadCoarsenedMap" and n.label != "ThreadBlockMap")

                        thread_block_map_entry = find_node_by_cond(kernel_state, kernel_entry,
                                                                   lambda n: isinstance(n, nodes.MapEntry) and n.map.label == "ThreadBlockMap")

                        if len(work_maps) == 0:
                            first_map_to_apply_mem_move = thread_block_map_entry
                        else:
                            first_map_to_apply_mem_move = find_node_by_cond(kernel_state, thread_block_map_entry,
                                                                            lambda n: isinstance(n, nodes.MapEntry))

                        if explicit_mem_move[0] and not is_assign_kernel:
                            ExplicitMemoryMove.apply_to(
                                sdfg=kernel_sdfg,
                                verify=True,
                                device_map_entry=kernel_entry,
                                thread_block_map_entry=thread_block_map_entry,
                                map_entry=first_map_to_apply_mem_move,
                                options={
                                    "memory_location": StorageType.GPU_Shared,
                                    "tiles_evenly": explicit_mem_move[1]
                                }
                            )

                        if save_steps and explicit_mem_move[0] and not is_assign_kernel:
                            kernel_sdfg.save(f"{guid}_mem_moved.sdfg")

                        first_inner_work_map = find_node_by_cond(kernel_state, thread_block_map_entry,
                                                                 lambda n: isinstance(n, nodes.MapEntry) and n.map.label.startswith("InnerWorkMap"))

                        if remainder_loop:
                            if len(work_maps) > 0:
                                RemainderLoop.apply_to(
                                    sdfg=kernel_sdfg,
                                    verify=True,
                                    inner_work_map_entry=first_inner_work_map
                                )
                            else:
                                thread_coarsened_map = find_node_by_cond(kernel_state, thread_block_map_entry,
                                                                         lambda n: isinstance(n, nodes.MapEntry) and n.map.label.startswith("ThreadCoarsenedMap"))
                                RemainderLoop.apply_to(
                                    sdfg=kernel_sdfg,
                                    verify=True,
                                    inner_work_map_entry=thread_coarsened_map
                                )

                        if not work_on_copy:
                            return None

                        if save_individual_kernels:
                            kernel_sdfg.save(f"{guid}_auto_tiled.sdfg")

                        inputs = auto_tile_util.generate_random_data(
                            kernel_sdfg, inputs)

                        compiled: dace.CompiledSDFG = kernel_sdfg.compile(
                            validate=True)

                        try:
                            compiled(**inputs)

                            if output != None:
                                result = output.get()
                            if reference_result and output:
                                if not np.allclose(result, reference_result):
                                    print(result)
                                    print(reference_result)
                                    print(result - reference_result)
                                    raise Exception(
                                        "Numerical Verification Failing")
                                elif verbose:
                                    s = f"For config {work_map_tiles}, {thread_tile}, {thread_block_size}, {
                                        explicit_mem_move}, {remainder_loop}: the transformations numerically verify"
                                    print(s)
                                    f.write(s + "\n")
                            else:
                                s = f"For config {work_map_tiles}, {thread_tile}, {thread_block_size}, {
                                    explicit_mem_move}, {remainder_loop}: no reference result"
                                print(s)

                            time = auto_tile_util.run_and_measure_time(
                                kernel_sdfg, inputs)
                            print(f"Transformed SDFG: {time} ms")
                            solved_flops = auto_tile_util.solve(flops, inputs)
                            solved_mem_access = auto_tile_util.solve(
                                mem_access, inputs)

                            if flops == 0:
                                percentage_of_peak = auto_tile_util.percentage_bandwidth(
                                    time, solved_mem_access, peak_bandwidth)
                            else:
                                percentage_of_peak = auto_tile_util.percentage_peak(
                                    time, solved_flops, solved_mem_access, peak_flops, peak_bandwidth)

                            if best_params == None or percentage_of_peak > threshold:
                                best_params = ((work_map_tiles, thread_tile, thread_block_size,
                                                explicit_mem_move, remainder_loop), time, percentage_of_peak)
                            if percentage_of_peak > threshold:
                                b = True

                            print(
                                f"Trandformed SDFG achieves {percentage_of_peak:.2f}% of the peak wrt. roofline model")
                            print(f"{kernel_entry} was transformed using parameters {
                                best_params}")
                            f.write(
                                f"Trandformed SDFG achieves {percentage_of_peak:.2f}% of the peak wrt. roofline model\n")

                        except Exception as ex:
                            print(
                                f"Transformations fail for config {work_map_tiles}, {thread_tile}, {thread_block_size}, {explicit_mem_move}, {remainder_loop}")
                            print("Exception:", ex)

    if best_params:
        apply_using_params(
            sdfg=sdfg,
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
            reference_result=reference_result,
            inputs=inputs,
            output=output,
            verbose=verbose,
            work_on_copy=False,
            save_steps=False,
            save_individual_kernels=False
        )

    return best_params


f = open("o.txt", "w")

kid = 0


def auto_apply(sdfg: SDFG,
               work_map_tiling_params: List[Tuple],
               thread_coarsening_params: List[Tuple],
               thread_block_params: List[Tuple],
               apply_explicit_memory_transfers: List[bool],
               apply_remainder_loop: List[bool],
               reference_result=None,
               inputs: Dict[Type[str], Any] = dict(),
               output: cp.array = None,
               verbose=False,
               save_steps=False,
               save_individual_kernels=False,
               re_apply=False
               ):
    # Any map that is GPU_Device is transformed applying the
    # AMM-guided transformations.
    # Anything inside the map is treated as work to be tiled
    # If reference result is not None then the transformed SDFG
    # is called and checked with the reference result (must be array)
    threshold = 85.0

    kperf = dict()

    sdfg_name = sdfg._name

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

    for state in sdfg.states():
        kernel_entry_guids = []
        for node in state.nodes():
            if isinstance(node, nodes.MapEntry) and \
                    node.map.schedule == ScheduleType.GPU_Device:
                kernel_entry_guids.append(node.guid)

        for kernel_entry_guid in kernel_entry_guids:
            kernel_entry = find_node_in_state_by_cond(
                state, lambda n: n.guid == kernel_entry_guid)
            flops, mem_access = auto_tile_util.get_flops_and_mem_access(
                sdfg, state, kernel_entry)
            if verbose:
                print(f"FLOPs as a symbolic expression: {flops}")
                print(f"Bytes accessed as a symbolic expression: {mem_access}")

            peak_flops, mem_bandwidth = peak_flops_and_badwidth_nvidia.get_peak_flops_and_mem_bandwidth(
                0)
            peak_flops *= 1e9
            mem_bandwidth *= 1e9
            flops, mem_access = auto_tile_util.get_flops_and_mem_access(
                sdfg, state, kernel_entry)

            if not kernel_entry.guid in kperf:
                try:
                    best_config = apply_using_params(
                        sdfg=sdfg,
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
                        reference_result=reference_result,
                        inputs=inputs,
                        output=output,
                        verbose=verbose,
                        work_on_copy=True,
                        save_steps=save_steps,
                        save_individual_kernels=save_individual_kernels
                    )
                    kperf[kernel_entry.guid] = (f"{kernel_entry.guid}_auto_tiled.sdfg", best_config[0], str(
                        flops), str(mem_access), float(best_config[1]), float(best_config[2]))
                except Exception as e:
                    best_config = None
                    print("Exception: on transforming {kernel_entry}:", e)
                    kperf[kernel_entry.guid] = (f"{kernel_entry.guid}_auto_tiled.sdfg", None, str(
                        flops), str(mem_access), float(-1.0), float("nan"))

            if verbose:
                print("Best config: ",
                      best_config[:-1] if best_config else "none")
                print("Percentage of the peak: ",
                      best_config[-1] if best_config else "none")

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
