import math
import itertools

from dace import SDFG, data, nodes, ScheduleType

import dace.optimization.superoptimization.utils as utils

import dace.transformation.helpers as xfh
from dace.transformation.dataflow import MapDimShuffle, MapExpansion, MapCollapse, MapTiling, InLocalStorage, OutLocalStorage, MapSchedule, Vectorization, AccumulateTransient


def map_schedule_enumerator(map: SDFG) -> SDFG:
    map_tmp = map.to_json()

    in_arrays, out_arrays = _arrays(map)
    all_params = utils.map_params(map)
    for permuted_params in _permutations(all_params):
        permuted_map = SDFG.from_json(map_tmp)
        _apply_permutation(permuted_map, permutation=permuted_params)

        permuted_map_tmp = permuted_map.to_json()
        for tiling in _tilings(permuted_params):
            tiled_map = SDFG.from_json(permuted_map_tmp)
            _apply_tiling(tiled_map, tiling)
            _expand_all_maps(tiled_map)

            tiled_map_tmp = tiled_map.to_json()
            expanded_params = utils.map_params(tiled_map)
            for local_storage in _local_storage(expanded_params, in_arrays, out_arrays):
                local_storage_map_ = SDFG.from_json(tiled_map_tmp)

                if not _apply_local_storage(local_storage_map_, local_storage=local_storage):
                    continue
                _collapse_all_maps(local_storage_map_)

                collapsed_params = utils.map_params(local_storage_map_)
                local_storage_map_tmp = local_storage_map_.to_json()
                for parallelization in _parallelizations(collapsed_params):
                    scheduled_map = SDFG.from_json(local_storage_map_tmp)
                    _apply_parallelization(scheduled_map, parallelization)

                    scheduled_map_tmp = scheduled_map.to_json()
                    for vec_len in _vectorizations():
                        final_map = SDFG.from_json(scheduled_map_tmp)
                        if _apply_vectorization(final_map, vec_len):
                            try:
                                final_map.validate()
                                schedule_desc = f"{permuted_params}:{tiling}:{local_storage}:{parallelization}:{vec_len}"
                                yield final_map, schedule_desc
                            except:
                                continue


def _permutations(all_params):
    perms = [list(itertools.permutations(level)) for level in all_params]
    perms = itertools.product(*perms)
    for perm in perms:
        yield perm


def _tilings(all_params, tile_sizes_range=range(0, 9)):
    tile_sizes = [2**k for k in tile_sizes_range]
    tilings = itertools.product(tile_sizes, repeat=len(all_params))

    for tiling in tilings:
        strategy = {}
        for i, group in enumerate(all_params):
            for param in group:
                strategy[param] = tiling[i]

        yield strategy


def _local_storage(all_params, in_arrays, out_arrays):
    arrays = list(in_arrays)
    arrays.extend(out_arrays)

    op = {"in": {}, "out": {}}
    if len(all_params) <= 1 or len(arrays) == 0:
        yield op
        return

    options = []
    for array in arrays:
        array_options = []
        zeros = [0] * (len(all_params) - 1)
        array_options.append(zeros)

        for i in range(len(all_params) - 1):
            bin = [0] * (len(all_params) - 1)
            bin[i] = 1
            array_options.append(bin)

        options.append(array_options)

    options = itertools.product(*options)

    for option in options:
        op = {"in": {}, "out": {}}
        for i in range(len(arrays)):
            array = arrays[i]
            if i < len(in_arrays):
                op["in"][array] = option[i]
            else:
                op["out"][array] = option[i]

        yield op


def _parallelizations(all_params):
    strategies = []
    strategies.append([0] * len(all_params))
    for i, group in enumerate(all_params):
        for j in range(1, len(group) + 1):
            par = [0] * len(all_params)
            par[i] = j
            strategies.append(par)

    for strategy in strategies:
        yield strategy


def _vectorizations():
    for vec_len in [1, 2, 4, 8, 16]:
        yield vec_len


def _apply_permutation(map: SDFG, permutation):
    levels = utils.map_levels(map)

    i = 0
    map_entry = None
    while map_entry in levels:
        map_entry = levels[map_entry]

        MapDimShuffle.apply_to(sdfg=map,
                               map_entry=map_entry,
                               options={"parameters": list(permutation[i])},
                               save=True,
                               verify=False)
        i = i + 1


def _apply_tiling(map: SDFG, tiling):
    levels = utils.map_levels(map)

    map_entry = None
    while map_entry in levels:
        map_entry = levels[map_entry]

        tile_sizes = [tiling[param] for param in map_entry.map.params]
        non_trivial = False
        for tile in tile_sizes:
            if tile > 1:
                non_trivial = True
                break

        if non_trivial:
            MapTiling.apply_to(sdfg=map,
                               options={
                                   "tile_sizes": tile_sizes,
                                   "tile_trivial": False
                               },
                               map_entry=map_entry,
                               save=True,
                               verify=False)


def _apply_local_storage(map: SDFG, local_storage):
    levels = utils.map_levels(map)

    levels_flat = []
    map_entry = None
    while map_entry in levels:
        map_entry = levels[map_entry]
        levels_flat.append(map_entry)

    in_local_storage = local_storage["in"]
    for array in in_local_storage:
        desc = in_local_storage[array]
        for i, flag in enumerate(desc):
            if flag == 0:
                continue

            outer_map_entry = levels_flat[i]
            inner_map_entry = levels_flat[i + 1]

            InLocalStorage.apply_to(
                sdfg=map,
                node_a=outer_map_entry,
                node_b=inner_map_entry,
                options={"array": array},
                save=True,
                verify=False,
            )

    out_local_storage = local_storage["out"]
    for array in out_local_storage:
        desc = out_local_storage[array]
        for i, flag in enumerate(desc):
            if flag == 0:
                continue

            outer_map_exit = map.start_state.exit_node(levels_flat[i])
            inner_map_exit = map.start_state.exit_node(levels_flat[i + 1])

            xform = OutLocalStorage()
            xform._sdfg = map
            xform.state_id = map.node_id(map.start_state)
            xform.node_a = inner_map_exit
            xform.node_b = outer_map_exit
            xform.array = array
            if xform.can_be_applied(map.start_state, sdfg=map, expr_index=0):
                OutLocalStorage.apply_to(
                    sdfg=map,
                    node_a=inner_map_exit,
                    node_b=outer_map_exit,
                    options={"array": array},
                    save=True,
                    verify=False,
                )
            else:
                xform = AccumulateTransient()
                xform._sdfg = map
                xform.state_id = map.node_id(map.start_state)
                xform.map_exit = inner_map_exit
                xform.outer_map_exit = outer_map_exit
                xform.array = array
                if xform.can_be_applied(map.start_state, sdfg=map, expr_index=0):
                    AccumulateTransient.apply_to(sdfg=map,
                                                 map_exit=inner_map_exit,
                                                 outer_map_exit=outer_map_exit,
                                                 options={"array": array},
                                                 save=True,
                                                 verify=False)
                else:
                    return False

    return True


def _apply_parallelization(map: SDFG, parallelization):
    levels = utils.map_levels(map)

    map_entry = None
    i = 0
    while map_entry in levels:
        map_entry = levels[map_entry]

        strategy = parallelization[i]
        if strategy == 0:
            schedule_type = ScheduleType.Sequential
            collapse = 1
        else:
            schedule_type = ScheduleType.CPU_Multicore
            collapse = strategy

        MapSchedule.apply_to(sdfg=map,
                             map_entry=map_entry,
                             options={
                                 "schedule_type": str(schedule_type),
                                 "collapse": collapse
                             },
                             save=True,
                             verify=False)
        i = i + 1


def _apply_vectorization(map: SDFG, vector_len: int):
    if vector_len == 1:
        return True

    levels = utils.map_levels(map)
    map_entry = None
    while map_entry in levels:
        map_entry = levels[map_entry]

    xform = Vectorization()
    xform._sdfg = map
    xform.state_id = map.node_id(map.start_state)
    xform.map_entry = map_entry
    if xform.can_be_applied(map.start_state, sdfg=map, expr_index=0):
        Vectorization.apply_to(sdfg=map,
                               map_entry=map_entry,
                               options={"vector_len": vector_len},
                               save=True,
                               verify=False)
        return True

    return False


def _expand_all_maps(map: SDFG):
    levels = utils.map_levels(map)
    map_entry = None
    while map_entry in levels:
        map_entry = levels[map_entry]
        if len(map_entry.map.params) > 1:
            MapExpansion.apply_to(sdfg=map, map_entry=map_entry, save=True, verify=False)


def _collapse_all_maps(map: SDFG):
    levels = utils.map_levels(map)

    map_entry = None
    levels_rev = []
    while map_entry in levels:
        map_entry = levels[map_entry]
        levels_rev.append(map_entry)
    levels_rev.reverse()

    inner = levels_rev[0]
    for i in range(1, len(levels_rev)):
        outer = levels_rev[i]

        xform = MapCollapse()
        xform._sdfg = map
        xform.state_id = map.node_id(map.start_state)
        xform.outer_map_entry = outer
        xform.inner_map_entry = inner

        if xform.can_be_applied(map.start_state, sdfg=map, expr_index=0):
            inner, _ = MapCollapse.apply_to(sdfg=map,
                                            outer_map_entry=outer,
                                            inner_map_entry=inner,
                                            save=True,
                                            verify=False)
        else:
            inner = outer


def _arrays(map: SDFG):
    parent_map_entry = None
    for node in map.start_state.nodes():
        if (not isinstance(node, nodes.MapEntry) or not xfh.get_parent_map(map.start_state, node) is None):
            continue

        parent_map_entry = node
        break

    in_arrays = set()
    for edge in map.start_state.in_edges(parent_map_entry):
        if not isinstance(edge.src, nodes.AccessNode):
            continue

        in_array = map.arrays[edge.data.data]
        if isinstance(in_array, data.Scalar):
            continue

        in_arrays.add(edge.data.data)

    parent_map_exit = map.start_state.exit_node(parent_map_entry)
    out_arrays = set()
    for edge in map.start_state.out_edges(parent_map_exit):
        if not isinstance(edge.dst, nodes.AccessNode):
            continue

        out_array = map.arrays[edge.data.data]
        if isinstance(out_array, data.Scalar):
            continue

        out_arrays.add(edge.data.data)

    return in_arrays, out_arrays
