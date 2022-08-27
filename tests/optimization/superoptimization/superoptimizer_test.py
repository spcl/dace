import sched
import dace
import copy

import dace.transformation.helpers as xfh

from dace.transformation.dataflow import MapTiling
from dace.optimization.superoptimization.utils import cutout_map, map_levels
from dace.optimization.superoptimization.enumerator.map_schedule_enumerator import _expand_all_maps, _collapse_all_maps
from dace.optimization.superoptimization.superoptimizer import Superoptimizer


def some_schedule(map_cutout: dace.SDFG):
    # Transform copy of map cutout
    map_cutout_transformed = copy.deepcopy(map_cutout)

    _expand_all_maps(map_cutout_transformed)

    outermost_map = None
    for node in map_cutout_transformed.start_state.nodes():
        if not isinstance(node, dace.nodes.MapEntry) or xfh.get_parent_map(map_cutout_transformed.start_state,
                                                                           node) is not None:
            continue

        outermost_map = node
        break

    MapTiling.apply_to(sdfg=map_cutout_transformed, map_entry=outermost_map, options={"tile_sizes": (8, )})

    _collapse_all_maps(map_cutout_transformed)

    schedule = []
    for trans in map_cutout_transformed.transformation_hist:
        schedule.append(trans.to_json())

    return schedule


@dace.program
def map_2d(A: dace.float64[10, 20], B: dace.float64[10, 20]):
    for i, j in dace.map[0:10, 0:20]:
        with dace.tasklet:
            a << A[i, j]
            b >> B[i, j]

            b = a * a


def test_apply_map_schedule():
    sdfg = map_2d.to_sdfg()
    sdfg.simplify()

    Superoptimizer._canonicalize_sdfg(sdfg, device_type=dace.DeviceType.CPU)

    outermost_map = None
    for node in sdfg.start_state.nodes():
        if not isinstance(node, dace.nodes.MapEntry) or xfh.get_parent_map(sdfg.start_state, node) is not None:
            continue

        outermost_map = node
        break
    map_cutout = cutout_map(sdfg.start_state, map_entry=outermost_map, make_copy=False)

    schedule = some_schedule(map_cutout)
    Superoptimizer._apply_map_schedule(sdfg, sdfg.start_state, cutout=map_cutout, transformations=schedule)

    levels = map_levels(map_cutout)
    maps = []
    map_entry = None
    while map_entry in levels:
        map_entry = levels[map_entry]
        maps.append(map_entry)

    assert len(maps) == 2

    outer_map = maps[0]
    assert len(outer_map.map.params) == 1
    assert outer_map.map.params[0] == "tile_i"

    inner_map = maps[1]
    assert len(inner_map.map.params) == 2
    assert inner_map.map.params == ["i", "j"]

    sdfg.validate()
