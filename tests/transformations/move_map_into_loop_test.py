import dace
import pytest
from dace.transformation.dataflow.tiling import MapTiling
from dace.transformation.interstate.move_map_into_loop import MoveMapIntoLoop
import copy

def _gen_sdfg() -> dace.SDFG:
    @dace.program
    def vadd(A: dace.float64[10], B: dace.float64[10], C: dace.float64[10]):
        for i in dace.map[0:10]:
            C[i] = A[i] + B[i]

    sdfg = vadd.to_sdfg()

    map_entries = set()
    for state in sdfg.all_states():
        for node in state.nodes():
            if isinstance(node, dace.nodes.MapEntry):
                map_entries.add(node)

    assert len(map_entries) == 1, "Expected exactly one map entry in the SDFG."

    MapTiling.apply_to(sdfg, map_entry=next(iter(map_entries)),
                       options={
                            'tile_sizes': [5],
                       })

    map_entries = set()
    for state in sdfg.all_states():
        for node in state.nodes():
            if isinstance(node, dace.nodes.MapEntry):
                map_entries.add(node)

    assert len(map_entries) == 2, "Expected exactly one map entry in the SDFG."

    # Get Outer and Inner Maps
    map_outer = next(iter(map_entries))
    map_inner = next(iter(map_entries - {map_outer}))
    if state.entry_node(map_outer) is not None:
        map_outer, map_inner = map_inner, map_outer

    map_exit_outer = state.exit_node(map_outer)
    map_exit_inner = state.exit_node(map_inner)

    print_tasklet = dace.nodes.Tasklet(
        'print_tasklet',
        inputs={},
        outputs={},
        code='printf("Hello from tasklet between map exits!\\n");',
        code_global='#include <stdio.h>',
        language=dace.dtypes.Language.CPP,
    )

    state.add_edge(map_exit_inner, None, print_tasklet, None,
                   dace.memlet.Memlet())
    state.add_edge(print_tasklet, None, map_exit_outer, None,
                  dace.memlet.Memlet())

    sdfg2 = copy.deepcopy(sdfg)
    MoveMapIntoLoop.apply_to(sdfg2, map_entry=map_inner)

    return sdfg, sdfg2


def test_tasklet_between_map_exits():
    sdfg_w_map, sdfg_w_loop = _gen_sdfg()
    sdfg_w_map.validate()
    sdfg_w_map.compile()
    sdfg_w_loop.validate()
    sdfg_w_loop.compile()

    A = dace.ndarray([10], dtype=dace.float64)
    B = dace.ndarray([10], dtype=dace.float64)
    C = dace.ndarray([10], dtype=dace.float64)
    C2 = dace.ndarray([10], dtype=dace.float64)
    # Compare many different SDFGs to each other

if __name__ == "__main__":
    test_tasklet_between_map_exits()
