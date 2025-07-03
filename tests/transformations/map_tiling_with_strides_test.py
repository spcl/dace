# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.

import dace
import numpy as np
from dace.transformation.dataflow import MapTiling
import pytest

N = dace.symbol('N')


def test_map_tiling_with_strides():
    s = 33

    @dace.program
    def vector_copy_strides(A: dace.uint32[N], B: dace.uint32[N]):
        for i in dace.map[0:N:s] @ dace.dtypes.ScheduleType.CPU_Multicore:
            A[i] = B[i]

    sdfg = vector_copy_strides.to_sdfg()

    state = sdfg.states()[0]
    sdfg_nodes = state.nodes()
    map_entry: dace.nodes.MapEntry = [n for n in sdfg_nodes if isinstance(n, dace.nodes.MapEntry)][0]

    tile_sizes = [32]
    MapTiling.apply_to(sdfg=sdfg,
                       options={
                           "prefix": "b",
                           "tile_sizes": tile_sizes,
                           "divides_evenly": False,
                           "tile_trivial": True,
                           "skew": False
                       },
                       map_entry=map_entry)
    inner_map_entry = map_entry
    outer_map_entry = state.entry_node(inner_map_entry)

    b_i = dace.symbol("b_i")
    inner_rangelist = [(b_i, dace.symbolic.SymExpr("Min(N - 1, b_i + 32*33 - 1)"), 33)]
    outer_rangelist = [(0, N - 1, 32 * 33)]
    inner_range = dace.subsets.Range(inner_rangelist)
    outer_range = dace.subsets.Range(outer_rangelist)

    sdfg.validate()
    assert inner_map_entry.map.range == inner_range
    assert outer_map_entry.map.range == outer_range


if __name__ == '__main__':
    test_map_tiling_with_strides()
