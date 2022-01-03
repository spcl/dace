# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np

from dace import nodes
from dace.transformation.dataflow import MapFusion


@dace.program
def rw_data_race(A: dace.float32[10, 10], B: dace.float32[10, 10]):
    tmp = A[:-1, :-1] + A[1:, 1:]
    B[1:-1, 1:-1] = tmp[:-1, :-1] + tmp[:-1, :-1]


def test_rw_data_race():
    sdfg = rw_data_race.to_sdfg(coarsen=True)
    sdfg.apply_transformations_repeated(MapFusion)
    map_entry_nodes = [
        n for n, _ in sdfg.all_nodes_recursive()
        if isinstance(n, nodes.MapEntry)
    ]
    assert (len(map_entry_nodes) > 1)


if __name__ == "__main__":
    test_rw_data_race()