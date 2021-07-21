# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np
from dace.transformation.dataflow import MapUnroll


def test_map_unroll():

    sdfg = dace.SDFG("unroll_schedule")
    sdfg.add_array("input_array", [3, 2, 1], dace.int32)
    sdfg.add_array("output_array", [1, 2, 3], dace.int32)
    state = sdfg.add_state("unroll_schedule")

    read = state.add_read("input_array")
    write = state.add_write("output_array")
    entry_outer, exit_outer = state.add_map("map_outer", {
        "i": "0:3",
        "j": "0:1"
    })
    entry_inner, exit_inner = state.add_map("map_inner", {"k": "0:2"})
    tasklet = state.add_tasklet("tasklet", {"x"}, {"y"}, "y = x + 1")
    state.add_memlet_path(read,
                          entry_outer,
                          entry_inner,
                          tasklet,
                          dst_conn="x",
                          memlet=dace.Memlet(f"input_array[i, k, j]"))
    state.add_memlet_path(tasklet,
                          exit_inner,
                          exit_outer,
                          write,
                          src_conn="y",
                          memlet=dace.Memlet(f"output_array[j, k, i]"))

    assert sdfg.apply_transformations_repeated(MapUnroll) == 4

    input_array = np.arange(0, 6, dtype=np.int32).reshape((3, 2, 1))
    output_array = np.empty((1, 2, 3), dtype=np.int32)
    sdfg(input_array=input_array, output_array=output_array)
    for i in range(3):
        for k in range(2):
            assert input_array[i, k, 0] + 1 == output_array[0, k, i]


if __name__ == "__main__":
    test_map_unroll()
