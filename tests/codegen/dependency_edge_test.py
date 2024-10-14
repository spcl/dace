# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.

import pytest
import dace
import numpy as np

@pytest.mark.parametrize("reverse", [True, False])
def test_mapped_dependency_edge(reverse):
    """ Tests dependency edges in a map scope """

    sdfg = dace.SDFG("mapped_dependency_edge")
    state = sdfg.add_state()

    sdfg.add_array("A", shape=[2], dtype=dace.int32)
    sdfg.add_array("B", shape=[2], dtype=dace.int32)
    sdfg.add_transient("tmp_A", shape=[1], dtype=dace.int32)
    sdfg.add_transient("tmp_B", shape=[1], dtype=dace.int32)

    map_entry, map_exit = state.add_map("map", {"i": "0:2"}, schedule=dace.dtypes.ScheduleType.Sequential)
    map_entry.add_in_connector("IN_A")
    map_entry.add_in_connector("IN_B")
    map_entry.add_out_connector("OUT_A")
    map_entry.add_out_connector("OUT_B")
    map_exit.add_in_connector("IN_A")
    map_exit.add_out_connector("OUT_A")

    A1 = state.add_read("A")
    A2 = state.add_write("A")
    A3 = state.add_write("A")
    A4 = state.add_write("A")
    B = state.add_read("B")
    tmp_A = state.add_write("tmp_A")
    tmp_B = state.add_write("tmp_B")

    state.add_edge(A1, None, map_entry, "IN_A", dace.Memlet("A[0:2]"))
    state.add_edge(B, None, map_entry, "IN_B", dace.Memlet("B[0:2]"))

    state.add_edge(map_entry, "OUT_A", tmp_A, None, dace.Memlet("A[i]"))
    state.add_edge(map_entry, "OUT_B", tmp_B, None, dace.Memlet("B[i]"))

    state.add_edge(tmp_A, None, A2, None, dace.Memlet("tmp_A[0] -> [((i+1)%2)]"))
    if not reverse:
      state.add_edge(A2, None, tmp_B, None, dace.Memlet()) # Dependency Edge
    state.add_edge(A2, None, map_exit, "IN_A", dace.Memlet("A[0:2]"))

    state.add_edge(tmp_B, None, A3, None, dace.Memlet("tmp_B[0] -> [((i+1)%2)]"))
    if reverse:
      state.add_edge(A3, None, tmp_A, None, dace.Memlet()) # Dependency Edge
    state.add_edge(A3, None, map_exit, "IN_A", dace.Memlet("A[0:2]"))

    state.add_edge(map_exit, "OUT_A", A4, None, dace.Memlet("A[0:2]"))

    sdfg.validate()
    a = np.random.randint(0, 100, 2).astype(np.int32)
    b = np.random.randint(0, 100, 2).astype(np.int32)
    sdfg(A=a, B=b)

    if reverse:
      assert a[0] == a[1]
    else:
      assert a[0] == b[1] and a[1] == b[0]


if __name__ == "__main__":
    test_mapped_dependency_edge(False)
    test_mapped_dependency_edge(True)
  
