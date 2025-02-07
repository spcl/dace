# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.

import dace
import numpy as np


def test():
    N = dace.symbol("N")

    sdfg = dace.SDFG("parallel_sections")

    sdfg.add_array("array_in", (2 * N, ), dace.dtypes.int32)
    sdfg.add_array("array_out", (N, ), dace.dtypes.int32)

    sdfg.add_stream("fifo_in_a", dace.dtypes.int32, 1, transient=True)
    sdfg.add_stream("fifo_in_b", dace.dtypes.int32, 1, transient=True)
    sdfg.add_stream("fifo_out", dace.dtypes.int32, 1, transient=True)

    state = sdfg.add_state("sections")

    array_in = state.add_access("array_in")
    array_out = state.add_access("array_out")

    fifo_in_a_0 = state.add_access("fifo_in_a")
    fifo_in_b_0 = state.add_access("fifo_in_b")
    fifo_in_a_1 = state.add_access("fifo_in_a")
    fifo_in_b_1 = state.add_access("fifo_in_b")
    fifo_out_0 = state.add_access("fifo_out")
    fifo_out_1 = state.add_access("fifo_out")

    ###########################################################################
    # First processing element: reads from memory into a stream

    read_a_entry, read_a_exit = state.add_map("read_map_a", {"i": "0:N"}, schedule=dace.dtypes.ScheduleType.Sequential)
    read_a_tasklet = state.add_tasklet("read_a", {"from_memory"}, {"to_stream"}, "to_stream = from_memory")

    # Inner edges
    state.add_edge(read_a_entry, None, read_a_tasklet, "from_memory", dace.memlet.Memlet.simple(array_in, "i"))
    state.add_edge(read_a_tasklet, "to_stream", read_a_exit, None, dace.memlet.Memlet.simple(fifo_in_a_0, "0"))

    # Outer edges
    state.add_edge(array_in, None, read_a_entry, None, dace.memlet.Memlet.simple(array_in, "0:N"))
    state.add_edge(read_a_exit, None, fifo_in_a_0, None, dace.memlet.Memlet.simple(fifo_in_a_0, "0"))

    ###########################################################################
    # Second processing element: reads from memory into a stream

    read_b_entry, read_b_exit = state.add_map("read_map_b", {"i": "N:2*N"},
                                              schedule=dace.dtypes.ScheduleType.Sequential)
    read_b_tasklet = state.add_tasklet("read_b", {"from_memory"}, {"to_stream"}, "to_stream = from_memory")

    # Inner edges
    state.add_edge(read_b_entry, None, read_b_tasklet, "from_memory", dace.memlet.Memlet.simple(array_in, "i"))
    state.add_edge(read_b_tasklet, "to_stream", read_b_exit, None, dace.memlet.Memlet.simple(fifo_in_b_0, "0"))

    # Outer edges
    state.add_edge(array_in, None, read_b_entry, None, dace.memlet.Memlet.simple(array_in, "0:N"))
    state.add_edge(read_b_exit, None, fifo_in_b_0, None, dace.memlet.Memlet.simple(fifo_in_b_0, "0"))

    ###########################################################################
    # Third processing element: reads from both input streams, adds the
    # numbers, the writes it to the output stream

    compute_entry, compute_exit = state.add_map("compute_map", {"i": "0:N"},
                                                schedule=dace.dtypes.ScheduleType.Sequential)
    compute_tasklet = state.add_tasklet("compute", {"a", "b"}, {"c"}, "c = a + b")

    # Inner edges
    state.add_edge(compute_entry, None, compute_tasklet, "a", dace.memlet.Memlet.simple(fifo_in_a_1, "0"))
    state.add_edge(compute_entry, None, compute_tasklet, "b", dace.memlet.Memlet.simple(fifo_in_b_1, "0"))
    state.add_edge(compute_tasklet, "c", compute_exit, None, dace.memlet.Memlet.simple(fifo_out_0, "0"))

    # Outer edges
    state.add_edge(fifo_in_a_1, None, compute_entry, None, dace.memlet.Memlet.simple(fifo_in_a_1, "0"))
    state.add_edge(fifo_in_b_1, None, compute_entry, None, dace.memlet.Memlet.simple(fifo_in_b_1, "0"))
    state.add_edge(compute_exit, None, fifo_out_0, None, dace.memlet.Memlet.simple(fifo_out_0, "0"))

    ###########################################################################
    # Fourth processing element: reads from stream into an array

    write_entry, write_exit = state.add_map("write_map", {"i": "0:N"}, schedule=dace.dtypes.ScheduleType.Sequential)
    write_tasklet = state.add_tasklet("write", {"from_stream"}, {"to_memory"}, "to_memory = from_stream")

    # Inner edges
    state.add_edge(write_entry, None, write_tasklet, "from_stream", dace.memlet.Memlet.simple(fifo_out_1, "0"))
    state.add_edge(write_tasklet, "to_memory", write_exit, None, dace.memlet.Memlet.simple(array_out, "i"))

    # Outer edges
    state.add_edge(fifo_out_1, None, write_entry, None, dace.memlet.Memlet.simple(fifo_out_1, "0"))
    state.add_edge(write_exit, None, array_out, None, dace.memlet.Memlet.simple(array_out, "0:N"))

    ###########################################################################
    N = 1024
    array_in = np.ndarray([2 * N], np.int32)
    array_in[:N] = range(0, N)
    array_in[N:] = range(0, N)
    array_out = np.ndarray([N], np.int32)
    sdfg(array_in=array_in, array_out=array_out, N=N)

    for i, val in enumerate(array_out):
        if val != 2 * i:
            print(i, val)
            raise ValueError


if __name__ == '__main__':
    test()
