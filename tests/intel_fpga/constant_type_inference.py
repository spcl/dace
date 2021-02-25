# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.

# With this test we want to ensure that type inference works
# well with constants (i.e., no auto are generated)

import dace
import numpy as np
import argparse

N = dace.symbol('N')
CONSTANT_ARRAY = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.float32)
CONSTANT_VALUE = float(1)


def make_sdfg():
    sdfg = dace.SDFG("constant_type_inference")

    sdfg.add_array("output", shape=[N], dtype=dace.float32)

    sdfg.add_array("device_output",
                   shape=[N],
                   dtype=dace.float32,
                   storage=dace.dtypes.StorageType.FPGA_Global,
                   transient=True)

    ###########################################################################
    # Copy data from FPGA
    copy_out_state = sdfg.add_state("copy_to_host")

    device_output = copy_out_state.add_read("device_output")
    host_output = copy_out_state.add_write("output")

    copy_out_state.add_memlet_path(device_output,
                                   host_output,
                                   memlet=dace.Memlet(f"{host_output}[0:N]"))

    ########################################################################
    # FPGA, First State

    # increment constant array of 1 elements

    fpga_state = sdfg.add_state("fpga_state")

    out = fpga_state.add_write("device_output")
    map_entry, map_exit = fpga_state.add_map(
        "increment_map", {"i": "0:N"}, schedule=dace.ScheduleType.FPGA_Device)

    # Force type inference for constant array
    tasklet = fpga_state.add_tasklet(
        "increment_tasklet", {}, {"out"}, "incr = constant_value\n"
        "tmp = constant_array[i]\n"
        "out = tmp + incr")

    fpga_state.add_memlet_path(map_entry, tasklet, memlet=dace.Memlet())
    fpga_state.add_memlet_path(tasklet,
                               map_exit,
                               out,
                               src_conn="out",
                               memlet=dace.Memlet("device_output[i]"))

    sdfg.add_edge(fpga_state, copy_out_state, dace.sdfg.sdfg.InterstateEdge())
    sdfg.fill_scope_connectors()
    sdfg.validate()
    return sdfg


if __name__ == "__main__":

    sdfg = make_sdfg()
    sdfg.add_constant('constant_array', CONSTANT_ARRAY)
    sdfg.add_constant('constant_value', CONSTANT_VALUE)

    out = dace.ndarray([CONSTANT_ARRAY.size], dtype=dace.float32)
    sdfg(N=CONSTANT_ARRAY.size, output=out)
    ref = CONSTANT_ARRAY + CONSTANT_VALUE
    diff = np.linalg.norm(ref - out) / CONSTANT_ARRAY.size
    if diff <= 1e-5:
        print("==== Program end ====")
    else:
        print("==== Program Error! ====")
