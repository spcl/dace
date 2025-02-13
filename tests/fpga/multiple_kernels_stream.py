# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.

# Two FPGA states that communicate through stream

import dace
import numpy as np
import argparse

from dace.memlet import Memlet


def make_sdfg(dtype=dace.float32):
    sdfg = dace.SDFG("multiple_kernels_streams")

    ###########################################################################
    # Copy data to FPGA

    copy_in_state = sdfg.add_state("copy_to_device")

    sdfg.add_array("a", shape=[1], dtype=dtype)
    sdfg.add_array("c", shape=[1], dtype=dtype)

    in_host_a = copy_in_state.add_read("a")
    in_host_c = copy_in_state.add_read("c")

    sdfg.add_array("device_a", shape=[1], dtype=dtype, storage=dace.dtypes.StorageType.FPGA_Global, transient=True)
    sdfg.add_array("device_c", shape=[1], dtype=dtype, storage=dace.dtypes.StorageType.FPGA_Global, transient=True)

    in_device_a = copy_in_state.add_write("device_a")
    in_device_c = copy_in_state.add_write("device_c")

    copy_in_state.add_memlet_path(in_host_a, in_device_a, memlet=Memlet(f"{in_host_a}[0]"))
    copy_in_state.add_memlet_path(in_host_c, in_device_c, memlet=Memlet(f"{in_host_c}[0]"))

    ###########################################################################
    # Copy data from FPGA
    copy_out_state = sdfg.add_state("copy_to_host")

    device_c = copy_out_state.add_read("device_c")
    host_c = copy_out_state.add_write("c")

    copy_out_state.add_memlet_path(device_c, host_c, memlet=Memlet(f"{host_c}[0]"))

    ########################################################################
    # FPGA, First State

    # create the stream for connecting the two states
    sdfg.add_stream('device_b_stream', dtype, buffer_size=32, storage=dace.dtypes.StorageType.FPGA_Local)

    fpga_state_0 = sdfg.add_state("fpga_state_0")

    a_in = fpga_state_0.add_read("device_a")
    b_stream_out = fpga_state_0.add_write("device_b_stream")

    state_0_tasklet = fpga_state_0.add_tasklet('state_0_tasklet', ['inCon'], ['outCon'], 'outCon = inCon + 1')

    fpga_state_0.add_memlet_path(a_in, state_0_tasklet, dst_conn='inCon', memlet=dace.Memlet(f"{a_in}[0]"))

    fpga_state_0.add_memlet_path(state_0_tasklet,
                                 b_stream_out,
                                 src_conn='outCon',
                                 memlet=dace.Memlet(f"{b_stream_out}[0]", dynamic=True))

    ########################################################################
    # FPGA, Second State

    fpga_state_1 = sdfg.add_state("fpga_state_1")

    b_stream_in = fpga_state_1.add_read("device_b_stream")
    c_out = fpga_state_1.add_write("device_c")

    state_1_tasklet = fpga_state_1.add_tasklet('state_1_tasklet', ['inCon'], ['outCon'], 'outCon = inCon + 1')

    fpga_state_1.add_memlet_path(b_stream_in,
                                 state_1_tasklet,
                                 dst_conn='inCon',
                                 memlet=dace.Memlet(f"{b_stream_in}[0]", dynamic=True))

    fpga_state_1.add_memlet_path(state_1_tasklet, c_out, src_conn='outCon', memlet=dace.Memlet(f"{c_out.data}[0]"))

    ######################################
    # Interstate edges
    sdfg.add_edge(copy_in_state, fpga_state_0, dace.sdfg.sdfg.InterstateEdge())
    sdfg.add_edge(fpga_state_0, fpga_state_1, dace.sdfg.sdfg.InterstateEdge())
    sdfg.add_edge(fpga_state_1, copy_out_state, dace.sdfg.sdfg.InterstateEdge())

    #########
    # Validate
    sdfg.fill_scope_connectors()
    sdfg.validate()
    return sdfg


if __name__ == "__main__":

    sdfg = make_sdfg()

    comp = sdfg.compile()

    a = np.random.rand(1).astype(np.float32)
    c = np.random.rand(1).astype(np.float32)
    ref_a = a[0]
    ref_c = c[0]
    print(a)
    print(c)
    comp(a=a, c=c)

    diff = ((ref_a + 2) - c) / c
    print("Ref_c {}, new c {}".format(ref_c, c))
    print("Difference:", diff)
    if diff <= 1e-5:
        print("==== Program end ====")
    else:
        print("==== Program Error! ====")

    exit(0 if diff <= 1e-5 else 1)
