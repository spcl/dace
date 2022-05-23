# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.

# DaCe program with two state that will be generated as two kernels

import dace
import numpy as np
import argparse

from dace.memlet import Memlet


def make_sdfg(dtype=dace.float32):
    sdfg = dace.SDFG("multiple_kernels")

    ###########################################################################
    # Copy data to FPGA

    copy_in_state = sdfg.add_state("copy_to_device")

    sdfg.add_array("a", shape=[1], dtype=dtype)
    sdfg.add_array("b", shape=[1], dtype=dtype)
    sdfg.add_array("c", shape=[1], dtype=dtype)

    in_host_a = copy_in_state.add_read("a")
    in_host_b = copy_in_state.add_read("b")
    in_host_c = copy_in_state.add_read("c")

    sdfg.add_array("device_a", shape=[1], dtype=dtype, storage=dace.dtypes.StorageType.FPGA_Global, transient=True)
    sdfg.add_array("device_b", shape=[1], dtype=dtype, storage=dace.dtypes.StorageType.FPGA_Global, transient=True)
    sdfg.add_array("device_c", shape=[1], dtype=dtype, storage=dace.dtypes.StorageType.FPGA_Global, transient=True)

    in_device_a = copy_in_state.add_write("device_a")
    in_device_b = copy_in_state.add_write("device_b")
    in_device_c = copy_in_state.add_write("device_c")

    copy_in_state.add_memlet_path(in_host_a, in_device_a, memlet=Memlet.simple(in_host_a, "0"))
    copy_in_state.add_memlet_path(in_host_b, in_device_b, memlet=Memlet.simple(in_host_b, "0"))
    copy_in_state.add_memlet_path(in_host_c, in_device_c, memlet=Memlet.simple(in_host_c, "0"))

    ###########################################################################
    # Copy data from FPGA
    copy_out_state = sdfg.add_state("copy_to_host")

    device_c = copy_out_state.add_read("device_c")
    host_c = copy_out_state.add_write("c")

    device_b = copy_out_state.add_read("device_b")
    host_b = copy_out_state.add_write("b")

    copy_out_state.add_memlet_path(device_c, host_c, memlet=Memlet.simple(host_c, "0"))

    copy_out_state.add_memlet_path(device_b, host_b, memlet=Memlet.simple(host_b, "0"))

    ########################################################################
    # FPGA, First State

    fpga_state_0 = sdfg.add_state("fpga_state_0")

    a_in = fpga_state_0.add_read("device_a")
    b_out = fpga_state_0.add_write("device_b")

    state_0_tasklet = fpga_state_0.add_tasklet('state_0_tasklet', ['inCon'], ['outCon'], 'outCon = inCon + 1')

    fpga_state_0.add_memlet_path(a_in, state_0_tasklet, dst_conn='inCon', memlet=dace.Memlet.simple(a_in.data, '0'))

    fpga_state_0.add_memlet_path(state_0_tasklet, b_out, src_conn='outCon', memlet=dace.Memlet.simple(b_out.data, '0'))

    ########################################################################
    # FPGA, Second State

    fpga_state_1 = sdfg.add_state("fpga_state_1")

    b_in = fpga_state_1.add_read("device_b")
    c_out = fpga_state_1.add_write("device_c")

    state_1_tasklet = fpga_state_1.add_tasklet('state_1_tasklet', ['inCon'], ['outCon'], 'outCon = inCon + 1')

    fpga_state_1.add_memlet_path(b_in, state_1_tasklet, dst_conn='inCon', memlet=dace.Memlet.simple(b_in.data, '0'))

    fpga_state_1.add_memlet_path(state_1_tasklet, c_out, src_conn='outCon', memlet=dace.Memlet.simple(c_out.data, '0'))

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
    b = np.random.rand(1).astype(np.float32)
    c = np.random.rand(1).astype(np.float32)
    ref_a = a[0]
    ref_b = b[0]
    ref_c = c[0]
    comp(a=a, b=b, c=c)

    diff1 = ((ref_a + 1) - b) / b
    diff2 = ((ref_a + 2) - c) / c
    if diff1 <= 1e-5 and diff2 <= 1e-5:
        print("==== Program end ====")
    else:
        raise Exception("==== Program Error! ====")
