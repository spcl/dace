# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.

# The purpose of this test is to verify that gearboxing functions are created only
# once, even if reused multiple times (so they are not defined multiple times)
# There are three different kernels, each one read/write to intermediate access nodes using gearboxing
# each of them increment by 1 the data

# NOTE: this is a nice case where we need streams that cross states

import dace
import numpy as np
import argparse

from dace.memlet import Memlet
from dace.fpga_testing import fpga_test

N = dace.symbol("N")


def make_sdfg(dtype=dace.float32, vec_width=4):
    sdfg = dace.SDFG("multiple_veclen_conversions")

    ###########################################################################
    # Copy data to FPGA

    copy_in_state = sdfg.add_state("copy_to_device")

    sdfg.add_array("a", shape=[N], dtype=dtype)
    sdfg.add_array("d", shape=[N], dtype=dtype)

    in_host_a = copy_in_state.add_read("a")
    in_host_d = copy_in_state.add_read("d")
    vec_type = dace.vector(dtype, vec_width)

    sdfg.add_array("device_a", shape=[N], dtype=dtype, storage=dace.dtypes.StorageType.FPGA_Global, transient=True)
    sdfg.add_array("device_b",
                   shape=[N / vec_width],
                   dtype=vec_type,
                   storage=dace.dtypes.StorageType.FPGA_Global,
                   transient=True)
    sdfg.add_array("device_c",
                   shape=[N / vec_width],
                   dtype=vec_type,
                   storage=dace.dtypes.StorageType.FPGA_Global,
                   transient=True)
    sdfg.add_array("device_d", shape=[N], dtype=dtype, storage=dace.dtypes.StorageType.FPGA_Global, transient=True)

    in_device_a = copy_in_state.add_write("device_a")
    in_device_d = copy_in_state.add_write("device_d")

    copy_in_state.add_memlet_path(in_host_a, in_device_a, memlet=Memlet(f"{in_host_a}[0:N]"))
    copy_in_state.add_memlet_path(in_host_d, in_device_d, memlet=Memlet(f"{in_host_d}[0:N]"))

    ###########################################################################
    # Copy data from FPGA
    copy_out_state = sdfg.add_state("copy_to_host")

    device_d = copy_out_state.add_read("device_d")
    host_d = copy_out_state.add_write("d")

    copy_out_state.add_memlet_path(device_d, host_d, memlet=Memlet(f"{host_d}[0:N]"))

    ########################################################################
    # FPGA, First State

    # reads data, increment by 1 pack

    fpga_state_0 = sdfg.add_state("fpga_state_0")

    a_in = fpga_state_0.add_read("device_a")
    b_out = fpga_state_0.add_write("device_b")

    # local storage to read and increment data
    sdfg.add_array('vec_data',
                   shape=[vec_width],
                   dtype=dtype,
                   transient=True,
                   storage=dace.dtypes.StorageType.FPGA_Registers)
    sdfg.add_array('inc_data',
                   shape=[vec_width],
                   dtype=dtype,
                   transient=True,
                   storage=dace.dtypes.StorageType.FPGA_Registers)
    vect_data = fpga_state_0.add_access("vec_data")
    inc_data = fpga_state_0.add_access("inc_data")
    # Read the data

    map_entry, map_exit = fpga_state_0.add_map("read_A", {
        "n0": "0:{}/{}".format(N, vec_width),
    },
                                               schedule=dace.ScheduleType.FPGA_Device)
    read_map_entry, read_map_exit = fpga_state_0.add_map("unrolled_reads", {"n1": "0:{}".format(vec_width)},
                                                         schedule=dace.ScheduleType.FPGA_Device,
                                                         unroll=True)

    # In the innermost map we read W=vec_width data elements and we store them into `vec_data`
    tasklet = fpga_state_0.add_tasklet("read_data", {"from_memory"}, {"to_kernel"}, "to_kernel = from_memory")
    fpga_state_0.add_memlet_path(a_in,
                                 map_entry,
                                 read_map_entry,
                                 tasklet,
                                 dst_conn="from_memory",
                                 memlet=dace.Memlet("device_a[n0*{}+n1]".format(vec_width)))

    fpga_state_0.add_memlet_path(tasklet,
                                 read_map_exit,
                                 vect_data,
                                 src_conn="to_kernel",
                                 memlet=dace.Memlet("vec_data[n1]"))

    # Increment all the elements by one
    inc_map_entry, inc_map_exit = fpga_state_0.add_map("unrolled_inc", {"n1": "0:{}".format(vec_width)},
                                                       schedule=dace.ScheduleType.FPGA_Device,
                                                       unroll=True)

    inc_tasklet = fpga_state_0.add_tasklet("increment", {"_a"}, {"_b"}, "_b = _a + 1 ")

    fpga_state_0.add_memlet_path(vect_data,
                                 inc_map_entry,
                                 inc_tasklet,
                                 dst_conn="_a",
                                 memlet=dace.Memlet("vec_data[n1]"))

    fpga_state_0.add_memlet_path(inc_tasklet, inc_map_exit, inc_data, src_conn="_b", memlet=dace.Memlet("inc_data[n1]"))

    # write it to memory (it wil pack it)
    fpga_state_0.add_memlet_path(inc_data,
                                 map_exit,
                                 b_out,
                                 src_conn="outCon",
                                 memlet=dace.Memlet(f"{b_out}[n0]", other_subset="0:{}".format(vec_width)))

    ########################################################################
    # FPGA, Second State:
    # this read, increment, unpack, re-pack and save to memory

    fpga_state_1 = sdfg.add_state("fpga_state_1")

    b_in = fpga_state_1.add_read("device_b")
    c_out = fpga_state_1.add_write("device_c")

    # unpack data
    sdfg.add_array('vec_data_B',
                   shape=[vec_width],
                   dtype=dtype,
                   transient=True,
                   storage=dace.dtypes.StorageType.FPGA_Registers)
    sdfg.add_array('inc_data_B',
                   shape=[vec_width],
                   dtype=dtype,
                   transient=True,
                   storage=dace.dtypes.StorageType.FPGA_Registers)

    map_entry, map_exit = fpga_state_1.add_map("read_B", {
        "n0": "0:{}/{}".format(N, vec_width),
    },
                                               schedule=dace.ScheduleType.FPGA_Device)

    vect_data = fpga_state_1.add_access("vec_data_B")
    inc_data = fpga_state_1.add_access("inc_data_B")

    # unpack data
    fpga_state_1.add_memlet_path(b_in,
                                 map_entry,
                                 vect_data,
                                 memlet=dace.Memlet("device_b[n0]", other_subset="0:{}".format(vec_width)))

    # Increment all the elements by one
    inc_map_entry, inc_map_exit = fpga_state_1.add_map("unrolled_inc", {"n1": "0:{}".format(vec_width)},
                                                       schedule=dace.ScheduleType.FPGA_Device,
                                                       unroll=True)

    inc_tasklet = fpga_state_1.add_tasklet("increment", {"_a"}, {"_b"}, "_b = _a + 1 ")

    fpga_state_1.add_memlet_path(vect_data,
                                 inc_map_entry,
                                 inc_tasklet,
                                 dst_conn="_a",
                                 memlet=dace.Memlet("vec_data_B[n1]"))

    fpga_state_1.add_memlet_path(inc_tasklet,
                                 inc_map_exit,
                                 inc_data,
                                 src_conn="_b",
                                 memlet=dace.Memlet("inc_data_B[n1]"))

    # then we copy that to C, we need other gearboxing
    fpga_state_1.add_memlet_path(inc_data,
                                 map_exit,
                                 c_out,
                                 src_conn="to_memory",
                                 memlet=dace.Memlet(f"{c_out.data}[n0]", other_subset="0:{}".format(vec_width)))

    ########################################################################
    # FPGA, third State, read from C, write unpacked to D

    fpga_state_2 = sdfg.add_state("fpga_state_2")

    c_in = fpga_state_2.add_read("device_c")
    d_out = fpga_state_2.add_write("device_d")

    # unpack data
    sdfg.add_array('vec_data_C',
                   shape=[vec_width],
                   dtype=dtype,
                   transient=True,
                   storage=dace.dtypes.StorageType.FPGA_Registers)

    map_entry, map_exit = fpga_state_2.add_map("read_C", {
        "n0": "0:{}/{}".format(N, vec_width),
    },
                                               schedule=dace.ScheduleType.FPGA_Device)
    vect_data = fpga_state_2.add_access("vec_data_C")
    write_map_entry, write_map_exit = fpga_state_2.add_map("unrolled_reads", {"n1": "0:{}".format(vec_width)},
                                                           schedule=dace.ScheduleType.FPGA_Device,
                                                           unroll=True)

    fpga_state_2.add_memlet_path(c_in,
                                 map_entry,
                                 vect_data,
                                 memlet=dace.Memlet("device_c[n0]", other_subset="0:{}".format(vec_width)))

    # then we copy that to memory
    tasklet = fpga_state_2.add_tasklet("write_D", {"from_kernel"}, {"to_memory"}, "to_memory = from_kernel")
    fpga_state_2.add_memlet_path(vect_data,
                                 write_map_entry,
                                 tasklet,
                                 dst_conn="from_kernel",
                                 memlet=dace.Memlet("vec_data_C[n1]"))

    fpga_state_2.add_memlet_path(tasklet,
                                 write_map_exit,
                                 map_exit,
                                 d_out,
                                 src_conn="to_memory",
                                 memlet=dace.Memlet(f"{d_out.data}[n0*{vec_width}+n1]"))

    ######################################
    # Interstate edges
    sdfg.add_edge(copy_in_state, fpga_state_0, dace.sdfg.sdfg.InterstateEdge())
    sdfg.add_edge(fpga_state_0, fpga_state_1, dace.sdfg.sdfg.InterstateEdge())
    sdfg.add_edge(fpga_state_1, fpga_state_2, dace.sdfg.sdfg.InterstateEdge())
    sdfg.add_edge(fpga_state_2, copy_out_state, dace.sdfg.sdfg.InterstateEdge())

    #########
    # Validate
    sdfg.fill_scope_connectors()
    sdfg.validate()
    return sdfg


@fpga_test(assert_ii_1=False)
def test_multiple_veclen_conversions_test():

    parser = argparse.ArgumentParser()
    parser.add_argument("N", type=int, nargs="?", default=32)
    args = vars(parser.parse_args())

    size_n = args["N"]

    sdfg = make_sdfg()
    comp = sdfg.compile()

    a = np.random.rand(size_n).astype(np.float32)
    d = np.random.rand(size_n).astype(np.float32)
    ref = a + 2
    comp(a=a, d=d, N=size_n)
    diff = np.linalg.norm(ref - d) / size_n
    print("Difference:", diff)
    if diff <= 1e-5:
        print("==== Program end ====")
    else:
        print("==== Program Error! ====")

    assert diff <= 1e-5

    return sdfg


if __name__ == "__main__":
    test_multiple_veclen_conversions_test(None)
