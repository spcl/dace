# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.

# The scope of the test is to verify channel predication.
# In Intel FPGA, the compiler does not deal well with static channel indexing, causing
# the compiler to predicate access to all the channel array.
# While this can be ok in some cases, in others it can generates problem because the compiler sees that
# channels are written in multiple places (due to predication) even if they are not
# We want to do explicitely, by generating proper code.
# In this test we have two maps: one that reads an array of N elements and puts each element in the correspondent
# channel into a channel (stream) array of N elements. The other one does symmetric operations, by reading
# the data from the different channel and putting it back to memory.

# NOTE: this problem is not detected by the emulation.

import dace
import numpy as np
import argparse

from dace.memlet import Memlet

N = dace.symbol("N")

def make_sdfg (dtype = dace.float32):
    sdfg = dace.SDFG("predicate_channel_access")


    ###########################################################################
    # Data

    sdfg.add_array("host_in", shape=[N], dtype=dtype)
    sdfg.add_array("host_out", shape=[N], dtype=dtype)

    sdfg.add_array("device_in", shape=[N], dtype=dtype, storage=dace.dtypes.StorageType.FPGA_Global,
                   transient=True)
    sdfg.add_array("device_out", shape=[N], dtype=dtype, storage=dace.dtypes.StorageType.FPGA_Global,
                   transient=True)

    ###########################################################################
    # Copy data to FPGA

    copy_in_state = sdfg.add_state("copy_to_device")


    in_host = copy_in_state.add_read("host_in")

    in_device = copy_in_state.add_write("device_in")

    copy_in_state.add_memlet_path(
        in_host, in_device,
        memlet=Memlet(f"{in_host}")
    )

    ###########################################################################
    # Copy data from FPGA
    copy_out_state = sdfg.add_state("copy_to_host")

    out_device = copy_out_state.add_read("device_out")
    out_host = copy_out_state.add_write("host_out")

    copy_out_state.add_memlet_path(
        out_device, out_host,
        memlet=Memlet(f"{out_host}")
    )


    ########################################################################
    # FPGA,

    # create the stream for connecting the two components
    sdfg.add_stream('device_stream', dtype, shape=[N],
                    storage=dace.dtypes.StorageType.FPGA_Local, transient=True)

    fpga_state = sdfg.add_state("fpga_state")

    # first component: each elements of the input array on a different channel
    data_in = fpga_state.add_read("device_in")
    data_stream_out = fpga_state.add_write("device_stream")

    read_entry, read_exit = fpga_state.add_map("read", {
        "n": "0:N",
    }, schedule=dace.ScheduleType.FPGA_Device)

    read_tasklet = fpga_state.add_tasklet(
        'readtasklet',
        ['inCon'],
        ['outCon'],
        'outCon = inCon'
    )

    fpga_state.add_memlet_path(
        data_in, read_entry, read_tasklet,
        dst_conn='inCon',
        memlet=dace.Memlet(f"{data_in}[n]")
    )
    fpga_state.add_memlet_path(
        read_tasklet, read_exit, data_stream_out,
        src_conn='outCon',
        memlet=dace.Memlet(f"{data_stream_out}[n]")
    )

    # second component: receive the elements from the different streams

    data_out = fpga_state.add_read("device_out")
    data_stream_in = fpga_state.add_read("device_stream")

    write_entry, write_exit = fpga_state.add_map("write", {
        "n": "0:N",
    }, schedule=dace.ScheduleType.FPGA_Device)

    write_tasklet = fpga_state.add_tasklet(
        'write_tasklet',
        ['inCon'],
        ['outCon'],
        'outCon = inCon'
    )

    fpga_state.add_memlet_path(
        data_stream_in, write_entry, write_tasklet,
        dst_conn='inCon',
        memlet=dace.Memlet(f"{data_stream_in}[n]")
    )
    fpga_state.add_memlet_path(
        write_tasklet, write_exit, data_out,
        src_conn='outCon',
        memlet=dace.Memlet(f"{data_out}[n]")
    )

    ######################################
    # Interstate edges
    sdfg.add_edge(copy_in_state, fpga_state,
                  dace.sdfg.sdfg.InterstateEdge())
    sdfg.add_edge(fpga_state, copy_out_state,
                  dace.sdfg.sdfg.InterstateEdge())

    #########
    # Validate
    sdfg.fill_scope_connectors()
    sdfg.validate()
    sdfg.save('/tmp/out.sdfg')
    return sdfg




if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("N", type=int)
    args = vars(parser.parse_args())

    N.set(args["N"])
    sdfg = make_sdfg()
    sdfg.specialize(dict(N=N))
    in_data = np.ndarray([N.get()], dtype=dace.float32.type)
    out_data = np.ndarray([N.get()], dtype=dace.float32.type)
    in_data[:] = np.random.rand(N.get()).astype(dace.float32.type)
    out_data[:] = np.random.rand(N.get()).astype(dace.float32.type)

    sdfg(host_in=in_data, host_out=out_data)

    if np.array_equal(in_data, out_data):
        print("==== Program end ====")
    else:
        print("==== Program Error! ====")

    exit(0 if np.array_equal(in_data, out_data) else 1)
