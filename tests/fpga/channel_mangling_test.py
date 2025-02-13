# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.

# The scope of the test is to verify channel mangling. In the
# SDFG we have two nested SDFG that increments some streaming data by one.
# The NestedSDFGs are similar, but they work on different sizes

# Note: to generate the code for both NSDFG, we set Dace config for compiler->unique_functions to False

import dace
import numpy as np
import argparse
import subprocess
from dace.config import Config

from dace.fpga_testing import intel_fpga_test
from dace.memlet import Memlet

N = dace.symbol("N")


def make_increment_sdfg(sdfg_name: str, dtype=dace.float32):
    inc_sdfg = dace.SDFG(sdfg_name)

    # FPGA State

    fpga_state = inc_sdfg.add_state("fpga_state")

    inc_sdfg.add_array("x", shape=[N], dtype=dtype, storage=dace.dtypes.StorageType.FPGA_Global)
    inc_sdfg.add_array("y", shape=[N], dtype=dtype, storage=dace.dtypes.StorageType.FPGA_Global)
    inc_sdfg.add_stream("what_a_nice_pipe", dtype, transient=True, storage=dace.dtypes.StorageType.FPGA_Local)

    data_in = fpga_state.add_read("x")
    data_out = fpga_state.add_write("y")
    pipe_write = fpga_state.add_write("what_a_nice_pipe")
    pipe_read = fpga_state.add_read("what_a_nice_pipe")

    # ---------- ----------
    read_map_entry, read_map_exit = fpga_state.add_map('read_incr_map',
                                                       dict(i='0:N'),
                                                       schedule=dace.dtypes.ScheduleType.FPGA_Device)

    incr_tasklet = fpga_state.add_tasklet('incr_task', ['in_con'], ['out_con'], 'out_con = in_con + 1')

    # From memory to increment
    fpga_state.add_memlet_path(data_in,
                               read_map_entry,
                               incr_tasklet,
                               dst_conn='in_con',
                               memlet=dace.Memlet(f"{data_in.data}[i]"))
    # from increment to pipe
    fpga_state.add_memlet_path(incr_tasklet,
                               read_map_exit,
                               pipe_write,
                               src_conn='out_con',
                               memlet=dace.Memlet("what_a_nice_pipe[0]"))

    # from pipe to memory
    write_map_entry, write_map_exit = fpga_state.add_map('write_map',
                                                         dict(i='0:N'),
                                                         schedule=dace.dtypes.ScheduleType.FPGA_Device)

    copy_tasklet = fpga_state.add_tasklet('copy_task', ['in_con'], ['out_con'], 'out_con = in_con ')

    fpga_state.add_memlet_path(pipe_read,
                               write_map_entry,
                               copy_tasklet,
                               dst_conn='in_con',
                               memlet=dace.Memlet("what_a_nice_pipe[0]"))
    fpga_state.add_memlet_path(copy_tasklet, write_map_exit, data_out, src_conn='out_con', memlet=dace.Memlet("y[i]"))

    #########
    # Validate
    inc_sdfg.fill_scope_connectors()
    inc_sdfg.validate()
    return inc_sdfg


def make_nested_sdfg_fpga(dtype=dace.float32):
    """
    Build an SDFG with two nested SDFGs, each one a different state
    """

    sdfg = dace.SDFG("channels_mangling")

    ###########################################################################
    # Copy data to FPGA

    copy_in_state = sdfg.add_state("copy_to_device")

    sdfg.add_array("X", shape=[N], dtype=dtype)

    in_host_x = copy_in_state.add_read("X")

    sdfg.add_array("device_X", shape=[N], dtype=dtype, storage=dace.dtypes.StorageType.FPGA_Global, transient=True)
    sdfg.add_array("device_tmp", shape=[N], dtype=dtype, storage=dace.dtypes.StorageType.FPGA_Global, transient=True)

    in_device_x = copy_in_state.add_write("device_X")

    copy_in_state.add_memlet_path(in_host_x, in_device_x, memlet=Memlet.simple(in_host_x, "0:N"))

    ###########################################################################
    # Copy data from FPGA

    copy_out_state = sdfg.add_state("copy_to_host")
    sdfg.add_array("Y", shape=[N], dtype=dtype)
    sdfg.add_array("device_Y", shape=[N], dtype=dtype, storage=dace.dtypes.StorageType.FPGA_Global, transient=True)

    out_device = copy_out_state.add_read("device_Y")
    out_host = copy_out_state.add_write("Y")

    copy_out_state.add_memlet_path(out_device, out_host, memlet=Memlet.simple(out_host, "0:N"))

    ########################################################################
    # First state
    state = sdfg.add_state("state")
    state.location["is_FPGA_kernel"] = False

    to_nest = make_increment_sdfg("nest_1", dtype)
    x = state.add_read("device_X")
    tmp = state.add_write("device_tmp")

    # add nested sdfg with symbol mapping
    nested_sdfg = state.add_nested_sdfg(to_nest, sdfg, {"x"}, {"y"})
    state.add_memlet_path(x, nested_sdfg, dst_conn="x", memlet=Memlet("device_X[0:N]"))
    state.add_memlet_path(nested_sdfg, tmp, src_conn="y", memlet=Memlet("device_tmp[0:N]"))

    ########################################################################
    # First state
    state2 = sdfg.add_state("state2")
    state2.location["is_FPGA_kernel"] = False

    to_nest = make_increment_sdfg("nest_2", dtype)
    tmp_read = state2.add_read("device_tmp")
    y = state2.add_write("device_Y")

    # add nested sdfg with symbol mapping
    nested_sdfg = state2.add_nested_sdfg(to_nest, sdfg, {"x"}, {"y"})
    state2.add_memlet_path(tmp_read, nested_sdfg, dst_conn="x", memlet=Memlet("device_tmp[0:N]"))
    state2.add_memlet_path(nested_sdfg, y, src_conn="y", memlet=Memlet("device_Y[0:N]"))

    ######################################
    # Interstate edges
    sdfg.add_edge(state, state2, dace.sdfg.sdfg.InterstateEdge())

    # Interstate edges
    sdfg.add_edge(copy_in_state, state, dace.sdfg.sdfg.InterstateEdge())
    sdfg.add_edge(state2, copy_out_state, dace.sdfg.sdfg.InterstateEdge())
    sdfg.validate()

    return sdfg


@intel_fpga_test()
def test_channel_mangling():

    parser = argparse.ArgumentParser()
    parser.add_argument("N", type=int, nargs="?", default=32)
    args = vars(parser.parse_args())

    size_n = args["N"]

    from dace.config import Config
    # set unique function to false to generate both sdfgs
    Config.set("compiler", "unique_functions", value="none")
    sdfg = make_nested_sdfg_fpga()

    X = np.random.rand(size_n).astype(np.float32)
    Y = np.random.rand(size_n).astype(np.float32)
    sdfg(X=X, Y=Y, N=size_n)
    ref = X + 2
    diff = np.linalg.norm(ref - Y) / size_n
    assert diff <= 1e-5

    return sdfg


if __name__ == "__main__":
    test_channel_mangling(None)
