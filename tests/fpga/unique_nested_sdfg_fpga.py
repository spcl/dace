# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.

# The scope of the test is to verify that code nested SDFGs with a unique name is generated only once
# The nested SDFG compute vector addition on FPGA, with vectorization

import dace
import numpy as np
import argparse
import subprocess

from dace.memlet import Memlet


def make_vecAdd_sdfg(sdfg_name: str, dtype=dace.float32):
    vecWidth = 4
    n = dace.symbol("size")
    vecAdd_sdfg = dace.SDFG(sdfg_name)
    vecType = dace.vector(dtype, vecWidth)

    x_name = "x"
    y_name = "y"
    z_name = "z"

    ###########################################################################
    # Copy data to FPGA

    copy_in_state = vecAdd_sdfg.add_state("copy_to_device")

    vecAdd_sdfg.add_array(x_name, shape=[n / vecWidth], dtype=vecType)
    vecAdd_sdfg.add_array(y_name, shape=[n / vecWidth], dtype=vecType)

    in_host_x = copy_in_state.add_read(x_name)
    in_host_y = copy_in_state.add_read(y_name)

    vecAdd_sdfg.add_array("device_x",
                          shape=[n / vecWidth],
                          dtype=vecType,
                          storage=dace.dtypes.StorageType.FPGA_Global,
                          transient=True)
    vecAdd_sdfg.add_array("device_y",
                          shape=[n / vecWidth],
                          dtype=vecType,
                          storage=dace.dtypes.StorageType.FPGA_Global,
                          transient=True)

    in_device_x = copy_in_state.add_write("device_x")
    in_device_y = copy_in_state.add_write("device_y")

    copy_in_state.add_memlet_path(in_host_x,
                                  in_device_x,
                                  memlet=Memlet.simple(
                                      in_host_x, "0:{}/{}".format(n, vecWidth)))
    copy_in_state.add_memlet_path(in_host_y,
                                  in_device_y,
                                  memlet=Memlet.simple(
                                      in_host_y, "0:{}/{}".format(n, vecWidth)))

    ###########################################################################
    # Copy data from FPGA
    vecAdd_sdfg.add_array(z_name, shape=[n / vecWidth], dtype=vecType)

    copy_out_state = vecAdd_sdfg.add_state("copy_to_host")

    vecAdd_sdfg.add_array("device_z",
                          shape=[n / vecWidth],
                          dtype=vecType,
                          storage=dace.dtypes.StorageType.FPGA_Global,
                          transient=True)

    out_device = copy_out_state.add_read("device_z")
    out_host = copy_out_state.add_write(z_name)

    copy_out_state.add_memlet_path(out_device,
                                   out_host,
                                   memlet=Memlet.simple(
                                       out_host, "0:{}/{}".format(n, vecWidth)))

    ########################################################################
    # FPGA State

    fpga_state = vecAdd_sdfg.add_state("fpga_state")

    x = fpga_state.add_read("device_x")
    y = fpga_state.add_read("device_y")
    z = fpga_state.add_write("device_z")

    # ---------- ----------
    # COMPUTE
    # ---------- ----------
    vecMap_entry, vecMap_exit = fpga_state.add_map(
        'vecAdd_map',
        dict(i='0:{0}/{1}'.format(n, vecWidth)),
        schedule=dace.dtypes.ScheduleType.FPGA_Device)

    vecAdd_tasklet = fpga_state.add_tasklet('vecAdd_task', ['x_con', 'y_con'],
                                            ['z_con'], 'z_con = x_con + y_con')

    fpga_state.add_memlet_path(x,
                               vecMap_entry,
                               vecAdd_tasklet,
                               dst_conn='x_con',
                               memlet=dace.Memlet.simple(x.data, "i"))

    fpga_state.add_memlet_path(y,
                               vecMap_entry,
                               vecAdd_tasklet,
                               dst_conn='y_con',
                               memlet=dace.Memlet.simple(y.data, 'i'))

    fpga_state.add_memlet_path(vecAdd_tasklet,
                               vecMap_exit,
                               z,
                               src_conn='z_con',
                               memlet=dace.Memlet.simple(z.data, 'i'))

    ######################################
    # Interstate edges
    vecAdd_sdfg.add_edge(copy_in_state, fpga_state,
                         dace.sdfg.sdfg.InterstateEdge())
    vecAdd_sdfg.add_edge(fpga_state, copy_out_state,
                         dace.sdfg.sdfg.InterstateEdge())

    #########
    # Validate
    vecAdd_sdfg.fill_scope_connectors()
    vecAdd_sdfg.validate()
    return vecAdd_sdfg


def make_nested_sdfg_fpga():
    '''
    Build an SDFG with two nested SDFGs, each one a different state
    '''

    n = dace.symbol("n")
    m = dace.symbol("m")

    sdfg = dace.SDFG("two_vecAdd")
    state = sdfg.add_state("state")

    # build the first axpy: works with x,y, and z of n-elements

    # ATTENTION: this two nested SDFG must have the same name as they are equal
    to_nest = make_vecAdd_sdfg("vecAdd")

    sdfg.add_array("x", [n], dace.float32)
    sdfg.add_array("y", [n], dace.float32)
    sdfg.add_array("z", [n], dace.float32)
    x = state.add_read("x")
    y = state.add_read("y")
    z = state.add_write("z")

    # add nested sdfg with symbol mapping
    nested_sdfg = state.add_nested_sdfg(to_nest, sdfg, {"x", "y"}, {"z"},
                                        {"size": "n"})

    state.add_memlet_path(x,
                          nested_sdfg,
                          dst_conn="x",
                          memlet=Memlet.simple(x, "0:n", num_accesses=n))
    state.add_memlet_path(y,
                          nested_sdfg,
                          dst_conn="y",
                          memlet=Memlet.simple(y, "0:n", num_accesses=n))
    state.add_memlet_path(nested_sdfg,
                          z,
                          src_conn="z",
                          memlet=Memlet.simple(z, "0:n", num_accesses=n))

    # Build the second axpy: works with v,w and u of m elements, use another state

    state2 = sdfg.add_state("state2")

    to_nest = make_vecAdd_sdfg("vecAdd")

    sdfg.add_array("v", [m], dace.float32)
    sdfg.add_array("w", [m], dace.float32)
    sdfg.add_array("u", [m], dace.float32)
    v = state2.add_read("v")
    w = state2.add_read("w")
    u = state2.add_write("u")

    nested_sdfg = state2.add_nested_sdfg(to_nest, sdfg, {"x", "y"}, {"z"},
                                         {"size": "m"})

    state2.add_memlet_path(v,
                           nested_sdfg,
                           dst_conn="x",
                           memlet=Memlet.simple(v, "0:m", num_accesses=m))
    state2.add_memlet_path(w,
                           nested_sdfg,
                           dst_conn="y",
                           memlet=Memlet.simple(w, "0:m", num_accesses=m))
    state2.add_memlet_path(nested_sdfg,
                           u,
                           src_conn="z",
                           memlet=Memlet.simple(u, "0:m", num_accesses=m))
    ######################################
    # Interstate edges
    sdfg.add_edge(state, state2, dace.sdfg.sdfg.InterstateEdge())
    sdfg.validate()

    return sdfg


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("N", type=int, nargs="?", default=32)
    parser.add_argument("M", type=int, nargs="?", default=64)
    args = vars(parser.parse_args())

    size_n = args["N"]
    size_m = args["M"]
    sdfg = make_nested_sdfg_fpga()

    two_axpy = sdfg.compile()

    x = np.random.rand(size_n).astype(np.float32)
    y = np.random.rand(size_n).astype(np.float32)
    z = np.random.rand(size_n).astype(np.float32)

    v = np.random.rand(size_m).astype(np.float32)
    w = np.random.rand(size_m).astype(np.float32)
    u = np.random.rand(size_m).astype(np.float32)

    two_axpy(x=x, y=y, z=z, v=v, w=w, u=u, n=size_n, m=size_m)

    ref1 = np.add(x, y)
    ref2 = np.add(v, w)

    diff1 = np.linalg.norm(ref1 - z) / size_n
    diff2 = np.linalg.norm(ref2 - u) / size_m
    if diff1 <= 1e-5 and diff2 <= 1e-5:
        print("==== Program end ====")
    else:
        raise Exception("==== Program Error! ====")

    # There is no need to check that the Nested SDFG has been generated only once. If this is not the case
    # the test will fail while compiling
