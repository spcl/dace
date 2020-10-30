# Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.

# The scope of the test is to verify that two nested SDFGs within the same state are generated
# as two different FPGA kernels. The second Nested SDFG uses the result produced by the previous one

# Given

import dace
import numpy as np
import argparse
import subprocess

from dace.memlet import Memlet


def make_vecAdd_sdfg(dtype=dace.float32):

    # Vector addition SDFG

    vecWidth = 4
    n = dace.symbol("size")
    vecAdd_sdfg = dace.SDFG("vecAdd")
    vecType = dace.vector(dtype, vecWidth)
    fpga_state = vecAdd_sdfg.add_state("vecAdd_state")

    vecAdd_sdfg.add_array('_device_x', shape=[n / vecWidth], dtype=vecType, storage=dace.dtypes.StorageType.FPGA_Global)
    vecAdd_sdfg.add_array('_device_y', shape=[n / vecWidth], dtype=vecType, storage=dace.dtypes.StorageType.FPGA_Global)
    vecAdd_sdfg.add_array('_device_z', shape=[n / vecWidth], dtype=vecType, storage=dace.dtypes.StorageType.FPGA_Global)

    x = fpga_state.add_read("_device_x")
    y = fpga_state.add_read("_device_y")
    z = fpga_state.add_write("_device_z")

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

    #########
    # Validate
    vecAdd_sdfg.fill_scope_connectors()
    vecAdd_sdfg.validate()
    return vecAdd_sdfg


def make_vecMul_sdfg(dtype=dace.float32):
    # Vector multiplication SDFG

    vecWidth = 4
    n = dace.symbol("size")
    vecMul_sdfg = dace.SDFG("vecMul")
    vecType = dace.vector(dtype, vecWidth)
    fpga_state = vecMul_sdfg.add_state("vecMul_state")

    vecMul_sdfg.add_array('_device_x', shape=[n / vecWidth], dtype=vecType, storage=dace.dtypes.StorageType.FPGA_Global)
    vecMul_sdfg.add_array('_device_y', shape=[n / vecWidth], dtype=vecType, storage=dace.dtypes.StorageType.FPGA_Global)
    vecMul_sdfg.add_array('_device_z', shape=[n / vecWidth], dtype=vecType, storage=dace.dtypes.StorageType.FPGA_Global)

    x = fpga_state.add_read("_device_x")
    y = fpga_state.add_read("_device_y")
    z = fpga_state.add_write("_device_z")

    # ---------- ----------
    # COMPUTE
    # ---------- ----------
    vecMap_entry, vecMap_exit = fpga_state.add_map(
        'vecMul_map',
        dict(i='0:{0}/{1}'.format(n, vecWidth)),
        schedule=dace.dtypes.ScheduleType.FPGA_Device)

    vecMul_tasklet = fpga_state.add_tasklet('vecMul_task', ['x_con', 'y_con'],
                                            ['z_con'], 'z_con = x_con * y_con')

    fpga_state.add_memlet_path(x,
                               vecMap_entry,
                               vecMul_tasklet,
                               dst_conn='x_con',
                               memlet=dace.Memlet.simple(x.data, "i"))

    fpga_state.add_memlet_path(y,
                               vecMap_entry,
                               vecMul_tasklet,
                               dst_conn='y_con',
                               memlet=dace.Memlet.simple(y.data, 'i'))

    fpga_state.add_memlet_path(vecMul_tasklet,
                               vecMap_exit,
                               z,
                               src_conn='z_con',
                               memlet=dace.Memlet.simple(z.data, 'i'))

    #########
    # Validate
    vecMul_sdfg.fill_scope_connectors()
    vecMul_sdfg.validate()
    return vecMul_sdfg


def make_fpga_sdfg():
    '''
    Build an SDFG with two nested SDFGs in a single FPGA state
    '''


    n = dace.symbol("n")
    vecWidth = 4
    vecType = dace.vector(dace.float32, vecWidth)
    sdfg = dace.SDFG("nested_sdfg_kernels")

    ###########################################################################
    # Copy data to FPGA

    copy_in_state = sdfg.add_state("copy_to_device")

    sdfg.add_array("x", shape=[n / vecWidth], dtype=vecType)
    sdfg.add_array("y", shape=[n / vecWidth], dtype=vecType)

    sdfg.add_array("v", shape=[n / vecWidth], dtype=vecType)
    # sdfg.add_array("w", shape=[n / vecWidth], dtype=vecType)

    in_host_x = copy_in_state.add_read("x")
    in_host_y = copy_in_state.add_read("y")

    in_host_v = copy_in_state.add_read("v")
    # in_host_w = copy_in_state.add_read("w")

    sdfg.add_array("device_x",
                          shape=[n / vecWidth],
                          dtype=vecType,
                          storage=dace.dtypes.StorageType.FPGA_Global,
                          transient=True)
    sdfg.add_array("device_y",
                          shape=[n / vecWidth],
                          dtype=vecType,
                          storage=dace.dtypes.StorageType.FPGA_Global,
                          transient=True)

    sdfg.add_array("device_v",
                   shape=[n / vecWidth],
                   dtype=vecType,
                   storage=dace.dtypes.StorageType.FPGA_Global,
                   transient=True)
    # sdfg.add_array("device_w",
    #                shape=[m / vecWidth],
    #                dtype=vecType,
    #                storage=dace.dtypes.StorageType.FPGA_Global,
    #                transient=True)

    in_device_x = copy_in_state.add_write("device_x")
    in_device_y = copy_in_state.add_write("device_y")

    in_device_v = copy_in_state.add_write("device_v")
    # in_device_w = copy_in_state.add_write("device_w")

    copy_in_state.add_memlet_path(in_host_x,
                                  in_device_x,
                                  memlet=Memlet.simple(
                                      in_host_x, "0:{}/{}".format(n, vecWidth)))
    copy_in_state.add_memlet_path(in_host_y,
                                  in_device_y,
                                  memlet=Memlet.simple(
                                      in_host_y, "0:{}/{}".format(n, vecWidth)))

    copy_in_state.add_memlet_path(in_host_v,
                                  in_device_v,
                                  memlet=Memlet.simple(
                                      in_host_v, "0:{}/{}".format(n, vecWidth)))
    # copy_in_state.add_memlet_path(in_host_w,
    #                               in_device_w,
    #                               memlet=Memlet.simple(
    #                                   in_host_w, "0:{}/{}".format(m, vecWidth)))

    ###########################################################################
    # Copy data from FPGA
    sdfg.add_array("z", shape=[n / vecWidth], dtype=vecType)
    sdfg.add_array("u", shape=[n / vecWidth], dtype=vecType)

    copy_out_state = sdfg.add_state("copy_to_host")

    sdfg.add_array("device_z",
                          shape=[n / vecWidth],
                          dtype=vecType,
                          storage=dace.dtypes.StorageType.FPGA_Global,
                          transient=True)

    sdfg.add_array("device_u",
                   shape=[n / vecWidth],
                   dtype=vecType,
                   storage=dace.dtypes.StorageType.FPGA_Global,
                   transient=True)

    out_device_z = copy_out_state.add_read("device_z")
    out_host_z = copy_out_state.add_write("z")

    out_device_u = copy_out_state.add_read("device_u")
    out_host_u = copy_out_state.add_write("u")

    copy_out_state.add_memlet_path(out_device_z,
                                   out_host_z,
                                   memlet=Memlet.simple(
                                       out_host_z, "0:{}/{}".format(n, vecWidth)))
    copy_out_state.add_memlet_path(out_device_u,
                                   out_host_u,
                                   memlet=Memlet.simple(
                                       out_host_u, "0:{}/{}".format(n, vecWidth)))
    ###########################################################################
    # FPGA state

    fpga_state = sdfg.add_state("I_do_not_want_to_be_fpga_kernel")
    fpga_state.location["is_FPGA_kernel"]=False
    # Build the vec addition SDFG and nest it

    to_nest = make_vecAdd_sdfg()
    # add nested sdfg with symbol mapping
    nested_sdfg = fpga_state.add_nested_sdfg(to_nest, sdfg, {"_device_x", "_device_y"}, {"_device_z"},
                                        {"size": "n"})

    fpga_state.add_memlet_path(in_device_x,
                          nested_sdfg,
                          dst_conn="_device_x",
                          memlet=Memlet.simple(in_device_x,  "0:{}/{}".format(n, vecWidth)))
    fpga_state.add_memlet_path(in_device_y,
                          nested_sdfg,
                          dst_conn="_device_y",
                          memlet=Memlet.simple(in_device_y,  "0:{}/{}".format(n, vecWidth)))
    fpga_state.add_memlet_path(nested_sdfg,
                          out_device_z,
                          src_conn="_device_z",
                          memlet=Memlet.simple(out_device_z,  "0:{}/{}".format(n, vecWidth)))

    # Build the second vec addition SDFG and nest it

    to_nest = make_vecAdd_sdfg()
    # add nested sdfg with symbol mapping
    nested_sdfg = fpga_state.add_nested_sdfg(to_nest, sdfg, {"_device_x", "_device_y"}, {"_device_z"},
                                             {"size": "n"})

    fpga_state.add_memlet_path(out_device_z,
                               nested_sdfg,
                               dst_conn="_device_x",
                               memlet=Memlet.simple(out_device_z, "0:{}/{}".format(n, vecWidth)))
    fpga_state.add_memlet_path(in_device_v,
                               nested_sdfg,
                               dst_conn="_device_y",
                               memlet=Memlet.simple(in_device_v, "0:{}/{}".format(n, vecWidth)))
    fpga_state.add_memlet_path(nested_sdfg,
                               out_device_u,
                               src_conn="_device_z",
                               memlet=Memlet.simple(out_device_u, "0:{}/{}".format(n, vecWidth)))

    ######################################
    # Interstate edges
    sdfg.add_edge(copy_in_state, fpga_state,
                         dace.sdfg.sdfg.InterstateEdge())
    sdfg.add_edge(fpga_state, copy_out_state,
                         dace.sdfg.sdfg.InterstateEdge())
    sdfg.fill_scope_connectors()
    sdfg.save('/tmp/out.sdfg')
    sdfg.validate()

    return sdfg



def make_fpga_sdfg_independent():
    '''
    Build an SDFG with two nested SDFGs in a single FPGA state
    '''


    n = dace.symbol("n")
    m = dace.symbol("m")
    vecWidth = 4
    vecType = dace.vector(dace.float32, vecWidth)
    sdfg = dace.SDFG("nested_sdfg_kernels")

    ###########################################################################
    # Copy data to FPGA

    copy_in_state = sdfg.add_state("copy_to_device")

    sdfg.add_array("x", shape=[n / vecWidth], dtype=vecType)
    sdfg.add_array("y", shape=[n / vecWidth], dtype=vecType)

    sdfg.add_array("v", shape=[n / vecWidth], dtype=vecType)
    sdfg.add_array("w", shape=[n / vecWidth], dtype=vecType)

    in_host_x = copy_in_state.add_read("x")
    in_host_y = copy_in_state.add_read("y")

    in_host_v = copy_in_state.add_read("v")
    in_host_w = copy_in_state.add_read("w")

    sdfg.add_array("device_x",
                          shape=[n / vecWidth],
                          dtype=vecType,
                          storage=dace.dtypes.StorageType.FPGA_Global,
                          transient=True)
    sdfg.add_array("device_y",
                          shape=[n / vecWidth],
                          dtype=vecType,
                          storage=dace.dtypes.StorageType.FPGA_Global,
                          transient=True)

    sdfg.add_array("device_v",
                   shape=[m / vecWidth],
                   dtype=vecType,
                   storage=dace.dtypes.StorageType.FPGA_Global,
                   transient=True)
    sdfg.add_array("device_w",
                   shape=[m / vecWidth],
                   dtype=vecType,
                   storage=dace.dtypes.StorageType.FPGA_Global,
                   transient=True)

    in_device_x = copy_in_state.add_write("device_x")
    in_device_y = copy_in_state.add_write("device_y")

    in_device_v = copy_in_state.add_write("device_v")
    in_device_w = copy_in_state.add_write("device_w")

    copy_in_state.add_memlet_path(in_host_x,
                                  in_device_x,
                                  memlet=Memlet.simple(
                                      in_host_x, "0:{}/{}".format(n, vecWidth)))
    copy_in_state.add_memlet_path(in_host_y,
                                  in_device_y,
                                  memlet=Memlet.simple(
                                      in_host_y, "0:{}/{}".format(n, vecWidth)))

    copy_in_state.add_memlet_path(in_host_v,
                                  in_device_v,
                                  memlet=Memlet.simple(
                                      in_host_v, "0:{}/{}".format(m, vecWidth)))
    copy_in_state.add_memlet_path(in_host_w,
                                  in_device_w,
                                  memlet=Memlet.simple(
                                      in_host_w, "0:{}/{}".format(m, vecWidth)))

    ###########################################################################
    # Copy data from FPGA
    sdfg.add_array("z", shape=[n / vecWidth], dtype=vecType)
    sdfg.add_array("u", shape=[m / vecWidth], dtype=vecType)

    copy_out_state = sdfg.add_state("copy_to_host")

    sdfg.add_array("device_z",
                          shape=[n / vecWidth],
                          dtype=vecType,
                          storage=dace.dtypes.StorageType.FPGA_Global,
                          transient=True)

    sdfg.add_array("device_u",
                   shape=[m / vecWidth],
                   dtype=vecType,
                   storage=dace.dtypes.StorageType.FPGA_Global,
                   transient=True)

    out_device_z = copy_out_state.add_read("device_z")
    out_host_z = copy_out_state.add_write("z")

    out_device_u = copy_out_state.add_read("device_u")
    out_host_u = copy_out_state.add_write("u")

    copy_out_state.add_memlet_path(out_device_z,
                                   out_host_z,
                                   memlet=Memlet.simple(
                                       out_host_z, "0:{}/{}".format(n, vecWidth)))
    copy_out_state.add_memlet_path(out_device_u,
                                   out_host_u,
                                   memlet=Memlet.simple(
                                       out_host_u, "0:{}/{}".format(m, vecWidth)))
    ###########################################################################
    # FPGA state

    fpga_state = sdfg.add_state("fpga_state")
    fpga_state.location["is_FPGA_kernel"]=False

    # Build the vec addition SDFG and nest it

    to_nest = make_vecAdd_sdfg()
    # add nested sdfg with symbol mapping
    nested_sdfg = fpga_state.add_nested_sdfg(to_nest, sdfg, {"_device_x", "_device_y"}, {"_device_z"},
                                        {"size": "n"})

    fpga_state.add_memlet_path(in_device_x,
                          nested_sdfg,
                          dst_conn="_device_x",
                          memlet=Memlet.simple(in_device_x,  "0:{}/{}".format(n, vecWidth)))
    fpga_state.add_memlet_path(in_device_y,
                          nested_sdfg,
                          dst_conn="_device_y",
                          memlet=Memlet.simple(in_device_y,  "0:{}/{}".format(n, vecWidth)))
    fpga_state.add_memlet_path(nested_sdfg,
                          out_device_z,
                          src_conn="_device_z",
                          memlet=Memlet.simple(out_device_z,  "0:{}/{}".format(n, vecWidth)))

    # Build the vec multiplication SDFG and nest it

    to_nest = make_vecMul_sdfg()
    # add nested sdfg with symbol mapping
    nested_sdfg = fpga_state.add_nested_sdfg(to_nest, sdfg, {"_device_x", "_device_y"}, {"_device_z"},
                                             {"size": "m"})

    fpga_state.add_memlet_path(in_device_v,
                               nested_sdfg,
                               dst_conn="_device_x",
                               memlet=Memlet.simple(in_device_v, "0:{}/{}".format(m, vecWidth)))
    fpga_state.add_memlet_path(in_device_w,
                               nested_sdfg,
                               dst_conn="_device_y",
                               memlet=Memlet.simple(in_device_w, "0:{}/{}".format(m, vecWidth)))
    fpga_state.add_memlet_path(nested_sdfg,
                               out_device_u,
                               src_conn="_device_z",
                               memlet=Memlet.simple(out_device_u, "0:{}/{}".format(m, vecWidth)))

    ######################################
    # Interstate edges
    sdfg.add_edge(copy_in_state, fpga_state,
                         dace.sdfg.sdfg.InterstateEdge())
    sdfg.add_edge(fpga_state, copy_out_state,
                         dace.sdfg.sdfg.InterstateEdge())
    sdfg.fill_scope_connectors()
    sdfg.save('/tmp/out.sdfg')
    sdfg.validate()

    return sdfg

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("N", type=int, nargs="?", default=32)
    parser.add_argument("M", type=int, nargs="?", default=32)
    args = vars(parser.parse_args())

    size_n = args["N"]
    size_m = args["M"]

    sdfg = make_fpga_sdfg()
    # sdfg = make_fpga_sdfg_independent()
    sdfg.save('/tmp/out.sdfg')

    vec_ops = sdfg.compile()

    x = np.random.rand(size_n).astype(np.float32)
    y = np.random.rand(size_n).astype(np.float32)
    z = np.random.rand(size_n).astype(np.float32)

    v = np.random.rand(size_n).astype(np.float32)
    u = np.random.rand(size_n).astype(np.float32)
    w = np.random.rand(size_n).astype(np.float32)

    vec_ops(x=x, y=y, z=z, v=v, u=u, n=size_n)
    ref1 = np.add(x, y)
    ref2 = np.add(ref1, v)

    # vec_ops(x=x, y=y, z=z, v=v,w=w, u=u, n=size_n, m=size_m)
    # ref1 = np.add(x, y)
    # ref2 = np.multiply(v,w)

    diff1 = np.linalg.norm(ref1 - z) / size_n
    diff2 = np.linalg.norm(ref2 - u) / size_n
    if diff1 <= 1e-5 and diff2 <= 1e-5:
        print("==== Program end ====")
    else:
        raise Exception("==== Program Error! ====")

    # There is no need to check that the Nested SDFG has been generated only once. If this is not the case
    # the test will fail while compiling


