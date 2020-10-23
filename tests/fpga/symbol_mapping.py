# Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.

# The scope of the test is to verify that nestedSDFG mapped symbols are handled correctly

import dace
import numpy as np
import argparse
import subprocess

from dace.memlet import Memlet


def make_vecAdd_sdfg(dtype=dace.float32, vecWidth=4):


    # ---------- ----------
    # SETUP GRAPH
    # ---------- ----------
    n = dace.symbol("n")
    vecType = dace.vector(dtype, vecWidth)

    vecAdd_sdfg = dace.SDFG('vecAdd_nested')
    vecAdd_state = vecAdd_sdfg.add_state()


    # ---------- ----------
    # MEMORY LOCATIONS
    # ---------- ----------
    # vecAdd_sdfg.add_scalar('_a', dtype=dtype, storage=dtypes.StorageType.FPGA_Global)
    vecAdd_sdfg.add_array('_x', shape=[n / vecWidth], dtype=vecType, storage=dace.dtypes.StorageType.FPGA_Global)
    vecAdd_sdfg.add_array('_y', shape=[n / vecWidth], dtype=vecType, storage=dace.dtypes.StorageType.FPGA_Global)
    vecAdd_sdfg.add_array('_res', shape=[n / vecWidth], dtype=vecType, storage=dace.dtypes.StorageType.FPGA_Global)

    x_in = vecAdd_state.add_read('_x')
    y_in = vecAdd_state.add_read('_y')
    z_out = vecAdd_state.add_write('_res')

    # ---------- ----------
    # COMPUTE
    # ---------- ----------
    vecMap_entry, vecMap_exit = vecAdd_state.add_map(
        'vecAdd_map',
        dict(i='0:{0}/{1}'.format(n, vecWidth)),
        schedule=dace.dtypes.ScheduleType.FPGA_Device)

    vecAdd_tasklet = vecAdd_state.add_tasklet(
        'vecAdd_task', ['x_con', 'y_con'], ['z_con'],
        'z_con =  x_con + y_con')

    vecAdd_state.add_memlet_path(x_in,
                                 vecMap_entry,
                                 vecAdd_tasklet,
                                 dst_conn='x_con',
                                 memlet=dace.Memlet.simple(
                                     x_in.data, 'i'))

    vecAdd_state.add_memlet_path(y_in,
                                 vecMap_entry,
                                 vecAdd_tasklet,
                                 dst_conn='y_con',
                                 memlet=dace.Memlet.simple(
                                     y_in.data, 'i'))

    vecAdd_state.add_memlet_path(vecAdd_tasklet,
                                 vecMap_exit,
                                 z_out,
                                 src_conn='z_con',
                                 memlet=dace.Memlet.simple(
                                     z_out.data, 'i'))

    return vecAdd_sdfg


def make_nested_sdfg_fpga():
    precision = dace.float32
    vecWidth = 4

    size = dace.symbol("size")
    sdfg = dace.SDFG("symbol_mapping")

    vecType = dace.vector(precision, vecWidth)

    ###########################################################################
    # Copy data to FPGA

    copy_in_state = sdfg.add_state("copy_to_device")

    sdfg.add_array("x", shape=[size / vecWidth], dtype=vecType)
    sdfg.add_array("y", shape=[size / vecWidth], dtype=vecType)

    in_host_x = copy_in_state.add_read("x")
    in_host_y = copy_in_state.add_read("y")

    sdfg.add_array("device_x", shape=[size / vecWidth], dtype=vecType, storage=dace.dtypes.StorageType.FPGA_Global,
                   transient=True)
    sdfg.add_array("device_y", shape=[size / vecWidth], dtype=vecType, storage=dace.dtypes.StorageType.FPGA_Global,
                   transient=True)

    in_device_x = copy_in_state.add_write("device_x")
    in_device_y = copy_in_state.add_write("device_y")

    copy_in_state.add_memlet_path(
        in_host_x, in_device_x,
        memlet=Memlet.simple(in_host_x, "0:{}/{}".format(size, vecWidth))
    )
    copy_in_state.add_memlet_path(
        in_host_y, in_device_y,
        memlet=Memlet.simple(in_host_y, "0:{}/{}".format(size, vecWidth))
    )

    ###########################################################################
    # Copy data from FPGA
    sdfg.add_array("z", shape=[size / vecWidth], dtype=vecType)

    copy_out_state = sdfg.add_state("copy_to_host")

    sdfg.add_array("device_z", shape=[size / vecWidth], dtype=vecType, storage=dace.dtypes.StorageType.FPGA_Global,
                   transient=True)

    out_device = copy_out_state.add_read("device_z")
    out_host = copy_out_state.add_write("z")

    copy_out_state.add_memlet_path(
        out_device, out_host,
        memlet=Memlet.simple(out_host, "0:{}/{}".format(size, vecWidth))
    )

    ########################################################################
    # FPGA State

    fpga_state = sdfg.add_state("fpga_state")

    x = fpga_state.add_read("device_x")
    y = fpga_state.add_read("device_y")
    z = fpga_state.add_write("device_z")


    vecAdd_node = make_vecAdd_sdfg( vecWidth=vecWidth)

    #nest the sdfg
    vecAdd_nested = fpga_state.add_nested_sdfg(vecAdd_node, sdfg, {"_x", "_y"}, {"_res"},
                                        {"n": "size"})


    fpga_state.add_memlet_path(x,
                               vecAdd_nested,
                               dst_conn="_x",
                               memlet=Memlet.simple(x, "0:{}/{}".format(size, vecWidth)))
    fpga_state.add_memlet_path(y,
                               vecAdd_nested,
                               dst_conn="_y",
                               memlet=Memlet.simple(y, "0:{}/{}".format(size, vecWidth)))
    fpga_state.add_memlet_path(vecAdd_nested,
                               z,
                               src_conn="_res",
                               memlet=Memlet.simple(z, "0:{}/{}".format(size, vecWidth)))

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
    return sdfg


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("N", type=int, nargs="?", default=32)
    args = vars(parser.parse_args())

    size= args["N"]
    sdfg = make_nested_sdfg_fpga()

    vecAdd = sdfg.compile()
    x = np.random.rand(size).astype(np.float32)
    y = np.random.rand(size).astype(np.float32)
    z = np.random.rand(size).astype(np.float32)


    vecAdd(x=x, y=y, z=z, size=size)

    ref1 = np.add(x, y)

    diff1 = np.linalg.norm(ref1 - z) / size
    if diff1 <= 1e-5 :
        print("==== Program end ====")
    else:
        raise Exception("==== Program Error! ====")

