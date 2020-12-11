#!/usr/bin/env python3
# Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.

import numpy as np

import argparse
import scipy
import random

import dace
from dace.memlet import Memlet

import dace.libraries.blas as blas
import dace.libraries.blas.utility.fpga_helper as streaming
from dace.libraries.blas.utility import memory_operations as memOps
from dace.transformation.interstate import GPUTransformSDFG

from dace.libraries.standard.memory import aligned_ndarray

from multiprocessing import Process, Queue


def run_program(program, a, b, c, alpha, testN, ref_result, queue):

    program(x1=a, y1=b, a=alpha, z1=c, n=np.int32(testN))
    ref_norm = np.linalg.norm(c - ref_result) / testN

    queue.put(ref_norm)


def run_test(configs, target, implementation, overwrite_y=False):

    testN = int(2**13)

    for config in configs:

        prec = np.float32 if config[2] == dace.float32 else np.float64
        a = aligned_ndarray(np.random.uniform(0, 100, testN).astype(prec),
                            alignment=256)
        b = aligned_ndarray(np.random.uniform(0, 100, testN).astype(prec),
                            alignment=256)
        b_ref = b.copy()

        c = aligned_ndarray(np.zeros(testN).astype(prec), alignment=256)
        alpha = np.float32(
            config[0]) if config[2] == dace.float32 else np.float64(config[0])

        ref_result = reference_result(a, b_ref, alpha)

        program = None
        if target == "fpga":
            program = stream_fpga_graph(config[1],
                                        config[2],
                                        implementation,
                                        test_case=config[3])
        elif target == "fpga_array":
            program = array_fpga_graph(config[1],
                                       config[2],
                                       implementation,
                                       test_case=config[3],
                                       expansion="fpga")
        elif target == "intel_fpga_dram":
            program = array_fpga_graph(config[1],
                                       config[2],
                                       implementation,
                                       test_case=config[3])
        else:
            program = pure_graph(config[1], config[2], test_case=config[3])

        ref_norm = 0
        if target in ["fpga", "intel_fpga_dram", "fpga_array"]:

            # Run FPGA tests in a different process to avoid issues with Intel OpenCL tools
            queue = Queue()
            p = Process(target=run_program,
                        args=(program, a, b, c, alpha, testN, ref_result,
                              queue))
            p.start()
            p.join()
            ref_norm = queue.get()

        elif overwrite_y:
            program(x1=a, y1=b, a=alpha, z1=b, n=np.int32(testN))
            ref_norm = np.linalg.norm(b - ref_result) / testN
        else:
            program(x1=a, y1=b, a=alpha, z1=c, n=np.int32(testN))
            ref_norm = np.linalg.norm(c - ref_result) / testN

        passed = ref_norm < 1e-5

        if not passed:
            raise RuntimeError(
                'AXPY {} implementation on target {} wrong test results on config: '
                .format(implementation, target), config)


# ---------- ----------
# Ref result
# ---------- ----------
def reference_result(x_in, y_in, alpha):
    return scipy.linalg.blas.saxpy(x_in, y_in, a=alpha)


# ---------- ----------
# Pure graph program
# ---------- ----------
def pure_graph(veclen, precision, implementation="pure", test_case="0"):

    n = dace.symbol("n")
    a = dace.symbol("a")

    prec = "single" if precision == dace.float32 else "double"
    test_sdfg = dace.SDFG("axpy_test_" + prec + "_v" + str(veclen) + "_" +
                          implementation + "_" + test_case)
    test_state = test_sdfg.add_state("test_state")

    vec_type = dace.vector(precision, veclen)

    test_sdfg.add_symbol(a.name, precision)

    test_sdfg.add_array('x1', shape=[n / veclen], dtype=vec_type)
    test_sdfg.add_array('y1', shape=[n / veclen], dtype=vec_type)
    test_sdfg.add_array('z1', shape=[n / veclen], dtype=vec_type)

    x_in = test_state.add_read('x1')
    y_in = test_state.add_read('y1')
    z_out = test_state.add_write('z1')

    saxpy_node = blas.axpy.Axpy("axpy", precision)
    saxpy_node.implementation = implementation

    test_state.add_memlet_path(x_in,
                               saxpy_node,
                               dst_conn='_x',
                               memlet=Memlet.simple(x_in,
                                                    "0:n/{}".format(veclen)))
    test_state.add_memlet_path(y_in,
                               saxpy_node,
                               dst_conn='_y',
                               memlet=Memlet.simple(y_in,
                                                    "0:n/{}".format(veclen)))

    test_state.add_memlet_path(saxpy_node,
                               z_out,
                               src_conn='_res',
                               memlet=Memlet.simple(z_out,
                                                    "0:n/{}".format(veclen)))

    # test_sdfg.expand_library_nodes()
    saxpy_node.expand(test_sdfg, test_state, vec_width=veclen)

    return test_sdfg.compile()


def test_pure():

    print("Run BLAS test: AXPY pure...")

    configs = [(1.0, 1, dace.float32, "0"), (0.0, 1, dace.float32, "1"),
               (random.random(), 1, dace.float32, "2"),
               (1.0, 1, dace.float64, "3"), (1.0, 4, dace.float64, "4")]

    run_test(configs, "pure", "pure")


# ---------- ----------
# FPGA graph programs
# ---------- ----------
def stream_fpga_graph(veclen, precision, vendor, test_case="0"):

    DATATYPE = precision

    n = dace.symbol("n")
    a = dace.symbol("a")

    vendor_mark = "x" if vendor == "xilinx" else "i"
    test_sdfg = dace.SDFG("axpy_test_" + vendor_mark + "_" + test_case)
    test_state = test_sdfg.add_state("test_state")

    vec_type = dace.vector(precision, veclen)

    test_sdfg.add_symbol(a.name, DATATYPE)

    test_sdfg.add_array('x1', shape=[n / veclen], dtype=vec_type)
    test_sdfg.add_array('y1', shape=[n / veclen], dtype=vec_type)
    test_sdfg.add_array('z1', shape=[n / veclen], dtype=vec_type)

    saxpy_node = blas.axpy.Axpy("axpy", DATATYPE, a=a)
    saxpy_node.implementation = 'fpga'

    x_stream = streaming.StreamReadVector('x1', n, DATATYPE, veclen=veclen)

    y_stream = streaming.StreamReadVector('y1', n, DATATYPE, veclen=veclen)

    z_stream = streaming.StreamWriteVector('z1', n, DATATYPE, veclen=veclen)

    preState, postState = streaming.fpga_setup_connect_streamers(
        test_sdfg,
        test_state,
        saxpy_node, [x_stream, y_stream], ['_x', '_y'],
        saxpy_node, [z_stream], ['_res'],
        input_memory_banks=[0, 1],
        output_memory_banks=[2])

    # test_sdfg.expand_library_nodes()
    saxpy_node.expand(test_sdfg, test_state, vec_width=veclen)

    mode = "simulation" if vendor == "xilinx" else "emulator"
    dace.config.Config.set("compiler", "fpga_vendor", value=vendor)
    dace.config.Config.set("compiler", vendor, "mode", value=mode)

    return test_sdfg.compile()


def array_fpga_graph(veclen,
                     precision,
                     vendor,
                     test_case="0",
                     expansion='Intel_FPGA_DRAM'):

    DATATYPE = precision

    n = dace.symbol("n")
    a = dace.symbol("a")

    test_name = "array_axpy_test_" + vendor + "_" + test_case
    if expansion == 'Intel_FPGA_DRAM':
        test_name = "axpy_test_intel_" + test_case

    test_sdfg = dace.SDFG(test_name)
    test_sdfg.add_symbol(a.name, DATATYPE)

    vec_type = precision if expansion == 'Intel_FPGA_DRAM' else dace.vector(
        precision, veclen)
    n_adj = n if expansion == 'Intel_FPGA_DRAM' else n / veclen

    test_sdfg.add_array('x1', shape=[n_adj], dtype=vec_type)
    test_sdfg.add_array('y1', shape=[n_adj], dtype=vec_type)
    test_sdfg.add_array('z1', shape=[n_adj], dtype=vec_type)

    ###########################################################################
    # Copy data to FPGA

    copy_in_state = test_sdfg.add_state("copy_to_device")

    in_host_x = copy_in_state.add_read("x1")
    in_host_y = copy_in_state.add_read("y1")

    test_sdfg.add_array("device_x",
                        shape=[n_adj],
                        dtype=vec_type,
                        storage=dace.dtypes.StorageType.FPGA_Global,
                        transient=True)
    test_sdfg.add_array("device_y",
                        shape=[n_adj],
                        dtype=vec_type,
                        storage=dace.dtypes.StorageType.FPGA_Global,
                        transient=True)

    in_device_x = copy_in_state.add_write("device_x")
    in_device_y = copy_in_state.add_write("device_y")

    copy_in_state.add_memlet_path(in_host_x,
                                  in_device_x,
                                  memlet=Memlet.simple(in_host_x,
                                                       "0:{}".format(n_adj)))
    copy_in_state.add_memlet_path(in_host_y,
                                  in_device_y,
                                  memlet=Memlet.simple(in_host_y,
                                                       "0:{}".format(n_adj)))

    ###########################################################################
    # Copy data from FPGA
    copy_out_state = test_sdfg.add_state("copy_to_host")

    test_sdfg.add_array("device_z",
                        shape=[n_adj],
                        dtype=vec_type,
                        storage=dace.dtypes.StorageType.FPGA_Global,
                        transient=True)

    out_device = copy_out_state.add_read("device_z")
    out_host = copy_out_state.add_write("z1")

    copy_out_state.add_memlet_path(out_device,
                                   out_host,
                                   memlet=Memlet.simple(out_host,
                                                        "0:{}".format(n_adj)))

    ########################################################################
    # FPGA State

    fpga_state = test_sdfg.add_state("fpga_state")

    x = fpga_state.add_read("device_x")
    y = fpga_state.add_read("device_y")
    z = fpga_state.add_write("device_z")

    saxpy_node = blas.axpy.Axpy("axpy", DATATYPE, a=a)
    saxpy_node.implementation = expansion

    fpga_state.add_memlet_path(x,
                               saxpy_node,
                               dst_conn="_x",
                               memlet=Memlet.simple(x, "0:{}".format(n_adj)))
    fpga_state.add_memlet_path(y,
                               saxpy_node,
                               dst_conn="_y",
                               memlet=Memlet.simple(y, "0:{}".format(n_adj)))
    fpga_state.add_memlet_path(saxpy_node,
                               z,
                               src_conn="_res",
                               memlet=Memlet.simple(z, "0:{}".format(n_adj)))

    ######################################
    # Interstate edges
    test_sdfg.add_edge(copy_in_state, fpga_state,
                       dace.sdfg.sdfg.InterstateEdge())
    test_sdfg.add_edge(fpga_state, copy_out_state,
                       dace.sdfg.sdfg.InterstateEdge())

    #########
    # Validate
    test_sdfg.fill_scope_connectors()
    test_sdfg.validate()

    # test_sdfg.~~~()
    saxpy_node.expand(test_sdfg, fpga_state, vec_width=veclen)

    mode = "simulation" if vendor == "xilinx" else "emulator"
    dace.config.Config.set("compiler", "fpga_vendor", value=vendor)
    dace.config.Config.set("compiler", vendor, "mode", value=mode)

    return test_sdfg.compile()


def test_fpga(type, vendor):

    print("Run BLAS test: AXPY fpga", vendor + "...")

    configs = [(0.0, 1, dace.float32, "0"), (1.0, 1, dace.float32, "1"),
               (random.random(), 1, dace.float32, "2"),
               (1.0, 1, dace.float64, "3"), (1.0, 4, dace.float64, "4")]

    run_test(configs, type, vendor)

    print(" --> passed")


if __name__ == "__main__":

    cmdParser = argparse.ArgumentParser(allow_abbrev=False)

    cmdParser.add_argument("--target", dest="target", default="pure")

    args = cmdParser.parse_args()

    if args.target == "intel_fpga" or args.target == "xilinx":
        test_fpga("fpga", args.target)
        test_fpga("fpga_array", args.target)
    elif args.target == "intel_fpga_dram":
        test_fpga("intel_fpga_dram", "intel_fpga")
    else:
        test_pure()
