#!/usr/bin/env python3

import numpy as np
from tqdm import tqdm

import argparse
import scipy
import random

import dace
from dace.memlet import Memlet

import dace.libraries.blas as blas
import dace.libraries.blas.utility.streaming as streaming
from dace.transformation.interstate import GPUTransformSDFG


# ---------- ----------
# Arguments & Utility
# ---------- ----------
cmdParser = argparse.ArgumentParser(allow_abbrev=False)

cmdParser.add_argument("--cublas", dest="cublas", action='store_true')
cmdParser.add_argument("--mkl", dest="mkl", action='store_true')
cmdParser.add_argument("--openblas", dest="openblas", action='store_true')
cmdParser.add_argument("--pure", dest="pure", action='store_true')
cmdParser.add_argument("--xilinx", dest="xilinx", action='store_true')
cmdParser.add_argument("--intel_fpga", dest="intel_fpga", action='store_true')

args = cmdParser.parse_args()


def aligned_ndarray(arr, alignment=64):
    """
    Allocates a and returns a copy of ``arr`` as an ``alignment``-byte aligned
    array. Useful for aligned vectorized access.
    
    Based on https://stackoverflow.com/a/20293172/6489142
    """
    if (arr.ctypes.data % alignment) == 0:
        return arr

    extra = alignment // arr.itemsize
    buf = np.empty(arr.size + extra, dtype=arr.dtype)
    ofs = (-buf.ctypes.data % alignment) // arr.itemsize
    result = buf[ofs:ofs + arr.size].reshape(arr.shape)
    np.copyto(result, arr)
    assert (result.ctypes.data % alignment) == 0
    return result


# ---------- ----------
# Ref result
# ---------- ----------
def reference_result(x_in, y_in, alpha):
    return scipy.linalg.blas.saxpy(x_in, y_in, a=alpha)


# ---------- ----------
# Pure graph program
# ---------- ----------
def pure_graph(vecWidth, precision, implementation="pure", testCase="0"):
    
    n = dace.symbol("n")
    a = dace.symbol("a")

    prec = "single" if precision == dace.float32 else "double"
    test_sdfg = dace.SDFG("axpy_test_" + prec + "_v" + str(vecWidth) + "_" + implementation + "_" + testCase)
    test_state = test_sdfg.add_state("test_state")

    test_sdfg.add_symbol(a.name, precision)

    test_sdfg.add_array('x1', shape=[n], dtype=precision)
    test_sdfg.add_array('y1', shape=[n], dtype=precision)
    test_sdfg.add_array('z1', shape=[n], dtype=precision)

    x_in = test_state.add_read('x1')
    y_in = test_state.add_read('y1')
    z_out = test_state.add_write('z1')

    saxpy_node = blas.axpy.Axpy("axpy", precision, vecWidth=vecWidth)
    saxpy_node.implementation = implementation

    test_state.add_memlet_path(
        x_in, saxpy_node,
        dst_conn='_x',
        memlet=Memlet.simple(x_in, "0:n", veclen=vecWidth)
    )
    test_state.add_memlet_path(
        y_in, saxpy_node,
        dst_conn='_y',
        memlet=Memlet.simple(y_in, "0:n", veclen=vecWidth)
    )

    test_state.add_memlet_path(
        saxpy_node, z_out,
        src_conn='_res',
        memlet=Memlet.simple(z_out, "0:n", veclen=vecWidth)
    )

    test_sdfg.expand_library_nodes()

    return test_sdfg.compile(optimizer=False)


def test_pure():

    print("Run BLAS test: AXPY pure...")

    configs = [
        (1.0, 1, dace.float32, "0"),
        (0.0, 1, dace.float32, "1"),
        (random.random(), 1, dace.float32, "2"),
        (1.0, 1, dace.float64, "3")
        # (1.0, 4, dace.float64, "4")
    ]

    testN = int(2**13)

    for config in configs:

        prec = np.float32 if config[2] == dace.float32 else np.float64
        a = aligned_ndarray(np.random.randint(100, size=testN).astype(prec), alignment=256)
        b = aligned_ndarray(np.random.randint(100, size=testN).astype(prec), alignment=256)
        b_ref = b.copy()

        c = aligned_ndarray(np.zeros(testN).astype(prec), alignment=256)
        alpha = np.float32(config[0]) if config[2] == dace.float32 else np.float64(config[0])

        ref_result = reference_result(a, b_ref, alpha)

        compiledGraph = pure_graph(config[1], config[2], testCase=config[3])

        compiledGraph(x1=a, y1=b, a=alpha, z1=c, n=np.int32(testN))

        ref_norm = np.linalg.norm(c - ref_result) / testN
        passed = ref_norm < 1e-5

        if not passed:
            raise RuntimeError('AXPY pure implementation wrong test results on config: ', config)

    print(" --> passed")


# ---------- ----------
# CPU library graph program
# ---------- ----------
def cpu_graph(precision, implementation, testCase="0"):
    
    n = dace.symbol("n")
    a = dace.symbol("a")

    prec = "single" if precision == dace.float32 else "double"
    test_sdfg = dace.SDFG("axpy_test_" + prec + "_" + implementation + "_" + testCase)
    test_state = test_sdfg.add_state("test_state")

    test_sdfg.add_symbol(a.name, precision)

    test_sdfg.add_array('x1', shape=[n], dtype=precision)
    test_sdfg.add_array('y1', shape=[n], dtype=precision)

    x_in = test_state.add_read('x1')
    y_in = test_state.add_read('y1')
    z_out = test_state.add_write('y1')

    saxpy_node = blas.axpy.Axpy("axpy", precision)
    saxpy_node.implementation = implementation

    test_state.add_memlet_path(
        x_in, saxpy_node,
        dst_conn='_x',
        memlet=Memlet.simple(x_in, "0:n")
    )
    test_state.add_memlet_path(
        y_in, saxpy_node,
        dst_conn='_y',
        memlet=Memlet.simple(y_in, "0:n")
    )

    test_state.add_memlet_path(
        saxpy_node, z_out,
        src_conn='_res',
        memlet=Memlet.simple(z_out, "0:n")
    )

    if saxpy_node.implementation == "cublas":
        test_sdfg.apply_transformations(GPUTransformSDFG)

    test_sdfg.expand_library_nodes()

    return test_sdfg.compile(optimizer=False)


def test_cpu(implementation):
    
    print("Run BLAS test: AXPY", implementation + "...")

    configs = [
        (1.0, 1, dace.float32, "0"),
        (0.0, 1, dace.float32, "1"),
        (random.random(), 1, dace.float32, "2"),
        (1.0, 1, dace.float64, "3")
    ]

    testN = int(2**13)

    for config in configs:

        prec = np.float32 if config[2] == dace.float32 else np.float64
        a = np.random.randint(100, size=testN).astype(prec)
        b = np.random.randint(100, size=testN).astype(prec)
        b_ref = b.copy()

        # c = np.zeros(testN).astype(prec)
        alpha = np.float32(config[0]) if config[2] == dace.float32 else np.float64(config[0])

        ref_result = reference_result(a, b_ref, alpha)

        compiledGraph = cpu_graph(config[2], implementation, testCase=config[3])

        compiledGraph(x1=a, y1=b, a=alpha, z1=b, n=np.int32(testN))

        ref_norm = np.linalg.norm(b - ref_result) / testN
        passed = ref_norm < 1e-5

        if not passed:
            raise RuntimeError("AXPY " + implementation + " implementation wrong test results")

    print(" --> passed")


# ---------- ----------
# GPU Cuda graph program
# ---------- ----------
# def gpu_graph():
#     return pure_graph(1, precision, implementation=implementation)


def test_gpu():
    test_cpu("cublas")



# ---------- ----------
# FPGA graph program
# ---------- ----------
def fpga_graph(vecWidth, precision, vendor):


    print("Run BLAS test: AXPY fpga...")
    
    DATATYPE = precision

    n = dace.symbol("n")
    a = dace.symbol("a")

    test_sdfg = dace.SDFG("axpy_test")
    test_state = test_sdfg.add_state("test_state")

    test_sdfg.add_symbol(a.name, DATATYPE)

    test_sdfg.add_array('x1', shape=[n], dtype=DATATYPE)
    test_sdfg.add_array('y1', shape=[n], dtype=DATATYPE)
    test_sdfg.add_array('z1', shape=[n], dtype=DATATYPE)

    saxpy_node = blas.level1.axpy.Axpy("axpy", DATATYPE , vecWidth=vecWidth, n=n, a=a)
    saxpy_node.implementation = 'fpga_stream'

    x_stream = streaming.streamReadVector(
        'x1',
        n,
        typeDace,
        vecWidth=vecWidth
    )

    y_stream = streaming.streamReadVector(
        'y1',
        n,
        typeDace,
        vecWidth=vecWidth
    )

    z_stream = streaming.streamWriteVector(
        'z1',
        n,
        typeDace,
        vecWidth=vecWidth
    )

    preState, postState = streaming.fpga_setupConnectStreamers(
        test_sdfg,
        test_state,
        saxpy_node,
        [x_stream, y_stream],
        ['_x', '_y'],
        saxpy_node,
        [z_stream],
        ['_res'],
        inputMemoryBanks=[0, 1],
        outputMemoryBanks=[2]
    )

    test_sdfg.expand_library_nodes()

    mode = "simulation" if vendor == "xilinx" else "emulator"
    dace.config.Config.set("compiler", "fpga_vendor", value=vendor)
    dace.config.Config.set("compiler", vendor, "mode", value=mode)

    return test_sdfg.compile(optimizer=False)


def test_fpga(vendor):
    pass



if __name__ == "__main__":
    
    if args.pure:
        test_pure()

    if args.mkl:
        test_cpu("mkl")

    if args.openblas:
        test_cpu("openblas")

    if args.cublas:
        test_gpu()

    if args.xilinx:
        test_fpga("xilinx")

    if args.intel_fpga:
        test_fpga("intel_fpga")