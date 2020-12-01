#!/usr/bin/env python3

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


def run_program(program, x, y, A, res, testN, ref_result, queue, alpha, beta):

    program(x1=x, y1=y, A1=A, res1=res, n=np.int32(testN), m=np.in32(testN), a=alpha, b=beta)
    ref_norm = np.linalg.norm(res - ref_result) / testN

    queue.put(ref_norm)


def run_test(configs, target, implementation, overwrite_y=False):

    # TODO: add testM to test non-square matrices
    testN = int(2**13)

    for config in configs:

        prec = np.float32 if config[2] == dace.float32 else np.float64
        A = aligned_ndarray(np.random.uniform(0, 100, testN * testN).astype(prec),
                            alignment=256)
        x = aligned_ndarray(np.random.uniform(0, 100, testN).astype(prec),
                            alignment=256)
        y = aligned_ndarray(np.random.uniform(0, 100, testN).astype(prec),
                            alignment=256)
        
        res = aligned_ndarray(np.zeros(testN).astype(prec), alignment=256)

        alpha = np.float32(
            config[0]) if config[2] == dace.float32 else np.float64(config[0])

        # TODO: add separate beta value to test config
        beta = np.float32(
            config[0]) if config[2] == dace.float32 else np.float64(config[0])


        ref_result = reference_result(A, x, y, alpha, beta)

        program = None
        if target == "fpga":
            program = fpga_graph(config[1],
                                 config[2],
                                 implementation,
                                 testCase=config[3])
        else:
            # TODO: add pure test
            pass

        ref_norm = 0
        if target == "fpga":

            # Run FPGA tests in a different process to avoid issues with Intel OpenCL tools
            queue = Queue()
            p = Process(target=run_program,
                        args=(program, x, y, A, res, testN, ref_result, queue, alpha, beta))
            p.start()
            p.join()
            ref_norm = queue.get()

        else:
            program(x1=x, y1=y, A1=A, res1=res, n=np.int32(testN), m=np.in32(testN), a=alpha, b=beta)
            ref_norm = np.linalg.norm(b - ref_result) / testN

        passed = ref_norm < 1e-5

        if not passed:
            raise RuntimeError(
                'GEMV {} implementation wrong test results on config: '.format(
                    implementation), config)


# ---------- ----------
# Ref result
# ---------- ----------
def reference_result(A_in, x_in, y_in, alpha, beta):
    return alpha * A_in @ x_in + beta * y_in


# ---------- ----------
# FPGA graph program
# ---------- ----------
def fpga_graph(veclen, precision, vendor, testCase="0"):

    DATATYPE = precision
    nRows = dace.symbol("nRows")
    mCols = dace.symbol("mCols")

    a = dace.symbol("a")
    b = dace.symbol("b")

    # TODO: expand tests to consider different tile size configs
    rowTile = 4
    colTile = 4
    partialWidth = 2
    vecM = veclen

    vendor_mark = "x" if vendor == "xilinx" else "i"
    test_sdfg = dace.SDFG("gemv_test_" + vendor_mark + "_" + testCase)
    test_state = test_sdfg.add_state("test_state")

    test_sdfg.add_symbol(a.name, DATATYPE)

    if b != 0:
        test_sdfg.add_symbol(b.name, DATATYPE)

    test_sdfg.add_array('A1', shape=[nRows*mCols], dtype=DATATYPE)
    test_sdfg.add_array('x1', shape=[mCols], dtype=DATATYPE)
    test_sdfg.add_array('y1', shape=[nRows], dtype=DATATYPE)
    test_sdfg.add_array('res1', shape=[nRows], dtype=DATATYPE)

    x_stream = streaming.StreamReadVector(
        'x1',
        mCols,
        DATATYPE,
        vecWidth=vecM,
        repeat='{}/{}'.format(nRows, rowTile)
    )

    y_stream = None
    if b != 0:
        y_stream = streaming.StreamReadVector(
            'y1',
            nRows,
            DATATYPE,
            vecWidth=1,
        )

    A_stream = streaming.StreamReadMatrixFull(
        'A1',
        nRows,
        mCols,
        rowTile,
        colTile,
        DATATYPE,
        tileByRow=True,
        vecWidth=vecM
    )

    res_stream = streaming.StreamWriteVector(
        'res1',
        nRows,
        DATATYPE
    )

    gemv_node = blas.gemv.Gemv(
        "blas_gemv",
        dtype=DATATYPE,
        nTile=rowTile,
        mTile=colTile,
        partialWidth=partialWidth,
        n=nRows,
        m=mCols,
        vecWidthM=vecM,
        a=a, b=b
    )
    gemv_node.implementation = 'fpga_stream'

    preState, postState = streaming.fpga_setup_connect_streamers(
        test_sdfg,
        test_state,
        gemv_node,
        [x_stream, y_stream, A_stream],
        ['_x', '_y', '_A'],
        gemv_node,
        [res_stream],
        ['_res']
    )

    test_sdfg.expand_library_nodes()

    mode = "simulation" if vendor == "xilinx" else "emulator"
    dace.config.Config.set("compiler", "fpga_vendor", value=vendor)
    dace.config.Config.set("compiler", vendor, "mode", value=mode)

    return test_sdfg.compile()


def test_fpga(vendor):

    print("Run BLAS test: GEMV fpga", vendor + "...")

    configs = [(0.0, 1, dace.float32, "0"), (1.0, 1, dace.float32, "1"),
               (random.random(), 1, dace.float32, "2"),
               (1.0, 1, dace.float64, "3"), (1.0, 4, dace.float64, "4")]

    run_test(configs, "fpga", vendor)

    print(" --> passed")


if __name__ == "__main__":

    cmdParser = argparse.ArgumentParser(allow_abbrev=False)
    cmdParser.add_argument("--target", dest="target", default="pure")

    args = cmdParser.parse_args()

    if args.target == "intel_fpga" or args.target == "xilinx":
        test_fpga(args.target)
    else:
        # TODO: add pure test
        pass
