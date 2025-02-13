#!/usr/bin/env python3
# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.

import numpy as np

import argparse
import scipy
import random

import dace
from dace.fpga_testing import fpga_test
from dace.memlet import Memlet

import dace.libraries.blas as blas
from dace.transformation.interstate import FPGATransformSDFG, InlineSDFG
from dace.transformation.dataflow import StreamingMemory

from dace.libraries.standard.memory import aligned_ndarray


def run_test(configs, target):

    n = int(1 << 13)

    for i, config in enumerate(configs):

        a, veclen, dtype = config

        x = aligned_ndarray(np.random.uniform(0, 100, n).astype(dtype.type), alignment=256)
        y = aligned_ndarray(np.random.uniform(0, 100, n).astype(dtype.type), alignment=256)
        y_ref = y.copy()

        a = dtype(a)

        ref_result = reference_result(x, y_ref, a)

        if target == "fpga_stream":
            sdfg = stream_fpga_graph(veclen, dtype, "fpga", i)
        elif target == "fpga_array":
            sdfg = fpga_graph(veclen, dtype, "fpga", i)
        else:
            sdfg = pure_graph(veclen, dtype, "pure", i)
        program = sdfg.compile()

        with dace.config.set_temporary('compiler', 'allow_view_arguments', value=True):

            if target in ["fpga_stream", "fpga_array"]:
                program(x=x, y=y, a=a, n=np.int32(n))
                ref_norm = np.linalg.norm(y - ref_result) / n
            else:
                program(x=x, y=y, a=a, n=np.int32(n))
                ref_norm = np.linalg.norm(y - ref_result) / n

        if ref_norm >= 1e-5:
            raise ValueError(f"Failed validation for target {target}.")

    return sdfg


def reference_result(x_in, y_in, alpha):
    return scipy.linalg.blas.saxpy(x_in, y_in, a=alpha)


def pure_graph(veclen, dtype, implementation, test_case):

    n = dace.symbol("n")
    a = dace.symbol("a")

    sdfg_name = f"axpy_test_{implementation}_{test_case}_w{veclen}"

    sdfg = dace.SDFG(sdfg_name)
    test_state = sdfg.add_state("test_state")

    vtype = dace.vector(dtype, veclen)

    sdfg.add_symbol(a.name, dtype)

    sdfg.add_array("x", shape=[n / veclen], dtype=vtype)
    sdfg.add_array("y", shape=[n / veclen], dtype=vtype)

    x_in = test_state.add_read("x")
    y_in = test_state.add_read("y")
    y_out = test_state.add_write("y")

    axpy_node = blas.axpy.Axpy("axpy", a)
    axpy_node.implementation = implementation

    test_state.add_memlet_path(x_in, axpy_node, dst_conn="_x", memlet=Memlet(f"x[0:n/{veclen}]"))
    test_state.add_memlet_path(y_in, axpy_node, dst_conn="_y", memlet=Memlet(f"y[0:n/{veclen}]"))
    test_state.add_memlet_path(axpy_node, y_out, src_conn="_res", memlet=Memlet(f"y[0:n/{veclen}]"))

    sdfg.expand_library_nodes()

    return sdfg


def test_pure():
    configs = [(0.5, 1, dace.float32), (1.0, 4, dace.float64)]
    run_test(configs, "pure")


def fpga_graph(veclen, dtype, test_case, expansion):
    sdfg = pure_graph(veclen, dtype, test_case, expansion)
    sdfg.apply_transformations_repeated([FPGATransformSDFG, InlineSDFG])
    return sdfg


def stream_fpga_graph(veclen, precision, test_case, expansion):
    sdfg = fpga_graph(veclen, precision, test_case, expansion)
    sdfg.expand_library_nodes()
    sdfg.apply_transformations_repeated([InlineSDFG, StreamingMemory], [{}, {"storage": dace.StorageType.FPGA_Local}])
    return sdfg


@fpga_test()
def test_axpy_fpga_array():
    configs = [(0.5, 1, dace.float32), (1.0, 4, dace.float64)]
    return run_test(configs, "fpga_array")


@fpga_test()
def test_axpy_fpga_stream():
    configs = [(0.5, 1, dace.float32), (1.0, 4, dace.float64)]
    return run_test(configs, "fpga_stream")


if __name__ == "__main__":

    cmdParser = argparse.ArgumentParser(allow_abbrev=False)

    cmdParser.add_argument("--target", dest="target", default="pure")

    args = cmdParser.parse_args()

    if args.target == "fpga":
        test_axpy_fpga_array(None)
        test_axpy_fpga_stream(None)
    elif args.target == "pure":
        test_pure()
    else:
        raise RuntimeError(f"Unknown target \"{args.target}\".")
