#!/usr/bin/env python3

import numpy as np

import argparse
import scipy

import dace
from dace.memlet import Memlet

import dace.libraries.blas as blas

import dace.libraries.blas.utility.fpga_helper as streaming


def fpga_graph(veclen, precision, vendor, test_case, expansion):

    DATATYPE = precision

    n = dace.symbol("n")
    a = dace.symbol("a")

    vendor_mark = "x" if vendor == "xilinx" else "i"
    test_sdfg = dace.SDFG("dot_test_" + vendor_mark + "_" + test_case)
    test_state = test_sdfg.add_state("test_state")

    vec_type = dace.vector(precision, veclen)

    test_sdfg.add_array('x', shape=[n / veclen], dtype=vec_type)
    test_sdfg.add_array('y', shape=[n / veclen], dtype=vec_type)
    test_sdfg.add_array('r', shape=[1], dtype=precision)

    dot_node = blas.Dot("dot")
    dot_node.implementation = expansion

    x_stream = streaming.StreamReadVector('x', n, DATATYPE, veclen=veclen)

    y_stream = streaming.StreamReadVector('y', n, DATATYPE, veclen=veclen)

    z_stream = streaming.StreamWriteVector('r', 1, DATATYPE)

    pre_state, post_state = streaming.fpga_setup_connect_streamers(
        test_sdfg,
        test_state,
        dot_node, [x_stream, y_stream], ['_x', '_y'],
        dot_node, [z_stream], ['_result'],
        input_memory_banks=[0, 1],
        output_memory_banks=[2])

    test_sdfg.expand_library_nodes()

    mode = "simulation" if vendor == "xilinx" else "emulator"
    dace.config.Config.set("compiler", "fpga_vendor", value=vendor)
    dace.config.Config.set("compiler", vendor, "mode", value=mode)

    test_sdfg.fill_scope_connectors()

    return test_sdfg


def pure_graph(dtype):
    n = dace.symbol("n")

    sdfg = dace.SDFG("dot_product")

    state = sdfg.add_state("dot")

    sdfg.add_array("x", [n], dtype)
    sdfg.add_array("y", [n], dtype)
    sdfg.add_array("r", [1], dtype)

    x = state.add_read("x")
    y = state.add_read("y")
    result = state.add_write("r")

    dot_node = blas.Dot("dot")
    dot_node.implementation = "pure"

    state.add_memlet_path(x,
                          dot_node,
                          dst_conn="_x",
                          memlet=Memlet.simple(x, "0:n", num_accesses=n))
    state.add_memlet_path(y,
                          dot_node,
                          dst_conn="_y",
                          memlet=Memlet.simple(y, "0:n", num_accesses=n))
    state.add_memlet_path(dot_node,
                          result,
                          src_conn="_result",
                          memlet=Memlet.simple(result, "0", num_accesses=1))

    return sdfg


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("N", type=int, nargs="?", default=64)
    parser.add_argument("--target", dest="target", default="pure")
    parser.add_argument("--vector-length", type=int, default=16)
    args = parser.parse_args()
    size = args.N

    if args.target == "pure":
        sdfg = pure_graph(dace.float32)
    elif args.target == "intel_fpga":
        sdfg = fpga_graph(args.vector_length,
                          dace.float32,
                          args.target,
                          "0",
                          expansion="FPGA_Accumulate")
    elif args.target == "xilinx":
        sdfg = fpga_graph(args.vector_length,
                          dace.float32,
                          args.target,
                          "0",
                          expansion="FPGA_PartialSums")
    else:
        print(f"Unsupported target: {args.target}")
        exit(-1)

    dot = sdfg.compile()

    x = np.ndarray(size, dtype=np.float32)
    y = np.ndarray(size, dtype=np.float32)
    result = np.ndarray(1, dtype=np.float32)

    x[:] = np.random.rand(size).astype(np.float32)
    y[:] = np.random.rand(size).astype(np.float32)

    result[0] = 0

    dot(x=x, y=y, r=result, n=size)

    ref = scipy.linalg.blas.sdot(x, y)

    diff = abs(result[0] - ref)
    if diff >= 1e-6 * ref:
        print("Unexpected result returned from dot product: "
              "got {}, expected {}".format(result[0], ref))
    else:
        print("Ok")
