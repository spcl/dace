#!/usr/bin/env python3
# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.

from dace import dtypes
from dace.sdfg import utils
import numpy as np

import argparse
import scipy
import random

import dace
from dace.memlet import Memlet

import dace.libraries.blas as blas
from dace.transformation.interstate import FPGATransformSDFG, InlineSDFG
from dace.transformation.dataflow import StreamingMemory
from dace.transformation.dataflow import hbm_copy_transform
from dace.transformation import optimizer

from dace.libraries.standard.memory import aligned_ndarray


def run_test(configs, target):

    n = int(1 << 13)

    for i, config in enumerate(configs):

        a, veclen, dtype = config

        x = aligned_ndarray(np.random.uniform(0, 100, n).astype(dtype.type),
                            alignment=256)
        y = aligned_ndarray(np.random.uniform(0, 100, n).astype(dtype.type),
                            alignment=256)
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

        if target in ["fpga_stream", "fpga_array"]:
            program(x=x, y=y, a=a, n=np.int32(n))
            ref_norm = np.linalg.norm(y - ref_result) / n
        else:
            program(x=x, y=y, a=a, n=np.int32(n))
            ref_norm = np.linalg.norm(y - ref_result) / n

        if ref_norm >= 1e-5:
            raise ValueError(f"Failed validation for target {target}.")


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

    test_state.add_memlet_path(x_in,
                               axpy_node,
                               dst_conn="_x",
                               memlet=Memlet(f"x[0:n/{veclen}]"))
    test_state.add_memlet_path(y_in,
                               axpy_node,
                               dst_conn="_y",
                               memlet=Memlet(f"y[0:n/{veclen}]"))
    test_state.add_memlet_path(axpy_node,
                               y_out,
                               src_conn="_res",
                               memlet=Memlet(f"y[0:n/{veclen}]"))

    sdfg.expand_library_nodes()

    return sdfg


def test_pure():
    configs = [(0.5, 1, dace.float32), (1.0, 4, dace.float64)]
    run_test(configs, "pure")


def fpga_graph(veclen, dtype, test_case, expansion):
    sdfg = pure_graph(veclen, dtype, test_case, expansion)
    sdfg.apply_transformations_repeated([FPGATransformSDFG, InlineSDFG])
    return sdfg

def fpga_hbm_graph(banks_per_array):
        N = dace.symbol("n")

        sdfg = dace.SDFG("axpy_test_hbm")
        sdfg.add_symbol("a", dace.float32)
        state = sdfg.add_state("axpy")
        axpy_node = blas.Axpy("saxpy_node")
        axpy_node.implementation = "fpga_hbm"
        state.add_node(axpy_node)
        create_hbm_access(state, "in1", f"hbm.0:{banks_per_array}", 
            [banks_per_array, N], axpy_node, "_x", False, "in1")
        create_hbm_access(state, "in2", f"hbm.{banks_per_array}:{2*banks_per_array}",
            [banks_per_array, N], axpy_node, "_y", False, "in2")
        create_hbm_access(state, "out", f"hbm.{2*banks_per_array}:{3*banks_per_array}",
            [banks_per_array, N], axpy_node, "_res", True, "out")
        axpy_node.expand(sdfg, state)
        
        sdfg.sdfg_list[2].symbols["a"] = sdfg.sdfg_list[1].symbols["a"] #Why does inference fail?

        sdfg.apply_fpga_transformations(False)
        utils.update_array_shape(sdfg, "in1", [banks_per_array*N])
        utils.update_array_shape(sdfg, "in2", [banks_per_array*N])
        utils.update_array_shape(sdfg, "out", [banks_per_array*N])
        sdfg.arrays["in1"].storage = dtypes.StorageType.CPU_Heap
        sdfg.arrays["in2"].storage = dtypes.StorageType.CPU_Heap
        sdfg.arrays["out"].storage = dtypes.StorageType.CPU_Heap
        for xform in optimizer.Optimizer(sdfg).get_pattern_matches(
            patterns=[hbm_copy_transform.HbmCopyTransform]):
            xform.apply(sdfg)

        return sdfg

def stream_fpga_graph(veclen, precision, test_case, expansion):
    sdfg = fpga_graph(veclen, precision, test_case, expansion)
    sdfg.expand_library_nodes()
    sdfg.apply_transformations_repeated(
        [InlineSDFG, StreamingMemory], [{}, {
            "storage": dace.StorageType.FPGA_Local
        }])
    return sdfg


def _test_fpga(target):
    configs = [(0.5, 1, dace.float32), (1.0, 4, dace.float64)]
    run_test(configs, target)

if __name__ == "__main__":

    cmdParser = argparse.ArgumentParser(allow_abbrev=False)

    cmdParser.add_argument("--target", dest="target", default="pure")

    args = cmdParser.parse_args()

    if args.target == "fpga":
        _test_fpga("fpga_array")
        _test_fpga("fpga_stream")
    elif args.target == "pure":
        test_pure()
    else:
        raise RuntimeError(f"Unknown target \"{args.target}\".")
