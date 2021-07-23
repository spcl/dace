#!/usr/bin/env python3
# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.

from dace import dtypes, subsets
from dace.sdfg import utils, nodes
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
        elif target == "fpga_hbm":
            sdfg = fpga_hbm_graph(veclen, dtype, i)
        else:
            sdfg = pure_graph(veclen, dtype, "pure", i)
        program = sdfg.compile()

        if target in ["fpga_stream", "fpga_array", "fpga_hbm"]:
            program(x=x, y=y, a=a, n=np.int32(n))
            ref_norm = np.linalg.norm(y - ref_result) / n
        else:
            program(x=x, y=y, a=a, n=np.int32(n))
            ref_norm = np.linalg.norm(y - ref_result) / n

        if ref_norm >= 1e-5:
            raise ValueError(f"Failed validation for target {target}.")


def reference_result(x_in, y_in, alpha):
    return scipy.linalg.blas.saxpy(x_in, y_in, a=alpha)


def pure_graph(veclen, dtype, implementation, test_case, expand_lib_node=True):

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

    if expand_lib_node:
        sdfg.expand_library_nodes()

    return sdfg


def test_pure():
    configs = [(0.5, 1, dace.float32), (1.0, 4, dace.float64)]
    run_test(configs, "pure")


def fpga_graph(veclen, dtype, test_case, expansion):
    sdfg = pure_graph(veclen, dtype, test_case, expansion, False)
    sdfg.apply_transformations_repeated([FPGATransformSDFG, InlineSDFG])
    return sdfg


def fpga_hbm_graph(veclen, dtype, expansion):
    sdfg = pure_graph(veclen, dtype, "fpga_hbm", expansion, False)

    banks_per_array = 2
    per_array_size = sdfg.arrays["x"].shape[0] / banks_per_array
    utils.update_array_shape(sdfg, "x", [banks_per_array, per_array_size])
    utils.update_array_shape(sdfg, "y", [banks_per_array, per_array_size])
    sdfg.arrays["x"].location["memorytype"] = "HBM"
    sdfg.arrays["y"].location["memorytype"] = "HBM"
    sdfg.arrays["x"].location["bank"] = f"0:{banks_per_array}"
    sdfg.arrays["y"].location["bank"] = f"{banks_per_array}:{2*banks_per_array}"
    state = sdfg.states()[0]
    for node in state:
        if isinstance(node, nodes.AccessNode):
            utils.update_path_subsets(
                state, node,
                subsets.Range.from_string(
                    f"0:{banks_per_array}, 0:{per_array_size}"))
    libnode = list(
        filter(lambda x: isinstance(x, nodes.LibraryNode), state.nodes()))[0]
    libnode.n = dace.symbolic.pystr_to_symbolic(f"n/{banks_per_array}")
    sdfg.apply_transformations_repeated([FPGATransformSDFG, InlineSDFG])
    sdfg.expand_library_nodes()
    utils.update_array_shape(sdfg, "x", [per_array_size * banks_per_array])
    utils.update_array_shape(sdfg, "y", [per_array_size * banks_per_array])
    sdfg.arrays["x"].storage = dtypes.StorageType.CPU_Heap
    sdfg.arrays["y"].storage = dtypes.StorageType.CPU_Heap
    for xform in optimizer.Optimizer(sdfg).get_pattern_matches(
            patterns=[hbm_copy_transform.HbmCopyTransform]):
        xform.apply(sdfg)
    sdfg.sdfg_list[3].symbols["a"] = sdfg.sdfg_list[2].symbols[
        "a"]  # Why does inference fail?
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
        if dace.Config.get("compiler", "fpga_vendor") == "xilinx": # Only supported by xilinx
            _test_fpga("fpga_hbm")
    elif args.target == "pure":
        test_pure()
    else:
        raise RuntimeError(f"Unknown target \"{args.target}\".")
