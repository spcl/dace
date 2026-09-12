#!/usr/bin/env python3
# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.

import numpy as np

import argparse
import scipy

import dace
from dace.memlet import Memlet

import dace.libraries.blas as blas

import pytest


def pure_graph(implementation, dtype, veclen):

    sdfg_name = f"dot_{implementation}_{dtype.ctype}_w{veclen}"
    sdfg = dace.SDFG(sdfg_name)

    state = sdfg.add_state("dot")

    n = dace.symbol("n")
    a = dace.symbol("a")

    vtype = dace.vector(dtype, veclen)

    sdfg.add_array("x", [n / veclen], vtype)
    sdfg.add_array("y", [n / veclen], vtype)
    sdfg.add_array("r", [1], dtype)

    x = state.add_read("x")
    y = state.add_read("y")
    result = state.add_write("r")

    dot_node = blas.Dot("dot")
    dot_node.implementation = implementation
    dot_node.n = n

    state.add_memlet_path(x, dot_node, dst_conn="_x", memlet=Memlet(f"x[0:{n}/{veclen}]"))
    state.add_memlet_path(y, dot_node, dst_conn="_y", memlet=Memlet(f"y[0:{n}/{veclen}]"))
    state.add_memlet_path(dot_node, result, src_conn="_result", memlet=Memlet(f"r[0]"))

    return sdfg


def run_test(target, size, vector_length):
    if target == "pure":
        sdfg = pure_graph("pure", dace.float32, vector_length)
    else:
        print(f"Unsupported target: {target}")
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
        raise ValueError("Unexpected result returned from dot product: "
                         "got {}, expected {}".format(result[0], ref))

    return sdfg


def test_validate_accepts_reparsed_symbol_instances():
    """Same-named ``n`` reaching Dot.validate at two dtypes (one operand's dace.int32
    instance vs the other's dace.int64 instance) must compare equal by name -- and a
    genuine size mismatch must still be rejected."""
    N32 = dace.symbol("N", dace.int32)
    N64 = dace.symbol("N", dace.int64)
    sdfg = dace.SDFG("dot_validate_symbol_identity")
    sdfg.add_array("x", [N32], dace.float64)
    sdfg.add_array("y", [N64], dace.float64)
    sdfg.add_array("r", [1], dace.float64)
    state = sdfg.add_state()
    node = blas.Dot("dot")
    state.add_node(node)
    state.add_edge(state.add_read("x"), None, node, "_x", Memlet.from_array("x", sdfg.arrays["x"]))
    state.add_edge(state.add_read("y"), None, node, "_y", Memlet.from_array("y", sdfg.arrays["y"]))
    state.add_edge(node, "_result", state.add_write("r"), None, Memlet.from_array("r", sdfg.arrays["r"]))
    node.validate(sdfg, state)  # must not raise

    sdfg.arrays["y"].shape = (dace.symbol("P", dace.int32), )
    y_edge = next(e for e in state.in_edges(node) if e.dst_conn == "_y")
    y_edge.data = Memlet.from_array("y", sdfg.arrays["y"])
    with pytest.raises(ValueError):
        node.validate(sdfg, state)


def test_dot_pure():
    assert isinstance(run_test("pure", 64, 1), dace.SDFG)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("N", type=int, nargs="?", default=64)
    parser.add_argument("--target", dest="target", default="pure")
    parser.add_argument("--vector-length", type=int, default=16)
    args = parser.parse_args()
    size = args.N

    run_test(args.target, size, args.vector_length)
