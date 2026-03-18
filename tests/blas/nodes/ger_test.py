#!/usr/bin/env python3
# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.

import numpy as np

import argparse
import scipy

import dace
import dace.libraries.blas as blas
from dace.libraries.standard.memory import aligned_ndarray
from dace.memlet import Memlet
from dace.transformation.interstate.sdfg_nesting import InlineSDFG


def pure_graph(implementation, dtype, veclen):

    m = dace.symbol("m")
    n = dace.symbol("n")
    vtype = dace.vector(dtype, veclen)

    sdfg = dace.SDFG("ger_test")

    state = sdfg.add_state("ger")

    sdfg.add_symbol("alpha", dtype)

    sdfg.add_array("x", shape=[m], dtype=dtype)
    sdfg.add_array("y", shape=[n / veclen], dtype=vtype)
    sdfg.add_array("A", shape=[m, n / veclen], dtype=vtype)
    sdfg.add_array("res", shape=[m, n / veclen], dtype=vtype)

    x = state.add_read("x")
    y = state.add_read("y")
    A = state.add_read("A")
    res = state.add_write("res")

    ger_node = blas.Ger(name="ger")
    ger_node.implementation = implementation

    state.add_memlet_path(x, ger_node, dst_conn="_x", memlet=Memlet("x[0:m]"))
    state.add_memlet_path(y, ger_node, dst_conn="_y", memlet=Memlet(f"y[0:n/{veclen}]"))
    state.add_memlet_path(A, ger_node, dst_conn="_A", memlet=Memlet(f"A[0:m, 0:n/{veclen}]"))
    state.add_memlet_path(ger_node, res, src_conn="_res", memlet=Memlet(f"res[0:m, 0:n/{veclen}]"))

    return ger_node, state, sdfg


def run_ger(target: str,
            n: int,
            m: int,
            tile_size_x: int,
            tile_size_y: int,
            alpha: float = 1,
            veclen: int = 1,
            eps: float = 1e-6):

    if target == "pure":
        ger_node, state, sdfg = pure_graph("pure", dace.float32, veclen)
        ger_node.expand(state)
        sdfg.apply_transformations_repeated([InlineSDFG])
    else:
        raise ValueError("Unsupported target")

    x = aligned_ndarray(np.random.rand(m).astype(np.float32), alignment=4 * veclen)
    y = aligned_ndarray(np.random.rand(n).astype(np.float32), alignment=4 * veclen)
    A = aligned_ndarray(np.random.rand(m, n).astype(np.float32), alignment=4 * veclen)
    res = aligned_ndarray(np.empty(A.shape, dtype=A.dtype), alignment=4 * veclen)
    ref = aligned_ndarray(np.empty(A.shape, dtype=A.dtype), alignment=4 * veclen)
    res[:] = A[:]
    ref[:] = A[:]

    with dace.config.set_temporary('compiler', 'allow_view_arguments', value=True):
        sdfg(x=x, y=y, A=A, res=res, m=dace.int32(m), n=dace.int32(n), alpha=alpha)

    ref = scipy.linalg.blas.sger(alpha=alpha, x=x, y=y, a=ref)

    diff = np.linalg.norm(res - ref)
    if diff >= eps * n * m:
        raise RuntimeError(f"Validation failed: {diff}")
    else:
        print("Validation successful.")

    return sdfg


def test_ger_pure():
    run_ger("pure", 256, 512, 16, 32)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("N", type=int, nargs="?", default=256)
    parser.add_argument("M", type=int, nargs="?", default=512)
    parser.add_argument("tile_size_x", type=int, nargs="?", default=16)
    parser.add_argument("tile_size_y", type=int, nargs="?", default=32)
    parser.add_argument("alpha", type=np.float32, nargs="?", default=1.0)
    parser.add_argument("--target", dest="target", default="pure")
    parser.add_argument("--veclen", type=int, default=8)
    parser.add_argument("--eps", type=float, default=1e-6)
    args = parser.parse_args()

    run_ger(args.target, args.N, args.M, args.tile_size_x, args.tile_size_y, args.alpha, args.veclen, args.eps)
