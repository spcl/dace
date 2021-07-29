# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.

from dace.sdfg import SDFG, SDFGState
from dace import Memlet
import dace
from dace.libraries import blas
from dace.transformation.dataflow import HbmTransform
from dace.transformation.interstate import InlineSDFG

def create_axpy_sdfg():
    sdfg: SDFG = SDFG("axpy")
    state = sdfg.add_state("axpy_state")
    n = dace.symbol("n")

    x = sdfg.add_array("x", [n], dace.float32)
    y = sdfg.add_array("y", [n], dace.float32)

    x_in = state.add_read("x")
    y_in = state.add_read("y")
    y_out = state.add_write("y")

    axpy_node = blas.axpy.Axpy("axpy", 1)
    axpy_node.implementation = "fpga"

    state.add_memlet_path(x_in,
                               axpy_node,
                               dst_conn="_x",
                               memlet=Memlet(f"x[0:n]"))
    state.add_memlet_path(y_in,
                               axpy_node,
                               dst_conn="_y",
                               memlet=Memlet(f"y[0:n]"))
    state.add_memlet_path(axpy_node,
                               y_out,
                               src_conn="_res",
                               memlet=Memlet(f"y[0:n]"))
    sdfg.expand_library_nodes()
    sdfg.apply_transformations(InlineSDFG)
    
    return sdfg

def test_direct_axpy():
    sdfg = create_axpy_sdfg()

    sdfg.apply_transformations(HbmTransform)

    sdfg.view()


test_direct_axpy()