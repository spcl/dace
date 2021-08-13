# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
from dace.fpga_testing import fpga_test
from dace.libraries.stencil import Stencil
import numpy as np
from dace.transformation.interstate import FPGATransformSDFG, InlineSDFG

ROWS = dace.symbol("rows")
COLS = dace.symbol("cols")
DTYPE = np.float32


def make_sdfg(implementation="pure"):

    sdfg = dace.SDFG("stencil_node_test")
    _, a_desc = sdfg.add_array("a", (ROWS, COLS), dtype=DTYPE)
    _, b_desc = sdfg.add_array("b", (ROWS, ), dtype=DTYPE)
    _, res_desc = sdfg.add_array("res", (ROWS, COLS), dtype=DTYPE)
    sdfg.add_symbol("c", DTYPE)

    state = sdfg.add_state("stencil_node_test")
    a = state.add_read("a")
    b = state.add_read("b")
    res = state.add_write("res")

    stencil_node = Stencil(
        "stencil_test",
        "res[0, 0] = c * b[0] * (a[-1, 0] + a[1, 0] + a[0, -1] + a[0, 1])",
        iterator_mapping={"b": (True, False)})
    stencil_node.implementation = implementation
    state.add_node(stencil_node)

    state.add_memlet_path(a,
                          stencil_node,
                          dst_conn="a",
                          memlet=dace.Memlet.from_array("a", a_desc))
    state.add_memlet_path(b,
                          stencil_node,
                          dst_conn="b",
                          memlet=dace.Memlet.from_array("b", b_desc))
    state.add_memlet_path(stencil_node,
                          res,
                          src_conn="res",
                          memlet=dace.Memlet.from_array("res", res_desc))

    return sdfg


def run_stencil(sdfg, specialize: bool):
    rows = 8
    cols = 4
    a = np.ones((rows, cols), dtype=DTYPE)
    a[1:-1, 1:-1] = 0
    b = np.empty((rows, ), dtype=DTYPE)
    b[:] = 1
    c = DTYPE(0.25)
    res = np.zeros((rows, cols), dtype=DTYPE)
    if specialize:
        sdfg.specialize({"cols": cols})
        sdfg(a=a, b=b, c=c, res=res, rows=rows)
    else:
        sdfg(a=a, b=b, c=c, res=res, rows=rows, cols=cols)
    expected = 0.25 * (a[2:, 1:-1] + a[:-2, 1:-1] + a[1:-1, 2:] + a[1:-1, :-2])
    print(expected)
    print(res)
    assert np.allclose(expected, res[1:-1, 1:-1])


def test_stencil_node():
    run_stencil(make_sdfg(), False)


def test_stencil_node_fpga():
    sdfg = make_sdfg(dace.Config.get("compiler", "fpga_vendor"))
    assert sdfg.apply_transformations(FPGATransformSDFG) == 1
    assert sdfg.apply_transformations(InlineSDFG) == 1
    run_stencil(sdfg, True)
    return sdfg


if __name__ == "__main__":
    test_stencil_node()
    test_stencil_node_fpga()
