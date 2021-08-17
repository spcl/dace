# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
from dace.fpga_testing import intel_fpga_test
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
    sdfg.add_symbol("c", DTYPE)
    _, d_desc = sdfg.add_array("d", (ROWS, COLS), dtype=DTYPE)
    _, res_desc = sdfg.add_array("res", (ROWS, COLS), dtype=DTYPE)

    state = sdfg.add_state("stencil_node_test")
    a = state.add_read("a")
    b = state.add_read("b")
    d = state.add_read("d")
    res = state.add_write("res")

    stencil_node = Stencil(
        "stencil_test",
        "res[0, 0] = c * b[0] * (a[-1, 0] + a[1, 0] + a[0, -1] + a[0, 1]) + d[0, -1] + d[0, 1]",
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
    state.add_memlet_path(d,
                          stencil_node,
                          dst_conn="d",
                          memlet=dace.Memlet.from_array("d", d_desc))
    state.add_memlet_path(stencil_node,
                          res,
                          src_conn="res",
                          memlet=dace.Memlet.from_array("res", res_desc))

    return sdfg


def run_stencil(sdfg, specialize: bool):
    rows = 16
    cols = 32
    a = np.zeros((rows, cols), dtype=DTYPE)
    a[1:-1, 1:-1] = np.arange(1, (rows - 2) * (cols - 2) + 1,
                              dtype=DTYPE).reshape((rows - 2, cols - 2))
    b = np.empty((rows, ), dtype=DTYPE)
    b[:] = 1
    c = DTYPE(0.25)
    d = 0.5 * np.ones((rows, cols), dtype=DTYPE)
    res = np.zeros((rows, cols), dtype=DTYPE)
    print("RUNNING SDFG")
    if specialize:
        sdfg.specialize({"cols": cols})
        sdfg(a=a, b=b, c=c, d=d, res=res, rows=rows)
    else:
        sdfg(a=a, b=b, c=c, d=d, res=res, rows=rows, cols=cols)
    print("FINISHED RUNNING SDFG")
    expected = (0.25 *
                (a[2:, 1:-1] + a[:-2, 1:-1] + a[1:-1, 2:] + a[1:-1, :-2]) + 1)
    assert np.allclose(expected, res[1:-1, 1:-1])


def test_stencil_node():
    run_stencil(make_sdfg(), False)


@intel_fpga_test()
def test_stencil_node_fpga():
    sdfg = make_sdfg(dace.Config.get("compiler", "fpga_vendor"))
    assert sdfg.apply_transformations(FPGATransformSDFG) == 1
    assert sdfg.apply_transformations(InlineSDFG) == 1
    run_stencil(sdfg, True)
    return sdfg


if __name__ == "__main__":
    test_stencil_node()
    test_stencil_node_fpga(None)
