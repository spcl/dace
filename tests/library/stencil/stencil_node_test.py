# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
from dace.libraries.stencil import Stencil
import numpy as np

ROWS = dace.symbol("rows")
COLS = dace.symbol("cols")
DTYPE = np.float32


def make_sdfg():

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


def test_stencil_node():
    sdfg = make_sdfg()
    rows = 16
    cols = 32
    a = np.ones((rows, cols), dtype=DTYPE)
    a[1:-1, 1:-1] = 0
    b = np.empty((rows, ), dtype=DTYPE)
    b[:] = 1
    c = DTYPE(0.25)
    res = np.empty((rows, cols), dtype=DTYPE)
    sdfg(a=a, b=b, c=c, res=res, rows=rows, cols=cols)
    assert np.allclose(
        0.25 * (a[2:, 1:-1] + a[:-2, 1:-1] + a[1:-1, 2:] + a[1:-1, :-2]),
        res[1:-1, 1:-1])


if __name__ == "__main__":
    test_stencil_node()
