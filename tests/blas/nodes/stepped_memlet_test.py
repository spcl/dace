# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import numpy as np

import dace
from dace.libraries import blas
from dace.memlet import Memlet


def test_dot_walks_the_memlet_step():
    sdfg = dace.SDFG("dot_stepped")
    sdfg.add_array("x", [8], dace.float64)
    sdfg.add_array("y", [4], dace.float64)
    sdfg.add_array("r", [1], dace.float64)
    state = sdfg.add_state()
    node = blas.Dot("dot")
    state.add_node(node)
    state.add_edge(state.add_read("x"), None, node, "_x", Memlet("x[0:8:2]"))
    state.add_edge(state.add_read("y"), None, node, "_y", Memlet("y[0:4]"))
    state.add_edge(node, "_result", state.add_write("r"), None, Memlet("r[0]"))

    x, y, r = np.arange(1.0, 9.0), np.ones(4), np.zeros(1)
    sdfg(x=x, y=y, r=r)

    assert r[0] == 1.0 + 3.0 + 5.0 + 7.0


if __name__ == "__main__":
    test_dot_walks_the_memlet_step()
