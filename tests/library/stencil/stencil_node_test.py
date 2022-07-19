# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
from dace.fpga_testing import intel_fpga_test
from dace.libraries.stencil import Stencil
import numpy as np
from dace.transformation.interstate import FPGATransformSDFG, InlineSDFG

SIZE = dace.symbol("size")
ROWS = dace.symbol("rows")
COLS = dace.symbol("cols")
DTYPE = np.float32


def make_sdfg_1d(implementation: str, vector_length: int):

    vtype = dace.vector(dace.typeclass(DTYPE), vector_length) if vector_length > 1 else DTYPE

    sdfg = dace.SDFG(f"stencil_node_test_1d_w{vector_length}")
    _, a_desc = sdfg.add_array("a", (SIZE / vector_length, ), dtype=vtype)
    _, res_desc = sdfg.add_array("res", (SIZE / vector_length, ), dtype=vtype)

    state = sdfg.add_state("stencil_node_test_1d")
    a = state.add_read("a")
    res = state.add_write("res")

    stencil_node = Stencil("stencil_test",
                           """\
tmp0 = (a[0] + a[1])
tmp1 = (tmp0 + a[2])
res[1] = (dace.float32(0.3333) * tmp1)""",
                           inputs={"a"},
                           outputs={"res"})
    stencil_node.implementation = implementation
    state.add_node(stencil_node)

    state.add_edge(a, None, stencil_node, "a", dace.Memlet.from_array("a", a_desc))
    state.add_edge(stencil_node, "res", res, None, dace.Memlet.from_array("res", res_desc))

    return sdfg


def make_sdfg_1d_with_scalar(implementation: str):

    sdfg = dace.SDFG("stencil_node_test_1d_with_scalar")
    _, a_desc = sdfg.add_array("a", (SIZE, ), dtype=DTYPE)
    _, scal_desc = sdfg.add_scalar("scal", dtype=DTYPE)
    _, res_desc = sdfg.add_array("res", (SIZE, ), dtype=DTYPE)

    state = sdfg.add_state("stencil_node_test_1d")
    a = state.add_read("a")
    scal = state.add_read("scal")
    res = state.add_write("res")

    stencil_node = Stencil("stencil_test",
                           """\
tmp0 = (a[0] + a[1])
tmp1 = (tmp0 + a[2])
res[1] = (scal * tmp1)""",
                           inputs={"a", "scal"},
                           outputs={"res"})
    stencil_node.implementation = implementation
    state.add_node(stencil_node)

    state.add_edge(a, None, stencil_node, "a",
                   dace.Memlet.from_array("a", a_desc))
    state.add_edge(scal, None, stencil_node, "scal",
                   dace.Memlet.from_array("scal", scal_desc))
    state.add_edge(stencil_node, "res", res, None,
                   dace.Memlet.from_array("res", res_desc))

    return sdfg


def make_sdfg_2d(implementation: str, vector_length: int):

    vtype = dace.vector(dace.typeclass(DTYPE), vector_length) if vector_length > 1 else DTYPE

    sdfg = dace.SDFG(f"stencil_node_test_2d_w{vector_length}")
    _, a_desc = sdfg.add_array("a", (ROWS, COLS / vector_length), dtype=vtype)
    _, b_desc = sdfg.add_array("b", (ROWS, ), dtype=DTYPE)
    sdfg.add_symbol("c", DTYPE)
    _, d_desc = sdfg.add_array("d", (ROWS, COLS / vector_length), dtype=vtype)
    _, res_desc = sdfg.add_array("res", (ROWS, COLS / vector_length), dtype=vtype)

    state = sdfg.add_state("stencil_node_test_2d")
    a = state.add_read("a")
    b = state.add_read("b")
    d = state.add_read("d")
    res = state.add_write("res")

    stencil_node = Stencil("stencil_test",
                           "res[0, 0] = c * b[0] * (a[-1, 0] + a[1, 0] + a[0, -1] + a[0, 1]) + d[0, -1] + d[0, 1]",
                           iterator_mapping={"b": (True, False)})
    stencil_node.implementation = implementation
    state.add_node(stencil_node)

    state.add_memlet_path(a, stencil_node, dst_conn="a", memlet=dace.Memlet.from_array("a", a_desc))
    state.add_memlet_path(b, stencil_node, dst_conn="b", memlet=dace.Memlet.from_array("b", b_desc))
    state.add_memlet_path(d, stencil_node, dst_conn="d", memlet=dace.Memlet.from_array("d", d_desc))
    state.add_memlet_path(stencil_node, res, src_conn="res", memlet=dace.Memlet.from_array("res", res_desc))

    return sdfg


def run_stencil_1d(sdfg, size):
    a = np.zeros((size, ), dtype=DTYPE)
    a[1:-1] = np.arange(1, size - 1, dtype=DTYPE).reshape((size - 2))
    res = np.zeros((size, ), dtype=DTYPE)
    sdfg.expand_library_nodes()
    sdfg(a=a, res=res, size=size)
    expected = 0.3333 * (a[:-2] + a[1:-1] + a[2:])
    assert np.allclose(expected, res[1:-1])


def test_stencil_node_1d():
    run_stencil_1d(make_sdfg_1d("pure", 1), 32)


def stencil_node_1d_fpga_array(vector_length: int):
    sdfg = make_sdfg_1d(dace.Config.get("compiler", "fpga", "vendor"), vector_length)
    assert sdfg.apply_transformations(FPGATransformSDFG) == 1
    assert sdfg.apply_transformations(InlineSDFG) == 1
    run_stencil_1d(sdfg, 32)
    return sdfg


def run_stencil_1d_with_scalar(sdfg, size):
    a = np.zeros((size, ), dtype=DTYPE)
    a[1:-1] = np.arange(1, size - 1, dtype=DTYPE).reshape((size - 2))
    scal = DTYPE(0.3333)
    res = np.zeros((size, ), dtype=DTYPE)
    sdfg(a=a, scal=scal, res=res, size=size)
    expected = 0.3333 * (a[:-2] + a[1:-1] + a[2:])
    assert np.allclose(expected, res[1:-1])


def test_stencil_node_1d_with_scalar():
    run_stencil_1d_with_scalar(make_sdfg_1d_with_scalar("pure"), 32)


@intel_fpga_test()
def test_stencil_node_1d_with_scalar_fpga_array():
    sdfg = make_sdfg_1d_with_scalar(dace.Config.get("compiler", "fpga_vendor"))
    assert sdfg.apply_transformations(FPGATransformSDFG) == 1
    assert sdfg.apply_transformations(InlineSDFG) == 1
    run_stencil_1d_with_scalar(sdfg, 32)
    return sdfg

@intel_fpga_test()
def test_stencil_node_1d_fpga_array():
    return stencil_node_1d_fpga_array(1)


def run_stencil_2d(sdfg, rows, cols, specialize: bool):
    a = np.zeros((rows, cols), dtype=DTYPE)
    a[1:-1, 1:-1] = np.arange(1, (rows - 2) * (cols - 2) + 1, dtype=DTYPE).reshape((rows - 2, cols - 2))
    b = np.empty((rows, ), dtype=DTYPE)
    b[:] = 1
    c = DTYPE(0.25)
    d = 0.5 * np.ones((rows, cols), dtype=DTYPE)
    res = np.zeros((rows, cols), dtype=DTYPE)
    if specialize:
        sdfg(a=a, b=b, c=c, d=d, res=res, rows=rows)
    else:
        sdfg(a=a, b=b, c=c, d=d, res=res, rows=rows, cols=cols)
    expected = (0.25 * (a[2:, 1:-1] + a[:-2, 1:-1] + a[1:-1, 2:] + a[1:-1, :-2]) + 1)
    assert np.allclose(expected, res[1:-1, 1:-1])


def test_stencil_node_2d():
    run_stencil_2d(make_sdfg_2d("pure", 1), 16, 32, False)


def stencil_node_2d_fpga_array(vector_length: int):
    sdfg = make_sdfg_2d(dace.Config.get("compiler", "fpga", "vendor"), vector_length)
    sdfg.specialize({"cols": 8})
    assert sdfg.apply_transformations(FPGATransformSDFG) == 1
    assert sdfg.apply_transformations(InlineSDFG) == 1
    run_stencil_2d(sdfg, 4, 8, True)
    return sdfg


@intel_fpga_test()
def test_stencil_node_2d_fpga_array():
    return stencil_node_2d_fpga_array(1)


@intel_fpga_test()
def test_stencil_node_2d_fpga_array_vectorized():
    return stencil_node_2d_fpga_array(4)


if __name__ == "__main__":
    test_stencil_node_1d()
    test_stencil_node_1d_fpga_array(None)
    test_stencil_node_1d_with_scalar()
    test_stencil_node_1d_with_scalar_fpga_array(None)
    test_stencil_node_2d()
    test_stencil_node_2d_fpga_array(None)
    test_stencil_node_2d_fpga_array_vectorized(None)
