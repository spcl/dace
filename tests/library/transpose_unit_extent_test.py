# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``Transpose`` over an operand with a unit extent, on every expansion that can run here.

A ``(1, 1)`` transpose is a one-element copy, and the BLAS expansions take ``_inp`` / ``_out`` as
buffers while the codegen types a single-element subset as a SCALAR -- so ``cblas_domatcopy`` was
handed a value where it declared a pointer and the BUILD failed. The frontend used to route every
unit extent around the node with a hand-written map; these pin that the node itself handles them,
which is what let that detour go.
"""
import numpy as np
import pytest

import dace
from dace.libraries.blas import environments as blas_environments
from dace.libraries.linalg import Transpose

#: Every 2D shape with a unit extent, plus one genuine matrix as the control.
SHAPES = [(1, 1), (5, 1), (1, 5), (4, 3)]


def build(shape, implementation):
    sdfg = dace.SDFG(f"transpose_{implementation}_{shape[0]}x{shape[1]}")
    sdfg.add_array("a", list(shape), dace.float64)
    sdfg.add_array("out", [shape[1], shape[0]], dace.float64)
    state = sdfg.add_state()
    node = Transpose("t", dace.float64)
    node.implementation = implementation
    state.add_node(node)
    state.add_edge(state.add_read("a"), None, node, "_inp", dace.Memlet.from_array("a", sdfg.arrays["a"]))
    state.add_edge(node, "_out", state.add_write("out"), None, dace.Memlet.from_array("out", sdfg.arrays["out"]))
    return sdfg


def run(shape, implementation):
    sdfg = build(shape, implementation)
    sdfg.expand_library_nodes()
    sdfg.validate()
    a = np.arange(shape[0] * shape[1], dtype=np.float64).reshape(shape)
    out = np.zeros((shape[1], shape[0]), dtype=np.float64)
    sdfg(a=a.copy(), out=out)
    # A transpose moves elements, it does not compute: anything but bitwise equality is a defect.
    assert np.array_equal(out, a.T), f"{implementation} {shape}: {out} != {a.T}"


@pytest.mark.parametrize("shape", SHAPES)
def test_transpose_pure_handles_a_unit_extent(shape):
    run(shape, "pure")


@pytest.mark.parametrize("shape", SHAPES)
def test_transpose_openblas_handles_a_unit_extent(shape):
    if not blas_environments.openblas.OpenBLAS.is_installed():
        pytest.skip("OpenBLAS is not installed")
    run(shape, "OpenBLAS")


@pytest.mark.parametrize("implementation", ["MKL", "OpenBLAS", "cuBLAS"])
def test_a_one_element_operand_never_reaches_a_blas_call(implementation):
    """The degenerate case must be intercepted BEFORE the library call is emitted, on every BLAS
    expansion -- including the ones this machine cannot build. Expanding to an SDFG rather than a
    CPP tasklet is what says the interception happened."""
    from dace.sdfg.nodes import NestedSDFG
    sdfg = build((1, 1), implementation)
    sdfg.expand_library_nodes()
    assert any(
        isinstance(n, NestedSDFG)
        for n, _ in sdfg.all_nodes_recursive()), (f"{implementation} emitted a library call for a one-element operand")


@pytest.mark.parametrize("implementation", ["MKL", "OpenBLAS", "cuBLAS"])
def test_a_genuine_matrix_still_reaches_the_blas_call(implementation):
    """...and the interception must not swallow the matrices the node exists for."""
    from dace.sdfg.nodes import Tasklet
    sdfg = build((4, 3), implementation)
    sdfg.expand_library_nodes()
    assert any(
        isinstance(n, Tasklet) and n.language == dace.dtypes.Language.CPP
        for n, _ in sdfg.all_nodes_recursive()), (f"{implementation} lost its library call")


if __name__ == '__main__':
    for s in SHAPES:
        test_transpose_pure_handles_a_unit_extent(s)
    for impl in ("MKL", "OpenBLAS", "cuBLAS"):
        test_a_one_element_operand_never_reaches_a_blas_call(impl)
        test_a_genuine_matrix_still_reaches_the_blas_call(impl)
