# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Scal, Ger, Symv, Symm, Syrk and Syr2k against numpy, in C and Fortran layout, on the CPU and through the GPU
vendor library. The triangle of a symmetric operand that the node must not read holds junk."""

import numpy as np
import pytest

import dace
from dace.codegen.common import get_gpu_backend
from dace.libraries.blas.nodes import Ger, Scal, Symm, Symv, Syr2k, Syrk

#: ``gpu`` stands for the vendor library of this build's GPU backend, resolved only when a GPU test runs.
IMPLEMENTATIONS = [
    "pure",
    pytest.param("OpenBLAS", marks=pytest.mark.lapack),
    pytest.param("gpu", marks=pytest.mark.gpu),
]
rng = np.random.default_rng(42)


def array(sdfg: dace.SDFG, name: str, value: np.ndarray, storage: dace.StorageType):
    """Adds ``value``'s container to ``sdfg`` with ``value``'s layout."""
    strides = [s // value.itemsize for s in value.strides]
    sdfg.add_array(name, list(value.shape), dace.float64, storage, strides=strides)


def run(
    name: str,
    node,
    implementation: str,
    inputs: dict[str, tuple[str, np.ndarray]],
    outputs: dict[str, tuple[str, np.ndarray]],
):
    """Connects ``node`` to one array per connector and runs it, in GPU memory for ``gpu``; writes the results back
    into the ``outputs`` arrays."""
    gpu = implementation == "gpu"
    if gpu:
        cupy = pytest.importorskip("cupy")
        implementation = "rocBLAS" if get_gpu_backend() == "hip" else "cuBLAS"
    sdfg = dace.SDFG(f"{name}_{implementation}")
    state = sdfg.add_state()
    node.implementation = implementation
    # A list, not a merged dict: an in-place node reads and writes one connector name
    operands = dict([*inputs.values(), *outputs.values()])
    storage = dace.StorageType.GPU_Global if gpu else dace.StorageType.Default
    for array_name, value in operands.items():
        array(sdfg, array_name, value, storage)
    for connector, (array_name, _) in inputs.items():
        state.add_edge(
            state.add_read(array_name),
            None,
            node,
            connector,
            dace.Memlet.from_array(array_name, sdfg.arrays[array_name]),
        )
    for connector, (array_name, _) in outputs.items():
        state.add_edge(
            node,
            connector,
            state.add_write(array_name),
            None,
            dace.Memlet.from_array(array_name, sdfg.arrays[array_name]),
        )
    if not gpu:
        sdfg(**operands)
        return
    device = {
        array_name: cupy.asarray(value, order="F" if value.ndim > 1 and value.flags.f_contiguous else "C")
        for array_name, value in operands.items()
    }
    sdfg(**device)
    for array_name, value in outputs.values():
        value[...] = device[array_name].get()


def matrix(shape, order: str) -> np.ndarray:
    return np.asarray(rng.standard_normal(shape), order=order)


def symmetric_from(a: np.ndarray, upper: bool) -> np.ndarray:
    """The symmetric matrix of which ``a`` holds the upper or the lower triangle."""
    triangle = np.triu(a) if upper else np.tril(a)
    return triangle + triangle.T - np.diag(np.diag(a))


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
def test_scal(implementation):
    x, res = rng.standard_normal(7), np.zeros(7)
    run("scal", Scal("scal", a=2.5), implementation, {"_x": ("x", x)}, {"_res": ("res", res)})
    np.testing.assert_allclose(res, 2.5 * x)


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
@pytest.mark.parametrize("order", ["C", "F"])
def test_ger(implementation, order):
    a, x, y = matrix((5, 4), order), rng.standard_normal(5), rng.standard_normal(4)
    res = np.zeros((5, 4), order=order)
    run(
        f"ger_{order}",
        Ger("ger", m=5, n=4, alpha=1.5),
        implementation,
        {"_A": ("A", a), "_x": ("x", x), "_y": ("y", y)},
        {"_res": ("res", res)},
    )
    np.testing.assert_allclose(res, a + 1.5 * np.outer(x, y))


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("upper", [False, True])
def test_symv(implementation, order, upper):
    a, x, y = matrix((6, 6), order), rng.standard_normal(6), rng.standard_normal(6)
    out = np.zeros(6)
    node = Symv("symv", uplo=upper, alpha=2.0, beta=0.5)
    run(
        f"symv_{order}_{upper}",
        node,
        implementation,
        {"_A": ("A", a), "_x": ("x", x), "_yin": ("y", y)},
        {"_yout": ("out", out)},
    )
    np.testing.assert_allclose(out, 2.0 * symmetric_from(a, upper) @ x + 0.5 * y)


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("right,upper", [(False, False), (True, True)])
def test_symm(implementation, order, right, upper):
    m, n = 5, 4
    a, b, c = matrix((n, n) if right else (m, m), order), matrix((m, n), order), matrix((m, n), order)
    c_in = c.copy(order="K")
    node = Symm("symm", side="R" if right else "L", uplo="U" if upper else "L", alpha=2.0, beta=0.5)
    run(
        f"symm_{order}_{right}_{upper}",
        node,
        implementation,
        {"_a": ("A", a), "_b": ("B", b), "_c": ("C", c)},
        {"_c": ("C", c)},
    )
    s = symmetric_from(a, upper)
    np.testing.assert_allclose(c, 2.0 * (b @ s if right else s @ b) + 0.5 * c_in)


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("transposed,upper", [(False, False), (True, True)])
def test_syrk(implementation, order, transposed, upper):
    n, k = 5, 3
    a, c = matrix((k, n) if transposed else (n, k), order), matrix((n, n), order)
    c_in = c.copy(order="K")
    node = Syrk("syrk", uplo="U" if upper else "L", trans="T" if transposed else "N", alpha=2.0, beta=0.5)
    run(
        f"syrk_{order}_{transposed}_{upper}",
        node,
        implementation,
        {"_a": ("A", a), "_c": ("C", c)},
        {"_c": ("C", c)},
    )
    # In place: the triangle Syrk does not reference keeps its old value
    full = 2.0 * (a.T @ a if transposed else a @ a.T) + 0.5 * c_in
    referenced = np.triu(np.ones((n, n), bool)) if upper else np.tril(np.ones((n, n), bool))
    np.testing.assert_allclose(c, np.where(referenced, full, c_in))


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("transposed,upper", [(False, False), (True, True)])
def test_syr2k(implementation, order, transposed, upper):
    n, k = 5, 3
    shape = (k, n) if transposed else (n, k)
    a, b, c = matrix(shape, order), matrix(shape, order), matrix((n, n), order)
    c_in = c.copy(order="K")
    node = Syr2k("syr2k", uplo="U" if upper else "L", trans="T" if transposed else "N", alpha=2.0, beta=0.5)
    run(
        f"syr2k_{order}_{transposed}_{upper}",
        node,
        implementation,
        {"_a": ("A", a), "_b": ("B", b), "_c": ("C", c)},
        {"_c": ("C", c)},
    )
    full = 2.0 * (a.T @ b + b.T @ a if transposed else a @ b.T + b @ a.T) + 0.5 * c_in
    referenced = np.triu(np.ones((n, n), bool)) if upper else np.tril(np.ones((n, n), bool))
    np.testing.assert_allclose(c, np.where(referenced, full, c_in))


@pytest.mark.parametrize("implementation", ["cuBLAS", "rocBLAS"])
def test_gpu_library_rejects_a_matrix_without_a_contiguous_dimension(implementation):
    """The vendor libraries take row- or column-major matrices only; a matrix strided in both dimensions is an error
    at expansion, not a call that reads the wrong elements. Expanding needs no device."""
    sdfg = dace.SDFG(f"symm_strided_{implementation}")
    state = sdfg.add_state()
    gpu = dace.StorageType.GPU_Global
    sdfg.add_array("A", [4, 4], dace.float64, gpu, strides=[8, 2])
    sdfg.add_array("B", [4, 3], dace.float64, gpu)
    sdfg.add_array("C", [4, 3], dace.float64, gpu)
    node = Symm("symm")
    node.implementation = implementation
    for connector, name in (("_a", "A"), ("_b", "B")):
        state.add_edge(state.add_read(name), None, node, connector, dace.Memlet.from_array(name, sdfg.arrays[name]))
    state.add_edge(node, "_c", state.add_write("C"), None, dace.Memlet.from_array("C", sdfg.arrays["C"]))
    with pytest.raises(ValueError, match="row- or column-major"):
        sdfg.expand_library_nodes()


if __name__ == "__main__":
    for implementation in ("pure", "OpenBLAS"):
        test_scal(implementation)
        for order in ("C", "F"):
            test_ger(implementation, order)
            for upper in (False, True):
                test_symv(implementation, order, upper)
            for flag, upper in ((False, False), (True, True)):
                test_symm(implementation, order, flag, upper)
                test_syrk(implementation, order, flag, upper)
                test_syr2k(implementation, order, flag, upper)
    test_gpu_library_rejects_a_matrix_without_a_contiguous_dimension("cuBLAS")
    test_gpu_library_rejects_a_matrix_without_a_contiguous_dimension("rocBLAS")
