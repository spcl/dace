# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Scal, Ger, Symv, Symm and Syrk against numpy, in C and Fortran layout. The triangle of a symmetric operand that
the node must not read holds junk."""

import numpy as np
import pytest

import dace
from dace.libraries.blas.nodes import Ger, Scal, Symm, Symv, Syrk

IMPLEMENTATIONS = ["pure", pytest.param("OpenBLAS", marks=pytest.mark.lapack)]
rng = np.random.default_rng(42)


def array(sdfg: dace.SDFG, name: str, value: np.ndarray):
    """Adds ``value``'s container to ``sdfg`` with ``value``'s layout."""
    strides = [s // value.itemsize for s in value.strides]
    sdfg.add_array(name, list(value.shape), dace.float64, strides=strides)


def run(
    name: str,
    node,
    implementation: str,
    inputs: dict[str, tuple[str, np.ndarray]],
    outputs: dict[str, tuple[str, np.ndarray]],
):
    """Connects ``node`` to one array per connector and runs it; returns the outputs."""
    sdfg = dace.SDFG(f"{name}_{implementation}")
    state = sdfg.add_state()
    node.implementation = implementation
    for connector, (array_name, value) in {**inputs, **outputs}.items():
        array(sdfg, array_name, value)
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
    sdfg(**dict({**inputs, **outputs}.values()))


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
    out = np.zeros((m, n), order=order)
    node = Symm("symm", side=right, uplo=upper, alpha=2.0, beta=0.5)
    run(
        f"symm_{order}_{right}_{upper}",
        node,
        implementation,
        {"_A": ("A", a), "_B": ("B", b), "_Cin": ("C", c)},
        {"_Cout": ("out", out)},
    )
    s = symmetric_from(a, upper)
    np.testing.assert_allclose(out, 2.0 * (b @ s if right else s @ b) + 0.5 * c)


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("transposed,upper", [(False, False), (True, True)])
def test_syrk(implementation, order, transposed, upper):
    n, k = 5, 3
    a, c = matrix((k, n) if transposed else (n, k), order), matrix((n, n), order)
    out = np.zeros((n, n), order=order)
    node = Syrk("syrk", uplo=upper, transA=transposed, alpha=2.0, beta=0.5)
    run(
        f"syrk_{order}_{transposed}_{upper}",
        node,
        implementation,
        {"_A": ("A", a), "_Cin": ("C", c)},
        {"_Cout": ("out", out)},
    )
    full = 2.0 * (a.T @ a if transposed else a @ a.T) + 0.5 * c
    referenced = np.triu(np.ones((n, n), bool)) if upper else np.tril(np.ones((n, n), bool))
    np.testing.assert_allclose(out, np.where(referenced, full, c))


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
