# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A scale that is no real literal is data: it reaches a BLAS node through ``_alpha`` or ``_beta``."""

import numpy as np
import pytest

import dace
from dace.libraries.blas.blas_helpers import connect_scale
from dace.libraries.blas.nodes import Axpy, Gemv, Ger, Scal, Symm, Symv, Syrk

IMPLEMENTATIONS = ["pure", pytest.param("OpenBLAS", marks=pytest.mark.lapack)]
ALPHA, BETA = 2.0 + 1.0j, 0.5 - 2.0j
Z = dace.complex128
rng = np.random.default_rng(7)


def complex_matrix(shape) -> np.ndarray:
    return rng.standard_normal(shape) + 1j * rng.standard_normal(shape)


def scaled_sdfg(name: str, node, inputs: dict, outputs: dict, implementation: str, **values) -> dace.SDFG:
    """``node`` over one complex array per entry of ``inputs`` and ``outputs``, each scale of ``values`` connected."""
    sdfg = dace.SDFG(f"{name}_{implementation}")
    state = sdfg.add_state()
    for array_name, value in {**inputs, **outputs}.values():
        if array_name not in sdfg.arrays:
            sdfg.add_array(array_name, list(value.shape), Z)
    node.implementation = implementation
    state.add_node(node)
    for connector, (array_name, _) in inputs.items():
        state.add_edge(state.add_read(array_name), None, node, connector, dace.Memlet(array_name))
    for connector, (array_name, _) in outputs.items():
        state.add_edge(node, connector, state.add_write(array_name), None, dace.Memlet(array_name))
    for name, value in values.items():
        attribute = "a" if isinstance(node, (Scal, Axpy)) else name
        setattr(node, attribute, connect_scale(sdfg, state, node, name, value))
    return sdfg


def run_case(name: str, implementation: str, value) -> tuple:
    """The result of the case ``name`` scaled by ``value``, and its numpy reference for ``ALPHA`` and ``BETA``."""
    x, y = complex_matrix(4), complex_matrix(4)
    a = complex_matrix((4, 4))
    symmetric = a + a.T
    b = complex_matrix((4, 3))
    res = np.zeros(4, dtype=np.complex128)
    if name == "scal":
        sdfg = scaled_sdfg("scal", Scal("scal"), {"_x": ("x", x)}, {"_res": ("res", res)}, implementation, alpha=value)
        sdfg(x=x, res=res)
        return res, value * x
    if name == "axpy":
        node = Axpy("axpy", n=4)
        connectors = {"_x": ("x", x), "_y": ("y", y)}
        sdfg = scaled_sdfg("axpy", node, connectors, {"_res": ("res", res)}, implementation, alpha=value)
        sdfg(x=x, y=y, res=res)
        return res, value * x + y
    if name == "ger":
        out = np.zeros((4, 3), dtype=np.complex128)
        w = complex_matrix(3)
        connectors = {"_A": ("b", b), "_x": ("x", x), "_y": ("w", w)}
        sdfg = scaled_sdfg("ger", Ger("ger"), connectors, {"_res": ("out", out)}, implementation, alpha=value)
        sdfg(b=b, x=x, w=w, out=out)
        return out, b + value * np.outer(x, w)
    if name == "symv":
        connectors = {"_A": ("a", symmetric), "_x": ("x", x), "_yin": ("y", y)}
        node = Symv("symv")
        sdfg = scaled_sdfg("symv", node, connectors, {"_yout": ("res", res)}, implementation, alpha=value, beta=BETA)
        sdfg(a=symmetric, x=x, y=y, res=res)
        return res, value * symmetric @ x + BETA * y
    if name == "symm":
        c = complex_matrix((4, 3))
        out = np.zeros((4, 3), dtype=np.complex128)
        connectors = {"_A": ("a", symmetric), "_B": ("b", b), "_Cin": ("c", c)}
        node = Symm("symm")
        sdfg = scaled_sdfg("symm", node, connectors, {"_Cout": ("out", out)}, implementation, alpha=value, beta=BETA)
        sdfg(a=symmetric, b=b, c=c, out=out)
        return out, value * symmetric @ b + BETA * c
    c = symmetric
    out = np.zeros((4, 4), dtype=np.complex128)
    connectors = {"_A": ("b", b), "_Cin": ("c", c)}
    node = Syrk("syrk")
    sdfg = scaled_sdfg("syrk", node, connectors, {"_Cout": ("out", out)}, implementation, alpha=value, beta=BETA)
    sdfg(b=b, c=c, out=out)
    return np.tril(out), np.tril(value * b @ b.T + BETA * c)


CASES = ("scal", "axpy", "ger", "symv", "symm", "syrk")


# Axpy has only the pure expansion
@pytest.mark.parametrize(
    "name, implementation",
    [(name, "pure") for name in CASES]
    + [pytest.param(name, "OpenBLAS", marks=pytest.mark.lapack) for name in CASES if name != "axpy"],
)
def test_a_complex_scale_reaches_the_expansion(name, implementation):
    """``1j`` on a symbolic property turns into a symbol named ``I`` that nothing defines."""
    result, reference = run_case(name, implementation, ALPHA)
    np.testing.assert_allclose(result, reference)


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
def test_gemv_reads_wired_scales(implementation):
    a, x, y = complex_matrix((3, 4)), complex_matrix(4), complex_matrix(3)
    expected = ALPHA * a @ x + BETA * y
    node = Gemv("gemv", beta=1)
    connectors = {"_A": ("a", a), "_x": ("x", x), "_y": ("y", y)}
    sdfg = scaled_sdfg("gemv_wired", node, connectors, {"_y": ("y", y)}, implementation, alpha=ALPHA, beta=BETA)
    sdfg(a=a, x=x, y=y)
    np.testing.assert_allclose(y, expected)


def test_a_complex_scale_on_a_symbolic_property_is_refused():
    with pytest.raises(ValueError, match="connect_scale"):
        Scal("scal", a=1j)


def symbolic_scale_sdfg(name: str) -> dace.SDFG:
    """A scale assigned to a symbol on an inter-state edge, wired to a Scal in the next state."""
    sdfg = dace.SDFG(name)
    sdfg.add_array("x", [4], dace.float64)
    sdfg.add_array("res", [4], dace.float64)
    sdfg.add_symbol("s", dace.float64)
    first = sdfg.add_state("first", is_start_block=True)
    state = sdfg.add_state_after(first, "call", assignments={"s": "2.5"})
    node = Scal("scal")
    node.implementation = "OpenBLAS"
    state.add_node(node)
    node.a = connect_scale(sdfg, state, node, "alpha", dace.symbol("s", dace.float64))
    state.add_edge(state.add_read("x"), None, node, "_x", dace.Memlet("x"))
    state.add_edge(node, "_res", state.add_write("res"), None, dace.Memlet("res"))
    return sdfg


@pytest.mark.lapack
def test_simplify_keeps_a_wired_symbol():
    """As data the scale is a use of the symbol, so its assignment survives."""
    sdfg = symbolic_scale_sdfg("simplified_scale")
    sdfg.simplify()
    x, res = np.arange(1.0, 5.0), np.zeros(4)
    sdfg(x=x, res=res)
    np.testing.assert_allclose(res, 2.5 * x)


@pytest.mark.lapack
def test_replace_reaches_a_wired_symbol():
    sdfg = symbolic_scale_sdfg("replaced_scale")
    sdfg.replace("s", "t")
    tasklet = next(node for node, _ in sdfg.all_nodes_recursive() if isinstance(node, dace.nodes.Tasklet))
    assert tasklet.code.as_string.strip() == "__out = t"
    x, res = np.arange(1.0, 5.0), np.zeros(4)
    sdfg(x=x, res=res)
    np.testing.assert_allclose(res, 2.5 * x)


def test_a_wired_scale_survives_a_json_round_trip():
    sdfg = dace.SDFG("json_scale")
    state = sdfg.add_state()
    sdfg.add_array("x", [4], Z)
    sdfg.add_array("res", [4], Z)
    node = Scal("scal")
    node.implementation = "pure"
    state.add_node(node)
    node.a = connect_scale(sdfg, state, node, "alpha", ALPHA)
    state.add_edge(state.add_read("x"), None, node, "_x", dace.Memlet("x"))
    state.add_edge(node, "_res", state.add_write("res"), None, dace.Memlet("res"))

    loaded = dace.SDFG.from_json(sdfg.to_json())
    loaded.name = "json_scale_loaded"
    x, res = complex_matrix(4), np.zeros(4, dtype=np.complex128)
    loaded(x=x, res=res)

    np.testing.assert_allclose(res, ALPHA * x)


if __name__ == "__main__":
    for implementation in ("pure", "OpenBLAS"):
        for name in CASES:
            if name != "axpy" or implementation == "pure":
                test_a_complex_scale_reaches_the_expansion(name, implementation)
        test_gemv_reads_wired_scales(implementation)
    test_a_complex_scale_on_a_symbolic_property_is_refused()
    test_simplify_keeps_a_wired_symbol()
    test_replace_reaches_a_wired_symbol()
    test_a_wired_scale_survives_a_json_round_trip()
