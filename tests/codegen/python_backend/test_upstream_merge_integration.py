# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Execution checks for the extended-branch lowering interfaces."""

import numpy as np
import pytest

import dace
from dace.libraries.blas.nodes.dot import Dot
from dace.transformation.passes.vectorization import VectorizeCuTile

N = dace.symbol("N")


@dace.program
def sum_then_add(a: dace.float64[N], b: dace.float64[N]):
    acc = 0.0
    for i in range(N):
        acc += a[i]
    b[:] = a + acc


@dace.program
def min_then_add(a: dace.float64[N], b: dace.float64[N]):
    acc = 1.0e9
    for i in range(N):
        acc = min(acc, a[i])
    b[:] = a + acc


@dace.program
def max_then_add(a: dace.float64[N], b: dace.float64[N]):
    acc = -1.0e9
    for i in range(N):
        acc = max(acc, a[i])
    b[:] = a + acc


@pytest.mark.gpu
@pytest.mark.parametrize("size", [64, 70])
@pytest.mark.parametrize("program,reduction", [(sum_then_add, np.sum), (min_then_add, np.min), (max_then_add, np.max)])
def test_scalar_reduction_across_tiles(program, reduction, size):
    sdfg = program.to_sdfg(simplify=False)
    VectorizeCuTile(widths=(32,)).apply_pass(sdfg, {})
    compiled = sdfg.compile()
    values = np.random.default_rng(size).normal(size=size)
    result = np.empty_like(values)
    compiled(a=values, b=result, N=size)
    np.testing.assert_allclose(result, values + reduction(values), rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize("conjugate", [False, True])
@pytest.mark.parametrize("implementation", ["pure", pytest.param("CuPy", marks=pytest.mark.gpu)])
def test_complex_dot_preserves_conjugation(conjugate, implementation):
    sdfg = dace.SDFG(f"complex_dot_{implementation}_{conjugate}")
    if implementation == "CuPy":
        sdfg.backend = dace.dtypes.BackendLanguage.Python
    for name in ("x", "y"):
        sdfg.add_array(name, (N,), dace.complex128)
    sdfg.add_array("result", (1,), dace.complex128)
    state = sdfg.add_state()
    dot = Dot("dot", conjugate=conjugate)
    dot.implementation = implementation
    state.add_node(dot)
    for name in ("x", "y"):
        state.add_edge(state.add_read(name), None, dot, f"_{name}", dace.Memlet(f"{name}[0:N]"))
    state.add_edge(dot, "_result", state.add_write("result"), None, dace.Memlet("result[0]"))
    rng = np.random.default_rng(12)
    x, y = [rng.normal(size=17) + 1j * rng.normal(size=17) for _ in range(2)]
    result = np.zeros(1, dtype=np.complex128)
    sdfg.compile()(x=x, y=y, result=result, N=17)
    np.testing.assert_allclose(result[0], np.vdot(x, y) if conjugate else np.dot(x, y), rtol=1e-13)


@pytest.mark.parametrize("expression", ["alpha - alpha", "alpha * 0"])
@pytest.mark.parametrize("kind", ["binop", "gather", "scatter"])
def test_inline_symbol_operands_keep_runtime_arguments(expression: str, kind: str):
    from dace.config import set_temporary
    from dace.libraries.tileops import TileBinop, TileGather, TileScatter

    sdfg = dace.SDFG(f"inline_symbol_liveness_{kind}")
    sdfg.add_symbol("alpha", dace.float64)
    sdfg.add_array("out", (8,), dace.float64)
    state = sdfg.add_state()
    if kind == "binop":
        node = TileBinop("sum", widths=(8,), op="+", kind_a="Symbol", expr_a=expression, kind_b="Symbol", expr_b="1")
        output = "_c"
    else:
        node_type = TileGather if kind == "gather" else TileScatter
        node = node_type("fill", widths=(8,), src_kind="Symbol", src_expr=f"({expression}) + 1")
        output = "_dst"
    node.implementation = "pure"
    state.add_edge(node, output, state.add_write("out"), None, dace.Memlet("out[0:8]"))
    sdfg.simplify()
    assert "alpha" in sdfg.symbols
    with set_temporary("compiler", "cpu", "implementation", value="legacy"):
        with set_temporary("compiler", "cpu", "explicit_copy", value=False):
            compiled = sdfg.compile()
    result = np.empty(8, dtype=np.float64)
    compiled(alpha=3.0, out=result)
    np.testing.assert_array_equal(result, np.ones(8))


def test_iota_functions_are_not_runtime_symbols():
    from dace.libraries.tileops import TileIota

    node = TileIota("indices", widths=(8,), expr="max(start, __l0) + _idx[__l0]", extra_inputs=("_idx",))
    assert node.free_symbols == {"start"}
