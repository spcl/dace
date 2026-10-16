# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Symbolic, multidimensional execution through both tile backends."""

import numpy as np
import pytest

import dace
from dace import dtypes
from dace.codegen.compiled_sdfg import CompiledSDFG
from dace.codegen.py.compiled_sdfg import PythonCompiledSDFG
from dace.config import set_temporary
from dace.libraries.tileops.nodes import TILEOPS_NODE_TYPES
from dace.transformation.passes.vectorization import VectorizeCPUMultiDim, VectorizeCuTile
from dace.transformation.passes.vectorization.config import VectorizeConfig

M = dace.symbol("M")
N = dace.symbol("N")


@dace.program
def mixed_matrix(A: dace.float32[M, N], B: dace.float64[M, N], alpha: dace.float64):
    for i, j in dace.map[0:M, 0:N]:
        B[i, j] = dace.float64(A[i, j]) * alpha


@pytest.fixture(scope="module", params=["pure", pytest.param("cutile", marks=pytest.mark.gpu)])
def compiled_matrix(request: pytest.FixtureRequest) -> CompiledSDFG | PythonCompiledSDFG:
    sdfg = mixed_matrix.to_sdfg(simplify=False)
    if request.param == "cutile":
        VectorizeCuTile(widths=(8, 4)).apply_pass(sdfg, {})
    else:
        config = VectorizeConfig(widths=(8, 4), target_isa="SCALAR", remainder_strategy="full_mask")
        VectorizeCPUMultiDim(config).apply_pass(sdfg, {})
        sdfg.backend = dtypes.BackendLanguage.CPP
    assert any(isinstance(node, TILEOPS_NODE_TYPES) for node, _ in sdfg.all_nodes_recursive())
    # The classic CPU backend isolates this test from unrelated readable-backend library conflicts.
    with set_temporary("compiler", "cpu", "implementation", value="legacy"):
        with set_temporary("compiler", "cpu", "explicit_copy", value=False):
            compiled = sdfg.compile()
    if request.param == "cutile":
        assert "@ct.kernel" in compiled.code
    return compiled


@pytest.mark.parametrize("shape", [(1, 1), (8, 4), (17, 9)])
def test_symbolic_matrix_mixed_dtypes(compiled_matrix, shape: tuple[int, int]):
    values = np.random.default_rng(17).normal(size=shape).astype(np.float32)
    output = np.full(shape, np.nan, dtype=np.float64)
    alpha = np.float64(1.0 + 2.0**-40)
    compiled_matrix(A=values, B=output, alpha=alpha, M=np.int64(shape[0]), N=np.int64(shape[1]))
    np.testing.assert_array_equal(output, values.astype(np.float64) * alpha)


def test_pure_binop_resolves_interstate_operand_types():
    from dace.libraries.tileops import TileBinop

    sdfg = dace.SDFG("scoped_binop_operand_types")
    sdfg.add_array("out", [1], dace.float32)
    init = sdfg.add_state(is_start_block=True)
    state = sdfg.add_state()
    sdfg.add_edge(
        init,
        state,
        dace.InterstateEdge(
            assignments={
                "left": "1099511627777",
                "right": "1099511627776",
            }
        ),
    )
    node = TileBinop(
        name="difference",
        widths=(1,),
        op="-",
        kind_a="Symbol",
        kind_b="Symbol",
        expr_a="left",
        expr_b="right",
    )
    node.implementation = "pure"
    state.add_node(node)
    state.add_edge(node, "_c", state.add_write("out"), None, dace.Memlet("out[0]"))
    assert "left" not in sdfg.symbols
    with set_temporary("compiler", "cpu", "implementation", value="legacy"):
        with set_temporary("compiler", "cpu", "explicit_copy", value=False):
            compiled = sdfg.compile()
    result = np.full(1, np.nan, dtype=np.float32)
    compiled(out=result)
    np.testing.assert_array_equal(result, [1.0])
