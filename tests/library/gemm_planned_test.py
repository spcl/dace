# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The planned OpenBLAS GEMM: a small result over a long K splits K across threads, ``X.T @ X`` takes syrk per
slice, any other shape is one threaded call; every path matches NumPy."""

import numpy as np
import pytest

import dace
from dace import dtypes
from dace.libraries.blas import Gemm
from dace.libraries.blas.environments.openblas import OpenBLAS
from dace.transformation.passes.canonicalize.finalize import canonicalize_set_fast_implementations

pytestmark = pytest.mark.skipif(not OpenBLAS.is_installed(), reason="OpenBLAS is not installed")

M = dace.symbol("M")
N = dace.symbol("N")
K = dace.symbol("K")


def gemm_sdfg(name: str, dtype, trans_a: bool, beta: float, same_operand: bool) -> dace.SDFG:
    """``C = A^(T) @ B + beta * C`` in place; with ``same_operand`` both inputs read ``A`` (``A.T @ A``)."""
    sdfg = dace.SDFG(name)
    sdfg.add_array("A", [K, M] if trans_a else [M, K], dtype)
    if not same_operand:
        sdfg.add_array("B", [K, N], dtype)
    sdfg.add_array("C", [M, N], dtype)
    state = sdfg.add_state()
    gemm = Gemm("gemm", transA=trans_a, alpha=1.0, beta=beta, cin=False)
    state.add_node(gemm)
    a_subset = "0:K, 0:M" if trans_a else "0:M, 0:K"
    state.add_edge(state.add_read("A"), None, gemm, "_a", dace.Memlet(f"A[{a_subset}]"))
    second = "A[0:K, 0:N]" if same_operand else "B[0:K, 0:N]"
    state.add_edge(state.add_read("A" if same_operand else "B"), None, gemm, "_b", dace.Memlet(second))
    state.add_edge(gemm, "_c", state.add_write("C"), None, dace.Memlet("C[0:M, 0:N]"))
    sdfg.validate()
    canonicalize_set_fast_implementations(sdfg, dtypes.DeviceType.CPU)
    assert gemm.implementation == "Planned"
    return sdfg


@pytest.mark.parametrize("dtype", [dace.float32, dace.float64])
@pytest.mark.parametrize("beta", [0.0, 0.5])
def test_a_tall_skinny_product_splits_k(dtype, beta):
    """``X.T @ W`` with a 12 x 7 result over 40000 rows: the split-K path."""
    sdfg = gemm_sdfg(f"planned_split_{dtype.to_string()}_{int(beta * 2)}", dtype, True, beta, False)
    rng = np.random.default_rng(0)
    npdt = dtype.type
    a, b = rng.random((40000, 12)).astype(npdt), rng.random((40000, 7)).astype(npdt)
    c = rng.random((12, 7)).astype(npdt)
    expected = a.T @ b + beta * c
    sdfg(A=a, B=b, C=c, M=12, N=7, K=40000)
    assert np.allclose(c, expected, rtol=1e-4 if dtype == dace.float32 else 1e-11)


def test_a_gramian_of_one_operand_is_symmetric():
    """``X.T @ X``: syrk per slice, the triangle mirrored into the full result."""
    sdfg = gemm_sdfg("planned_gram", dace.float64, True, 0.0, True)
    rng = np.random.default_rng(1)
    a = rng.random((30000, 9))
    c = np.zeros((9, 9))
    sdfg(A=a, C=c, M=9, N=9, K=30000)
    assert np.allclose(c, a.T @ a, rtol=1e-11)
    assert np.array_equal(c, c.T)


def test_a_square_product_is_one_call():
    """A 200 x 200 result is past the split: the plain threaded call, same numbers."""
    sdfg = gemm_sdfg("planned_square", dace.float64, False, 0.0, False)
    rng = np.random.default_rng(2)
    a, b = rng.random((200, 300)), rng.random((300, 200))
    c = np.zeros((200, 200))
    sdfg(A=a, B=b, C=c, M=200, N=200, K=300)
    assert np.allclose(c, a @ b, rtol=1e-12)


if __name__ == "__main__":
    for dtype in [dace.float32, dace.float64]:
        for beta in [0.0, 0.5]:
            test_a_tall_skinny_product_splits_k(dtype, beta)
    test_a_gramian_of_one_operand_is_symmetric()
    test_a_square_product_is_one_call()
