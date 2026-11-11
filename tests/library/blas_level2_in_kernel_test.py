# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``Gemv``, ``Ger`` and ``Symv`` inside a GPU kernel run on the kernel's thread block, each lane on its share of the
output, rather than pinning the map around them to the host."""

import numpy as np
import pytest

from dace.libraries.blas import Gemv, Ger, Symv
from tests.library.blas_in_kernel_common import lowered_on_the_block, per_block, random_inputs, run_on_the_block

M, N = 300, 7


def gemv_per_block(name: str, trans: bool, beta: float):
    """``y[b] = 2 * op(A[b]) @ x[b] + beta * y[b]``, ``op(A)`` ``M x N``."""
    node = Gemv("gemv", transA=trans, alpha=2.0, beta=beta)
    operands = [("_A", "A", (N, M) if trans else (M, N), False), ("_x", "x", (N,), False), ("_y", "y", (M,), True)]
    if beta != 0:
        operands.append(("_y", "y", (M,), False))
    sdfg, state = per_block(name, node, operands)
    return sdfg, state, node


def ger_per_block(name: str):
    node = Ger("ger", alpha=1.5)
    operands = [("_A", "A", (M, N), False), ("_x", "x", (M,), False), ("_y", "y", (N,), False)]
    sdfg, state = per_block(name, node, operands + [("_res", "A", (M, N), True)])
    return sdfg, state, node


def symv_per_block(name: str):
    """Only the upper triangle of ``A`` is read."""
    node = Symv("symv", uplo=True, alpha=2.0, beta=0.5)
    operands = [("_A", "A", (N, N), False), ("_x", "x", (N,), False), ("_yin", "y", (N,), False)]
    sdfg, state = per_block(name, node, operands + [("_yout", "y", (N,), True)])
    return sdfg, state, node


def test_a_gemv_in_a_kernel_lowers_to_the_block():
    code = lowered_on_the_block(*gemv_per_block("gemv_block_lowering", False, 0.0))
    assert "__acc = __acc + __A[" in code and "__beta" not in code


def test_a_ger_in_a_kernel_lowers_to_the_block():
    code = lowered_on_the_block(*ger_per_block("ger_block_lowering"))
    assert "__alpha * __x[" in code


def test_a_symv_in_a_kernel_reads_one_triangle_on_the_block():
    code = lowered_on_the_block(*symv_per_block("symv_block_lowering"))
    assert "__i <= __j" in code


@pytest.mark.gpu
@pytest.mark.parametrize("trans, beta", [(False, 0.0), (True, 0.5)])
def test_a_gemv_on_the_block_matches_numpy(trans: bool, beta: float):
    built = gemv_per_block(f"gemv_block_run_{int(trans)}_{int(beta != 0)}", trans, beta)
    inputs = random_inputs({"A": (N, M) if trans else (M, N), "x": (N,), "y": (M,)})
    out = run_on_the_block(built, inputs)
    matrix = inputs["A"].transpose(0, 2, 1) if trans else inputs["A"]
    expected = 2.0 * np.einsum("bij,bj->bi", matrix, inputs["x"]) + beta * inputs["y"]
    np.testing.assert_allclose(out["y"], expected, rtol=1e-13)


@pytest.mark.gpu
def test_a_ger_on_the_block_matches_numpy():
    built = ger_per_block("ger_block_run")
    inputs = random_inputs({"A": (M, N), "x": (M,), "y": (N,)})
    out = run_on_the_block(built, inputs)
    np.testing.assert_allclose(out["A"], inputs["A"] + 1.5 * np.einsum("bi,bj->bij", inputs["x"], inputs["y"]))


@pytest.mark.gpu
def test_a_symv_on_the_block_matches_numpy():
    built = symv_per_block("symv_block_run")
    inputs = random_inputs({"A": (N, N), "x": (N,), "y": (N,)})
    # The lower triangle is never read: poison it
    inputs["A"][:, np.tril(np.ones((N, N), dtype=bool), -1)] = np.nan
    out = run_on_the_block(built, inputs)
    upper = np.triu(np.nan_to_num(inputs["A"]))
    symmetric = upper + np.triu(upper, 1).transpose(0, 2, 1)
    expected = 2.0 * np.einsum("bij,bj->bi", symmetric, inputs["x"]) + 0.5 * inputs["y"]
    np.testing.assert_allclose(out["y"], expected, rtol=1e-13)


if __name__ == "__main__":
    test_a_gemv_in_a_kernel_lowers_to_the_block()
    test_a_ger_in_a_kernel_lowers_to_the_block()
    test_a_symv_in_a_kernel_reads_one_triangle_on_the_block()
    test_a_gemv_on_the_block_matches_numpy(False, 0.0)
    test_a_gemv_on_the_block_matches_numpy(True, 0.5)
    test_a_ger_on_the_block_matches_numpy()
    test_a_symv_on_the_block_matches_numpy()
