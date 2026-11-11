# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``Symm``, ``Syrk``, ``Syr2k``, ``BatchedMatMul`` and ``Einsum`` inside a GPU kernel run on the kernel's thread block,
each lane on its share of the output, rather than pinning the map around them to the host."""

import numpy as np
import pytest

from dace.libraries.blas import BatchedMatMul, Einsum, Symm, Syr2k, Syrk
from tests.library.blas_in_kernel_common import lowered_on_the_block, per_block, random_inputs, run_on_the_block

M, N, K = 20, 17, 9


def symm_per_block(name: str):
    """``C = 2 * B @ A + 0.5 * C`` (right side) with only the lower triangle of ``A`` read."""
    node = Symm("symm", side=True, uplo=False, alpha=2.0, beta=0.5)
    operands = [("_A", "A", (N, N), False), ("_B", "B", (M, N), False), ("_Cin", "C", (M, N), False)]
    sdfg, state = per_block(name, node, operands + [("_Cout", "C", (M, N), True)])
    return sdfg, state, node


def rank_k_per_block(name: str, two: bool):
    """``C = 1.5 * (A^T @ B + B^T @ A) + 0.5 * C`` (``syr2k``) or ``1.5 * A^T @ A + 0.5 * C``, upper triangle only."""
    node = (Syr2k if two else Syrk)("rank_k", uplo=True, transA=True, alpha=1.5, beta=0.5)
    operands = [("_A", "A", (K, N), False), ("_Cin", "C", (N, N), False), ("_Cout", "C", (N, N), True)]
    if two:
        operands.append(("_B", "B", (K, N), False))
    sdfg, state = per_block(name, node, operands)
    return sdfg, state, node


def bmm_per_block(name: str):
    """Two products per block, ``B`` broadcast over them."""
    node = BatchedMatMul("bmm")
    operands = [("_a", "A", (2, M, K), False), ("_b", "B", (1, K, N), False), ("_c", "C", (2, M, N), True)]
    sdfg, state = per_block(name, node, operands)
    return sdfg, state, node


def einsum_per_block(name: str):
    """A contraction over two indices: ``out[i] = 3 * sum_jk a[i, j, k] * b[k, j]``."""
    node = Einsum("einsum")
    node.einsum_str, node.alpha, node.beta = "ijk,kj->i", 3.0, 0.0
    for conn in ("a", "b"):
        node.add_in_connector(conn)
    node.add_out_connector("out")
    operands = [("a", "A", (M, N, K), False), ("b", "B", (K, N), False), ("out", "C", (M,), True)]
    sdfg, state = per_block(name, node, operands)
    return sdfg, state, node


def test_a_symm_in_a_kernel_reads_one_triangle_on_the_block():
    code = lowered_on_the_block(*symm_per_block("symm_block_lowering"))
    assert "__k >= __j" in code


@pytest.mark.parametrize("two", [False, True])
def test_a_rank_k_update_in_a_kernel_keeps_the_other_triangle_on_the_block(two: bool):
    code = lowered_on_the_block(*rank_k_per_block(f"rank_k_block_lowering_{int(two)}", two))
    assert "(__i <= __j) ?" in code


def test_a_batched_matmul_in_a_kernel_broadcasts_on_the_block():
    code = lowered_on_the_block(*bmm_per_block("bmm_block_lowering"))
    assert "__b1" in code


def test_an_einsum_in_a_kernel_contracts_on_the_block():
    code = lowered_on_the_block(*einsum_per_block("einsum_block_lowering"))
    assert "__ej" in code and "__ek" in code


@pytest.mark.gpu
def test_a_symm_on_the_block_matches_numpy():
    built = symm_per_block("symm_block_run")
    inputs = random_inputs({"A": (N, N), "B": (M, N), "C": (M, N)})
    inputs["A"][:, np.triu(np.ones((N, N), dtype=bool), 1)] = np.nan
    out = run_on_the_block(built, inputs)
    lower = np.tril(np.nan_to_num(inputs["A"]))
    symmetric = lower + np.tril(lower, -1).transpose(0, 2, 1)
    np.testing.assert_allclose(out["C"], 2.0 * inputs["B"] @ symmetric + 0.5 * inputs["C"], rtol=1e-13)


@pytest.mark.gpu
@pytest.mark.parametrize("two", [False, True])
def test_a_rank_k_update_on_the_block_matches_numpy(two: bool):
    built = rank_k_per_block(f"rank_k_block_run_{int(two)}", two)
    inputs = random_inputs({"A": (K, N), "B": (K, N), "C": (N, N)} if two else {"A": (K, N), "C": (N, N)})
    out = run_on_the_block(built, inputs)
    a, c = inputs["A"], inputs["C"]
    at = a.transpose(0, 2, 1)
    product = at @ inputs["B"] + inputs["B"].transpose(0, 2, 1) @ a if two else at @ a
    upper = np.triu(np.ones((N, N), dtype=bool))
    np.testing.assert_allclose(out["C"], np.where(upper, 1.5 * product + 0.5 * c, c), rtol=1e-13)


@pytest.mark.gpu
def test_a_batched_matmul_on_the_block_matches_numpy():
    built = bmm_per_block("bmm_block_run")
    inputs = random_inputs({"A": (2, M, K), "B": (1, K, N), "C": (2, M, N)})
    out = run_on_the_block(built, inputs)
    np.testing.assert_allclose(out["C"], inputs["A"] @ inputs["B"], rtol=1e-13)


@pytest.mark.gpu
def test_an_einsum_on_the_block_matches_numpy():
    built = einsum_per_block("einsum_block_run")
    inputs = random_inputs({"A": (M, N, K), "B": (K, N), "C": (M,)})
    out = run_on_the_block(built, inputs)
    np.testing.assert_allclose(out["C"], 3.0 * np.einsum("bijk,bkj->bi", inputs["A"], inputs["B"]), rtol=1e-13)


if __name__ == "__main__":
    test_a_symm_in_a_kernel_reads_one_triangle_on_the_block()
    test_a_rank_k_update_in_a_kernel_keeps_the_other_triangle_on_the_block(False)
    test_a_rank_k_update_in_a_kernel_keeps_the_other_triangle_on_the_block(True)
    test_a_batched_matmul_in_a_kernel_broadcasts_on_the_block()
    test_an_einsum_in_a_kernel_contracts_on_the_block()
    test_a_symm_on_the_block_matches_numpy()
    test_a_rank_k_update_on_the_block_matches_numpy(False)
    test_a_rank_k_update_on_the_block_matches_numpy(True)
    test_a_batched_matmul_on_the_block_matches_numpy()
    test_an_einsum_on_the_block_matches_numpy()
