# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A guarded product of gathered blocks in a loop nest becomes ONE gathered ``BatchedMatMul`` on the GPU target."""

import os

os.environ.setdefault("OMPI_MCA_pml", "ob1")
os.environ.setdefault("OMPI_MCA_btl", "self,vader")
os.environ.setdefault("UCX_VFS_ENABLE", "n")
os.environ.setdefault("MPI4PY_RC_INITIALIZE", "0")

import numpy as np
import pytest

import dace
from dace.libraries.blas.nodes.batched_matmul import BATCH_INDEX, BatchedMatMul
from dace.libraries.blas.nodes.matmul import MatMul
from dace.transformation.passes.canonicalize.pipeline import canonicalize

NA, NB, BS, R = (dace.symbol(s) for s in ("NA", "NB", "BS", "R"))


@dace.program
def gathered_block_products(
    a_index: dace.int64[NA, 3],
    b_index: dace.int64[NB, 3],
    a_blocks: dace.float64[NA, BS, BS],
    b_blocks: dace.float64[NB, BS, BS],
    C: dace.float64[R * BS, R * BS],
):
    for ia in range(NA):
        a_bid = a_index[ia, 2]
        if a_bid < 0:
            continue
        a_row = a_index[ia, 0]
        a_k = a_index[ia, 1]
        for ib in range(NB):
            b_bid = b_index[ib, 2]
            if b_bid < 0:
                continue
            b_k = b_index[ib, 0]
            if a_k != b_k:
                continue
            b_col = b_index[ib, 1]
            product = a_blocks[a_bid] @ b_blocks[b_bid]
            for bi in range(BS):
                for bj in range(BS):
                    C[a_row * BS + bi, b_col * BS + bj] += product[bi, bj]


def blocks(seed: int, n: int, r: int, bs: int):
    """Block coordinates ``(row or k, k or col, id)`` with some slots unused (``id`` -1), and their payloads."""
    rng = np.random.default_rng(seed)
    a_index = np.stack([rng.integers(0, r, n), rng.integers(0, r, n), rng.permutation(n)], axis=1)
    b_index = np.stack([rng.integers(0, r, n), rng.integers(0, r, n), rng.permutation(n)], axis=1)
    a_index[rng.random(n) < 0.3, 2] = -1
    b_index[rng.random(n) < 0.3, 2] = -1
    return a_index, b_index, rng.random((n, bs, bs)), rng.random((n, bs, bs))


def reference(a_index, b_index, a_blocks, b_blocks, r, bs):
    C = np.zeros((r * bs, r * bs))
    for row, k, a in a_index:
        for k2, col, b in b_index:
            if a >= 0 and b >= 0 and k == k2:
                C[row * bs : (row + 1) * bs, col * bs : (col + 1) * bs] += a_blocks[a] @ b_blocks[b]
    return C


def batched_sdfg(name: str):
    sdfg = gathered_block_products.to_sdfg(simplify=True)
    sdfg.name = name
    canonicalize(sdfg, target="gpu")
    return sdfg


def test_the_guarded_block_products_become_one_gathered_batched_product():
    """Every product the guard lets through runs in one ``BatchedMatMul`` reading both pools through an index."""
    sdfg = batched_sdfg("gathered_block_products_structure")
    libs = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, (MatMul, BatchedMatMul))]
    assert len(libs) == 1 and isinstance(libs[0], BatchedMatMul), libs
    assert set(BATCH_INDEX.values()) <= set(libs[0].in_connectors)


@pytest.mark.gpu
def test_the_gathered_batched_product_matches_numpy():
    """The device run sums each guarded pair's product into its block, as the nest does."""
    import cupy as cp

    from dace.transformation.passes.canonicalize.finalize import finalize_for_target, offload_to_gpu

    sdfg = batched_sdfg("gathered_block_products_gpu")
    offload_to_gpu(sdfg)
    finalize_for_target(sdfg, "gpu")
    n, r, bs = 24, 4, 5
    a_index, b_index, a_blocks, b_blocks = blocks(0, n, r, bs)
    C = cp.zeros((r * bs, r * bs))
    sdfg(
        a_index=cp.asarray(a_index),
        b_index=cp.asarray(b_index),
        a_blocks=cp.asarray(a_blocks),
        b_blocks=cp.asarray(b_blocks),
        C=C,
        NA=n,
        NB=n,
        BS=bs,
        R=r,
    )
    assert np.allclose(C.get(), reference(a_index, b_index, a_blocks, b_blocks, r, bs), rtol=1e-12)


if __name__ == "__main__":
    test_the_guarded_block_products_become_one_gathered_batched_product()
    test_the_gathered_batched_product_matches_numpy()
