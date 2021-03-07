# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import numpy as np
import pytest

import dace


def test_redundant_array_removal():
    @dace.program
    def reshape(data: dace.float64[9], reshaped: dace.float64[3, 3]):
        reshaped[:] = np.reshape(data, [3, 3])

    @dace.program
    def test_redundant_array_removal(A: dace.float64[9], B: dace.float64[3]):
        A_reshaped = dace.define_local([3, 3], dace.float64)
        reshape(A, A_reshaped)
        return A_reshaped + B

    A = np.arange(9).astype(np.float64)
    B = np.arange(3).astype(np.float64)
    result = test_redundant_array_removal(A.copy(), B.copy())
    assert np.allclose(result, A.reshape(3, 3) + B)

    data_accesses = {
        n.data
        for n, _ in
        test_redundant_array_removal.to_sdfg(strict=True).all_nodes_recursive()
        if isinstance(n, dace.nodes.AccessNode)
    }
    assert "A_reshaped" not in data_accesses


@pytest.mark.gpu
def test_libnode_expansion():
    @dace.program
    def test_broken_matmul(A: dace.float64[8, 2, 4], B: dace.float64[4, 3]):
        return np.einsum("aik,kj->aij", A, B)

    sdfg = test_broken_matmul.to_sdfg()
    sdfg.expand_library_nodes()
    sdfg.apply_gpu_transformations()
    sdfg.apply_strict_transformations()

    A = np.random.rand(8, 2, 4).astype(np.float64)
    B = np.random.rand(4, 3).astype(np.float64)
    C = test_broken_matmul(A.copy(), B.copy())

    assert np.allclose(A @ B, C)
