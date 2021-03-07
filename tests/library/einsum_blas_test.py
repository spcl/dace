# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
"""
These tests are mostly from daceml, testing that the einsums for the BERT encoder are correctly specialized to BLAS
nodes.
"""

import pytest
import numpy as np
import dace
from dace.library import change_default
from dace.libraries import blas


def assert_used_environment(sdfg, impl):
    implementation_to_env = {"MKL": "IntelMKL", "cuBLAS": "cuBLAS"}
    all_tasklets = (n for n, _ in sdfg.all_nodes_recursive()
                    if isinstance(n, dace.nodes.Tasklet))
    environments = {env for n in all_tasklets for env in n.environments}

    assert implementation_to_env[impl] in environments


@pytest.mark.mkl
def test_gemm_fails_storage_mkl():

    with change_default(blas, "MKL"):
        with pytest.raises(ValueError) as info:

            @dace.program
            def test_failing_mkl(A: dace.float32[10, 5], B: dace.float32[5, 3],
                                 C: dace.float32[10, 3]):
                C[:] = A @ B

            sdfg = test_failing_mkl.to_sdfg()
            sdfg.apply_gpu_transformations()
            A = np.random.rand(10, 5).astype(np.float32)
            B = np.random.rand(5, 3).astype(np.float32)
            C = np.zeros((10, 3)).astype(np.float32)
            sdfg(A=A, B=B, C=C)
        assert "cannot access" in str(info.value)


@pytest.mark.gpu
def test_gemm_fails_storage_cuda():

    with change_default(blas, "cuBLAS"):
        with pytest.raises(ValueError) as info:

            @dace.program
            def test_failing_cublas(A: dace.float32[10, 5],
                                    B: dace.float32[5, 3], C: dace.float32[10,
                                                                           3]):
                C[:] = A @ B

            sdfg = test_failing_cublas.to_sdfg()
            A = np.random.rand(10, 5).astype(np.float32)
            B = np.random.rand(5, 3).astype(np.float32)
            C = np.zeros((10, 3)).astype(np.float32)
            sdfg(A=A, B=B, C=C)
        assert "cannot access" in str(info.value)
