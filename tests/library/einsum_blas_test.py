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

MKL_AND_CUBLAS = [pytest.param("cuBLAS", marks=pytest.mark.gpu), pytest.param("MKL", marks=pytest.mark.mkl)]


def test_change_default():
    old_default = blas.default_implementation

    blas.default_implementation = "hello"

    with change_default(blas, "MKL"):
        assert blas.default_implementation == "MKL"
    assert blas.default_implementation == "hello"
    blas.default_implementation = old_default


def assert_used_environment(sdfg, impl):
    implementation_to_env = {
        "MKL": blas.environments.IntelMKL.full_class_path(),
        "cuBLAS": blas.environments.cuBLAS.full_class_path()
    }
    all_tasklets = (n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.Tasklet))
    environments = {env for n in all_tasklets for env in n.environments}

    assert implementation_to_env[impl] in environments


@pytest.mark.mkl
def test_gemm_fails_storage_mkl():

    with change_default(blas, "MKL"):
        with pytest.raises(ValueError) as info:

            @dace.program
            def test_failing_mkl(A: dace.float32[10, 5], B: dace.float32[5, 3], C: dace.float32[10, 3]):
                C[:] = A @ B

            sdfg = test_failing_mkl.to_sdfg()
            sdfg.apply_gpu_transformations()
            A = np.random.rand(10, 5).astype(np.float32)
            B = np.random.rand(5, 3).astype(np.float32)
            C = np.zeros((10, 3)).astype(np.float32)
            sdfg(A=A, B=B, C=C)
        assert "cannot access" in str(info.value)


@pytest.mark.parametrize("impl", MKL_AND_CUBLAS)
def test_simple(impl):
    A_desc = dace.float32[10, 5]
    B_desc = dace.float32[5, 3]
    C_desc = dace.float32[10, 3]
    with change_default(blas, impl):

        @dace.program
        def test_simple_einsum(A: A_desc, B: B_desc, C: C_desc):
            C[:] = np.einsum("ik,kj->ij", A, B)

        A = np.random.rand(*A_desc.shape).astype(np.float32)
        B = np.random.rand(*B_desc.shape).astype(np.float32)
        C = np.zeros(C_desc.shape).astype(np.float32)

        sdfg: dace.SDFG = test_simple_einsum.to_sdfg()
        sdfg.name = impl + "_einsum_simple"
        if impl == "cuBLAS":
            sdfg.apply_gpu_transformations()
        sdfg.expand_library_nodes()

        assert_used_environment(sdfg, impl)

        sdfg(A=A, B=B, C=C)
        assert np.allclose(A @ B, C)


@pytest.mark.parametrize("impl", MKL_AND_CUBLAS)
def test_3x2(impl):
    A_desc = dace.float32[8, 10, 12]
    B_desc = dace.float32[12, 5]
    C_desc = dace.float32[8, 10, 5]
    with change_default(blas, impl):

        @dace.program
        def test_3x2(A: A_desc, B: B_desc, C: C_desc):
            C[:] = np.einsum("aik,kj->aij", A, B)

        A = np.random.rand(*A_desc.shape).astype(np.float32)
        B = np.random.rand(*B_desc.shape).astype(np.float32)
        C = np.zeros(C_desc.shape).astype(np.float32)

        sdfg: dace.SDFG = test_3x2.to_sdfg()
        sdfg.name = impl + "_einsum_3x2"
        if impl == "cuBLAS":
            sdfg.apply_gpu_transformations()
        sdfg.expand_library_nodes()

        assert_used_environment(sdfg, impl)

        sdfg(A=A, B=B, C=C)
        assert np.allclose(A @ B, C)


@pytest.mark.parametrize("impl", MKL_AND_CUBLAS)
def test_4x4(impl):
    A_desc = dace.float32[8, 12, 5, 3]
    B_desc = dace.float32[8, 12, 3, 6]
    C_desc = dace.float32[8, 12, 5, 6]
    with change_default(blas, impl):

        @dace.program
        def test_4x4(A: A_desc, B: B_desc, C: C_desc):
            C[:] = np.einsum("abik,abkj->abij", A, B)

        A = np.random.rand(*A_desc.shape).astype(np.float32)
        B = np.random.rand(*B_desc.shape).astype(np.float32)
        C = np.zeros(C_desc.shape).astype(np.float32)

        sdfg: dace.SDFG = test_4x4.to_sdfg()
        sdfg.name = impl + "_einsum_4x4"
        if impl == "cuBLAS":
            sdfg.apply_gpu_transformations()
        sdfg.expand_library_nodes()

        assert_used_environment(sdfg, impl)

        sdfg(A=A, B=B, C=C)
        assert np.allclose(A @ B, C)
