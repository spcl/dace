# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.

import dace
import copy
import pytest
import numpy
from dace.transformation.passes.explicit_vectorization_cpu import ExplicitVectorizationPipelineCPU
from dace.transformation.passes.explicit_vectorization_gpu import ExplicitVectorizationPipelineGPU

# Vector Addition Symbols
N = dace.symbol('N')
# Tasklet in NestedSDFGs Symbols
S1 = dace.symbol("S1")
S2 = dace.symbol("S2")
S = dace.symbol("S")
# CloudSC Symbols
klev = dace.symbol("klev")
kidia = dace.symbol("kidia")
kfdia = dace.symbol("kfdia")
# SpMV Symbols
n = dace.symbol('n')  # number of rows
m = dace.symbol('m')  # number of columns
nnz = dace.symbol('nnz')  # number of nonzeros


@dace.program
def vadds_gpu(A: dace.float64[N, N] @ dace.dtypes.StorageType.GPU_Global,
              B: dace.float64[N, N] @ dace.dtypes.StorageType.GPU_Global):
    for i, j in dace.map[0:N, 0:N] @ dace.dtypes.ScheduleType.GPU_Device:
        A[i, j] = A[i, j] + B[i, j]
    for i, j in dace.map[0:N, 0:N] @ dace.dtypes.ScheduleType.GPU_Device:
        B[i, j] = 3 * B[i, j]**2.0 + 2.0


@dace.program
def vadds_cpu(A: dace.float64[N, N], B: dace.float64[N, N]):
    for i, j in dace.map[0:N, 0:N]:
        A[i, j] = A[i, j] + B[i, j]
    for i, j in dace.map[0:N, 0:N]:
        B[i, j] = 3 * B[i, j]**2.0 + 2.0


@dace.program
def no_maps(A: dace.float64[N, N], B: dace.float64[N, N]):
    i = 8
    j = 7
    A[i, j] = 2.0 * A[i, j]
    B[i + 1, j + 1] = B[i, j] / 1.5


@dace.program
def cloudsc_snippet_one(za: dace.float64[klev, kfdia], zliqfrac: dace.float64[klev, kfdia],
                        zicefrac: dace.float64[klev, kfdia], zqx: dace.float64[klev, kfdia, 5],
                        zli: dace.float64[klev, kfdia], zy: dace.float64[klev, kfdia, 5],
                        zx: dace.float64[klev, kfdia, 4], rlmin: dace.float64, z1: dace.int64, z2: dace.int64):
    for i in range(1, klev + 1):
        for j in range(kidia + 1, kfdia + 1):
            za[i - 1, j - 1] = 2.0 * za[i - 1, j - 1] - 5
            cond1 = rlmin > (0.5 * (zqx[i - 1, j - 1, z1] + zqx[i, j, z2]))
            if cond1:
                zliqfrac[i - 1, j - 1] = zqx[i - 1, j - 1, z1] * zli[i - 1, j - 1]
                zicefrac[i - 1, j - 1] = 1 - zliqfrac[i - 1, j - 1]
            else:
                zliqfrac[i - 1, j - 1] = 0
                zicefrac[i - 1, j - 1] = 0
            for m in dace.map[1:5:1]:
                zx[i - 1, j - 1, m - 1] = zy[i - 1, z1, z2]


@dace.program
def tasklet_in_nested_sdfg(
    a: dace.float64[S, S],
    b: dace.float64[S, S],
    offset1: dace.int64,
    offset2: dace.int64,
):
    for i, j in dace.map[S1:S2:1, S1:S2:1] @ dace.dtypes.ScheduleType.Sequential:
        # Complicated NestedSDFG with offset1 and offset2 in the NestedSDFG as symbols
        a[i + offset1, j + offset2] = ((1.5 * b[i + offset1, j + offset2]) + (2.0 * a[i + offset1, j + offset2])) / 3.5


@dace.program
def tasklet_in_nested_sdfg_2(
    a: dace.float64[S, S],
    b: dace.float64[S, S],
    offset1: dace.int64,
    offset2: dace.int64,
):
    # If a scalar is always added to a map param
    # Then move the scalar to the loop like this
    #for i, j in dace.map[S1:S2:1, S1:S2:1] @ dace.dtypes.ScheduleType.Sequential:
    #    a[i + offset1, j + offset2] = (
    #        (1.5 * b[i + offset1, j + offset2]) +
    #        (2.0 * a[i + offset1, j + offset2])
    #    ) / 3.5
    for i, j in dace.map[S1 - offset1:S2 - offset1:1,
                         S1 - offset2:S2 - offset2:1] @ dace.dtypes.ScheduleType.Sequential:
        a[i, j] = ((1.5 * b[i, j]) + (2.0 * a[i, j])) / 3.5


@dace.program
def tasklets_in_if(
    a: dace.float64[S, S],
    b: dace.float64[S, S],
    d: dace.float64[S, S],
    c: dace.float64,
):
    for i in dace.map[S1:S2:1]:
        for j in dace.map[S1:S2:1]:
            if a[i, j] > c:
                b[i, j] = b[i, j] + d[i, j]
            else:
                b[i, j] = b[i, j] - d[i, j]
            b[i, j] = (1 - a[i, j]) * c


@dace.program
def tasklets_in_if_two(
    a: dace.float64[S, S],
    b: dace.float64,
    c: dace.float64[S, S],
    d: dace.float64[S, S],
    e: dace.float64[S, S],
    f: dace.float64,
):
    for i in dace.map[0:S - 1:1]:
        for j in dace.map[0:S - 1:1]:
            if a[i, j] + a[i + 1, j + 1] < b:
                g = f * a[i, j]
                d[i, j] = c[i, j] * g
                e[i, j] = d[i, j] * 2.0 - a[i, j]


@dace.program
def spmv_csr(indptr: dace.int64[n + 1], indices: dace.int64[nnz], data: dace.float64[nnz], x: dace.float64[m],
             y: dace.float64[n]):
    n_rows = len(indptr) - 1

    for i in dace.map[0:n_rows:1]:
        row_start = indptr[i]
        row_end = indptr[i + 1]
        tmp = 0.0
        for idx in dace.map[row_start:row_end:1]:
            j = indices[idx]
            tmp = tmp + data[idx] * x[j]
        y[i] = tmp


@dace.program
def spmv_csr_2(indptr: dace.int64[n + 1], indices: dace.int64[nnz], data: dace.float64[nnz], x: dace.float64[m],
               y: dace.float64[n]):
    n_rows = len(indptr) - 1

    for i in dace.map[0:n_rows:1]:
        row_start = indptr[i]
        row_end = indptr[i + 1]
        tmp = 0.0
        for idx in dace.map[row_start:row_end:1]:
            j = indices[idx]
            tmp = tmp + data[idx] * x[j]
        y[i] = tmp


@pytest.mark.gpu
def _test_simple_gpu():
    import cupy

    # Allocate 64x64 GPU arrays using CuPy
    A_gpu = cupy.random.random((64, 64), dtype=cupy.float64)
    B_gpu = cupy.random.random((64, 64), dtype=cupy.float64)

    # Create copies for comparison
    A_orig = cupy.copy(A_gpu)
    B_orig = cupy.copy(B_gpu)
    A_vec = cupy.copy(A_gpu)
    B_vec = cupy.copy(B_gpu)

    # Original SDFG
    sdfg = vadds_gpu.to_sdfg()
    c_sdfg = sdfg.compile()

    # Vectorized SDFG
    copy_sdfg = copy.deepcopy(sdfg)
    ExplicitVectorizationPipelineGPU(vector_width=4).apply_pass(copy_sdfg, {})
    c_copy_sdfg = copy_sdfg.compile()

    c_sdfg(A=A_orig, B=B_orig, N=64)
    c_copy_sdfg(A=A_vec, B=B_vec, N=64)

    # Compare results
    assert cupy.allclose(A_orig, A_vec)
    assert cupy.allclose(B_orig, B_vec)


def test_simple_cpu():
    import numpy

    # Allocate 64x64 CPU arrays using NumPy
    A_cpu = numpy.random.random((64, 64))
    B_cpu = numpy.random.random((64, 64))

    # Create copies for comparison
    A_orig = A_cpu.copy()
    B_orig = B_cpu.copy()
    A_vec = A_cpu.copy()
    B_vec = B_cpu.copy()

    # Original SDFG
    sdfg = vadds_cpu.to_sdfg()
    c_sdfg = sdfg.compile()

    # Vectorized SDFG
    copy_sdfg = copy.deepcopy(sdfg)
    copy_sdfg.save("vadd.sdfg")
    ExplicitVectorizationPipelineCPU(vector_width=4).apply_pass(copy_sdfg, {})
    c_copy_sdfg = copy_sdfg.compile()
    copy_sdfg.save("vadd_vectorized.sdfg")

    c_sdfg(A=A_orig, B=B_orig, N=64)
    c_copy_sdfg(A=A_vec, B=B_vec, N=64)

    # Compare results
    assert numpy.allclose(A_orig, A_vec), f"{A_orig - A_vec}"
    assert numpy.allclose(B_orig, B_vec), f"{B_orig - B_vec}"


def test_nested_sdfg():
    _S1 = 1
    _S2 = 65
    _S = 64
    A = numpy.random.random((_S, _S))
    B = numpy.random.random((_S, _S))

    # Create copies for comparison
    A_orig = copy.deepcopy(A)
    B_orig = copy.deepcopy(B)
    A_vec = copy.deepcopy(A)
    B_vec = copy.deepcopy(B)

    # Original SDFG
    sdfg = tasklet_in_nested_sdfg.to_sdfg()

    # Vectorized SDFG
    copy_sdfg = copy.deepcopy(sdfg)
    sdfg.save("nested_tasklets.sdfg")
    ExplicitVectorizationPipelineCPU(vector_width=8).apply_pass(copy_sdfg, {})
    copy_sdfg.save("nested_tasklets_vectorized.sdfg")

    c_sdfg = sdfg.compile()
    c_copy_sdfg = copy_sdfg.compile()

    c_sdfg(a=A_orig, b=B_orig, S=_S, S1=_S1, S2=_S2, offset1=-1, offset2=-1)
    c_copy_sdfg(a=A_vec, b=B_vec, S=_S, S1=_S1, S2=_S2, offset1=-1, offset2=-1)

    # Compare results
    assert numpy.allclose(A_orig, A_vec), f"{A_orig - A_vec}"
    assert numpy.allclose(B_orig, B_vec), f"{B_orig - B_vec}"


def test_no_maps():
    _N = 16
    A = numpy.random.random((_N, _N))
    B = numpy.random.random((_N, _N))

    # Create copies for comparison
    A_orig = copy.deepcopy(A)
    B_orig = copy.deepcopy(B)
    A_vec = copy.deepcopy(A)
    B_vec = copy.deepcopy(B)

    # Original SDFG
    sdfg = no_maps.to_sdfg()

    # Vectorized SDFG
    copy_sdfg = copy.deepcopy(sdfg)
    sdfg.save("no_maps.sdfg")
    ExplicitVectorizationPipelineCPU(vector_width=8).apply_pass(copy_sdfg, {})
    copy_sdfg.save("no_maps_vectorized.sdfg")

    c_sdfg = sdfg.compile()
    c_copy_sdfg = copy_sdfg.compile()

    c_sdfg(A=A_orig, B=B_orig, N=_N)
    c_copy_sdfg(A=A_vec, B=B_vec, N=_N)

    # Compare results
    assert numpy.allclose(A_orig, A_vec), f"{A_orig - A_vec}"
    assert numpy.allclose(B_orig, B_vec), f"{B_orig - B_vec}"


def test_tasklets_in_if():
    _S = 64
    A = numpy.random.random((_S, _S))
    B = numpy.random.random((_S, _S))

    # Create copies for comparison
    A_orig = copy.deepcopy(A)
    B_orig = copy.deepcopy(B)
    A_vec = copy.deepcopy(A)
    B_vec = copy.deepcopy(B)

    # Original SDFG
    sdfg = tasklets_in_if.to_sdfg()

    # Vectorized SDFG
    copy_sdfg = copy.deepcopy(sdfg)
    sdfg.save("nested_tasklets.sdfg")
    ExplicitVectorizationPipelineCPU(vector_width=8).apply_pass(copy_sdfg, {})
    copy_sdfg.save("nested_tasklets_vectorized.sdfg")

    c_sdfg = sdfg.compile()
    c_copy_sdfg = copy_sdfg.compile()

    c_sdfg(a=A_orig, b=B_orig, S=_S)
    c_copy_sdfg(a=A_vec, b=B_vec, S=_S)

    # Compare results
    assert numpy.allclose(A_orig, A_vec), f"{A_orig - A_vec}"
    assert numpy.allclose(B_orig, B_vec), f"{B_orig - B_vec}"


def test_tasklets_in_if_two():
    _S = 64
    A = numpy.random.random((_S, _S))
    B = numpy.random.random((1, ))
    C = numpy.random.random((_S, _S))
    D = numpy.random.random((_S, _S))
    E = numpy.random.random((_S, _S))
    F = numpy.random.random((1, ))

    # Create copies for comparison
    A_orig = copy.deepcopy(A)
    B_orig = copy.deepcopy(B)
    C_orig = copy.deepcopy(C)
    D_orig = copy.deepcopy(D)
    E_orig = copy.deepcopy(E)
    F_orig = copy.deepcopy(F)
    A_vec = copy.deepcopy(A)
    B_vec = copy.deepcopy(B)
    C_vec = copy.deepcopy(C)
    D_vec = copy.deepcopy(D)
    E_vec = copy.deepcopy(E)
    F_vec = copy.deepcopy(F)

    # Original SDFG
    sdfg = tasklets_in_if.to_sdfg()

    # Vectorized SDFG
    copy_sdfg = copy.deepcopy(sdfg)
    sdfg.save("nested_tasklets.sdfg")
    ExplicitVectorizationPipelineCPU(vector_width=8).apply_pass(copy_sdfg, {})
    copy_sdfg.save("nested_tasklets_vectorized.sdfg")

    c_sdfg = sdfg.compile()
    c_copy_sdfg = copy_sdfg.compile()

    c_sdfg(a=A_orig, b=B_orig, c=C_orig, d=D_orig, e=E_orig, f=F_orig, S=_S)
    c_copy_sdfg(a=A_vec, b=B_vec, c=C_vec, d=D_vec, e=E_vec, f=F_vec, S=_S)

    # Compare results
    assert numpy.allclose(A_orig, A_vec), f"{A_orig - A_vec}"
    assert numpy.allclose(B_orig, B_vec), f"{B_orig - B_vec}"
    assert numpy.allclose(C_orig, C_vec), f"{C_orig - C_vec}"
    assert numpy.allclose(D_orig, D_vec), f"{D_orig - D_vec}"
    assert numpy.allclose(E_orig, E_vec), f"{E_orig - E_vec}"
    assert numpy.allclose(F_orig, F_vec), f"{F_orig - F_vec}"


def _dense_to_csr(dense: numpy.ndarray):
    """
    Convert a 2D dense numpy array to CSR arrays (data, indices, indptr).
    Keeps the same ordering usually used by CSR: row-major.
    """
    data = []
    indices = []
    indptr = [0]
    nrows, ncols = dense.shape
    for i in range(nrows):
        row_nnz = 0
        for j in range(ncols):
            v = dense[i, j]
            if v != 0:
                data.append(v)
                indices.append(j)
                row_nnz += 1
        indptr.append(indptr[-1] + row_nnz)
    return numpy.array(data, dtype=dense.dtype), numpy.array(indices, dtype=numpy.int64), numpy.array(indptr,
                                                                                                      dtype=numpy.int64)


def trim_to_multiple_of_8(dense: numpy.ndarray) -> numpy.ndarray:
    """
    For each row in the dense matrix, drop (set to zero) the last few nonzeros
    so that the number of nonzeros becomes a multiple of 8.
    """
    A = dense.copy()
    for i in range(A.shape[0]):
        nz_idx = numpy.flatnonzero(A[i])
        excess = len(nz_idx) % 8
        if excess:
            # zero out the last 'excess' nonzeros
            A[i, nz_idx[-excess:]] = 0
    return A


def test_spmv():
    _N = 32
    density = 0.25
    dense = numpy.random.random((_N, _N))
    mask = numpy.random.random((_N, _N)) < density
    dense = dense * mask  # many zeros

    # Create CSR arrays (data, indices, indptr)
    dense = trim_to_multiple_of_8(dense)
    data, indices, indptr = _dense_to_csr(dense)

    # input / output vectors
    x = numpy.random.random((_N, ))
    y_orig = numpy.zeros_like(x)
    y_vec = numpy.zeros_like(x)
    _nnz = len(data)

    # Original SDFG
    sdfg = spmv_csr.to_sdfg()

    # Vectorized SDFG
    copy_sdfg = copy.deepcopy(sdfg)
    sdfg.save("spmv.sdfg")
    ExplicitVectorizationPipelineCPU(vector_width=8).apply_pass(copy_sdfg, {})
    copy_sdfg.save("auto_vectorized_spmv.sdfg")

    c_sdfg = sdfg.compile()
    c_copy_sdfg = copy_sdfg.compile()

    c_sdfg(data=data, indices=indices, indptr=indptr, x=x, y=y_orig, n=_N, nnz=_nnz)
    c_copy_sdfg(data=data, indices=indices, indptr=indptr, x=x, y=y_vec, n=_N, nnz=_nnz)

    # Compare results
    assert numpy.allclose(y_orig, y_vec), f"{y_orig - y_vec}"


if __name__ == "__main__":
    test_tasklets_in_if()
    test_tasklets_in_if_two()
    #test_nested_sdfg()
    #test_simple_cpu()
    #test_no_maps()
    #test_spmv()
