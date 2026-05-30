# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import copy
import dace
import pytest
import numpy
from dace.transformation.passes.vectorization.vectorize_cpu import VectorizeCPU
from tests.passes.vectorization.helpers.harness import (
    run_vectorization_test,
    N,
    S,
    S1,
    S2,
    n,
    m,
    nnz,
    _get_map_inside_nested_map,
)


@dace.program
def no_maps(A: dace.float64[N, N], B: dace.float64[N, N]):
    i = 8
    j = 7
    A[i, j] = 2.0 * A[i, j]
    B[i + 1, j + 1] = B[i, j] / 1.5


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
def tasklet_in_nested_sdfg_two(
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


def test_nested_sdfg():
    _S1 = 1
    _S2 = 65
    _S = 64
    A = numpy.random.random((_S, _S))
    B = numpy.random.random((_S, _S))

    run_vectorization_test(dace_func=tasklet_in_nested_sdfg,
                           arrays={
                               'a': A,
                               'b': B
                           },
                           params={
                               'S': _S,
                               'S1': _S1,
                               'S2': _S2,
                               'offset1': -1,
                               'offset2': -1
                           },
                           vector_width=8,
                           sdfg_name="nested_tasklets")


def test_no_maps():
    _N = 16
    A = numpy.random.random((_N, _N))
    B = numpy.random.random((_N, _N))

    run_vectorization_test(dace_func=no_maps,
                           arrays={
                               'A': A,
                               'B': B
                           },
                           params={'N': _N},
                           vector_width=8,
                           sdfg_name="no_maps")


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


@dace.program
def overlapping_access(A: dace.float64[2, 2, S], B: dace.float64[S]):
    for i in dace.map[0:S:1]:
        B[i] = A[0, 0, i] + A[1, 0, i]


@dace.program
def overlapping_access_same_src_and_dst(A: dace.float64[2, 2, S]):
    for i in dace.map[0:S:1]:
        A[0, 0, i] = A[0, 0, i] + A[1, 0, i]


@dace.program
def overlapping_access_same_src_and_dst_in_nestedsdfg(A: dace.float64[2, 2, S], js: dace.int64):
    for i in dace.map[0:S:1]:
        j = js + 1
        A[0, j, i] = A[0, j, i] + A[1, j, i]


@pytest.mark.simple  # canonical: test_overlapping_access_same_src_and_dst_in_nestedsdfg (hardest — NSDFG path)
def test_overlapping_access():
    _S = 64
    A = numpy.random.random((2, 2, _S))
    B = numpy.random.random((_S))

    run_vectorization_test(dace_func=overlapping_access,
                           arrays={
                               'A': A,
                               'B': B
                           },
                           params={
                               'S': _S,
                           },
                           vector_width=8,
                           sdfg_name="overlapping_access")


@pytest.mark.simple  # canonical: test_overlapping_access_same_src_and_dst_in_nestedsdfg (hardest — NSDFG path)
def test_overlapping_access_same_src_and_dst():
    _S = 64
    A = numpy.random.random((2, 2, _S))

    run_vectorization_test(dace_func=overlapping_access_same_src_and_dst,
                           arrays={
                               'A': A,
                           },
                           params={
                               'S': _S,
                           },
                           vector_width=8,
                           sdfg_name="overlapping_access_same_src_and_dst")


@pytest.mark.parametrize("run_id", [i for i in range(4)])
def test_overlapping_access_same_src_and_dst_in_nestedsdfg(run_id):
    _S = 64
    A = numpy.random.random((2, 2, _S))

    run_vectorization_test(dace_func=overlapping_access_same_src_and_dst_in_nestedsdfg,
                           arrays={
                               'A': A,
                           },
                           params={
                               'S': _S,
                               "js": 0,
                           },
                           vector_width=8,
                           sdfg_name=f"overlapping_access_same_src_and_dst_in_nestedsdfg_runid_{run_id}")


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
    copy_sdfg.name = sdfg.name + "_vectorized"
    VectorizeCPU(vector_width=8, fail_on_unvectorizable=True).apply_pass(copy_sdfg, {})

    c_sdfg = sdfg.compile()
    c_copy_sdfg = copy_sdfg.compile()

    c_sdfg(data=data, indices=indices, indptr=indptr, x=x, y=y_orig, n=_N, nnz=_nnz)
    c_copy_sdfg(data=data, indices=indices, indptr=indptr, x=x, y=y_vec, n=_N, nnz=_nnz)

    # Compare results
    assert numpy.allclose(y_orig, y_vec, atol=1e-12), f"{y_orig - y_vec}"


_OPT_PARAMS = [(True, True), (True, False), (False, True), (False, False)]


@pytest.mark.parametrize("opt_parameters", _OPT_PARAMS)
def test_map_inside_nested_map(opt_parameters):
    fuse_overlapping_loads, insert_copies = opt_parameters

    sdfg = _get_map_inside_nested_map()
    sdfg.name = "map_inside_nested_map"
    sdfg.validate()

    # Symbolic values requested by the user
    klon = 64
    klev = 64
    kidia = 1
    kfdia = 32

    # Map of array shapes (from the SDFG snippet): only the shape tuples matter for creating arrays
    arr_shapes = {
        "int_array": (klon, 5, 5),
        "int_array2": (klon, 5, 5),
    }

    # Create Fortran-ordered NumPy arrays
    arrays = {name: numpy.random.random(shape).astype(numpy.float64, order='F') for name, shape in arr_shapes.items()}
    # Create scalars requested
    scalars = {"kfdia": numpy.int64(kfdia), "kidia": numpy.int64(kidia), "klon": numpy.int64(klon)}

    # Quick verification display: shape and contiguity / strides
    run_vectorization_test(dace_func=sdfg,
                           from_sdfg=True,
                           arrays=arrays,
                           params=scalars,
                           vector_width=8,
                           sdfg_name=sdfg.name,
                           fuse_overlapping_loads=fuse_overlapping_loads,
                           insert_copies=insert_copies,
                           no_inline=True,
                           param_tag=f"param{_OPT_PARAMS.index(opt_parameters)}")
