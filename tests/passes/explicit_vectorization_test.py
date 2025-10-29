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
def vsubs_cpu(A: dace.float64[N, N], B: dace.float64[N, N]):
    for i, j in dace.map[0:N, 0:N]:
        A[i, j] = A[i, j] - B[i, j]


@dace.program
def vsubs_two_cpu(A: dace.float64[N, N], B: dace.float64[N, N]):
    for i, j in dace.map[0:N, 0:N]:
        A[i, j] = B[i, j] - A[i, j]


@dace.program
def v_const_subs_cpu(A: dace.float64[N, N]):
    for i, j in dace.map[0:N, 0:N]:
        A[i, j] = A[i, j] - 1.0


@dace.program
def v_const_subs_two_cpu(A: dace.float64[N, N]):
    for i, j in dace.map[0:N, 0:N]:
        A[i, j] = 1.0 - A[i, j]


@dace.program
def unsupported_op(A: dace.float64[N, N], B: dace.float64[N, N]):
    for i, j in dace.map[0:N, 0:N]:
        A[i, j] = B[i, j] > 0.0


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
    for i in dace.map[0:S:1]:
        for j in dace.map[0:S:1]:
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
    for i in dace.map[0:S:1]:
        for j in dace.map[0:S:1]:
            if a[i, j] < b:
                e[i, j] = (c[i, j] * f * a[i, j] * 2.0) - a[i, j]


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


def run_vectorization_test(dace_func,
                           arrays,
                           params,
                           vector_width=8,
                           simplify=True,
                           skip_simplify=None,
                           save_sdfgs=False,
                           sdfg_name=None):
    """
    Run vectorization test and compare results.

    Args:
        dace_func: DaCe program function to test
        arrays: Dict of numpy arrays (will be copied internally)
        params: Dict of additional parameters to pass to compiled functions
        vector_width: Vector width for vectorization
        simplify: Whether to simplify the SDFG
        skip_simplify: Set of passes to skip during simplification
        save_sdfgs: Whether to save SDFGs to disk
        sdfg_name: Base name for saved SDFGs
    """
    # Create copies for comparison
    arrays_orig = {k: copy.deepcopy(v) for k, v in arrays.items()}
    arrays_vec = {k: copy.deepcopy(v) for k, v in arrays.items()}

    # Original SDFG
    sdfg = dace_func.to_sdfg(simplify=False)
    if simplify:
        sdfg.simplify(validate=True, validate_all=True, skip=skip_simplify or set())
    if save_sdfgs and sdfg_name:
        sdfg.save(f"{sdfg_name}.sdfg")
    c_sdfg = sdfg.compile()

    # Vectorized SDFG
    copy_sdfg = copy.deepcopy(sdfg)
    copy_sdfg.name = copy_sdfg.name + "_vectorized"

    ExplicitVectorizationPipelineCPU(vector_width=vector_width).apply_pass(copy_sdfg, {})

    if save_sdfgs and sdfg_name:
        copy_sdfg.save(f"{sdfg_name}_vectorized.sdfg")
    c_copy_sdfg = copy_sdfg.compile()

    # Run both
    c_sdfg(**arrays_orig, **params)

    c_copy_sdfg(**arrays_vec, **params)

    # Compare results
    for name in arrays.keys():
        assert numpy.allclose(arrays_orig[name], arrays_vec[name]), \
            f"{name} Diff: {arrays_orig[name] - arrays_vec[name]}"


def test_vsubs_cpu():
    N = 64
    A = numpy.random.random((N, N))
    B = numpy.random.random((N, N))

    run_vectorization_test(
        dace_func=vsubs_cpu,
        arrays={
            'A': A,
            'B': B
        },
        params={'N': N},
        vector_width=8,
        save_sdfgs=True,
        sdfg_name="vsubs_one",
    )


def test_vsubs_two_cpu():
    N = 64
    A = numpy.random.random((N, N))
    B = numpy.random.random((N, N))

    run_vectorization_test(
        dace_func=vsubs_two_cpu,
        arrays={
            'A': A,
            'B': B
        },
        params={'N': N},
        vector_width=8,
        save_sdfgs=True,
        sdfg_name="vsubs_two",
    )


def test_v_const_subs_cpu():
    N = 64
    A = numpy.random.random((N, N))

    run_vectorization_test(
        dace_func=v_const_subs_cpu,
        arrays={'A': A},
        params={'N': N},
        vector_width=8,
        save_sdfgs=True,
        sdfg_name="v_const_subs_one",
    )


def test_v_const_subs_two_cpu():
    N = 64
    A = numpy.random.random((N, N))

    run_vectorization_test(
        dace_func=v_const_subs_two_cpu,
        arrays={'A': A},
        params={'N': N},
        vector_width=8,
        sdfg_name="v_const_subs_two",
        save_sdfgs=True,
    )


def test_simple_cpu():
    A = numpy.random.random((64, 64))
    B = numpy.random.random((64, 64))

    run_vectorization_test(
        dace_func=vadds_cpu,
        arrays={
            'A': A,
            'B': B
        },
        params={'N': 64},
        vector_width=4,
        sdfg_name="simple_cpu",
        save_sdfgs=True,
    )


def test_unsupported_op():
    A = numpy.random.random((64, 64))
    B = numpy.random.random((64, 64))

    run_vectorization_test(dace_func=unsupported_op,
                           arrays={
                               'A': A,
                               'B': B
                           },
                           params={'N': 64},
                           vector_width=4,
                           skip_simplify={"ScalarToSymbolPromotion"},
                           save_sdfgs=True,
                           sdfg_name="unsupported_op")


def test_unsupported_op_two():
    A = numpy.random.random((64, 64))
    B = numpy.random.random((64, 64))

    run_vectorization_test(dace_func=unsupported_op,
                           arrays={
                               'A': A,
                               'B': B
                           },
                           params={'N': 64},
                           vector_width=4,
                           save_sdfgs=True,
                           sdfg_name="unsupported_op")


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
                           save_sdfgs=True,
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
                           save_sdfgs=True,
                           sdfg_name="no_maps")


def test_tasklets_in_if():
    _S = 64
    A = numpy.random.random((_S, _S))
    B = numpy.random.random((_S, _S))
    D = numpy.random.random((_S, _S))
    C = numpy.random.random((1, ))

    arrays_orig = {'a': copy.deepcopy(A), 'b': copy.deepcopy(B), 'c': copy.deepcopy(C), 'd': copy.deepcopy(D)}
    arrays_vec = {'a': copy.deepcopy(A), 'b': copy.deepcopy(B), 'c': copy.deepcopy(C), 'd': copy.deepcopy(D)}

    sdfg = tasklets_in_if.to_sdfg()
    copy_sdfg = copy.deepcopy(sdfg)
    sdfg.save("nested_tasklets_in_if.sdfg")
    ExplicitVectorizationPipelineCPU(vector_width=8).apply_pass(copy_sdfg, {})
    copy_sdfg.save("nested_tasklets_in_if_vectorized.sdfg")

    c_sdfg = sdfg.compile()
    c_copy_sdfg = copy_sdfg.compile()

    c_sdfg(a=arrays_orig['a'], b=arrays_orig['b'], c=arrays_orig['c'][0], d=arrays_orig['d'], S=_S)
    c_copy_sdfg(a=arrays_vec['a'], b=arrays_vec['b'], c=arrays_vec['c'][0], d=arrays_vec['d'], S=_S)

    assert numpy.allclose(arrays_orig['a'], arrays_vec['a']), f"A Diff: {arrays_orig['a'] - arrays_vec['a']}"
    assert numpy.allclose(arrays_orig['b'], arrays_vec['b']), f"B Diff: {arrays_orig['b'] - arrays_vec['b']}"
    assert numpy.allclose(arrays_orig['c'], arrays_vec['c']), f"C Diff: {arrays_orig['c'] - arrays_vec['c']}"
    assert numpy.allclose(arrays_orig['d'], arrays_vec['d']), f"D Diff: {arrays_orig['d'] - arrays_vec['d']}"


# There was a non-deterministic bug, therefore run it multiple times
@pytest.mark.parametrize("run_id", [i for i in range(8)])
def test_tasklets_in_if_two(run_id):
    _S = 64
    A = numpy.random.random((_S, _S))
    B = numpy.random.random((1, ))
    C = numpy.random.random((_S, _S))
    D = numpy.random.random((_S, _S))
    E = numpy.random.random((_S, _S))
    F = numpy.random.random((1, ))

    arrays_orig = {
        'a': copy.deepcopy(A),
        'b': copy.deepcopy(B),
        'c': copy.deepcopy(C),
        'd': copy.deepcopy(D),
        'e': copy.deepcopy(E),
        'f': copy.deepcopy(F)
    }
    arrays_vec = {
        'a': copy.deepcopy(A),
        'b': copy.deepcopy(B),
        'c': copy.deepcopy(C),
        'd': copy.deepcopy(D),
        'e': copy.deepcopy(E),
        'f': copy.deepcopy(F)
    }

    sdfg = tasklets_in_if_two.to_sdfg(simplify=False)
    sdfg.simplify(skip=["ScalarToSymbolPromotion"])
    copy_sdfg = copy.deepcopy(sdfg)
    sdfg.save("nested_tasklets.sdfg")
    ExplicitVectorizationPipelineCPU(vector_width=8).apply_pass(copy_sdfg, {})
    copy_sdfg.save("nested_tasklets_vectorized.sdfg")

    c_sdfg = sdfg.compile()
    c_copy_sdfg = copy_sdfg.compile()

    c_sdfg(a=arrays_orig['a'],
           b=arrays_orig['b'][0],
           c=arrays_orig['c'],
           d=arrays_orig['d'],
           e=arrays_orig['e'],
           f=arrays_orig['f'][0],
           S=_S)
    c_copy_sdfg(a=arrays_vec['a'],
                b=arrays_vec['b'][0],
                c=arrays_vec['c'],
                d=arrays_vec['d'],
                e=arrays_vec['e'],
                f=arrays_vec['f'][0],
                S=_S)

    assert numpy.allclose(arrays_orig['a'], arrays_vec['a']), f"{arrays_orig['a'] - arrays_vec['a']}"
    assert numpy.allclose(arrays_orig['b'], arrays_vec['b']), f"{arrays_orig['b'] - arrays_vec['b']}"
    assert numpy.allclose(arrays_orig['c'], arrays_vec['c']), f"{arrays_orig['c'] - arrays_vec['c']}"
    assert numpy.allclose(arrays_orig['d'], arrays_vec['d']), f"{arrays_orig['d'] - arrays_vec['d']}"
    assert numpy.allclose(arrays_orig['e'], arrays_vec['e']), f"{arrays_orig['e'] - arrays_vec['e']}"
    assert numpy.allclose(arrays_orig['f'], arrays_vec['f']), f"{arrays_orig['f'] - arrays_vec['f']}"


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


def test_overlapping_access():
    _S = 64
    A = numpy.random.random((2, 2, _S))
    B = numpy.random.random((_S))

    sdfg = overlapping_access.to_sdfg()
    sdfg.save("s.sdfg")

    run_vectorization_test(dace_func=overlapping_access,
                           arrays={
                               'A': A,
                               'B': B
                           },
                           params={
                               'S': _S,
                           },
                           vector_width=8,
                           save_sdfgs=True,
                           sdfg_name="overlapping_access")


def test_overlapping_access_same_src_and_dst():
    _S = 64
    A = numpy.random.random((2, 2, _S))

    sdfg = overlapping_access_same_src_and_dst.to_sdfg()
    sdfg.save("s.sdfg")

    run_vectorization_test(dace_func=overlapping_access_same_src_and_dst,
                           arrays={
                               'A': A,
                           },
                           params={
                               'S': _S,
                           },
                           vector_width=8,
                           save_sdfgs=True,
                           sdfg_name="overlapping_access_same_src_and_dst")


@pytest.mark.parametrize("run_id", [i for i in range(4)])
def test_overlapping_access_same_src_and_dst_in_nestedsdfg(run_id):
    _S = 64
    A = numpy.random.random((2, 2, _S))

    sdfg = overlapping_access_same_src_and_dst_in_nestedsdfg.to_sdfg()
    sdfg.save("s.sdfg")

    run_vectorization_test(dace_func=overlapping_access_same_src_and_dst_in_nestedsdfg,
                           arrays={
                               'A': A,
                           },
                           params={
                               'S': _S,
                               "js": 0,
                           },
                           vector_width=8,
                           save_sdfgs=True,
                           sdfg_name="overlapping_access_same_src_and_dst_in_nestedsdfg")


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


@dace.program
def jacobi2d(A: dace.float64[S, S], B: dace.float64[S, S], tsteps: dace.int64):  #, N, tsteps):
    for t in range(tsteps):
        for i, j in dace.map[0:S - 2, 0:S - 2]:
            B[i + 1, j + 1] = 0.2 * (A[i + 1, j + 1] + A[i, j + 1] + A[i + 2, j + 1] + A[i + 1, j] + A[i + 1, j + 2])

        for i, j in dace.map[0:S - 2, 0:S - 2]:
            A[i + 1, j + 1] = 0.2 * (B[i + 1, j + 1] + B[i, j + 1] + B[i + 2, j + 1] + B[i + 1, j] + B[i + 1, j + 2])


def test_jacobi2d():
    _S = 64
    A = numpy.random.random((_S, _S))
    B = numpy.random.random((_S, _S))

    sdfg = jacobi2d.to_sdfg()
    sdfg.save("s.sdfg")

    run_vectorization_test(dace_func=overlapping_access,
                           arrays={
                               'A': A,
                               'B': B
                           },
                           params={
                               'S': _S,
                           },
                           vector_width=8,
                           save_sdfgs=True,
                           sdfg_name="jacobi2d")


if __name__ == "__main__":
    test_vsubs_cpu()
    test_vsubs_two_cpu()
    test_v_const_subs_cpu()
    test_v_const_subs_two_cpu()
    test_unsupported_op()
    test_unsupported_op_two()
    test_tasklets_in_if()
    for i in range(8):
        test_tasklets_in_if_two(i)
    test_nested_sdfg()
    test_simple_cpu()
    test_no_maps()
    test_spmv()
    test_overlapping_access()
    test_jacobi2d()
    test_overlapping_access_same_src_and_dst()
    test_overlapping_access_same_src_and_dst_in_nestedsdfg()
