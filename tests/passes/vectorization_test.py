# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
import math
from typing import Tuple
import dace
import copy
import pytest
import numpy
from dace.properties import CodeBlock
from dace.sdfg.state import ConditionalBlock
from dace.transformation.interstate import branch_elimination
from dace.transformation.passes.vectorization.vectorize_cpu import VectorizeCPU
from dace.transformation.passes.vectorization.vectorize_gpu import VectorizeGPU

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
def vexp_cpu(A: dace.float64[N, N]):
    for i, j in dace.map[0:N, 0:N]:
        A[i, j] = math.exp(A[i, j])


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
        A[i, j] = math.exp(B[i, j])


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
def spmv_csr_two(indptr: dace.int64[n + 1], indices: dace.int64[nnz], data: dace.float64[nnz], x: dace.float64[m],
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
                           sdfg_name=None,
                           fuse_overlapping_loads=False,
                           insert_copies=False):
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
    sdfg.name = sdfg_name
    if simplify:
        sdfg.simplify(validate=True, validate_all=True, skip=skip_simplify or set())
    if save_sdfgs and sdfg_name:
        sdfg.save(f"{sdfg_name}.sdfg")
    c_sdfg = sdfg.compile()

    # Vectorized SDFG
    copy_sdfg = copy.deepcopy(sdfg)
    copy_sdfg.name = copy_sdfg.name + "_vectorized"

    VectorizeCPU(vector_width=vector_width, fuse_overlapping_loads=fuse_overlapping_loads,
                 insert_copies=insert_copies).apply_pass(copy_sdfg, {})

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

    return copy_sdfg


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


def test_vexp_cpu():
    N = 64
    A = numpy.random.random((N, N))

    run_vectorization_test(
        dace_func=vexp_cpu,
        arrays={
            'A': A,
        },
        params={'N': N},
        vector_width=8,
        save_sdfgs=True,
        sdfg_name="vexp_one",
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
    VectorizeCPU(vector_width=8).apply_pass(copy_sdfg, {})
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
def test_tasklets_in_if_two():
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
    copy_sdfg.name = copy_sdfg.name + "_vectorized"
    sdfg.save("nested_tasklets.sdfg")
    VectorizeCPU(vector_width=8).apply_pass(copy_sdfg, {})
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
    sdfg.save("spmv.sdfg")
    VectorizeCPU(vector_width=8).apply_pass(copy_sdfg, {})
    copy_sdfg.save("spmv_vectorized.sdfg")

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
    _S = 66
    A = numpy.random.random((_S, _S))
    B = numpy.random.random((_S, _S))

    run_vectorization_test(dace_func=jacobi2d,
                           arrays={
                               'A': A,
                               'B': B
                           },
                           params={
                               'S': _S,
                               'tsteps': 5,
                           },
                           vector_width=8,
                           save_sdfgs=True,
                           sdfg_name="jacobi2d")


def test_jacobi2d_with_fuse_overlapping_loads():
    _S = 66
    A = numpy.random.random((_S, _S))
    B = numpy.random.random((_S, _S))

    vectorized_sdfg: dace.SDFG = run_vectorization_test(dace_func=jacobi2d,
                                                        arrays={
                                                            'A': A,
                                                            'B': B
                                                        },
                                                        params={
                                                            'S': _S,
                                                            'tsteps': 5,
                                                        },
                                                        vector_width=8,
                                                        save_sdfgs=True,
                                                        sdfg_name="jacobi2d_with_fuse_overlapping_loads",
                                                        fuse_overlapping_loads=True)

    # Should have 1 access node between two maps
    inner_map_entries = {(n, g)
                         for n, g in vectorized_sdfg.all_nodes_recursive()
                         if isinstance(n, dace.nodes.MapEntry) and g.scope_dict()[n] is not None}
    for inner_map_entry, state in inner_map_entries:
        src_access_nodes = {
            ie.src
            for ie in state.in_edges(inner_map_entry) if isinstance(ie.src, dace.nodes.AccessNode)
        }

        src_src_access_nodes = set()
        for src_acc_node in src_access_nodes:
            src_src_access_nodes = src_src_access_nodes.union(
                {ie.src
                 for ie in state.in_edges(src_acc_node) if isinstance(ie.src, dace.nodes.AccessNode)})

        assert len(src_src_access_nodes
                   ) == 1, f"Excepted one access node got {len(src_src_access_nodes)}, ({src_src_access_nodes})"


@pytest.mark.parametrize("param_tuple", [(True, True), (True, False), (False, True), (False, False)])
def test_jacobi2d_with_parameters(param_tuple):
    fuse_overlapping_loads, insert_copies = param_tuple
    _S = 66
    A = numpy.random.random((_S, _S))
    B = numpy.random.random((_S, _S))

    run_vectorization_test(
        dace_func=jacobi2d,
        arrays={
            'A': A,
            'B': B
        },
        params={
            'S': _S,
            'tsteps': 5,
        },
        vector_width=8,
        save_sdfgs=True,
        sdfg_name=f"jacobi2d_with_fuse_overlapping_loads_{fuse_overlapping_loads}_with_insert_copies_{insert_copies}",
        fuse_overlapping_loads=fuse_overlapping_loads,
        insert_copies=insert_copies)


C = 32


def _get_disjoint_chain_sdfg(trivial_if: bool, fortran_layout: bool = False) -> dace.SDFG:
    sd1 = dace.SDFG("disjoint_chain")
    cb1 = ConditionalBlock("cond_if_cond_58", sdfg=sd1, parent=sd1)
    ss1 = sd1.add_state(label="pre", is_start_block=True)
    sd1.add_node(cb1, is_start_block=False)

    cfg1 = dace.ControlFlowRegion(label="cond_58_true", sdfg=sd1, parent=cb1)
    s1 = cfg1.add_state("main_1", is_start_block=True)
    cfg2 = dace.ControlFlowRegion(label="cond_58_false", sdfg=sd1, parent=cb1)
    s2 = cfg2.add_state("main_2", is_start_block=True)

    cb1.add_branch(
        condition=CodeBlock("_if_cond_58 == 1" if not trivial_if else "1 == 1"),
        branch=cfg1,
    )
    cb1.add_branch(
        condition=None,
        branch=cfg2,
    )
    for arr_name, shape in [
        ("zsolqa", (5, 5, C)),
        ("zrainaut", (C, )),
        ("zrainacc", (C, )),
        ("ztp1", (C, )),
    ]:
        sd1.add_array(arr_name, shape, dace.float64)
    sd1.add_scalar("rtt", dace.float64)
    sd1.add_symbol("_if_cond_58", dace.float64)
    sd1.add_symbol("_for_it_52", dace.int64)
    sd1.add_edge(src=ss1, dst=cb1, data=dace.InterstateEdge(assignments={
        "_if_cond_58": "ztp1[_for_it_52] <= rtt",
    }, ))

    for state, d1_access_str, zsolqa_access_str, zsolqa_access_str_rev in [
        (s1, "_for_it_52", "0,3,_for_it_52", "3,0,_for_it_52"), (s2, "_for_it_52", "0,2,_for_it_52", "2,0,_for_it_52")
    ]:
        zrainaut = state.add_access("zrainaut")
        zrainacc = state.add_access("zrainacc")
        zsolqa1 = state.add_access("zsolqa")
        zsolqa2 = state.add_access("zsolqa")
        zsolqa3 = state.add_access("zsolqa")
        zsolqa4 = state.add_access("zsolqa")
        zsolqa5 = state.add_access("zsolqa")
        for i, (tasklet_code, in1, instr1, in2, instr2, out, outstr) in enumerate([
            ("_out = _in1 + _in2", zrainaut, d1_access_str, zsolqa1, zsolqa_access_str, zsolqa2, zsolqa_access_str),
            ("_out = _in1 + _in2", zrainacc, d1_access_str, zsolqa2, zsolqa_access_str, zsolqa3, zsolqa_access_str),
            ("_out = (-_in1) + _in2", zrainaut, d1_access_str, zsolqa3, zsolqa_access_str_rev, zsolqa4,
             zsolqa_access_str_rev),
            ("_out = (-_in1) + _in2", zrainacc, d1_access_str, zsolqa4, zsolqa_access_str_rev, zsolqa5,
             zsolqa_access_str_rev),
        ]):
            t1 = state.add_tasklet("t1", {"_in1", "_in2"}, {"_out"}, tasklet_code)
            state.add_edge(in1, None, t1, "_in1", dace.memlet.Memlet(f"{in1.data}[{instr1}]"))
            state.add_edge(in2, None, t1, "_in2", dace.memlet.Memlet(f"{in2.data}[{instr2}]"))
            state.add_edge(t1, "_out", out, None, dace.memlet.Memlet(f"{out.data}[{outstr}]"))

    sd1.validate()

    sd2 = dace.SDFG("disjoin_chain_sdfg")
    p_s1 = sd2.add_state("p_s1", is_start_block=True)

    map_entry, map_exit = p_s1.add_map(name="map1", ndrange={"_for_it_52": dace.subsets.Range([(0, C - 1, 1)])})
    nsdfg = p_s1.add_nested_sdfg(sdfg=sd1,
                                 inputs={"zsolqa", "ztp1", "zrainaut", "zrainacc", "rtt"},
                                 outputs={"zsolqa"},
                                 symbol_mapping={"_for_it_52": "_for_it_52"})
    for arr_name, shape in [("zsolqa", (5, 5, C)), ("zrainaut", (C, )), ("zrainacc", (C, )), ("ztp1", (C, ))]:
        sd2.add_array(arr_name, shape, dace.float64)
    sd2.add_scalar("rtt", dace.float64)
    for input_name in {"zsolqa", "ztp1", "zrainaut", "zrainacc", "rtt"}:
        a = p_s1.add_access(input_name)
        p_s1.add_edge(a, None, map_entry, f"IN_{input_name}",
                      dace.memlet.Memlet.from_array(input_name, sd2.arrays[input_name]))
        p_s1.add_edge(map_entry, f"OUT_{input_name}", nsdfg, input_name,
                      dace.memlet.Memlet.from_array(input_name, sd2.arrays[input_name]))
        map_entry.add_in_connector(f"IN_{input_name}")
        map_entry.add_out_connector(f"OUT_{input_name}")
    for output_name in {"zsolqa"}:
        a = p_s1.add_access(output_name)
        p_s1.add_edge(map_exit, f"OUT_{output_name}", a, None,
                      dace.memlet.Memlet.from_array(output_name, sd2.arrays[output_name]))
        p_s1.add_edge(nsdfg, output_name, map_exit, f"IN_{output_name}",
                      dace.memlet.Memlet.from_array(output_name, sd2.arrays[output_name]))
        map_exit.add_in_connector(f"IN_{output_name}")
        map_exit.add_out_connector(f"OUT_{output_name}")

    nsdfg.sdfg.parent_nsdfg_node = nsdfg

    sd1.validate()
    sd2.validate()
    return sd2, p_s1


@pytest.mark.parametrize("trivial_if_demote_symbols", [(True, True), (True, False), (False, True), (False, False)])
def test_disjoint_chain_split_branch_only(trivial_if_demote_symbols: Tuple[bool, bool]):
    trivial_if, demote_symbols = trivial_if_demote_symbols
    sdfg, nsdfg_parent_state = _get_disjoint_chain_sdfg(trivial_if)
    zsolqa = numpy.random.choice([0.001, 5.0], size=(C, 5, 5))
    zrainacc = numpy.random.choice([0.001, 5.0], size=(C, ))
    zrainaut = numpy.random.choice([0.001, 5.0], size=(C, ))
    ztp1 = numpy.random.choice([3.5, 5.0], size=(C, ))
    rtt = numpy.random.choice([4.0], size=(1, ))

    sdfg.name = f"{sdfg.name}_{str(trivial_if).lower()}_{str(demote_symbols).lower()}"
    copy_sdfg = copy.deepcopy(sdfg)
    copy_sdfg.name = f"{copy_sdfg.name}_{str(trivial_if).lower()}_{str(demote_symbols).lower()}_vectorized"

    arrays = {"zsolqa": zsolqa, "zrainacc": zrainacc, "zrainaut": zrainaut, "ztp1": ztp1, "rtt": rtt[0]}

    sdfg.validate()
    sdfg.save(f"disjoint_chain_{str(trivial_if).lower()}_{str(demote_symbols).lower()}.sdfg")
    out_no_fuse = {k: v.copy() for k, v in arrays.items()}
    sdfg(**out_no_fuse)

    # Run SDFG version (with transformation)
    if trivial_if:
        from dace.transformation.passes.lift_trivial_if import LiftTrivialIf
        LiftTrivialIf().apply_pass(copy_sdfg, {})
    else:
        xform = branch_elimination.BranchElimination()
        cblocks = {n for n, g in copy_sdfg.all_nodes_recursive() if isinstance(n, ConditionalBlock)}
        assert len(cblocks) == 1
        cblock = cblocks.pop()

        xform.conditional = cblock
        xform.parent_nsdfg_state = nsdfg_parent_state
        xform.sequentialize_if_else_branch_if_disjoint_subsets(cblock.parent_graph)

    out_fused = {k: v.copy() for k, v in arrays.items()}

    vectorizer = VectorizeCPU(vector_width=8)
    vectorizer.try_to_demote_symbols_in_nsdfgs = demote_symbols
    vectorizer.apply_pass(copy_sdfg, {})
    copy_sdfg.save("disjoint_chain_vectorized.sdfg")

    copy_sdfg(**out_fused)

    for name in arrays.keys():
        numpy.testing.assert_allclose(out_no_fuse[name], out_fused[name], atol=1e-12)


if __name__ == "__main__":
    test_vsubs_cpu()
    test_vsubs_two_cpu()
    test_v_const_subs_cpu()
    test_v_const_subs_two_cpu()
    test_unsupported_op()
    test_unsupported_op_two()
    test_tasklets_in_if()
    test_tasklets_in_if_two()
    test_nested_sdfg()
    test_simple_cpu()
    test_no_maps()
    test_spmv()
    test_overlapping_access()
    test_jacobi2d()
    test_overlapping_access_same_src_and_dst()
    test_overlapping_access_same_src_and_dst_in_nestedsdfg()
    for argtuple in [(True, True), (True, False), (False, True), (False, False)]:
        test_disjoint_chain_split_branch_only(argtuple)
        test_jacobi2d_with_parameters(argtuple)
    test_jacobi2d_with_fuse_overlapping_loads()
