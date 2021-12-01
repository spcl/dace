# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np
import scipy as sp
from dace.transformation.dataflow import Vectorization
from dace.transformation.dataflow.sve.vectorization import SVEVectorization
import pytest
from dace import SDFG
import dace.dtypes as dtypes

N = dace.symbol('N')
N_positive = dace.symbol('N', positive=True)
W = dace.symbol('W')
H = dace.symbol('H')
nnz = dace.symbol('nnz')


def find_connector_by_name(sdfg: SDFG, name: str):
    """
    Utility function to obtain the type of a connector by its name
    """
    for node, _ in sdfg.start_state.all_nodes_recursive():
        if name in node.in_connectors:
            return node.in_connectors[name]
        elif name in node.out_connectors:
            return node.out_connectors[name]

    raise RuntimeError(f'Could not find connector "{name}"')


@dace.program
def tovec(A: dace.float64[20]):
    return A + A


def regression(A, ratio):
    return A[np.where(A > ratio)]


@dace.program(dace.float64, dace.float64[N], dace.float64[N])
def axpy(A, X, Y):
    @dace.map(_[0:N])
    def multiplication(i):
        in_A << A
        in_X << X[i]
        in_Y << Y[i]
        out >> Y[i]

        out = in_A * in_X + in_Y


@dace.program(dace.uint32[H + 1], dace.uint32[nnz], dace.float32[nnz],
              dace.float32[W], dace.float32[H])
def spmv(A_row, A_col, A_val, x, b):
    @dace.mapscope(_[0:H])
    def compute_row(i):
        @dace.map(_[A_row[i]:A_row[i + 1]])
        def compute(j):
            a << A_val[j]
            in_x << x[A_col[j]]
            out >> b(1, lambda x, y: x + y)[i]

            out = a * in_x


@dace.program(dace.float32[N], dace.float32[N], dace.uint32[1], dace.float32)
def pbf(A, out, outsz, ratio):
    ostream = dace.define_stream(dace.float32, N)

    @dace.map(_[0:N])
    def filter(i):
        a << A[i]
        r << ratio
        b >> ostream(-1)
        osz >> outsz(-1, lambda x, y: x + y, 0)

        if a > r:
            b = a
            osz = 1

    ostream >> out


@dace.program
def tovec_sym(x: dace.float32[N], y: dace.float32[N], z: dace.float32[N]):
    @dace.map
    def sum(i: _[0:N]):
        xx << x[i]
        yy << y[i]
        zz << z[i]
        out >> z[i]

        out = xx + yy + zz


@dace.program
def tovec_uneven(A: dace.float64[N + 2]):
    for i in dace.map[1:N + 1]:
        with dace.tasklet:
            a << A[i]
            b >> A[i]
            b = a + a


@pytest.mark.sve
def test_axpy_sve():
    print("==== Program start ====")

    N.set(24)

    print('Scalar-vector multiplication %d' % (N.get()))

    # Initialize arrays: Randomize A and X, zero Y
    A = dace.float64(np.random.rand())
    X = np.random.rand(N.get()).astype(np.float64)
    Y = np.random.rand(N.get()).astype(np.float64)

    A_regression = np.float64()
    X_regression = np.ndarray([N.get()], dtype=np.float64)
    Y_regression = np.ndarray([N.get()], dtype=np.float64)
    A_regression = A
    X_regression[:] = X[:]
    Y_regression[:] = Y[:]

    sdfg = axpy.to_sdfg(strict=True)
    assert sdfg.apply_transformations(SVEVectorization) == 1

    sdfg(A=A, X=X, Y=Y, N=N)

    c_axpy = sp.linalg.blas.get_blas_funcs('axpy',
                                           arrays=(X_regression, Y_regression))
    if dace.Config.get_bool('profiling'):
        dace.timethis('axpy', 'BLAS', (2 * N.get()), c_axpy, X_regression,
                      Y_regression, N.get(), A_regression)
    else:
        c_axpy(X_regression, Y_regression, N.get(), A_regression)

    diff = np.linalg.norm(Y_regression - Y) / N.get()
    print("Difference:", diff)
    print("==== Program end ====")
    assert diff <= 1e-5


@pytest.mark.sve
def test_filter_sve():
    N_positive.set(64)
    ratio = np.float32(0.5)

    print('Predicate-Based Filter. size=%d, ratio=%f' %
          (N_positive.get(), ratio))

    A = np.random.rand(N_positive.get()).astype(np.float32)
    B = np.zeros_like(A)
    outsize = dace.scalar(dace.uint32)
    outsize[0] = 0

    sdfg = pbf.to_sdfg(strict=True)
    assert sdfg.apply_transformations(SVEVectorization) == 1

    sdfg(A=A, out=B, outsz=outsize, ratio=ratio, N=N_positive)

    if dace.Config.get_bool('profiling'):
        dace.timethis('filter', 'numpy', 0, regression, A, ratio)

    filtered = regression(A, ratio)

    if len(filtered) != outsize[0]:
        print(
            "Difference in number of filtered items: %d (DaCe) vs. %d (numpy)" %
            (outsize[0], len(filtered)))
        totalitems = min(outsize[0], N_positive.get())
        print('DaCe:', B[:totalitems].view(type=np.ndarray))
        print('Regression:', filtered.view(type=np.ndarray))
        exit(1)

    # Sort the outputs
    filtered = np.sort(filtered)
    B[:outsize[0]] = np.sort(B[:outsize[0]])

    if len(filtered) == 0:
        print("==== Program end ====")
        exit(0)

    diff = np.linalg.norm(filtered - B[:outsize[0]]) / float(outsize[0])
    print("Difference:", diff)
    if diff > 1e-5:
        totalitems = min(outsize[0], N_positive.get())
        print('DaCe:', B[:totalitems].view(type=np.ndarray))
        print('Regression:', filtered.view(type=np.ndarray))

    print("==== Program end ====")
    assert diff <= 1e-5


@pytest.mark.sve
def test_spmv_sve():
    W.set(64)
    H.set(64)
    nnz.set(640)

    print('Sparse Matrix-Vector Multiplication %dx%d (%d non-zero elements)' %
          (W.get(), H.get(), nnz.get()))

    A_row = dace.ndarray([H + 1], dtype=dace.uint32)
    A_col = dace.ndarray([nnz], dtype=dace.uint32)
    A_val = dace.ndarray([nnz], dtype=dace.float32)

    x = dace.ndarray([W], dace.float32)
    b = dace.ndarray([H], dace.float32)

    # Assuming uniform sparsity distribution across rows
    nnz_per_row = nnz.get() // H.get()
    nnz_last_row = nnz_per_row + (nnz.get() % H.get())
    if nnz_last_row > W.get():
        print('Too many nonzeros per row')
        exit(1)

    # RANDOMIZE SPARSE MATRIX
    A_row[0] = dace.uint32(0)
    A_row[1:H.get()] = dace.uint32(nnz_per_row)
    A_row[-1] = dace.uint32(nnz_last_row)
    A_row = np.cumsum(A_row, dtype=np.uint32)

    # Fill column data
    for i in range(H.get() - 1):
        A_col[nnz_per_row*i:nnz_per_row*(i+1)] = \
            np.sort(np.random.choice(W.get(), nnz_per_row, replace=False))
    # Fill column data for last row
    A_col[nnz_per_row * (H.get() - 1):] = np.sort(
        np.random.choice(W.get(), nnz_last_row, replace=False))

    A_val[:] = np.random.rand(nnz.get()).astype(dace.float32.type)
    #########################

    x[:] = np.random.rand(W.get()).astype(dace.float32.type)
    b[:] = dace.float32(0)

    # Setup regression
    A_sparse = sp.sparse.csr_matrix((A_val, A_col, A_row),
                                    shape=(H.get(), W.get()))

    sdfg = spmv.to_sdfg(strict=True)
    assert sdfg.apply_transformations(SVEVectorization) == 1

    sdfg(A_row=A_row, A_col=A_col, A_val=A_val, x=x, b=b, H=H, W=W, nnz=nnz)

    if dace.Config.get_bool('profiling'):
        dace.timethis('spmv', 'scipy', 0, A_sparse.dot, x)

    diff = np.linalg.norm(A_sparse.dot(x) - b) / float(H.get())
    print("Difference:", diff)
    print("==== Program end ====")
    assert diff <= 1e-5


def test_basic_stride_sve():
    @dace.program
    def program(A: dace.float32[N], B: dace.float32[N]):
        for i in dace.map[0:N]:
            with dace.tasklet:
                a << A[i]
                b >> B[i]
                b = a

    sdfg = program.to_sdfg(strict=True)
    assert sdfg.apply_transformations(SVEVectorization) == 1


def test_irregular_stride_sve():
    @dace.program
    def program(A: dace.float32[N], B: dace.float32[N]):
        for i in dace.map[0:N * N]:
            with dace.tasklet:
                a << A[i * i]
                b >> B[i * i]
                b = a

    sdfg = program.to_sdfg(strict=True)
    # [i * i] has a stride of 2i + 1 which is not constant (cannot be vectorized)
    assert sdfg.apply_transformations(SVEVectorization) == 0


def test_diagonal_stride_sve():
    @dace.program
    def program(A: dace.float32[N, N], B: dace.float32[N, N]):
        for i in dace.map[0:N]:
            with dace.tasklet:
                a << A[i, i]
                b >> B[i, i]
                b = a

    sdfg = program.to_sdfg(strict=True)
    # [i, i] has a stride of N + 1, so it is perfectly fine
    assert sdfg.apply_transformations(SVEVectorization) == 1


def test_unsupported_type_sve():
    @dace.program
    def program(A: dace.complex64[N], B: dace.complex64[N]):
        for i in dace.map[0:N]:
            with dace.tasklet:
                a << A[i]
                b >> B[i]
                b = a

    sdfg = program.to_sdfg(strict=True)
    # Complex datatypes are currently not supported by the codegen
    assert sdfg.apply_transformations(SVEVectorization) == 0


def test_supported_wcr_sve():
    @dace.program
    def program(A: dace.float32[N], B: dace.int32[1]):
        for i in dace.map[0:N]:
            with dace.tasklet:
                a << A[i]
                b >> B(-1, lambda x, y: x + y)[0]
                b = a

    sdfg = program.to_sdfg(strict=True)
    # Complex datatypes are currently not supported by the codegen
    assert sdfg.apply_transformations(SVEVectorization) == 1


def test_first_level_vectorization_sve():
    @dace.program
    def program(A: dace.float32[N], B: dace.float32[N]):
        for i, j in dace.map[0:N, 0:N]:
            with dace.tasklet:
                a_scal << A[i]
                a_vec << A[j]
                b >> B[j]
                b = a_vec

    sdfg = program.to_sdfg(strict=True)
    sdfg.apply_transformations(SVEVectorization)

    # i is constant in the vectorized map
    assert not isinstance(find_connector_by_name(sdfg, 'a_scal'), dtypes.vector)
    # j is the innermost param
    assert isinstance(find_connector_by_name(sdfg, 'a_vec'), dtypes.vector)


def test_stream_push_sve():
    @dace.program(dace.float32[N], dace.float32[N])
    def program(A, B):
        S_out = dace.define_stream(dace.float32, N)
        for i in dace.map[0:N]:
            with dace.tasklet:
                a << A[i]
                b >> S_out(-1)
                b = a
        S_out >> B

    sdfg = program.to_sdfg(strict=True)
    # Stream push is possible
    assert sdfg.apply_transformations(SVEVectorization) == 1


def test_stream_pop_sve():
    @dace.program(dace.float32[N], dace.float32[N])
    def program(A, B):
        S_in = dace.define_stream(dace.float32, N)
        S_in << A
        for i in dace.map[0:N]:
            with dace.tasklet:
                a << S_in(-1)
                b >> B[i]
                b = a

    sdfg = program.to_sdfg(strict=True)
    # Stream pop is not implemented yet
    assert sdfg.apply_transformations(SVEVectorization) == 0


def test_vectorization():
    sdfg: dace.SDFG = tovec.to_sdfg()
    assert sdfg.apply_transformations(Vectorization, options={'vector_len':
                                                              2}) == 1
    assert 'vec<double, 2>' in sdfg.generate_code()[0].code
    A = np.random.rand(20)
    B = sdfg(A=A)
    assert np.allclose(B, A * 2)


def test_vectorization_uneven():
    sdfg: dace.SDFG = tovec_uneven.to_sdfg()

    A = np.ones([22], np.float64)
    result = np.array([1.] + [2.] * 20 + [1.], dtype=np.float64)
    sdfg(A=A, N=20)
    assert np.allclose(A, result)

    sdfg.apply_strict_transformations()
    assert sdfg.apply_transformations(Vectorization, options={'vector_len':
                                                              2}) == 1
    assert 'vec<double, 2>' in sdfg.generate_code()[0].code

    A = np.ones([22], np.float64)
    sdfg(A=A, N=20)
    assert np.allclose(A, result)


def test_vectorization_postamble():
    sdfg: dace.SDFG = tovec_sym.to_sdfg()
    sdfg.apply_strict_transformations()
    assert sdfg.apply_transformations(Vectorization) == 1
    assert 'vec<float, 4>' in sdfg.generate_code()[0].code
    csdfg = sdfg.compile()

    for N in range(24, 29):
        x = np.random.rand(N).astype(np.float32)
        y = np.random.rand(N).astype(np.float32)
        z = np.random.rand(N).astype(np.float32)
        expected = x + y + z

        csdfg(x=x, y=y, z=z, N=N)
        assert np.allclose(z, expected)


def test_propagate_parent():
    sdfg: dace.SDFG = tovec.to_sdfg()
    assert sdfg.apply_transformations(Vectorization,
                                      options={
                                          'vector_len': 2,
                                          'propagate_parent': True
                                      }) == 1
    assert 'vec<double, 2>' in sdfg.generate_code()[0].code
    A = np.random.rand(20)
    B = sdfg(A=A)
    assert np.allclose(B.reshape(20), A * 2)


if __name__ == '__main__':
    # test_axpy_sve()
    # test_filter_sve()
    # test_spmv_sve()

    # test_basic_stride_sve()
    # test_irregular_stride_sve()
    # test_diagonal_stride_sve()
    # test_unsupported_type_sve()
    # test_supported_wcr_sve()
    test_first_level_vectorization_sve()
    # test_stream_push_sve()
    # test_stream_pop_sve()



    # test_vectorization()
    # test_vectorization_uneven()
    # test_vectorization_postamble()
    # test_propagate_parent()
