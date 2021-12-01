# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
from dace.codegen.targets.sve.util import NotSupportedError
import dace
import dace.dtypes
import pytest
from dace.transformation.dataflow.sve.vectorization import SVEVectorization

N = dace.symbol('N')
M = dace.symbol('M')


def vectorize(program):
    sdfg = program.to_sdfg(strict=True)
    sdfg.apply_transformations(SVEVectorization)
    return sdfg


def get_code(program):
    return vectorize(program).generate_code()[0].clean_code


def test_assign_scalar():
    @dace.program(dace.float32[N], dace.float32[N])
    def program(A, B):
        for i in dace.map[0:N]:
            with dace.tasklet:
                a << A[i]
                b >> B[i]
                b = 0.0

    code = get_code(program)

    # Scalar must be duplicated and brought into right type
    assert 'svdup_f32' in code
    assert f'({dace.float32})' in code


def test_assign_pointer():
    @dace.program(dace.float64[N], dace.float64[N])
    def program(A, B):
        for i in dace.map[0:N]:
            with dace.tasklet:
                a << A[:]
                b >> B[i]
                b = a

    # Assigning a pointer to a vector is bad!
    with pytest.raises(NotSupportedError):
        get_code(program)


def test_compare_scalar_vector():
    @dace.program(dace.float64[N], dace.float64[N])
    def program(A, B):
        for i in dace.map[0:N]:
            with dace.tasklet:
                a << A[i]
                b >> B[i]
                b = a if 0.0 < a else a * 2.0

    code = get_code(program)

    assert 'svcmplt' in code


def test_if_block():
    @dace.program(dace.float64[N], dace.float64[N])
    def program(A, B):
        for i in dace.map[0:N]:
            with dace.tasklet:
                a << A[i]
                b >> B[i]
                if a > 0:
                    b = 0
                else:
                    b *= 2

    code = get_code(program)

    # Accumulator must be used for predicates
    assert '__pg_acc' in code


def test_assign_new_variable():
    @dace.program(dace.float64[N], dace.float64[N])
    def program(A, B):
        for i in dace.map[0:N]:
            with dace.tasklet:
                a << A[i]
                b >> B[i]
                if a > 0 and a < 1:
                    c = a
                else:
                    c = 0
                b = a

    code = get_code(program)

    # c will be once defined as vector, once as scalar (locally)
    assert 'svfloat64_t c = ' in code
    assert f'{dace.int64} c = ' in code


def test_math_functions():
    @dace.program(dace.float64[N], dace.float64[N])
    def program(A, B):
        for i in dace.map[0:N]:
            with dace.tasklet:
                a << A[i]
                b >> B[i]
                b = math.max(42, a)
                b = math.sqrt(a)
                b = math.max(41, 42)

    code = get_code(program)

    # Vectorized max
    assert 'svmax' in code
    # Vectorized sqrt
    assert 'svsqrt' in code
    # Regular max (on scalars)
    assert 'dace::math::max' in code
    # Assigning scalar max to vector
    assert 'svdup' in code


def test_fused_operations():
    @dace.program(dace.float64[N], dace.float64[N])
    def program(A, B):
        for i in dace.map[0:N]:
            with dace.tasklet:
                a << A[i]
                b >> B[i]
                b = a * a + a
                b = a + a * a
                b = a * a - a
                b = a - a * a
                c = 0 * 1 + a

    code = get_code(program)

    # All fused ops
    assert 'svmad' in code
    assert 'svmla' in code
    assert 'svmls' in code
    assert 'svmsb' in code

    # No fusion if less than 2 vectors
    assert 'svadd' in code


def test_map_simple():
    # One dimensional
    @dace.program(dace.float64[N], dace.float64[N])
    def program(A, B):
        for i in dace.map[0:N]:
            with dace.tasklet:
                a << A[i]
                b >> B[i]
                b = a

    code = get_code(program)

    assert '__pg_i' in code


def test_map_advanced():
    # Multidimensional + stride
    @dace.program(dace.float64[16 * N], dace.float64[16 * N])
    def program(A, B):
        for i, j, k in dace.map[0:N, 0:N:2, 1:8 * N + 1:N * 2]:
            with dace.tasklet:
                a << A[k]
                b >> B[k]
                b = a

    code = get_code(program)

    # Only innermost should be SVE
    assert '__pg_i' not in code
    assert '__pg_j' not in code

    # Check for stride of N * 2
    assert '(2 * N)' in code

    # Offset initial
    assert 'k = 1' in code

    # Upper bound (minus 1)
    assert '(8 * N)' in code


def test_contiguous_map():
    @dace.program(dace.float64[N], dace.float64[N])
    def program(A, B):
        for i in dace.map[0:N]:
            with dace.tasklet:
                a << A[i]
                b >> B[i]
                b = a

    code = get_code(program)

    assert 'svld1(' in code
    assert 'svst1(' in code


def test_stride_map():
    @dace.program(dace.float64[N], dace.float64[N])
    def program(A, B):
        for i in dace.map[0:N:2]:
            with dace.tasklet:
                a << A[i]
                b >> B[i]
                b = a

    code = get_code(program)

    assert 'svld1_gather' in code
    assert 'svst1_scatter' in code
    assert '(0, 2)' in code


def test_fake_stride():
    @dace.program(dace.float64[N], dace.float64[N])
    def program(A, B):
        for i in dace.map[0:N:2]:
            with dace.tasklet:
                a << A[i / 2]
                b >> B[i]
                b = a

    code = get_code(program)

    # Load is contiguous even though it doesn't look like it
    assert 'svld1(' in code

    # Store is stride
    assert 'svst1_scatter' in code


def test_matrix_stride():
    @dace.program(dace.float64[N, M], dace.float64[M, N])
    def program(A, B):
        for i, j in dace.map[0:N, 0:M]:
            with dace.tasklet:
                a << A[i, j]
                b >> B[j, i]
                b = a

    code = get_code(program)

    # Contiguous load of entries
    assert 'svld1' in code
    # Stride N store
    assert 'svst1_scatter' in code
    assert '(0, N)' in code


def test_indirect_load_explicit():
    @dace.program(dace.int64[N], dace.int64[N], dace.int64[N])
    def program(A, B, C):
        for i in dace.map[0:N]:
            with dace.tasklet:
                a << A[:]
                b << B[i]
                c >> C[i]
                c = a[b]

    code = get_code(program)

    assert 'svld1_gather_index' in code


def test_indirect_load_implicit():
    @dace.program(dace.int64[N], dace.int64[N], dace.int64[N])
    def program(A, B, C):
        for i in dace.map[0:N]:
            with dace.tasklet:
                a << A[B[i]]
                c >> C[i]
                c = a

    code = get_code(program)

    # This is still an indirect load (uses Indirection tasklet)
    assert 'svld1_gather_index' in code


def test_stream_push():
    @dace.program(dace.float32[N], dace.float32[N])
    def program(A, B):
        stream = dace.define_stream(dace.float32, N)
        for i in dace.map[0:N]:
            with dace.tasklet:
                a << A[i]
                s >> stream(-1)
                s = 42.0

        stream >> B

    code = get_code(program)

    assert 'stream.push' in code
    assert 'svcompact' in code


def test_wcr_sum():
    @dace.program(dace.float64[N], dace.float64[1])
    def program(A, B):
        for i in dace.map[0:N]:
            with dace.tasklet:
                a << A[i]
                b >> B(1, lambda x, y: x + y)[0]
                b = a

    code = get_code(program)

    assert 'ReductionType::Sum' in code
    assert 'svaddv' in code


def test_wcr_min():
    @dace.program(dace.float64[N], dace.float64[1])
    def program(A, B):
        for i in dace.map[0:N]:
            with dace.tasklet:
                a << A[i]
                b >> B(1, lambda x, y: min(x, y))[0]
                b = a

    code = get_code(program)

    assert 'ReductionType::Min' in code
    assert 'svminv' in code


def test_wcr_max():
    @dace.program(dace.float64[N], dace.float64[1])
    def program(A, B):
        for i in dace.map[0:N]:
            with dace.tasklet:
                a << A[i]
                b >> B(1, lambda x, y: max(x, y))[0]
                b = a

    code = get_code(program)

    assert 'ReductionType::Max' in code
    assert 'svmaxv' in code


if __name__ == '__main__':
    test_assign_scalar()
    test_assign_pointer()
    test_compare_scalar_vector()
    test_if_block()
    test_assign_new_variable()
    test_math_functions()
    test_fused_operations()
    test_map_simple()
    test_map_advanced()
    test_contiguous_map()
    test_stride_map()
    test_fake_stride()
    test_matrix_stride()
    test_indirect_load_explicit()
    test_indirect_load_implicit()
    test_stream_push()
    test_wcr_sum()
    test_wcr_min()
    test_wcr_max()
