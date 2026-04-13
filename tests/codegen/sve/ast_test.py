# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
from dace.codegen.targets.sve.util import NotSupportedError
import dace
import dace.dtypes
from tests.codegen.sve.common import get_code
import pytest
from dace.codegen.targets.sve.type_compatibility import IncompatibleTypeError

N = dace.symbol('N')
M = dace.symbol('M')


def test_assign_scalar():

    @dace.program
    def program(A: dace.float32[N], B: dace.float32[N]):
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

    @dace.program
    def program(A: dace.float64[N], B: dace.float64[N]):
        for i in dace.map[0:N]:
            with dace.tasklet:
                a << A[:]
                b >> B[i]
                b = a

    # Assigning a pointer to a vector is bad!
    with pytest.raises(NotSupportedError):
        get_code(program)


def test_compare_scalar_vector():

    @dace.program
    def program(A: dace.float64[N], B: dace.float64[N]):
        for i in dace.map[0:N]:
            with dace.tasklet:
                a << A[i]
                b >> B[i]
                b = a if 0.0 < a else a * 2.0

    code = get_code(program)

    assert 'svcmplt' in code


def test_if_block():

    @dace.program
    def program(A: dace.float64[N], B: dace.float64[N]):
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

    @dace.program
    def program(A: dace.float64[N], B: dace.float64[N]):
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

    @dace.program
    def program(A: dace.float64[N], B: dace.float64[N]):
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

    @dace.program
    def program(A: dace.float64[N], B: dace.float64[N]):
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
