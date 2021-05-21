# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
from tests.codegen.sve.common import get_code
import pytest
from dace.codegen.targets.sve.util import NotSupportedError

N = dace.symbol('N')


def test_wcr_sum():
    @dace.program(dace.float64[N], dace.float64[1])
    def program(A, B):
        for i in dace.map[0:N]:
            with dace.tasklet:
                a << A[i]
                b >> B(1, lambda x, y: x + y)[0]
                b = a

    code = get_code(program, 'i')

    assert 'ReductionType::Sum' in code
    assert 'svaddv(__pg_i, b)' in code


def test_wcr_min():
    @dace.program(dace.float64[N], dace.float64[1])
    def program(A, B):
        for i in dace.map[0:N]:
            with dace.tasklet:
                a << A[i]
                b >> B(1, lambda x, y: min(x, y))[0]
                b = a

    code = get_code(program, 'i')

    assert 'ReductionType::Min' in code
    assert 'svminv(__pg_i, b)' in code


def test_wcr_max():
    @dace.program(dace.float64[N], dace.float64[1])
    def program(A, B):
        for i in dace.map[0:N]:
            with dace.tasklet:
                a << A[i]
                b >> B(1, lambda x, y: max(x, y))[0]
                b = a

    code = get_code(program, 'i')

    assert 'ReductionType::Max' in code
    assert 'svmaxv(__pg_i, b)' in code


def test_wcr_unsupported():
    # Product reduction not supported
    @dace.program(dace.float64[N], dace.float64[N])
    def program_prod(A, B):
        for i in dace.map[0:N]:
            with dace.tasklet:
                a << A[i]
                b >> B(1, lambda x, y: x * y)[i]
                b = a

    with pytest.raises(NotSupportedError):
        get_code(program_prod, 'i')

    # WCR using the SVE param is unvectorizable (no atomic bulk store)
    @dace.program(dace.float64[N], dace.float64[N])
    def program_wcr_sve_param(A, B):
        for i in dace.map[0:N]:
            with dace.tasklet:
                a << A[i]
                b >> B(1, lambda x, y: x + y)[i]
                b = a

    with pytest.raises(NotSupportedError):
        get_code(program_wcr_sve_param, 'i')
