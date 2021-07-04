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

