# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
from tests.codegen.sve.common import get_code
import pytest

N = dace.symbol('N')


def test_stream_push():

    @dace.program
    def program(A: dace.float32[N], B: dace.float32[N]):
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
