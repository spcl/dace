# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
from tests.sve.common import get_code

N = dace.symbol('N')


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


if __name__ == '__main__':
    test_map_simple()
    test_map_advanced()
