# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import numpy as np
import pytest

import dace
from dace.frontend.python.common import DaceSyntaxError

N = dace.symbol('N')


def test_iterator_after_loop_holds_last_value():

    @dace.program
    def last(A: dace.int64[1]):
        for i in range(2, 10, 3):
            pass
        A[0] = i

    A = np.zeros(1, dtype=np.int64)
    last(A)
    assert A[0] == 8


def test_iterator_after_break_holds_break_value():

    @dace.program
    def broken(A: dace.int64[1]):
        for i in range(10):
            if i == 4:
                break
        A[0] = i

    A = np.zeros(1, dtype=np.int64)
    broken(A)
    assert A[0] == 4


def test_iterator_after_loop_that_may_not_run_raises():

    @dace.program
    def maybe_empty(A: dace.int64[1]):
        for i in range(N):
            pass
        A[0] = i

    with pytest.raises(DaceSyntaxError, match='may not run'):
        maybe_empty.to_sdfg()


if __name__ == '__main__':
    test_iterator_after_loop_holds_last_value()
    test_iterator_after_break_holds_break_value()
    test_iterator_after_loop_that_may_not_run_raises()
