# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import numpy as np
import pytest

import dace
from dace.frontend.python.common import DaceSyntaxError

N = dace.symbol("N")


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


def test_iterator_after_loop_that_may_not_run_holds_last_value():

    @dace.program
    def maybe_empty(A: dace.int64[1]):
        for i in range(N):
            pass
        A[0] = i

    A = np.zeros(1, dtype=np.int64)
    maybe_empty(A, N=5)
    assert A[0] == 4


def test_iterator_after_loop_that_never_runs_raises():

    @dace.program
    def empty(A: dace.int64[1]):
        for i in range(3, 0):
            pass
        A[0] = i

    with pytest.raises(DaceSyntaxError, match="never runs"):
        empty.to_sdfg()


def test_iterator_of_every_loop_that_bound_the_name_is_read():

    @dace.program
    def branches(A: dace.float64[5], c: dace.int64):
        if c > 0:
            for i in range(3):
                A[i] = 1
        else:
            for i in range(4):
                A[i] = 2
        return i

    @dace.program
    def in_sequence(A: dace.float64[5], n: dace.int64):
        for i in range(3):
            A[i] = 1
        for i in range(n):
            A[i] = 2
        return i

    @dace.program
    def last_never_runs(A: dace.float64[5]):
        for i in range(3):
            A[i] = 1
        for i in range(0):
            A[i] = 2
        return i

    assert branches(np.zeros(5), 1)[0] == 2
    assert branches(np.zeros(5), 0)[0] == 3
    assert in_sequence(np.zeros(5), 0)[0] == 2
    assert last_never_runs(np.zeros(5))[0] == 2


def test_variable_reused_as_iterator_reads_symbolically():

    @dace.program
    def indexed(A: dace.float64[5]):
        x = 0
        for x in range(1, 3):
            A[x - 1] = x

    A = np.zeros(5)
    indexed(A)
    assert np.array_equal(A, [1, 2, 0, 0, 0])


def test_variable_reused_as_iterator_keeps_what_the_body_bound():

    @dace.program
    def rebound(n: dace.int64):
        x = 5
        for x in range(n):
            x = 7
        return x

    assert rebound(2)[0] == 7
    assert rebound(0)[0] == 5


def test_variable_reused_as_iterator_leaves_its_bound_alone():

    @dace.program
    def own_bound(m: dace.int64, A: dace.float64[5]):
        x = m + 2
        for x in range(x):
            A[x] = 1
        return x

    A = np.zeros(5)
    assert own_bound(1, A)[0] == 2
    assert np.array_equal(A, [1, 1, 1, 0, 0])
    assert own_bound(-5, np.zeros(5))[0] == -3


def test_view_reused_as_iterator_is_not_written_through():

    @dace.program
    def viewed(A: dace.float64[5]):
        v = A[1:3]
        for v in range(3):
            pass
        return A

    assert np.array_equal(viewed(np.arange(5.0)), np.arange(5.0))


if __name__ == "__main__":
    test_iterator_after_loop_holds_last_value()
    test_iterator_after_break_holds_break_value()
    test_iterator_after_loop_that_may_not_run_holds_last_value()
    test_iterator_after_loop_that_never_runs_raises()
    test_iterator_of_every_loop_that_bound_the_name_is_read()
    test_variable_reused_as_iterator_reads_symbolically()
    test_variable_reused_as_iterator_keeps_what_the_body_bound()
    test_variable_reused_as_iterator_leaves_its_bound_alone()
    test_view_reused_as_iterator_is_not_written_through()
