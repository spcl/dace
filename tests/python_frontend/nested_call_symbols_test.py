# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Symbols crossing a call: which names a callee receives, and what a call leaves behind in it."""

import numpy as np
import sympy

import dace

N = dace.symbol("N", dace.int64, positive=True)
P = dace.symbol("P")


@dace.program
def fill(B):
    B[:] = 1


@dace.program
def set_first(B):
    B[0] = 1


@dace.program
def set_at(n: dace.compiletime, A):
    A[n] = 1


@dace.program
def sized_by(n: dace.compiletime, A):
    B = np.zeros(n + 1)
    B[0] = 2
    A[0] = B[0]


def test_a_callee_receives_the_loop_iterator_its_slice_names():

    @dace.program
    def caller(A: dace.float64[5]):
        for i in range(3):
            for j in range(i + 1, 5):
                fill(A[i:j])

    A = np.zeros(5)
    caller(A)
    assert np.array_equal(A, [1, 1, 1, 1, 0])


def test_a_compile_time_argument_stays_an_expression():

    @dace.program
    def positional(A: dace.float64[P]):
        set_at(P - 1, A)

    @dace.program
    def keyword(A: dace.float64[P]):
        set_at(n=P - 1, A=A)

    @dace.program
    def as_a_size(A: dace.float64[P]):
        sized_by(P - 1, A)

    for program, expected in ((positional, [0, 0, 0, 1]), (keyword, [0, 0, 0, 1]), (as_a_size, [2, 0, 0, 0])):
        A = np.zeros(4)
        program(A, P=4)
        assert np.array_equal(A, expected)


def test_a_call_leaves_no_iterator_in_the_callee():
    """The ranges of one loop are not facts of the next call, of this program or another."""

    @dace.program
    def callee(B):
        B[0] = 1

    @dace.program
    def two_loops(A: dace.float64[5]):
        for i in range(1, 3):
            callee(A[:i])
        for i in range(0, 3):
            callee(A[:i])

    @dace.program
    def in_a_map(A: dace.float64[5]):
        for i in dace.map[0:5]:
            callee(A[:i])

    two_loops.to_sdfg(simplify=False).validate()
    assert "i" not in callee.global_vars
    in_a_map.to_sdfg(simplify=False).validate()


def test_a_slice_with_an_undetermined_extent_leaves_the_declared_size_alone():

    @dace.program
    def caller(A: dace.float64[N]):
        for i in range(1, 3):
            set_first(A[i:])

    caller.to_sdfg(simplify=False).validate()


def test_a_size_promoted_in_a_loop_outlives_the_loop():

    @dace.program
    def caller(m: dace.int64):
        for i in range(3):
            n = m + 1
            B = np.zeros(n)
            B[0] = 1
        y = B.shape[0] + 1
        return y

    assert caller(2)[0] == 4


def test_split_of_a_symbolic_length_is_decided_by_the_facts():

    @dace.program
    def caller(A: dace.float64[2 * N]):
        first, second = np.split(A, 2)
        return second

    assert np.array_equal(caller(np.arange(6.0), N=3), [3, 4, 5])


def test_a_symbolic_default_argument_is_typed_by_the_program():

    @dace.program
    def caller(A: dace.float64[5], k=N + 1):
        A[0] = k

    A = np.zeros(5)
    caller(A, N=3)
    assert A[0] == 4


def test_a_descending_iterator_claims_no_sign_it_loses():
    """``range(3, -3, -1)`` reaches -2, so the body of the loop is not nested under ``0 < i``."""

    @dace.program
    def descending(A: dace.float64[6]):
        for i in range(3, -3, -1):
            for j in dace.map[0:2]:
                A[j] = i

    sdfg = descending.to_sdfg(simplify=False)
    nested = [n for s in sdfg.states() for n in s.nodes() if isinstance(n, dace.nodes.NestedSDFG)]
    assert nested
    zero, i = sympy.Integer(0), dace.symbolic.symbol("i")
    positive = dace.symbolic.Relation(dace.symbolic.RelationKind.LT, zero, i)
    assert not any(dace.symbolic.holds(positive, node.sdfg.facts()) for node in nested)


if __name__ == "__main__":
    test_a_callee_receives_the_loop_iterator_its_slice_names()
    test_a_compile_time_argument_stays_an_expression()
    test_a_call_leaves_no_iterator_in_the_callee()
    test_a_slice_with_an_undetermined_extent_leaves_the_declared_size_alone()
    test_a_size_promoted_in_a_loop_outlives_the_loop()
    test_split_of_a_symbolic_length_is_decided_by_the_facts()
    test_a_symbolic_default_argument_is_typed_by_the_program()
    test_a_descending_iterator_claims_no_sign_it_loses()
