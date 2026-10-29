# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A size computed in the program (``nt = Nt + 1; np.empty(nt)``) is read into a symbol and used as a shape; the
size descriptor stays, and every shape sized from one version of it shares one symbol."""

import numpy as np
import pytest

import dace

N = dace.symbol("N")


@dace.program
def size_from_empty(a: dace.float64[N], Nt: dace.int64, out: dace.float64[N]):
    b = np.empty(Nt + 1, dace.float64)
    for i in range(N):
        b[i] = a[i] * 2.0
    for i in range(N):
        out[i] = b[i]


@dace.program
def size_from_empty_into_a_branch(a: dace.float64[N], Nt: dace.int64, out: dace.float64[N]):
    b = np.empty(Nt + 1, dace.float64)
    if Nt > 2:
        for i in range(N):
            b[i] = a[i] * 2.0
        for i in range(N):
            out[i] = b[i]
    else:
        for i in range(N):
            out[i] = 0.0


@dace.program
def calls_size_from_empty(a: dace.float64[N], Nt: dace.int64, out: dace.float64[N]):
    size_from_empty(a, Nt, out)


def test_size_descriptor_survives_its_use_as_a_shape():

    @dace.program
    def read_after_use(a: dace.float64[N], Nt: dace.int64, out: dace.float64[N]):
        m = Nt + 1
        b = np.empty(m, dace.float64)
        b[0] = 1.0
        out[:] = a + m

    @dace.program
    def reassigned_after_use(a: dace.float64[N], Nt: dace.int64, out: dace.float64[N]):
        m = Nt + 1
        b = np.empty(m, dace.float64)
        b[0] = 1.0
        m = 99
        out[:] = a + m

    a = np.arange(5, dtype=np.float64)
    for program, added in ((read_after_use, 8), (reassigned_after_use, 99)):
        out = np.zeros(5)
        program(a, np.int64(7), out, N=5)
        assert np.allclose(out, a + added)


def test_two_arrays_from_a_reassigned_size_keep_their_own_extents():

    @dace.program
    def two_arrays(Nt: dace.int64, out: dace.float64[1]):
        m = Nt
        b = np.empty(m, dace.float64)
        for i in range(64):
            b[i] = 1.0
        m = 2
        c = np.empty(m, dace.float64)
        c[0] = 0.0
        out[0] = np.sum(b)

    out = np.zeros(1)
    two_arrays(np.int64(64), out)
    assert out[0] == 64.0


def test_a_size_reused_as_an_index_does_not_rebind_the_extent():

    @dace.program
    def reused_as_index(out: dace.float64[1]):
        m = 8
        a = np.empty(m, dace.float64)
        for i in range(8):
            a[i] = i * 1.0
        m = 2
        out[0] = a[m]

    out = np.zeros(1)
    reused_as_index(out)
    assert out[0] == 2.0


@pytest.mark.parametrize("simplify", [False, True])
def test_shape_stays_correct_through_simplify(simplify):
    a = np.arange(6, dtype=np.float64)
    out = np.zeros(6)
    size_from_empty.to_sdfg(simplify=simplify)(a=a, Nt=np.int64(9), out=out, N=6)
    assert np.allclose(out, a * 2.0)


def test_a_size_one_array_is_read_through_a_subscript():

    @dace.program
    def from_size_one_array(nt: dace.int64[1], out: dace.float64[4]):
        b = np.empty(nt, dace.float64)
        b[0] = 1.0
        out[0] = b[0]

    out = np.zeros(4)
    from_size_one_array(np.array([4], dtype=np.int64), out)
    assert np.allclose(out, [1.0, 0.0, 0.0, 0.0])


@pytest.mark.parametrize("program", [size_from_empty_into_a_branch, calls_size_from_empty])
def test_a_size_assigned_into_a_region_is_defined_before_the_allocation(program):
    a = np.arange(6, dtype=np.float64)
    out = np.zeros(6)
    program(a=a, Nt=np.int64(9), out=out)
    assert np.allclose(out, a * 2.0)


def test_a_size_promoted_in_one_branch_is_not_reused_in_a_sibling():
    """The promotion edge lives in the branch that allocated first; the sibling must read the scalar again."""

    @dace.program
    def sibling_branches(Nt: dace.int64, out: dace.float64[1]):
        m = Nt + 1
        if Nt > 2:
            b = np.empty(m, dace.float64)
            b[0] = 1.0
            out[0] = b[0]
        else:
            c = np.empty(m, dace.float64)
            c[0] = 2.0
            out[0] = c[0]

    for nt, expected in ((1, 2.0), (5, 1.0)):
        out = np.zeros(1)
        sibling_branches(np.int64(nt), out)
        assert out[0] == expected


def test_two_arrays_from_one_size_share_their_extent():

    @dace.program
    def counts(p: dace.float64[N]):
        a_grid = np.zeros(9, dtype=np.int64)
        b_grid = np.zeros(16, dtype=np.int64)
        n = a_grid.size * b_grid.size
        count_c = p.size - np.ones(n, dtype=np.int64) - np.ones(n, dtype=np.int64)
        p[0] = count_c[0]

    p = np.zeros(5)
    counts(p)
    assert p[0] == 3


def test_an_array_made_in_a_loop_shares_the_extent_of_one_made_before_it():

    @dace.program
    def loop_body_array(nib: dace.int64[1], out: dace.float64[N, 2]):
        my_n = int(nib[0])
        acc = np.zeros((N, 2, my_n), dtype=np.float64)
        for ip in range(2):
            acc[:, ip, :] = np.ones((N, my_n), dtype=np.float64)
        out[:] = np.sum(acc, axis=2)

    out = np.zeros((5, 2))
    loop_body_array(np.array([3], dtype=np.int64), out)
    assert np.allclose(out, 3.0)


def test_a_slice_bounded_by_a_size_shares_its_extent():

    @dace.program
    def slice_by_size(nib: dace.int64[1], psi: dace.float64[N, N], out: dace.float64[N]):
        my_n = int(nib[0])
        tg = np.zeros((N, my_n), dtype=np.float64)
        tg[:, :] = psi[:, :my_n]
        out[:] = np.sum(tg, axis=1)

    psi = np.arange(25, dtype=np.float64).reshape(5, 5).copy()
    out = np.zeros(5)
    slice_by_size(np.array([3], dtype=np.int64), psi, out)
    assert np.allclose(out, psi[:, :3].sum(axis=1))


def test_a_shape_in_a_loop_reads_the_size_the_previous_iteration_wrote():

    @dace.program
    def size_grows_in_a_loop(out: dace.float64[4]):
        n = 2
        out[3] = np.sum(np.ones(n))
        for k in range(3):
            buf = np.ones(n)
            out[k] = np.sum(buf[:n])
            n = n + 1

    out = np.zeros(4)
    size_grows_in_a_loop(out)
    assert np.allclose(out, [2.0, 3.0, 4.0, 2.0])


if __name__ == "__main__":
    test_size_descriptor_survives_its_use_as_a_shape()
    test_two_arrays_from_a_reassigned_size_keep_their_own_extents()
    test_a_size_reused_as_an_index_does_not_rebind_the_extent()
    test_shape_stays_correct_through_simplify(False)
    test_shape_stays_correct_through_simplify(True)
    test_a_size_one_array_is_read_through_a_subscript()
    test_a_size_assigned_into_a_region_is_defined_before_the_allocation(size_from_empty_into_a_branch)
    test_a_size_assigned_into_a_region_is_defined_before_the_allocation(calls_size_from_empty)
    test_a_size_promoted_in_one_branch_is_not_reused_in_a_sibling()
    test_two_arrays_from_one_size_share_their_extent()
    test_an_array_made_in_a_loop_shares_the_extent_of_one_made_before_it()
    test_a_slice_bounded_by_a_size_shares_its_extent()
    test_a_shape_in_a_loop_reads_the_size_the_previous_iteration_wrote()
