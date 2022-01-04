# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np

N = dace.symbol('N')

# yapf: disable


@dace.program
def comments(A: dace.float32[N], B: dace.float32[N]):
    # The DaCe program sets B equal to A
    B[:] = A[:]  # for i in 0 .. N-1; B[i] = A[i]


def test_comments():
    N.set(128)
    A = np.random.rand(N.get()).astype(np.float32)
    B = np.zeros((N.get(),), dtype=np.float32)
    comments(A, B)
    assert np.allclose(A, B)


@dace.program
def explicit_line_joining(A: dace.float32[N], B: dace.float32[N]):
    # The DaCe program sets B equal to A
    B[:] = \
        A[:]  # for i in 0 .. N-1; B[i] = A[i]


def test_explicit_line_joining():
    N.set(128)
    A = np.random.rand(N.get()).astype(np.float32)
    B = np.zeros((N.get(),), dtype=np.float32)
    explicit_line_joining(A, B)
    assert np.allclose(A, B)


@dace.program
def implicit_line_joining(A: dace.float32[N], B: dace.float32[N]):
    # The DaCe programs sets B equal to A
    tmp = dace.define_local(
        (N,),  # shape
        dtype=dace.float32  # type
    )
    tmp[:] = A[:]  # for i in 0 .. N-1; tmp[i] = A[i]
    B[:] = tmp[:]  # for i in 0 .. N-1; B[i] = tmp[i]


def test_implicit_line_joining():
    N.set(128)
    A = np.random.rand(N.get()).astype(np.float32)
    B = np.zeros((N.get(),), dtype=np.float32)
    implicit_line_joining(A, B)
    assert np.allclose(A, B)


@dace.program
def blank_lines(A: dace.float32[N], B: dace.float32[N]):

    # The DaCe programs sets B equal to A


    tmp = dace.define_local(
        (N,),  # shape



        dtype=dace.float32  # type

    )
    tmp[:] = A[:]  # for i in 0 .. N-1; tmp[i] = A[i]

    B[:] = tmp[:]  # for i in 0 .. N-1; B[i] = tmp[i]

# yapf: enable


def test_blank_lines():
    N.set(128)
    A = np.random.rand(N.get()).astype(np.float32)
    B = np.zeros((N.get(), ), dtype=np.float32)
    blank_lines(A, B)
    assert np.allclose(A, B)


if __name__ == "__main__":
    test_comments()
    test_explicit_line_joining()
    test_implicit_line_joining()
    test_blank_lines()
