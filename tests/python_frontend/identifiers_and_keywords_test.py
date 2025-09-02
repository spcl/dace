# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np
import pytest
from typing import Optional

N = dace.symbol('N')


@dace.program
def keyword_false(A: dace.float32[N], B: dace.float32[N], C: dace.bool):
    if C is False:
        B[:] = A[:]


def test_keyword_false():
    N = 128
    A = np.random.rand(N).astype(np.float32)
    B = np.zeros((N, ), dtype=np.float32)
    C = False
    keyword_false(A, B, C)
    assert np.allclose(A, B)


@dace.program
def keyword_none(A: dace.float32[N], B: dace.float32[N], C: Optional[dace.int32[20]]):
    if C is None:
        B[:] = A[:]


def test_keyword_none():
    N = 128
    A = np.random.rand(N).astype(np.float32)
    B = np.zeros((N, ), dtype=np.float32)
    C = None
    keyword_none(A, B, C)
    assert np.allclose(A, B)


@dace.program
def keyword_true(A: dace.float32[N], B: dace.float32[N], C: dace.bool):
    if C is True:
        B[:] = A[:]


def test_keyword_true():
    N = 128
    A = np.random.rand(N).astype(np.float32)
    B = np.zeros((N, ), dtype=np.float32)
    C = True
    keyword_true(A, B, C)
    assert np.allclose(A, B)


@dace.program
def keyword_and(A: dace.float32[N], B: dace.float32[N], C: dace.bool, D: dace.bool):
    if C and D:
        B[:] = A[:]


def test_keyword_and():
    N = 128
    A = np.random.rand(N).astype(np.float32)
    B = np.zeros((N, ), dtype=np.float32)
    C = True
    D = True
    keyword_and(A, B, C, D)
    assert np.allclose(A, B)


@dace.program
def keyword_assert(A: dace.float32[N], B: dace.float32[N], C: dace.bool, D: dace.bool):
    with C as A:
        from dace import symbolic
        a = 5
        del a
        assert (C == True)
        if C and D:
            B[:] = A[:]


def test_keyword_assert():
    N = 128
    A = np.random.rand(N).astype(np.float32)
    B = np.zeros((N, ), dtype=np.float32)
    C = True
    D = True
    with pytest.raises(Exception):
        keyword_assert(A, B, C, D)


@dace.program
def keyword_ifelse(A: dace.float32[N], B: dace.float32[N], C: dace.int32):
    if C == 0:
        B[:] = -A[:]
    elif C == 1:
        B[:] = A[:] * A[:]
    else:
        B[:] = A


def test_keyword_ifelse():
    N = 128
    A = np.random.rand(N).astype(np.float32)
    B = np.zeros((N, ), dtype=np.float32)
    C = np.int32(2)
    keyword_ifelse(A, B, C)
    assert np.allclose(A, B)


@dace.program
def keyword_for(A: dace.float32[N], B: dace.float32[N]):
    for i in range(N):
        B[i] = A[i]


def test_keyword_for():
    N = 128
    A = np.random.rand(N).astype(np.float32)
    B = np.zeros((N, ), dtype=np.float32)
    keyword_for(A, B)
    assert np.allclose(A, B)


@dace.program
def keyword_while(A: dace.float32[N], B: dace.float32[N]):
    i = dace.define_local_scalar(dtype=dace.int32)
    i = 0
    while True:
        B[i] = A[i] + i - i
        i += 1
        if i < N:
            continue
        else:
            break


def test_keyword_while():
    N = 128
    A = np.random.rand(N).astype(np.float32)
    B = np.zeros((N, ), dtype=np.float32)
    keyword_while(A, B)
    assert np.allclose(A, B)


@dace.program
def keyword_return(A: dace.float32[N]):
    i = dace.define_local_scalar(dtype=dace.int32)
    i = 0
    B = dace.define_local((N, ), dtype=dace.float32)
    while True:
        B[i] = A[i] + i - i
        i += 1
        if i < N:
            continue
        else:
            break
    return B


def test_keyword_return():
    N = 128
    A = np.random.rand(N).astype(np.float32)
    B = np.zeros((N, ), dtype=np.float32)
    B[:] = keyword_return(A)
    assert np.allclose(A, B)


@dace.program
def keyword_notor(A: dace.float32[N], B: dace.float32[N], C: dace.bool, D: dace.bool):
    if not C or D:
        B[:] = A[:]


def test_keyword_notor():
    N = 128
    A = np.random.rand(N).astype(np.float32)
    B = np.zeros((N, ), dtype=np.float32)
    C = False
    D = True
    keyword_notor(A, B, C, D)
    assert np.allclose(A, B)


@dace.program
def keyword_lambda(A: dace.float32[N], B: dace.float32[N]):
    x = lambda a: a
    for i in range(N):
        B[i] = x(A[i])


def test_keyword_lambda():
    N = 128
    A = np.random.rand(N).astype(np.float32)
    B = np.zeros((N, ), dtype=np.float32)
    with pytest.raises(Exception):
        keyword_lambda(A, B)


if __name__ == "__main__":
    test_keyword_false()
    test_keyword_none()
    test_keyword_true()
    test_keyword_and()
    test_keyword_assert()
    test_keyword_ifelse()
    test_keyword_for()
    test_keyword_while()
    test_keyword_return()
    test_keyword_notor()
    test_keyword_lambda()
