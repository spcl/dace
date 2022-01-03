# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np

N = dace.symbol('N')


@dace.program
def keyword_false(A: dace.float32[N], B: dace.float32[N], C: dace.bool):
    if C is False:
        B[:] = A[:]


def test_keyword_false():
    N.set(128)
    A = np.random.rand(N.get()).astype(np.float32)
    B = np.zeros((N.get(), ), dtype=np.float32)
    C = False
    try:
        keyword_false(A, B, C)
    except Exception as e:
        print(e)
        return False
    assert np.allclose(A, B)


@dace.program
def keyword_none(A: dace.float32[N], B: dace.float32[N], C: dace.pointer(dace.int32)):
    if C is None:
        B[:] = A[:]


def test_keyword_none():
    N.set(128)
    A = np.random.rand(N.get()).astype(np.float32)
    B = np.zeros((N.get(), ), dtype=np.float32)
    C = None
    try:
        keyword_none(A, B, C)
    except Exception as e:
        print(e)
        return False
    assert np.allclose(A, B)


@dace.program
def keyword_true(A: dace.float32[N], B: dace.float32[N], C: dace.bool):
    if C is True:
        B[:] = A[:]


def test_keyword_true():
    N.set(128)
    A = np.random.rand(N.get()).astype(np.float32)
    B = np.zeros((N.get(), ), dtype=np.float32)
    C = True
    try:
        keyword_true(A, B, C)
    except Exception as e:
        print(e)
        return False
    assert np.allclose(A, B)


@dace.program
def keyword_and(A: dace.float32[N], B: dace.float32[N], C: dace.bool, D: dace.bool):
    if C and D:
        B[:] = A[:]


def test_keyword_and():
    N.set(128)
    A = np.random.rand(N.get()).astype(np.float32)
    B = np.zeros((N.get(), ), dtype=np.float32)
    C = True
    D = True
    try:
        keyword_and(A, B, C, D)
    except Exception as e:
        print(e)
        return False
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
    N.set(128)
    A = np.random.rand(N.get()).astype(np.float32)
    B = np.zeros((N.get(), ), dtype=np.float32)
    C = True
    D = True
    try:
        keyword_assert(A, B, C, D)
    except Exception as e:
        print(e)
        return True
    assert np.allclose(A, B)


@dace.program
def keyword_ifelse(A: dace.float32[N], B: dace.float32[N], C: dace.int32):
    if C == 0:
        B[:] = -A[:]
    elif C == 1:
        B[:] = A[:] * A[:]
    else:
        B[:] = A


def test_keyword_ifelse():
    N.set(128)
    A = np.random.rand(N.get()).astype(np.float32)
    B = np.zeros((N.get(), ), dtype=np.float32)
    C = np.int32(2)
    try:
        keyword_ifelse(A, B, C)
    except Exception as e:
        print(e)
        return False
    assert np.allclose(A, B)


@dace.program
def keyword_for(A: dace.float32[N], B: dace.float32[N]):
    for i in range(N):
        B[i] = A[i]


def test_keyword_for():
    N.set(128)
    A = np.random.rand(N.get()).astype(np.float32)
    B = np.zeros((N.get(), ), dtype=np.float32)
    try:
        keyword_for(A, B)
    except Exception as e:
        print(e)
        return False
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
    N.set(128)
    A = np.random.rand(N.get()).astype(np.float32)
    B = np.zeros((N.get(), ), dtype=np.float32)
    try:
        keyword_while(A, B)
    except Exception as e:
        print(e)
        return False
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
    N.set(128)
    A = np.random.rand(N.get()).astype(np.float32)
    B = np.zeros((N.get(), ), dtype=np.float32)
    try:
        B[:] = keyword_return(A)
    except Exception as e:
        print(e)
        return False
    assert np.allclose(A, B)


@dace.program
def keyword_notor(A: dace.float32[N], B: dace.float32[N], C: dace.bool, D: dace.bool):
    if not C or D:
        B[:] = A[:]


def test_keyword_notor():
    N.set(128)
    A = np.random.rand(N.get()).astype(np.float32)
    B = np.zeros((N.get(), ), dtype=np.float32)
    C = False
    D = True
    try:
        keyword_notor(A, B, C, D)
    except Exception as e:
        print(e)
        return False
    assert np.allclose(A, B)


@dace.program
def keyword_lambda(A: dace.float32[N], B: dace.float32[N]):
    x = lambda a: a
    for i in range(N):
        B[i] = x(A[i])


def test_keyword_lambda():
    N.set(128)
    A = np.random.rand(N.get()).astype(np.float32)
    B = np.zeros((N.get(), ), dtype=np.float32)
    try:
        keyword_lambda(A, B)
    except Exception as e:
        print(e)
        return True
    assert np.allclose(A, B)


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
