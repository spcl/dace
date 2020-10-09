# Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
import numpy as np
import dace

M, N = 24, 24


@dace.program
def exponent(A: dace.complex64[M, N], B: dace.complex64[M, N]):
    B[:] = exp(A)


@dace.program
def sine(A: dace.complex64[M, N], B: dace.complex64[M, N]):
    B[:] = sin(A)


@dace.program
def cosine(A: dace.complex64[M, N], B: dace.complex64[M, N]):
    B[:] = cos(A)


@dace.program
def square_root(A: dace.complex64[M, N], B: dace.complex64[M, N]):
    B[:] = sqrt(A)


@dace.program
def logarithm(A: dace.complex64[M, N], B: dace.complex64[M, N]):
    B[:] = log(A)


@dace.program
def conjugate(A: dace.complex64[M, N], B: dace.complex64[M, N]):
    B[:] = conj(A)


@dace.program
def real_part(A: dace.complex64[M, N], B: dace.float32[M, N]):
    B[:] = real(A)


@dace.program
def imag_part(A: dace.complex64[M, N], B: dace.float32[M, N]):
    B[:] = imag(A)


@dace.program
def exponent_m(A: dace.complex64[M, N], B: dace.complex64[M, N]):
    for i in dace.map[0:M]:
        B[i] = exp(A[i])


@dace.program
def exponent_t(A: dace.complex64[M, N], B: dace.complex64[M, N]):
    for i, j in dace.map[0:M, 0:N]:
        B[i, j] = exp(A[i, j])


if __name__ == '__main__':
    A = np.random.rand(M, N).astype(
        np.float32) + 1j * np.random.rand(M, N).astype(np.float32)

    def validate(program, func, op, restype=None):
        if restype is None:
            restype = op.dtype
        daceB = np.zeros([M, N], dtype=restype)
        exec('{p}(op, daceB)'.format(p=program))
        numpyB = daceB.copy()
        exec('numpyB[:] = np.{f}(op)'.format(f=func))
        relerr = np.linalg.norm(numpyB - daceB) / np.linalg.norm(numpyB)
        print('Relative error:', relerr)
        assert relerr < 1e-5

    for p, f in {('exponent', 'exp'), ('sine', 'sin'), ('cosine', 'cos'),
                 ('square_root', 'sqrt'), ('logarithm', 'log'),
                 ('conjugate', 'conj')}:
        validate(p, f, A)

    for p, f in {('real_part', 'real'), ('imag_part', 'imag')}:
        validate(p, f, A, restype=np.float32)

    validate('exponent_m', 'exp', A)
    validate('exponent_t', 'exp', A)
