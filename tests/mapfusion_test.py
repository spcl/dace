import numpy as np
import os
import dace
from dace.transformation.dataflow import MapFusion


@dace.program
def fusion(A: dace.float32[10, 20], B: dace.float32[10, 20],
           out: dace.float32[1]):
    tmp = dace.define_local([10, 20], dtype=A.dtype)
    tmp_2 = dace.define_local([10, 20], dtype=A.dtype)
    for i, j in dace.map[0:10, 0:20]:
        with dace.tasklet:
            a << A[i, j]
            b >> tmp[i, j]

            b = a * a

    for k, l in dace.map[0:20, 0:10]:
        with dace.tasklet:
            a << tmp[l, k]
            b << B[l, k]
            c >> tmp_2[l, k]

            c = a + b

    for m, n in dace.map[0:10, 0:20]:
        with dace.tasklet:
            a << tmp_2[m, n]
            b >> out(1, lambda a, b: a + b)[0]

            b = a


@dace.program
def multiple_fusions(A: dace.float32[10, 20], B: dace.float32[10, 20],
                     C: dace.float32[10, 20], out: dace.float32[1]):
    A_prime = dace.define_local([10, 20], dtype=A.dtype)
    A_prime_copy = dace.define_local([10, 20], dtype=A.dtype)
    for i, j in dace.map[0:10, 0:20]:
        with dace.tasklet:
            inp << A[i, j]
            out1 >> out(1, lambda a, b: a + b)[0]
            out2 >> A_prime[i, j]
            out3 >> A_prime_copy[i, j]
            out1 = inp
            out2 = inp * inp
            out3 = inp * inp

    for i, j in dace.map[0:10, 0:20]:
        with dace.tasklet:
            inp << A_prime[i, j]
            out1 >> B[i, j]
            out1 = inp + 1

    for i, j in dace.map[0:10, 0:20]:
        with dace.tasklet:
            inp << A_prime_copy[i, j]
            out2 >> C[i, j]
            out2 = inp + 2


if __name__ == '__main__':
    sdfg = fusion.to_sdfg()
    sdfg.apply_transformations([MapFusion])
    sdfg.save(os.path.join('_dotgraphs', 'after.sdfg'))

    A = np.random.rand(10, 20).astype(np.float32)
    B = np.random.rand(10, 20).astype(np.float32)
    out = np.zeros(shape=1, dtype=np.float32)
    sdfg(A=A, B=B, out=out)

    diff = np.sum(A * A + B) - out
    print('Difference:', diff)
    if diff > 1e-4:
        exit(1)

    # Second test
    sdfg = multiple_fusions.to_sdfg()
    sdfg.apply_transformations([MapFusion])
    sdfg.save(os.path.join('_dotgraphs', 'after.sdfg'))
    A = np.random.rand(10, 20).astype(np.float32)
    B = np.zeros_like(A)
    C = np.zeros_like(A)
    out = np.zeros(shape=1, dtype=np.float32)
    sdfg(A=A, B=B, C=C, out=out)
    diff1 = np.linalg.norm(A * A + 1 - B)
    diff2 = np.linalg.norm(A * A + 2 - C)
    print('Difference1:', diff1)
    if diff1 > 1e-4:
        exit(1)

    print('Difference2:', diff2)
    if diff2 > 1e-4:
        exit(1)
