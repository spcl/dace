import numpy as np
import dace
from dace.transformation.dataflow import MapFusion


@dace.program
def fusion(A: dace.float32[10, 20], B: dace.float32[10, 20], out: dace.float32[1]):
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
            #c >> out(1, lambda a, b: a + b)[0]
            c = a + b

    for m, n in dace.map[0:10, 0:20]:
        with dace.tasklet:
            a << tmp_2[m, n]
            b >> out(1, lambda a, b: a + b)[0]
            b = a


if __name__ == '__main__':
    sdfg = fusion.to_sdfg()
    sdfg.apply_transformations([MapFusion])
    sdfg.draw_to_file(filename='after.dot')

    A = np.random.rand(10, 20).astype(np.float32)
    B = np.random.rand(10, 20).astype(np.float32)
    out = np.zeros(shape=1, dtype=np.float32)
    sdfg(A=A, B=B, out=out)

    diff = np.sum(A * A + B) - out
    print('Difference:', diff)
    if diff > 1e-5:
        exit(1)
