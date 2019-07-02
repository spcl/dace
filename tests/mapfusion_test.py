import numpy as np
import dace
from dace.transformation.dataflow import MapFusion


@dace.program
def fusion(A: dace.float32[20, 20], out: dace.float32[1]):
    tmp = dace.define_local([20, 20], dtype=A.dtype)
    for i, j in dace.map[0:20, 0:20]:
        with dace.tasklet:
            a << A[i, j]
            b >> tmp[i, j]
            b = a * a

    for k, l in dace.map[0:20, 0:20]:
        with dace.tasklet:
            a << tmp[k, l]
            b >> out(1, lambda a, b: a + b)[0]
            b = a * 2


if __name__ == '__main__':
    sdfg = fusion.to_sdfg()
    sdfg.apply_transformations([MapFusion])

    A = np.random.rand(20, 20).astype(np.float32)
    out = np.zeros(shape=1, dtype=np.float32)
    sdfg(A=A, out=out)

    diff = 2 * np.sum(A * A) - out
    print('Difference:', diff)
    if diff > 1e-5:
        exit(1)
