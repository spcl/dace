import dace
from dace.transformation import optimizer
from dace.transformation.dataflow import GPUTransformMap
import numpy as np


@dace.program
def prog1(A: dace.float32[32], B: dace.float32[32]):
    @dace.map
    def work1(i: _[0:32]):
        a << A[i]
        b >> B[i]
        b = a * 2.0


@dace.program
def prog2(A: dace.float32[32], B: dace.float32[32]):
    @dace.map
    def work2(i: _[0:32]):
        a << A[i]
        b >> B[i]
        b = a / 2.0


######################################
if __name__ == '__main__':
    print('Multi-program CUDA test')

    A = np.random.rand(32).astype(np.float32)
    B = np.random.rand(32).astype(np.float32)
    C = np.random.rand(32).astype(np.float32)

    s1 = prog1.to_sdfg()
    s1.apply_transformations(GPUTransformMap)

    s2 = prog2.to_sdfg()
    s2.apply_transformations(GPUTransformMap)

    s1func = s1.compile(optimizer='')
    s2func = s2.compile(optimizer='')

    s1func(A=A, B=B)
    s2func(A=B, B=C)

    diff = np.linalg.norm(A - C)

    print('Difference:', diff)
    exit(0 if diff <= 1e-5 else 1)
