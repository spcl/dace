
import dace
import numpy as np
from dace.transformation.interstate import GPUTransformSDFG

        
@dace.program
def gpu_vector_add(A: dace.int32[20], B: dace.int32[20], C: dace.int32[20]):
    for i in dace.map[0:20]:       # parallelization construct
       C[i] =  A[i] + B[i]

if __name__ == '__main__':
    sdfg = gpu_vector_add.to_sdfg(simplify=False)   # compiled SDFG
    sdfg.apply_transformations(GPUTransformSDFG)

    # call with values
    A = np.ones((20), dtype=np.int32)   # 1,1,1,1,...
    B = np.ones((20), dtype=np.int32)   # 1,1,1,1,...
    C = np.zeros((20), dtype=np.int32)  # 0,0,0,0,...
    sdfg(A, B, C)

    # ref = np.full(20, 2, dtype=np.int32)     # 2,2,2,2,...
    # assert np.array_equal(ref, C)