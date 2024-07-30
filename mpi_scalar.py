
import dace
import numpy as np
from dace.transformation.dataflow import MPITransformMap
        
@dace.program
def mpi_scalar_add(A: dace.int32, B: dace.int32, C: dace.int32):
    C = A + B

if __name__ == '__main__':
    sdfg = mpi_scalar_add.to_sdfg(simplify=False)   # compiled SDFG
    sdfg.apply_transformations(MPITransformMap)
    
    A = np.int32(1)   # 1,1,1,1,...
    B = np.int32(1)   # 1,1,1,1,...
    C = np.int32(0)  # 0,0,0,0,...
    sdfg(A, B, C)