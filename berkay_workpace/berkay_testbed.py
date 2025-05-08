import dace
import cupy as cp
import random

from dace import registry
from dace.sdfg.scope import ScopeSubgraphView
from dace.codegen.prettycode import CodeIOStream
from dace.codegen.targets.target import TargetCodeGenerator
from dace.codegen.targets.framecode import DaCeCodeGenerator
from dace.codegen.targets.cpp import sym2cpp
from IPython.display import Code




N = dace.symbol('N')

@dace.program
def vector_copy4(A: dace.float64[N] @ dace.dtypes.StorageType.GPU_Global, B: dace.float64[N] @ dace.dtypes.StorageType.GPU_Global):
    for i in dace.map[0:N:32] @ dace.dtypes.ScheduleType.GPU_Device:
        for j in dace.map[0:32] @ dace.dtypes.ScheduleType.GPU_ThreadBlock:
            if i + j < N:
                A[i + j] = B[i + j]

n = random.randint(40, 150)
# Initialize random CUDA arrays
A = cp.zeros(n, dtype=cp.float64)  # Output array
B = cp.random.rand(n).astype(cp.float64)  # Random input array

sdfg = vector_copy4.to_sdfg()
sdfg(A=A, B=B, N=n)
equal_at_end = cp.all(A == B)

