import dace
import random
import cupy as cp

from dace import registry
from dace.sdfg.scope import ScopeSubgraphView
from dace.codegen.prettycode import CodeIOStream
from dace.codegen.targets.target import TargetCodeGenerator
from dace.codegen.targets.framecode import DaCeCodeGenerator
from dace.codegen.targets.cpp import sym2cpp
from IPython.display import Code
from dace.config import Config

print(Config.get('compiler', 'cuda', 'implementation'))


@dace.program
def warpLevel(A: dace.float64[512] @ dace.dtypes.StorageType.GPU_Global, B: dace.float64[512] @ dace.dtypes.StorageType.GPU_Global):
    for i in dace.map[0:512:512] @ dace.dtypes.ScheduleType.GPU_Device:
        for j in dace.map[0:512] @ dace.dtypes.ScheduleType.GPU_ThreadBlock:
            for k in dace.map[0:16] @ dace.dtypes.ScheduleType.GPU_Warp:
                A[k] = A[k] + 1


sdfg = warpLevel.to_sdfg()
Code(sdfg.generate_code()[0].clean_code, language='cpp')


"""
"""


"""
@dace.program
def vector_copy3(A: dace.float64[64] @ dace.dtypes.StorageType.GPU_Global, B: dace.float64[64] @ dace.dtypes.StorageType.GPU_Global):
    for j in dace.map[0:32] @ dace.dtypes.ScheduleType.GPU_ThreadBlock:
        A[j] = B[j]

sdfg = vector_copy3.to_sdfg()
Code(sdfg.generate_code()[0].clean_code, language='cpp')




"""