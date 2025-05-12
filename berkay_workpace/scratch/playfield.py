import dace
import random
import cupy as cp
from dace.frontend.python.interface import inline


from dace import registry
from dace.sdfg.scope import ScopeSubgraphView
from dace.codegen.prettycode import CodeIOStream
from dace.codegen.targets.target import TargetCodeGenerator
from dace.codegen.targets.framecode import DaCeCodeGenerator
from dace.codegen.targets.cpp import sym2cpp
from IPython.display import Code
from dace.config import Config


@dace.program
def reduce_add_sync(mask: dace.uint32, value: dace.uint32):

    result = dace.define_local_scalar(dace.uint32)
    
    with dace.tasklet(dace.Language.CPP):
        inp_mask << mask
        inp_value << value
        out_result >> result
        """
        out_result = __reduce_add_sync(inp_mask, inp_value);
        """
    return result



@dace.program
def warpLevel(A: dace.uint32[512] @ dace.dtypes.StorageType.GPU_Global, B: dace.uint32[512] @ dace.dtypes.StorageType.GPU_Global):
    for _ in dace.map[0:512:512] @ dace.dtypes.ScheduleType.GPU_Device:
        for j in dace.map[0:512] @ dace.dtypes.ScheduleType.GPU_ThreadBlock:

            value = A[j]
            mask = 0xffffffff
            result = 0

            for _ in dace.map[0:16] @ dace.dtypes.ScheduleType.GPU_Warp:
                
                result = reduce_add_sync(mask, value)
            
            B[j] = result


A = cp.ones(512, cp.uint32)
B = cp.random.rand(512).astype(cp.uint32)

sdfg = warpLevel.to_sdfg()
sdfg(A=A, B=B)

print(B)