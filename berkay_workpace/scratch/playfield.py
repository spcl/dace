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


bs = 512
ns = 1024
BS = dace.symbol('BS')
NS = dace.symbol('NS')

START = dace.symbol('START')
WS = dace.symbol('WS')
STRIDE = dace.symbol('STRIDE')

start = 2
stride = 3
ws = 16
@dace.program
def symbolic_warp_map(A: dace.uint32[NS] @ dace.dtypes.StorageType.GPU_Global, B: dace.uint32[NS] @ dace.dtypes.StorageType.GPU_Global):
    """
    Focus is in the use of symbolic variables in the MAP.
    """
    A[:] = B[:]

sdfg = symbolic_warp_map.to_sdfg()

Code(sdfg.generate_code()[0].clean_code, language='cpp')