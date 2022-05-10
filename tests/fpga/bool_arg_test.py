# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.

# Test Intel FPGA backend with bool argument (bool is not a supported type in OpenCL)

import dace
import numpy as np
import random
from dace.transformation.interstate import FPGATransformSDFG, InlineSDFG

@dace.program
def bool_arg(flag:dace.uint32, x:dace.float32):

    if flag:
        tmp = x + 1
    else:
        tmp = x- 1
    return tmp



if __name__ == "__main__":
    x =  np.random.rand(1).astype(np.float32)
    flag = bool(random.getrandbits(1))
    sdfg = bool_arg.to_sdfg()
    applied = sdfg.apply_transformations([FPGATransformSDFG])

    gt = x+1 if flag else x-1
    res=sdfg(flag=flag, x=x)
    assert np.allclose(gt, res)





