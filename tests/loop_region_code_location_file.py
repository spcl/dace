# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.
import numpy as np
import dace

N = dace.symbol('N')

@dace.program(use_experimental_cfg_blocks=True)
def function_for(a: dace.float64[N]):
    b = np.zeros((N,))
    for i in range(N):
        b[i] = \
            a[i] + 10
    return b

@dace.program(use_experimental_cfg_blocks=True)
def function_while(a: dace.float64[N]):
    b = np.zeros((N,))
    i = 0
    while i < N:
        b[i] = \
            a[i] + 10
        i += 1
    return b

