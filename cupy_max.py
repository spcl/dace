import dace
import cupy as cp
import numpy as np


@dace.program
def multidimred(a: dace.float64[3], b: dace.float64[1]):
    b[:] = cp.max(a, axis=(0, 1))


#a = cp.random.rand(2, 2, 6)
#b = cp.random.rand(6)

sdfg = multidimred.to_sdfg()
csdfg = sdfg.compile()

a = cp.random.rand(2, 2, 6)
b = cp.random.rand(6)
csdfg(a, b)