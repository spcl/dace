#!/usr/bin/env python
import dace
import math as mt
import numpy as np


@dace.program
def myprint(input, N, M):
    @dace.tasklet
    def myprint():
        a << input
        for i in range(0, N):
            for j in range(0, M):
                printf("%f\n", mt.sin(a[i, j]))


input = dace.ndarray([10, 10], dtype=dace.float32)
input[:] = np.random.rand(10, 10).astype(dace.float32.type)

myprint(input, 10, 10)
