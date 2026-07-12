import numpy as np


def kernel(TSTEPS, A, B):

    # The DaCe corpus kernel loops `for t in range(tsteps)` (TSTEPS sweeps); match it.
    for t in range(TSTEPS):
        B[1:-1] = 0.33333 * (A[:-2] + A[1:-1] + A[2:])
        A[1:-1] = 0.33333 * (B[:-2] + B[1:-1] + B[2:])
