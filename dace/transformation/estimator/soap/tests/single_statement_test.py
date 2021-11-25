import dace
from soap.analysis_classes import *
from soap.utils import *
import pytest
import numpy as np
import sympy as sp

# parameters
N = dace.symbol('N')
M = dace.symbol('M')
K = dace.symbol('K')
Ss = dace.symbol('Ss')

# iteration variables
i = dace.symbol('i')
j = dace.symbol('j')
k = dace.symbol('k')

rngs = [(i,sp.sympify(1), N), (j, sp.sympify(1), M), (k, sp.sympify(1), K)]
soap_ranges = {"scope1" : rngs}

# arrays
A = dace.symbol('A')
B = dace.symbol('B')
C = dace.symbol('C')

# phis
phi_C = {'i*j*k' : {(0, 0, 0)}}
phi_A = {'i*k' : {(0, 0)}}
phi_B = {'k*j' : {(0, 0)}}

# output access
out_C = ('i*j*k' , (0, 0, 1), True)




S = dace.symbol('S')
remoteMatlab = False

def test_MMM_lowerbound():
    """ Test MMM I/O lower bound Q >= 2*M*N*K / sqrt(S). """
    S = SOAP_statement()
    S.phis['C'] = phi_C
    S.phis['A'] = phi_A
    S.phis['B'] = phi_B

    S.outputAccesses['C'] = out_C

    S.ranges = soap_ranges

    solver = solver()
    solver.StartSolver(remoteMatlab)
    S.solve(solver)

    assert S.Q == 2*M*N*K / sp.sqrt(Ss)


if __name__ == "__main__":
    test_MMM_lowerbound()