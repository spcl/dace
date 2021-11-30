import sys
sys.path.insert(0,"C:/gk_pliki/uczelnia/soap/dace")
import dace
from dace.transformation.estimator.soap.soap import SoapStatement, AccessParams, OutputArrayParams
from dace.transformation.estimator.soap.solver import Solver
from dace.transformation.estimator.soap.utils import SOAPParameters, d2sp
import numpy as np
import sympy as sp
import copy
from collections import namedtuple

# parameters
n = dace.symbol('n') 
M = dace.symbol('M')
K = dace.symbol('K')
Ss = dace.symbol('Ss')

# iteration variables
i = dace.symbol('i')
j = dace.symbol('j')
k = dace.symbol('k')

rngs = [(i,sp.sympify(1), n), (j, sp.sympify(1), M), (k, sp.sympify(1), K)]
soap_ranges = {"scope1" : rngs}

# arrays
A = dace.symbol('A')
B = dace.symbol('B')
C = dace.symbol('C')

# phis
params_C  = AccessParams(ssa_dim=[k], offsets={(0,0)})
params_A  = AccessParams(ssa_dim=[], offsets={(0,0)})
params_B  = AccessParams(ssa_dim=[], offsets={(0,0)})
phi_C = {'i*j' : params_C}
phi_A = {'i*k' : params_A}
phi_B = {'k*j' : params_B}

# output access (for a single SOAP statement)
out_C = OutputArrayParams('i*j' , True, (0, 1), ssa_dim=[[k,k,1]])

# output arrays (list of all read-and-write arrays in the entire SDFG)
out_arrays = {"out1" : copy.deepcopy(out_C)}




S = dace.symbol('S')

def test_MMM_lowerbound():
    """ Test MMM I/O lower bound Q >= 2*M*N*K / sqrt(S). """
    S = SoapStatement()
    S.phis['C'] = phi_C
    S.phis['A'] = phi_A
    S.phis['B'] = phi_B
    S.output_accesses['C'] = out_C
    S.output_arrays = out_arrays
    S.ranges = soap_ranges
    S.loop_ranges = []
    params = SOAPParameters()
    solver = Solver()
    solver.start_solver(remoteMatlab = True)
    S.solve(solver, params)
    assert d2sp(S.Q) == d2sp(2*M*n*K / sp.sqrt(Ss))


if __name__ == "__main__":
    test_MMM_lowerbound()