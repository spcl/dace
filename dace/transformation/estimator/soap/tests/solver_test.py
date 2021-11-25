import dace
from soap.analysis_classes import *
from soap.utils import *
from soap.SDG import SDG
import pytest
import numpy as np
import sympy as sp

def TestSolver(solver):
    print('Testing')
    S = SOAP_statement(solver)
    S.DomV = sp.sympify("i*j + i*k + k*j")
    S.VhSize = sp.sympify("i*j*k")
    m = dace.symbol('M')
    n = dace.symbol('N')
    k = dace.symbol('K')
    S.V = m*n*k
    
    S.Solve()
    S.Q = S.V / S.rhoOpts
    print('Matrix multiplication I/O lower bound: ' + str(S.Q))