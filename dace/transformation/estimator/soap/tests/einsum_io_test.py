import sys
from dace.transformation.estimator.soap.einsum_to_sdfg import sdfg_gen
from dace.transformation.estimator.soap.io_analysis import perform_soap_analysis
from dace.transformation.estimator.soap.utils import d2sp
import numpy as np
import sympy as sp


def test_mttkrp_io():
    """ 
    Test MTTKRP I/O lower bound Q >= 3*N^4/S**(2/3)
    """
    # get MTTKRP sdfg with auto-generated default tensor sizes  
    sdfg = sdfg_gen('ijk,jl,kl->il')

    soap_result = perform_soap_analysis(sdfg)
    # test MTTKRP I/O bound
    assert d2sp(soap_result.Q) == sp.sympify('3*S0*S1*S2*S3/Ss**(2/3)')


def test_opt_einsum_example_io():
    """ 
    Test the example from the opt_einsum documentation pi,qj,ijkl,rk,sl->pqrs
    """
    # get MTTKRP sdfg with auto-generated default tensor sizes  
    sdfg = sdfg_gen('pi,qj,ijkl,rk,sl->pqrs')

    soap_result = perform_soap_analysis(sdfg)
    # test MTTKRP I/O bound
    assert d2sp(soap_result.Q) == sp.sympify('2*S0*S5*(S1*S3*S4 + S2*(S3*S4 + S6*(S4 + S7)))/sqrt(Ss)')

if __name__ == "__main__":
    test_opt_einsum_example_io()