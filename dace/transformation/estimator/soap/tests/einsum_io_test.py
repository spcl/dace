import sys
sys.path.insert(0,"C:/gk_pliki/uczelnia/soap/dace")
from dace.transformation.estimator.soap.einsum_to_sdfg import sdfg_gen
from dace.transformation.estimator.soap.io_result import perform_soap_analysis
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

if __name__ == "__main__":
    test_mttkrp_io()