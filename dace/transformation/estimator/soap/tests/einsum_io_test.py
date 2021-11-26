import sys
sys.path.insert(0,"C:/gk_pliki/uczelnia/soap/dace")
import dace
from dace.transformation.estimator.soap.einsum_to_sdfg import sdfg_gen
from dace.transformation.estimator.soap.io_result import perform_soap_analysis
import numpy as np
import sympy as sp


def test_mttkrop_io():
    """ 
    Test MTTKRP I/O lower bound Q >= 3*N^4/S**(2/3)
    """

    # get MTTKRP sdfg:
    einsum = 'ijk,jl,kl->il'
    dim = 30      
    inputs = einsum.replace(' ', '').split('->')[0].split(',')
    inp_arrays = []            
    for input in inputs:
        order = len(input)
        A = np.random.rand(dim**order).reshape([dim] * order)           
        inp_arrays.append(A)
            
    sdfg = sdfg_gen(einsum, inp_arrays)
    soap_result = perform_soap_analysis(sdfg)

    # test MTTKRP I/O bound
    assert soap_result.Q == sp.sympify('3*S0*S1*S2*S3/Ss**(2/3)')

    a = 1


if __name__ == "__main__":
    test_mttkrop_io()