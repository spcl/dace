import sys
from dace.transformation.estimator.soap.einsum_to_sdfg import sdfg_gen
from dace.transformation.estimator.soap.io_analysis import perform_soap_analysis, perform_soap_analysis_einsum
from dace.transformation.estimator.soap.utils import d2sp
import numpy as np
import sympy as sp


def test_mttkrp_io():
    """ 
    Test MTTKRP I/O lower bound Q >= 3*N^4/S**(2/3)
    """
    # get MTTKRP sdfg with auto-generated default tensor sizes  
    # einsum_str = 'ijk,jl,kl->il' 
    # einsum_str = 'ik,kj,jl->il'
    # i0-4 = abcde
    # j0-4 = fghij
    einsum_str = 'af,bg,ch,di,ej,bcgh,fghij->abcde'
    # decomp_params=[("p", 17), ("Ss", 1024), ("S0", 256), ("S1", 256), ("S2", 256), ("S3", 256)]
    # decomp_params=[("p", 64), ("Ss", 1024), ("S0", 6400 * 4), ("S1", 4400 * 4), ("S2", 7200 * 4), ("S3", 256)]
    decomp_params = [("p", 64), ("Ss", 1024)]
    for i in range(10):
        decomp_params.append((f"S{i}", 2))
    # p = 128
    # NI = 6400
    # NJ = 7200
    # NK = 4400
    # NL = 4800
    # def scale(N: int):
    #     return int(np.ceil(int(N * np.cbrt(p)) / 2) * 2)
    # S0 = scale(NI)
    # S1 = scale(NK)
    # S2 = scale(NJ)
    # S3 = scale(NL)
    # decomp_params=[("p", p), ("Ss", 1024), ("S0", S0), ("S1", S1), ("S2", S2), ("S3", S3)]
    einsum_str = 'ijk,jl,kl->il'
    # einsum_str = 'ik,kj,jl->il'  # ik,kj -> ij.....    ij, jl -> il
    decomp_params=[("p", 17), ("Ss", 1024), ("S0", 256), ("S1", 256), ("S2", 256), ("S3", 25600)]
    soap_result = perform_soap_analysis_einsum(einsum_str, decomp_params, generate_schedule=True)
    for i, sgraph in enumerate(soap_result.subgraphs):
        print(f"Subgraph {i}==================================================")
        print(f"Variables: {sgraph.variables}")
        print(f"Local domains: {sgraph.loc_domain_dims}")
        print(f"Grid: {sgraph.p_grid}")
    # test MTTKRP I/O bound
    # assert d2sp(soap_result.Q) == sp.sympify('3*S0*S1*S2*S3/Ss**(2/3)')


def test_opt_einsum_example_io():
    """ 
    Test the example from the opt_einsum documentation pi,qj,ijkl,rk,sl->pqrs
    """
    sdfg = sdfg_gen('pi,qj,ijkl,rk,sl->pqrs')

    soap_result = perform_soap_analysis(sdfg)
    # test MTTKRP I/O bound
    assert d2sp(soap_result.Q) == sp.sympify('2*S0*S5*(S1*S3*S4 + S2*(S3*S4 + S6*(S4 + S7)))/sqrt(Ss)')

if __name__ == "__main__":
    test_mttkrp_io()
    # test_opt_einsum_example_io()
