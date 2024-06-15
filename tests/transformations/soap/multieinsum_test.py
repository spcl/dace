import sys
from dace.transformation.estimator.soap.einsum_to_sdfg import sdfg_gen, sdfg_multigen
from dace.transformation.estimator.soap.io_analysis import perform_soap_analysis, perform_soap_analysis_einsum
from dace.transformation.estimator.soap.utils import d2sp
from dace.transformation.estimator.soap.solver import Solver
from dace.transformation.estimator.soap.sdg import SDG
import numpy as np
import sympy as sp


def test_mttkrp_io():
    """ 
    Test MTTKRP I/O lower bound Q >= 3*N^4/S**(2/3)
    """
    # get MTTKRP sdfg with auto-generated default tensor sizes  
    #'ik,kj->ij'
    # einsum_str = 'ijk,jl,kl->il' 
    einsum_str = 'ij,jk,kl->il'
    
    decomp_params = [("p", 64), ("Ss", 1024), ("S0", 200), ("S1", 200), ("S2", 200), ("S3", 200), ("S4", 200)]
    decomp_params = [("p", 256), ("Ss", 1e10)]
    for i in range(10):
        decomp_params.append((f"S{i}", 100))

    solver = Solver()
    solver.start_solver()
    solver.set_timeout(300)
    sdfg = sdfg_multigen(['ik,kj->ij', 'il,lj->ij','ij,jm->im'], {0: {1: -1}, 1:{2: 0}})
    #sdfg.view()
    sdg = SDG(sdfg, solver)
    Q, subgraphs= sdg.calculate_IO_of_SDG()
    solver.end_solver()

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
