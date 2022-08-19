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
    'ik,kj->ij'
    einsum_str = 'ijk,jl,kl->il' 
    # einsum_str = 'a,b,bc,cd->ad'
    # einsum_str = 'ab,bc,c,d->ad'
    # einsum_str = 'ik,kj,jl->il'
    # einsum_str = 'af,bg,ch,di,ej,bcgh,fghij->abcde'
    # einsum_str = 'ab,ab->ab'
    # einsum_str = 'abc,jb,kc->ajk'
    # einsum_str = 'ijkl,ia,kc,ld->jacd'  # 4-mode TTMc
    # einsum_str = 'ijkl,ia,ja,ka->la'  # 4-mode MTTKRP
    # einsum_str = 'ijklm,ia,ja,la,ma->ka' # 5-mode MTTKRP
    # einsum_str = 'ijklm,ja,ka,la,ma->ia' # 5-mode MTTKRP
    # einsum_str = 'ijklm,ia,ja,ka,la->ma' # 5-mode MTTKRP
    # einsum_str = 'ijklm,jb,kc,ld,me->ibcde' # 5-mode TTMc
    # einsum_str = 'ijk,jb,kc->ibc' # 3-mode TTMc
    # einsum_str = 'ijklm,jklmbcde->ibcde' # 5-mode TTMc test
    # einsum_str = 'ijklm,jb,kc,ld,me,il,ke,jd->ibcde' # 5-mode TTMc redistr
    # einsum_str = 'ik, kj, jl, lm -> im' # 3mm
    # einsum_str = 'ijk,lj,mk->ilm'
    # einsum_str = 'ij, jk -> ik'
    # einsum_str = 'ij, jk, kl -> il'
    # einsum_str = 'ij, jk, kl, lm -> im'
    # einsum_str = 'ijk, ja, ka -> ia'
    # einsum_str = 'ijk, ia, ka -> ja'
    # einsum_str = 'ijk, ia, ja -> ka'
    # einsum_str = 'ijk, ja, ka, al -> il'
    decomp_params = [("p", 64), ("Ss", 1024), ("S0", 200), ("S1", 200), ("S2", 200), ("S3", 200), ("S4", 200)]
    decomp_params = [("p", 256), ("Ss", 1e10)]
    for i in range(10):
        decomp_params.append((f"S{i}", 100))
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
