from os import name
from typing import Dict
import sympy as sp
from dace.transformation.estimator.soap.sdg import SDG
from dace.transformation.estimator.soap.utils import parse_params, Solver
from dataclasses import dataclass, field
from dace import SDFG

@dataclass
class io_result_subgraph():
    name : int
    Q: sp.core.Expr
    rho : sp.core.Expr
    input_arrays : Dict
    variables : list[sp.core.Expr]
    inner_tile : list[sp.core.Expr]
    outer_tile : list[sp.core.Expr]

    # The following three fields define the parallel distribution.
    # they require user to specify numerical value of parameters, 
    # such as P, S, and problem sizes. The default values of some of these 
    # parameters are specified in utils.param_values
    loc_domain_dims : list[sp.core.Expr] = field(default_factory=list)
    p_grid : list[sp.core.Expr] = field(default_factory=list)
    dimensions_ordered : list[sp.core.Expr] = field(default_factory=list)

    def get_data_decomposition(self, pp):
        # p_pos is an N -> N^d mapping, translating a global rank to a cooridnate rank in 
        # the d-dimensional iteration space  (p  -> [p_x, p_y, p_z, ....])
        if (sp.prod(self.p_grid) <= pp):
            print("\n\nDecomposition uses only {} processors. Given rank {} will be idle!\n\n".format(sp.prod(self.p_grid), pp))
            return {}
        
        div = lambda x,y: (x-x%y)/y
        p_pos = []
        for dim_num, dim_size in enumerate(self.p_grid):
            p_dim = div(pp, sp.prod(self.p_grid[dim_num+1 :]))
            p_pos.append(p_dim)
            pp -= p_dim * sp.prod(self.p_grid[dim_num+1 :])
            
        iter_ranges = {str(var) : 
            (p_pos[i]* self.loc_domain_dims[i], min((p_pos[i] + 1)* self.loc_domain_dims[i], dim_size))  
            for i, (var, dim_size) in enumerate(zip(self.variables, self.dimensions_ordered))}
        
        
        par_distribution = {}
        for arr, accesses in self.input_arrays.items():
            for access, pars in accesses.items():
                rngs = {}
                for it in access.split('*'):
                    rngs[it] = iter_ranges[it]
            par_distribution[arr] = rngs
                            
        return par_distribution   

@dataclass
class io_result():
    name : int
    Q: sp.core.Expr
    sdg: SDG
    subgraphs : list[io_result_subgraph]


def perform_soap_analysis(sdfg : SDFG, generate_schedule : bool = True) -> io_result_subgraph:
    """
    Main interface of the SOAP analysis. 

    Input:
    sdfg: sdfg to be analyzed

    Output:
    io_result_subgraph: dataclass containing SOAP analysis results, 
    such as I/O lower bound (Q), computational intensity (rho), symbolic directed graph (SDG),
    and the list of subgraphs (subgraphs) with their optimal tilings and kernel merging
    """
    params = parse_params()
    solver = Solver()
    solver.start_solver(params.remoteMatlab)
    solver.set_timeout(300)
    params.solver = solver
    sdg = SDG(sdfg, params)
    Q, subgraphs= sdg.calculate_IO_of_SDG(params)

    subgraphs_res = []
    for subgr in subgraphs:
        io_res_sg = io_result_subgraph(name= subgr.name, Q = subgr.Q, rho = subgr.rhoOpts, 
                    variables = subgr.variables, inner_tile = subgr.inner_tile, 
                    outer_tile = subgr.outer_tile, input_arrays = subgr.phis)                    
        if generate_schedule:
            subgr.init_decomposition(params.param_values,  params)
        
        io_res_sg.loc_domain_dims = subgr.loc_domain_dims
        io_res_sg.p_grid = subgr.p_grid
        io_res_sg.dimensions_ordered = subgr.dimensions_ordered
        
        subgraphs_res.append(io_res_sg)
    
    return(io_result(SDFG.__name__, Q, sdg, subgraphs_res))

