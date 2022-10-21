from os import name
from typing import Dict, List, Union, Set
import sympy as sp
from dace.sdfg.graph import SubgraphView
from dace.transformation.estimator.soap.sdg import SDG
from dace.transformation.estimator.soap.solver import Solver
from dace.transformation.estimator.soap.einsum_to_sdfg import sdfg_gen
from dataclasses import dataclass, field
from dace import SDFG, Config
import networkx as nx

@dataclass
class IOAnalysisSubgraph():
    name : int
    Q: sp.Expr
    rho : sp.Expr
    input_arrays : Dict
    tasklets : Set     
    variables :  List[sp.Expr]
    inner_tile : List[sp.Expr]
    outer_tile : List[sp.Expr]

    # The following three fields define the parallel distribution.
    # they require user to specify numerical value of parameters, 
    # such as P, S, and problem sizes. The default values of some of these 
    # parameters are specified in utils.param_values
    loc_domain_dims : List[sp.Expr] = field(default_factory=list)
    p_grid : List[sp.Expr] = field(default_factory=list)
    dimensions_ordered : List[sp.Expr] = field(default_factory=list)

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
class IOAnalysis():
    name : int
    Q: sp.Expr
    sdg: SDG
    subgraphs : List[IOAnalysisSubgraph]


def perform_soap_analysis(sdfg : Union[SDFG, SubgraphView], decomp_params: List = [],
                    generate_schedule : bool = False, solver_timeout : int = 60) -> IOAnalysis:
    """
    Main interface of the SOAP analysis. 

    Input:
    sdfg (SDFG): sdfg to be analyzed
    generate_schedule [Optional] (bool): Whether the parallel decomposition should be evaluated for each subgraph.
                                         The decomposition parameters are specified in params.param_values.
    decomp_params: specifies numerical values of the symbolic parameters as a list of tuples
                 (e.g., [("p", 64, "Ss", 1024, "N", 128)]). If not specified, default values are taken
                 from Config.get("soap", "decomposition", "decomposition_params")

    Output:
    io_result_subgraph: dataclass containing SOAP analysis results, 
    such as I/O lower bound (Q), computational intensity (rho), symbolic directed graph (SDG),
    and the list of subgraphs (subgraphs) with their optimal tilings and kernel merging
    """
    solver = Solver()
    solver.start_solver()
    solver.set_timeout(solver_timeout)
        
    sdg = SDG(sdfg, solver)
    # check if the created SDG is correct
    assert(nx.number_weakly_connected_components(sdg.graph) == 1)
    Q, subgraphs = sdg.calculate_IO_of_SDG()
    solver.end_solver()

    subgraphs_res = []
    for subgr in subgraphs:
        io_res_sg = IOAnalysisSubgraph(name= subgr.name, Q = subgr.Q, rho = subgr.rhoOpts, 
                    variables = subgr.variables, inner_tile = subgr.inner_tile, 
                    outer_tile = subgr.outer_tile, input_arrays = subgr.phis, 
                    tasklets = subgr.tasklet)                    
        if generate_schedule:
            if len(decomp_params) == 0:
                decomp_list = Config.get("soap", "decomposition", "decomposition_params")
                decomp_params = list(zip(decomp_list[::2],decomp_list[1::2]))
            subgr.init_decomposition(decomp_params)
            io_res_sg.loc_domain_dims = subgr.loc_domain_dims
            io_res_sg.p_grid = subgr.p_grid
            io_res_sg.dimensions_ordered = subgr.dimensions_ordered       
            io_res_sg.get_data_decomposition = subgr.get_data_decomposition 
        subgraphs_res.append(io_res_sg)
    
    return(IOAnalysis(SDFG.__name__, Q, sdg, subgraphs_res))



def perform_soap_analysis_einsum(einsum_string : str, decomp_params: List = [],
            generate_schedule : bool = True) -> IOAnalysis:
    """
    Specialization of the main interface dedicated for the einsum tensor operations.

    Input:
    einsum_str: einsum srtring to be analyzed
    decomp_params: specifies numerical values of the symbolic parameters as a list of tuples
                 (e.g., [("p", 64, "Ss", 1024, "N", 128)]). If not specified, default values are taken
                 from Config.get("soap", "decomposition", "decomposition_params")
    Output:
    io_result_subgraph: dataclass containing SOAP analysis results, 
    such as I/O lower bound (Q), computational intensity (rho), symbolic directed graph (SDG),
    and the list of subgraphs (subgraphs) with their optimal tilings and kernel merging
    """
    solver = Solver()
    solver.start_solver()
    solver.set_timeout(300)

    sdfg, _ = sdfg_gen(einsum_string)
    sdg = SDG(sdfg, solver)
    Q, subgraphs= sdg.calculate_IO_of_SDG()
    solver.end_solver()

    # Scaling
    scaling = {
        1: (1024, 24),
        2: (1218, 30),
        4: (1450, 34),
        8: (1724, 42),
        12: (1908, 48),
        16: (2048, 48),
        27: (2337, 57),
        32: (2436, 60),
        64: (2900, 68),
        125: (3425, 85),
        128: (3448, 88),
        252: (4116, 126),
        256: (4096, 96),
        512: (4872, 120)
    }

    procs = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    for p in procs:

        print(f"########## PROCS := {p} ##########\n")
        decomp_params = [('p', p)] + [decomp_params[1]] + [(f'S{i}', scaling[p][0]) for i in range(3)] + [(f'S{i}', scaling[p][1]) for i in range(3,5)]
        print(decomp_params)
        subgraphs_res = []
        for subgr in subgraphs:
            io_res_sg = IOAnalysisSubgraph(name= subgr.name, Q = subgr.Q, rho = subgr.rhoOpts, 
                        variables = subgr.variables, inner_tile = subgr.inner_tile, 
                        outer_tile = subgr.outer_tile, input_arrays = subgr.phis,
                        tasklets = subgr.tasklet)                    
            if generate_schedule and hasattr(subgr, 'varsOpt'):
                if decomp_params == []:
                    decomp_list = Config.get("soap", "decomposition", "decomposition_params")
                    decomp_params = list(zip(decomp_list[::2],decomp_list[1::2]))
                subgr.init_decomposition(decomp_params)
            
                io_res_sg.loc_domain_dims = subgr.loc_domain_dims
                io_res_sg.p_grid = subgr.p_grid
                io_res_sg.dimensions_ordered = subgr.dimensions_ordered        
            subgraphs_res.append(io_res_sg)
        
        for i, sgraph in enumerate(subgraphs_res):
            print(f"Subgraph {i}==================================================")
            print(f"Variables: {sgraph.variables}")
            print(f"Local domains: {sgraph.loc_domain_dims}")
            print(f"Grid: {sgraph.p_grid}")
        
        print()
    
    return(IOAnalysis(SDFG.__name__, Q, sdg, subgraphs_res))

