from os import name
from typing import Dict, List, Union, Set
import sympy as sp
from dace.sdfg.graph import SubgraphView
from dace.transformation.estimator.soap.soap import SoapStatement, LoopBody, Loop, LoopHeader
from dace.transformation.estimator.soap.sdg import SDG
from dace.transformation.estimator.soap.solver import Solver
from dace.transformation.estimator.soap.einsum_to_sdfg import sdfg_gen
from dataclasses import dataclass, field
from dace import SDFG, Config
import networkx as nx
from collections import OrderedDict


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


import re

def perform_soap_analysis_from_ir(ir : str, decomp_params: List = [],
                                  generate_schedule : bool = False,
                                  solver_timeout: int = 60) -> IOAnalysis:
    # stupid sympy cannot parse the symbolic variable N - it confuses it with
    # some built-in function. So we replace it with a symbol Nnn
    ir = re.sub(r'\bN\b', 'n', ir)

    solver = Solver()
    solver.start_solver()
    solver.set_timeout(solver_timeout)
    sdg = SDG(solver=solver)
    sdg.from_ir(ir)
    res = perform_soap_analysis(sdg, decomp_params, generate_schedule)
    solver.end_solver()
    return res
    

def perform_soap_analysis_from_sdfg(sdfg : Union[SDFG, SubgraphView], decomp_params: List = [],
                                  generate_schedule : bool = False,
                                  solver_timeout: int = 60) -> IOAnalysis:
    solver = Solver()
    solver.start_solver()
    solver.set_timeout(solver_timeout)
    sdg = SDG(sdfg, solver)
    res = perform_soap_analysis(sdg, decomp_params, generate_schedule)
    solver.end_solver()
    return res


def perform_soap_analysis(sdg : SDG, decomp_params: List = [],
                    generate_schedule : bool = False) -> IOAnalysis:
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

    # check if the created SDG is correct
    assert(nx.number_weakly_connected_components(sdg.graph) == 1)
    Q, subgraphs_graph = sdg.calculate_IO_of_SDG()

    generate_code(subgraphs_graph, decomp_params)

    subgraphs_res = []
    for subgr_name in nx.topological_sort(subgraphs_graph):
        subgr = subgraphs_graph.nodes[subgr_name]['st']
        io_res_sg = IOAnalysisSubgraph(name= subgr.name, Q = subgr.Q, rho = subgr.rhoOpts, 
                    variables = subgr.variables, inner_tile = subgr.inner_tile, 
                    outer_tile = subgr.outer_tile, input_arrays = subgr.phis, 
                    tasklets = subgr.tasklet)                    
        if generate_schedule:
            if len(decomp_params) == 0:
                decomp_list = Config.get("soap", "decomposition", "decomposition_params")
                decomp_params = list(zip(decomp_list[::2],decomp_list[1::2]))
            nontrivial = subgr.init_decomposition(decomp_params)
            if nontrivial:
                io_res_sg.loc_domain_dims = subgr.loc_domain_dims
                io_res_sg.p_grid = subgr.p_grid
                io_res_sg.dimensions_ordered = subgr.dimensions_ordered
                io_res_sg.get_data_decomposition = subgr.get_data_decomposition
        subgraphs_res.append(io_res_sg)
    
    return(IOAnalysis(SDFG.__name__, Q, sdg, subgraphs_res))


def generate_code(subgraphs_graph: nx.DiGraph, decomp_params: List = []):
    """
    Generates the code for the given subgraphs graph. The code is in the form
    of a nested loop program. Each subgraph includes a list of fused kernels
    nested withing its iteration domain. The code is  tiled in two levels: inner and outer. Outer tiling represents parallel decomposition: each tile is executed by a separate processor. Inner tiling represents a sequential schedule within a tile of a single processor (e.g., CUDA warp or OpenMP thread block). The size of the inner tile should correspond to the instruction-level parallelism, e.g., loop unrolling, Tensor Core size or 
    warp schedule. The size of the outer tile should correspond to the data parallelism, e.g., the number of processors or CUDA blocks.

    Input:
    subgraphs_graph: the graph of subgraphs
    decomp_params: specifies numerical values of the symbolic parameters as a list of tuples
                 (e.g., [("p", 64, "Ss", 1024, "M", 128, "N", 128)]). If not specified, default values are taken
                 from Config.get("soap", "decomposition", "decomposition_params")
                 "p" is the number of processors, "Ss" is the size of the local memory (cache, registers on GPU, etc), "M", "N"... are the problem sizes
    """

    """
    The nested loop program is represented as a nested dictionary:
    the keys reprensents the loop levels, the values are the loop bodies
    (possibly nested dictionaries with the same structure)
    """
    # code = OrderedDict()
    code = LoopBody()
    code_str = ""

    if len(decomp_params) == 0:
        decomp_list = Config.get("soap", "decomposition", "decomposition_params")
        decomp_params = list(zip(decomp_list[::2],decomp_list[1::2]))

    # merge trivial kernels into the previous or the next kernel's loop
    for subgr_name in nx.topological_sort(subgraphs_graph):
        subgr: SoapStatement = subgraphs_graph.nodes[subgr_name]['st']
        trivial = subgr.init_decomposition(decomp_params)        
        if trivial:
            # check if this kernel has predecessors. If yes, merge it into the previous kernel
            pred = list(subgraphs_graph.predecessors(subgr_name))
            if len(pred) > 0:
                pred = pred[0]
                pred_subgr: SoapStatement = subgraphs_graph.nodes[pred]['st']
                pred_subgr.append_trivial_kernel_end(subgr)
            else:
                # if not, merge it into the next kernel
                succ = list(subgraphs_graph.successors(subgr_name))
                assert(len(succ) > 0) # if no predecesssors, there must be at least one successor
                succ = succ[0]
                succ_subgr: SoapStatement = subgraphs_graph.nodes[succ]['st']
                succ_subgr.append_trivial_kernel_start(subgr)
            

    for subgr_name in nx.topological_sort(subgraphs_graph):
        subgr: SoapStatement = subgraphs_graph.nodes[subgr_name]['st']
        
        if len(subgr.outer_tile) == 0:
            # it means that this kernel is trivial and has been merged into the previous or the next kernel
            continue
        kernel_code = subgr.generate_code(decomp_params)
        code.statements += kernel_code.statements
        code_str += nested_loop2str(code.statements)
            
    return code

def nested_loop2str(statements) -> str:
    code_str = ""
    for statement in statements:
        if isinstance(statement, Loop):
            inner_loop_header = statement.header
            inner_loop_body = statement.body
            inner_loop_body_str = nested_loop2str(inner_loop_body.statements)
            code_str += str(inner_loop_header) + "\n" + inner_loop_body_str
        else:
            code_str += str(statement) + "\n"
    return code_str

# def code2str(code:LoopBody) -> str:
#     code_str = ""
    
#     for st in code.statements:
#         code_str += nested_loop2str(st) + "\n"
#     return code_str



def perform_soap_analysis_einsum(einsum_string : str, decomp_params: List = [],
            generate_schedule : bool = False) -> IOAnalysis:
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

