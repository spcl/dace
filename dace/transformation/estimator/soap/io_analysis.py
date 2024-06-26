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
import dace
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
        if subgr.trivial:
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
        else:
            # If the subgraph is not trivial, check if its output is used by the next kernel.
            # Why we need this: even if subgr cannot be merged with the next kernel, the final result
            # (e.g., after the reduction) can be used by the next kernel. In this case, we generate a single 
            # kernel that includes two sequential parts: first, the entire computation of subgr local domain,
            # and then, second kernel that reuses local results from the first kernel.
            
            # The important thing is index renaming. E.g., if the first kernel computes C[i,j] += A[i,k] * B[k,j],
            # and the second kernel computes D[i,j] += C[i,k] * E[k,j], the index k in the second kernel should be renamed
            # to j. This is because the local domain of the second kernel is C[i,k] and not C[i,j].
            succ = [subgraphs_graph.nodes[s]['st'] for s in list(subgraphs_graph.successors(subgr_name))]
            if len(succ) > 0:
                # if more than one successor, find which one uses the output of this kernel
                output_arrs = set(subgr.output_arrays)
                
                # succ_matching are the successors with nonzero intersection between output_arrs and succ.subgraph_inputs
                succ_matching = [s for s in succ if len(output_arrs & set(s.phis)) > 0]
                assert(len(succ_matching) == 1)
                succ_matching = succ_matching[0]
                matching_arrays = output_arrs & set(succ_matching.phis)
                assert(len(matching_arrays) == 1)
                matching_arr = matching_arrays.pop()
                
                # check if the matching array is used in the same way in both kernels
                out_access = subgr.output_accesses[matching_arr].baseAccess
                assert(len(succ_matching.phis[matching_arr]) == 1)
                [in_access,acc_params] = list(succ_matching.phis[matching_arr].items())[0]
                if in_access != out_access:
                    # if the access is different, we need to rename the index
                    # find the index that is different
                    out_indices = out_access.split('*')
                    in_indices = in_access.split('*')
                    swaplist = {}
                    inv_swaplist = {}
                    for in_ind, out_ind in zip(in_indices, out_indices):
                        if in_ind != out_ind:
                            swaplist[in_ind] = out_ind
                            inv_swaplist[out_ind] = in_ind
                            
                    # succ_matching.swap_iter_vars(swaplist, inv_swaplist)
                
                # find the reuse dimension of the matching array. It is the reduction dimension of the 
                # successor kernel (taken from its output access)
                assert(len(succ_matching.output_accesses))
                red_dim = list(succ_matching.output_accesses.values())[0].ssa_dim[0]
                if isinstance(red_dim[0], dace.symbolic.symbol):
                    subgr.later_reuse_dim = str(red_dim[0])
                    if subgr.later_reuse_dim in swaplist:
                        subgr.later_reuse_dim = swaplist[subgr.later_reuse_dim]
                else:
                    raise ValueError("The reduction dimension must be a symbolic variable")
            
        subgr.init_decomposition(decomp_params)    
                
            
    statements = []

    for subgr_name in nx.topological_sort(subgraphs_graph):
        subgr: SoapStatement = subgraphs_graph.nodes[subgr_name]['st']
        statements.append(subgr)
    #     if len(subgr.outer_tile) == 0:
    #         # it means that this kernel is trivial and has been merged into the previous or the next kernel
    #         continue
    #     kernel_code = subgr.generate_code(decomp_params)
    #     code.statements += kernel_code.statements
    #     code_str += nested_loop2str(code.statements)
    #     kernel = parallelize_nested_loops(code.statements)
    #     # code = str(kernel) 
    
    code = ParallelKernel()
    code.statements = statements
    code_str = code.generate_code(decomp_params)
    a = 1
    # kernel = parallelize_nested_loops(code.statements) 
    # code = str(kernel) 
    # return code




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


@dataclass
class SequentialKernel:
    id: int
    id_counter: int = 0
    name : str = ""
    arrays: set[str] = field(default_factory=set)
    loop_headers: List[LoopHeader] = field(default_factory=list)
    outer_variables: List[str] = field(default_factory=list)
    
    statements: list = field(default_factory=list)
    
    def get_name(self):
        if self.name == "":
            self.name = f"sequential_kernel_{self.id}"
        return self.name
    
    def __str__(self):
        if self.name == "":
            self.name = f"sequential_kernel_{self.id}"
        
        if self.arrays == set():
            self.get_arrays()
            
        if self.outer_variables == []:
            self.outer_variables = [h.loopvar+"_outer" for h in self.loop_headers]
            
        code_str = f"def {self.name}({', '.join(self.arrays)}, {', '.join(self.outer_variables)}):\n"
        code_str += nested_loop2str(self.sequential_loop_head)
        return code_str
    
    def __init__(self):
        SequentialKernel.id_counter += 1
        self.id = SequentialKernel.id_counter
        self.name = ""
        self.arrays = set()
        self.loop_headers = []
        self.statements = []
        self.sequential_loop_head = None
        self.outer_variables = []
        
    def get_arrays(self):
        '''
        Iterate over all statements and collect all arrays that are accessed in the kernel.
        Each statement is a string of a form "arr_1[accesses] = arr_2[accesses] op arr_3[accesses]", e.g. , "A[i,j] = B[i,j] + C[i,j]".
        The arrays returned by this function are the arrays that are accessed in the kernel, e.g., A, B, C.
        '''
        arrays = set()
        for statement in self.statements:
            # this is a statement
            arrs = re.findall(r"\w+\[", statement)
            # remove the last character, which is "["
            arrs = [arr[:-1] for arr in arrs]
            arrays.update(arrs)
        
        self.arrays = arrays
        return self.arrays



@dataclass
class ParallelKernel:
    id: int
    id_counter: int = 0
    parallel_ker_name : str = ""
    arrays: List[str] = field(default_factory=list)
    statements: List[SoapStatement] = field(default_factory=list)
    # headers: List[LoopHeader] = field(default_factory=list)
    
    # kernels: list[SequentialKernel] = field(default_factory=list)
    
    def __init__(self):
        ParallelKernel.id_counter += 1
        self.id = ParallelKernel.id_counter
        self.parallel_ker_name = ""
        self.seqential_ker_name = ""
        self.arrays = set()
        
    def generate_code(self, decomposition_params: List = []):
        if len(self.arrays) == 0:
            self.get_arrays()
        code = self.generate_sequential_kernels()
        code += "\n\n" + self.generate_parallel_call(decomposition_params)        
        return code
        
    def generate_sequential_kernels(self):
        code_str = "\n"
        for i, statement in enumerate(self.statements):
            code_str += f"def sequential_kernel_{i}({', '.join(self.arrays)}, p):\n"
            code_str += statement.generate_CUDA_code()
            code_str += "\n\n"
        return code_str
    
    def generate_parallel_call(self, decomposition_params: List = []):
        if self.parallel_ker_name == "":
            self.parallel_ker_name = f"parallel_kernel_{self.id}"
            
        P = [x[1] for x in decomposition_params if x[0] == "P"][0]
        
        code_str = f"def {self.parallel_ker_name}({', '.join(self.arrays)}):\n"
        code_str += f"    with concurrent.futures.ThreadPoolExecutor() as executor:\n"
        for i, _ in enumerate(self.statements):
            code_str += f"        futures = [executor.submit(sequential_kernel_{i}, {', '.join(self.arrays)}, p)"
            code_str += f" for p in range({P})"
            code_str += f"]\n        concurrent.futures.wait(futures)\n\n"
        return code_str
    
    def get_arrays(self):
        '''
        Collect all arrays that are accessed in all sequential kernels in
        this parallel kernel using the get_arrays method of the SequentialKernel class. Store them in self.arrays
        '''
        for st in self.statements:
            self.arrays |= set(st.phis.keys()) | set(st.output_arrays.keys())
        self.arrays = sorted(list(self.arrays))


# @dataclass
# class ParallelKernel:
#     id: int
#     id_counter: int = 0
#     name : str = ""
#     arrays: List[str] = field(default_factory=list)
#     headers: List[LoopHeader] = field(default_factory=list)
    
#     kernels: list[SequentialKernel] = field(default_factory=list)
    
#     def __init__(self):
#         ParallelKernel.id_counter += 1
#         self.id = ParallelKernel.id_counter
#         self.name = ""
#         self.arrays = set()
#         self.headers = []
#         self.kernels = []
        
#     def __str__(self):
#         '''
#         generate the parallel kernel with the given outer_variables, ranges, and tile_sizes.
#         Example kernel:
        
#         def parallel_outer_sequential_inner(A):
#             with concurrent.futures.ThreadPoolExecutor() as executor:
#                 futures = [executor.submit(process_tile, i_outer, j_outer, tile_size, A) for i_outer in range(0, N, tile_size) for j_outer in range(0, N, tile_size)]
#                 concurrent.futures.wait(futures)   
#         '''
#         if self.name == "":
#             self.name = f"parallel_kernel_{self.id}"
            
#         # instantiate self.arrays, self.tile_sizes, self.outer_variables, self.starts, self.ends from the list of LoopHeaders
#         [self.outer_variables, self.starts, self.ends, self.tile_sizes] = \
#             [list(x) for x in zip(*[(h.loopvar, h.start, h.end, h.step) for h in self.headers])]
#         # [(h.loopvar, h.start, h.end, h.step) for h in self.headers]
        
#         if len(self.arrays) == 0:
#             self.get_arrays()
           
#         code_str = "\n" 
#         for kernel in self.kernels:
#             code_str += str(kernel)
        
#         code_str += "\n\n"
            
#         code_str += f"def {self.name}({', '.join(self.arrays)}):\n"
#         code_str += f"    with concurrent.futures.ThreadPoolExecutor() as executor:\n"
#         code_str += f"        futures = []\n"
#         for kernel in self.kernels:
#             code_str += f"        futures += [executor.submit({kernel.get_name()}, {', '.join(self.arrays)}, {', '.join(self.outer_variables)})"
#         for (out_var, start, end, tile_size) in zip(self.outer_variables, self.starts, self.ends, self.tile_sizes):
#             code_str += f" for {out_var} in range({start}, {end}, {tile_size})"
#         code_str += f"]\n        concurrent.futures.wait(futures)\n\n"
    
        

#         return code_str
    
#     def get_arrays(self):
#         '''
#         Collect all arrays that are accessed in all sequential kernels in
#         this parallel kernel using the get_arrays method of the SequentialKernel class. Store them in self.arrays
#         '''
#         for kernel in self.kernels:
#             self.arrays |= kernel.get_arrays()
        

def parallelize_nested_loops(statements:LoopBody) -> ParallelKernel:
    par_kernel = ParallelKernel()
    parallelize_nested_loops_recursive(statements, par_kernel)
    return par_kernel
    
def parallelize_nested_loops_recursive(statements:LoopBody, par_kernel: ParallelKernel):
    for statement in statements:
        if isinstance(statement, Loop):
            inner_loop_header: LoopHeader = statement.header
            if 'parallel' in inner_loop_header.pragmas:
                par_kernel.headers.append(inner_loop_header)
                parallelize_nested_loops_recursive(statement.body.statements, par_kernel)                
            else:
                # this means that the inner loop is sequential
                kernel = SequentialKernel()
                kernel.sequential_loop_head = [statement]
                kernel.loop_headers.append(inner_loop_header)
                process_sequential_nested_loops(statement.body.statements, kernel)
                par_kernel.kernels.append(kernel)


def process_sequential_nested_loops(statements:LoopBody, seq_kernel: SequentialKernel):
    
    for statement in statements:
        if isinstance(statement, Loop):
            inner_loop_header = statement.header
            seq_kernel.loop_headers.append(inner_loop_header)
            inner_loop_body = statement.body
            process_sequential_nested_loops(inner_loop_body.statements, seq_kernel)
        else:
            seq_kernel.statements.append(str(statement))
    





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


