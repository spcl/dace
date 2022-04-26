""" Prototype Deinsum implementation. """

import re
from dace import nodes
from dace.transformation.estimator.soap.einsum_to_sdfg import sdfg_gen
from dace.transformation.estimator.soap.io_analysis import IOAnalysisSubgraph
from dace.transformation.estimator.soap.sdg import SDG
from dace.transformation.estimator.soap.solver import Solver
from typing import List, Union


def deinsum(desc: Union[str, List[str]]) -> str:

    # TODO: Currently support only a single einsum
    assert(isinstance(desc, str) or (isinstance(desc, (list, tuple)) and len(desc) == 1))
    if isinstance(desc, str):
        desc = [desc]

    # Create shared-memory SDFG
    sdfg = None
    contraction_list = []
    for einsum in desc:
        partial_sdfg, contractions = sdfg_gen(einsum)
        contraction_list.extend(contractions)
        if sdfg:
            raise NotImplementedError
        else:
            sdfg = partial_sdfg
    
    # I/O analysis
    solver = Solver()
    solver.start_solver()
    solver.set_timeout(300)
    sdg = SDG(sdfg, solver)
    Q, subgraphs= sdg.calculate_IO_of_SDG()
    solver.end_solver()
    subgraphs_res = []
    for subgr in subgraphs:
            io_res_sg = IOAnalysisSubgraph(name= subgr.name, Q = subgr.Q, rho = subgr.rhoOpts, 
                        variables = subgr.variables, inner_tile = subgr.inner_tile, 
                        outer_tile = subgr.outer_tile, input_arrays = subgr.phis,
                        tasklets = subgr.tasklet)                          
            subgraphs_res.append(io_res_sg)

    symbols = []

    # Header
    code = """
import dace
import numpy as np

    """

    allvars = set()
    inputs = dict()
    for i, subgraph in enumerate(reversed(subgraphs_res)):
        allvars.update(set(subgraph.variables))
        for arr, arrdesc in subgraph.input_arrays.items():
            if arr.startswith('inp'):
                for k, _ in arrdesc.items():
                    if isinstance(k, str):
                        tokens = k.split('*')
                        inputs[f"grid{i}_{arr}"] = ','.join([f"S{t}G{i}" for t in tokens])
                        break
        code += f"""
{','.join([f"S{v}G{i}" for v in subgraph.variables])} = (dace.symbol(s) for s in ({','.join([f"'S{v}G{i}'" for v in subgraph.variables])}))
{','.join([f"P{v}G{i}" for v in subgraph.variables])} = (dace.symbol(s) for s in ({','.join([f"'P{v}G{i}'" for v in subgraph.variables])}))
        """

    code += f"""
{','.join([f"S{v}G" for v in allvars])} = (dace.symbol(s) for s in ({','.join([f"'S{v}'" for v in allvars])}))


dctype = dace.float64
nptype = np.float64

@dace.program
def deinsum_program({", ".join([f"{arr}: dctype[{shape}]" for arr, shape in inputs.items()])}):
    """

    # for arr, shape in inputs.items():
    #     code += f"""
    # {arr}: dctype[{shape}],
    #     """

    # code += "):"

    for i, subgraph in enumerate(reversed(subgraphs_res)):
        contractions_num = len(subgraph.tasklets)
        contractions = contraction_list[:contractions_num]

        pgrid = ','.join(f"P{v}G{i}" for v in subgraph.variables)
        variables = [str(v) for v in subgraph.variables]
        
        code += f"""
    grid{i} = dace.comm.Cart_create([{pgrid}])
        """
    
        if i > 0:
            prev_subgraph = subgraphs_res[len(subgraphs_res) - i]
            prev_variables = [str(v) for v in prev_subgraph.variables]
            for arr, arrdesc in subgraph.input_arrays.items():
                if arr.startswith('out'):
                    for k, _ in arrdesc.items():
                        if isinstance(k, str):
                            tokens = k.split('*')
                            # print(arr, tokens)
                            gather = [v in tokens for v in prev_variables]
                            reduce = [v not in tokens for v in prev_variables]
                            scatter = [v in tokens for v in variables]
                            bcast = [v not in tokens for v in variables]
                            global_shape = ','.join([f"S{t}" for t in tokens])
                            prev_local_shape = ','.join([f"S{t}G{i-1}" for t in tokens])
                            local_shape = ','.join([f"S{t}G{i}" for t in tokens])
                            code += f"""
    grid{i-1}_{arr}_gather = dace.comm.Cart_sub(grid{i-1}, {gather}, exact_grid=0)
    grid{i-1}_{arr}_reduce = dace.comm.Cart_sub(grid{i-1}, {reduce})
    grid{i-1}_{arr}_subarray = dace.comm.Subarray([{global_shape}], [{prev_local_shape}], nptype, process_grid=grid{i-1}_{arr}_gather)
    grid{i}_{arr}_scatter = dace.comm.Cart_sub(grid{i}, {scatter}, exact_grid=0)
    grid{i}_{arr}_bcast = dace.comm.Cart_sub(grid{i}, {bcast})
    grid{i}_{arr}_subarray = dace.comm.Subarray([{global_shape}], [{local_shape}], nptype, process_grid=grid{i}_{arr}_scatter)
    
    dace.comm.Reduce(grid{i-1}_{arr}, 'MPI_SUM', grid=grid{i-1}_{arr}_reduce)
    grid{i}_{arr} = np.empty_like(grid{i-1}_{arr}, shape=[{local_shape}])
    dace.comm.Redistribute(grid{i-1}_{arr}, grid{i-1}_{arr}_subarray, grid{i}_{arr}, grid{i}_{arr}_subarray)
    dace.comm.Bcast(grid{i}_{arr}, grid=grid{i}_{arr}_bcast)
                            """
                            break

        for contraction, tasklet in zip(contractions, subgraph.tasklets):
            einsum = contraction[2]
            operation = contraction[4]
            tokens = re.split(',|->', einsum)
            contr = list(set(tokens[0]).intersection(set(tokens[1])))
            for n, s in sdfg.all_nodes_recursive():
                if isinstance(n, nodes.Tasklet) and n.name == tasklet.name:
                    state = s
                    node = n
                    break
            inputs = [e.data.data for e in state.in_edges(node)]
            output = [e.data.data for e in state.out_edges(node)][0]
            if operation in ('GEMM', 'TDOT'):
                axes_a = [tokens[0].index(t) for t in contr]
                axes_b = [tokens[1].index(t) for t in contr]
                code += f"""
    grid{i}_{output} = np.tensordot(grid{i}_{inputs[0]}, grid{i}_{inputs[1]}, axes=({axes_a}, {axes_b}))
                """
            else:
                params = set(tokens[0]).union(set(tokens[1]))
                ranges = [f"0:S{p}" for p in params]
                if contr:
                    code += f"""
    grid{i}_{output} = np.zeros({sdfg.arrays[output].shape}, dtype=nptype)
    for {','.join(params)} in dace.map[{','.join(ranges)}]:
        grid{i}_{output}[{','.join(tokens[2])}] += grid{i}_{inputs[0]}[{','.join(tokens[0])}] * grid{i}_{inputs[1]}[{','.join(tokens[1])}]
                    """
                else:
                    code += f"""
    grid{i}_{output} = np.empty({sdfg.arrays[output].shape}, dtype=nptype)
    for {','.join(params)} in dace.map[{','.join(ranges)}]:
        grid{i}_{output}[{','.join(tokens[2])}] = grid{i}_{inputs[0]}[{','.join(tokens[0])}] * grid{i}_{inputs[1]}[{','.join(tokens[1])}]
                    """

        contraction_list = contraction_list[contractions_num:]

    if i == len(subgraphs) - 1:
        tokens = [t for t in tokens[2]]
        reduce = [v not in tokens for v in variables]
        code += f"""
    grid{i}_{output}_reduce = dace.comm.Cart_sub(grid{i}, {reduce})
    dace.comm.Allreduce(grid{i}_{output}, 'MPI_SUM', grid=grid{i}_{output}_{reduce})
        """

    print(code)