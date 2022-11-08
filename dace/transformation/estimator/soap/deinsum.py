""" Prototype Deinsum implementation. """

import re
from dace import config, nodes
from dace.transformation.estimator.soap.einsum_to_sdfg import sdfg_gen
from dace.transformation.estimator.soap.io_analysis import IOAnalysisSubgraph
from dace.transformation.estimator.soap.sdg import SDG
from dace.transformation.estimator.soap.solver import Solver
from typing import List, Union


def deinsum(desc: Union[str, List[str]]):

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
            io_res_sg.pgrid = dict()
            if hasattr(subgr, 'varsOpt'):                    
                # for p in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]:
                for p in [1, 2, 4, 8]:
                    decomp_params = [('p', p), ('Ss', 1024)] + [(f'S{i}', 1024) for i in range(20)]
                    with config.set_temporary('soap', 'decomposition', 'chosen_par_setup', value='memory_independent'):
                        subgr.init_decomposition(decomp_params)
                        io_res_sg.pgrid[p] = subgr.p_grid
            subgraphs_res.append(io_res_sg)

    # Header
    code = """
import dace
import numpy as np
import opt_einsum as oe

from dace.sdfg import utils

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
        
pgrid{i} = {subgraph.pgrid}         
        """

    code += f"""
{','.join([f"S{v}" for v in allvars])} = (dace.symbol(s) for s in ({','.join([f"'S{v}'" for v in allvars])}))


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

    visited_arrays = set()
    reversed_subgraphs = list(reversed(subgraphs_res))
    for i, subgraph in enumerate(reversed_subgraphs):
        contractions_num = len(subgraph.tasklets)
        contractions = contraction_list[:contractions_num]

        pgrid = ','.join(f"P{v}G{i}" for v in subgraph.variables)
        variables = [str(v) for v in subgraph.variables]
        
        code += f"""
    grid{i} = dace.comm.Cart_create([{pgrid}])
        """
    
    #     if i > 0:
    #         prev_subgraph = subgraphs_res[len(subgraphs_res) - i]
    #         prev_variables = [str(v) for v in prev_subgraph.variables]
    #         for arr, arrdesc in subgraph.input_arrays.items():
    #             if arr.startswith('out') and arr in visited_arrays:
    #                 visited_arrays.remove(arr)
    #                 for k, _ in arrdesc.items():
    #                     if isinstance(k, str):
    #                         tokens = k.split('*')
    #                         # print(arr, tokens)
    #                         gather = [v in tokens for v in prev_variables]
    #                         reduce = [v not in tokens for v in prev_variables]
    #                         scatter = [v in tokens for v in variables]
    #                         bcast = [v not in tokens for v in variables]
    #                         global_shape = ','.join([f"S{t}" for t in tokens])
    #                         prev_local_shape = ','.join([f"S{t}G{i-1}" for t in tokens])
    #                         local_shape = ','.join([f"S{t}G{i}" for t in tokens])
    #                         code += f"""
    # grid{i-1}_{arr}_gather = dace.comm.Cart_sub(grid{i-1}, {gather}, exact_grid=0)
    # grid{i-1}_{arr}_reduce = dace.comm.Cart_sub(grid{i-1}, {reduce})
    # grid{i-1}_{arr}_subarray = dace.comm.Subarray([{global_shape}], [{prev_local_shape}], dctype, process_grid=grid{i-1}_{arr}_gather)
    # grid{i}_{arr}_scatter = dace.comm.Cart_sub(grid{i}, {scatter}, exact_grid=0)
    # grid{i}_{arr}_bcast = dace.comm.Cart_sub(grid{i}, {bcast})
    # grid{i}_{arr}_subarray = dace.comm.Subarray([{global_shape}], [{local_shape}], dctype, process_grid=grid{i}_{arr}_scatter)
    
    # dace.comm.Reduce(grid{i-1}_{arr}, 'MPI_SUM', grid=grid{i-1}_{arr}_reduce)
    # grid{i}_{arr} = np.empty_like(grid{i-1}_{arr}, shape=[{local_shape}])
    # dace.comm.Redistribute(grid{i-1}_{arr}, grid{i-1}_{arr}_subarray, grid{i}_{arr}, grid{i}_{arr}_subarray)
    # dace.comm.Bcast(grid{i}_{arr}, grid=grid{i}_{arr}_bcast)
    #                         """
    #                         break

        if i > 0:
            # prev_subgraph = subgraphs_res[len(subgraphs_res) - i]
            # prev_variables = [str(v) for v in prev_subgraph.variables]
            for arr, arrdesc in subgraph.input_arrays.items():
                if arr.startswith('out') and arr in visited_arrays:
                    visited_arrays.remove(arr)
                    previdx = -1
                    for j in range(i):
                        prev_subgraph = reversed_subgraphs[j]
                        for tasklet in prev_subgraph.tasklets:
                            state, node = None, None
                            for n, s in sdfg.all_nodes_recursive():
                                if isinstance(n, nodes.Tasklet) and n.name == tasklet.name:
                                    state = s
                                    node = n
                                    break
                            assert (state is not None) and (node is not None)
                            outdata = [e.data.data for e in state.out_edges(node)][0]
                            if arr == outdata:
                                previdx = j
                                break
                        if previdx > -1:
                            break
                    assert previdx > -1
                    prev_subgraph = reversed_subgraphs[previdx]
                    prev_variables = [str(v) for v in prev_subgraph.variables]
                    for k, _ in arrdesc.items():
                        if isinstance(k, str):
                            tokens = k.split('*')
                            # print(arr, tokens)
                            gather = [v in tokens for v in prev_variables]
                            gather_filter = [v for v in prev_variables if v in tokens]
                            gather_corr = ','.join([str(gather_filter.index(t)) for t in tokens])
                            reduce = [v not in tokens for v in prev_variables]
                            scatter = [v in tokens for v in variables]
                            scatter_filter = [v for v in variables if v in tokens]
                            scatter_corr = ','.join([str(scatter_filter.index(t)) for t in tokens])
                            bcast = [v not in tokens for v in variables]
                            global_shape = ','.join([f"S{t}" for t in tokens])
                            prev_local_shape = ','.join([f"S{t}G{previdx}" for t in tokens])
                            local_shape = ','.join([f"S{t}G{i}" for t in tokens])
                            code += f"""
    grid{previdx}_{arr}_gather = dace.comm.Cart_sub(grid{previdx}, {gather}, exact_grid=0)
    grid{previdx}_{arr}_reduce = dace.comm.Cart_sub(grid{previdx}, {reduce})
    grid{previdx}_{arr}_subarray = dace.comm.Subarray([{global_shape}], [{prev_local_shape}], dctype, process_grid=grid{previdx}_{arr}_gather, correspondence=({gather_corr}))
    grid{i}_{arr}_scatter = dace.comm.Cart_sub(grid{i}, {scatter}, exact_grid=0)
    grid{i}_{arr}_bcast = dace.comm.Cart_sub(grid{i}, {bcast})
    grid{i}_{arr}_subarray = dace.comm.Subarray([{global_shape}], [{local_shape}], dctype, process_grid=grid{i}_{arr}_scatter, correspondence=({scatter_corr}))
    
    dace.comm.Reduce(grid{previdx}_{arr}, 'MPI_SUM', grid=grid{previdx}_{arr}_reduce)
    grid{i}_{arr} = np.empty_like(grid{previdx}_{arr}, shape=[{local_shape}])
    dace.comm.Redistribute(grid{previdx}_{arr}, grid{previdx}_{arr}_subarray, grid{i}_{arr}, grid{i}_{arr}_subarray)
    dace.comm.Bcast(grid{i}_{arr}, grid=grid{i}_{arr}_bcast)
                            """
                            break

        # for contraction, tasklet in zip(contractions, subgraph.tasklets):
        for contraction in contractions:

            einsum = contraction[2]
            operation = contraction[4]
            tokens = re.split(',|->', einsum)
            contr = list(set(tokens[0]).intersection(set(tokens[1])))

            found = False
            for tasklet in subgraph.tasklets:
                for n, s in sdfg.all_nodes_recursive():
                    if isinstance(n, nodes.Tasklet) and n.name == tasklet.name:
                        state = s
                        node = n
                        break
                assert (state is not None) and (node is not None)
                outtoken =  [''.join(re.split(',', str(e.data.subset).replace(' ', ''))) for e in state.out_edges(node)][0]   
                if outtoken == tokens[2]:
                    found = True
                    break
            assert found

            # for n, s in sdfg.all_nodes_recursive():
            #     if isinstance(n, nodes.Tasklet) and n.name == tasklet.name:
            #         state = s
            #         node = n
            #         break
            cinputs = [(e.data.data, ''.join(re.split(',', str(e.data.subset).replace(' ', '')))) for e in state.in_edges(node)]
            print(tokens)
            print(cinputs)
            cinput0 = next(data for data, t in cinputs if t == tokens[0])
            cinput1 = next(data for data, t in cinputs if t == tokens[1])
            # cinput1 = [e.data.data for e in state.in_edges_by_connector(node, '__in1')][0]
            # cinputs = [e.data.data for e in state.in_edges(node)]
            coutput = [e.data.data for e in state.out_edges(node)][0]
            if operation in ('GEMM', 'TDOT'):
                axes_a = [tokens[0].index(t) for t in contr]
                axes_b = [tokens[1].index(t) for t in contr]
                outtoken = [t for t in tokens[0] if t not in contr] + [t for t in tokens[1] if t not in contr]
                out_axes = [outtoken.index(t) for t in tokens[2]]
                # if operation == 'GEMM':
                #     cinput0, cinput1 = cinput1, cinput0
                #     axes_a, axes_b = axes_b, axes_a
                code += f"""
    grid{i}_{coutput} = np.tensordot(grid{i}_{cinput0}, grid{i}_{cinput1}, axes=({axes_a}, {axes_b}), out_axes={out_axes})
                """
            else:
                params = set(tokens[0]).union(set(tokens[1]))
                ranges = [f"0:S{p}G{i}" for p in params]
                if contr:
                    code += f"""
    grid{i}_{coutput} = np.zeros(({','.join([f"S{t}G{i}" for t in tokens[2]])}), dtype=nptype)
    for {','.join(params)} in dace.map[{','.join(ranges)}]:
        grid{i}_{coutput}[{','.join(tokens[2])}] += grid{i}_{cinput0}[{','.join(tokens[0])}] * grid{i}_{cinput1}[{','.join(tokens[1])}]
                    """
                else:
                    code += f"""
    grid{i}_{coutput} = np.empty(({','.join([f"S{t}G{i}" for t in tokens[2]])}), dtype=nptype)
    for {','.join(params)} in dace.map[{','.join(ranges)}]:
        grid{i}_{coutput}[{','.join(tokens[2])}] = grid{i}_{cinput0}[{','.join(tokens[0])}] * grid{i}_{cinput1}[{','.join(tokens[1])}]
                    """
            
            visited_arrays.add(coutput)

        contraction_list = contraction_list[contractions_num:]

    if i == len(subgraphs) - 1:
        tokens = [t for t in tokens[2]]
        reduce = [v not in tokens for v in variables]
        code += f"""
    grid{i}_{coutput}_reduce = dace.comm.Cart_sub(grid{i}, {reduce})
    dace.comm.Allreduce(grid{i}_{coutput}, 'MPI_SUM', grid=grid{i}_{coutput}_reduce)
    return grid{i}_{coutput}
        """
    
    # Main
    code += f"""
if __name__ == "__main__":

    from mpi4py import MPI
    commworld = MPI.COMM_WORLD
    rank = commworld.Get_rank()
    size = commworld.Get_size()

    if size not in pgrid0:
        raise ValueError("Selected number of MPI processes is not supported.")
    
    sdfg = None
    if rank == 0:
        sdfg = deinsum_program.to_sdfg(simplify=True)
    func = utils.distributed_compile(sdfg, commworld)

    grid_dims = set()
    """

    for i, subgraph in enumerate(reversed(subgraphs_res)):
        code += f"""
    grid_dims.update(pgrid{i}[size])
        """
    
    code += f"""
    lcm = np.lcm.reduce(list(grid_dims))
    S = np.int32(2 * lcm)
    """

    for i, subgraph in enumerate(reversed(subgraphs_res)):
        code += f"""
    PG{i} = pgrid{i}[size]
    SG{i} = [S // np.int32(p) for p in PG{i}]
        """

    for arr, shape in inputs.items():
        shape = shape.split(',')
        code += f"""
    {arr} = np.arange(S**{len(shape)}, dtype=nptype).reshape({','.join('S' * len(shape))}).copy()
        """
    
    code += f"""
    ##### Reference #####
    if rank == 0:
        ref = oe.contract('{desc[0]}', {', '.join(sorted(list(inputs.keys())))})
    commworld.Barrier()

    ##### Deinsum #####
    """

    for i, subgraph in enumerate(reversed(subgraphs_res)):
        code += f"""
    cart_comm = commworld.Create_cart(pgrid{i}[size])
    coords = cart_comm.Get_coords(rank)
        """
        sgrid = [f"S{v}G{i}" for v in subgraph.variables]
        for arr, shape in inputs.items():
            if not arr.startswith(f"grid{i}"):
                continue
            shape = shape.split(',')
            indices = [sgrid.index(s) for s in shape]
            code += f"""
    l{arr} = {arr}[{','.join([f"coords[{idx}] * SG{i}[{idx}] : (coords[{idx}] + 1) * SG{i}[{idx}]" for idx in indices])}].copy()
            """
    
    code += f"""
    val = func({','.join([f"{arr}=l{arr}" for arr in inputs.keys()])},
               {','.join([f"S{v} = S" for v in allvars])},
    """
    
    for i, subgraph in enumerate(reversed(subgraphs_res)):
        code += f"""
               {','.join([f"P{v}G{i} = PG{i}[{idx}]" for idx, v in enumerate(subgraph.variables)])},
               {','.join([f"S{v}G{i} = SG{i}[{idx}]" for idx, v in enumerate(subgraph.variables)])},
        """
    code += ')'

    tokens = re.split(',|->', desc[-1])
    tokens = [t for t in tokens[-1].strip()]
    subgraph = subgraphs_res[0]
    i = len(subgraphs_res) - 1
    sgrid = [f"S{v}G{i}" for v in subgraph.variables]
    shape = [f"S{t}G{i}" for t in tokens]
    indices = [sgrid.index(s) for s in shape]

    code += f"""
    if rank > 0:
        commworld.Send(val, 0)
    else:
        out = np.ndarray([{','.join(['S'] * len(tokens))}], dtype=nptype)
        out[{','.join([f"coords[{idx}] * SG{i}[{idx}] : (coords[{idx}] + 1) * SG{i}[{idx}]" for idx in indices])}] = val
        
        buf = np.ndarray([{','.join([f"SG{i}[{idx}]" for idx in indices])}], dtype=nptype)
        for r in range(1, size):
            commworld.Recv(buf, r)
            coords = cart_comm.Get_coords(r)
            out[{','.join([f"coords[{idx}] * SG{i}[{idx}] : (coords[{idx}] + 1) * SG{i}[{idx}]" for idx in indices])}] = buf
    """

    code += f"""
        print(f"\\nRelative error: {{np.linalg.norm(out-ref) / np.linalg.norm(ref)}}", flush=True)
        assert(np.allclose(out, ref))
    
    commworld.Barrier()
    """

    f = open("tensor_tmp.py", "w")
    f.write(code)
    f.close()
