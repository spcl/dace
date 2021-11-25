
import sys
import os
#sys.path.insert(0, os.getcwd())
sys.path.insert(0,"C:/gk_pliki/uczelnia/soap/dace")
import dace
from dace.transformation.estimator.soap.utils import *
from dace.transformation.estimator.soap.sdg import SDG
from dace.transformation.estimator.soap.analysis import Perform_SDG_analysis

def test_sdg_constructor():
 #   test_sdfg = "sample-sdfgs/various/tal_example.sdfg"
    params = global_parameters()
    solver = Solver()
    params.solver = solver
    if params.IOanalysis:
        solver.start_solver(params.remoteMatlab)
        solver.set_debug(True)
    test_sdfg = "sample-sdfgs/polybench_optimized/2mm-perf.sdfg"
    params.exp = "fdtd2d"
    sdfg = dace.SDFG.from_file(test_sdfg)
    sdg = SDG(sdfg, params)
    Q = sdg.calculate_IO_of_SDG(params)
    a = 1
    

def test_polybench_kernels():
    params = parse_params()
    kernels = get_kernels(params)
    solver = Solver()
    params.solver = solver
    
    
    if params.IOanalysis:
        solver.start_solver(params.remoteMatlab)
        solver.set_debug(True)
        solver.set_timeout(300)

    final_analysisStr = {}
    final_analysisSym = polybenchRes
    preprocessed = False

    for [sdfg, exp] in kernels:
        
        exp = exp.split('-')[0]
        sdfg_to_evaluate = ""
        for node, state in sdfg.all_nodes_recursive():
            if isinstance(node, dace.nodes.NestedSDFG) and 'kernel_' in node.label:
                sdfg_to_evaluate = node.sdfg
                break
        if not sdfg_to_evaluate:
            warnings.warn('NESTED SDFG NOT FOUND')
            sdfg_to_evaluate = sdfg
        sdfg=sdfg_to_evaluate

        
        print("Evaluating ", exp, ":\n")
        params.exp = exp
        if not preprocessed:
            # Parallelize as many loops as possible
            from dace.transformation.interstate import LoopToMap, RefineNestedAccess
            sdfg.apply_transformations_repeated([LoopToMap, RefineNestedAccess])
            # Remove as many transients and views as possible
            # NOTE: We are breaking program semantics here
            # NOTE: This must be done after parallelization, because it may
            # generate dependencies among loop iterations.
            # TODO: (Maybe) make it a proper dace transformations
            # First pattern: Non-transient data -> Map Entries -> Transient/View
            # Second pattern: Transient/View -> Map Exits -> Non-transient data
            # from dace import data, nodes
            # for state in sdfg.nodes():
            #     for node in state.nodes():
            #         if not isinstance(node, nodes.AccessNode):
            #             continue
            #         desc = sdfg.arrays[node.data]
            #         if not (isinstance(desc, data.View) or desc.transient):
            #             continue
            #         in_edges = state.in_edges(node)
            #         out_edges = state.out_edges(node)
            #         is_read = False
            #         if len(in_edges) == 1:
            #             in_edge = in_edges[0]
            #             in_subset = in_edge.data.get_src_subset(in_edge, state)
            #             src = state.memlet_path(in_edge)[0].src
            #             if (in_subset and isinstance(src, nodes.AccessNode) and
            #                     not sdfg.arrays[src.data].transient):
            #                 is_read = True
            #         if is_read:
            #             composable = True
            #             for e in out_edges:
            #                 try:
            #                     subset = e.data.get_src_subset(e, state)
            #                     src_subset = copy.deepcopy(in_subset)
            #                     src_subset.compose(subset)
            #                 except:
            #                     composable = False
            #                     break
            #             if composable:
            #                 state.remove_edge(in_edge)
            #                 for e1 in out_edges:
            #                     for e2 in state.memlet_path(e1):
            #                         subset = e2.data.get_src_subset(e2, state)
            #                         src_subset = copy.deepcopy(in_subset)
            #                         src_subset.compose(subset)
            #                         e2.data.data = src.data
            #                         e2.data.src_subset = src_subset
            #                     state.remove_edge(e1)
            #                     state.add_edge(in_edge.src, in_edge.src_conn, e1.dst, e1.dst_conn, e1.data)
            #                 state.remove_node(node)
            #             continue
            #         is_write = False
            #         if len(out_edges) == 1:
            #             out_edge = out_edges[0]
            #             out_subset = out_edge.data.get_dst_subset(out_edge, state)
            #             dst = state.memlet_path(out_edge)[-1].dst
            #             if (out_subset and isinstance(dst, nodes.AccessNode) and
            #                     not sdfg.arrays[dst.data].transient):
            #                 is_write = True
            #         if is_write:
            #             composable = True
            #             for e in in_edges:
            #                 try:
            #                     subset = e.data.get_dst_subset(e, state)
            #                     dst_subset = copy.deepcopy(out_subset)
            #                     dst_subset.compose(subset)
            #                 except:
            #                     composable = False
            #                     break
            #             if composable:
            #                 state.remove_edge(out_edge)
            #                 for e1 in in_edges:
            #                     for e2 in state.memlet_path(e1):
            #                         subset = e2.data.get_dst_subset(e2, state)
            #                         dst_subset = copy.deepcopy(out_subset)
            #                         dst_subset.compose(subset)
            #                         e2.data.data = dst.data
            #                         e2.data.dst_subset = dst_subset
            #                     state.remove_edge(e1)
            #                     state.add_edge(e1.src, e1.src_conn, out_edge.dst, out_edge.dst_conn, e1.data)
            #                 state.remove_node(node)
            dace.propagate_memlets_sdfg(sdfg)
            sdfg.save("tmp.sdfg", hash=False)        

        else:
            sdfg = dace.SDFG.from_file("tmp.sdfg")

        sdg = SDG(sdfg, params)
        Perform_SDG_analysis(sdg, final_analysisStr, final_analysisSym, exp, params)
 
    if params.IOanalysis:
        solver.end_solver()
    
    outputStr = ""

    if params.IOanalysis:
        if params.latex:            
            colNames = ["kernel", "our I/O bound", "previous bound"]
            outputStr = GenerateLatexTable(final_analysisStr, colNames, params.suiteName)
        else:
            for kernel, result in final_analysisStr.items():
                outputStr += "{0:30}Q: {1:40}\n".format(kernel, result)
    if params.WDanalysis:
        for kernel, result in final_analysisStr.items():
            outputStr += "{0:30}W: {1:30}D: {2:30}\n".format(kernel, result[0], result[1])
    print(outputStr)


if __name__ == "__main__":
    test_polybench_kernels()