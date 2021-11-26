
import sys
import os
#sys.path.insert(0, os.getcwd())
sys.path.insert(0,"C:/gk_pliki/uczelnia/soap/dace")
import dace
from dace.transformation.estimator.soap.utils import *
from dace.transformation.estimator.soap.sdg import SDG
from dace.transformation.estimator.soap.analysis import Perform_SDG_analysis
from dace.transformation.estimator.soap.io_result import *


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
            outputStr = generate_latex_table(final_analysisStr, colNames, params.suiteName)
        else:
            for kernel, result in final_analysisStr.items():
                outputStr += "{0:30}Q: {1:40}\n".format(kernel, result)
    if params.WDanalysis:
        for kernel, result in final_analysisStr.items():
            outputStr += "{0:30}W: {1:30}D: {2:30}\n".format(kernel, result[0], result[1])
    print(outputStr)


if __name__ == "__main__":
    test_polybench_kernels()