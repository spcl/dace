import sys
import warnings
import dace
from dace import Config
from dace.transformation.estimator.soap.einsum_to_sdfg import sdfg_gen
from dace.transformation.estimator.soap.io_analysis import perform_soap_analysis
from dace.transformation.estimator.soap.utils import get_kernels, polybenchRes, get_lead_term, generate_latex_table
import sympy as sp
import os

os.environ['SYMPY_USE_CACHE'] = 'no'


def test_manual_polybench_kernels():
    Config.set("soap", "tests", "suite_name", value="manual_polybench")
    kernels = get_kernels()
    final_analysisStr = {}
    final_analysisSym = polybenchRes
    preprocessed = False

    for [sdfg, exp] in kernels:
        
        exp = exp.split('_')[0]
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
        if not preprocessed:
            # Parallelize as many loops as possible
            from dace.transformation.interstate import LoopToMap, RefineNestedAccess
            sdfg.apply_transformations_repeated([LoopToMap, RefineNestedAccess])
           
            dace.propagate_memlets_sdfg(sdfg)
            sdfg.save("tmp.sdfg", hash=False)        

        else:
            sdfg = dace.SDFG.from_file("tmp.sdfg")

        soap_result = perform_soap_analysis(sdfg)
        Q = soap_result.Q
    
        if len(Q.free_symbols) > 0:
            Q = get_lead_term(Q)
        strQ = (str(sp.simplify(Q))).replace('Ss', 'S').replace("**", "^").\
                replace('TMAX', 'T').replace('tsteps','T').replace('dace_','').\
                replace('_0', '').replace('m', 'M').replace('n', 'N').replace('k', 'K').\
                replace('i', 'I').replace('j', 'J').replace('l', 'L')


        if exp in final_analysisSym.keys():
            assert strQ in final_analysisSym[exp], 'Test failed! For exp ' + exp + ', old bound: ' + str(final_analysisSym[exp]) + ", new bound: " + strQ
        elif any(exp in k for k in final_analysisSym.keys()):
            exp = [k for k in final_analysisSym.keys() if exp in k][0]
            assert final_analysisSym[exp] == strQ, 'Test failed! For exp ' + exp + ', old bound: ' + str(final_analysisSym[exp]) + ", new bound: " + strQ
        else:
            final_analysisSym[exp] = Q

        if Config.get("soap", "output", "latex"):
            strQ = (sp.printing.latex(Q)).replace('Ss', 'S').replace("**", "^").replace('TMAX', 'T').replace('tsteps','T')    
        
        #strQ = solver.Command("LatexSimplify;"+strQ)
        # print('Total data movement ' + strQ)
        final_analysisStr[exp] = strQ 
    
    
    outputStr = ""

    if Config.get("soap", "output", "latex"):            
        colNames = ["kernel", "our I/O bound", "previous bound"]
        outputStr = generate_latex_table(final_analysisStr, colNames, "manual polybench")
    else:
        for kernel, result in final_analysisStr.items():
            outputStr += "{0:30}Q: {1:40}\n".format(kernel, result)
    if Config.get("soap", "analysis", "wd_analysis"):
        for kernel, result in final_analysisStr.items():
            outputStr += "{0:30}W: {1:30}D: {2:30}\n".format(kernel, result[0], result[1])
    print(outputStr)



def test_npbench_polybench_kernels():
    Config.set("soap", "tests", "suite_name", value="npbench")
    kernels = get_kernels()
    final_analysisStr = {}
    final_analysisSym = polybenchRes
    preprocessed = False

    for [sdfg, exp] in kernels:
        
        exp = exp.split('_')[0]
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
        if not preprocessed:
            # Parallelize as many loops as possible
            from dace.transformation.interstate import LoopToMap, RefineNestedAccess
            sdfg.apply_transformations_repeated([LoopToMap, RefineNestedAccess])
           
            dace.propagate_memlets_sdfg(sdfg)
            sdfg.save("tmp.sdfg", hash=False)        

        else:
            sdfg = dace.SDFG.from_file("tmp.sdfg")

        soap_result = perform_soap_analysis(sdfg)
        Q = soap_result.Q
    
        if len(Q.free_symbols) > 0:
            Q = get_lead_term(Q)
        strQ = (str(sp.simplify(Q))).replace('Ss', 'S').replace("**", "^").\
                replace('TMAX', 'T').replace('tsteps','T').replace('dace_','').\
                replace('_0', '').replace('m', 'M').replace('n', 'N').replace('k', 'K').\
                replace('i', 'I').replace('j', 'J').replace('l', 'L')

        if exp in final_analysisSym.keys():
            assert strQ in final_analysisSym[exp], 'Test failed! For exp ' + exp + ', old bound: ' + str(final_analysisSym[exp]) + ", new bound: " + strQ
        elif any(exp in k for k in final_analysisSym.keys()):
            exp = [k for k in final_analysisSym.keys() if exp in k][0]
            assert final_analysisSym[exp] == strQ, 'Test failed! For exp ' + exp + ', old bound: ' + str(final_analysisSym[exp]) + ", new bound: " + strQ
        else:
            final_analysisSym[exp] = Q

        if Config.get("soap", "output", "latex"):
            strQ = (sp.printing.latex(Q)).replace('Ss', 'S').replace("**", "^").replace('TMAX', 'T').replace('tsteps','T')    
        
        #strQ = solver.Command("LatexSimplify;"+strQ)
        # print('Total data movement ' + strQ)
        final_analysisStr[exp] = strQ 
    
    
    outputStr = ""

    if Config.get("soap", "output", "latex"):            
        colNames = ["kernel", "our I/O bound", "previous bound"]
        outputStr = generate_latex_table(final_analysisStr, colNames, "manual polybench")
    else:
        for kernel, result in final_analysisStr.items():
            outputStr += "{0:30}Q: {1:40}\n".format(kernel, result)
    if Config.get("soap", "analysis", "wd_analysis"):
        for kernel, result in final_analysisStr.items():
            outputStr += "{0:30}W: {1:30}D: {2:30}\n".format(kernel, result[0], result[1])
    print(outputStr)


if __name__ == "__main__":
    test_manual_polybench_kernels()    
    test_npbench_polybench_kernels()
    