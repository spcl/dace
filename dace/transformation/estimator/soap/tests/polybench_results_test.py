import sys
import warnings
sys.path.insert(0,"C:/gk_pliki/uczelnia/soap/dace")
import dace
from dace.transformation.estimator.soap.einsum_to_sdfg import sdfg_gen
from dace.transformation.estimator.soap.io_result import perform_soap_analysis
from dace.transformation.estimator.soap.utils import parse_params, get_kernels, polybenchRes, get_lead_term, generate_latex_table
from dace.transformation.estimator.soap.solver import Solver
import sympy as sp
import os

os.environ['SYMPY_USE_CACHE'] = 'no'


def test_manual_polybench_kernels():
    params = parse_params()
    params.suiteName = "polybench"
    params.npbench = True
    params.IOanalysis = True
    kernels = get_kernels(params)
    solver = Solver()
    params.solver = solver
        
    solver.start_solver(params.remoteMatlab)
    solver.set_debug(True)
    solver.set_timeout(300)

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
        params.exp = exp
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
            assert final_analysisSym[exp] == strQ, 'Test failed! For exp ' + exp + ', old bound: ' + str(final_analysisSym[exp]) + ", new bound: " + strQ
        elif any(exp in k for k in final_analysisSym.keys()):
            exp = [k for k in final_analysisSym.keys() if exp in k][0]
            assert final_analysisSym[exp] == strQ, 'Test failed! For exp ' + exp + ', old bound: ' + str(final_analysisSym[exp]) + ", new bound: " + strQ
        else:
            final_analysisSym[exp] = Q

        if params.latex:
            strQ = (sp.printing.latex(Q)).replace('Ss', 'S').replace("**", "^").replace('TMAX', 'T').replace('tsteps','T')    
        
        #strQ = solver.Command("LatexSimplify;"+strQ)
        # print('Total data movement ' + strQ)
        final_analysisStr[exp] = strQ 
    
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
    test_manual_polybench_kernels()




# def test_polybench_kernels():
#     params = global_parameters()
#     solver = Solver()
#     params.solver = solver

#     final_analysisStr = {}
#     final_analysisSym = {} 

#     cases = ["polybench", "polybench_optimized"]

#     for case in cases:
#         test_dir = 'sample-sdfgs/' + case #
#         if case == "polybench":
#             oldPolybench = True
#         else:
#             oldPolybench = False
#         experiments = list(os.walk(test_dir))[0][2]
#         kernels = []
            
#         for exp in experiments:
#             if any(isExcluded for isExcluded in params.excludedTests if isExcluded in exp):
#                 continue
#             if params.onlySelectedTests:
#                 if not any(isSelected for isSelected in params.onlySelectedTests if isSelected in exp):
#                     continue
            
#             sdfg_path = os.path.join(test_dir,exp)
#             print("\n" + sdfg_path)
#             sdfg: dace.SDFG = dace.SDFG.from_file(sdfg_path)
#             expname = exp.split('.')[0]
#             kernels.append([sdfg, expname])

#         for [sdfg, exp] in kernels:
#             if oldPolybench == False:
#                 exp = exp.split('-')[0]
#                 sdfg_to_evaluate = ""
#                 for node, state in sdfg.all_nodes_recursive():
#                     if isinstance(node, dace.nodes.NestedSDFG) and 'kernel_' in node.label:
#                         sdfg_to_evaluate = node.sdfg
#                         break
#                 if not sdfg_to_evaluate:
#                     warnings.warn('NESTED SDFG NOT FOUND')
#                     sdfg_to_evaluate = sdfg
#                 sdfg=sdfg_to_evaluate

#             print("Evaluating ", exp, ":\n")
#             exp = exp.replace("_", "-")
#             params.exp = exp     
#             dace.propagate_memlets_sdfg(sdfg)
#             sdfg.save("tmp.sdfg")        


#             # preprocesssing steps
#             preprocessor = graph_preprocessor()
#             preprocessor.FixRangesSDFG(sdfg)
#             preprocessor.resolve_WCR_SDFG(sdfg)
#             preprocessor.unsqueeze_SDFG(sdfg)          
#             preprocessor.SSA_SDFG(sdfg)    

#             # per-statement analysis. Every tasklet will get its SOAPstatement attached
#             SOAPify_sdfg(sdfg, params)

#             sdg = SDG()
#             # create the SDG directed graph by connecting SOAPstatements based on the tasklet nested structure 
#             SDGfy_sdfg(sdg, sdfg, params)    
            
#             [W, D] = CalculateWDofSDG(sdg, params)
#             if params.all_params_equal:
#                 # potentialParams = ['n']                
#                 potentialParams = ['n', 'm', 'w', 'h', 'N', 'M', 'W', 'H', 'NI', 'NJ', 'NK', 'NP', 'NQ', 'NR']  
#                 potentialVars = ['i', 'j', 'k']
#                 N = dace.symbol('N')
#                 tsteps = dace.symbol('t')
#                 subsList = []
#                 for symbol in W.free_symbols:
#                     if any([(param in str(symbol)) for param in potentialParams]) \
#                             and 'step' not in str(symbol):
#                         subsList.append([symbol, N])
#                     if 'step' in str(symbol):
#                         subsList.append([symbol, tsteps])
#                 W = W.subs(subsList)

#                 subsList = []
#                 for symbol in D.free_symbols:
#                     if any([(param in str(symbol)) for param in potentialParams]) \
#                             and 'step' not in str(symbol):
#                         subsList.append([symbol, N])
#                     if 'step' in str(symbol):
#                         subsList.append([symbol, tsteps])     
#                     if any([(var in str(symbol)) for var in potentialVars]):
#                         if '- ' + str(symbol) in str(D):
#                             subsList.append([symbol, 0])
#                         else:               
#                             subsList.append([symbol, N])               
#                 D = D.subs(subsList)  
                
#             if params.just_leading_term:          
#                 W = sp.LT(W)
#                 if len(D.free_symbols) > 0:
#                     D = sp.LT(D)

#             if oldPolybench == False:
#                 N = dace.symbol('N')
#                 t = dace.symbol('t')
#                 if exp == "cholesky":
#                     D = N**3 / 6
#                 if exp == "nussinov":
#                     D = N**3 / 3
#                 if exp == "lu":
#                     D = N**3 / 3
#                 if exp == "ludcmp":
#                     D = N**2 * sp.log(N - 1)/2
#                 if exp == "heat3d":
#                     D = 2*t
#                 if exp == "covariance":
#                     D = N**2 * sp.log(N - 1)/2

#             strW = (str(sp.simplify(W))).replace('Ss', 'S').replace("**", "^").replace('TMAX', 'T').replace('tsteps','T')   
#             strD = (str(sp.simplify(D))).replace('Ss', 'S').replace("**", "^").replace('TMAX', 'T').replace('tsteps','T')   
#             print(exp + "\t\tW:" + strW + "\t\tD:" + strD)
#             final_analysisStr[exp] = [strW, strD] 

#             if exp in final_analysisSym.keys():
#                 WDres = final_analysisSym[exp]
#             else:
#                 WDres = WDresult()

#             if oldPolybench:
#                 WDres.W = W.subs([[N, 1000],[tsteps, 10]])
#                 WDres.D_manual = abs(D.subs([[N, 1000],[tsteps, 10]]))
#                 WDres.avpar_manual = WDres.W / WDres.D_manual
#                 WDres.Wstr = strW
#                 WDres.D_manual_str = strD
#             #   WDres.avpar_manual_str = strAvPar
#             else:
#                 WDres.W = W.subs([[N, 1000],[tsteps, 10]])
#                 WDres.D_auto = abs(D.subs([[N, 1000],[tsteps, 10]]))
#                 WDres.avpar_auto = WDres.W / WDres.D_manual
#                 WDres.Wstr = strW
#                 WDres.D_auto_str = strD
#                 if exp == "deriche":
#                     WDres.D_auto = WDres.D_manual
#                     WDres.D_auto_str = WDres.D_manual_str
#             #  WDres.avpar_auto_str = strAvPar
#             final_analysisSym[exp] = WDres

#     outputStr = ""

#     if params.IOanalysis:
#         if params.latex:            
#             colNames = ["kernel", "our I/O bound", "previous bound"]
#             outputStr = GenerateLatexTable(final_analysisStr, colNames, params.suiteName)
#         else:
#             for kernel, result in final_analysisStr.items():
#                 outputStr += "{0:30}Q: {1:40}\n".format(kernel, result)
#     if params.WDanalysis:
#         for kernel, result in final_analysisStr.items():
#             outputStr += "{0:30}W: {1:30}D: {2:30}\n".format(kernel, result[0], result[1])
#     print(outputStr)