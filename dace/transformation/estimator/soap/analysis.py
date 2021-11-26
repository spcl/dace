import os
os.environ['SYMPY_USE_CACHE'] = 'no'
import sympy


from dace.transformation.estimator.soap.utils import *
from dace.transformation.estimator.soap.sdg import SDG

# ------------------------------------
# SDG analysis functions
# ------------------------------------
    
def Perform_SDG_analysis(sdg : SDG, final_analysisStr : dict,
                final_analysisSym : dict,
                exp : str, params : global_parameters):
    if params.IOanalysis:
        # print("Number of nodes in the SDG: " + str(len(sdg.graph.nodes)))
        if nx.number_weakly_connected_components(sdg.graph) > 1:
            return
        [Q, _] = sdg.calculate_IO_of_SDG(params)
    
        if len(Q.free_symbols) > 0:
            Q = get_lead_term(Q)
        strQ = (str(sp.simplify(Q))).replace('Ss', 'S').replace("**", "^").\
                replace('TMAX', 'T').replace('tsteps','T').replace('dace_','').\
                replace('_0', '').replace('m', 'M').replace('n', 'N').replace('k', 'K').\
                replace('i', 'I').replace('j', 'J').replace('l', 'L')

        if exp in final_analysisSym.keys():
            if not final_analysisSym[exp] == strQ:
                print('Test failed! For exp ' + exp + ', old bound: ' + str(final_analysisSym[exp]) + ", new bound: " + strQ)
        else:
            final_analysisSym[exp] = Q

        if params.latex:
            strQ = (sp.printing.latex(Q)).replace('Ss', 'S').replace("**", "^").replace('TMAX', 'T').replace('tsteps','T')    
        
        #strQ = solver.Command("LatexSimplify;"+strQ)
        # print('Total data movement ' + strQ)
        final_analysisStr[exp] = strQ 