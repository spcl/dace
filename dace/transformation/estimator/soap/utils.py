from collections import defaultdict
from dataclasses import dataclass
import numpy as np
from sympy.core.compatibility import ordered
from dace.sdfg.nodes import *
from dace.subsets import Range
import sympy as sp
from typing import Optional, Tuple
import copy
from dace.sdfg.graph import MultiConnectorEdge
import dace
from dace.sdfg.nodes import *
from dace.subsets import Range, Indices
from dace import subsets
import re
import os
from dace.transformation.estimator.soap.solver import Solver
from dace.transformation.estimator.soap.einsum_to_sdfg import sdfg_gen
import time
import argparse
import sys
import networkx as nx

# FOR DEBUGGING/DEVELOPMENT PURPOSES ONLY!!!
# These parameters are specified in config_schema.yml and initialized in .dace.conf

# ---------------------------------------
#  CONFIG PARAMETERS
# ---------------------------------------
# -----------------------
# -- parallel schedule -- 
avail_par_setups = ["memory_independent", "memory_dependent"]
chosen_par_setup = avail_par_setups[0]

# default numerical parameters for the schedue generation
decompostition_params = [("p", 8), ("Ss", 32*1024), ("S0", 512), ("S1", 512), ("S2", 512), ("S3", 512)]

# -----------------------
# -- test parameters --
available_setups = ["old_tals_sdfgs", "c2dace", "npbench", # different polybench SDFGs
                    "einsum_string", "einsum_strings_from_file", 
                    "other"]
chosen_setup = available_setups[3]
only_selected_tests = ["doitgen"] #["lenet"] #["deriche", "symm", "mvt"]
excluded_tests = ["cholesky2", "outer", "ssa", "deriche", "adi"]
# einsum_string = 'ijk,jl,kl->il'
einsum_string = 'pi,qj,ijkl,rk,sl->pqrs'
# einsum_string = 'ik,kj->ij'

# this applies ONLY when chosen_setup = available_setups[5] (other)
# sdfg_path may either point to a single SDFG file or to a directory. 
# In the latter case, all sdfgs in the directtory will be evaluated
sdfg_path = 'tensors/test.sdfg'

# path to polybench kernels
abs_test_path  = 'C:/gk_pliki/uczelnia/soap/soap_code/sdg'

# ------------------
# -- solver setup --
use_remote_matlab_server = True
caching_solver_solutions = True
only_cached = False
solver_local_path = "C:\\gk_pliki\\uczelnia\\soap\\soap_code\\matlab"
solver_remote_access = 'galilei.inf.ethz.ch'  
solver_db_path = "dace/transformation/estimator/soap/solver_cache/solver_cache.txt"



# ----------------------------------------
# initialization and launch configurations
# ----------------------------------------
# TODO: take input arguments instead of relying on the Config
# TODO: path to SDFG folders
def get_kernels(suite_name : str, kernel_name : str = "", sdfg_path : str = ""):
    kernels = []
    selected_kernels = [kernel_name] if kernel_name != ""  \
            else Config.get("soap", "tests", "only_selected_tests") \
            if len(Config.get("soap", "tests", "only_selected_tests")) > 0  else []
    test_dir = os.path.join(Config.get("soap", "tests", "abs_test_path"), sdfg_path)
    if suite_name == "einsum": #params.suiteName == "einsum":
        dim = 30
        for einsum in Config.get("soap", "tests", "einsum_string"):            
            sdfg = sdfg_gen(einsum)
            kernels.append([sdfg, einsum])
        return kernels
    
    if suite_name == "npbench":
        experiments = list(os.walk(test_dir))[0][1]
                
        for exp in experiments:
            if any(isExcluded for isExcluded in \
                        Config.get("soap", "tests", "excluded_tests") \
                        if isExcluded in exp):
                continue
            if len(selected_kernels) > 0:
                if not any(isSelected for isSelected in \
                        selected_kernels \
                        if isSelected in exp):
                    continue              
            try:
                sdfg_path = os.path.join(test_dir,exp, exp + "_dace.py")
                kernels += sdfgs_from_npbench(sdfg_path)
            except:
                pass

        return kernels
            # for n, k in zip(experiments, kernels):
            #     k.save(f'{n}.sdfg')
              
    if suite_name == "manual_polybench":
        experiments = list(os.walk(test_dir))[0][2]
            
        for exp in experiments:
            if any(isExcluded for isExcluded in \
                        Config.get("soap", "tests", "excluded_tests") \
                        if isExcluded in exp):
                continue
            if len(selected_kernels) > 0:
                if not any(isSelected for isSelected in \
                        selected_kernels \
                        if isSelected in exp):
                    continue  
            
            try: 
                sdfg_path = os.path.join(test_dir,exp)
                print("\n" + sdfg_path)
                sdfg: dace.SDFG = dace.SDFG.from_file(sdfg_path)
                expname = exp.split('.')[0]
                kernels.append([sdfg, expname])
            except:
                pass
        

    return kernels




import dace.frontend.python.parser
import importlib
import sys
import os

def sdfgs_from_npbench(path):
    fname = os.path.basename(path).split('.')[0]
    sys.path.append(os.path.dirname(path))
    mod = importlib.import_module(fname)
    progs = [getattr(mod, k) for k in dir(mod) if isinstance(getattr(mod, k), dace.frontend.python.parser.DaceProgram)]
    kernels = []
    for prog in progs:
        try:
            sdfg = prog.to_sdfg()
            if 'kernel' not in sdfg.name:
                continue
            sdfg.expand_library_nodes()
            # sdfg.save('tmp.sdfg')
            kernels.append((sdfg, (fname + "_" + sdfg.name)))
        except:
            pass
        
    return kernels




# ----------------------------------------
# various helper functions
# ----------------------------------------
def rng_global2dict(ranges_scopes):
    rng_dict = {}
    for scope, ranges in ranges_scopes.items():
        rng_dict = {**rng_dict, **rng_list2dict(ranges)}
    return rng_dict


def rng_list2dict(ranges) -> Dict[str, Tuple]:
    return dict([(it, (rng_low, rng_high)) for (it, rng_low, rng_high) in ranges])


def rng_dict2list(ranges_dict):
    return list(ranges_dict.items())


def strip(array_name_with_version : str) -> str:
    return '_'.join(array_name_with_version.split('_')[:-1])


def get_access_from_memlet(memlet : dace.Memlet, iter_vars) -> Tuple[str, str, Tuple[int]]:
    arrayName = memlet.data # + "_" + str(memlet.soap_array.version)
    baseAccess = ""
    offsets = []
    for looop in memlet.subset.ndrange():
        if looop[0] != looop[1] or looop[2] != 1:
            raise ValueError('Malformed program')      
        # check if we are updating only a subset of array. If yes, then this statement
        # does NOT count the array as a whole
        if not looop[0].free_symbols:            
            # TODO: new experimental. Instead of discarding, we need to handle it
            if len(memlet.subset.ndrange()) > 1:
                # return (None, None, None)
                continue
            # this is the case when we have a WCR on a transient scalar
            else:
                continue

        # a) remove constants from the access function (e.g., A[i + N] -> A[i])
        # b) split dimensions with multiple iteration variables into multiple dimensions (e.g., A[i-k] -> A[i,k])
        [access, offset] = extract_access(looop[0], iter_vars)  
        if access:            
            baseAccess += str(access) + "*"
            offsets += offset
            # subtract the currently accessed iteration variable to avoid situations like e.g.,:
            # A[i,i] -> base_access = 'i*i'. It should be base_access = 'i' (just one i). 
            iter_vars = iter_vars - access.free_symbols
    baseAccess = baseAccess[:-1]

    return (arrayName, baseAccess, tuple(offsets))


def base_in_list(base_str : str, swaplist : Dict[str, str]) -> bool:
    return any(any((re.search(r'\b%s\b' % iter, swap_el))  for swap_el in swaplist.keys())
               for iter in base_str.split('*')) 


def swap_in_string(base_str, swaplist, inv_swaplist) -> str:
    if not base_in_list(base_str, swaplist) and not base_in_list(base_str, inv_swaplist):
        return base_str
    if base_in_list(base_str, swaplist) and base_in_list(base_str, inv_swaplist):
        for iter_new, iter_old in swaplist.items():
            base_str = re.sub(iter_new , "tmppp", base_str)
            base_str = re.sub(iter_old , iter_new, base_str)
            base_str = re.sub("tmppp" , iter_old, base_str)
        return base_str                          
    else:
        if base_in_list(base_str, swaplist):
            cur_swaplist = swaplist
        else:
            cur_swaplist = inv_swaplist  
        rep = dict((re.escape(k), v) for k, v in cur_swaplist.items()) 
        pattern = re.compile( r"\b" + (r"\b|\b".join(rep.keys())) + r"\b" )
        return pattern.sub(lambda m:rep[re.escape(m.group(0))], base_str)
        


# ------------------------------------------- #
# --------- SYMBOLIC PROCESSING ------------- #
# ------------------------------------------- #

# remove floors and ceilings
def int_to_real(expr):
    """ Remove floors and ceilings from the expression
    """
    nexpr = expr
    if not isinstance(expr, sp.Basic):
        return expr

    a = sp.Wild('a')
    processed = 1
    while processed > 0:
        processed = 0
        for ceil in nexpr.find(sp.ceiling):
            # Simple ceiling
            m = ceil.match( sp.ceiling(a))
            if m is not None:
                nexpr = nexpr.subs(ceil, m[a])
                processed += 1
                continue
            
    processed = 1
    while processed > 0:
        processed = 0
        for fl in nexpr.find(sp.floor):
            # Simple ceiling
            m = fl.match( sp.floor(a))
            if m is not None:
                nexpr = nexpr.subs(fl, m[a])
                processed += 1
                continue
            
    
    return nexpr


def compare_Q(Q_1, Q_2):
    Ss = sp.sympify('Ss')
    Q_new_val = Q_1.subs(Ss, 10000)
    subsList = []
    for symbol in Q_new_val.free_symbols:
        subsList.append([symbol, 100000])
    Q_new_val = Q_new_val.subs(subsList)

    if Q_2 != 0:
        Q_old_val = Q_2.subs(Ss, 10000)
        subsList = []
        for symbol in Q_old_val.free_symbols:
            subsList.append([symbol, 100000])
        Q_old_val = Q_old_val.subs(subsList)
        
        return Q_new_val > Q_old_val
    
    else:
        return True
    
    
def compare_st(subgraph_st, subgraph_opt, Q_old_val = -1):
    Ss = sp.sympify('Ss')
    Q_new_val = subgraph_st.Q.subs(Ss, 10000)
    subsList = []
    for symbol in Q_new_val.free_symbols:
        subsList.append([symbol, 100000])
    Q_new_val = Q_new_val.subs(subsList)

    if Q_old_val == -1:
        Q_old_val = subgraph_opt.Q.subs(Ss, 10000)
        subsList = []
        for symbol in Q_old_val.free_symbols:
            subsList.append([symbol, 100000])
        Q_old_val = Q_old_val.subs(subsList)
        
    

    # decide which subgraph is larger (we prefer larger merges)
    if len(subgraph_st.name.split(';')) >= len(subgraph_opt.name.split(';')):
        larger_subgraph = subgraph_st
        larger_sg_Q = Q_new_val
        smaller_subgraph = subgraph_opt
        smaller_sg_Q = Q_old_val
    else:
        larger_subgraph = subgraph_opt
        larger_sg_Q = Q_old_val
        smaller_subgraph = subgraph_st
        smaller_sg_Q = Q_new_val
    
    # smaller subgraph must have much smaller Q to be preferable:
    if smaller_sg_Q != 0 and (1.2 * smaller_sg_Q < larger_sg_Q):
        return [smaller_subgraph, smaller_sg_Q]
    else:
        return [larger_subgraph, larger_sg_Q]


# ------------------------------------------------------ #
# --------------- POLYBENCH SUITE ---------------------- #
# ------------------------------------------------------ #


def MatchPolybenchKernelNames(exp : str) -> str:     
    exp = exp.split('-')[0]
    if exp == "k2mm":
        exp = "2mm"
    if exp == "k3mm":
        exp = "3mm"
    if "floyd" in exp:
        exp = "floyd-warshall"
    if exp == "j1d":
        exp = "jacobi1d"
    if exp == "j2d":
        exp = "jacobi2d"
    if exp == "heat":
        exp = "heat3d"
    if exp == "seidel":
        exp = "seidel2d"
    return exp



def rng_to_subset(ranges : dict):
    iter_vars = list(map(dace.symbol, [v for v in ranges.keys()]))
    return [(i, i, 1) for i in iter_vars]


def d2sp(expression):
    return sp.sympify(str(expression).replace('N', 'n'))


def extract_access(accessFun, itervars):
    # remove parameters from the access function (e.g., A[N*i] -> A[i])
    allVars = accessFun.free_symbols
    params = allVars - itervars
    if params != allVars:
        for param in params:
            accessFun = accessFun.subs(param, 1)
    else:
        return [None, None]

    accessPol = sp.Poly(accessFun)
    accessVars = accessPol.gens
    baseAccess = sp.sympify(0)
    offset = sp.sympify(0)
    for k, c in zip(accessPol.monoms(), accessPol.coeffs()):
        monom = c * sp.prod(x**k1 for x, k1 in zip(accessVars,k))
        if sum(k) > 0:            
            baseAccess += monom
        else:
            offset = monom
    baseAccess = sp.prod(baseAccess.free_symbols)

    # if the access has more than one iteration variable (e.g., A[k - i]), we replace it with A[k,i])
    # then, the dimension of the offset must match
    offset = [offset] * len(baseAccess.free_symbols)
    return [baseAccess, offset] 



def get_lead_term(expression):
    q_pol = sp.Poly(expression)
    q_vars = list(q_pol.gens)
    q_monoms = [list(monom) for monom in q_pol.monoms()]
    q_coeffs = list(q_pol.coeffs())
    
    # check if one of the generators is 1/S (instead of S). Then, we need to flip it.
    S = sp.sympify('Ss')
    for i in range(len(q_vars)):              
        if q_vars[i] == S:
            a = 1
            
        if q_vars[i] == 1/S:
            q_vars[i] = S
            for monom in q_monoms:
                monom[i] = -monom[i]
            
    max_deg = max([sum(monom) for monom in q_monoms])
    simpQ = sp.sympify(0)
    for k, c in zip(q_monoms, q_coeffs):
        if sum(k) >= max_deg:
            monom = c * sp.prod(x**k1 for x, k1 in zip(q_vars,k))
            simpQ += monom
    return simpQ


# Checks whether two base accessses are the same. It is NOT enough to just have
# a string comparison (e.g., "i*j" == "i*j"), as sometimes the same iteration variables
# have different names (e.g., A[dace_tmp_3] == A[dace_tmp_4] in jacobi1d)
def eq_accesses(base_access1, base_access2):
    accesses_1 = base_access1.split('*')
    accesses_2 = base_access2.split('*')
    if any((acc1 in accesses_2) for acc1 in accesses_1):
        return base_access1 == base_access2
    else:
        # if there are no common iteration variables, we conservatively assume they are the same
        # (e.g., A[dace_tmp_3] == A[dace_tmp_4])
        return len(accesses_1) == len(accesses_2)


def generate_latex_table(final_analysis, colNames, suiteName):
    outputStr = ""

    # table header
    outputStr += "\\begin{table} \n" + \
            "\\begin{tabular}{l|" + "l" * len(colNames) + "} \n" + \
            "\\toprule \n" + \
            "".join([" & \\textbf{" + cn + "}" for cn in colNames]) + "\\\\ \n" + \
            "\\midrule"

    # table first (shared, rotated) column 
    outputStr += "\multirow{" + str(len(final_analysis)) + "}{*}{\\begin{turn}{90}\\textbf{" + \
        suiteName + "}\end{turn}}"

    # table contents
    for kernel, result in final_analysis.items():
        parsedKernel = kernel.replace('_', '\\_')

        parsedRes = re.sub(r"([A-Z])([a-zA-Z])", r"\1_\2",result)

        knownBound = [prevBounds[ker] for ker in prevBounds.keys() if ker in kernel]
        if len(knownBound) > 0:
            knownBound = knownBound[0]
        else:
            knownBound = "---"

        outputStr += "& " + parsedKernel + " & $" + parsedRes + "$ & $" + \
            knownBound + "$" + " & [empty]"*(len(colNames) - 3) + "\\\\ \n"

    outputStr += "\\bottomrule \n" + \
        "\\end{tabular} \n" + \
	    "\\caption{\\textmd{[empty]}} \n" + \
	    "\\label{[empty]} \n" + \
        "\end{table}"
    return outputStr


# ------------------------------------------
# pretty latex output
# ------------------------------------------

    #     if Config.get("soap", "output", "latex"):
    #         strQ = (sp.printing.latex(Q)).replace('Ss', 'S').replace("**", "^").replace('TMAX', 'T').replace('tsteps','T')    
        
    #     final_analysisStr[exp] = strQ 
    
    
    # outputStr = ""

    # if Config.get("soap", "output", "latex"):            
    #     colNames = ["kernel", "our I/O bound", "previous bound"]
    #     outputStr = generate_latex_table(final_analysisStr, colNames, "manual polybench")
    # else:
    #     for kernel, result in final_analysisStr.items():
    #         outputStr += "{0:30}Q: {1:40}\n".format(kernel, result)
    # if Config.get("soap", "analysis", "wd_analysis"):
    #     for kernel, result in final_analysisStr.items():
    #         outputStr += "{0:30}W: {1:30}D: {2:30}\n".format(kernel, result[0], result[1])
    # print(outputStr)


# ------------------------------------------
# plotting with matplotlib
# ------------------------------------------




# def plotWD(final_analysisSym):
#     final_analysisSym_list = list(final_analysisSym)
#     final_analysisSym_all = final_analysisSym
#     l = len(final_analysisSym_list)
#     middle = int(l/2)
#     for part in range(2):
#         if part == 0:
#             final_analysis_part_list = final_analysisSym_list[:middle]
#         else:
#             final_analysis_part_list = final_analysisSym_list[middle:]
        
#         final_analysisSym = { exp : final_analysisSym_all[exp] for exp in final_analysis_part_list }
#         plt.rcParams["figure.figsize"] = (20,5)
#         plt.rc('text', usetex=True)
#         font = {'family' : 'arial',
#             #  'weight' : 'bold',
#                 'size'   : 25}
#         size_labels = 21
                
#         plt.rc('font', **font)
#         fig, ax = plt.subplots()
#         labels = list(name.replace("floyd-warshall","f-w").replace("gramschmidt","g-sch") for name in final_analysisSym.keys())
#         depths_manual = list([res[1].D_manual for res in final_analysisSym.items()])
#         depths_auto = list([res[1].D_auto for res in final_analysisSym.items()])
#         works = list([res[1].W for res in final_analysisSym.items()])

#         depths_manual_str = list([res[1].D_manual_str.replace('N - 1', 'N').replace('N - 2', 'N'). \
#                 replace('*', '').replace('log', '\log') for res in final_analysisSym.items()])
#         depths_auto_str = list([res[1].D_auto_str.replace('N - 1', 'N').replace('N - 2', 'N'). \
#                 replace('*', '').replace('log', '\log') for res in final_analysisSym.items()])
#         works_str = list([res[1].Wstr.replace('*', '') for res in final_analysisSym.items()])

#         x = np.arange(len(labels)) 
#         width = 0.27
#         rectsW = ax.bar(x-width, works, width, label = r'$P=1$', color = 'gray')
#         rectsD_manual = ax.bar(x, depths_manual, width, label = r'$P=\infty$, manual par.', color = 'turquoise')
#         rectsD_auto = ax.bar(x+width, depths_auto, width, label = r'$P=\infty$, auto par.', color = 'lightcoral')
#         ax.set_ylabel('Idealized runtime [cycles]')
#         # ax.set_title('Scores by group and gender')
#         ax.set_xticks(x)
#         ax.set_xticklabels(labels, rotation = 90)
#         ax.legend(ncol=len(x))
#         ax.set_ylim([1,100000000000000])
#         plt.yscale('log')
#         plt.xlim(-0.5,len(x)-.5)

#         for rect, work_str in zip(rectsW, works_str):
#             height = rect.get_height()
#             lab = r"$" + work_str +"$"
#             ax.annotate(lab,
#                         xy=(rect.get_x() + rect.get_width() / 2, height),
#                         xytext=(0, 3),  # 3 points vertical offset
#                         textcoords="offset points",
#                         ha='center', va='bottom', rotation=90, fontsize=size_labels)

#         for rect, work_str in zip(rectsD_manual, depths_manual_str):
#             height = rect.get_height()
#             lab = r"$" + work_str +"$"
#             ax.annotate(lab,
#                         xy=(rect.get_x() + rect.get_width() / 2, height),
#                         xytext=(0, 3),  # 3 points vertical offset
#                         textcoords="offset points",
#                         ha='center', va='bottom', rotation=90, fontsize=size_labels)

#         for rect, work_str in zip(rectsD_auto, depths_auto_str):
#             height = rect.get_height()
#             lab = r"$" + work_str +"$"
#             ax.annotate(lab,
#                         xy=(rect.get_x() + rect.get_width() / 2, height),
#                         xytext=(0, 3),  # 3 points vertical offset
#                         textcoords="offset points",
#                         ha='center', va='bottom', rotation=90, fontsize=size_labels)

        


#         fig.tight_layout()

#         # plt.show()
#         plt.savefig('depth_ ' + str(part) + '.pdf')  
#         from overlord import overlord
#         ol = overlord()
#         ol.upload_file("MyGreatResearch", "plots", "myplot.pdf")


# import collections

# def flatten(d, parent_key='', sep='_'):
#     items = []
#     for k, v in d.items():
#         new_key = str(parent_key) + sep + str(k) if parent_key else k
#         if isinstance(v, collections.MutableMapping):
#             items.extend(flatten(v, new_key, sep=sep).items())
#         else:
#             items.append((new_key, v))
#     return dict(items)
