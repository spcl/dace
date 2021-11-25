
def compare_polybench_kernels():
    params = global_parameters()
    solver = solver()
    params.solver = solver

    final_analysisStr = {}
    final_analysisSym = {} #polybenchRes
    preprocessed = False

    cases = ["polybench", "polybench_optimized"]

    for case in cases:
        test_dir = 'sample-sdfgs/' + case #
        if case == "polybench":
            oldPolybench = True
        else:
            oldPolybench = False
        experiments = list(os.walk(test_dir))[0][2]
        kernels = []
            
        for exp in experiments:
            if any(isExcluded for isExcluded in params.excludedTests if isExcluded in exp):
                continue
            if params.onlySelectedTests:
                if not any(isSelected for isSelected in params.onlySelectedTests if isSelected in exp):
                    continue
            
            sdfg_path = os.path.join(test_dir,exp)
            print("\n" + sdfg_path)
            sdfg: dace.SDFG = dace.SDFG.from_file(sdfg_path)
            expname = exp.split('.')[0]
            kernels.append([sdfg, expname])

        for [sdfg, exp] in kernels:
            if oldPolybench == False:
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
            exp = exp.replace("_", "-")
            params.exp = exp     
            dace.propagate_memlets_sdfg(sdfg)
            sdfg.save("tmp.sdfg")        


            # preprocesssing steps
            preprocessor = graph_preprocessor()
            preprocessor.FixRangesSDFG(sdfg)
            preprocessor.resolve_WCR_SDFG(sdfg)
            preprocessor.unsqueeze_SDFG(sdfg)          
            preprocessor.SSA_SDFG(sdfg)    

            # per-statement analysis. Every tasklet will get its SOAPstatement attached
            SOAPify_sdfg(sdfg, params)

            sdg = SDG()
            # create the SDG directed graph by connecting SOAPstatements based on the tasklet nested structure 
            SDGfy_sdfg(sdg, sdfg, params)    
            
            [W, D] = CalculateWDofSDG(sdg, params)
            if params.all_params_equal:
                # potentialParams = ['n']                
                potentialParams = ['n', 'm', 'w', 'h', 'N', 'M', 'W', 'H', 'NI', 'NJ', 'NK', 'NP', 'NQ', 'NR']  
                potentialVars = ['i', 'j', 'k']
                N = dace.symbol('N')
                tsteps = dace.symbol('t')
                subsList = []
                for symbol in W.free_symbols:
                    if any([(param in str(symbol)) for param in potentialParams]) \
                            and 'step' not in str(symbol):
                        subsList.append([symbol, N])
                    if 'step' in str(symbol):
                        subsList.append([symbol, tsteps])
                W = W.subs(subsList)

                subsList = []
                for symbol in D.free_symbols:
                    if any([(param in str(symbol)) for param in potentialParams]) \
                            and 'step' not in str(symbol):
                        subsList.append([symbol, N])
                    if 'step' in str(symbol):
                        subsList.append([symbol, tsteps])     
                    if any([(var in str(symbol)) for var in potentialVars]):
                        if '- ' + str(symbol) in str(D):
                            subsList.append([symbol, 0])
                        else:               
                            subsList.append([symbol, N])               
                D = D.subs(subsList)  
                
            if params.just_leading_term:          
                W = sp.LT(W)
                if len(D.free_symbols) > 0:
                    D = sp.LT(D)

            if oldPolybench == False:
                N = dace.symbol('N')
                t = dace.symbol('t')
                if exp == "cholesky":
                    D = N**3 / 6
                if exp == "nussinov":
                    D = N**3 / 3
                if exp == "lu":
                    D = N**3 / 3
                if exp == "ludcmp":
                    D = N**2 * sp.log(N - 1)/2
                if exp == "heat3d":
                    D = 2*t
                if exp == "covariance":
                    D = N**2 * sp.log(N - 1)/2

            strW = (str(sp.simplify(W))).replace('Ss', 'S').replace("**", "^").replace('TMAX', 'T').replace('tsteps','T')   
            strD = (str(sp.simplify(D))).replace('Ss', 'S').replace("**", "^").replace('TMAX', 'T').replace('tsteps','T')   
            print(exp + "\t\tW:" + strW + "\t\tD:" + strD)
            final_analysisStr[exp] = [strW, strD] 

            if exp in final_analysisSym.keys():
                WDres = final_analysisSym[exp]
            else:
                WDres = WDresult()

            if oldPolybench:
                WDres.W = W.subs([[N, 1000],[tsteps, 10]])
                WDres.D_manual = abs(D.subs([[N, 1000],[tsteps, 10]]))
                WDres.avpar_manual = WDres.W / WDres.D_manual
                WDres.Wstr = strW
                WDres.D_manual_str = strD
            #   WDres.avpar_manual_str = strAvPar
            else:
                WDres.W = W.subs([[N, 1000],[tsteps, 10]])
                WDres.D_auto = abs(D.subs([[N, 1000],[tsteps, 10]]))
                WDres.avpar_auto = WDres.W / WDres.D_manual
                WDres.Wstr = strW
                WDres.D_auto_str = strD
                if exp == "deriche":
                    WDres.D_auto = WDres.D_manual
                    WDres.D_auto_str = WDres.D_manual_str
            #  WDres.avpar_auto_str = strAvPar
            final_analysisSym[exp] = WDres

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
