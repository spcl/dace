function varsOpt = GuessSolutionOld2(input, whichOnes, vars, grad)
    monomials = children(input);
    ComplicatedCode = 0;
    numVars = length(vars);
    u = sym('u', [1 numVars]);
    assume(u <= 0);    
    syms A X;
    assume(X >= 1001);
    assume(A >= 0);
    varsOpt = [];
    %if there are more variables than equations, we have to guess at least
    %some of the variables
    numVariablesToGuess =  length(vars) - length(monomials);
    variablesToGuess = vars(1:numVariablesToGuess);
    vars = vars(numVariablesToGuess+1:end);
    constants = 1;
    %generate possible exponents and leading terms
    if ComplicatedCode == 0
        if numVariablesToGuess > 0
            for i = 1:numVars+1
                constants = union(constants,([1:i+1] -1 )./i);
            end
        end
    else
        constants = ([1:numVars] )./(numVars + 1);
    end
    
    numConstants = length(constants);

    if ComplicatedCode == 0
        numCombinations = numConstants^(2*numVariablesToGuess);
    else
        numCombinations = numConstants^(numVars-1);
        optExponents = sym([2/7, 3/7, 1/7, 2/7, 3/7, 4/7]);
        vars = symvar(input);
        syms i1 i2 i3 i4 i5 i6;
        var6 = (1-(i1*i2*i4+i2*i3*i5+i1*i4*i5))/(i1*i3 + 2*i3*i4 + i2);
    end
    constantChoice = ones([2*numVariablesToGuess,1]);
    %a vector of a form
    %[c1 e1 c2 e2 .... cn en], where n = numVariablesToGuess, and ci and ei
    %refer to a specific constant in the "constants" table

    varsChar = arrayfun(@char, vars, 'uniform', 0);
    overalSum = 1/(length(monomials) - nnz(whichOnes));
    for combination = 0:numCombinations-1
        badCombination = 0;
        varsOpt = sym(varsOpt);
        c = sym('c', size(vars));
        e = sym('e', size(vars));
        assume(c > 0);
        assume(e >= 0);
        combRemainder = combination;
        
        if ComplicatedCode == 0
            for constantInd = 2*numVariablesToGuess:-1:1
                constantChoice(constantInd) = floor(combRemainder / ( numConstants^(constantInd-1)) + 1);
                combRemainder = combRemainder - (constantChoice(constantInd) - 1) * ( numConstants^(constantInd-1));
                if mod(constantInd, 2 ) == 1
                    if (constantChoice(constantInd)+1 > numConstants)
                        badCombination = 1;
                        break
                    end
                    varsOpt((constantInd+1)/2) = constants(constantChoice(constantInd)+1) * (X-nnz(whichOnes))^(constants(constantChoice(constantInd+1)));
                end
            end 
        else
            for constantInd = numVars-1:-1:1
                constantChoice(constantInd) = floor(combRemainder / ( numConstants^(constantInd-1)) + 1);
                combRemainder = combRemainder - (constantChoice(constantInd) - 1) * ( numConstants^(constantInd-1));                
                varsOpt(constantInd) = constants(constantChoice(constantInd)) * (X-nnz(whichOnes))^optExponents(constantInd);
            end 
            varsOpt(6) = subs(var6, [i1 i2 i3 i4 i5], constants(constantChoice))* (X-nnz(whichOnes))^optExponents(6);
            'constants choice'
        constantChoice
            if (subs(input, vars, varsOpt) ~= X)
                continue
            end
            gradOpt = solve(subs(grad,[variablesToGuess vars],varsOpt) == 0, [A u]);
            if isempty(gradOpt) || isempty(gradOpt.(char(A)))
                varsOpt = [];
                continue
            end
            success = 1;
        end
        if badCombination == 1
            continue
        end
        'constants choice'
        constantChoice
        
        monomialsCur = subs(monomials, variablesToGuess, varsOpt);
    
        for i = 1:length(monomials)
            monomial = monomialsCur(i);
            monoVars = arrayfun(@char, symvar(monomial), 'uniform', 0);
            sys2(i) = e * transpose(contains(varsChar, monoVars)) - 1 + whichOnes(i);
            sys3(i) = prod(c.^ contains(varsChar, monoVars)) - overalSum + (overalSum - 1)*whichOnes(i);
        end
        sol2 = solve(sys2 == 0, e);
        sol3 = solve(sys3 == 0, c);
        if isempty(sol2.(char(e(1)))) || isempty(sol3.(char(c(1))))
            varsOpt = [];
            continue
        end
        for i = 1:length(vars)
            e(i) = sol2.(char(e(i)));
            c(i) = sol3.(char(c(i)));                
            varsOpt(numVariablesToGuess+i) = c(i) * (X- nnz(whichOnes))^e(i);
        end
        gradOpt = solve(subs(grad,[variablesToGuess vars],varsOpt) == 0, [A u]);
        if isempty(gradOpt) || isempty(gradOpt.(char(A)))
            varsOpt = [];
            continue
        end
        solVars = symvar(gradOpt.(char(A)));
        for i = 1:length(vars)
            solVars = union(solVars, symvar(gradOpt.(char(u(i)))));
        end
        if length(solVars) > 1
            varsOpt = [];
        end
        if ~isempty(varsOpt)
            return
        end
    end
end

