function [ rhoOpts, varsOpt, Xopts, vars, inner_tile, outer_tile ] = MaxRho( input, output )    
    onlyOneSolution = 1;    
    input = simplify(input);
    output = simplify(output);
    if nargin == 1
        output = prod(symvar(input));
    else
        if ~isequal(symvar(input), symvar(output))
            'something is fishy....'
            rhoOpts = -1;
            varsOpt = 'missing variables between inputs and outputs';
            Xopts = input;
            vars = output;
            return;
            dif1 = setdiff(symvar(input), symvar(output));
            dif2 = setdiff(symvar(output), symvar(input));
            if ~isempty(dif1)
                input = subs(input, dif1, 1);
            end
            if ~isempty(dif2)
                rhoOpts = prod(symvar(dif2));
                varsOpt = [];
                return
            end
        end
    end
    vars = union(symvar(input),symvar(output));   
    varsOpt = sym([]);
    u = sym('u', size(vars));
    syms A X S;
    assume(u <= 0);    
    assume(S >= 1000);
    assume(X >= 1001);
    assume(A >= 0);
    assume(vars >= 1);
    ineqconstr = u.*(vars - 1);
    %form Lagrangian from the KKT multipliers
    L = output - A*(input - X) - sum(ineqconstr);
    try
        grad = gradient(L, [vars A]);
    catch
        'Malformed input'
        input
        rhoOpts = -1;
        varsOpt = 'Malformed input';
        Xopts = input;
        vars = output;
        return
    end
    sys = [grad; transpose(ineqconstr)];    
    %KKT optimiality conditions
    warning('off');
    %sys
    sol = solve(sys == 0, [vars A u],'Real', true, 'IgnoreAnalyticConstraints', true);
    warning('on');
    numSols = length(sol.(char(vars(1))));
    
   % sol = [];
   % numSols = 1;
    rhoOpts(1) = [X];
    Xopts(1) = [X];
    if ~isempty(sol) && ~isempty(sol.(char(vars(1))))
        %find optimal X
        k = 1;
        for i = 1:numSols
            varsOpt(i,1) = [X];
            for j = 1:length(vars)
                sols = sol.(char(vars(j)));
                varsOpt(k,j) = simplify(sols(i));
            end
            rho = subs(output/ ( X - S), vars, varsOpt(k,:));
            dRho = diff(rho, X);
            Xopt = solve(dRho == 0, X);
            
            if isempty(Xopt)
                rhoOpts(end+1) = simplify(limit(rho, X, inf));
                Xopts(end+1) = inf;                
            else
                rhoOpts(end+1:end+length(Xopt)) = simplify(subs(rho, X, Xopt));
                Xopts(end+1:end+length(Xopt)) = Xopt;
                for j = 1:(length(Xopt) - 1)
                    numSols = numSols + 1;
                    k = k + 1;
                    varsOpt(k, :) = varsOpt(k - 1, :);
                end
            end
            k = k + 1;
        end  
        Xopts = Xopts(2:end);
    else
        %check if output is a function of only X
        output2 = subs(output, input, X);
        if length(symvar(output2)) == 1
            %good to go
            rho = output2/(X-S);
            dRho = diff(rho, X);
            Xopt = solve(dRho == 0, X);
            if isempty(Xopt)
                rhoOpts(end+1) = simplify(limit(rho, X, inf));
            else
                rhoOpts(end+1:end+length(Xopt)) = simplify(subs(rho, X, Xopt));
            end
        else
        %try brute force guesses 
            monomials = children(input);
            varsOpt = [];
            iter = 0;
            while isempty(varsOpt)
                whichOnes = de2bi(iter);
                whichOnes = [whichOnes,zeros(1, length(monomials) - length(whichOnes))];
                %varsOpt = GuessSolution(input, whichOnes, vars, sys);
                varsOpt = GuessSolution(input); %, whichOnes, vars, sys);
                iter = iter + 1;
            end

            rho = subs(output/ ( X - S), vars, varsOpt);
            dRho = diff(rho, X);
            Xopt = solve(dRho == 0, X);
            if isempty(Xopt)
                rhoOpts(end+1) = simplify(limit(rho, X, inf));
            else
                rhoOpts(end+1:end+length(Xopt)) = simplify(subs(rho, X, Xopt));
            end        
        end
    end
    rhoOpts = rhoOpts(2:end);
    
    % order solutions
    for i = 1:numSols
        for j = 1:numSols-i
          % this can also be replaced by
          % if eval(limit(rhoOpts(j) / rhoOpts(j+1), S, inf) S < 0)
            if eval(subs(rhoOpts(j) - rhoOpts(j+1), S, 1000) < 0)
                rhoOpts([j j+1]) = rhoOpts([j+1 j]);
                Xopts([j j+1]) = Xopts([j+1 j]);
                varsOpt([j j+1], :) = varsOpt([j+1 j], :);
            end
        end
    end
    
    if (onlyOneSolution == 1)
        rhoOpts = rhoOpts(1);
        Xopts = Xopts(1);
        varsOpt = varsOpt(1, :);
        inner_tile = simplify(subs(varsOpt, X, Xopts));
        outer_tile = simplify(subs(varsOpt, X, S));
    end
end

