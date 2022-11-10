function [guess, res] = GuessSolution(dom,guess, output)
    if nargin < 3
        output = prod(symvar(dom));
    else
        if ~isequal(symvar(dom), symvar(output))
            'something is fishy....'
            dif1 = setdiff(symvar(dom), symvar(output));
            dif2 = setdiff(symvar(output), symvar(dom));
            if ~isempty(dif1)
                input = subs(dom, dif1, 1);
            end
            if ~isempty(dif2)
                rhoOpts = prod(symvar(dif2));
                varsOpt = [];
                return
            end
        end
    end
    
    
    dom = expand(dom);
    
    syms Lambda X M;
    assume(X > 0.1);
    vars = union(symvar(dom),symvar(output));   
    n = length(vars);
    c = children(dom);
    m = length(c);
    deg = 1;
    for i = 1:m
        deg = max(deg, length(symvar(c(i))));
    end
    L = output - Lambda*(dom - X);
    
    if nargin < 2
        grad = 1;        
        % LET'S DO THE GUESSING !   
        % denoting n - number of variables (I, J, K, ...) and
        % deg as maximum degree of any monomial in the dominator set,
        % we assume that the solution ( [I(X), J(X), K(X),...]) is of the
        % following form:
        % ------------------ %
        % baseGuess * A + B, where:
        % baseGuess = c3*(X - c1)^c2 for some scalar c1, c2, c3
        % ------------------ %
        % A=[a1,...,an], B=[b1,...,bn] are n-dimensional rational vectors
        % all unknowns: c1, c2, a1,...,an, b1,...,bn
        % are expected to be of the form p^(-1/q), where 
        % p \in [1,..,n]
        % q \in [1,..,deg]
        
        
%         % another heuristic. 
%         % We expect that only variables which occur in fewest monomials will
%         % have a non-1 value:
%         % e.g., for Dom = I*J + I*K, variable I is in two monomials, 
%         % and K and J are only in one. Therefore, we expect that I = 1 and 
%         % we solve only for K and J.
%         monomials = children(expand(dom));
%         counter = zeros([n 1]);
%         minOccur = length(monomials);
%         for i = 1:n
%             for monomial = monomials
%                 if ismember(vars(i), symvar(monomial))
%                     counter(i) = counter(i) + 1;
%                 end
%             end
%             minOccur = min(counter(i), minOccur);                
%         end
%         % now we get rid of all non-minimal variables and set them to 1.
%         nOrig = n;
%         dom = subs(dom, vars(counter > minOccur), ones([1, length(vars(counter > minOccur))]));               
%         vars = symvar(dom); 
%         n = length(vars);
%         c = children(expand(dom));
%         m = length(c);
%         deg = 1;
%         for i = 1:m
%             deg = max(deg, length(symvar(c(i))));
%         end
%         
%         % collect all free terms in the dominator set
%         a0 = 0;
%         for monomial = c
%             if isempty(symvar(monomial))
%                 a0 = a0 + monomial;
%             end
%         end
%         X = X - a0;
        
        % formulate lagrangian
        L = output - Lambda*(dom - X);
    
        % generate possible choices:
        span = sym(zeros([1, (n-1)*deg + 2]));        
        for ii = 2:n
            for jj = 1:deg
                span((ii-2)*deg + jj + 2) = sym(ii^(sym(-1/jj)));
            end
        end
        span(1) = 0;
        span(2) = 1;
        numChoices = length(span);
        
        c1 = 0;
        c2 = 1;
        c3 = 1;
        choicesC3 = span;
        % remove zero from possible choices of c3
        choicesC3(1) = [];
        curChoiceC3 = 1;
        
        A = sym(ones([1 n]));
        B = sym(zeros([1 n]));        
        choicesA = sym(zeros([n,numChoices]));
        choicesB = sym(zeros([n,numChoices]));
        for i = 1:n
            choicesA(i,:) = span;
            choicesB(i,:) = span;
        end
        
        % uWeights and guessWeights capture symmetries in the equation:
        % e.g., if the dominator set equation is symmetric for some
        % variables I, J, then we expect the corresponding pairs of parameters
        % (ai, aj) and (bi, bj) to be equal
        uWeights = sym('u', [1 n]);  
        guessWeights = uWeights;
        for i = 1:n
            for j = i+1:n
                if simplify(dom - subs(dom,[vars(i) vars(j)], [vars(j) vars(i)])) == 0
                    guessWeights(j) = guessWeights(i);
                    choicesA(j,:) = zeros([1 numChoices]);
                    choicesB(j,:) = zeros([1 numChoices]);
                end
            end            
        end
        
        % indicators which choice of guessed parameter is currenty
        % evaluated
        curChoiceA = ones([n 1]);
        curChoiceB = ones([n 1]);
        
        % default choice for A is 1, default choice for B is 0
        choicesA(:,[1 2]) = choicesA(:,[2 1]);
        
        blindGuessing = 0;
        
        while (grad ~= 0)
            baseGuess = c3*(X - c1)^(c2);
            guess = simplify(baseGuess*subs(guessWeights, uWeights, A) + subs(guessWeights, uWeights, B));        
          %  guess
%             if (curChoiceC3 == 5 && curChoiceA(1) == 1 && curChoiceA(3) == 3 && curChoiceB(1) == 2 && curChoiceB(3) == 3)
%                 a = 1;
%             end
            res = simplify(gradient(subs(L, vars, guess), [vars Lambda]));   
            grad = res(end);
            % done. Locally optimal solution
            if grad == 0
                break
            end
            
            % no finite maximum, increases to infinity
            if ~ismember(X, symvar(grad))
                if grad < 0
              %      guess = subs(guess, X, Inf);
                    break;
                end
            end
            
            if blindGuessing == 0
                % get the highest power of X in gradient
                a = children(GetLeadTerms(grad,1,X));
                b = a(1) / coeffs(a(1));           
                e = simplify(log(b) / log(X));
                if (e ~= 1)
                    c2 = c2/e;
                else
                    a = children(grad);
                    if (~isempty(symvar(simplify(a(1)/X))))
                        c1 = solve(grad == 0, X);
                    else
                        blindGuessing = 1;
                    end
                end
            end
            
            if blindGuessing == 1
                % now we start blid guessing...
                % we try to increment i-th row of curChoiceA
                if (curChoiceC3 < nnz(choicesC3))
                    curChoiceC3 = curChoiceC3 + 1;
                else
                    curChoiceC3 = 1;
                    i = 1;
                    while(1)
                        curChoiceA(i) = curChoiceA(i) + 1;
                        if (curChoiceA(i) > nnz(choicesA(i, :)) + 1)
                            curChoiceA(i) = 1;
                            i = i + 1;
                            if (i > n)
                                break;
                            end
                        else
                            break
                        end                        
                    end

                    % we checked all choices for A. Now let's increment
                    % B'th choices
                    if (i > n)
                        i = 1;
                        while(1)
                            curChoiceB(i) = curChoiceB(i) + 1;
                            if (curChoiceB(i) > nnz(choicesB(i, :)) + 1)
                                curChoiceB(i) = 1;
                                i = i + 1;
                                if (i > n)
                                    break;
                                end
                            else
                                break
                            end                        
                        end
                    end

                    if (i > n)
                        % something went wrong. We didn't find any
                        % solution... ;(
                        a = 1;
                        exit('Wtf?');
                    end
                end

%                 curChoiceC3 = 5;
%                 curChoiceA(1) = 1;
%                 curChoiceA(3) = 3;
%                 curChoiceB(1) = 2;
%                 curChoiceB(3) = 3;
                % now pick up current c3, A and B according to 
                % the current choice
                c3 = choicesC3(curChoiceC3);
                for i = 1:n
                    A(i) = choicesA(i, curChoiceA(i));
                    B(i) = choicesB(i, curChoiceB(i));
                end
            end
           % gradPrev = grad;
           % prevGuess = guess;
        end
        
%         fullGuess = sym(ones([1, nOrig]));
%         fullGuess(counter == minOccur) = guess;
%         guess = fullGuess;
    else 
        res = simplify(gradient(subs(L, vars, guess), [vars Lambda]));
    end
    a = 1
end

