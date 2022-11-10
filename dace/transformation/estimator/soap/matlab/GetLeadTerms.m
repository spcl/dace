function [ z ] = GetLeadTerms( expr, n, v )
    if nargin == 2
        syms S;
        v = S;
    end
    [k,~] = size(expr);    
    if (k > 1)
        for i = 1:k
            z(i) = evalin(symengine,strcat('expr(series(', char(expr(i)) ,', ', char(v), ' = infinity,',num2str(n),'))'));            
            syms O;
            z(i) = subs(z(i),O,0);
        end
        z = transpose(z);
    else
        z = evalin(symengine,strcat('expr(series(', char(expr) ,', ', char(v), ' = infinity,',num2str(n),'))'));
        syms O;
        z = subs(z,O,0);
    end
end

