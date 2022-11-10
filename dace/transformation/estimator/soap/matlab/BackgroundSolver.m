function [X] = BackgroundSolver(port, delay)
    debug = 1;
    delay = 0.5;
    timeout = 60;
    matServer = tcpip('localhost', port, 'NetworkRole', 'server');
    if debug == 1
        'waiting for the python client on port'
	port
    end
    fopen(matServer);
    if debug == 1
        'Matlab server running'
    end
    

    syms S;
    counter = 0;
    cmdBuff = '';
    'Initial timeout value'
    timeout
    while(1)
        if matServer.BytesAvailable > 0
            counter = 0;
            cmdBuff = strcat(cmdBuff, ...
                    fscanf(matServer, '%c', matServer.BytesAvailable));
            tmp = strsplit(cmdBuff, '@');
            cmds = tmp(1:end-1);
            cmdBuff = tmp{end};
            for i = 1:length(cmds)     
                command = cmds{i};
                if command == "end"
                    break
                elseif startsWith(command,'simplify')
                    problem = split(command, ';');
                    if debug == 1
                        'problem to simplify:'
                        problem(2)
                    end
                    Q = evalin(symengine, problem(2)); 
                    syms S N M T;
                    Qsimp = expand(GetLeadTerms(Q, 1, S));
                    Qsimp = expand(GetLeadTerms(Qsimp, 1, N));
                    Qsimp = expand(GetLeadTerms(Qsimp, 1, M));
                    Qsimp = expand(GetLeadTerms(Qsimp, 1, T));
                    fwrite(matServer, string(Qsimp));    
                elseif startsWith(command,'debug')
                    debugStr = split(command, ';');                
                    debug = str2double(debugStr{2});
                    'setting debug to'
                    debug
                elseif startsWith(command,'timeout')
                    timeoutStr = split(command, ';');
                    timeout = str2double(timeoutStr{2});  
                    'Setting timeout to'
                    timeout
                elseif startsWith(command,'eval')
                    problem = split(command, ';');
                    if debug == 1
                        'problem to eval:'
                        problem(2)
                    end
                    if debug == 1
                        Q1 = evalin(symengine, problem(2))
                        Q2 = eval(Q1)
                        Q3 = simplify(Q2)
                    end
                    Q = simplify(eval(evalin(symengine, problem(2))));                
                    Qsimp = expand(GetLeadTerms(Q, 1, S));
                    fwrite(matServer, string(Qsimp));                               
                else
                    try
                        problem = split(command, ';');
                        input = evalin(symengine, problem{1});
                        output = evalin(symengine, problem{2});
                        if debug == 1
                            'command'
                            command
                            'solving input'
                            input
                            'output'
                            output
                        end
                        [rhoOpts, varsOpt, Xopts, vars, inner_tile, outer_tile ] = MaxRho( input, output );  
                        outputJson = "{" + newline;
                        outputJson = outputJson + '"variables": ["';
                        outputJson = outputJson + strjoin(string(vars), '", "') + '"],' + newline;
                        outputJson = outputJson + '"rhoOpts": ["';
                        outputJson = outputJson + strjoin(string(rhoOpts), '", "') + '"],' + newline;
                        outputJson = outputJson + '"varsOpt": ["';
                        outputJson = outputJson + strjoin(string(varsOpt), '", "') + '"],' + newline;
                        outputJson = outputJson + '"Xopts": ["';
                        outputJson = outputJson + strjoin(string(Xopts), '", "') + '"],' + newline;
                        outputJson = outputJson + '"inner_tile": ["';
                        outputJson = outputJson + strjoin(string(inner_tile), '", "') + '"],' + newline;
                        outputJson = outputJson + '"outer_tile": ["';
                        outputJson = outputJson + strjoin(string(outer_tile), '", "') + '"]' + newline;
                        outputJson = outputJson + "}";
                        if debug == 1
                            'writing output'
                            outputJson
                        end
                        fwrite(matServer, outputJson);
                    catch
                        'Unknown command received: '
                        command
                        outputJson = "{" + newline;
                        outputJson = outputJson + '"variables": ["-1"],' + newline;
                        outputJson = outputJson + '"rhoOpts": ["-2"],' + newline;
                        outputJson = outputJson + '"varsOpt": ["Unknown command received"],' + newline;
                        outputJson = outputJson + '"Xopts": ["';
                        outputJson = outputJson + strjoin(string(command), '", "') + '"]' + newline;
                        outputJson = outputJson + "}";
                        fwrite(matServer, outputJson);
                    end
                end
            end
        else
            pause(delay);
            counter = counter + 1;
            if counter > timeout
                 break
            end
        end
    end
    fclose(matServer);
end
