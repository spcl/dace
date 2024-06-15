from os import error, path
import socket
import time
from subprocess import call, TimeoutExpired
from dataclasses import dataclass, field
from typing import Dict
import json
from dace import Config
import numpy as np
import sympy as sp
from sympy import symbols
import copy
from dace.transformation.estimator.soap.utils import get_default_gateway_ip
# MATLAB SERVER CONFIGURATION 

@dataclass
class Solver():     
    cached_only : bool = False
    caching_solutions: bool = True
    debug: bool = False
    conn : socket = field(default_factory = socket.socket, hash = False)
    status: str = "disconnected"
    solver_cache: Dict = field(default_factory=dict)

    def __post_init__(self):
        self.cached_only = Config.get("soap", "solver", "only_db") 
        self.caching_solutions = Config.get("soap", "solver", "caching_solver_solutions") 
    
    def start_solver(self):
        if self.caching_solutions:
            if path.getsize(Config.get("soap", "solver", "db_path")) > 0:
                with open(Config.get("soap", "solver", "db_path"), "r") as solver_cache_file:
                    self.solver_cache = json.load(solver_cache_file)
        if self.cached_only:
            return
        # configuration
        port = 30000
        if Config.get("soap", "solver", "remote_matlab") :
            address = Config.get("soap", "solver", "remote_solver_address")
        else:
            # address = '192.168.6.166'# '127.0.0.1'
            # address = '127.0.0.1'
            # address = '172.22.80.1'
            address = get_default_gateway_ip()
            # start matlab in background
            call("matlab.exe -nosplash -nodesktop -r \"cd('" + Config.get("soap", "solver", "local_solver_path") + 
                "'); BackgroundSolver(" + str(port) + ");exit\"", shell=True)
            # try:
            #     call("matlab.exe -nosplash -nodesktop -r \"cd('/home/alexnick/Projects/sdg/matlab'); BackgroundSolver(" + str(port) + ");exit\"", shell=True, timeout=1)
            # except TimeoutExpired:
            #     pass



        # initialize matlab connection
        self.conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM) 
        if self.debug:        
            print('\nWaiting for the Matlab server to start....\n')
        while(True):
            try:
                self.conn.connect((address,port))
                break
            except Exception:
                time.sleep(1)
        if self.debug:        
            self.print('\nConnected\n')
        self.status = "connected"
        #fromSolver = toSolver  # timos: we really don't need two sockets, but I didn't want to change too much of this code

        time.sleep(2)
        
        #for debugging only:
        self.set_timeout(60)

        return [self.conn, self.conn]

    
    def set_debug(self, debug = True):
        if self.cached_only:
            return
        self.debug = debug
        self.conn.sendall(("debug;" + str(int(debug)) + '@').encode())  
        self.set_timeout(300)

    
    def set_timeout(self, timeout : int):
        if self.cached_only:
            return
        self.conn.sendall(("timeout;" + str(timeout) + '@').encode())


    def end_solver(self):
        if self.caching_solutions:
            with open(Config.get("soap", "solver", "db_path"), "w") as solver_cache_file:
                json.dump(self.solver_cache, solver_cache_file)
        if self.cached_only:
            return
        self.conn.sendall("end@".encode())
        self.status = "disconnected"
        call('stty sane', shell=True)
        

    def send_command(self, cmd : str):
        if cmd in self.solver_cache.keys():
            return self.solver_cache[cmd]
        if self.cached_only:
            raise Exception("Using offline (cached-only) solver. The input command {} not found in cache".format(cmd))
        self.conn.sendall((cmd + '@').encode())   
        ret_val = self.conn.recv(2048).decode() 
        if self.caching_solutions:
            self.solver_cache[cmd] = ret_val
        return ret_val

    

    def sympy_solve(self, dom_set, h_size):
        vars = np.array(list(dom_set.free_symbols))
        lam, X, S = symbols('lam X S')
        assumps = [(X, 1001), (S, 1000)]
        u = np.array(symbols(['u'+str(i) for i in range(len(vars))]))
        ineqconstr = np.multiply(u, vars-1)
        L = h_size - lam*(dom_set - X) - np.sum(ineqconstr)
        grad = [L.diff(x) for x in list(vars) + [lam]]
        sys = grad + list(ineqconstr)
        sols = sp.solvers.nonlinsolve(sys, list(vars) + [lam] + list(u))

        opt_prob = h_size/ (X-S)
        varsOpts = []
        rhoOpts = []
        Xoptss = []
        if len(sols) > 0:
            for i,sol in enumerate(sols):
                if is_valid_sol(sol, len(vars)):
                    varsOpt = list(zip(vars, sol[:len(vars)]))
                    varsOpts.append(varsOpt)
                    rho = opt_prob.subs(varsOpt)
                    dRho = sp.diff(rho, X)
                    Xopts = [x for x in sp.solvers.solve(dRho, X) if x.subs(assumps) > 0]
                    if len(Xopts) == 0:
                        rhoOpts.append(sp.limit(rho,X, sp.oo))
                        Xoptss.append(sp.oo)
                    else:
                        for Xopt in Xopts:
                            rhoOpts.append(sp.simplify(rho.subs(X, Xopt)))
                            Xoptss.append(Xopt)
        ordered_sols = sorted(list(zip(rhoOpts, Xoptss, varsOpts)), key=lambda x: x[0].subs(S, 1000))

        sol = ordered_sols[0]
        rhoOpts = sol[0]
        Xopts = sol[1]
        varsOpt = sol[2]
        inner_tile = [sp.simplify(v[1].subs(X,Xopts)) for v in varsOpt]
        outer_tile = [sp.simplify(v[1].subs(X,S)) for v in varsOpt]
        xpart_dims = copy.deepcopy(outer_tile)
        return [rhoOpts, Xopts, varsOpt, inner_tile, outer_tile, xpart_dims]


def is_valid_sol(sol, var_len):
    X, S = symbols('X S')
    u_vals = sol[-var_len:]
    assumps = [(X, 1001), (S, 1000)]
    if all([u_val.subs(assumps) <= 0 for u_val in u_vals]):
        var_vals = sol[:var_len]
        if all([var_val.subs(assumps) >= 1 for var_val in var_vals]):
            return True
    return False


