from os import error, path
import socket
import time
from subprocess import call
from dataclasses import dataclass, field
from typing import Dict
import json
from dace import Config
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
    
    def start_solver(self, remoteMatlab : bool = True):
        if self.caching_solutions:
            if path.getsize(Config.get("soap", "solver", "db_path")) > 0:
                with open(Config.get("soap", "solver", "db_path"), "r") as solver_cache_file:
                    self.solver_cache = json.load(solver_cache_file)
        if self.cached_only:
            return
        # configuration
        port = 30000
        if remoteMatlab:
            address = Config.get("soap", "solver", "remote_solver_address")
        else:
            address = 'localhost'
            # start matlab in background
            call("matlab.exe -nosplash -nodesktop -r \"cd('" + Config.get("soap", "solver", "local_solver_path") + 
                "'); BackgroundSolver(" + str(port) + ");exit\"", shell=True)



        # initialize matlab connection
        self.conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)    
        if self.debug:        
            print('\nWaiting for the Matlab server to start....\n')
        while(True):
            try:
                self.conn.connect((address,port))
                break
            except:
                time.sleep(1)
        if self.debug:        
            self.print('\nConnected\n')
        self.status = "connected"
        #fromSolver = toSolver  # timos: we really don't need two sockets, but I didn't want to change too much of this code

        time.sleep(2)
        
        #for debugging only:
        self.set_timeout(10)

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

