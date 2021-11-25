import socket
import time
from subprocess import call
# MATLAB SERVER CONFIGURATION 

class Solver(): 
    def __init__(self) -> None:        
        self.debug_no_solve = False
        self.conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.status = "disconnected"
    
    def start_solver(self, remoteMatlab):

        # configuration
        port = 30000
        if remoteMatlab:
            address = 'galilei.inf.ethz.ch'   
     #       address = '172.23.224.1'        
        else:
            # address = '172.31.64.1'
            address = 'localhost'
            # start matlab in background
            call("matlab.exe -nosplash -nodesktop -r \"cd('C:\\gk_pliki\\uczelnia\\doktorat\\performance_modelling\\repo"
                       "\\DAAPCe\\daapce_official\\matlab'); BackgroundSolver(" + str(port) + ");exit\"", shell=True)



        # initialize matlab connection
        self.conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)            
        print('\nWaiting for the Matlab server to start....\n')
        while(True):
            try:
                self.conn.connect((address,port))
                break
            except:
                time.sleep(1)
        print('\nConnected\n')
        self.status = "connected"
        #fromSolver = toSolver  # timos: we really don't need two sockets, but I didn't want to change too much of this code

        time.sleep(2)
        
        #for debugging only:
        self.set_timeout(10)

        return [self.conn, self.conn]

    
    def set_debug(self, debug = True):
        self.conn.sendall(("debug;" + str(int(debug)) + '@').encode())  
        self.set_timeout(300)

    
    def set_timeout(self, timeout : int):
        self.conn.sendall(("timeout;" + str(timeout) + '@').encode())


    def end_solver(self):
        self.conn.sendall("end@".encode())
        self.status = "disconnected"

    def Command(self, cmd : str):
        self.conn.sendall((cmd + '@').encode())   
        return self.conn.recv(2048).decode()     
        