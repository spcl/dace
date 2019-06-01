# Translates DIODE1 optscripts to equivalent diode2_client commands
import argparse, sys

class diode_optscript_parser:

    def __init__(self, outputfile, diode_client_path="./"):
        self._diode_client_path = diode_client_path
        if outputfile != "":
            self._of = open(outputfile, "w")
        else:
            self._of = sys.stdout

        self._chain_ended = False
        self._chain = []
        
    def close(self):
        if self._of != sys.stdout:
            self._of.close()

    def OpenPythonFile(self, filepath):
        # Open as dace file and compile
        if self._chain_ended: self._of.write(" && ")
        self._of.write("cat {fpath} | python3 {cpath}diode2_client.py --code --compile --extract sdfg txform_detail runnercode".format(fpath=filepath, cpath=self._diode_client_path))

        # Build the chain to recreate when necessary
        self._chain.append(lambda: self.OpenPythonFile(filepath))
        

    def Run(self):
        # Run
        #print("Run called")
        self._of.write(" | python3 {cpath}diode2_client.py --run ".format(cpath=self._diode_client_path))

        self._chain.append(lambda: self.Run())
        self._chain_ended = True

    def ExpandNode(self, nodename):
        # This did open a new node in the old DIODE optgraph/opttree.
        # For diode2, the process is more involved:
        # 1) Get the current SDFG and optimizations
        # 2) Filter transformations, find the node name. Special care is needed for the global suffixes
        # 3) Send a new compile request with the selected optimization

        if self._chain_ended:
            self._of.write(" && ")
            # Restart the chain
            for x in self._chain:
                x()

        self._of.write()

    def ActivateNode(self, nodename):
        # #TODO
        raise ValueError("Not implemented yet")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("-f", "--file",
                    help="Path to optscript")

    parser.add_argument("-of", "--outfile", default="",
                    help="Path to output file. If not specified, the output file is stdout")

    parser.add_argument("-d2p", "--diode2_client_path", default="./",
                    help="Path of the diode2_client.py")

    args = parser.parse_args()

    path = args.file
    with open(path) as f:
        tr = diode_optscript_parser(args.outfile, args.diode2_client_path)
        exec(f.read(), globals(), {'diode': tr})
        tr.close()
