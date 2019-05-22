#!/usr/bin/python3
# DIODE2 client using a command line interface


# Usage example: cat ../samples/simple/gemm.py | ./diode2_client.py --code --compile
import argparse, requests, json, sys

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--connect", default="localhost",
                    help="Connect to Server IP. Default is localhost.")

parser.add_argument("-p", "--port", default="5000",
                    help="Set server port. Default is 5000")

parser.add_argument("-compile", "--compile", action="store_true",
                    help="Compiles the SDFG and returns resulting structures.")

parser.add_argument("-code", "--code", action="store_true",
                    help="Setting this indicates that the input is dace code. Default is false (compile JSON serialization of SDFG)")

parser.add_argument("-u", "--user", default="default",
                    help="Setting this indicates that the input is dace code. Default is false (compile JSON serialization of SDFG)")


parser.add_argument("-ver", "--version", default="1.0",
                    help="Sets the REST API Version to use.")
args = parser.parse_args()

if args.compile:
    url = 'http://' + str(args.connect) + ":" + str(args.port)
    data = {}
    stdin_input = sys.stdin.read()
    if args.code:
        # Compile from code
        data['code'] = stdin_input
    else:
        # Compile from serialized data
        data['sdfg'] = stdin_input
    
    data['client_id'] = args.user
    
    #data = json.dumps(data)
    response = requests.post(url + "/dace/api/v" + args.version + "/compile/dace", json=data)

    sys.stdout.write(response.text)

