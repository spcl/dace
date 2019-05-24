#!/usr/bin/python3
# DIODE2 client using a command line interface


# Usage example: cat ../samples/simple/gemm.py | ./diode2_client.py --code --compile
import argparse, requests, json, sys

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--connect", default="localhost", metavar="IP",
                    help="Connect to Server IP. Default is localhost.")

parser.add_argument("-p", "--port", default="5000",
                    help="Set server port. Default is 5000")

parser.add_argument("-compile", "--compile", action="store_true",
                    help="Compiles the SDFG and returns resulting structures.")

parser.add_argument("-code", "--code", action="store_true",
                    help="Setting this indicates that the input is dace code. Default is false (compile JSON serialization of SDFG)")

parser.add_argument("-u", "--user", default="default",
                    help="Setting this indicates that the input is dace code. Default is false (compile JSON serialization of SDFG)")

parser.add_argument("-e", "--extract", nargs="+", choices=["txform", "sdfg", "structure", "struct_noprop", "outcode", "txform_detail"])

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

    resp_json = response.json()
    
    def dict_scanner(d, nometa=False):
        if not isinstance(d, dict):
            return None
        else:
            for x in list(d.keys()):
                if nometa:
                    if x.startswith("_meta_"):
                        del d[x]
                        continue
                d[x] = dict_scanner(d[x], nometa=nometa)
        return d

    def get_transformations(resp_json, cb):
        for x in resp_json['compounds'].keys():
            sys.stdout.write(x + ":\n")
            l = resp_json['compounds'][x]['matching_opts']
            encountered = {}
            for c in l:
                if c['opt_name'] not in encountered:
                    encountered[c['opt_name']] = 0
                else:
                    encountered[c['opt_name']] += 1
                name_str = "" if encountered[c['opt_name']] == 0 else ("$" + str(encountered[c['opt_name']]))
                #sys.stdout.write(c['opt_name'] + name_str + '\n')
                cb(x, c['opt_name'] + name_str, c)

    # Extract if requested
    if args.extract:
        if "sdfg" in args.extract:
            # Output SDFG
            pass
        if "txform" in args.extract:
            # Output available transformations
            get_transformations(resp_json, lambda a, b, c: sys.stdout.write(b + '\n'))
        if "txform_detail" in args.extract:
            get_transformations(resp_json, lambda a, b, c: sys.stdout.write(b + '\n' + json.dumps(c, indent=2) + '\n\n'))
        if "structure" in args.extract:
            # Remove values; only output skeleton structure (i.e. only true tree nodes, no leafs)
            new_d = dict_scanner(resp_json)
            sys.stdout.write(json.dumps(new_d, indent=2))
        if "struct_noprop" in args.extract:
            new_d = dict_scanner(resp_json, nometa=True)
            sys.stdout.write(json.dumps(new_d, indent=2))
        if "outcode" in args.extract:
            for x in resp_json['compounds'].keys():
                sys.stdout.write("//" + x + ":\n")
                l = resp_json['compounds'][x]['generated_code']
                for c in l:
                    sys.stdout.write("// #### Next ####")
                    sys.stdout.write(c)
    
    else:
        sys.stdout.write(response.text)

