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

parser.add_argument("-r", "--run", action="store_true",
                    help = "Executes the SDFG on the target machine specified in Config and prints the execution output (blocking)")

parser.add_argument("-code", "--code", action="store_true",
                    help="Setting this indicates that the input is dace code. Default is false (compile JSON serialization of SDFG)")

parser.add_argument("-u", "--user", default="default",
                    help="Setting this indicates that the input is dace code. Default is false (compile JSON serialization of SDFG)")

parser.add_argument("-e", "--extract", nargs="+", choices=["txform", "sdfg", "structure", "struct_noprop", "outcode", "txform_detail"])

parser.add_argument("-ver", "--version", default="1.0",
                    help="Sets the REST API Version to use.")
args = parser.parse_args()

if args.compile or args.run:
    url = 'http://' + str(args.connect) + ":" + str(args.port)
    data = {}
    stdin_input = sys.stdin.read()
    if args.code:
        # Compile from code
        data['code'] = stdin_input
    else:
        # Compile from serialized data
        try:
            data['sdfg'] = json.loads(stdin_input)
        except:
            sys.stderr.write("Failed to parse serialized SDFG input, is it in a correct json format?")
            sys.stdout.write("Invalid data: " + str(stdin_input))
            sys.exit(-3)
    
    data['client_id'] = args.user
    
    #data = json.dumps(data)
    cmdstr = "run/" if args.run else "compile/dace"

    nofail = False
    for i in range(0, 5):
        uri = url + "/dace/api/v" + args.version + "/" + cmdstr
        try:
            response = requests.post(uri, json=data)
        except Exception as e:
            print("Failed to request url + '" + uri + "' with error " + str(e))
            import time
            time.sleep(2)
            continue

        # Break if there was no exception
        nofail = True
        break
    if not nofail:
        # Cannot continue
        sys.exit(-2)


    if args.run:
        first_out = response.text
        output_ok = False
        #resptext = ""
        # Output is a json asking to use a different URL to read the output
        for i in range(0, 5):
            import time
            time.sleep(1)
            response = requests.post(url + "/dace/api/v" + args.version + "/run/status/", json={'client_id': args.user})
            #resptext = ""
            #for l in response.iter_lines(decode_unicode=True):
            #    resptext += str(l) + '\n'
            try:
                tmp = json.loads(response.text)
                #tmp = json.loads(resptext)
            except:
                # Got valid data
                output_ok = True
                break
        if not output_ok:
            sys.stderr.write("Failed to get run reference\n")
            sys.exit(-1)
        sys.stdout.write(response.text)
        #sys.stdout.write(resptext)
        sys.exit(0)

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
            comps = resp_json['compounds']
            ret = {}
            for k, v in comps.items():
                ret[k] = v['sdfg']
            sys.stdout.write(json.dumps(ret, indent=2))
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
                    sys.stdout.write("// #### Next ####\n")
                    sys.stdout.write(c)
    
    else:
        sys.stdout.write(response.text)

