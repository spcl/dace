#!/usr/bin/python3
# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" DIODE client using a command line interface. """

# Usage example: cat ../samples/simple/matmul.py | ./diode_client.py --code --compile
import argparse, requests, json, sys

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-c",
                        "--connect",
                        default="localhost",
                        metavar="IP",
                        help="Connect to Server IP. Default is localhost.")

    parser.add_argument("-p", "--port", default="5000", help="Set server port. Default is 5000")

    parser.add_argument("-compile",
                        "--compile",
                        action="store_true",
                        help="Compiles the SDFG and returns resulting structures.")

    parser.add_argument(
        "-tf",
        "--transform",
        default="",
        help=
        "Sets the name of the transform to apply. If the transformation name is ambiguous, the first transformation with that name is chosen."
    )

    parser.add_argument(
        "-r",
        "--run",
        action="store_true",
        help="Executes the SDFG on the target machine specified in Config and prints the execution output (blocking)")

    parser.add_argument(
        "-code",
        "--code",
        action="store_true",
        help="Setting this indicates that the input is dace code. Default is false (compile JSON serialization of SDFG)"
    )

    parser.add_argument(
        "-u",
        "--user",
        default="default",
        help="Setting this indicates that the input is dace code. Default is false (compile JSON serialization of SDFG)"
    )

    parser.add_argument(
        "-e",
        "--extract",
        nargs="+",
        choices=["txform", "sdfg", "structure", "struct_noprop", "outcode", "txform_detail", "runnercode"])

    parser.add_argument("-ver", "--version", default="1.0", help="Sets the REST API Version to use.")
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
                data['sdfg'] = json.loads(stdin_input)['sdfg']
            except:
                sys.stderr.write("Failed to parse serialized SDFG input, " "is it in a correct json format?")
                sys.stdout.write("Invalid data: " + str(stdin_input))
                sys.exit(-3)

            # Append runnercode if available
            try:
                data['code'] = json.loads(stdin_input)['runnercode'][0]
            except:
                pass

        data['client_id'] = args.user

        #data = json.dumps(data)
        cmdstr = "run/" if args.run else "compile/dace"

        if args.transform:
            if args.code:
                sys.stderr.write(
                    "Cannot combine --code and --transform. Compile using '--code --extract sdfg txform_detail' first, then pipe the output into a command with --transform"
                )
                sys.exit(-4)

            try:
                transforms = json.loads(stdin_input)['advanced_transform']
            except:
                sys.stderr.write(
                    "Commands executed with --transform need an input file generated previously that includes --extract txform_detail. (Not passing --extract is not valid)"
                )
                sys.exit(-4)

            # Apply default transform (no property change)
            txf_found = False
            txform_sdfg = ""
            for k, v in transforms.items():
                # Compound level (key = target sdfg name, value = Object of transforms)

                try:
                    txform = v[args.transform]
                    txf_found = True
                    txform_sdfg = k
                    break
                except:
                    # Key not found
                    continue
            if not txf_found:
                sys.stderr.write("Could not find a transformation named " + args.transform)
                sys.exit(-5)
            # Else we have a transform to apply

            # Build the format for the transformation manually
            data['optpath'] = {txform_sdfg: [{'name': args.transform, 'params': {'props': txform}}]}

        nofail = False
        for i in range(0, 5):
            uri = url + "/dace/api/v" + args.version + "/" + cmdstr
            try:
                response = requests.post(uri, json=data)
            except Exception as e:
                print("Failed to request url '" + uri + "' with error " + str(e))
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
            # Output is a json asking to use a different URL to read the output
            for i in range(0, 5):
                import time
                time.sleep(1)
                response = requests.post(url + "/dace/api/v" + args.version + "/run/status/",
                                         json={'client_id': args.user})
                try:
                    tmp = json.loads(response.text)
                except:
                    # Got valid data
                    output_ok = True
                    break
            if not output_ok:
                sys.stderr.write("Failed to get run reference\n")
                sys.exit(-1)
            sys.stdout.write(response.text)
            sys.exit(0)

        resp_json = response.json()
        if "error" in resp_json:
            s = ""
            if "traceback" in resp_json:
                s += resp_json["traceback"]
            raise ValueError(s + resp_json["error"])

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
            clist = resp_json['compounds'].keys()
            for x in clist:
                sys.stdout.write('"' + x + '"' + ":\n{")
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
                    if c != l[-1]: sys.stdout.write(',')

                sys.stdout.write("}")
                if x != list(clist)[-1]: sys.stdout.write(',')

        extract_list = list(args.extract)
        # Extract if requested
        if args.extract:
            if len(args.extract) == 1 and "outcode" in args.extract:
                pass
            else:
                sys.stdout.write("{")
            for elem in extract_list:

                if "sdfg" == elem:
                    # Output SDFG
                    comps = resp_json['compounds']
                    ret = {}
                    for k, v in comps.items():
                        ret[k] = v['sdfg']
                    sys.stdout.write('"sdfg":')
                    sys.stdout.write(json.dumps(ret, indent=2))
                    if "sdfg" != args.extract[-1]: sys.stdout.write(',')
                if "txform" == elem:
                    # Output available transformations
                    sys.stdout.write('"simple_transform":')
                    sys.stdout.write("{")
                    get_transformations(resp_json, lambda a, b, c: sys.stdout.write(b + '\n'))
                    sys.stdout.write("}")
                    if "txform" != args.extract[-1]: sys.stdout.write(',')
                if "txform_detail" == elem:
                    # Output available transformations in json-format (necessary to apply)
                    sys.stdout.write('"advanced_transform":')
                    sys.stdout.write("{")
                    get_transformations(
                        resp_json,
                        lambda a, b, c: sys.stdout.write('"' + b + '":\n' + json.dumps(c, indent=2) + '\n\n'))
                    sys.stdout.write("}")
                    if "txform_detail" != args.extract[-1]:
                        sys.stdout.write(',')
                if "structure" == elem:
                    # Remove values; only output skeleton structure (i.e. only true tree nodes, no leafs)
                    sys.stdout.write('"structure":')
                    new_d = dict_scanner(resp_json)
                    sys.stdout.write(json.dumps(new_d, indent=2))
                if "struct_noprop" == elem:
                    sys.stdout.write('"struct_noprop":')
                    new_d = dict_scanner(resp_json, nometa=True)
                    sys.stdout.write(json.dumps(new_d, indent=2))
                if "outcode" == elem:
                    # Don't add objects if this is the only requested output
                    as_json = False
                    if len(args.extract) > 1:
                        sys.stdout.write('"outcode": ')
                        as_json = True
                    if as_json:
                        sys.stdout.write(json.dumps({k: v['generated_code']
                                                     for k, v in resp_json['compounds'].items()}))

                    else:
                        try:
                            for x in resp_json['compounds'].keys():
                                sys.stdout.write("//" + x + ":\n")
                                l = resp_json['compounds'][x]['generated_code']
                                for c in l:
                                    sys.stdout.write("// #### Next ####\n")
                                    sys.stdout.write(c)
                        except:
                            if 'error' in resp_json:
                                print('ERROR:', resp_json['error'])
                                if 'traceback' in resp_json:
                                    print(resp_json['traceback'])
                            else:
                                print('Received erroneous JSON:', resp_json)
                            raise
                    if len(args.extract) > 1:
                        if "outcode" != extract_list[-1]:
                            sys.stdout.write(',')
                if "runnercode" == elem:

                    sys.stdout.write('"runnercode": ')
                    # Pass the input code through

                    # Check if the code was in already passed-through data input (read json)
                    runnercode = ""
                    try:
                        d = json.loads(stdin_input)
                        runnercode = d['runnercode'][0]
                    except:
                        pass
                    if not args.code and runnercode == "":
                        sys.stderr.write("Error: Cannot extract runnercode as it was not in input.")
                        sys.exit(-4)
                    elif args.code:
                        # Take stdin_input
                        runnercode = stdin_input

                    sys.stdout.write(json.dumps([runnercode]))

                    if "runnercode" != extract_list[-1]:
                        sys.stdout.write(',')

            if len(args.extract) == 1 and "outcode" in args.extract:
                pass
            else:
                sys.stdout.write("}")

        else:
            sys.stdout.write(response.text)
