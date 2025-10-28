import dace


sdfg = dace.SDFG.from_file("cloudsc_inner_fresh.sdfgz")

for n, g in sdfg.all_nodes_recursive():
    if isinstance(n, dace.nodes.NestedSDFG) and (n.label == "foedelta_srt4" or n.sdfg.label == "foedelta_srt4"):
        print("ptare_var_0 in symbols?", "ptare_var_0" in n.sdfg.symbols)
        print("ptare_var_0 in arrays?", "ptare_var_0" in n.sdfg.arrays)
        if "ptare_var_0" in n.sdfg.arrays:
            print(n.sdfg.arrays["ptare_var_0"], type(n.sdfg.arrays["ptare_var_0"]))
        print("rtt_var_1 in symbols?", "rtt_var_1" in n.sdfg.symbols)
        print("rtt_var_1 in arrays?", "rtt_var_1" in n.sdfg.arrays)
        if "rtt_var_1" in n.sdfg.arrays:
            print(n.sdfg.arrays["rtt_var_1"], type(n.sdfg.arrays["rtt_var_1"]))
