def create_deeply_nested_sdfg():
    sdfg = dace.SDFG("deepnest_test")
    state : dace.SDFGState = sdfg.add_state("init")
    xarr = state.add_array("x", [4, 100, 100], dace.float32)
    sdfg.arrays["x"].location["hbmbank"] = "0:4"
    yarr = state.add_array("y", [4, 100, 100], dace.float32)
    sdfg.arrays["y"].location["hbmbank"] = "4:8"

    topMapEntry, topMapExit = state.add_map("topmap", dict(k="0:2"))
    
    nsdfg = dace.SDFG("nest")
    nstate = nsdfg.add_state("nested_state")
    xRead = nstate.add_array("xin", [4, 100, 100], dace.float32)
    xWrite = nstate.add_array("xout", [4, 100, 100], dace.float32)
    mapEntry, mapExit = nstate.add_map("map1", dict(w="0:2"))
    nope = nstate.add_tasklet("nop", dict(_in=None), dict(_out=None),
                                "_out = _in")
    inputMem = mem.Memlet("xin[2*k+w]")
    outputMem = mem.Memlet("xout[2*k+w]")
    nstate.add_memlet_path(xRead, mapEntry, nope, memlet=inputMem, dst_conn="_in")
    nstate.add_memlet_path(nope, mapExit, xWrite, memlet=outputMem, src_conn="_out")
    nsdfg_node = state.add_nested_sdfg(nsdfg, state, set(["xin"]), set(['xout']))

    state.add_memlet_path(xarr, topMapEntry, nsdfg_node, 
                        memlet=mem.Memlet.from_array("x", sdfg.arrays["x"]), dst_conn="xin")
    state.add_memlet_path(nsdfg_node, topMapExit, yarr, 
                        memlet=mem.Memlet.from_array("y", sdfg.arrays["y"]), src_conn="xout")
    sdfg.apply_fpga_transformations(validate=False)

    return sdfg
