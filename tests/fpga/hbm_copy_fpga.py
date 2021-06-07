"""
Performs several tests for HBM-copies and nesting of accesses to HBM.
Checked:
ND-Copy between Host/Device and on Device (with openCL) 
Nested unrolls
Nested sdfgs
Copy between HBM and DDR -no
Copy between HBM and StreamArray -no
"""

from sympy.abc import Z
from dace import subsets as sbs, dtypes, memlet as mem
import dace
import numpy as np

#Checks ND-copies from/to HBM via cpu
def create_cpu_copy_test():
    sdfg = dace.SDFG("ndcopytest")
    cpuis : dace.SDFGState = sdfg.add_state("cpu init state")
    host3darray = cpuis.add_array("h", [100, 100, 100], dace.float32)
    dev2darray_x = cpuis.add_array("x", [2, 50, 50], dace.float32,
                                    dtypes.StorageType.FPGA_Global)
    sdfg.arrays["x"].location["hbmbank"] = sbs.Range.from_string("0:1")
    dev2darray_y = cpuis.add_array("y", [100, 100], dace.float32,
                                    dtypes.StorageType.FPGA_Global)
    sdfg.arrays["y"].location["hbmbank"] = sbs.Range.from_string("2:2")
    dev3darray_z = cpuis.add_array("z", [2, 10, 10, 10], dace.float32,
                                    dtypes.StorageType.FPGA_Global)
    sdfg.arrays["z"].location["hbmbank"] = sbs.Range.from_string("3:4")
    xmirror = cpuis.add_array("xmirror", [2, 50, 50], dace.float32)
    ymirror = cpuis.add_array("ymirror", [100, 100], dace.float32)
    zmirror = cpuis.add_array("zmirror", [2, 10, 10, 10], dace.float32)

    h2y = mem.Memlet("h[3, 0:100, 0:100]")
    h2x = mem.Memlet("h[0:50, 0:50, 3]->0, 0:50, 0:50")
    y2x = mem.Memlet("y[50:100, 50:100]->1, 0:50, 0:50")
    h2z = mem.Memlet("h[90:100, 90:100, 90:100]->0, 0:10, 0:10, 0:10")
    xmir = mem.Memlet("xmirror")
    ymir = mem.Memlet("ymirror")
    zmir = mem.Memlet("zmirror")
    cpuis.add_memlet_path(host3darray, dev2darray_y, memlet=h2y)
    cpuis.add_memlet_path(host3darray, dev2darray_x, memlet=h2x)
    cpuis.add_memlet_path(dev2darray_y, dev2darray_x, memlet=y2x)
    cpuis.add_memlet_path(host3darray, dev3darray_z, memlet=h2z)
    cpuis.add_memlet_path(dev2darray_x, xmirror, memlet=xmir)
    cpuis.add_memlet_path(dev2darray_y, ymirror, memlet=ymir)
    cpuis.add_memlet_path(dev3darray_z, zmirror, memlet=zmir)

    return sdfg

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

def create_HBM_stream_and_DDR_test():
    sdfg = dace.SDFG("hbmstream")
    state = sdfg.add_state("hbmstream")

    xarr = sdfg.add_array("x", [2, 10], dace.float32)
    yarr = sdfg.add_array("y", [2, 10, 10], dace.float32)
    sdfg.arrays["x"].location["hbmbank"] = sbs.Range("0:1")
    sdfg.arrays["y"].location["hbmbank"] = sbs.Range("2:3")
    streamarr = sdfg.add_stream("st", dace.float32, 1, [10, 10])
    ddrarr0 = sdfg.add_array("z", [10], dace.float32)
    ddrarr1 = sdfg.add_array("w", [10], dace.float32)
    sdfg.arrays["z"].location["bank"] = 0
    sdfg.arrays["w"].location["bank"] = 1

    hbm2stream = mem.Memlet("y[0, 0:10, 0:10]->0:10, 0:10")
    stream2ddr = mem.Memlet("st[0, 0:10]->0:10")
    hbm2ddr = mem.Memlet("x[0, 5:10]->5:10")

    

sdfg = create_deeply_nested_sdfg()
sdfg.view()