from dace.sdfg.state import SDFGState
from dace import dtypes, subsets, memlet
from dace.transformation import dataflow
from dace.transformation.dataflow import hbm_copy_transform
from typing import Iterable
from dace.sdfg import utils
from dace.sdfg.sdfg import SDFG
import dace
import dace.libraries.blas.nodes as blas
import numpy as np
import dace.sdfg.nodes as nd
from dace.transformation import optimizer

################
# Helpers

def create_hbm_access(state: SDFGState, name, locstr, shape, lib_node,
    lib_conn, is_write, mem_str, dtype=dace.float32):
    if name not in state.parent.arrays:
        state.parent.add_array(name, shape, dtype)
        if locstr is not None:
            state.parent.arrays[name].location["bank"] = locstr
    if is_write:
        access = state.add_write(name)
        state.add_edge(lib_node, lib_conn, access, None,
            memlet.Memlet(mem_str))
    else:
        access = state.add_read(name)
        state.add_edge(access, None, lib_node, lib_conn, 
            memlet.Memlet(mem_str))

def random_array(size, type = np.float32, fix_constant = None):
    if not isinstance(size, Iterable):
        size = (size,)
    if fix_constant is None:
        a = np.random.rand(*size)
        a = a.astype(type)
    else:
        a = np.ones(size, type) * fix_constant
    return a

def create_or_load(load_from, create_method):
    if load_from is None:
        sdfg = create_method()
    else:
        sdfg = utils.load_precompiled_sdfg(load_from)
    return sdfg

def exec_axpy(data_size_per_bank: int, banks_per_array: int, load_from=None):
    def create_axpy_sdfg():
        N = dace.symbol("n")

        sdfg = SDFG("hbm_axpy")
        sdfg.add_symbol("a", dace.float32)
        state = sdfg.add_state("axpy")
        axpy_node = blas.Axpy("saxpy_node")
        axpy_node.implementation = "fpga_hbm"
        state.add_node(axpy_node)
        create_hbm_access(state, "in1", f"hbm.0:{banks_per_array}", 
            [banks_per_array, N], axpy_node, "_x", False, "in1")
        create_hbm_access(state, "in2", f"hbm.{banks_per_array}:{2*banks_per_array}",
            [banks_per_array, N], axpy_node, "_y", False, "in2")
        create_hbm_access(state, "out", f"hbm.{2*banks_per_array}:{3*banks_per_array}",
            [banks_per_array, N], axpy_node, "_res", True, "out")
        axpy_node.expand(sdfg, state)
        
        sdfg.sdfg_list[2].symbols["a"] = sdfg.sdfg_list[1].symbols["a"] #Why does inference fail?

        sdfg.apply_fpga_transformations(False)
        utils.update_array_shape(sdfg, "in1", [banks_per_array*N])
        utils.update_array_shape(sdfg, "in2", [banks_per_array*N])
        utils.update_array_shape(sdfg, "out", [banks_per_array*N])
        sdfg.arrays["in1"].storage = dtypes.StorageType.CPU_Heap
        sdfg.arrays["in2"].storage = dtypes.StorageType.CPU_Heap
        sdfg.arrays["out"].storage = dtypes.StorageType.CPU_Heap
        for xform in optimizer.Optimizer(sdfg).get_pattern_matches(
            patterns=[hbm_copy_transform.HbmCopyTransform]):
            xform.apply(sdfg)

        return sdfg

    sdfg = create_or_load(load_from, create_axpy_sdfg)
    x = random_array(data_size_per_bank*banks_per_array)
    y = random_array(data_size_per_bank*banks_per_array)
    alpha = random_array(1)
    result = np.zeros(data_size_per_bank*banks_per_array, dtype=np.float32)
    check = (alpha[0] * x) + y
    sdfg(in1=x, in2=y, out=result, a=alpha[0], n=data_size_per_bank)
    assert np.allclose(result, check)

if __name__ == "__main__":
    exec_axpy(1000, 8)