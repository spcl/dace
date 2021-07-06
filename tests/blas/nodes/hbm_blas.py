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

def expandlibnode(sdfg: dace.SDFG, state: dace.SDFGState, 
    node: nd.LibraryNode, impl: str, dryExpand: bool = False, *args, **kwargs):
    node.implementation = impl
    if not dryExpand:
        node.expand(sdfg, state, *args, **kwargs)

def expand_first_libnode(sdfg: dace.SDFG, impl: str, dryExpand: bool=False,
    *args, **kwargs):
    for node, state in sdfg.all_nodes_recursive():
        if(isinstance(node, nd.LibraryNode)):
            expandlibnode(state.parent, state, node, impl, dryExpand, *args, **kwargs)
            return node

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

def on_nested_sdfgs(sdfg: SDFG, action):
    if sdfg.parent_nsdfg_node is not None:
        action(sdfg, sdfg.parent_nsdfg_node)
    else:
        action(sdfg, None)
    for state in sdfg.states():
        for n in state.nodes():
            if isinstance(n, nd.NestedSDFG):
                on_nested_sdfgs(n.sdfg, action)

def print_nested_symbols(sdfg: SDFG):
    def print_sym(sdfg, node):
        if node is not None:
            print(node.label)
            print(node.symbol_mapping)
        print(sdfg.symbols)
        print("----------------")
    on_nested_sdfgs(sdfg, print_sym)

################
# Execution methods

def exec_dot_hbm(data_size_per_bank: int, banks_per_input: int, load_from=None):
    def create_dot_sdfg():
        N = dace.symbol("N")

        sdfg = SDFG("hbm_dot")
        state = sdfg.add_state("sdot")
        dot_node = blas.Dot("sdot_node")
        dot_node.implementation = "FPGA_HBM_PartialSums"
        create_hbm_access(state, "in1", f"hbm.0:{banks_per_input}", 
            [banks_per_input, N], dot_node, "_x", False, "in1")
        create_hbm_access(state, "in2", f"hbm.{banks_per_input}:{2*banks_per_input}",
            [banks_per_input, N], dot_node, "_y", False, "in2")
        create_hbm_access(state, "out", "ddr.0", [1],
            dot_node, "_result", True, "out", dace.float32)
        dot_node.expand(sdfg, state, partial_width=16)

        sdfg.apply_fpga_transformations(False, validate=False)

        utils.update_array_shape(sdfg, "in1", [banks_per_input*N])
        utils.update_array_shape(sdfg, "in2", [banks_per_input*N])
        sdfg.arrays["in1"].storage = dtypes.StorageType.CPU_Heap
        sdfg.arrays["in2"].storage = dtypes.StorageType.CPU_Heap
        for xform in optimizer.Optimizer(sdfg).get_pattern_matches(
            patterns=[hbm_copy_transform.HbmCopyTransform]):
            xform.apply(sdfg)

        return sdfg

    sdfg = create_or_load(load_from, create_dot_sdfg)
    
    x = random_array(data_size_per_bank*banks_per_input, fix_constant=1)
    y = random_array(data_size_per_bank*banks_per_input, fix_constant=1)
    result = np.zeros(1, dtype=np.float32)
    check = np.dot(x, y)
    sdfg(in1=x, in2=y, out=result, N=data_size_per_bank)
    print(check)
    print(result)
    assert np.allclose(result, check)

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
    sdfg.view()
    x = random_array(data_size_per_bank*banks_per_array)
    y = random_array(data_size_per_bank*banks_per_array)
    alpha = random_array(1)
    result = np.zeros(data_size_per_bank*banks_per_array, dtype=np.float32)
    check = (alpha[0] * x) + y
    sdfg(in1=x, in2=y, out=result, a=alpha[0], n=data_size_per_bank)
    assert np.allclose(result, check)

def ref_gemv():
    N = dace.symbol("N")
    M = dace.symbol("M")

    sdfg = SDFG("gemv")
    state = sdfg.add_state("gemv")
    node = blas.Gemv("gemv", None, True, 0.5, 0.5)
    node.implementation = "FPGA_TilesByColumn"
    create_hbm_access(state, "A", None, [N, M], node,
        "_A", False, "A")
    create_hbm_access(state, "x", None, [N], node,
        "_x", False, "x")
    create_hbm_access(state, "y", None, [M], node,
        "_y", False, "y")
    create_hbm_access(state, "z", None, [M], node,
        "_y", True, "z")
    node.expand(sdfg, state)
    
    sdfg.view()

def exec_gemv(banks_A, m_size_per_bank, n_size, transpose_A = True, load_from : str = None):
    def create_gemv_sdfg():
        a = dace.symbol("a", dace.float32)
        b = dace.symbol("b", dace.float32)
        N = dace.symbol("N")
        M = dace.symbol("M")
        if transpose_A:
            shape_A = [banks_A, N , M]
            true_shape_A = [N, banks_A * M]
        else:
            shape_A = [banks_A, M, N]
            true_shape_A = [banks_A * M, N]
        shape_x = [N]
        shape_y = [banks_A, M]

        sdfg = SDFG("hbm_gemv")
        sdfg.add_symbol("a", dace.float32)
        sdfg.add_symbol("b", dace.float32)
        state = sdfg.add_state("gemv")
        gemv_node = blas.Gemv("sgemv_node", None, transpose_A, a, b)
        gemv_node.implementation = "FPGA_TilesByColumnHbm"
        state.add_node(gemv_node)
        create_hbm_access(state, "A", f"hbm.2:{banks_A+2}", 
            shape_A, gemv_node, "_A", False, "A")
        create_hbm_access(state, "x", "hbm.0", shape_x, gemv_node,
            "_x", False, "x")
        create_hbm_access(state, "y", "hbm.1", shape_y, gemv_node,
            "_y", False, "y")
        create_hbm_access(state, "y", None, None, gemv_node,
            "_y", True, "y")
        gemv_node.expand(sdfg, state, tile_size_x=16, tile_size_y=16)
        sdfg.sdfg_list[2].symbols["a"] = sdfg.sdfg_list[0].symbols["a"]
        sdfg.sdfg_list[2].symbols["b"] = sdfg.sdfg_list[0].symbols["b"]

        sdfg.apply_fpga_transformations(False)
        utils.update_array_shape(sdfg, "A", true_shape_A)
        utils.update_array_shape(sdfg, "y", [banks_A * M])
        sdfg.arrays["A"].storage = dtypes.StorageType.CPU_Heap
        sdfg.arrays["y"].storage = dtypes.StorageType.CPU_Heap
        for xform in optimizer.Optimizer(sdfg).get_pattern_matches(
            patterns=[hbm_copy_transform.HbmCopyTransform]):
            graph = sdfg.nodes()[xform.state_id]
            src = graph.nodes()[xform.subgraph[hbm_copy_transform.HbmCopyTransform._src_node]]
            if src.data == "A":
                if transpose_A:
                    xform.split_array_info = [1, banks_A]
                else:
                    xform.split_array_info = [banks_A, 1]
            xform.apply(sdfg)

        return sdfg

    sdfg = create_or_load(load_from, create_gemv_sdfg)
    sdfg.view()
    if transpose_A:
        size_fst, size_snd = n_size, m_size_per_bank * banks_A
    else:
        size_fst, size_snd = m_size_per_bank * banks_A, n_size
    A = random_array((size_fst, size_snd), fix_constant=1)
    x = random_array((size_snd, ), fix_constant=1)
    y = random_array((size_fst, ), fix_constant=1)
    alpha = random_array((1, ), fix_constant=0.5)
    beta = random_array((1, ), fix_constant=0.5)

    sdfg(A=A, x=x, y=y, a=alpha[0], b=beta[0])
    check = alpha[0] * (A @ x) + beta[0]*y
    assert np.allclose(check, y)

def createGemm(target : str = None):
    N = dace.symbol("N")
    M = dace.symbol("M")
    
    @dace.program
    def sgemmtest(A : dace.float32[M, N], B : dace.float32[N, 100], C : dace.float32[M, 100]):
        C[:] = A @ B

    sdfg = sgemmtest.to_sdfg()
    sdfg.apply_fpga_transformations(False)
    expand_first_libnode(sdfg, "specialize")
    expand_first_libnode(sdfg, "FPGA1DSystolic")
    sdfg.compile()
    
#exec_dot_hbm(1000, 8)
#sdfg = utils.load_precompiled_sdfg("mycompiledstuff")
#exec_axpy(1000, 10)
#ref_gemv()
exec_gemv(2, 100, 100, True)