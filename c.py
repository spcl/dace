import dace
import math
import os
from typing import Tuple
import dace
import copy
import numpy as np
from dace import InterstateEdge
from dace import Union
from dace.memlet import Memlet
from dace.properties import CodeBlock
from dace.sdfg import ControlFlowRegion
from dace.sdfg.graph import Edge
from dace.sdfg.state import ConditionalBlock
from dace.transformation.interstate import LoopToMap, branch_elimination
from dace.transformation.passes import eliminate_branches
from dace.transformation.passes.vectorization.vectorize_cpu import VectorizeCPU
from dace.transformation.passes.vectorization.vectorize_gpu import VectorizeGPU
import ctypes
import subprocess
import pathlib
from math import sin, cos, log, exp, pow
from dace.transformation.passes.fusion_inline import InlineSDFGs

LEN_1D = dace.symbol("LEN_1D")
ITERATIONS = dace.symbol("ITERATIONS")

@dace.program
def dace_s1421(
    b: dace.float64[LEN_1D],
    a: dace.float64[LEN_1D],
):
    # xx = &b[LEN_1D/2]; b[i] = xx[i] + a[i]
    half = LEN_1D // 2
    for nl in range(8 * ITERATIONS):
        for i in range(half):
            b[i] = b[half + i] + a[i]


sdfg = dace_s1421.to_sdfg()



def set_arrdtype(sdfg: dace.SDFG):
    for n, g in sdfg.all_nodes_recursive():
        if isinstance(n, dace.nodes.MapEntry):
            n.map.schedule = dace.dtypes.ScheduleType.Sequential
        if isinstance(n, dace.nodes.NestedSDFG):
            for arr_name, arr in n.sdfg.arrays.items():
                if arr.storage == dace.dtypes.StorageType.Default and arr.transient is True:
                    arr.storage = dace.dtypes.StorageType.Register
    for arr_name, arr in sdfg.arrays.items():
        if arr.storage == dace.dtypes.StorageType.Default and arr.transient is True:
            arr.storage = dace.dtypes.StorageType.Register


def run_vectorization_test(dace_func: Union[dace.SDFG, callable],
                           vector_width=8,
                           simplify=True,
                           skip_simplify=None,
                           save_sdfgs=True,
                           sdfg_name=None,
                           fuse_overlapping_loads=False,
                           filter_map=-1,
                           cleanup=False,
                           from_sdfg=False,
                           no_inline=False,
                           exact=None,
                           apply_loop_to_map=False,
                           break_vectorize=False,
                           split_all_branches=False):
    # Create copies for comparison


    # Original SDFG
    if not from_sdfg:
        sdfg: dace.SDFG = dace_func.to_sdfg(simplify=False)
        sdfg.name = sdfg_name
        if simplify:
            sdfg.simplify(validate=True, validate_all=True, skip=skip_simplify or set())
    else:
        sdfg: dace.SDFG = dace_func

    assert apply_loop_to_map is True
    if apply_loop_to_map:
        sdfg.apply_transformations_repeated(LoopToMap)
        sdfg.simplify()
        InlineSDFGs().apply_pass(sdfg, {})

    if save_sdfgs and sdfg_name:
        sdfg.save(f"{sdfg_name}.sdfg")
    c_sdfg = sdfg.compile()

    # Vectorized SDFG
    copy_sdfg: dace.SDFG = copy.deepcopy(sdfg)
    copy_sdfg.name = copy_sdfg.name + "_vectorized"
    copy_sdfg.instrument = dace.dtypes.InstrumentationType.Timer

    if split_all_branches:
        cblocks = {n for n, g in sdfg.all_nodes_recursive() if isinstance(n, ConditionalBlock)}
        for cblock in cblocks:
            xform = branch_elimination.BranchElimination()
            xform.conditional = cblock
            xform._split_branches(parent_graph=cblock.parent_graph, if_block=cblock)

    if cleanup:
        for e, g in copy_sdfg.all_edges_recursive():
            if isinstance(g, dace.SDFGState):
                if (isinstance(e.src, dace.nodes.AccessNode) and isinstance(e.dst, dace.nodes.AccessNode)
                        and isinstance(g.sdfg.arrays[e.dst.data], dace.data.Scalar)
                        and e.data.other_subset is not None):
                    # Add assignment taskelt
                    src_data = e.src.data
                    src_subset = e.data.subset if e.data.data == src_data else e.data.other_subset
                    dst_data = e.dst.data
                    dst_subset = e.data.subset if e.data.data == dst_data else e.data.other_subset
                    g.remove_edge(e)
                    t = g.add_tasklet(name=f"assign_dst_{dst_data}_from_{src_data}",
                                      code="_out = _in",
                                      inputs={"_in"},
                                      outputs={"_out"})
                    g.add_edge(e.src, e.src_conn, t, "_in",
                               dace.memlet.Memlet(data=src_data, subset=copy.deepcopy(src_subset)))
                    g.add_edge(t, "_out", e.dst, e.dst_conn,
                               dace.memlet.Memlet(data=dst_data, subset=copy.deepcopy(dst_subset)))
        copy_sdfg.validate()

    for n, g in copy_sdfg.all_nodes_recursive():
        if isinstance(n, dace.nodes.MapEntry):
            for oe in g.out_edges(n):
                if isinstance(oe.dst, dace.nodes.AccessNode):
                    if oe.data.other_subset is not None:
                        print(oe, oe.data.other_subset)
                        if isinstance(g.sdfg.arrays[oe.dst.data], dace.data.Scalar):
                            subs = oe.data.subset
                            data = oe.data.data
                            print(oe.data.subset)
                            g.remove_edge(oe)
                            for oe2 in g.out_edges(oe.dst):
                                g.add_edge(oe.src, oe.src_conn, oe2.dst, oe2.dst_conn,
                                           dace.memlet.Memlet(
                                               data=data,
                                               subset=subs,
                                           ))
                            g.remove_node(oe.dst)
    copy_sdfg.validate()

    if filter_map != -1:
        map_labels = [n.map.label for (n, g) in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.MapEntry)]
        filter_map_labels = map_labels[0:filter_map]
        filter_map = filter_map_labels
    else:
        filter_map = None

    for insert_copies in [True, False]:
        pass_info = dict()
        if not break_vectorize:
            VectorizeCPU(vector_width=vector_width,
                        fuse_overlapping_loads=fuse_overlapping_loads,
                        insert_copies=insert_copies).apply_pass(copy_sdfg, pass_info)
        else:
            from dace.transformation.passes.vectorization.vectorize_break import VectorizeBreak
            VectorizeBreak(vector_width=vector_width).apply_pass(copy_sdfg, pass_info)
        if insert_copies is True:
            copy_sdfg.name += "_cpy"
        else:
            pass

        copy_sdfg.validate()

        for n, g in copy_sdfg.all_nodes_recursive():
            if isinstance(n, dace.nodes.MapEntry):
                print(n, n.map.schedule)
                n.map.schedule = dace.dtypes.ScheduleType.Sequential
        for n, g in copy_sdfg.all_nodes_recursive():
            if isinstance(n, dace.nodes.MapEntry):
                print(n.map.schedule)
                assert n.map.schedule == dace.dtypes.ScheduleType.Sequential


        set_arrdtype(copy_sdfg)

        if save_sdfgs and sdfg_name:
            copy_sdfg.save(f"{sdfg_name}_vectorized.sdfg")
        c_copy_sdfg = copy_sdfg.compile()

        return c_copy_sdfg


run_vectorization_test(dace_s1421, sdfg_name="s1421", apply_loop_to_map=True)