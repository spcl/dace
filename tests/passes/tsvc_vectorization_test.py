# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
import math
from typing import Tuple
import dace
import copy
import pytest
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

LEN_1D = dace.symbol("LEN_1D")

def run_vectorization_test(dace_func: Union[dace.SDFG, callable],
                           arrays,
                           params,
                           vector_width=8,
                           simplify=True,
                           skip_simplify=None,
                           save_sdfgs=False,
                           sdfg_name=None,
                           fuse_overlapping_loads=False,
                           insert_copies=True,
                           filter_map=-1,
                           cleanup=False,
                           from_sdfg=False,
                           no_inline=False,
                           exact=None,
                           apply_loop_to_map=False):

    # Create copies for comparison
    arrays_orig = {k: copy.deepcopy(v) for k, v in arrays.items()}
    arrays_vec = {k: copy.deepcopy(v) for k, v in arrays.items()}

    # Original SDFG
    if not from_sdfg:
        sdfg: dace.SDFG = dace_func.to_sdfg(simplify=False)
        sdfg.name = sdfg_name
        if simplify:
            sdfg.simplify(validate=True, validate_all=True, skip=skip_simplify or set())
    else:
        sdfg: dace.SDFG = dace_func


    if apply_loop_to_map:
        sdfg.apply_transformations_repeated(LoopToMap())
        sdfg.simplify()

    if save_sdfgs and sdfg_name:
        sdfg.save(f"{sdfg_name}.sdfg")
    c_sdfg = sdfg.compile()

    # Vectorized SDFG
    copy_sdfg: dace.SDFG = copy.deepcopy(sdfg)
    copy_sdfg.name = copy_sdfg.name + "_vectorized"

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

    if filter_map != -1:
        map_labels = [n.map.label for (n, g) in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.MapEntry)]
        filter_map_labels = map_labels[0:filter_map]
        filter_map = filter_map_labels
    else:
        filter_map = None

    VectorizeCPU(vector_width=vector_width,
                 fuse_overlapping_loads=fuse_overlapping_loads,
                 insert_copies=insert_copies,
                 apply_on_maps=filter_map,
                 no_inline=no_inline,
                 fail_on_unvectorizable=True).apply_pass(copy_sdfg, {})
    copy_sdfg.validate()

    if save_sdfgs and sdfg_name:
        copy_sdfg.save(f"{sdfg_name}_vectorized.sdfg")
    c_copy_sdfg = copy_sdfg.compile()

    # Run both
    c_sdfg(**arrays_orig, **params)

    c_copy_sdfg(**arrays_vec, **params)

    # Compare results
    for name in arrays.keys():
        assert np.allclose(arrays_orig[name], arrays_vec[name], rtol=1e-32), \
            f"{name} Diff: {arrays_orig[name] - arrays_vec[name]}"
        if exact is not None:
            diff = arrays_vec[name] - exact
            assert np.allclose(arrays_vec[name], exact, rtol=0, atol=1e-300), \
                f"{name} Diff: max abs diff = {np.max(np.abs(diff))}"
    return copy_sdfg


def initialise_arrays():
    # Create array handles equivalent to the globals in C
    # Adjust shapes to match your actual code.
    a  = np.zeros(LEN_1D, dtype=np.float64)
    b  = np.zeros(LEN_1D, dtype=np.float64)
    c  = np.zeros(LEN_1D, dtype=np.float64)
    d  = np.zeros(LEN_1D, dtype=np.float64)
    e  = np.zeros(LEN_1D, dtype=np.float64)
    aa = np.zeros(LEN_1D, dtype=np.float64)
    bb = np.zeros(LEN_1D, dtype=np.float64)
    cc = np.zeros(LEN_1D, dtype=np.float64)
    return a, b, c, d, e, aa, bb, cc


@dace.program
def dace_s317(q: dace.float64[1]):
    for nl in range(50):
        q[0] = 1.0
        # Inner reduction: q *= 0.99 repeated LEN_1D/2 times
        # Equivalent to: q = 0.99**(LEN_1D/2) but we follow the loop exactly.
        for i in range(LEN_1D // 2):
            q[0] *= 0.99


def test_s317():
    q = np.zeros(1, dtype=np.float64)
    run_vectorization_test(dace_func=dace_s317,
                           arrays={"q": q},
                           params={"LEN_1D": 64},
                           save_sdfgs=True,
                           sdfg_name="dace_s317",
                           apply_loop_to_map=True)

@dace.program
def dace_s3251(a: dace.float64[LEN_1D],
               b: dace.float64[LEN_1D],
               c: dace.float64[LEN_1D],
               d: dace.float64[LEN_1D],
               e: dace.float64[LEN_1D]):
    for nl in range(10):
        for i in range(LEN_1D - 1):
            a[i + 1] = b[i] + c[i]
            b[i]     = c[i] * e[i]
            d[i]     = a[i] * e[i]

@dace.program
def dace_s491(a: dace.float64[LEN_1D],
              b: dace.float64[LEN_1D],
              c: dace.float64[LEN_1D],
              d: dace.float64[LEN_1D],
              ip: dace.int32[LEN_1D]):
    for nl in range(10):
        for i in range(LEN_1D):
            a[ip[i]] = b[i] + c[i] * d[i]

def test_s491():
    LEN_1D_val = 64

    a = np.random.rand(LEN_1D_val).astype(np.float64)
    b = np.random.rand(LEN_1D_val).astype(np.float64)
    c = np.random.rand(LEN_1D_val).astype(np.float64)
    d = np.random.rand(LEN_1D_val).astype(np.float64)

    ip = np.random.permutation(LEN_1D_val).astype(np.int32)

    run_vectorization_test(
        dace_func=dace_s491,
        arrays={"a": a, "b": b, "c": c, "d": d, "ip": ip},
        params={"LEN_1D": LEN_1D_val},
        save_sdfgs=True,
        sdfg_name="dace_s491",
        apply_loop_to_map=True,
    )

    return a

@dace.program
def dace_s293(a: dace.float64[LEN_1D]):
    for nl in range(50):
        # loop peeling structure preserved
        a0 = a[0]
        for i in range(LEN_1D):
            a[i] = a0

def test_s293():
    LEN_1D_val = 64  # example, adjust as needed

    # Allocate array a
    a = np.random.rand(LEN_1D_val).astype(np.float64)

    # Run DaCe test harness
    run_vectorization_test(
        dace_func=dace_s293,
        arrays={"a": a},
        params={"LEN_1D": LEN_1D_val},  # or your iteration count
        save_sdfgs=True,
        sdfg_name="dace_s293",
        apply_loop_to_map=True,
    )

    return a



def test_s3251():
    LEN_1D_val = 64

    a = np.random.rand(LEN_1D_val).astype(np.float64)
    b = np.random.rand(LEN_1D_val).astype(np.float64)
    c = np.random.rand(LEN_1D_val).astype(np.float64)
    d = np.random.rand(LEN_1D_val).astype(np.float64)
    e = np.random.rand(LEN_1D_val).astype(np.float64)

    run_vectorization_test(
        dace_func=dace_s3251,
        arrays={"a": a, "b": b, "c": c, "d": d, "e": e},
        params={"LEN_1D": LEN_1D_val},
        save_sdfgs=True,
        sdfg_name="dace_s3251",
        apply_loop_to_map=True,
    )

    return a, b, c, d, e


@dace.program
def dace_s441(a: dace.float64[LEN_1D],
              b: dace.float64[LEN_1D],
              c: dace.float64[LEN_1D],
              d: dace.float64[LEN_1D]):
    for nl in range(50):  # or iterations parameter
        for i in range(LEN_1D):
            if d[i] < 0.0:
                a[i] = a[i] + b[i] * c[i]
            elif d[i] == 0.0:
                a[i] = a[i] + b[i] * b[i]
            else:
                a[i] = a[i] + c[i] * c[i]

def test_s441():
    LEN_1D_val = 64  # example length

    # Allocate random inputs
    a = np.random.rand(LEN_1D_val).astype(np.float64)
    b = np.random.rand(LEN_1D_val).astype(np.float64)
    c = np.random.rand(LEN_1D_val).astype(np.float64)
    d = np.random.randn(LEN_1D_val).astype(np.float64)  # includes negative, zero, positive

    sdfg = dace_s441.to_sdfg()
    eliminate_branches.EliminateBranches().apply_pass(sdfg, {})
    branches = {n for (n,g) in sdfg.all_nodes_recursive() if isinstance(n, ConditionalBlock)}
    if len(branches) > 0:
        sdfg.save("branch_elimination_failed_s441.sdfg")
        assert False

    # Run DaCe test harness (your helper function)
    run_vectorization_test(
        dace_func=dace_s441,
        arrays={"a": a, "b": b, "c": c, "d": d},
        params={"LEN_1D": LEN_1D_val},
        save_sdfgs=True,
        sdfg_name="dace_s441",
        apply_loop_to_map=True,
    )

    return a