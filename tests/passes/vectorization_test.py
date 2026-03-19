# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
import math
from typing import Tuple
import dace
import copy
import pytest
import numpy
from dace import InterstateEdge
from dace import Union
from dace.memlet import Memlet
from dace.properties import CodeBlock
from dace.sdfg import ControlFlowRegion
from dace.sdfg.graph import Edge
from dace.sdfg.state import ConditionalBlock
from dace.transformation.interstate import branch_elimination
from dace.transformation.passes.vectorization.tasklet_preprocessing_passes import ReplaceSTDExpWithDaCeExp, ReplaceSTDLogWithDaCeLog, ReplaceSTDPowWithDaCePow
from dace.transformation.passes.vectorization.vectorize_cpu import VectorizeCPU
from dace.transformation.passes.vectorization.vectorize_gpu import VectorizeGPU

# Vector Addition Symbols
N = dace.symbol('N')
# Tasklet in NestedSDFGs Symbols
S1 = dace.symbol("S1")
S2 = dace.symbol("S2")
S = dace.symbol("S")
# CloudSC Symbols
klev = dace.symbol("klev")
kidia = dace.symbol("kidia")
kfdia = dace.symbol("kfdia")
# SpMV Symbols
n = dace.symbol('n')  # number of rows
m = dace.symbol('m')  # number of columns
nnz = dace.symbol('nnz')  # number of nonzeros


@dace.program
def vadds_gpu(A: dace.float64[N, N] @ dace.dtypes.StorageType.GPU_Global,
              B: dace.float64[N, N] @ dace.dtypes.StorageType.GPU_Global):
    for i, j in dace.map[0:N, 0:N] @ dace.dtypes.ScheduleType.GPU_Device:
        A[i, j] = A[i, j] + B[i, j]
    for i, j in dace.map[0:N, 0:N] @ dace.dtypes.ScheduleType.GPU_Device:
        B[i, j] = 3 * B[i, j]**2.0 + 2.0


@dace.program
def vadds_cpu(A: dace.float64[N, N], B: dace.float64[N, N]):
    for i, j in dace.map[0:N, 0:N]:
        A[i, j] = A[i, j] + B[i, j]
    for i, j in dace.map[0:N, 0:N]:
        B[i, j] = 3 * B[i, j]**2.0 + 2.0


@dace.program
def vabs(A: dace.float64[N, N]):
    for i, j in dace.map[0:N, 0:N]:
        A[i, j] = abs(A[i, j])


@dace.program
def unary_symbol(A: dace.float64[N, N]):
    for i, j in dace.map[0:N, 0:N]:
        A[i, j] = -999


@dace.program
def vadd_int(A: dace.int64[N, N], B: dace.int64[N, N]):
    for i, j in dace.map[0:N, 0:N]:
        A[i, j] = A[i, j] + B[i, j]
    for i, j in dace.map[0:N, 0:N]:
        B[i, j] = 3 * B[i, j] * 2 + 20


@dace.program
def vadd_with_different_types(A: dace.int64[N, N], B: dace.float64[N, N]):
    for i, j in dace.map[0:N, 0:N]:
        A[i, j] = A[i, j] + B[i, j]
    for i, j in dace.map[0:N, 0:N]:
        B[i, j] = 3 * B[i, j] * 2 + 20


@dace.program
def vadd_int_with_scalars(A: dace.int64[N, N], B: dace.int64[N, N], c1: dace.int64, c2: dace.int64):
    for i, j in dace.map[0:N, 0:N]:
        c3 = c1 * c2
        A[i, j] = A[i, j] + B[i, j] + c3
    for i, j in dace.map[0:N, 0:N]:
        c4 = c1 + c2
        B[i, j] = 3 * B[i, j] * 2 + 20 + c4


@dace.program
def vsubs_cpu(A: dace.float64[N, N], B: dace.float64[N, N]):
    for i, j in dace.map[0:N, 0:N]:
        A[i, j] = A[i, j] - B[i, j]


@dace.program
def vexp_cpu(A: dace.float64[N, N]):
    for i, j in dace.map[0:N, 0:N]:
        A[i, j] = math.exp(A[i, j])


@dace.program
def vsubs_two_cpu(A: dace.float64[N, N], B: dace.float64[N, N]):
    for i, j in dace.map[0:N, 0:N]:
        A[i, j] = B[i, j] - A[i, j]


@dace.program
def v_const_subs_cpu(A: dace.float64[N, N]):
    for i, j in dace.map[0:N, 0:N]:
        A[i, j] = A[i, j] - 1.0


@dace.program
def v_const_subs_two_cpu(A: dace.float64[N, N]):
    for i, j in dace.map[0:N, 0:N]:
        A[i, j] = 1.0 - A[i, j]


@dace.program
def unsupported_op(A: dace.float64[N, N], B: dace.float64[N, N]):
    for i, j in dace.map[0:N, 0:N]:
        A[i, j] = math.exp(B[i, j])


@dace.program
def memset(A: dace.float64[N, N]):
    for i in dace.map[0:N]:
        for j in dace.map[0:N]:
            A[i, j] = 0.0


@dace.program
def v_const_subs_4d(A: dace.float64[N, N, N, N]):
    for i, j, k, m in dace.map[0:N, 0:N, 0:N, 0:N]:
        A[i, j, k, m] = 1.0 - A[i, j, k, m]


@dace.program
def v_const_subs_4d_indirect_access(A: dace.float64[N, N, N, N], c: dace.int64):
    for i, j, k in dace.map[0:N, 0:N, 0:N]:
        A[c + 1, i, j, k] = 1.0 - A[c + 1, i, j, k]


@dace.program
def memset_4d(A: dace.float64[N, N, N, N]):
    for i in dace.map[0:N]:
        for j in dace.map[0:N]:
            for k in dace.map[0:N]:
                for m in dace.map[0:N]:
                    A[i, j, k, m] = 0.0


def test_memset_4d():
    N = 8
    A = numpy.random.random((N, N, N, N))
    c = 0.8

    run_vectorization_test(
        dace_func=memset_4d,
        arrays={
            'A': A,
        },
        params={
            'N': N,
        },
        vector_width=8,
        save_sdfgs=True,
        sdfg_name="memset_4d",
    )


def test_v_const_subs_4d():
    N = 8
    A = numpy.random.random((N, N, N, N))

    run_vectorization_test(
        dace_func=v_const_subs_4d,
        arrays={
            'A': A,
        },
        params={
            'N': N,
        },
        vector_width=8,
        save_sdfgs=True,
        sdfg_name="v_const_subs_4d",
    )


def test_v_const_subs_4d_indirect_access():
    N = 8
    A = numpy.random.random((N, N, N, N))

    run_vectorization_test(
        dace_func=v_const_subs_4d_indirect_access,
        arrays={
            'A': A,
        },
        params={
            'N': N,
            'c': 0,
        },
        vector_width=8,
        save_sdfgs=True,
        sdfg_name="v_const_subs_4d_indirect_access",
    )


def foo(A):
    A = 0.0


@dace.program
def nested_memset(A: dace.float64[N, N]):
    for i in dace.map[0:N]:
        for j in dace.map[0:N]:
            foo(A[i, j])


@dace.program
def max_with_constant(A: dace.float64[N, N], c: dace.float64):
    for i in dace.map[0:N]:
        for j in dace.map[0:N]:
            A[i, j] = max(c, A[i, j])


@dace.program
def max_with_constant_reversed_order(A: dace.float64[N, N], c: dace.float64):
    for i in dace.map[0:N]:
        for j in dace.map[0:N]:
            A[i, j] = max(A[i, j], c)


def test_max_with_constant():
    N = 64
    A = numpy.random.random((N, N))
    c = 0.8

    run_vectorization_test(
        dace_func=max_with_constant,
        arrays={
            'A': A,
        },
        params={
            'N': N,
            'c': c
        },
        vector_width=8,
        save_sdfgs=True,
        sdfg_name="max_with_constant",
    )


def test_max_with_constant_reversed_order():
    N = 64
    A = numpy.random.random((N, N))
    c = 0.8

    run_vectorization_test(
        dace_func=max_with_constant_reversed_order,
        arrays={
            'A': A,
        },
        params={
            'N': N,
            'c': c
        },
        vector_width=8,
        save_sdfgs=True,
        sdfg_name="max_with_constant_reversed_order",
    )


@dace.program
def division_by_zero(A: dace.float64[N], B: dace.float64[N], c: dace.float64):
    for i in dace.map[
            0:N,
    ]:
        if A[i] > 0.0:
            B[i] = c / A[i]
        else:
            B[i] = 0.0


@dace.program
def no_maps(A: dace.float64[N, N], B: dace.float64[N, N]):
    i = 8
    j = 7
    A[i, j] = 2.0 * A[i, j]
    B[i + 1, j + 1] = B[i, j] / 1.5


@dace.program
def cloudsc_snippet_one(
    za: dace.float64[kfdia, klev],
    zliqfrac: dace.float64[kfdia, klev],
    zicefrac: dace.float64[kfdia, klev],
    zqx: dace.float64[5, klev + 1, kfdia + 1],
    zli: dace.float64[kfdia, klev],
    rlmin: dace.float64,
    z1: dace.int64,
):
    # note: outer loop over j (kfdia) first, then i (klev) to match column-major
    for j in range(kfdia):
        for i in range(klev):
            zaji = za[j, i]
            za[j, i] = 2.0 * zaji - 5
            cond1 = rlmin > 0.5 * (zqx[z1, i, j] + zqx[z1, j + 1, i + 1])
            if cond1:
                zliqfrac[j, i] = zqx[z1, j, i] * zli[j, i]
                zicefrac[j, i] = 1 - zliqfrac[j, i]
            else:
                zliqfrac[j, i] = 0
                zicefrac[j, i] = 0


@dace.program
def tasklet_in_nested_sdfg(
    a: dace.float64[S, S],
    b: dace.float64[S, S],
    offset1: dace.int64,
    offset2: dace.int64,
):
    for i, j in dace.map[S1:S2:1, S1:S2:1] @ dace.dtypes.ScheduleType.Sequential:
        # Complicated NestedSDFG with offset1 and offset2 in the NestedSDFG as symbols
        a[i + offset1, j + offset2] = ((1.5 * b[i + offset1, j + offset2]) + (2.0 * a[i + offset1, j + offset2])) / 3.5


@dace.program
def tasklet_in_nested_sdfg_two(
    a: dace.float64[S, S],
    b: dace.float64[S, S],
    offset1: dace.int64,
    offset2: dace.int64,
):
    # If a scalar is always added to a map param
    # Then move the scalar to the loop like this
    #for i, j in dace.map[S1:S2:1, S1:S2:1] @ dace.dtypes.ScheduleType.Sequential:
    #    a[i + offset1, j + offset2] = (
    #        (1.5 * b[i + offset1, j + offset2]) +
    #        (2.0 * a[i + offset1, j + offset2])
    #    ) / 3.5
    for i, j in dace.map[S1 - offset1:S2 - offset1:1,
                         S1 - offset2:S2 - offset2:1] @ dace.dtypes.ScheduleType.Sequential:
        a[i, j] = ((1.5 * b[i, j]) + (2.0 * a[i, j])) / 3.5


@dace.program
def tasklets_in_if(
    a: dace.float64[S, S],
    b: dace.float64[S, S],
    d: dace.float64[S, S],
    c: dace.float64,
):
    for i in dace.map[0:S:1]:
        for j in dace.map[0:S:1]:
            if a[i, j] > c:
                b[i, j] = b[i, j] + d[i, j]
            else:
                b[i, j] = b[i, j] - d[i, j]
            b[i, j] = (1 - a[i, j]) * c


@dace.program
def tasklets_in_if_two(
    a: dace.float64[S, S],
    b: dace.float64,
    c: dace.float64[S, S],
    d: dace.float64[S, S],
    e: dace.float64[S, S],
    f: dace.float64,
):
    for i in dace.map[0:S:1]:
        for j in dace.map[0:S:1]:
            if a[i, j] < b:
                e[i, j] = (c[i, j] * f * a[i, j] * 2.0) - a[i, j]


@dace.program
def spmv_csr(indptr: dace.int64[n + 1], indices: dace.int64[nnz], data: dace.float64[nnz], x: dace.float64[m],
             y: dace.float64[n]):
    n_rows = len(indptr) - 1

    for i in dace.map[0:n_rows:1]:
        row_start = indptr[i]
        row_end = indptr[i + 1]
        tmp = 0.0
        for idx in dace.map[row_start:row_end:1]:
            j = indices[idx]
            tmp = tmp + data[idx] * x[j]
        y[i] = tmp


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
                           exact=None):

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

    c_copy_sdfg = copy_sdfg.compile()

    # Run both
    c_sdfg(**arrays_orig, **params)

    c_copy_sdfg(**arrays_vec, **params)

    # Compare results
    for name in arrays.keys():
        assert numpy.allclose(arrays_orig[name], arrays_vec[name], rtol=1e-32), \
            f"{name} Diff: {arrays_orig[name] - arrays_vec[name]}"
        if exact is not None:
            diff = arrays_vec[name] - exact
            assert numpy.allclose(arrays_vec[name], exact, rtol=0, atol=1e-300), \
                f"{name} Diff: max abs diff = {numpy.max(numpy.abs(diff))}"
    return copy_sdfg


def test_division_by_zero_cpu():
    N = 256
    A = numpy.random.random((N, ))
    B = numpy.random.random((N, ))

    run_vectorization_test(
        dace_func=division_by_zero,
        arrays={
            'A': A,
            'B': B
        },
        params={
            'N': N,
            "c": 8.9
        },
        vector_width=8,
        save_sdfgs=True,
        sdfg_name="division_by_zero",
    )


def test_vsubs_cpu():
    N = 64
    A = numpy.random.random((N, N))
    B = numpy.random.random((N, N))

    run_vectorization_test(
        dace_func=vsubs_cpu,
        arrays={
            'A': A,
            'B': B
        },
        params={'N': N},
        vector_width=8,
        save_sdfgs=True,
        sdfg_name="vsubs_one",
    )


def test_memset():
    N = 64
    A = numpy.random.random((N, N))

    run_vectorization_test(dace_func=memset,
                           arrays={
                               'A': A,
                           },
                           params={'N': N},
                           vector_width=8,
                           save_sdfgs=True,
                           sdfg_name="memset",
                           exact=0.0)


def test_memset_with_fuse_and_copyin_enabled():
    N = 64
    A = numpy.random.random((N, N))

    run_vectorization_test(dace_func=memset,
                           arrays={
                               'A': A,
                           },
                           params={'N': N},
                           vector_width=8,
                           save_sdfgs=True,
                           sdfg_name="memset_with_fuse_and_copy_in_enabled",
                           fuse_overlapping_loads=True,
                           insert_copies=True,
                           exact=0.0)


def test_nested_memset_with_fuse_and_copyin_enabled():
    N = 64
    A = numpy.random.random((N, N))

    run_vectorization_test(dace_func=nested_memset,
                           arrays={
                               'A': A,
                           },
                           params={'N': N},
                           vector_width=8,
                           save_sdfgs=True,
                           sdfg_name="nested_memset_with_fuse_and_copy_in_enabled",
                           fuse_overlapping_loads=True,
                           insert_copies=True,
                           simplify=False,
                           exact=0.0)


def test_vexp_cpu():
    N = 64
    A = numpy.random.random((N, N))

    run_vectorization_test(
        dace_func=vexp_cpu,
        arrays={
            'A': A,
        },
        params={'N': N},
        vector_width=8,
        save_sdfgs=True,
        sdfg_name="vexp_one",
    )


def test_vsubs_two_cpu():
    N = 64
    A = numpy.random.random((N, N))
    B = numpy.random.random((N, N))

    run_vectorization_test(
        dace_func=vsubs_two_cpu,
        arrays={
            'A': A,
            'B': B
        },
        params={'N': N},
        vector_width=8,
        save_sdfgs=True,
        sdfg_name="vsubs_two",
    )


def test_v_const_subs_cpu():
    N = 64
    A = numpy.random.random((N, N))

    run_vectorization_test(
        dace_func=v_const_subs_cpu,
        arrays={'A': A},
        params={'N': N},
        vector_width=8,
        save_sdfgs=True,
        sdfg_name="v_const_subs_one",
    )


def test_v_const_subs_two_cpu():
    N = 64
    A = numpy.random.random((N, N))

    run_vectorization_test(
        dace_func=v_const_subs_two_cpu,
        arrays={'A': A},
        params={'N': N},
        vector_width=8,
        sdfg_name="v_const_subs_two",
        save_sdfgs=True,
    )


def test_simple_cpu():
    A = numpy.random.random((64, 64))
    B = numpy.random.random((64, 64))

    run_vectorization_test(
        dace_func=vadds_cpu,
        arrays={
            'A': A,
            'B': B
        },
        params={'N': 64},
        vector_width=4,
        sdfg_name="simple_cpu",
        save_sdfgs=True,
    )


def test_unsupported_op():
    A = numpy.random.random((64, 64))
    B = numpy.random.random((64, 64))

    run_vectorization_test(dace_func=unsupported_op,
                           arrays={
                               'A': A,
                               'B': B
                           },
                           params={'N': 64},
                           vector_width=4,
                           skip_simplify={"ScalarToSymbolPromotion"},
                           save_sdfgs=True,
                           sdfg_name="unsupported_op")


def test_unsupported_op_two():
    A = numpy.random.random((64, 64))
    B = numpy.random.random((64, 64))

    run_vectorization_test(dace_func=unsupported_op,
                           arrays={
                               'A': A,
                               'B': B
                           },
                           params={'N': 64},
                           vector_width=4,
                           save_sdfgs=True,
                           sdfg_name="unsupported_op_two")


def test_nested_sdfg():
    _S1 = 1
    _S2 = 65
    _S = 64
    A = numpy.random.random((_S, _S))
    B = numpy.random.random((_S, _S))

    run_vectorization_test(dace_func=tasklet_in_nested_sdfg,
                           arrays={
                               'a': A,
                               'b': B
                           },
                           params={
                               'S': _S,
                               'S1': _S1,
                               'S2': _S2,
                               'offset1': -1,
                               'offset2': -1
                           },
                           vector_width=8,
                           save_sdfgs=True,
                           sdfg_name="nested_tasklets")


def test_no_maps():
    _N = 16
    A = numpy.random.random((_N, _N))
    B = numpy.random.random((_N, _N))

    run_vectorization_test(dace_func=no_maps,
                           arrays={
                               'A': A,
                               'B': B
                           },
                           params={'N': _N},
                           vector_width=8,
                           save_sdfgs=True,
                           sdfg_name="no_maps")


def test_tasklets_in_if():
    _S = 64
    A = numpy.random.random((_S, _S))
    B = numpy.random.random((_S, _S))
    D = numpy.random.random((_S, _S))
    C = numpy.random.random((1, ))

    arrays_orig = {'a': copy.deepcopy(A), 'b': copy.deepcopy(B), 'c': copy.deepcopy(C), 'd': copy.deepcopy(D)}
    arrays_vec = {'a': copy.deepcopy(A), 'b': copy.deepcopy(B), 'c': copy.deepcopy(C), 'd': copy.deepcopy(D)}

    sdfg = tasklets_in_if.to_sdfg()
    copy_sdfg = copy.deepcopy(sdfg)
    copy_sdfg.name = sdfg.name + "_vectorized"
    sdfg.save("nested_tasklets_in_if.sdfg")
    VectorizeCPU(vector_width=8, fail_on_unvectorizable=True).apply_pass(copy_sdfg, {})
    copy_sdfg.save("nested_tasklets_in_if_vectorized.sdfg")

    c_sdfg = sdfg.compile()
    c_copy_sdfg = copy_sdfg.compile()

    c_sdfg(a=arrays_orig['a'], b=arrays_orig['b'], c=arrays_orig['c'][0], d=arrays_orig['d'], S=_S)
    c_copy_sdfg(a=arrays_vec['a'], b=arrays_vec['b'], c=arrays_vec['c'][0], d=arrays_vec['d'], S=_S)

    assert numpy.allclose(arrays_orig['a'], arrays_vec['a'],
                          atol=1e-12), f"A Diff: {arrays_orig['a'] - arrays_vec['a']}"
    assert numpy.allclose(arrays_orig['b'], arrays_vec['b'],
                          atol=1e-12), f"B Diff: {arrays_orig['b'] - arrays_vec['b']}"
    assert numpy.allclose(arrays_orig['c'], arrays_vec['c'],
                          atol=1e-12), f"C Diff: {arrays_orig['c'] - arrays_vec['c']}"
    assert numpy.allclose(arrays_orig['d'], arrays_vec['d'],
                          atol=1e-12), f"D Diff: {arrays_orig['d'] - arrays_vec['d']}"


# There was a non-deterministic bug, therefore run it multiple times
def test_tasklets_in_if_two():
    _S = 64
    A = numpy.random.random((_S, _S))
    B = numpy.random.random((1, ))
    C = numpy.random.random((_S, _S))
    D = numpy.random.random((_S, _S))
    E = numpy.random.random((_S, _S))
    F = numpy.random.random((1, ))

    arrays_orig = {
        'a': copy.deepcopy(A),
        'b': copy.deepcopy(B),
        'c': copy.deepcopy(C),
        'd': copy.deepcopy(D),
        'e': copy.deepcopy(E),
        'f': copy.deepcopy(F)
    }
    arrays_vec = {
        'a': copy.deepcopy(A),
        'b': copy.deepcopy(B),
        'c': copy.deepcopy(C),
        'd': copy.deepcopy(D),
        'e': copy.deepcopy(E),
        'f': copy.deepcopy(F)
    }

    sdfg = tasklets_in_if_two.to_sdfg(simplify=False)
    sdfg.simplify(skip=["ScalarToSymbolPromotion"])
    copy_sdfg = copy.deepcopy(sdfg)
    copy_sdfg.name = copy_sdfg.name + "_vectorized"
    sdfg.save("nested_tasklets.sdfg")
    VectorizeCPU(vector_width=8, fail_on_unvectorizable=True).apply_pass(copy_sdfg, {})
    copy_sdfg.save("nested_tasklets_vectorized.sdfg")

    c_sdfg = sdfg.compile()
    c_copy_sdfg = copy_sdfg.compile()

    c_sdfg(a=arrays_orig['a'],
           b=arrays_orig['b'][0],
           c=arrays_orig['c'],
           d=arrays_orig['d'],
           e=arrays_orig['e'],
           f=arrays_orig['f'][0],
           S=_S)
    c_copy_sdfg(a=arrays_vec['a'],
                b=arrays_vec['b'][0],
                c=arrays_vec['c'],
                d=arrays_vec['d'],
                e=arrays_vec['e'],
                f=arrays_vec['f'][0],
                S=_S)

    assert numpy.allclose(arrays_orig['a'], arrays_vec['a'], atol=1e-12), f"{arrays_orig['a'] - arrays_vec['a']}"
    assert numpy.allclose(arrays_orig['b'], arrays_vec['b'], atol=1e-12), f"{arrays_orig['b'] - arrays_vec['b']}"
    assert numpy.allclose(arrays_orig['c'], arrays_vec['c'], atol=1e-12), f"{arrays_orig['c'] - arrays_vec['c']}"
    assert numpy.allclose(arrays_orig['d'], arrays_vec['d'], atol=1e-12), f"{arrays_orig['d'] - arrays_vec['d']}"
    assert numpy.allclose(arrays_orig['e'], arrays_vec['e'], atol=1e-12), f"{arrays_orig['e'] - arrays_vec['e']}"
    assert numpy.allclose(arrays_orig['f'], arrays_vec['f'], atol=1e-12), f"{arrays_orig['f'] - arrays_vec['f']}"


def _dense_to_csr(dense: numpy.ndarray):
    """
    Convert a 2D dense numpy array to CSR arrays (data, indices, indptr).
    Keeps the same ordering usually used by CSR: row-major.
    """
    data = []
    indices = []
    indptr = [0]
    nrows, ncols = dense.shape
    for i in range(nrows):
        row_nnz = 0
        for j in range(ncols):
            v = dense[i, j]
            if v != 0:
                data.append(v)
                indices.append(j)
                row_nnz += 1
        indptr.append(indptr[-1] + row_nnz)
    return numpy.array(data, dtype=dense.dtype), numpy.array(indices, dtype=numpy.int64), numpy.array(indptr,
                                                                                                      dtype=numpy.int64)


def trim_to_multiple_of_8(dense: numpy.ndarray) -> numpy.ndarray:
    """
    For each row in the dense matrix, drop (set to zero) the last few nonzeros
    so that the number of nonzeros becomes a multiple of 8.
    """
    A = dense.copy()
    for i in range(A.shape[0]):
        nz_idx = numpy.flatnonzero(A[i])
        excess = len(nz_idx) % 8
        if excess:
            # zero out the last 'excess' nonzeros
            A[i, nz_idx[-excess:]] = 0
    return A


@dace.program
def overlapping_access(A: dace.float64[2, 2, S], B: dace.float64[S]):
    for i in dace.map[0:S:1]:
        B[i] = A[0, 0, i] + A[1, 0, i]


@dace.program
def overlapping_access_same_src_and_dst(A: dace.float64[2, 2, S]):
    for i in dace.map[0:S:1]:
        A[0, 0, i] = A[0, 0, i] + A[1, 0, i]


@dace.program
def overlapping_access_same_src_and_dst_in_nestedsdfg(A: dace.float64[2, 2, S], js: dace.int64):
    for i in dace.map[0:S:1]:
        j = js + 1
        A[0, j, i] = A[0, j, i] + A[1, j, i]


def test_overlapping_access():
    _S = 64
    A = numpy.random.random((2, 2, _S))
    B = numpy.random.random((_S))

    run_vectorization_test(dace_func=overlapping_access,
                           arrays={
                               'A': A,
                               'B': B
                           },
                           params={
                               'S': _S,
                           },
                           vector_width=8,
                           save_sdfgs=True,
                           sdfg_name="overlapping_access")


def test_overlapping_access_same_src_and_dst():
    _S = 64
    A = numpy.random.random((2, 2, _S))

    run_vectorization_test(dace_func=overlapping_access_same_src_and_dst,
                           arrays={
                               'A': A,
                           },
                           params={
                               'S': _S,
                           },
                           vector_width=8,
                           save_sdfgs=True,
                           sdfg_name="overlapping_access_same_src_and_dst")


@pytest.mark.parametrize("run_id", [i for i in range(4)])
def test_overlapping_access_same_src_and_dst_in_nestedsdfg(run_id):
    _S = 64
    A = numpy.random.random((2, 2, _S))

    run_vectorization_test(dace_func=overlapping_access_same_src_and_dst_in_nestedsdfg,
                           arrays={
                               'A': A,
                           },
                           params={
                               'S': _S,
                               "js": 0,
                           },
                           vector_width=8,
                           save_sdfgs=True,
                           sdfg_name=f"overlapping_access_same_src_and_dst_in_nestedsdfg_runid_{run_id}")


def test_spmv():
    _N = 32
    density = 0.25
    dense = numpy.random.random((_N, _N))
    mask = numpy.random.random((_N, _N)) < density
    dense = dense * mask  # many zeros

    # Create CSR arrays (data, indices, indptr)
    dense = trim_to_multiple_of_8(dense)
    data, indices, indptr = _dense_to_csr(dense)

    # input / output vectors
    x = numpy.random.random((_N, ))
    y_orig = numpy.zeros_like(x)
    y_vec = numpy.zeros_like(x)
    _nnz = len(data)

    # Original SDFG
    sdfg = spmv_csr.to_sdfg()

    # Vectorized SDFG
    copy_sdfg = copy.deepcopy(sdfg)
    copy_sdfg.name = sdfg.name + "_vectorized"
    sdfg.save("spmv.sdfg")
    VectorizeCPU(vector_width=8, fail_on_unvectorizable=True).apply_pass(copy_sdfg, {})
    copy_sdfg.save("spmv_vectorized.sdfg")

    c_sdfg = sdfg.compile()
    c_copy_sdfg = copy_sdfg.compile()

    c_sdfg(data=data, indices=indices, indptr=indptr, x=x, y=y_orig, n=_N, nnz=_nnz)
    c_copy_sdfg(data=data, indices=indices, indptr=indptr, x=x, y=y_vec, n=_N, nnz=_nnz)

    # Compare results
    assert numpy.allclose(y_orig, y_vec, atol=1e-12), f"{y_orig - y_vec}"


@dace.program
def jacobi2d(A: dace.float64[S, S], B: dace.float64[S, S], tsteps: dace.int64):  #, N, tsteps):
    for t in range(tsteps):
        for i, j in dace.map[0:S - 2, 0:S - 2]:
            B[i + 1, j + 1] = 0.2 * (A[i + 1, j + 1] + A[i, j + 1] + A[i + 2, j + 1] + A[i + 1, j] + A[i + 1, j + 2])

        for i, j in dace.map[0:S - 2, 0:S - 2]:
            A[i + 1, j + 1] = 0.2 * (B[i + 1, j + 1] + B[i, j + 1] + B[i + 2, j + 1] + B[i + 1, j] + B[i + 1, j + 2])


def test_jacobi2d():
    _S = 66
    A = numpy.random.random((_S, _S))
    B = numpy.random.random((_S, _S))

    run_vectorization_test(dace_func=jacobi2d,
                           arrays={
                               'A': A,
                               'B': B
                           },
                           params={
                               'S': _S,
                               'tsteps': 5,
                           },
                           vector_width=8,
                           save_sdfgs=True,
                           sdfg_name="jacobi2d")


def test_jacobi2d_with_filter_map():
    _S = 66
    A = numpy.random.random((_S, _S))
    B = numpy.random.random((_S, _S))

    sdfg = jacobi2d.to_sdfg()

    run_vectorization_test(dace_func=jacobi2d,
                           arrays={
                               'A': A,
                               'B': B
                           },
                           params={
                               'S': _S,
                               'tsteps': 5,
                           },
                           vector_width=8,
                           save_sdfgs=True,
                           sdfg_name="jacobi2d_with_filter_map",
                           filter_map=1)


def test_jacobi2d_with_fuse_overlapping_loads():
    _S = 66
    A = numpy.random.random((_S, _S))
    B = numpy.random.random((_S, _S))

    vectorized_sdfg: dace.SDFG = run_vectorization_test(dace_func=jacobi2d,
                                                        arrays={
                                                            'A': A,
                                                            'B': B
                                                        },
                                                        params={
                                                            'S': _S,
                                                            'tsteps': 5,
                                                        },
                                                        vector_width=8,
                                                        save_sdfgs=True,
                                                        sdfg_name="jacobi2d_with_fuse_overlapping_loads",
                                                        fuse_overlapping_loads=True,
                                                        insert_copies=True)

    # Should have 1 access node between two maps
    inner_map_entries = {(n, g)
                         for n, g in vectorized_sdfg.all_nodes_recursive()
                         if isinstance(n, dace.nodes.MapEntry) and g.scope_dict()[n] is not None}
    for inner_map_entry, state in inner_map_entries:
        src_access_nodes = {
            ie.src
            for ie in state.in_edges(inner_map_entry) if isinstance(ie.src, dace.nodes.AccessNode)
        }

        src_src_access_nodes = set()
        for src_acc_node in src_access_nodes:
            src_src_access_nodes = src_src_access_nodes.union(
                {ie.src
                 for ie in state.in_edges(src_acc_node) if isinstance(ie.src, dace.nodes.AccessNode)})

        assert len(src_src_access_nodes
                   ) == 1, f"Excepted one access node got {len(src_src_access_nodes)}, ({src_src_access_nodes})"


@pytest.mark.parametrize("param_tuple", [(True, True), (True, False), (False, True), (False, False)])
def test_jacobi2d_with_parameters(param_tuple):
    fuse_overlapping_loads, insert_copies = param_tuple
    _S = 66
    A = numpy.random.random((_S, _S))
    B = numpy.random.random((_S, _S))

    run_vectorization_test(
        dace_func=jacobi2d,
        arrays={
            'A': A,
            'B': B
        },
        params={
            'S': _S,
            'tsteps': 5,
        },
        vector_width=8,
        save_sdfgs=True,
        sdfg_name=f"jacobi2d_with_fuse_overlapping_loads_{fuse_overlapping_loads}_with_insert_copies_{insert_copies}",
        fuse_overlapping_loads=fuse_overlapping_loads,
        insert_copies=insert_copies)


C = 32


def _get_disjoint_chain_sdfg(trivial_if: bool, fortran_layout: bool = False) -> dace.SDFG:
    sd1 = dace.SDFG("disjoint_chain")
    cb1 = ConditionalBlock("cond_if_cond_58", sdfg=sd1, parent=sd1)
    ss1 = sd1.add_state(label="pre", is_start_block=True)
    sd1.add_node(cb1, is_start_block=False)

    cfg1 = dace.ControlFlowRegion(label="cond_58_true", sdfg=sd1, parent=cb1)
    s1 = cfg1.add_state("main_1", is_start_block=True)
    cfg2 = dace.ControlFlowRegion(label="cond_58_false", sdfg=sd1, parent=cb1)
    s2 = cfg2.add_state("main_2", is_start_block=True)

    cb1.add_branch(
        condition=CodeBlock("_if_cond_58 == 1" if not trivial_if else "1 == 1"),
        branch=cfg1,
    )
    cb1.add_branch(
        condition=None,
        branch=cfg2,
    )
    for arr_name, shape in [
        ("zsolqa", (5, 5, C)),
        ("zrainaut", (C, )),
        ("zrainacc", (C, )),
        ("ztp1", (C, )),
    ]:
        sd1.add_array(arr_name, shape, dace.float64)
    sd1.add_scalar("rtt", dace.float64)
    sd1.add_symbol("_if_cond_58", dace.float64)
    sd1.add_symbol("_for_it_52", dace.int64)
    sd1.add_edge(src=ss1, dst=cb1, data=dace.InterstateEdge(assignments={
        "_if_cond_58": "ztp1[_for_it_52] <= rtt",
    }, ))

    for state, d1_access_str, zsolqa_access_str, zsolqa_access_str_rev in [
        (s1, "_for_it_52", "0,3,_for_it_52", "3,0,_for_it_52"), (s2, "_for_it_52", "0,2,_for_it_52", "2,0,_for_it_52")
    ]:
        zrainaut = state.add_access("zrainaut")
        zrainacc = state.add_access("zrainacc")
        zsolqa1 = state.add_access("zsolqa")
        zsolqa2 = state.add_access("zsolqa")
        zsolqa3 = state.add_access("zsolqa")
        zsolqa4 = state.add_access("zsolqa")
        zsolqa5 = state.add_access("zsolqa")
        for i, (tasklet_code, in1, instr1, in2, instr2, out, outstr) in enumerate([
            ("_out = _in1 + _in2", zrainaut, d1_access_str, zsolqa1, zsolqa_access_str, zsolqa2, zsolqa_access_str),
            ("_out = _in1 + _in2", zrainacc, d1_access_str, zsolqa2, zsolqa_access_str, zsolqa3, zsolqa_access_str),
            ("_out = (-_in1) + _in2", zrainaut, d1_access_str, zsolqa3, zsolqa_access_str_rev, zsolqa4,
             zsolqa_access_str_rev),
            ("_out = (-_in1) + _in2", zrainacc, d1_access_str, zsolqa4, zsolqa_access_str_rev, zsolqa5,
             zsolqa_access_str_rev),
        ]):
            t1 = state.add_tasklet("t1", {"_in1", "_in2"}, {"_out"}, tasklet_code)
            state.add_edge(in1, None, t1, "_in1", dace.memlet.Memlet(f"{in1.data}[{instr1}]"))
            state.add_edge(in2, None, t1, "_in2", dace.memlet.Memlet(f"{in2.data}[{instr2}]"))
            state.add_edge(t1, "_out", out, None, dace.memlet.Memlet(f"{out.data}[{outstr}]"))

    sd1.validate()

    sd2 = dace.SDFG("disjoin_chain_sdfg")
    p_s1 = sd2.add_state("p_s1", is_start_block=True)

    map_entry, map_exit = p_s1.add_map(name="map1", ndrange={"_for_it_52": dace.subsets.Range([(0, C - 1, 1)])})
    nsdfg = p_s1.add_nested_sdfg(sdfg=sd1,
                                 inputs={"zsolqa", "ztp1", "zrainaut", "zrainacc", "rtt"},
                                 outputs={"zsolqa"},
                                 symbol_mapping={"_for_it_52": "_for_it_52"})
    for arr_name, shape in [("zsolqa", (5, 5, C)), ("zrainaut", (C, )), ("zrainacc", (C, )), ("ztp1", (C, ))]:
        sd2.add_array(arr_name, shape, dace.float64)
    sd2.add_scalar("rtt", dace.float64)
    for input_name in {"zsolqa", "ztp1", "zrainaut", "zrainacc", "rtt"}:
        a = p_s1.add_access(input_name)
        p_s1.add_edge(a, None, map_entry, f"IN_{input_name}",
                      dace.memlet.Memlet.from_array(input_name, sd2.arrays[input_name]))
        p_s1.add_edge(map_entry, f"OUT_{input_name}", nsdfg, input_name,
                      dace.memlet.Memlet.from_array(input_name, sd2.arrays[input_name]))
        map_entry.add_in_connector(f"IN_{input_name}")
        map_entry.add_out_connector(f"OUT_{input_name}")
    for output_name in {"zsolqa"}:
        a = p_s1.add_access(output_name)
        p_s1.add_edge(map_exit, f"OUT_{output_name}", a, None,
                      dace.memlet.Memlet.from_array(output_name, sd2.arrays[output_name]))
        p_s1.add_edge(nsdfg, output_name, map_exit, f"IN_{output_name}",
                      dace.memlet.Memlet.from_array(output_name, sd2.arrays[output_name]))
        map_exit.add_in_connector(f"IN_{output_name}")
        map_exit.add_out_connector(f"OUT_{output_name}")

    nsdfg.sdfg.parent_nsdfg_node = nsdfg

    sd1.validate()
    sd2.validate()
    return sd2, p_s1


@pytest.mark.parametrize("trivial_if_demote_symbols", [(True, True), (True, False), (False, True), (False, False)])
def test_disjoint_chain_split_branch_only(trivial_if_demote_symbols: Tuple[bool, bool]):
    trivial_if, demote_symbols = trivial_if_demote_symbols
    sdfg, nsdfg_parent_state = _get_disjoint_chain_sdfg(trivial_if)
    zsolqa = numpy.random.choice([0.001, 5.0], size=(C, 5, 5))
    zrainacc = numpy.random.choice([0.001, 5.0], size=(C, ))
    zrainaut = numpy.random.choice([0.001, 5.0], size=(C, ))
    ztp1 = numpy.random.choice([3.5, 5.0], size=(C, ))
    rtt = numpy.random.choice([4.0], size=(1, ))

    sdfg.name = f"{sdfg.name}_{str(trivial_if).lower()}_{str(demote_symbols).lower()}"
    copy_sdfg = copy.deepcopy(sdfg)
    copy_sdfg.name = f"{copy_sdfg.name}_{str(trivial_if).lower()}_{str(demote_symbols).lower()}_vectorized"

    arrays = {"zsolqa": zsolqa, "zrainacc": zrainacc, "zrainaut": zrainaut, "ztp1": ztp1, "rtt": rtt[0]}

    sdfg.validate()
    sdfg.save(f"disjoint_chain_{str(trivial_if).lower()}_{str(demote_symbols).lower()}.sdfg")
    out_no_fuse = {k: v.copy() for k, v in arrays.items()}
    sdfg(**out_no_fuse)

    # Run SDFG version (with transformation)
    if trivial_if:
        from dace.transformation.passes.lift_trivial_if import LiftTrivialIf
        LiftTrivialIf().apply_pass(copy_sdfg, {})
    else:
        xform = branch_elimination.BranchElimination()
        cblocks = {n for n, g in copy_sdfg.all_nodes_recursive() if isinstance(n, ConditionalBlock)}
        assert len(cblocks) == 1
        cblock = cblocks.pop()

        xform.conditional = cblock
        xform.parent_nsdfg_state = nsdfg_parent_state
        xform.sequentialize_if_else_branch_if_disjoint_subsets(cblock.parent_graph)

    out_fused = {k: v.copy() for k, v in arrays.items()}

    vectorizer = VectorizeCPU(vector_width=8, fail_on_unvectorizable=True)
    vectorizer.try_to_demote_symbols_in_nsdfgs = demote_symbols
    vectorizer.apply_pass(copy_sdfg, {})
    copy_sdfg.save("disjoint_chain_vectorized.sdfg")

    copy_sdfg(**out_fused)

    for name in arrays.keys():
        numpy.testing.assert_allclose(out_no_fuse[name], out_fused[name], atol=1e-12)


@dace.program
def cloudsc_snippet_two(
    A: dace.float64[2, N, N],
    B: dace.float64[N, N],
    c: dace.float64,
    D: dace.float64[N, N],
    E: dace.float64[N, N],
):
    for i in dace.map[0:N]:
        for j in dace.map[0:N]:
            B[i, j] = A[1, i, j] + A[0, i, j]
            _if_cond_5 = B[i, j] > c
            if _if_cond_5:
                D[i, j] = B[i, j] / A[0, i, j]
                E[i, j] = 1.0 - D[i, j]
            else:
                D[i, j] = 0.0
                E[i, j] = 0.0


def test_snippet_from_cloudsc_two():
    _S = 64
    A = numpy.random.random((2, _S, _S))
    B = numpy.random.random((_S, _S))
    c = 0.1
    D = numpy.random.random((_S, _S))
    E = numpy.random.random((_S, _S))

    run_vectorization_test(dace_func=cloudsc_snippet_two,
                           arrays={
                               'A': A,
                               'B': B,
                               'D': D,
                               'E': E,
                           },
                           params={
                               'c': c,
                               'N': _S
                           },
                           vector_width=8,
                           save_sdfgs=True,
                           sdfg_name="cloudsc_snippet_two")


def has_no_inner_maps(state: dace.SDFGState, map_entry: dace.nodes.MapEntry):
    for inode in state.all_nodes_between(map_entry, state.exit_node(map_entry)):
        if isinstance(inode, dace.nodes.MapEntry):
            return False
    return True


def test_snippet_from_cloudsc_two_fuse_overlapping_loads():
    _S = 64
    A = numpy.random.random((2, _S, _S))
    B = numpy.random.random((_S, _S))
    c = 0.1
    D = numpy.random.random((_S, _S))
    E = numpy.random.random((_S, _S))

    vectorized_sdfg = run_vectorization_test(dace_func=cloudsc_snippet_two,
                                             arrays={
                                                 'A': A,
                                                 'B': B,
                                                 'D': D,
                                                 'E': E,
                                             },
                                             params={
                                                 'c': c,
                                                 'N': _S
                                             },
                                             vector_width=8,
                                             save_sdfgs=True,
                                             fuse_overlapping_loads=True,
                                             sdfg_name="cloudsc_snippet_two_fuse_overlapping_loads")

    # Should have 1 access node between two maps
    nsdfgs = {(n, g) for n, g in vectorized_sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.NestedSDFG)}
    for nsdfg, state in nsdfgs:
        src_access_nodes = {ie.src for ie in state.in_edges(nsdfg) if isinstance(ie.src, dace.nodes.AccessNode)}

        src_src_access_nodes = set()
        for src_acc_node in src_access_nodes:
            src_src_access_nodes = src_src_access_nodes.union(
                {ie.src
                 for ie in state.in_edges(src_acc_node) if isinstance(ie.src, dace.nodes.AccessNode)})

        vectorized_sdfg.save("vec.sdfg")
        assert len(
            src_src_access_nodes
        ) == 1, f"Excepted one access node got {len(src_src_access_nodes)}, ({src_src_access_nodes}) (from: ({src_access_nodes}))"


def test_snippet_from_cloudsc_one():
    klev = 64
    kfdia = 32

    # reverse dimensions to match Fortran layout
    za = numpy.random.random((kfdia, klev))
    zliqfrac = numpy.random.random((kfdia, klev))
    zicefrac = numpy.random.random((kfdia, klev))
    zqx = numpy.random.random((5, kfdia + 1, klev + 1))
    zli = numpy.random.random((kfdia, klev))

    rlmin = 0.1
    z1 = 1

    run_vectorization_test(
        dace_func=cloudsc_snippet_one,
        arrays={
            'za': za,
            'zliqfrac': zliqfrac,
            'zicefrac': zicefrac,
            'zqx': zqx,
            'zli': zli,
        },
        params={
            'rlmin': rlmin,
            'z1': z1,
            'kfdia': kfdia,
            'klev': klev,
        },
        vector_width=8,
        save_sdfgs=True,
        sdfg_name="cloudsc_snippet_one",
        cleanup=True,
    )


def _get_disjoint_chain_sdfg_two() -> dace.SDFG:
    sd1 = dace.SDFG("disjoint_chain_two")
    cb1 = ConditionalBlock("cond_if_cond_58", sdfg=sd1, parent=sd1)
    ss1 = sd1.add_state(label="pre", is_start_block=True)
    sd1.add_node(cb1, is_start_block=False)
    sd1.add_symbol("N", dace.int64)

    cfg1 = ControlFlowRegion(label="cond_58_true", sdfg=sd1, parent=cb1)
    s1 = cfg1.add_state("main_1", is_start_block=True)
    cfg2 = ControlFlowRegion(label="cond_58_false", sdfg=sd1, parent=cb1)
    s2 = cfg2.add_state("main_2", is_start_block=True)

    cb1.add_branch(
        condition=CodeBlock("_if_cond_58 == 1"),
        branch=cfg1,
    )
    cb1.add_branch(
        condition=None,
        branch=cfg2,
    )
    for arr_name, shape in [
        ("zsolqa", (5, 5, N)),
        ("zrainaut", (N, )),
        ("zrainacc", (N, )),
        ("ztp1", (N, )),
    ]:
        sd1.add_array(arr_name, shape, dace.float64)
    sd1.add_scalar("rtt", dace.float64)
    sd1.add_symbol("_if_cond_58", dace.float64)
    sd1.add_symbol("_for_it_52", dace.int64)
    sd1.add_edge(src=ss1, dst=cb1, data=InterstateEdge(assignments={
        "_if_cond_58": "ztp1[_for_it_52] <= rtt",
    }, ))

    for state, d1_access_str, zsolqa_access_str, zsolqa_access_str_rev in [
        (s1, "_for_it_52", "3,0,_for_it_52", "0,3,_for_it_52"), (s2, "_for_it_52", "2,0,_for_it_52", "0,2,_for_it_52")
    ]:
        zrainaut = state.add_access("zrainaut")
        zrainacc = state.add_access("zrainacc")
        zsolqa1 = state.add_access("zsolqa")
        zsolqa2 = state.add_access("zsolqa")
        zsolqa3 = state.add_access("zsolqa")
        zsolqa4 = state.add_access("zsolqa")
        zsolqa5 = state.add_access("zsolqa")
        for i, (tasklet_code, in1, instr1, in2, instr2, out, outstr) in enumerate([
            ("_out = _in1 + _in2", zrainaut, d1_access_str, zsolqa1, zsolqa_access_str, zsolqa2, zsolqa_access_str),
            ("_out = _in1 + _in2", zrainacc, d1_access_str, zsolqa2, zsolqa_access_str, zsolqa3, zsolqa_access_str),
            ("_out = (-_in1) + _in2", zrainaut, d1_access_str, zsolqa3, zsolqa_access_str_rev, zsolqa4,
             zsolqa_access_str_rev),
            ("_out = (-_in1) + _in2", zrainacc, d1_access_str, zsolqa4, zsolqa_access_str_rev, zsolqa5,
             zsolqa_access_str_rev),
        ]):
            t1 = state.add_tasklet("t1", {"_in1", "_in2"}, {"_out"}, tasklet_code)
            state.add_edge(in1, None, t1, "_in1", dace.memlet.Memlet(f"{in1.data}[{instr1}]"))
            state.add_edge(in2, None, t1, "_in2", dace.memlet.Memlet(f"{in2.data}[{instr2}]"))
            state.add_edge(t1, "_out", out, None, dace.memlet.Memlet(f"{out.data}[{outstr}]"))

    sd1_s2 = sd1.add_state_after(cb1, label="extra")

    z1 = sd1_s2.add_access("zrainacc")
    z2 = sd1_s2.add_access("zrainacc")
    t1 = sd1_s2.add_tasklet("increment", {"_in"}, {"_out"}, "_out = _in + 1")
    sd1_s2.add_edge(z1, None, t1, "_in", dace.memlet.Memlet("zrainacc[_for_it_52]"))
    sd1_s2.add_edge(t1, "_out", z2, None, dace.memlet.Memlet("zrainacc[_for_it_52]"))
    sd1.validate()

    sd2 = dace.SDFG("sd2")
    sd2.add_symbol("N", dace.int64)
    p_s1 = sd2.add_state("p_s1", is_start_block=True)

    map_entry, map_exit = p_s1.add_map(name="map1", ndrange={"_for_it_52": dace.subsets.Range([(0, N - 1, 1)])})
    nsdfg = p_s1.add_nested_sdfg(sdfg=sd1,
                                 inputs={"zsolqa", "ztp1", "zrainaut", "zrainacc", "rtt"},
                                 outputs={"zsolqa", "zrainacc"},
                                 symbol_mapping={"_for_it_52": "_for_it_52"})
    for arr_name, shape in [("zsolqa", (5, 5, N)), ("zrainaut", (N, )), ("zrainacc", (N, )), ("ztp1", (N, ))]:
        sd2.add_array(arr_name, shape, dace.float64)
    sd2.add_scalar("rtt", dace.float64)

    for input_name in {"zsolqa", "ztp1", "zrainaut", "zrainacc", "rtt"}:
        a = p_s1.add_access(input_name)
        p_s1.add_edge(a, None, map_entry, f"IN_{input_name}",
                      dace.memlet.Memlet.from_array(input_name, sd2.arrays[input_name]))
        p_s1.add_edge(map_entry, f"OUT_{input_name}", nsdfg, input_name,
                      dace.memlet.Memlet.from_array(input_name, sd2.arrays[input_name]))
        map_entry.add_in_connector(f"IN_{input_name}")
        map_entry.add_out_connector(f"OUT_{input_name}")
    for output_name in {"zsolqa", "zrainacc"}:
        a = p_s1.add_access(output_name)
        p_s1.add_edge(map_exit, f"OUT_{output_name}", a, None,
                      dace.memlet.Memlet.from_array(output_name, sd2.arrays[output_name]))
        p_s1.add_edge(nsdfg, output_name, map_exit, f"IN_{output_name}",
                      dace.memlet.Memlet.from_array(output_name, sd2.arrays[output_name]))
        map_exit.add_in_connector(f"IN_{output_name}")
        map_exit.add_out_connector(f"OUT_{output_name}")

    nsdfg.sdfg.parent_nsdfg_node = nsdfg

    sd1.validate()
    sd2.validate()
    return sd2, p_s1


def test_disjoint_chain_with_overlapping_region_fusion():
    sdfg, nsdfg_parent_state = _get_disjoint_chain_sdfg_two()
    sdfg.name = f"disjoint_chain_split_two_rtt_val_4_2_with_overlapping_region_fusion"
    _N = 64
    zsolqa = numpy.random.choice([0.001, 5.0], size=(5, 5, _N))
    zrainacc = numpy.random.choice([0.001, 5.0], size=(_N, ))
    zrainaut = numpy.random.choice([0.001, 5.0], size=(_N, ))
    ztp1 = numpy.random.choice([3.5, 5.0], size=(_N, ))
    rtt = numpy.array([4.2], numpy.float64)
    _N = numpy.array([64], numpy.int64)
    sdfg.validate()
    sdfg.save(f"{sdfg.name}.sdfgz", compress=True)

    copy_sdfg = copy.deepcopy(sdfg)
    copy_sdfg.name = sdfg.name + "_vectorized"
    arrays = {"zsolqa": zsolqa, "zrainacc": zrainacc, "zrainaut": zrainaut, "ztp1": ztp1, "rtt": rtt[0], "N": _N[0]}

    sdfg.validate()
    out_no_fuse = {k: v.copy() for k, v in arrays.items()}
    sdfg(**out_no_fuse)
    # Run SDFG version (with transformation)
    VectorizeCPU(vector_width=8, insert_copies=True, fuse_overlapping_loads=True,
                 fail_on_unvectorizable=True).apply_pass(copy_sdfg, {})

    out_fused = {k: v.copy() for k, v in arrays.items()}
    copy_sdfg.validate()
    copy_sdfg.save(f"{copy_sdfg.name}.sdfgz", compress=True)

    # There is should be no `_union` access nodes

    access_nodes_of_unions = set()
    for state in copy_sdfg.all_states():
        for node in state.nodes():
            if isinstance(node, dace.nodes.AccessNode):
                if node.data.endswith("_union"):
                    access_nodes_of_unions.add(node)
    assert len(access_nodes_of_unions) == 0

    copy_sdfg(**out_fused)

    for name in arrays.keys():
        numpy.testing.assert_allclose(out_no_fuse[name], out_fused[name], atol=1e-12)


def test_disjoint_chain():
    sdfg, nsdfg_parent_state = _get_disjoint_chain_sdfg_two()
    sdfg.name = f"disjoint_chain"
    _N = 64
    zsolqa = numpy.random.choice([0.001, 5.0], size=(5, 5, _N))
    zrainacc = numpy.random.choice([0.001, 5.0], size=(_N, ))
    zrainaut = numpy.random.choice([0.001, 5.0], size=(_N, ))
    ztp1 = numpy.random.choice([3.5, 5.0], size=(_N, ))
    rtt = numpy.array([4.2], numpy.float64)
    _N = numpy.array([64], numpy.int64)
    sdfg.validate()
    sdfg.save(f"{sdfg.name}.sdfgz", compress=True)

    copy_sdfg = copy.deepcopy(sdfg)
    copy_sdfg.name = sdfg.name + "_vectorized"
    arrays = {"zsolqa": zsolqa, "zrainacc": zrainacc, "zrainaut": zrainaut, "ztp1": ztp1, "rtt": rtt[0], "N": _N[0]}

    sdfg.validate()
    out_no_fuse = {k: v.copy() for k, v in arrays.items()}
    sdfg(**out_no_fuse)
    # Run SDFG version (with transformation)
    VectorizeCPU(vector_width=8, insert_copies=True, fuse_overlapping_loads=False,
                 fail_on_unvectorizable=True).apply_pass(copy_sdfg, {})

    out_fused = {k: v.copy() for k, v in arrays.items()}
    copy_sdfg.validate()
    copy_sdfg.save(f"{copy_sdfg.name}.sdfgz", compress=True)

    for state in copy_sdfg.all_states():
        for node in state.nodes():
            if isinstance(node, dace.nodes.AccessNode):
                if "zrainacc_vec" in node.data or "zrainaut_vec" in node.data:
                    for ie in state.in_edges(node):
                        assert ie.data.subset != dace.subsets.Range([(0, N - 1, 1)])

    copy_sdfg(**out_fused)

    for name in sorted(arrays.keys()):
        print(f"Compare {name}")
        numpy.testing.assert_allclose(out_no_fuse[name], out_fused[name], atol=1e-12)
        print(f"{name} OK")


def _get_cloudsc_snippet_three(add_scalar: bool, map_range_dependent_subset: bool = False):
    klon = dace.symbolic.symbol("klon")
    klev = dace.symbolic.symbol("klev")
    # Add all arrays to the SDFGs
    in_arrays = {"tendency_tmp_q", "pa", "pq", "tendency_tmp_t", "tendency_tmp_a", "pt"}
    in_scalars = {"kfdia", "kidia", "ptsphy"}
    if add_scalar:
        in_scalars = in_scalars.union({"ralvdcp"})
    out_arrays = {"zqx0", "zqx", "ztp1", "zaorig", "za"}
    arr_shapes = {
        "tendency_tmp_q": ((klon, klev), (1, klon), dace.float64),
        "pa": ((klon, klev), (1, klon), dace.float64),
        "pq": ((klon, klev), (1, klon), dace.float64),
        "tendency_tmp_t": ((klon, klev), (1, klon), dace.float64),
        "tendency_tmp_a": ((klon, klev), (1, klon), dace.float64),
        "pt": ((klon, klev), (1, klon), dace.float64),
        "zqx0": ((klon, klev, 5), (1, klon, klon * klev), dace.float64),
        "zqx": ((klon, klev, 5), (1, klon, klon * klev), dace.float64),
        "ztp1": ((klon, klev), (1, klon), dace.float64),
        "zaorig": ((klon, klev), (1, klon), dace.float64),
        "za": ((klon, klev), (1, klon), dace.float64),
    }
    scalar_dtypes = {
        "kfdia": dace.int64,
        "kidia": dace.int64,
        "ptsphy": dace.float64,
    }
    if add_scalar:
        scalar_dtypes["ralvdcp"] = dace.float64
    outer_sdfg = dace.SDFG("outer")
    inner_sdfg = dace.SDFG("inner")
    inner_state = inner_sdfg.add_state("inner_compute_state", is_start_block=True)
    outer_state = outer_sdfg.add_state("outer_compute_state", is_start_block=True)

    symbols = {"i", "j", "klev", "klon"}
    for sym in symbols:
        inner_sdfg.add_symbol(sym, dace.int64)
        outer_sdfg.add_symbol(sym, dace.int64)

    for sdfg in [inner_sdfg, outer_sdfg]:
        for arr_name in in_arrays.union(out_arrays):
            shape, strides, dtype = arr_shapes[arr_name]
            sdfg.add_array(arr_name, shape, dtype, strides=strides, transient=False)
        for scalar_name in in_scalars:
            if sdfg == inner_sdfg and scalar_name in {"kfdia", "kidia"}:
                continue
            sdfg.add_scalar(scalar_name, scalar_dtypes[scalar_name], transient=False)

    for transient_scl in {"t0_s1", "t0_s2", "t0_s3", "t0_s4", "t0_s5"}:
        inner_sdfg.add_scalar(transient_scl, dace.float64, dace.dtypes.StorageType.Register, True)

    # All tasklets for the inner SDFG
    if add_scalar:
        tasklets = {
            ("ralvdcp", "0", None, None, "_out = - _in1", "t0_s1", "0"),
            ("ptsphy", "0", "tendency_tmp_q", "i, j", "_out = _in2 * _in1", "t0_s2", "0"),
            ("ptsphy", "0", "tendency_tmp_q", "i, j", "_out = _in2 * _in1", "t0_s3", "0"),
            ("ptsphy", "0", "tendency_tmp_a", "i, j", "_out = _in2 * _in1", "t0_s4", "0"),
            ("ptsphy", "0", "tendency_tmp_a", "i, j", "_out = _in2 * _in1", "t0_s5", "0"),
            ("t0_s1", "0", "pt", "i, j", "_out = _in1 + _in2", "ztp1", "i, j"),
            ("t0_s2", "0", "pq", "i, j", "_out = _in2 + _in1", "zqx", "i, j, 4"),
            ("t0_s3", "0", "pq", "i, j", "_out = _in2 + _in1", "zqx0", "i, j, 4"),
            ("t0_s4", "0", "pa", "i, j", "_out = _in2 + _in1", "za", "i, j"),
            ("t0_s5", "0", "pa", "i, j", "_out = _in2 + _in1", "zaorig", "i, j"),
        }
    else:
        tasklets = {
            ("ptsphy", "0", "tendency_tmp_t", "i, j", "_out = _in2 * _in1", "t0_s1", "0"),
            ("ptsphy", "0", "tendency_tmp_q", "i, j", "_out = _in2 * _in1", "t0_s2", "0"),
            ("ptsphy", "0", "tendency_tmp_q", "i, j", "_out = _in2 * _in1", "t0_s3", "0"),
            ("ptsphy", "0", "tendency_tmp_a", "i, j", "_out = _in2 * _in1", "t0_s4", "0"),
            ("ptsphy", "0", "tendency_tmp_a", "i, j", "_out = _in2 * _in1", "t0_s5", "0"),
            ("t0_s1", "0", "pt", "i, j", "_out = _in1 + _in2", "ztp1", "i, j"),
            ("t0_s2", "0", "pq", "i, j", "_out = _in2 + _in1", "zqx", "i, j, 4"),
            ("t0_s3", "0", "pq", "i, j", "_out = _in2 + _in1", "zqx0", "i, j, 4"),
            ("t0_s4", "0", "pa", "i, j", "_out = _in2 + _in1", "za", "i, j"),
            ("t0_s5", "0", "pa", "i, j", "_out = _in2 + _in1", "zaorig", "i, j"),
        }
    access_nodes = dict()
    for in1_arr, in1_subset, in2_arr, in2_subset, tasklet_code, out_arr, out_subset in tasklets:
        in1_an = inner_state.add_access(in1_arr) if in1_arr not in access_nodes else access_nodes[in1_arr]
        if in2_arr is not None:
            in2_an = inner_state.add_access(in2_arr) if in2_arr not in access_nodes else access_nodes[in2_arr]
        out_an = inner_state.add_access(out_arr) if out_arr not in access_nodes else access_nodes[out_arr]
        access_nodes[in1_arr] = in1_an
        if in2_arr is not None:
            access_nodes[in2_arr] = in2_an
        access_nodes[out_arr] = out_an

        t = inner_state.add_tasklet("t_" + out_arr, {"_in1", "_in2"} if in2_arr is not None else {"_in1"}, {"_out"},
                                    tasklet_code)
        access_str1 = f"{in1_arr}[{in1_subset}]" if in1_subset != "0" else in1_arr
        if in2_arr is not None:
            access_str2 = f"{in2_arr}[{in2_subset}]" if in2_subset != "0" else in2_arr
        inner_state.add_edge(in1_an, None, t, "_in1", dace.memlet.Memlet(access_str1))
        if in2_arr is not None:
            inner_state.add_edge(in2_an, None, t, "_in2", dace.memlet.Memlet(access_str2))

        access_str3 = f"{out_arr}[{out_subset}]" if out_subset != "0" else out_arr
        inner_state.add_edge(t, "_out", out_an, None, dace.memlet.Memlet(access_str3))

    inner_symbol_mapping = {sym: sym for sym in symbols}
    in_args = in_arrays.union({"ptsphy"})
    if add_scalar:
        in_args.add("ralvdcp")
    nsdfg = outer_state.add_nested_sdfg(inner_sdfg, in_args, out_arrays, inner_symbol_mapping)

    m1_entry, m1_exit = outer_state.add_map(name="m1", ndrange={"j": "0:klev:1"})
    m2_entry, m2_exit = outer_state.add_map(name="m2", ndrange={"i": "kidia-1:kfdia:1"})

    inner_sdfg.validate()

    # Access nodes to map entry
    for arr in in_arrays.union(in_scalars):
        outer_state.add_edge(outer_state.add_access(arr), None, m1_entry, f"IN_{arr}",
                             dace.memlet.Memlet.from_array(arr, outer_state.sdfg.arrays[arr]))
        m1_entry.add_in_connector(f"IN_{arr}")
    # Access nodes to map entry1 to map entry 2 and nsdfg
    for arr in in_arrays.union(in_scalars):
        if map_range_dependent_subset:
            if arr in in_arrays:
                if arr in {"zqx", "zqx0"}:
                    mem = dace.memlet.Memlet(f"{arr}[0:i+1, 0:j+1, 0:5]")
                else:
                    mem = dace.memlet.Memlet(f"{arr}[0:i+1, 0:j+1]")
            else:
                mem = dace.memlet.Memlet.from_array(arr, outer_state.sdfg.arrays[arr])
        else:
            mem = dace.memlet.Memlet.from_array(arr, outer_state.sdfg.arrays[arr])

        outer_state.add_edge(m1_entry, f"OUT_{arr}", m2_entry, f"IN_{arr}" if arr not in {"kidia", "kfdia"} else arr,
                             dace.memlet.Memlet.from_array(arr, outer_state.sdfg.arrays[arr]))
        m1_entry.add_out_connector(f"OUT_{arr}", force=True)
        m2_entry.add_in_connector(f"IN_{arr}" if arr not in {"kidia", "kfdia"} else arr, force=True)
        if arr not in {"kidia", "kfdia"}:
            outer_state.add_edge(m2_entry, f"OUT_{arr}", nsdfg, arr, copy.deepcopy(mem))
            m2_entry.add_out_connector(f"OUT_{arr}", force=True)
    # Same for exit nodes
    for arr in out_arrays:
        if map_range_dependent_subset:
            if arr in in_arrays:
                if arr in {"zqx", "zqx0"}:
                    mem = dace.memlet.Memlet(f"{arr}[0:i+1, 0:j+1, 0:5]")
                else:
                    mem = dace.memlet.Memlet(f"{arr}[0:i+1, 0:j+1]")
            else:
                mem = dace.memlet.Memlet.from_array(arr, outer_state.sdfg.arrays[arr])
        else:
            mem = dace.memlet.Memlet.from_array(arr, outer_state.sdfg.arrays[arr])
        outer_state.add_edge(nsdfg, arr, m2_exit, f"IN_{arr}", copy.deepcopy(mem))
        m2_exit.add_in_connector(f"IN_{arr}", force=True)
        outer_state.add_edge(m2_exit, f"OUT_{arr}", m1_exit, f"IN_{arr}",
                             dace.memlet.Memlet.from_array(arr, outer_state.sdfg.arrays[arr]))
        m1_exit.add_in_connector(f"IN_{arr}", force=True)
        m2_exit.add_out_connector(f"OUT_{arr}", force=True)

    for arr in out_arrays:
        outer_state.add_edge(m1_exit, f"OUT_{arr}", outer_state.add_access(arr), None,
                             dace.memlet.Memlet.from_array(arr, outer_state.sdfg.arrays[arr]))
        m1_exit.add_out_connector(f"OUT_{arr}", force=True)
    sdfg.validate()
    return sdfg


def _get_cloudsc_snippet_four():
    klon = dace.symbolic.symbol("klon")
    klev = dace.symbolic.symbol("klev")
    kidia = dace.symbolic.symbol("kidia")
    kfdia = dace.symbolic.symbol("kfdia")
    # Add all arrays to the SDFGs
    in_arrays = {"zsolqb", "zfallsink", "zqlhs"}
    out_arrays = {"zqlhs"}
    arr_shapes = {
        "zqlhs": ((klon, 5, 5), (1, klon, 5 * klon), dace.float64),
        "zfallsink": ((
            klon,
            5,
        ), (
            1,
            klon,
        ), dace.float64),
        "zsolqb": ((klon, 5, 5), (1, klon, 5 * klon), dace.float64),
    }
    outer_sdfg = dace.SDFG("outer")
    inner_sdfg = dace.SDFG("inner")
    inner_state = inner_sdfg.add_state("inner_compute_state", is_start_block=True)
    outer_state = outer_sdfg.add_state("outer_compute_state", is_start_block=True)

    symbols = {"_for_it_93", "_for_it_91", "_for_it_92", "for_i", "klev", "klon", "kfdia", "kidia"}
    for sym in symbols:
        inner_sdfg.add_symbol(sym, dace.int64)
        outer_sdfg.add_symbol(sym, dace.int64)

    for sdfg in [inner_sdfg, outer_sdfg]:
        for arr_name in in_arrays.union(out_arrays):
            shape, strides, dtype = arr_shapes[arr_name]
            sdfg.add_array(arr_name, shape, dtype, strides=strides, transient=False)

    for transient_scl in {"t0_s1", "t0_s2", "t0_s3", "t0_s4", "t0_s5"}:
        inner_sdfg.add_scalar(transient_scl, dace.float64, dace.dtypes.StorageType.Register, True)

    tasklets = {
        ("zfallsink", "_for_it_93, _for_it_91", None, None, "_out = _in1 + 1.0", "zqlhs@1",
         "_for_it_93, _for_it_92, _for_it_91"),
        ("zqlhs@1", "_for_it_93, _for_it_92, _for_it_91", "zsolqb", "_for_it_93, 0, _for_it_91", "_out = _in1 + _in2",
         "zqlhs@2", "_for_it_93, _for_it_92, _for_it_91"),
        ("zqlhs@2", "_for_it_93, _for_it_92, _for_it_91", "zsolqb", "_for_it_93, 1, _for_it_91", "_out = _in1 + _in2",
         "zqlhs@3", "_for_it_93, _for_it_92, _for_it_91"),
        ("zqlhs@3", "_for_it_93, _for_it_92, _for_it_91", "zsolqb", "_for_it_93, 2, _for_it_91", "_out = _in1 + _in2",
         "zqlhs@4", "_for_it_93, _for_it_92, _for_it_91"),
        ("zqlhs@4", "_for_it_93, _for_it_92, _for_it_91", "zsolqb", "_for_it_93, 3, _for_it_91", "_out = _in1 + _in2",
         "zqlhs@5", "_for_it_93, _for_it_92, _for_it_91"),
        ("zqlhs@5", "_for_it_93, _for_it_92, _for_it_91", "zsolqb", "_for_it_93, 4, _for_it_91", "_out = _in1 + _in2",
         "zqlhs@6", "_for_it_93, _for_it_92, _for_it_91"),
    }
    zqlhs_ans = list()
    for i in range(6):
        zqlhs_an = inner_state.add_access("zqlhs")
        zqlhs_ans.append(zqlhs_an)

    access_nodes = dict()
    for in1_arr, in1_subset, in2_arr, in2_subset, tasklet_code, out_arr, out_subset in tasklets:
        if in1_arr.startswith("zqlhs@"):
            in1_an = zqlhs_ans[int(in1_arr.split("@")[1]) - 1]
            in1_arr = "zqlhs"
        else:
            in1_an = inner_state.add_access(in1_arr) if in1_arr not in access_nodes else access_nodes[in1_arr]

        if in2_arr is not None:
            if in2_arr.startswith("zqlhs@"):
                in2_an = zqlhs_ans[int(in2_arr.split("@")[1]) - 1]
                in2_arr = "zqlhs"
            else:
                in2_an = inner_state.add_access(in2_arr) if in2_arr not in access_nodes else access_nodes[in2_arr]

        if out_arr.startswith("zqlhs@"):
            out_an = zqlhs_ans[int(out_arr.split("@")[1]) - 1]
            out_arr = "zqlhs"
        else:
            out_an = inner_state.add_access(out_arr) if out_arr not in access_nodes else access_nodes[out_arr]

        access_nodes[in1_arr] = in1_an
        if in2_arr is not None:
            access_nodes[in2_arr] = in2_an
        access_nodes[out_arr] = out_an

        t = inner_state.add_tasklet("t_" + out_arr, {"_in1", "_in2"} if in2_arr is not None else {"_in1"}, {"_out"},
                                    tasklet_code)
        access_str1 = f"{in1_arr}[{in1_subset}]" if in1_subset != "0" else in1_arr
        if in2_arr is not None:
            access_str2 = f"{in2_arr}[{in2_subset}]" if in2_subset != "0" else in2_arr
        inner_state.add_edge(in1_an, None, t, "_in1", dace.memlet.Memlet(access_str1))
        if in2_arr is not None:
            inner_state.add_edge(in2_an, None, t, "_in2", dace.memlet.Memlet(access_str2))

        access_str3 = f"{out_arr}[{out_subset}]" if out_subset != "0" else out_arr
        inner_state.add_edge(t, "_out", out_an, None, dace.memlet.Memlet(access_str3))

    inner_symbol_mapping = {sym: sym for sym in symbols}
    in_args = in_arrays
    nsdfg = outer_state.add_nested_sdfg(inner_sdfg, in_args, out_arrays, inner_symbol_mapping)

    m1_entry, m1_exit = outer_state.add_map(name="m1", ndrange={"_for_it_93": "kidia-1:kfdia:1"})

    inner_sdfg.validate()

    # Access nodes to map entry
    for arr in in_arrays:
        outer_state.add_edge(outer_state.add_access(arr), None, m1_entry, f"IN_{arr}",
                             dace.memlet.Memlet.from_array(arr, outer_state.sdfg.arrays[arr]))
        m1_entry.add_in_connector(f"IN_{arr}")
        mem = dace.memlet.Memlet.from_array(arr, outer_state.sdfg.arrays[arr])
        m1_entry.add_out_connector(f"OUT_{arr}", force=True)
        outer_state.add_edge(m1_entry, f"OUT_{arr}", nsdfg, arr, copy.deepcopy(mem))
    # Same for exit nodes
    for arr in out_arrays:
        mem = dace.memlet.Memlet.from_array(arr, outer_state.sdfg.arrays[arr])
        outer_state.add_edge(nsdfg, arr, m1_exit, f"IN_{arr}", copy.deepcopy(mem))
        outer_state.add_edge(m1_exit, f"OUT_{arr}", outer_state.add_access(arr), None,
                             dace.memlet.Memlet.from_array(arr, outer_state.sdfg.arrays[arr]))
        m1_exit.add_out_connector(f"OUT_{arr}", force=True)
        m1_exit.add_in_connector(f"IN_{arr}", force=True)
    sdfg.validate()
    return sdfg


def _get_map_inside_nested_map():
    klon = dace.symbolic.symbol("klon")
    # Add all arrays to the SDFGs
    in_arrays = set()
    in_scalars = {
        "kfdia",
        "kidia",
    }
    out_arrays = {"int_array", "int_array2"}
    arr_shapes = {
        "int_array": ((klon, 5, 5), (1, klon, klon * 5), dace.int64),
        "int_array2": ((klon, 5, 5), (1, klon, klon * 5), dace.int64),
    }
    scalar_dtypes = {
        "kfdia": dace.int64,
        "kidia": dace.int64,
    }
    outer_sdfg = dace.SDFG("outer")
    inner_sdfg = dace.SDFG("inner")
    inner_state = inner_sdfg.add_state("inner_compute_state", is_start_block=True)
    outer_state = outer_sdfg.add_state("outer_compute_state", is_start_block=True)

    symbols = {"i", "j", "k", "klev", "klon"}
    for sym in symbols:
        inner_sdfg.add_symbol(sym, dace.int64)
        outer_sdfg.add_symbol(sym, dace.int64)

    for sdfg in [inner_sdfg, outer_sdfg]:
        for arr_name in in_arrays.union(out_arrays):
            shape, strides, dtype = arr_shapes[arr_name]
            sdfg.add_array(arr_name, shape, dtype, strides=strides, transient=False)
        for scalar_name in in_scalars:
            sdfg.add_scalar(scalar_name, scalar_dtypes[scalar_name], transient=False)

    t, map_entry, map_exit = inner_state.add_mapped_tasklet(name="assign",
                                                            map_ranges={
                                                                "j": dace.subsets.Range([(0, 4, 1)]),
                                                                "k": dace.subsets.Range([(kfdia - 1, kidia - 1, 1)])
                                                            },
                                                            inputs=dict(),
                                                            code="_out = 0",
                                                            outputs={
                                                                "_out": dace.memlet.Memlet("int_array[k, j, i]"),
                                                            },
                                                            external_edges=True,
                                                            input_nodes=dict(),
                                                            output_nodes={
                                                                "int_array": inner_state.add_access("int_array"),
                                                            })
    for scl_name in {"kfdia", "kidia"}:
        an = inner_state.add_access(scl_name)
        inner_state.add_edge(an, None, map_entry, scl_name, dace.memlet.Memlet(scl_name))
        map_entry.add_in_connector(scl_name)

    inner_symbol_mapping = {sym: sym for sym in symbols}
    in_args = in_scalars
    nsdfg = outer_state.add_nested_sdfg(inner_sdfg, in_args, out_arrays, inner_symbol_mapping)

    m1_entry, m1_exit = outer_state.add_map(name="m1", ndrange={"i": "0:5:1"})

    inner_sdfg.validate()

    t2 = inner_state.add_tasklet("t2", set(), {"_out"}, "_out = 1")
    inner_state.add_edge(t2, "_out", inner_state.add_access("int_array2"), None,
                         dace.memlet.Memlet("int_array2[1,1,1]"))

    # Access nodes to map entry
    for arr in in_arrays.union(in_scalars):
        outer_state.add_edge(outer_state.add_access(arr), None, m1_entry, f"IN_{arr}",
                             dace.memlet.Memlet.from_array(arr, outer_state.sdfg.arrays[arr]))
        m1_entry.add_in_connector(f"IN_{arr}")
        outer_state.add_edge(m1_entry, f"OUT_{arr}", nsdfg, arr,
                             dace.memlet.Memlet.from_array(arr, outer_state.sdfg.arrays[arr]))
        m1_entry.add_out_connector(f"OUT_{arr}")
    # Same for exit nodes
    for arr in out_arrays:
        outer_state.add_edge(nsdfg, arr, m1_exit, f"IN_{arr}",
                             dace.memlet.Memlet.from_array(arr, outer_state.sdfg.arrays[arr]))
        m1_exit.add_in_connector(f"IN_{arr}", force=True)
        outer_state.add_edge(m1_exit, f"OUT_{arr}", outer_state.add_access(arr), None,
                             dace.memlet.Memlet.from_array(arr, outer_state.sdfg.arrays[arr]))
        m1_exit.add_out_connector(f"OUT_{arr}", force=True)
    sdfg.validate()
    return sdfg


def test_snippet_from_cloudsc_four():
    sdfg = _get_cloudsc_snippet_four()
    sdfg.name = f"cloudsc_snippet_four"
    sdfg.validate()

    # Symbolic values requested by the user
    klon = 64
    klev = 64
    kidia = 1
    kfdia = 32
    _for_it_92 = 0
    _for_it_91 = 0

    # Map of array shapes (from the SDFG snippet): only the shape tuples matter for creating arrays
    arr_shapes = {
        "zfallsink": (klon, klev, 5),
        "zqlhs": (klon, klev, 5),
        "zsolqb": (klon, klev, 5),
    }

    # Create Fortran-ordered NumPy arrays
    arrays = {name: numpy.random.random(shape).astype(numpy.float64, order='F') for name, shape in arr_shapes.items()}
    # Create scalars requested
    scalars = {
        "kfdia": numpy.int64(kfdia),
        "kidia": numpy.int64(kidia),
        "klev": numpy.int64(klev),
        "klon": numpy.int64(klon),
        "_for_it_92": numpy.int64(_for_it_92),
        "_for_it_91": numpy.int64(_for_it_91),
    }

    # Quick verification display: shape and contiguity / strides
    run_vectorization_test(dace_func=sdfg,
                           from_sdfg=True,
                           arrays=arrays,
                           params=scalars,
                           vector_width=8,
                           save_sdfgs=True,
                           sdfg_name=sdfg.name,
                           fuse_overlapping_loads=False,
                           insert_copies=True)


@pytest.mark.parametrize("opt_parameters", [(True, True), (True, False), (False, True), (False, False)])
def test_snippet_from_cloudsc_three(opt_parameters):
    fuse_overlapping_loads, insert_copies = opt_parameters

    sdfg = _get_cloudsc_snippet_three(add_scalar=False)
    sdfg.name = f"cloudsc_snippet_three_fuse_overlapping_loads_{fuse_overlapping_loads}_insert_copies_{insert_copies}"
    sdfg.validate()

    # Symbolic values requested by the user
    klon = 64
    klev = 64
    kidia = 1
    kfdia = 32

    # Map of array shapes (from the SDFG snippet): only the shape tuples matter for creating arrays
    arr_shapes = {
        "tendency_tmp_q": (klon, klev),
        "pa": (klon, klev),
        "pq": (klon, klev),
        "tendency_tmp_t": (klon, klev),
        "tendency_tmp_a": (klon, klev),
        "pt": (klon, klev),
        "zqx0": (klon, klev, 5),
        "zqx": (klon, klev, 5),
        "ztp1": (klon, klev),
        "zaorig": (klon, klev),
        "za": (klon, klev),
    }

    # Create Fortran-ordered NumPy arrays
    arrays = {name: numpy.random.random(shape).astype(numpy.float64, order='F') for name, shape in arr_shapes.items()}
    # Create scalars requested
    scalars = {
        "kfdia": numpy.int64(kfdia),
        "kidia": numpy.int64(kidia),
        "ptsphy": numpy.float64(0.0),
        "klev": numpy.int64(klev),
        "klon": numpy.int64(klon),
    }

    # Quick verification display: shape and contiguity / strides
    run_vectorization_test(dace_func=sdfg,
                           from_sdfg=True,
                           arrays=arrays,
                           params=scalars,
                           vector_width=8,
                           save_sdfgs=True,
                           sdfg_name=sdfg.name,
                           fuse_overlapping_loads=fuse_overlapping_loads,
                           insert_copies=insert_copies)


@pytest.mark.parametrize("opt_parameters", [(True, True), (True, False), (False, True), (False, False)])
def test_snippet_from_cloudsc_three_with_partial_subset(opt_parameters):
    fuse_overlapping_loads, insert_copies = opt_parameters

    sdfg = _get_cloudsc_snippet_three(add_scalar=False, map_range_dependent_subset=True)
    sdfg.name = f"cloudsc_snippet_three_with_partial_subset_fuse_overlapping_loads_{fuse_overlapping_loads}_insert_copies_{insert_copies}"
    sdfg.validate()

    # Symbolic values requested by the user
    klon = 64
    klev = 64
    kidia = 1
    kfdia = 32

    # Map of array shapes (from the SDFG snippet): only the shape tuples matter for creating arrays
    arr_shapes = {
        "tendency_tmp_q": (klon, klev),
        "pa": (klon, klev),
        "pq": (klon, klev),
        "tendency_tmp_t": (klon, klev),
        "tendency_tmp_a": (klon, klev),
        "pt": (klon, klev),
        "zqx0": (klon, klev, 5),
        "zqx": (klon, klev, 5),
        "ztp1": (klon, klev),
        "zaorig": (klon, klev),
        "za": (klon, klev),
    }

    # Create Fortran-ordered NumPy arrays
    arrays = {name: numpy.random.random(shape).astype(numpy.float64, order='F') for name, shape in arr_shapes.items()}
    # Create scalars requested
    scalars = {
        "kfdia": numpy.int64(kfdia),
        "kidia": numpy.int64(kidia),
        "ptsphy": numpy.float64(0.0),
        "klev": numpy.int64(klev),
        "klon": numpy.int64(klon),
    }

    # Quick verification display: shape and contiguity / strides
    run_vectorization_test(dace_func=sdfg,
                           from_sdfg=True,
                           arrays=arrays,
                           params=scalars,
                           vector_width=8,
                           save_sdfgs=True,
                           sdfg_name=sdfg.name,
                           fuse_overlapping_loads=fuse_overlapping_loads,
                           insert_copies=insert_copies,
                           no_inline=True)


@pytest.mark.parametrize("opt_parameters", [(True, True), (True, False), (False, True), (False, False)])
def test_snippet_from_cloudsc_three_with_partial_subset_without_inline(opt_parameters):
    fuse_overlapping_loads, insert_copies = opt_parameters

    sdfg = _get_cloudsc_snippet_three(add_scalar=False, map_range_dependent_subset=True)
    sdfg.name = f"cloudsc_snippet_three_with_partial_subset_without_inline_fuse_overlapping_loads_{fuse_overlapping_loads}_insert_copies_{insert_copies}"
    sdfg.validate()

    # Symbolic values requested by the user
    klon = 64
    klev = 64
    kidia = 1
    kfdia = 32

    # Map of array shapes (from the SDFG snippet): only the shape tuples matter for creating arrays
    arr_shapes = {
        "tendency_tmp_q": (klon, klev),
        "pa": (klon, klev),
        "pq": (klon, klev),
        "tendency_tmp_t": (klon, klev),
        "tendency_tmp_a": (klon, klev),
        "pt": (klon, klev),
        "zqx0": (klon, klev, 5),
        "zqx": (klon, klev, 5),
        "ztp1": (klon, klev),
        "zaorig": (klon, klev),
        "za": (klon, klev),
    }

    # Create Fortran-ordered NumPy arrays
    arrays = {name: numpy.random.random(shape).astype(numpy.float64, order='F') for name, shape in arr_shapes.items()}
    # Create scalars requested
    scalars = {
        "kfdia": numpy.int64(kfdia),
        "kidia": numpy.int64(kidia),
        "ptsphy": numpy.float64(0.0),
        "klev": numpy.int64(klev),
        "klon": numpy.int64(klon),
    }

    # Quick verification display: shape and contiguity / strides
    run_vectorization_test(dace_func=sdfg,
                           from_sdfg=True,
                           arrays=arrays,
                           params=scalars,
                           vector_width=8,
                           save_sdfgs=True,
                           sdfg_name=sdfg.name,
                           fuse_overlapping_loads=fuse_overlapping_loads,
                           insert_copies=insert_copies,
                           no_inline=True)


@pytest.mark.parametrize("opt_parameters", [(True, True), (True, False), (False, True), (False, False)])
def test_map_inside_nested_map(opt_parameters):
    fuse_overlapping_loads, insert_copies = opt_parameters

    sdfg = _get_map_inside_nested_map()
    sdfg.name = f"map_inside_nested_map_fuse_overlapping_loads_{fuse_overlapping_loads}_insert_copies_{insert_copies}"
    sdfg.validate()

    # Symbolic values requested by the user
    klon = 64
    klev = 64
    kidia = 1
    kfdia = 32

    # Map of array shapes (from the SDFG snippet): only the shape tuples matter for creating arrays
    arr_shapes = {
        "int_array": (klon, 5, 5),
        "int_array2": (klon, 5, 5),
    }

    # Create Fortran-ordered NumPy arrays
    arrays = {name: numpy.random.random(shape).astype(numpy.float64, order='F') for name, shape in arr_shapes.items()}
    # Create scalars requested
    scalars = {"kfdia": numpy.int64(kfdia), "kidia": numpy.int64(kidia), "klon": numpy.int64(klon)}

    # Quick verification display: shape and contiguity / strides
    run_vectorization_test(dace_func=sdfg,
                           from_sdfg=True,
                           arrays=arrays,
                           params=scalars,
                           vector_width=8,
                           save_sdfgs=True,
                           sdfg_name=sdfg.name,
                           fuse_overlapping_loads=fuse_overlapping_loads,
                           insert_copies=insert_copies,
                           no_inline=True)


@pytest.mark.parametrize("opt_parameters", [(True, True), (True, False), (False, True), (False, False)])
def test_snippet_from_cloudsc_three_without_inline_sdfgs(opt_parameters):
    fuse_overlapping_loads, insert_copies = opt_parameters

    sdfg = _get_cloudsc_snippet_three(add_scalar=False)
    sdfg.name = f"cloudsc_snippet_three_without_inline_sdfgs_fuse_overlapping_loads_{fuse_overlapping_loads}_insert_copies_{insert_copies}"
    sdfg.validate()

    # Symbolic values requested by the user
    klon = 64
    klev = 64
    kidia = 1
    kfdia = 32

    # Map of array shapes (from the SDFG snippet): only the shape tuples matter for creating arrays
    arr_shapes = {
        "tendency_tmp_q": (klon, klev),
        "pa": (klon, klev),
        "pq": (klon, klev),
        "tendency_tmp_t": (klon, klev),
        "tendency_tmp_a": (klon, klev),
        "pt": (klon, klev),
        "zqx0": (klon, klev, 5),
        "zqx": (klon, klev, 5),
        "ztp1": (klon, klev),
        "zaorig": (klon, klev),
        "za": (klon, klev),
    }

    # Create Fortran-ordered NumPy arrays
    arrays = {name: numpy.random.random(shape).astype(numpy.float64, order='F') for name, shape in arr_shapes.items()}
    # Create scalars requested
    scalars = {
        "kfdia": numpy.int64(kfdia),
        "kidia": numpy.int64(kidia),
        "ptsphy": numpy.float64(0.0),
        "klev": numpy.int64(klev),
        "klon": numpy.int64(klon),
    }

    # Quick verification display: shape and contiguity / strides
    run_vectorization_test(dace_func=sdfg,
                           from_sdfg=True,
                           arrays=arrays,
                           params=scalars,
                           vector_width=8,
                           save_sdfgs=True,
                           sdfg_name=sdfg.name,
                           fuse_overlapping_loads=fuse_overlapping_loads,
                           insert_copies=insert_copies,
                           no_inline=True)


@pytest.mark.parametrize("opt_parameters", [(True, True), (True, False), (False, True), (False, False)])
def test_snippet_from_cloudsc_three_with_scalar_use(opt_parameters):
    fuse_overlapping_loads, insert_copies = opt_parameters

    sdfg = _get_cloudsc_snippet_three(add_scalar=True)
    sdfg.name = f"cloudsc_snippet_three_with_scalar_use_fuse_overlapping_loads_{fuse_overlapping_loads}_insert_copies_{insert_copies}"
    sdfg.validate()

    # Symbolic values requested by the user
    klon = 64
    klev = 64
    kidia = 1
    kfdia = 32

    # Map of array shapes (from the SDFG snippet): only the shape tuples matter for creating arrays
    arr_shapes = {
        "tendency_tmp_q": (klon, klev),
        "pa": (klon, klev),
        "pq": (klon, klev),
        "tendency_tmp_t": (klon, klev),
        "tendency_tmp_a": (klon, klev),
        "pt": (klon, klev),
        "zqx0": (klon, klev, 5),
        "zqx": (klon, klev, 5),
        "ztp1": (klon, klev),
        "zaorig": (klon, klev),
        "za": (klon, klev),
    }

    # Create Fortran-ordered NumPy arrays
    arrays = {name: numpy.random.random(shape).astype(numpy.float64, order='F') for name, shape in arr_shapes.items()}
    # Create scalars requested
    scalars = {
        "kfdia": numpy.int64(kfdia),
        "kidia": numpy.int64(kidia),
        "ptsphy": numpy.float64(0.0),
        "klev": numpy.int64(klev),
        "klon": numpy.int64(klon),
        "ralvdcp": numpy.float64(2.3),
    }

    # Quick verification display: shape and contiguity / strides
    run_vectorization_test(dace_func=sdfg,
                           from_sdfg=True,
                           arrays=arrays,
                           params=scalars,
                           vector_width=8,
                           save_sdfgs=True,
                           sdfg_name=sdfg.name,
                           fuse_overlapping_loads=fuse_overlapping_loads,
                           insert_copies=insert_copies)


@dace.program
def vadd_with_unary_scalar_cpu(A: dace.float64[N, N], B: dace.float64[N, N], c: dace.float64):
    for i, j in dace.map[0:N, 0:N]:
        c2 = -c
        c3 = B[i, j] + c2
        A[i, j] = A[i, j] + c3


@dace.program
def vadd_with_scalar_scalar_cpu(A: dace.float64[N, N], B: dace.float64[N, N], c1: dace.float64, c2: dace.float64):
    for i, j in dace.map[0:N, 0:N]:
        c3 = -c1
        c4 = c3 * c2
        c5 = B[i, j] + c4
        A[i, j] = A[i, j] + c5


def test_vadd_with_unary_scalar_cpu():
    N = 64
    A = numpy.random.random((N, N))
    B = numpy.random.random((N, N))
    c = numpy.float64(0.5)

    run_vectorization_test(
        dace_func=vadd_with_unary_scalar_cpu,
        arrays={
            'A': A,
            'B': B
        },
        params={
            'N': N,
            'c': c
        },
        vector_width=8,
        save_sdfgs=True,
        sdfg_name="vadd_with_unary_scalar_cpu",
    )


def test_vabs():
    N = 64
    A = numpy.random.random((N, N))

    run_vectorization_test(
        dace_func=vabs,
        arrays={
            'A': A,
        },
        params={
            'N': N,
        },
        vector_width=8,
        save_sdfgs=True,
        sdfg_name="vabs",
    )


def test_unary_symbol():
    N = 64
    A = numpy.random.random((N, N))

    run_vectorization_test(
        dace_func=unary_symbol,
        arrays={
            'A': A,
        },
        params={
            'N': N,
        },
        vector_width=8,
        save_sdfgs=True,
        sdfg_name="unary_symbol",
    )


def test_vadd_with_scalar_scalar_cpu():
    N = 64
    A = numpy.random.random((N, N))
    B = numpy.random.random((N, N))
    c1 = numpy.float64(0.5)
    c2 = numpy.float64(0.7)

    run_vectorization_test(
        dace_func=vadd_with_scalar_scalar_cpu,
        arrays={
            'A': A,
            'B': B
        },
        params={
            'N': N,
            'c1': c1,
            'c2': c2
        },
        vector_width=8,
        save_sdfgs=True,
        sdfg_name="vadd_with_scalar_scalar_cpu",
    )


def test_vadd_int():
    N = 64
    A = numpy.random.random((N, N)).astype(numpy.int64)
    B = numpy.random.random((N, N)).astype(numpy.int64)

    run_vectorization_test(
        dace_func=vadd_int,
        arrays={
            'A': A,
            'B': B
        },
        params={'N': N},
        vector_width=8,
        save_sdfgs=True,
        sdfg_name="vadd_int",
    )


def test_vadd_with_different_types():
    N = 64
    A = numpy.random.random((N, N)).astype(numpy.int64)
    B = numpy.random.random((N, N)).astype(numpy.float64)

    run_vectorization_test(
        dace_func=vadd_with_different_types,
        arrays={
            'A': A,
            'B': B
        },
        params={'N': N},
        vector_width=8,
        save_sdfgs=True,
        sdfg_name="vadd_with_different_types",
    )


def test_vadd_with_scalars_int():
    N = 64
    A = numpy.random.random((N, N)).astype(numpy.int64)
    B = numpy.random.random((N, N)).astype(numpy.int64)
    c1 = numpy.int64(5)
    c2 = numpy.int64(7)

    run_vectorization_test(
        dace_func=vadd_int_with_scalars,
        arrays={
            'A': A,
            'B': B
        },
        params={
            'N': N,
            'c1': c1,
            'c2': c2
        },
        vector_width=8,
        save_sdfgs=True,
        sdfg_name="vadd_int_with_scalars",
    )


@dace.program
def interstate_boolean_op_one(A: dace.float64[N, N], B: dace.float64[N, N], c0: dace.int64):
    for i, j in dace.map[0:N, 0:N]:
        c1 = i
        c2 = j
        c3 = (c1 > c0) or (c2 > c0)
        if c3:
            A[i, j] = A[i, j] + B[i, j]


@dace.program
def interstate_boolean_op_two(A: dace.float64[N, N], B: dace.float64[N, N], c0: dace.int64):
    for i, j in dace.map[0:N, 0:N]:
        c1 = i
        c2 = j
        c3 = (c1 > c0) or (c2 > c0)
        c4 = c3 or (A[i, j] > B[i, j])
        if not c4:
            A[i, j] = A[i, j] + B[i, j]


def test_interstate_boolean_op_one():
    N = 64
    A = numpy.random.random((N, N)).astype(numpy.int64)
    B = numpy.random.random((N, N)).astype(numpy.int64)
    c0 = numpy.int64(0)

    run_vectorization_test(
        dace_func=interstate_boolean_op_one,
        arrays={
            'A': A,
            'B': B
        },
        params={
            'N': N,
            'c0': c0,
        },
        vector_width=8,
        save_sdfgs=True,
        sdfg_name="interstate_boolean_op_one",
    )


def test_interstate_boolean_op_two():
    N = 64
    A = numpy.random.random((N, N)).astype(numpy.int64)
    B = numpy.random.random((N, N)).astype(numpy.int64)
    c0 = numpy.int64(0)

    run_vectorization_test(
        dace_func=interstate_boolean_op_two,
        arrays={
            'A': A,
            'B': B
        },
        params={
            'N': N,
            'c0': c0,
        },
        vector_width=8,
        save_sdfgs=True,
        sdfg_name="interstate_boolean_op_two",
    )


def test_interstate_boolean_op_three():
    sdfg = interstate_boolean_op_one.to_sdfg()
    sdfg.name = "interstate_boolean_op_three"
    nsdfg = {n for (n, g) in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.NestedSDFG)}.pop()
    inner_sdfg: dace.SDFG = nsdfg.sdfg
    syms = inner_sdfg.symbols
    last_state = {s for s in inner_sdfg.nodes() if inner_sdfg.out_degree(s) == 0}.pop()
    last_last_state = inner_sdfg.add_state_after(last_state, "ssss", assignments={"symsym": "__tmp0 or __tmp1"})
    inner_sdfg.add_symbol("symsym", dace.int64)
    sdfg.validate()
    sdfg.save("interstate_boolean_op_three.sdfg")
    VectorizeCPU(vector_width=8,
                 fuse_overlapping_loads=True,
                 insert_copies=True,
                 apply_on_maps=None,
                 no_inline=False,
                 fail_on_unvectorizable=True).apply_pass(sdfg, {})
    sdfg.validate()
    sdfg.save("interstate_boolean_op_three_vectorized.sdfg")


def _get_dependency_edge_to_unary_symbol_sdfg():
    klon = dace.symbolic.symbol("klon")
    # Add all arrays to the SDFGs
    in_arrays = {"int_array"}
    in_scalars = {}
    out_arrays = {"int_array2"}
    arr_shapes = {
        "int_array": ((klon, ), (1, ), dace.int64),
        "int_array2": ((klon, ), (1, ), dace.int64),
    }
    scalar_dtypes = {}
    outer_sdfg = dace.SDFG("outer")
    inner_sdfg = dace.SDFG("inner")
    inner_state = inner_sdfg.add_state("inner_compute_state", is_start_block=True)
    outer_state = outer_sdfg.add_state("outer_compute_state", is_start_block=True)

    an1_src = inner_state.add_access("int_array")
    an2_dst = inner_state.add_access("int_array2")
    an_tmp = inner_state.add_access("tmp0")
    inner_state.sdfg.add_scalar("tmp0", dace.int64, dace.dtypes.StorageType.Register, True)
    t1 = inner_state.add_tasklet("t1", set(), {"_out"}, "_out = i + 1")
    t2 = inner_state.add_tasklet("t2", {"_in1", "_in2"}, {"_out"}, "_out = _in1 > _in2")

    inner_state.add_edge(an1_src, None, t1, None, dace.memlet.Memlet(None))
    inner_state.add_edge(t1, "_out", an_tmp, None, dace.memlet.Memlet("tmp0"))
    inner_state.add_edge(an_tmp, None, t2, "_in1", dace.memlet.Memlet("tmp0"))
    inner_state.add_edge(an1_src, None, t2, "_in2", dace.memlet.Memlet("int_array[i]"))
    inner_state.add_edge(t2, "_out", an2_dst, None, dace.memlet.Memlet("int_array2[i]"))

    symbols = {"i", "klon"}
    for sym in symbols:
        inner_sdfg.add_symbol(sym, dace.int64)
        outer_sdfg.add_symbol(sym, dace.int64)

    for sdfg in [inner_sdfg, outer_sdfg]:
        for arr_name in in_arrays.union(out_arrays):
            shape, strides, dtype = arr_shapes[arr_name]
            sdfg.add_array(arr_name, shape, dtype, strides=strides, transient=False)
        for scalar_name in in_scalars:
            sdfg.add_scalar(scalar_name, scalar_dtypes[scalar_name], transient=False)

    inner_symbol_mapping = {sym: sym for sym in symbols}
    in_args = in_arrays.union(in_scalars)
    nsdfg = outer_state.add_nested_sdfg(inner_sdfg, in_args, out_arrays, inner_symbol_mapping)

    m1_entry, m1_exit = outer_state.add_map(name="m1", ndrange={"i": "0:klon:1"})

    inner_sdfg.validate()

    # Access nodes to map entry
    for arr in in_arrays.union(in_scalars):
        outer_state.add_edge(outer_state.add_access(arr), None, m1_entry, f"IN_{arr}",
                             dace.memlet.Memlet.from_array(arr, outer_state.sdfg.arrays[arr]))
        m1_entry.add_in_connector(f"IN_{arr}")
        outer_state.add_edge(m1_entry, f"OUT_{arr}", nsdfg, arr,
                             dace.memlet.Memlet.from_array(arr, outer_state.sdfg.arrays[arr]))
        m1_entry.add_out_connector(f"OUT_{arr}")
    # Same for exit nodes
    for arr in out_arrays:
        outer_state.add_edge(nsdfg, arr, m1_exit, f"IN_{arr}",
                             dace.memlet.Memlet.from_array(arr, outer_state.sdfg.arrays[arr]))
        m1_exit.add_in_connector(f"IN_{arr}", force=True)
        outer_state.add_edge(m1_exit, f"OUT_{arr}", outer_state.add_access(arr), None,
                             dace.memlet.Memlet.from_array(arr, outer_state.sdfg.arrays[arr]))
        m1_exit.add_out_connector(f"OUT_{arr}", force=True)
    sdfg.validate()
    return sdfg


def test_dependency_edge_to_unary_symbol():
    sdfg = _get_dependency_edge_to_unary_symbol_sdfg()
    N = 64
    A = numpy.random.random((N, )).astype(numpy.float64)
    B = numpy.random.random((N, )).astype(numpy.float64)

    run_vectorization_test(
        dace_func=sdfg,
        arrays={
            'int_array': A,
            'int_array2': B
        },
        params={
            'klon': N,
        },
        vector_width=8,
        save_sdfgs=True,
        sdfg_name="dependency_edge_to_unary_symbol",
        from_sdfg=True,
        no_inline=True,
    )


@dace.program
def mid_sdfg(pap: dace.float64[N], ptsphy: dace.float64, r2es: dace.float64, r3ies: dace.float64, r4ies: dace.float64,
             rcldtopcf: dace.float64, rd: dace.float64, rdepliqrefdepth: dace.float64, rdepliqrefrate: dace.float64,
             rg: dace.float64, riceinit: dace.float64, rlmin: dace.float64, rlstt: dace.float64, rtt: dace.float64,
             rv: dace.float64, za: dace.float64[N], zdp: dace.float64[N], zfokoop: dace.float64[N],
             zicecld: dace.float64[N], zrho: dace.float64[N], ztp1: dace.float64[N], zcldtopdist: dace.float64[N],
             zicenuclei: dace.float64[N], zqxfg: dace.float64[N], zsolqa: dace.float64[N]):
    for it_47 in dace.map[
            0:N:1,
    ]:
        # Ice nucleation and deposition
        if ztp1[it_47] < rtt and zqxfg[it_47] > rlmin:
            # Calculate ice saturation vapor pressure
            tmp_arg_72 = (r3ies * (ztp1[it_47] - rtt)) / (ztp1[it_47] - r4ies)
            zicenuclei[it_47] = 2.0 * numpy.exp(tmp_arg_72)
            # Deposition calculation parameters
            zadd = (1.6666666666667 * rlstt * (rlstt / ztp1[it_47]))
            zbdd = (0.452488687782805 * pap[it_47] * rv * ztp1[it_47])
            # Update mixing ratios
            zqxfg[it_47] = zqxfg[it_47] + zadd
            zsolqa[it_47] = zqxfg[it_47] + zbdd


@dace.program
def huge_sdfg(pap: dace.float64[N], ptsphy: dace.float64, r2es: dace.float64, r3ies: dace.float64, r4ies: dace.float64,
              rcldtopcf: dace.float64, rd: dace.float64, rdepliqrefdepth: dace.float64, rdepliqrefrate: dace.float64,
              rg: dace.float64, riceinit: dace.float64, rlmin: dace.float64, rlstt: dace.float64, rtt: dace.float64,
              rv: dace.float64, za: dace.float64[N], zdp: dace.float64[N], zfokoop: dace.float64[N],
              zicecld: dace.float64[N], zrho: dace.float64[N], ztp1: dace.float64[N], zcldtopdist: dace.float64[N],
              zicenuclei: dace.float64[N], zqxfg: dace.float64[N], zsolqa: dace.float64[N]):
    for it_47 in dace.map[
            0:N:1,
    ]:
        # Check if crossing cloud top threshold
        if za[it_47] < rcldtopcf and za[it_47] >= rcldtopcf:
            zcldtopdist[it_47] = 0.0
        else:
            zcldtopdist[it_47] = zcldtopdist[it_47] + (zdp[it_47] / (rg * zrho[it_47]))

        # Ice nucleation and deposition
        if ztp1[it_47] < rtt and zqxfg[it_47] > rlmin:
            # Calculate ice saturation vapor pressure
            tmp_arg_72 = (r3ies * (ztp1[it_47] - rtt)) / (ztp1[it_47] - r4ies)
            tmp_call_47 = r2es * numpy.exp(tmp_arg_72)
            zvpice = (rv * tmp_call_47) / rd

            # Calculate liquid vapor pressure
            zvpliq = zfokoop[it_47] * numpy.log(zvpice)

            # Ice nuclei concentration
            tmp_arg_27 = -0.639 + ((-1.96 * zvpice + 1.96 * zvpliq) / zvpliq)
            zicenuclei[it_47] = 1000.0 * numpy.exp(tmp_arg_27)

            # Nucleation factor
            zinfactor = min(1.0, 6.66666666666667e-05 * zicenuclei[it_47])

            # Deposition calculation parameters
            zadd = (1.6666666666667 * rlstt * (rlstt / (rv * ztp1[it_47]) - 1.0)) / ztp1[it_47]
            zbdd = (0.452488687782805 * pap[it_47] * rv * ztp1[it_47]) / zvpice

            tmp_call_49 = (zicenuclei[it_47] / zrho[it_47])
            zcvds = (7.8 * tmp_call_49 * (zvpliq - zvpice)) / (zvpice * (zadd + zbdd))

            # Initial ice content
            zice0 = max(riceinit * zicenuclei[it_47] / zrho[it_47], zicecld[it_47])

            # New ice after deposition
            tmp_arg_30 = 0.666 * ptsphy * zcvds + zice0
            zinew = tmp_arg_30**1.5

            # Deposition amount
            zdepos1 = max(0.0, za[it_47] * (zinew - zice0))
            zdepos2 = min(zdepos1, 1.1)

            # Apply nucleation factor and cloud top distance factor
            tmp_arg_33 = zinfactor + (1.0 - zinfactor) * (rdepliqrefrate + zcldtopdist[it_47] / rdepliqrefdepth)
            zdepos3 = zdepos2 * min(1.0, tmp_arg_33)

            # Update mixing ratios
            zqxfg[it_47] = zqxfg[it_47] + zdepos3
            zsolqa[it_47] = zsolqa[it_47] + zdepos3


def test_huge_sdfg_with_log_exp_div():
    """Generate test data for the loop body function"""
    eps_operator_type_for_log_and_div: str = "add"
    _N = 32
    data = {
        'ptsphy': numpy.float64(36.0),  # timestep (s)
        'r2es': numpy.float64(6.11),  # saturation vapor pressure constant (hPa)
        'r3ies': numpy.float64(12.0),  # ice saturation constant
        'r4ies': numpy.float64(15.5),  # ice saturation constant
        'rcldtopcf': numpy.float64(16.8),  # cloud top threshold
        'rd': numpy.float64(287.0),  # gas constant for dry air (J/kg/K)
        'rdepliqrefdepth': numpy.float64(20.0),  # reference depth
        'rdepliqrefrate': numpy.float64(17.3),  # reference rate
        'rg': numpy.float64(9.81),  # gravity (m/s²)
        'riceinit': numpy.float64(5.3),  # initial ice content (kg/m³)
        'rlmin': numpy.float64(3.9),  # minimum liquid water (kg/m³)
        'rlstt': numpy.float64(2.5e6),  # latent heat (J/kg)
        'rtt': numpy.float64(273.15),  # triple point temperature (K)
        'rv': numpy.float64(461.5),  # gas constant for water vapor (J/kg/K)
        'N': numpy.int64(_N),
    }

    # 1D arrays with safe ranges
    rng = numpy.random.default_rng(0)

    def safe_uniform(low, high, size):
        """Avoid near-zero or extreme values that could cause NaN in log/div."""
        return rng.uniform(low, high, size).astype(numpy.float64)

    # State variables (N = grid size)
    data['pap'] = safe_uniform(1.0, 2.0, (_N, ))  # pressure-like
    data['za'] = safe_uniform(0.9, 1.5, (_N, ))  # altitude/cloud-top
    data['ztp1'] = safe_uniform(260.0, 280.0, (_N, ))  # temperature near freezing
    data['zqxfg'] = safe_uniform(5.0, 11.0, (_N, ))  # mixing ratios
    data['zsolqa'] = safe_uniform(5.0, 11.0, (_N, ))  # ice tendencies

    data['zdp'] = safe_uniform(0.5, 2.0, (_N, ))  # layer depth
    data['zfokoop'] = safe_uniform(0.95, 1.05, (_N, ))  # correction factor
    data['zicecld'] = safe_uniform(10.0, 11.0, (_N, ))  # cloud ice
    data['zrho'] = safe_uniform(0.9, 1.2, (_N, ))  # density
    data['zcldtopdist'] = safe_uniform(0.1, 1.0, (_N, ))  # distance to cloud top
    data['zicenuclei'] = safe_uniform(1e2, 1e4, (_N, ))  # ice nuclei concentration

    sdfg = huge_sdfg.to_sdfg()
    sdfg.name = f"huge_sdfg_with_log_exp_div_operator_{eps_operator_type_for_log_and_div}"
    sdfg.validate()
    #it_23: dace.int64, it_47: dace.int64
    sdfg.validate()
    sdfg.auto_optimize(dace.dtypes.DeviceType.CPU, True, True)
    sdfg.validate()
    out_no_fuse = {k: v.copy() for k, v in data.items()}
    sdfg(**out_no_fuse)
    sdfg.save(f"{sdfg.name}.sdfg")

    # Apply transformation
    copy_sdfg = copy.deepcopy(sdfg)
    VectorizeCPU(vector_width=8, insert_copies=False).apply_pass(copy_sdfg, {})
    copy_sdfg.name = f"huge_sdfg_with_log_exp_div_operator_{eps_operator_type_for_log_and_div}_vectorized"
    copy_sdfg.save(f"{copy_sdfg.name}.sdfg")

    # Run SDFG version (with transformation)
    out_fused = {k: v.copy() for k, v in data.items()}

    copy_sdfg(**out_fused)

    # Compare all arrays
    for name in data.keys():
        print(name)
        print(out_fused[name] - out_no_fuse[name])
        numpy.testing.assert_allclose(out_fused[name], out_no_fuse[name], atol=1e-12)


def test_mid_sdfg_with_log_exp_div():
    """Generate test data for the loop body function"""
    eps_operator_type_for_log_and_div = "add"
    _N = 32

    data = {
        'ptsphy': numpy.float64(36.0),  # timestep (s)
        'r2es': numpy.float64(6.11),  # saturation vapor pressure constant (hPa)
        'r3ies': numpy.float64(12.0),  # ice saturation constant
        'r4ies': numpy.float64(15.5),  # ice saturation constant
        'rcldtopcf': numpy.float64(16.8),  # cloud top threshold
        'rd': numpy.float64(287.0),  # gas constant for dry air (J/kg/K)
        'rdepliqrefdepth': numpy.float64(20.0),  # reference depth
        'rdepliqrefrate': numpy.float64(17.3),  # reference rate
        'rg': numpy.float64(9.81),  # gravity (m/s²)
        'riceinit': numpy.float64(5.3),  # initial ice content (kg/m³)
        'rlmin': numpy.float64(3.9),  # minimum liquid water (kg/m³)
        'rlstt': numpy.float64(2.5e6),  # latent heat (J/kg)
        'rtt': numpy.float64(273.15),  # triple point temperature (K)
        'rv': numpy.float64(461.5),  # gas constant for water vapor (J/kg/K)
        'N': numpy.int64(_N),
    }

    # 1D arrays with safe ranges
    rng = numpy.random.default_rng(0)

    def safe_uniform(low, high, size):
        """Avoid near-zero or extreme values that could cause NaN in log/div."""
        return rng.uniform(low, high, size).astype(numpy.float64)

    # State variables (N = grid size)
    data['pap'] = safe_uniform(1.0, 2.0, (_N, ))  # pressure-like
    data['za'] = safe_uniform(0.9, 1.5, (_N, ))  # altitude/cloud-top
    data['ztp1'] = safe_uniform(260.0, 280.0, (_N, ))  # temperature near freezing
    data['zqxfg'] = safe_uniform(5.0, 11.0, (_N, ))  # mixing ratios
    data['zsolqa'] = safe_uniform(5.0, 11.0, (_N, ))  # ice tendencies

    data['zdp'] = safe_uniform(0.5, 2.0, (_N, ))  # layer depth
    data['zfokoop'] = safe_uniform(0.95, 1.05, (_N, ))  # correction factor
    data['zicecld'] = safe_uniform(10.0, 11.0, (_N, ))  # cloud ice
    data['zrho'] = safe_uniform(0.9, 1.2, (_N, ))  # density
    data['zcldtopdist'] = safe_uniform(0.1, 1.0, (_N, ))  # distance to cloud top
    data['zicenuclei'] = safe_uniform(1e2, 1e4, (_N, ))  # ice nuclei concentration
    sdfg = mid_sdfg.to_sdfg()
    sdfg.name = f"mid_sdfg_with_log_exp_div_operator_{eps_operator_type_for_log_and_div}"
    copy_sdfg = copy.deepcopy(sdfg)
    copy_sdfg.name = f"mid_sdfg_with_log_exp_div_operator_{eps_operator_type_for_log_and_div}_branch_eliminated"

    sdfg.validate()
    copy_sdfg.validate()

    #it_23: dace.int64, it_47: dace.int64

    sdfg.validate()
    sdfg.auto_optimize(dace.dtypes.DeviceType.CPU, True, True)
    sdfg.validate()
    out_no_fuse = {k: v.copy() for k, v in data.items()}
    sdfg(**out_no_fuse)

    # Apply transformation
    VectorizeCPU(vector_width=8, insert_copies=False).apply_pass(copy_sdfg, {})

    # Run SDFG version (with transformation)
    out_fused = {k: v.copy() for k, v in data.items()}

    copy_sdfg(**out_fused)

    # Compare all arrays
    for name in data.keys():
        print(name)
        print(out_fused[name] - out_no_fuse[name])
        numpy.testing.assert_allclose(out_fused[name], out_no_fuse[name], atol=1e-12)


ssym = dace.symbolic.symbol("ssym")


@dace.program
def vecscale_unit_stride(src: dace.float64[N], dst: dace.float64[N], scale: dace.float64):
    for i, in dace.map[0:N:1]:
        dst[i] = src[i] * scale


@dace.program
def gather_load(src: dace.float64[N], idx: dace.int64[N], dst: dace.float64[N], scale: dace.float64):
    for i, in dace.map[0:N:1]:
        dst[i] = src[idx[i]] * scale


@dace.program
def gather_load_matrix_specialized(A: dace.float32[4, 8192], B: dace.int32[4, 8192], C: dace.float32[4, 8192]):
    for i, j in dace.map[0:4:1, 0:8192:1]:
        C[i, j] = A[i, B[i, j]] * 2.0


@dace.program
def strided_load_stride_2(src: dace.float64[2 * N], dst: dace.float64[N], scale: dace.float64):
    for i, in dace.map[0:N:1]:
        dst[i] = src[i * 2] * scale


@dace.program
def strided_load_stride_ssym(src: dace.float64[ssym * N], dst: dace.float64[N], scale: dace.float64):
    for i, in dace.map[0:N:1]:
        dst[i] = src[i * ssym] * scale


@dace.program
def strided_load_stride_3(src: dace.float64[3 * N], dst: dace.float64[N], scale: dace.float64):
    for i, in dace.map[0:N:1]:
        dst[i] = src[i * 3] * scale


@dace.program
def strided_load_stride_4(src: dace.float64[4 * N], dst: dace.float64[N], scale: dace.float64):
    for i, in dace.map[0:N:1]:
        dst[i] = src[i * 4] * scale


@dace.program
def strided_load_stride_5(src: dace.float64[5 * N], dst: dace.float64[N], scale: dace.float64):
    for i, in dace.map[0:N:1]:
        dst[i] = src[i * 5] * scale


@dace.program
def strided_load_stride_6(src: dace.float64[6 * N], dst: dace.float64[N], scale: dace.float64):
    for i, in dace.map[0:N:1]:
        dst[i] = src[i * 6] * scale


@dace.program
def strided_load_stride_7(src: dace.float64[7 * N], dst: dace.float64[N], scale: dace.float64):
    for i, in dace.map[0:N:1]:
        dst[i] = src[i * 7] * scale


@dace.program
def strided_load_stride_8(src: dace.float64[8 * N], dst: dace.float64[N], scale: dace.float64):
    for i, in dace.map[0:N:1]:
        dst[i] = src[i * 8] * scale


@dace.program
def strided_load_stride_16(src: dace.float64[16 * N], dst: dace.float64[N], scale: dace.float64):
    for i, in dace.map[0:N:1]:
        dst[i] = src[i * 16] * scale


@dace.program
def scatter_store(src: dace.float64[N], idx: dace.int64[N], dst: dace.float64[N], scale: dace.float64):
    for i, in dace.map[0:N:1]:
        dst[idx[i]] = src[i] * scale


@dace.program
def strided_store_stride_2(src: dace.float64[N], dst: dace.float64[2 * N], scale: dace.float64):
    for i, in dace.map[0:N:1]:
        dst[i * 2] = src[i] * scale


@dace.program
def strided_store_stride_ssym(src: dace.float64[N], dst: dace.float64[ssym * N], scale: dace.float64):
    for i, in dace.map[0:N:1]:
        dst[i * ssym] = src[i] * scale


@dace.program
def strided_store_stride_3(src: dace.float64[N], dst: dace.float64[3 * N], scale: dace.float64):
    for i, in dace.map[0:N:1]:
        dst[i * 3] = src[i] * scale


@dace.program
def strided_store_stride_4(src: dace.float64[N], dst: dace.float64[4 * N], scale: dace.float64):
    for i, in dace.map[0:N:1]:
        dst[i * 4] = src[i] * scale


@dace.program
def strided_store_stride_5(src: dace.float64[N], dst: dace.float64[5 * N], scale: dace.float64):
    for i, in dace.map[0:N:1]:
        dst[i * 5] = src[i] * scale


@dace.program
def strided_store_stride_6(src: dace.float64[N], dst: dace.float64[6 * N], scale: dace.float64):
    for i, in dace.map[0:N:1]:
        dst[i * 6] = src[i] * scale


@dace.program
def strided_store_stride_7(src: dace.float64[N], dst: dace.float64[7 * N], scale: dace.float64):
    for i, in dace.map[0:N:1]:
        dst[i * 7] = src[i] * scale


@dace.program
def strided_store_stride_8(src: dace.float64[N], dst: dace.float64[8 * N], scale: dace.float64):
    for i, in dace.map[0:N:1]:
        dst[i * 8] = src[i] * scale


@dace.program
def strided_store_stride_16(src: dace.float64[N], dst: dace.float64[16 * N], scale: dace.float64):
    for i, in dace.map[0:N:1]:
        dst[i * 16] = src[i] * scale


def test_vecscale_unit_stride():
    N = 64
    src = numpy.random.random(N)
    dst = numpy.zeros(N)
    run_vectorization_test(
        dace_func=vecscale_unit_stride,
        arrays={
            "src": src,
            "dst": dst
        },
        params={
            "N": N,
            "scale": 1.5
        },
        vector_width=8,
        save_sdfgs=True,
        sdfg_name="vecscale_unit_stride",
    )


def test_gather_load():
    N = 64
    src = numpy.random.random(N)
    idx = numpy.random.permutation(N).astype(numpy.int64)
    dst = numpy.zeros(N)
    run_vectorization_test(
        dace_func=gather_load,
        arrays={
            "src": src,
            "idx": idx,
            "dst": dst
        },
        params={
            "N": N,
            "scale": 1.5
        },
        vector_width=8,
        save_sdfgs=True,
        sdfg_name="gather_load",
    )


def test_gather_load_matrix_specialized():
    Y_val = 4
    X_val = 8192
    A = numpy.random.rand(Y_val, X_val).astype(numpy.float32)  # Random float32 values
    B = numpy.random.randint(0, X_val, size=(Y_val, X_val), dtype=numpy.int32)  # Random indices in [0, 8192)
    C = numpy.zeros((Y_val, X_val), dtype=numpy.float32)  # Output array initialized to zeros

    run_vectorization_test(
        dace_func=gather_load_matrix_specialized,
        arrays={
            "A": A,
            "B": B,
            "C": C
        },
        params={},
        vector_width=32,
        save_sdfgs=True,
        sdfg_name="gather_load_matrix_specialized",
    )


def test_strided_load_stride_2():
    N = 64
    src = numpy.random.random(2 * N)
    dst = numpy.zeros(N)
    run_vectorization_test(
        dace_func=strided_load_stride_2,
        arrays={
            "src": src,
            "dst": dst
        },
        params={
            "N": N,
            "scale": 1.5
        },
        vector_width=8,
        save_sdfgs=True,
        sdfg_name="strided_load_stride_2",
    )


def test_strided_load_stride_ssym():
    N = 64
    _ssym = 2
    src = numpy.random.random(_ssym * N)
    dst = numpy.zeros(N)
    run_vectorization_test(
        dace_func=strided_load_stride_ssym,
        arrays={
            "src": src,
            "dst": dst
        },
        params={
            "N": N,
            "scale": 1.5,
            "ssym": _ssym
        },
        vector_width=8,
        save_sdfgs=True,
        sdfg_name="strided_load_stride_ssym",
    )


def test_strided_load_stride_3():
    N = 64
    src = numpy.random.random(3 * N)
    dst = numpy.zeros(N)
    run_vectorization_test(
        dace_func=strided_load_stride_3,
        arrays={
            "src": src,
            "dst": dst
        },
        params={
            "N": N,
            "scale": 1.5
        },
        vector_width=8,
        save_sdfgs=True,
        sdfg_name="strided_load_stride_3",
    )


def test_strided_load_stride_4():
    N = 64
    src = numpy.random.random(4 * N)
    dst = numpy.zeros(N)
    run_vectorization_test(
        dace_func=strided_load_stride_4,
        arrays={
            "src": src,
            "dst": dst
        },
        params={
            "N": N,
            "scale": 1.5
        },
        vector_width=8,
        save_sdfgs=True,
        insert_copies=False,
        sdfg_name="strided_load_stride_4",
    )


def test_strided_load_stride_5():
    N = 64
    src = numpy.random.random(5 * N)
    dst = numpy.zeros(N)
    run_vectorization_test(
        dace_func=strided_load_stride_5,
        arrays={
            "src": src,
            "dst": dst
        },
        params={
            "N": N,
            "scale": 1.5
        },
        vector_width=8,
        save_sdfgs=True,
        sdfg_name="strided_load_stride_5",
    )


def test_strided_load_stride_6():
    N = 64
    src = numpy.random.random(6 * N)
    dst = numpy.zeros(N)
    run_vectorization_test(
        dace_func=strided_load_stride_6,
        arrays={
            "src": src,
            "dst": dst
        },
        params={
            "N": N,
            "scale": 1.5
        },
        vector_width=8,
        save_sdfgs=True,
        insert_copies=True,
        fuse_overlapping_loads=True,
        sdfg_name="strided_load_stride_6",
    )


def test_strided_load_stride_7():
    N = 64
    src = numpy.random.random(7 * N)
    dst = numpy.zeros(N)
    run_vectorization_test(
        dace_func=strided_load_stride_7,
        arrays={
            "src": src,
            "dst": dst
        },
        params={
            "N": N,
            "scale": 1.5
        },
        vector_width=8,
        save_sdfgs=True,
        sdfg_name="strided_load_stride_7",
    )


def test_strided_load_stride_8():
    N = 64
    src = numpy.random.random(8 * N)
    dst = numpy.zeros(N)
    run_vectorization_test(
        dace_func=strided_load_stride_8,
        arrays={
            "src": src,
            "dst": dst
        },
        params={
            "N": N,
            "scale": 1.5
        },
        vector_width=8,
        save_sdfgs=True,
        sdfg_name="strided_load_stride_8",
    )


def test_strided_load_stride_16():
    N = 64
    src = numpy.random.random(16 * N)
    dst = numpy.zeros(N)
    run_vectorization_test(
        dace_func=strided_load_stride_16,
        arrays={
            "src": src,
            "dst": dst
        },
        params={
            "N": N,
            "scale": 1.5
        },
        vector_width=8,
        save_sdfgs=True,
        sdfg_name="strided_load_stride_16",
    )


def test_scatter_store():
    N = 64
    src = numpy.random.random(N)
    idx = numpy.random.permutation(N).astype(numpy.int64)
    dst = numpy.zeros(N)
    run_vectorization_test(
        dace_func=scatter_store,
        arrays={
            "src": src,
            "idx": idx,
            "dst": dst
        },
        params={
            "N": N,
            "scale": 1.5
        },
        vector_width=8,
        save_sdfgs=True,
        sdfg_name="scatter_store",
    )


def test_strided_store_stride_2():
    N = 64
    src = numpy.random.random(N)
    dst = numpy.zeros(2 * N)
    run_vectorization_test(
        dace_func=strided_store_stride_2,
        arrays={
            "src": src,
            "dst": dst
        },
        params={
            "N": N,
            "scale": 1.5
        },
        vector_width=8,
        save_sdfgs=True,
        sdfg_name="strided_store_stride_2",
    )


def test_strided_store_stride_ssym():
    N = 64
    src = numpy.random.random(N)
    dst = numpy.zeros(2 * N)
    _ssym = numpy.int64(2)
    run_vectorization_test(
        dace_func=strided_store_stride_ssym,
        arrays={
            "src": src,
            "dst": dst
        },
        params={
            "N": N,
            "scale": 1.5,
            "ssym": _ssym
        },
        vector_width=8,
        save_sdfgs=True,
        sdfg_name="strided_store_stride_ssym",
    )


def test_strided_store_stride_3():
    N = 64
    src = numpy.random.random(N)
    dst = numpy.zeros(3 * N)
    run_vectorization_test(
        dace_func=strided_store_stride_3,
        arrays={
            "src": src,
            "dst": dst
        },
        params={
            "N": N,
            "scale": 1.5
        },
        vector_width=8,
        save_sdfgs=True,
        insert_copies=True,
        fuse_overlapping_loads=True,
        sdfg_name="strided_store_stride_3",
    )


def test_strided_store_stride_4():
    N = 64
    src = numpy.random.random(N)
    dst = numpy.zeros(4 * N)
    run_vectorization_test(
        dace_func=strided_store_stride_4,
        arrays={
            "src": src,
            "dst": dst
        },
        params={
            "N": N,
            "scale": 1.5
        },
        vector_width=8,
        save_sdfgs=True,
        sdfg_name="strided_store_stride_4",
    )


def test_strided_store_stride_5():
    N = 64
    src = numpy.random.random(N)
    dst = numpy.zeros(5 * N)
    run_vectorization_test(
        dace_func=strided_store_stride_5,
        arrays={
            "src": src,
            "dst": dst
        },
        params={
            "N": N,
            "scale": 1.5
        },
        vector_width=8,
        save_sdfgs=True,
        sdfg_name="strided_store_stride_5",
    )


def test_strided_store_stride_6():
    N = 64
    src = numpy.random.random(N)
    dst = numpy.zeros(6 * N)
    run_vectorization_test(
        dace_func=strided_store_stride_6,
        arrays={
            "src": src,
            "dst": dst
        },
        params={
            "N": N,
            "scale": 1.5
        },
        vector_width=8,
        save_sdfgs=True,
        sdfg_name="strided_store_stride_6",
    )


def test_strided_store_stride_7():
    N = 64
    src = numpy.random.random(N)
    dst = numpy.zeros(7 * N)
    run_vectorization_test(
        dace_func=strided_store_stride_7,
        arrays={
            "src": src,
            "dst": dst
        },
        params={
            "N": N,
            "scale": 1.5
        },
        vector_width=8,
        save_sdfgs=True,
        sdfg_name="strided_store_stride_7",
    )


def test_strided_store_stride_8():
    N = 64
    src = numpy.random.random(N)
    dst = numpy.zeros(8 * N)
    run_vectorization_test(
        dace_func=strided_store_stride_8,
        arrays={
            "src": src,
            "dst": dst
        },
        params={
            "N": N,
            "scale": 1.5
        },
        vector_width=8,
        save_sdfgs=True,
        sdfg_name="strided_store_stride_8",
    )


def test_strided_store_stride_16():
    N = 64
    src = numpy.random.random(N)
    dst = numpy.zeros(16 * N)
    run_vectorization_test(
        dace_func=strided_store_stride_16,
        arrays={
            "src": src,
            "dst": dst
        },
        params={
            "N": N,
            "scale": 1.5
        },
        vector_width=8,
        save_sdfgs=True,
        sdfg_name="strided_store_stride_16",
    )


def test_strided_store_stride_ssym():
    N = 64
    _ssym = 2
    src = numpy.random.random(N)
    dst = numpy.zeros(2 * N)
    run_vectorization_test(
        dace_func=strided_store_stride_ssym,
        arrays={
            "src": src,
            "dst": dst
        },
        params={
            "N": N,
            "scale": 1.5,
            "ssym": _ssym
        },
        vector_width=8,
        save_sdfgs=True,
        sdfg_name="strided_store_stride_ssym",
    )


Y = dace.symbolic.symbol("Y")
X = dace.symbolic.symbol("X")


@dace.program
def nested_matrix_gather_load(A: dace.float32[Y, X], B: dace.int32[Y, X], C: dace.float32[Y, X], scale: dace.float32):
    for i, j in dace.map[0:Y:1, 0:X:1]:
        C[i, j] = A[i, B[i, j]] * scale


@dace.program
def nested_matrix_gather_load_specialized(A: dace.float32[Y, X], B: dace.int32[Y, X], C: dace.float32[Y, X]):
    for i, j in dace.map[0:Y:1, 0:X:1]:
        C[i, j] = A[i, B[i, j]] * 2.0


def test_nested_matrix_gather_load():
    X_val = 32
    Y_val = 32
    A = numpy.random.rand(Y_val, X_val).astype(numpy.float32)
    B = numpy.random.randint(0, X_val, size=(Y_val, X_val), dtype=numpy.int32)
    C = numpy.zeros((Y_val, X_val), dtype=numpy.float32)
    run_vectorization_test(
        dace_func=nested_matrix_gather_load,
        arrays={
            "A": A,
            "B": B,
            "C": C,
        },
        params={
            "X": X_val,
            "Y": Y_val,
            "scale": 2.0
        },
        vector_width=8,
        save_sdfgs=True,
        sdfg_name="nested_matrix_gather_load",
    )


def test_nested_matrix_gather_load_specialized():
    X_val = 32
    Y_val = 32
    A = numpy.random.rand(Y_val, X_val).astype(numpy.float32)
    B = numpy.random.randint(0, X_val, size=(Y_val, X_val), dtype=numpy.int32)
    C = numpy.zeros((Y_val, X_val), dtype=numpy.float32)
    run_vectorization_test(
        dace_func=nested_matrix_gather_load_specialized,
        arrays={
            "A": A,
            "B": B,
            "C": C,
        },
        params={
            "X": X_val,
            "Y": Y_val,
        },
        vector_width=8,
        save_sdfgs=True,
        sdfg_name="nested_matrix_gather_load_specialized",
    )


def _get_unstructured_access_cloudsc_sdfg(layout: str = "C") -> dace.SDFG:
    klon = dace.symbolic.symbol("klon")
    klev = dace.symbolic.symbol("klev")
    sdfg_outer = dace.SDFG(f"unstructured_access_cloudsc_sdfg_{layout.lower()}")
    sdfg_inner = dace.SDFG("inner")

    outer_symbols = {("klon", dace.int64), ("klev", dace.int64)}
    inner_symbols = {
        ("jo", dace.int64),
        ("_for_it_88", dace.int64),
    }  #  ("_for_it_85", dace.int64)
    # Add inner symbols to inner SDFG
    for sname, stype in inner_symbols:
        sdfg_inner.add_symbol(sname, stype)

    # Add outer symbols to both
    for sdfg in [sdfg_outer, sdfg_inner]:
        for sname, stype in outer_symbols:
            sdfg.add_symbol(sname, stype)

    if layout == "Fortran":
        arrays = {("iorder", dace.int64, (klon, 5), (1, klon)),
                  ("zqx", dace.float64, (klon, klev, 5), (1, klon, klon * klev)),
                  ("zsinksum", dace.float64, (klon, 5), (1, klon)), ("zratio", dace.float64, (klon, 5), (1, klon))}
    else:
        assert layout == "C"
        arrays = {("iorder", dace.int64, (5, klon), (klon, 1)),
                  ("zqx", dace.float64, (5, klev, klon), (klon * klev, klon, 1)),
                  ("zsinksum", dace.float64, (5, klon), (klon, 1)), ("zratio", dace.float64, (5, klon), (klon, 1))}
    scalars = {("zmm", dace.float64), ("zrr", dace.float64)}

    # Add arrays
    for sdfg in [sdfg_outer, sdfg_inner]:
        for arr_name, dtype, shape, stride in arrays:
            sdfg.add_array(arr_name, shape, dtype, strides=stride)

    # Add scalars to inner SDFG
    for sname, dtype in scalars:
        sdfg.add_scalar(sname, dtype, dace.dtypes.StorageType.Register, True)

    # Add states
    state_outer = sdfg_outer.add_state("outer_s1", is_start_block=True)
    state_inner1 = sdfg_inner.add_state("inner_s1", is_start_block=True)
    state_inner2 = sdfg_inner.add_state("inner_s2")

    # Populate inner SDFG
    # ==============================
    if layout == "Fortran":
        sdfg_inner.add_edge(state_inner1, state_inner2, InterstateEdge(assignments={"jo": "iorder[_for_it_88, 0]"}))
    else:
        sdfg_inner.add_edge(state_inner1, state_inner2, InterstateEdge(assignments={"jo": "iorder[0, _for_it_88]"}))

    zqx = state_inner2.add_access("zqx")
    zsinksum = state_inner2.add_access("zsinksum")
    zratio = state_inner2.add_access("zratio")
    zrr = state_inner2.add_access("zrr")
    zmm = state_inner2.add_access("zmm")

    t1 = state_inner2.add_tasklet("t1", {"_in1"}, {"_out"}, "_out = max(1e-14, _in1)")
    t2 = state_inner2.add_tasklet("t2", {"_in1", "_in2"}, {"_out"}, "_out = max(_in1, _in2)")
    t3 = state_inner2.add_tasklet("t3", {"_in1", "_in2"}, {"_out"}, "_out = _in1 / _in2")

    if layout == "Fortran":
        state_inner2.add_edge(zqx, None, t1, "_in1", dace.memlet.Memlet("zqx[_for_it_88, 0, jo - 1]"))
    else:
        state_inner2.add_edge(zqx, None, t1, "_in1", dace.memlet.Memlet("zqx[ jo - 1, 0, _for_it_88]"))

    state_inner2.add_edge(t1, "_out", zmm, None, dace.memlet.Memlet("zmm"))

    if layout == "Fortran":
        state_inner2.add_edge(zsinksum, None, t2, "_in2", dace.memlet.Memlet("zsinksum[_for_it_88, jo-1]"))
    else:
        state_inner2.add_edge(zsinksum, None, t2, "_in2", dace.memlet.Memlet("zsinksum[jo-1, _for_it_88]"))

    state_inner2.add_edge(zmm, None, t2, "_in1", dace.memlet.Memlet("zmm"))
    state_inner2.add_edge(t2, "_out", zrr, None, dace.memlet.Memlet("zrr"))

    state_inner2.add_edge(zrr, None, t3, "_in2", dace.memlet.Memlet("zrr"))
    state_inner2.add_edge(zmm, None, t3, "_in1", dace.memlet.Memlet("zmm"))

    if layout == "Fortran":
        state_inner2.add_edge(t3, "_out", zratio, None, dace.memlet.Memlet("zratio[_for_it_88, jo -1]"))
    else:
        state_inner2.add_edge(t3, "_out", zratio, None, dace.memlet.Memlet("zratio[jo -1, _for_it_88]"))

    # ==============================

    map_entry, map_exit = state_outer.add_map("m1", {"_for_it_88": "0:klon:1"})

    nsdfg = state_outer.add_nested_sdfg(sdfg_inner,
                                        inputs={"zqx", "zsinksum", "iorder"},
                                        outputs={"zratio"},
                                        symbol_mapping={"_for_it_88": "_for_it_88"})

    # Add in arrays
    for arr_name in {"zqx", "zsinksum", "iorder"}:
        an = state_outer.add_access(arr_name)
        state_outer.add_edge(an, None, map_entry, f"IN_{arr_name}",
                             dace.memlet.Memlet.from_array(arr_name, state_outer.sdfg.arrays[arr_name]))
        state_outer.add_edge(map_entry, f"OUT_{arr_name}", nsdfg, arr_name,
                             dace.memlet.Memlet.from_array(arr_name, state_outer.sdfg.arrays[arr_name]))
        map_entry.add_in_connector(f"IN_{arr_name}")
        map_entry.add_out_connector(f"OUT_{arr_name}")

    # Add out arrays
    for arr_name in {"zratio"}:
        an = state_outer.add_access(arr_name)
        state_outer.add_edge(nsdfg, arr_name, map_exit, f"IN_{arr_name}",
                             dace.memlet.Memlet.from_array(arr_name, state_outer.sdfg.arrays[arr_name]))
        state_outer.add_edge(map_exit, f"OUT_{arr_name}", an, None,
                             dace.memlet.Memlet.from_array(arr_name, state_outer.sdfg.arrays[arr_name]))
        map_exit.add_in_connector(f"IN_{arr_name}")
        map_exit.add_out_connector(f"OUT_{arr_name}")

    sdfg_outer.validate()
    return sdfg_outer


@pytest.mark.parametrize("layout", ["C", "Fortran"])
def test_unstructured_access_pattern(layout: str):
    klon_val = 32
    klev_val = 32
    """
        arrays = {("iorder", dace.int64, ("klon", 5), (1, "klon")),
                    ("zqx", dace.float64, ("klon", "klev", 5), (1, "klon", "klon * klev")),
                    ("zsinksum", dace.float64, ("klon", 5), (1, "klon")),
                    ("zratio", dace.float64, ("klon", 5), (1, "klon"))}
    """
    zqx = numpy.random.rand(5, klev_val, klon_val).astype(numpy.float64)
    iorder = numpy.random.randint(1, 6, size=(5, klon_val), dtype=numpy.int64)
    zratio = numpy.zeros((5, klon_val), dtype=numpy.float64)
    zsinksum = numpy.zeros((5, klon_val), dtype=numpy.float64)
    run_vectorization_test(
        dace_func=_get_unstructured_access_cloudsc_sdfg(layout=layout),
        arrays={
            "zqx": zqx,
            "zratio": zratio,
            "zsinksum": zsinksum,
            "iorder": iorder,
        },
        params={
            "klon": klon_val,
            "klev": klev_val,
        },
        vector_width=8,
        save_sdfgs=True,
        from_sdfg=True,
        sdfg_name=f"unstructured_access_pattern_layout_{layout.lower()}",
    )


KLON = dace.symbol('KLON')
KLEV = dace.symbol('KLEV')
NCLDQL = dace.symbol('NCLDQL')
NCLDQI = dace.symbol('NCLDQI')


@dace.program
def cloud_fraction_update(
        ZA: dace.float64[KLEV, KLON],
        ZQX: dace.float64[5, KLEV, KLON],  # last dim just example
        ZLI: dace.float64[KLEV, KLON],
        ZLIQFRAC: dace.float64[KLEV, KLON],
        ZICEFRAC: dace.float64[KLEV, KLON],
        RLMIN: dace.float64):
    # Loop over levels and horizontal domain
    for jk in dace.map[0:KLEV]:
        for jl in dace.map[0:KLON]:

            # 1. Clip ZA to [0, 1]
            ZA[jk, jl] = max(0.0, min(1.0, ZA[jk, jl]))

            # 2. Compute total liquid+ice
            ZLI[jk, jl] = (ZQX[NCLDQL, jk, jl] + ZQX[NCLDQI, jk, jl])

            if ZLI[jk, jl] > RLMIN:
                ZLIQFRAC[jk, jl] = (ZQX[NCLDQL, jk, jl] / ZLI[jk, jl])
                ZICEFRAC[jk, jl] = (1.0 - ZLIQFRAC[jk, jl])
            else:
                ZLIQFRAC[jk, jl] = 0.0
                ZICEFRAC[jk, jl] = 0.0


def test_cloud_fraction_update():
    # Example sizes
    klev = 16
    klon = 32

    # Pick any valid indexes for QL/QI
    ncldql = 0
    ncldqi = 1

    # Create test arrays
    ZA = numpy.random.uniform(-0.2, 1.2, size=(klev, klon))
    ZQX = numpy.abs(numpy.random.randn(5, klev, klon)) * 1e-4
    ZLI = numpy.zeros((klev, klon))
    ZLIQFRAC = numpy.zeros((klev, klon))
    ZICEFRAC = numpy.zeros((klev, klon))

    RLMIN = 1e-12

    run_vectorization_test(
        dace_func=cloud_fraction_update,
        arrays={
            "ZA": ZA,
            "ZQX": ZQX,
            "ZLI": ZLI,
            "ZLIQFRAC": ZLIQFRAC,
            "ZICEFRAC": ZICEFRAC,
        },
        params={
            "RLMIN": RLMIN,
            "KLON": klon,
            "KLEV": klev,
            "NCLDQL": ncldql,
            "NCLDQI": ncldqi
        },
        vector_width=8,
        save_sdfgs=True,
        sdfg_name=f"cloud_fraction_update",
    )


from math import log, exp, pow


@dace.program
def log_implementations(A: dace.float64[S], B: dace.float64[S]):
    for i in dace.map[0:S]:
        B[i] = log(A[i])


@dace.program
def exp_implementations(A: dace.float64[S], B: dace.float64[S]):
    for i in dace.map[0:S]:
        B[i] = exp(A[i])


@dace.program
def pow_implementations(A: dace.float64[S], B: dace.float64[S]):
    for i in dace.map[0:S]:
        B[i] = pow(A[i], 3.3)


def test_log():
    # Create test arrays
    _S = 64
    A = numpy.random.uniform(0.2, 1.2, size=(_S, ))
    B = numpy.abs(numpy.random.randn(_S, )) * 1e-4

    # Baseline SDFG
    log_implementations_std_sdfg = log_implementations.to_sdfg()
    ReplaceSTDLogWithDaCeLog().apply_pass(log_implementations_std_sdfg, {})

    run_vectorization_test(
        dace_func=log_implementations_std_sdfg,
        arrays={
            "A": A,
            "B": B
        },
        params={"S": _S},
        vector_width=8,
        save_sdfgs=True,
        sdfg_name=f"test_log",
        from_sdfg=True,
    )


def test_exp():
    # Create test arrays
    _S = 64
    A = numpy.random.uniform(0.2, 1.2, size=(_S, ))
    B = numpy.abs(numpy.random.randn(_S, )) * 1e-4

    # Baseline SDFG
    exp_implementations_std_sdfg = exp_implementations.to_sdfg()
    ReplaceSTDExpWithDaCeExp().apply_pass(exp_implementations_std_sdfg, {})

    run_vectorization_test(
        dace_func=exp_implementations_std_sdfg,
        arrays={
            "A": A,
            "B": B
        },
        params={"S": _S},
        vector_width=8,
        save_sdfgs=True,
        sdfg_name=f"test_exp",
        from_sdfg=True,
    )


@pytest.mark.skip
def test_pow():
    # Create test arrays
    _S = 64
    A = numpy.random.uniform(0.2, 1.2, size=(_S, ))
    B = numpy.abs(numpy.random.randn(_S, )) * 1e-4

    # Baseline SDFG
    pow_implementations_std_sdfg = pow_implementations.to_sdfg()
    ReplaceSTDPowWithDaCePow().apply_pass(pow_implementations_std_sdfg, {})

    run_vectorization_test(dace_func=pow_implementations,
                           arrays={
                               "A": A,
                               "B": B
                           },
                           params={"S": _S},
                           vector_width=8,
                           save_sdfgs=True,
                           sdfg_name=f"test_pow",
                           from_sdfg=True)


@dace.program
def dace_s000(A: dace.float64[S], B: dace.float64[S]):
    for nl in range(2):
        for i in dace.map[0:S:1]:
            A[i] = B[i] + 1.0


def test_s000():
    # Create test arrays
    _S = 64
    A = numpy.random.uniform(0.2, 1.2, size=(_S, ))
    B = numpy.abs(numpy.random.randn(_S, )) * 1e-4

    run_vectorization_test(dace_func=dace_s000,
                           arrays={
                               "A": A,
                               "B": B
                           },
                           params={"S": _S},
                           vector_width=8,
                           save_sdfgs=True,
                           sdfg_name=f"s000",
                           from_sdfg=False)


if __name__ == "__main__":
    test_log()
    test_exp()
    #test_pow()
    test_dependency_edge_to_unary_symbol()
    test_vabs()
    test_interstate_boolean_op_one()
    test_interstate_boolean_op_two()
    test_interstate_boolean_op_three()
    test_memset_4d()
    test_vadd_int()
    test_vadd_with_scalars_int()
    test_vadd_with_unary_scalar_cpu()
    test_vadd_with_scalar_scalar_cpu()
    test_v_const_subs_4d()
    test_v_const_subs_4d_indirect_access()
    test_disjoint_chain()
    test_disjoint_chain_with_overlapping_region_fusion()
    test_vsubs_cpu()
    test_vsubs_two_cpu()
    test_v_const_subs_cpu()
    test_v_const_subs_two_cpu()
    test_unsupported_op()
    test_unsupported_op_two()
    test_tasklets_in_if()
    test_tasklets_in_if_two()
    test_nested_sdfg()
    test_simple_cpu()
    test_no_maps()
    test_spmv()
    test_overlapping_access()
    test_jacobi2d()
    test_overlapping_access_same_src_and_dst()
    test_overlapping_access_same_src_and_dst_in_nestedsdfg()
    for argtuple in [(True, True), (True, False), (False, True), (False, False)]:
        test_disjoint_chain_split_branch_only(argtuple)
        test_jacobi2d_with_parameters(argtuple)
        test_snippet_from_cloudsc_three(argtuple)
        test_snippet_from_cloudsc_three_with_scalar_use(argtuple)
        test_snippet_from_cloudsc_three_without_inline_sdfgs(argtuple)
        test_snippet_from_cloudsc_three_with_partial_subset(argtuple)
        test_snippet_from_cloudsc_three_with_partial_subset_without_inline(argtuple)
        test_map_inside_nested_map(argtuple)
    test_jacobi2d_with_fuse_overlapping_loads()
    test_division_by_zero_cpu()
    test_memset_with_fuse_and_copyin_enabled()
    test_nested_memset_with_fuse_and_copyin_enabled()
    test_snippet_from_cloudsc_one()
    test_snippet_from_cloudsc_two()
    test_snippet_from_cloudsc_two_fuse_overlapping_loads()
    test_snippet_from_cloudsc_four()
    test_max_with_constant()
    test_vadd_with_different_types()
    test_unary_symbol()
    test_mid_sdfg_with_log_exp_div()
    test_huge_sdfg_with_log_exp_div()
    test_vecscale_unit_stride()
    test_gather_load()
    test_strided_load_stride_2()
    test_strided_load_stride_ssym()
    test_strided_load_stride_3()
    test_strided_load_stride_4()
    test_strided_load_stride_5()
    test_strided_load_stride_6()
    test_strided_load_stride_7()
    test_strided_load_stride_8()
    test_strided_load_stride_16()
    test_scatter_store()
    test_strided_store_stride_2()
    test_strided_store_stride_ssym()
    test_strided_store_stride_3()
    test_strided_store_stride_4()
    test_strided_store_stride_5()
    test_strided_store_stride_6()
    test_strided_store_stride_7()
    test_strided_store_stride_8()
    test_strided_store_stride_16()
    test_nested_matrix_gather_load()
    test_nested_matrix_gather_load_specialized()
    test_gather_load_matrix_specialized()
    test_unstructured_access_pattern("C")
    test_unstructured_access_pattern("Fortran")
    test_cloud_fraction_update()
