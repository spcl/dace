# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import pytest
import numpy
from tests.passes.vectorization._harness import (
    run_vectorization_test,
    assert_fused_nsdfg_structure,
    S,
)


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

    vectorized_sdfg = run_vectorization_test(
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
        sdfg_name=f"jacobi2d_with_fuse_overlapping_loads_{fuse_overlapping_loads}_with_insert_copies_{insert_copies}",
        fuse_overlapping_loads=fuse_overlapping_loads,
        insert_copies=insert_copies)
    if fuse_overlapping_loads and insert_copies:
        assert_fused_nsdfg_structure(vectorized_sdfg, ("A", ))


@dace.program
def jacobi1d(A: dace.float64[S], B: dace.float64[S], tsteps: dace.int64):
    for t in range(tsteps):
        for i in dace.map[0:S - 2]:
            B[i + 1] = 0.33333 * (A[i] + A[i + 1] + A[i + 2])

        for i in dace.map[0:S - 2]:
            A[i + 1] = 0.33333 * (B[i] + B[i + 1] + B[i + 2])


@pytest.mark.parametrize("param_tuple", [(True, True), (True, False), (False, True), (False, False)])
def test_jacobi1d_with_parameters(param_tuple):
    fuse_overlapping_loads, insert_copies = param_tuple
    _S = 130
    A = numpy.random.random((_S, ))
    B = numpy.random.random((_S, ))

    vectorized_sdfg = run_vectorization_test(
        dace_func=jacobi1d,
        arrays={
            'A': A,
            'B': B
        },
        params={
            'S': _S,
            'tsteps': 5,
        },
        vector_width=8,
        sdfg_name=f"jacobi1d_with_fuse_overlapping_loads_{fuse_overlapping_loads}_with_insert_copies_{insert_copies}",
        fuse_overlapping_loads=fuse_overlapping_loads,
        insert_copies=insert_copies)
    if fuse_overlapping_loads and insert_copies:
        assert_fused_nsdfg_structure(vectorized_sdfg, ("A", ))


@dace.program
def heat3d(A: dace.float64[S, S, S], B: dace.float64[S, S, S], tsteps: dace.int64):
    for t in range(tsteps):
        for i, j, k in dace.map[0:S - 2, 0:S - 2, 0:S - 2]:
            B[i + 1, j + 1, k + 1] = (0.125 * (A[i + 2, j + 1, k + 1] - 2.0 * A[i + 1, j + 1, k + 1] +
                                                A[i, j + 1, k + 1]) + 0.125 *
                                      (A[i + 1, j + 2, k + 1] - 2.0 * A[i + 1, j + 1, k + 1] + A[i + 1, j, k + 1]) +
                                      0.125 * (A[i + 1, j + 1, k + 2] - 2.0 * A[i + 1, j + 1, k + 1] +
                                               A[i + 1, j + 1, k]) + A[i + 1, j + 1, k + 1])

        for i, j, k in dace.map[0:S - 2, 0:S - 2, 0:S - 2]:
            A[i + 1, j + 1, k + 1] = (0.125 * (B[i + 2, j + 1, k + 1] - 2.0 * B[i + 1, j + 1, k + 1] +
                                                B[i, j + 1, k + 1]) + 0.125 *
                                      (B[i + 1, j + 2, k + 1] - 2.0 * B[i + 1, j + 1, k + 1] + B[i + 1, j, k + 1]) +
                                      0.125 * (B[i + 1, j + 1, k + 2] - 2.0 * B[i + 1, j + 1, k + 1] +
                                               B[i + 1, j + 1, k]) + B[i + 1, j + 1, k + 1])


@pytest.mark.parametrize("param_tuple", [(True, True), (True, False), (False, True), (False, False)])
def test_heat3d_with_parameters(param_tuple):
    fuse_overlapping_loads, insert_copies = param_tuple
    _S = 18
    A = numpy.random.random((_S, _S, _S))
    B = numpy.random.random((_S, _S, _S))

    vectorized_sdfg = run_vectorization_test(
        dace_func=heat3d,
        arrays={
            'A': A,
            'B': B
        },
        params={
            'S': _S,
            'tsteps': 3,
        },
        vector_width=8,
        sdfg_name=f"heat3d_with_fuse_overlapping_loads_{fuse_overlapping_loads}_with_insert_copies_{insert_copies}",
        fuse_overlapping_loads=fuse_overlapping_loads,
        insert_copies=insert_copies)
    if fuse_overlapping_loads and insert_copies:
        assert_fused_nsdfg_structure(vectorized_sdfg, ("A", ))
