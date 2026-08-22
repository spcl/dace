# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Regression tests for the "pick a memlet side by comparing data names" bug class.

``Memlet.subset`` names one endpoint of a copy and ``other_subset`` the other, but WHICH is carried
by the memlet's ``_is_data_src`` flag -- never derivable from the endpoint data names. On a self-copy
(``A -> A``, e.g. the CloudSC flux level shift ``pfsqlf[jk] = pfsqlf[jk-1]``) both endpoints match
``memlet.data``, so a name test picks a side arbitrarily and silently reverses half of them.

The confirmed instance was ``InsertExplicitCopies._replace_direct_copies``. These tests cover the
other passes that carried the same pattern. Each runs a self-copy in BOTH memlet orientations through
the pass and then COMPILES AND RUNS the rewritten graph -- the original bug survived a test that
asserted on subsets and never executed the graph.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MPI4PY_RC_INITIALIZE", "0")
os.environ.setdefault("OMPI_MCA_pml", "ob1")
os.environ.setdefault("OMPI_MCA_btl", "self,vader")
os.environ.setdefault("UCX_VFS_ENABLE", "n")

import numpy as np
import pytest

import dace
from dace.sdfg import nodes
from dace.transformation.passes.buffer_expansion import BufferExpansion
from dace.transformation.passes.insert_assign_tasklets_at_map_boundary import InsertAssignTaskletsAtMapBoundary
from dace.transformation.passes.insert_unit_copy_assign_tasklets import InsertAssignTaskletsForUnitCopies

READ = (1, 4)
WRITE = (1, 3)


def src_relative(name: str):
    """Unit self-copy ``p[1, 3] = p[1, 4]`` with ``subset`` = the READ element.

    The orientation ``Memlet.try_initialize`` defaults to when both endpoints name the same array.
    """
    return build(name, dace.Memlet(data='p', subset=f'{READ[0]}, {READ[1]}', other_subset=f'{WRITE[0]}, {WRITE[1]}'))


def dst_relative(name: str):
    """The SAME copy with ``subset`` = the WRITTEN element (``_is_data_src`` False).

    An anonymous-source ``->`` memlet takes ``data`` from the destination; ``try_initialize`` keeps an
    already-set flag, so the orientation survives attachment to a ``p -> p`` edge.
    """
    return build(name, dace.Memlet(f'[{READ[0]}, {READ[1]}] -> p[{WRITE[0]}, {WRITE[1]}]'))


def build(name: str, memlet: dace.Memlet):
    """4x5 array ``p`` with one ``AccessNode -> AccessNode`` edge on the SAME array."""
    sdfg = dace.SDFG(name)
    sdfg.add_array('p', [4, 5], dace.float64)
    state = sdfg.add_state('s')
    state.add_edge(state.add_access('p'), None, state.add_access('p'), None, memlet)
    return sdfg


def run(sdfg: dace.SDFG) -> np.ndarray:
    p = np.arange(20, dtype=np.float64).reshape(4, 5).copy()
    sdfg(p=p)
    return p


def expected() -> np.ndarray:
    p = np.arange(20, dtype=np.float64).reshape(4, 5).copy()
    p[WRITE] = p[READ]
    return p


ORIENTATIONS = [src_relative, dst_relative]


@pytest.mark.parametrize('build_sdfg', ORIENTATIONS)
def test_insert_unit_copy_assign_tasklets_keeps_self_copy_direction(build_sdfg):
    """``InsertAssignTaskletsForUnitCopies`` splits a unit ``A -> A`` copy into
    ``A -> tasklet -> A``. Choosing the two halves by data name reads the destination element and
    writes the source one, i.e. runs the copy backwards."""
    sdfg = build_sdfg(f'unit_copy_{build_sdfg.__name__}')
    assert InsertAssignTaskletsForUnitCopies().apply_pass(sdfg, {}) == 1
    assert any(isinstance(n, nodes.Tasklet) for n, _ in sdfg.all_nodes_recursive())
    sdfg.validate()
    np.testing.assert_array_equal(run(sdfg), expected())


@pytest.mark.parametrize('build_sdfg', ORIENTATIONS)
def test_insert_assign_tasklets_at_map_boundary_keeps_self_copy_direction(build_sdfg):
    """Same split, done by ``InsertAssignTaskletsAtMapBoundary`` for edges carrying
    ``other_subset``; same failure mode."""
    sdfg = build_sdfg(f'map_boundary_{build_sdfg.__name__}')
    assert InsertAssignTaskletsAtMapBoundary().apply_pass(sdfg, {}) == 1
    assert any(isinstance(n, nodes.Tasklet) for n, _ in sdfg.all_nodes_recursive())
    sdfg.validate()
    np.testing.assert_array_equal(run(sdfg), expected())


@pytest.mark.parametrize('build_sdfg', ORIENTATIONS)
def test_buffer_expansion_arr_subset_separates_the_two_ends(build_sdfg):
    """``BufferExpansion._arr_subset`` reports the region an AccessNode touches on an incident edge.

    On a self-copy the source node's out-edge and the destination node's in-edge are the SAME edge, so
    a name-based lookup returns one region for both -- the read region then counts as a write, and a
    loop-carried level shift ``p[k] = p[k-1]`` looks like a write that covers its own read.
    """
    sdfg = build_sdfg(f'buffer_expansion_{build_sdfg.__name__}')
    state = sdfg.states()[0]
    edge = state.edges()[0]

    read = BufferExpansion._arr_subset(state, edge, edge.src)
    write = BufferExpansion._arr_subset(state, edge, edge.dst)
    assert str(read) == f'{READ[0]}, {READ[1]}', f'source node reads {read}'
    assert str(write) == f'{WRITE[0]}, {WRITE[1]}', f'destination node writes {write}'
    assert str(read) != str(write)


@pytest.mark.parametrize('build_sdfg', ORIENTATIONS)
def test_an_side_subset_separates_the_two_ends(build_sdfg):
    """``an_side_subset`` / ``infer_edge_endpoints`` are the vectorization pipeline's endpoint
    readers. Selecting by ``memlet.data == an.data`` matches on BOTH ends of a self-copy, so both
    ends got the same region and ``infer_edge_endpoints`` reported ``src == dst`` -- wrong in either
    orientation, not just the dst-relative one."""
    from dace.transformation.passes.vectorization.utils.subsets import an_side_subset, infer_edge_endpoints

    sdfg = build_sdfg(f'an_side_{build_sdfg.__name__}')
    state = sdfg.states()[0]
    edge = state.edges()[0]

    assert str(an_side_subset(edge, edge.src, sdfg, state)) == f'{READ[0]}, {READ[1]}'
    assert str(an_side_subset(edge, edge.dst, sdfg, state)) == f'{WRITE[0]}, {WRITE[1]}'

    src_data, src_subset, dst_data, dst_subset = infer_edge_endpoints(edge, sdfg, state)
    assert src_data == dst_data == 'p'
    assert str(src_subset) == f'{READ[0]}, {READ[1]}'
    assert str(dst_subset) == f'{WRITE[0]}, {WRITE[1]}'
    assert str(src_subset) != str(dst_subset), 'a self-copy reads and writes different elements'


@pytest.mark.parametrize('build_sdfg', ORIENTATIONS)
def test_canonicalize_node_side_subset_separates_the_two_ends(build_sdfg):
    """``LoopToSymmetrize`` / ``LoopToTranspose`` share a ``_node_side_subset`` helper that resolves
    an endpoint's region on a copy edge. ``LoopToSymmetrize`` matches in-place transposed self-copies
    by construction, and it derives ``source_upper`` -- which triangle survives -- from which of the
    two regions is the read. Swapping them mirrors the wrong triangle.
    """
    from dace.transformation.passes.canonicalize.loop_to_symmetrize import _node_side_subset as symmetrize_side
    from dace.transformation.passes.canonicalize.loop_to_transpose import _node_side_subset as transpose_side

    sdfg = build_sdfg(f'node_side_{build_sdfg.__name__}')
    state = sdfg.states()[0]
    edge = state.edges()[0]

    for side in (symmetrize_side, transpose_side):
        assert str(side(state, edge, edge.src)) == f'{READ[0]}, {READ[1]}'
        assert str(side(state, edge, edge.dst)) == f'{WRITE[0]}, {WRITE[1]}'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
