# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Pins the invariant behind a whole class of silent-miscompile bugs.

A copy memlet stores one endpoint's region in ``subset`` and the other's in ``other_subset``.
**Which** endpoint ``subset`` names is carried by the memlet's own ``_is_data_src`` flag; it is NOT
derivable from the endpoint data names. On a self-copy (``A -> A``, e.g. the level shift
``p[jk] = p[jk-1]``) the names cannot disambiguate at all: both ends match ``memlet.data``.

Consequence: any code that picks a side by comparing data names silently swaps source and
destination on half of all self-copies. ``Memlet.get_src_subset`` / ``Memlet.get_dst_subset`` are the
only correct readers, and this file asserts that -- against the values the compiled program actually
produces, not against the memlet's own bookkeeping.
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

READ_COL = 4
WRITE_COL = 3


def build_self_copy(name: str, memlet: dace.Memlet):
    """``p`` (4x5) with a single ``AccessNode -> AccessNode`` edge on the SAME array carrying
    ``memlet``. Returns ``(sdfg, edge, state)``."""
    sdfg = dace.SDFG(name)
    sdfg.add_array('p', [4, 5], dace.float64)
    state = sdfg.add_state('s')
    edge = state.add_edge(state.add_access('p'), None, state.add_access('p'), None, memlet)
    return sdfg, edge, state


def src_relative_self_copy(name: str):
    """``subset`` = the READ column, ``other_subset`` = the WRITTEN one.

    This is the orientation ``Memlet.try_initialize`` assigns by default when both endpoints name the
    same array, and the one the DaCe Python frontend emits.
    """
    return build_self_copy(name, dace.Memlet(data='p', subset=f'0:4, {READ_COL}', other_subset=f'0:4, {WRITE_COL}'))


def dst_relative_self_copy(name: str):
    """The SAME copy spelled destination-relative: ``subset`` = the WRITTEN column.

    ``src -> dst`` memlet syntax with an anonymous source picks ``data`` from the destination, which
    sets ``_is_data_src = False``. ``try_initialize`` preserves an already-set flag, so this survives
    being attached to a ``p -> p`` edge.
    """
    return build_self_copy(name, dace.Memlet(f'[0:4, {READ_COL}] -> p[0:4, {WRITE_COL}]'))


def run(sdfg: dace.SDFG) -> np.ndarray:
    """Run ``sdfg`` on a seeded 4x5 buffer and return it."""
    p = np.arange(20, dtype=np.float64).reshape(4, 5).copy()
    sdfg(p=p)
    return p


def expected() -> np.ndarray:
    """``p[:, WRITE_COL] = p[:, READ_COL]`` on the seeded buffer."""
    p = np.arange(20, dtype=np.float64).reshape(4, 5).copy()
    p[:, WRITE_COL] = p[:, READ_COL]
    return p


@pytest.mark.parametrize('build', [src_relative_self_copy, dst_relative_self_copy])
def test_accessors_name_the_ends_a_name_test_cannot(build):
    """``get_src_subset`` / ``get_dst_subset`` return the read and written columns in BOTH
    orientations; the name-based selection ``subset if data == node.data else other_subset`` returns
    the same subset for both ends and is therefore wrong for one of them."""
    sdfg, edge, state = build(sdfg_name_of(build))
    memlet = edge.data

    assert str(memlet.get_src_subset(edge, state)) == f'0:4, {READ_COL}'
    assert str(memlet.get_dst_subset(edge, state)) == f'0:4, {WRITE_COL}'

    # What a name test would answer: both endpoints match ``memlet.data``, so both get ``subset``.
    assert edge.src.data == memlet.data and edge.dst.data == memlet.data
    name_based_src = memlet.subset if memlet.data == edge.src.data else memlet.other_subset
    name_based_dst = memlet.subset if memlet.data == edge.dst.data else memlet.other_subset
    assert str(name_based_src) == str(name_based_dst), 'a name test cannot tell a self-copy\'s ends apart'
    assert str(name_based_src) != str(memlet.get_dst_subset(edge, state)) or \
        str(name_based_dst) != str(memlet.get_src_subset(edge, state)), \
        'the name test must disagree with the accessors on at least one end'


def sdfg_name_of(build) -> str:
    """Distinct SDFG name per orientation (build folders must not collide)."""
    return f'self_copy_{build.__name__}'


@pytest.mark.parametrize('build', [src_relative_self_copy, dst_relative_self_copy])
def test_compiled_copy_follows_the_accessors(build):
    """The generated code moves data in the direction ``get_src_subset`` / ``get_dst_subset`` name.

    This is what makes the invariant testable rather than tautological: both orientations describe the
    SAME copy, so both must produce the same buffer.
    """
    sdfg, _edge, _state = build(sdfg_name_of(build))
    np.testing.assert_array_equal(run(sdfg), expected())


def test_both_orientations_are_the_same_program():
    """The two spellings differ only in bookkeeping -- their compiled results must be identical."""
    np.testing.assert_array_equal(run(src_relative_self_copy('self_copy_pair_src')[0]),
                                  run(dst_relative_self_copy('self_copy_pair_dst')[0]))


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
