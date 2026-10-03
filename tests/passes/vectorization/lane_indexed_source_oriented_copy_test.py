# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A lane-indexed transient stays recognizable when a SOURCE-oriented copy writes it.

``WidenAccesses`` widens a lane-dependent transient into a per-lane tile of shape ``widths``, which
only a scalar-like buffer supports; a transient that already carries the lane axis in its own
indexing (CloudSC's ``zsolqa[jm, jn, jl]``, shape ``(5, 5, klon)``) is exempted by
:func:`data_is_lane_indexed` because widening it would add a SECOND lane axis. That test matched
edges on ``edge.data.data``, but an AN-to-AN copy names whichever endpoint the memlet is ORIENTED
on: a copy built from the source names the source, so every access to the transient it writes was
invisible and the exemption never fired -- the pass then refused the kernel outright with
``NotImplementedError: ... has non-scalar shape``.
"""
import dace
import pytest
from dace.transformation.passes.vectorization.utils.tile_access import data_is_lane_indexed
from dace.transformation.passes.vectorization.widen_accesses import WidenAccesses

N = dace.symbol('N')


def copy_into_lane_indexed_transient(orientation: str) -> dace.SDFG:
    """``A[1, 1, i] -> buf[1, 1, i]`` as one AN-to-AN copy, written with the given orientation.

    :param orientation: ``'source'`` names ``A`` in the memlet, ``'destination'`` names ``buf``.
        Both spell the same copy -- only which side ``subset`` describes differs.
    :returns: the SDFG.
    """
    sdfg = dace.SDFG(f'lane_indexed_{orientation}')
    sdfg.add_array('A', [5, 5, N], dace.float64)
    sdfg.add_transient('buf', [5, 5, N], dace.float64)
    state = sdfg.add_state('block', is_start_block=True)
    named = 'A' if orientation == 'source' else 'buf'
    state.add_edge(state.add_read('A'), None, state.add_write('buf'), None,
                   dace.Memlet(data=named, subset='1, 1, i', other_subset='1, 1, i'))
    return sdfg


@pytest.mark.parametrize('orientation', ['destination', 'source'])
def test_lane_indexing_is_seen_through_either_orientation(orientation):
    """``buf`` is indexed by the tile iter-var however the copy that writes it is spelled."""
    sdfg = copy_into_lane_indexed_transient(orientation)
    assert data_is_lane_indexed(sdfg, 'buf', ('i', )) is True


@pytest.mark.parametrize('orientation', ['destination', 'source'])
def test_lane_indexed_transient_is_not_refused(orientation):
    """The propagation exempts it instead of raising, and leaves it out of the widening set.

    Being absent from the returned set is the point: a name in it gets its descriptor swapped for
    a per-lane tile, which is exactly the wrong treatment for a buffer that already spans the lane
    axis.
    """
    sdfg = copy_into_lane_indexed_transient(orientation)
    widened = WidenAccesses()._propagate_lane_dep(sdfg, ('i', ), {'A'})
    assert 'buf' not in widened


def test_constant_indexed_transient_is_still_refused():
    """The exemption must stay narrow: a genuine per-lane buffer indexed by constants still refuses.

    Without this the fix would read as "never refuse", and a real multi-element per-lane window
    would be silently left under-widened rather than reported.
    """
    sdfg = dace.SDFG('per_lane_window')
    sdfg.add_array('A', [5, 5, N], dace.float64)
    sdfg.add_transient('win', [2], dace.float64)
    state = sdfg.add_state('block', is_start_block=True)
    state.add_edge(state.add_read('A'), None, state.add_write('win'), None,
                   dace.Memlet(data='A', subset='1, 1, i', other_subset='0:2'))

    with pytest.raises(NotImplementedError, match='non-scalar shape'):
        WidenAccesses()._propagate_lane_dep(sdfg, ('i', ), {'A'})


def scalar_staging_copy(orientation: str) -> dace.SDFG:
    """``A[i] -> s[0]`` as one AN-to-AN copy, written with the given orientation.

    ``s`` is scalar-like, so it IS a genuine per-lane buffer: ``_widen_transient`` swaps its
    descriptor for a ``(W,)`` tile and must rewrite the copy's ``s`` side to match.

    :param orientation: which endpoint the memlet names in ``data``.
    :returns: the SDFG.
    """
    sdfg = dace.SDFG(f'staging_{orientation}')
    sdfg.add_array('A', [N], dace.float64)
    sdfg.add_transient('s', [1], dace.float64)
    state = sdfg.add_state('block', is_start_block=True)
    source_named = orientation == 'source'
    state.add_edge(
        state.add_read('A'), None, state.add_write('s'), None,
        dace.Memlet(data='A' if source_named else 's',
                    subset='i' if source_named else '0',
                    other_subset='0' if source_named else 'i'))
    return sdfg


@pytest.mark.parametrize('orientation', ['destination', 'source'])
def test_widened_transient_side_matches_its_new_descriptor(orientation):
    """The ``s`` side of the copy becomes the tile range however the copy is spelled.

    Left stale, ``s`` reads ``[0]`` -- one element -- against a descriptor that is now ``(W,)``,
    which ``validate`` rejects as a dimensionality mismatch and codegen would copy one lane of W.
    """
    sdfg = scalar_staging_copy(orientation)
    assert WidenAccesses(widths=(8, ))._widen_transient(sdfg, 's', {'s'}) is True

    edge = next(iter(next(iter(sdfg.states())).edges()))
    s_side = edge.data.subset if edge.data.data == 's' else edge.data.other_subset
    assert str(s_side) == '0:8', f"the widened side still reads {s_side}"
    assert edge.data.volume == 8


def test_the_other_arrays_side_is_left_alone():
    """Widening ``s`` must not rewrite ``A[i]``: that side is the other array's own region.

    Overwriting it with the tile range turns ``A[i]`` into ``A[0:W]`` -- every lane reading from
    the array's head instead of its own element, which is silently wrong rather than invalid.
    """
    sdfg = scalar_staging_copy('source')
    WidenAccesses(widths=(8, ))._widen_transient(sdfg, 's', {'s'})

    edge = next(iter(next(iter(sdfg.states())).edges()))
    assert str(edge.data.subset) == 'i', f"the A side was rewritten to {edge.data.subset}"


def test_a_scalar_like_buffer_is_never_lane_indexed():
    """A shape-``(1,)`` buffer holds one value, so it has no lane axis to carry.

    It is a per-lane buffer whose DESCRIPTOR gets swapped for a ``(W,)`` tile. Calling it
    lane-indexed instead seeds it for in-place widening: the memlets go to ``[0:W]`` while the
    descriptor stays ``(1,)``, which is the "Memlet subset out-of-bounds" TSVC-2.5
    ``scan_conditional`` hits on its ``_then__scan_in_out`` arm temp. The copy below is indexed by
    the tile iter-var on its SOURCE side, so the orientation-aware lookup does reach a subset
    naming ``i`` -- the descriptor is what settles it.
    """
    sdfg = scalar_staging_copy('source')
    assert data_is_lane_indexed(sdfg, 's', ('i', )) is False
    # The multi-element array the exemption exists for still answers True.
    assert data_is_lane_indexed(copy_into_lane_indexed_transient('source'), 'buf', ('i', )) is True


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
