# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for NormalizeScheduleForLayout: after a Block lays an array out as [.., N/b, .., b], the
schedule is re-tiled by b so the innermost loop iterates the block. Oracle: the tiled schedule
computes the same result (bit-exact), tiling is idempotent, and an unblocked kernel is untouched."""
import numpy
import dace

from dace.transformation.layout.split_dimensions import SplitDimensions
from dace.transformation.layout.normalize_schedule import normalize_schedule_for_layout, NormalizeScheduleForLayout

N = dace.symbol("N")


@dace.program
def ew1d(A: dace.float64[N], C: dace.float64[N]):
    for i in dace.map[0:N] @ dace.ScheduleType.Sequential:
        C[i] = A[i] * 2.0 + 1.0


@dace.program
def ew2d(A: dace.float64[N, N], C: dace.float64[N, N]):
    for i, j in dace.map[0:N, 0:N] @ dace.ScheduleType.Sequential:
        C[i, j] = A[i, j] * 3.0


def _num_maps(sdfg):
    return sum(1 for st in sdfg.all_states() for n in st.nodes() if isinstance(n, dace.nodes.MapEntry))


def test_normalize_tiles_by_block_width_bitexact():
    """Block A and C by 4, then normalize: the single map is tiled (one nested map added) and the
    result is unchanged."""
    sdfg = ew1d.to_sdfg(simplify=True)
    SplitDimensions(split_map={"A": ([True], [4]), "C": ([True], [4])}).apply_pass(sdfg, {})
    before = _num_maps(sdfg)
    assert normalize_schedule_for_layout(sdfg, divides_evenly=True) == 1
    assert _num_maps(sdfg) == before + 1
    sdfg.validate()

    _N = 16
    A = numpy.random.rand(_N // 4, 4)
    C = numpy.zeros((_N // 4, 4))
    sdfg(A=A, C=C, N=_N)
    assert numpy.allclose(C, A * 2.0 + 1.0)


def test_normalize_idempotent():
    """A second normalize pass tiles nothing (the block signal is a point access, absent on the
    already-tiled map)."""
    sdfg = ew1d.to_sdfg(simplify=True)
    SplitDimensions(split_map={"A": ([True], [4]), "C": ([True], [4])}).apply_pass(sdfg, {})
    assert normalize_schedule_for_layout(sdfg, divides_evenly=True) == 1
    assert normalize_schedule_for_layout(sdfg, divides_evenly=True) == 0


def test_normalize_skips_unblocked():
    """An unblocked kernel has no block-width signal, so no map is tiled."""
    sdfg = ew1d.to_sdfg(simplify=True)
    assert normalize_schedule_for_layout(sdfg) == 0


def test_normalize_2d_both_dims_bitexact():
    """A 2D map whose both dimensions are blocked by 4 is tiled by (4, 4)."""
    sdfg = ew2d.to_sdfg(simplify=True)
    SplitDimensions(split_map={"A": ([True, True], [4, 4]), "C": ([True, True], [4, 4])}).apply_pass(sdfg, {})
    before = _num_maps(sdfg)
    assert normalize_schedule_for_layout(sdfg, divides_evenly=True) == 1
    assert _num_maps(sdfg) == before + 1
    sdfg.validate()

    _N = 8
    # Blocked layout [N/4, N/4, 4, 4]; build a contiguous array of that shape.
    A = numpy.random.rand(_N // 4, _N // 4, 4, 4)
    C = numpy.zeros((_N // 4, _N // 4, 4, 4))
    sdfg(A=A, C=C, N=_N)
    assert numpy.allclose(C, A * 3.0)


def test_normalize_partial_block_not_tiled():
    """A map with only ONE of two dims blocked is not uniformly blocked, so it is left alone (this
    version tiles only maps where every dimension shares a block width)."""
    sdfg = ew2d.to_sdfg(simplify=True)
    SplitDimensions(split_map={"A": ([True, False], [4, 1]), "C": ([True, False], [4, 1])}).apply_pass(sdfg, {})
    assert normalize_schedule_for_layout(sdfg, divides_evenly=True) == 0


if __name__ == "__main__":
    test_normalize_tiles_by_block_width_bitexact()
    test_normalize_idempotent()
    test_normalize_skips_unblocked()
    test_normalize_2d_both_dims_bitexact()
    test_normalize_partial_block_not_tiled()
    print("normalize_schedule tests PASS")
