# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Carriers whose dependence does not live on every axis, and reads that arrive as ranges.

``WavefrontSkew`` used to require a POINT access on a 2-D carrier, which put the whole
row-granularity family out of reach: a sweep ``A[i, 1:N-1] = f(A[i-1, 1:N-1], ...)`` touches a
span on the column axis, and memlet consolidation folds its three neighbour reads into the single
``A[i-1 : i+2, 1:N-1]``. Two steps recover it -- drop the axis every access spans identically
(:func:`uniform_axes`), then expand what is left back into the points it covers
(:func:`reduced_points`) -- after which the carrier has ONE index and the distances come from
program order rather than from inverting a write map.

These are the unit-level contracts. The end-to-end skew, and the exact ``tau = (2, 1)`` it must
find, are pinned in ``tests/canonicalize/canonicalize_wavefront_skew_test.py``.
"""
import pytest

from dace import symbolic
from dace.transformation.passes.canonicalize.wavefront_skew import (MAX_RANGE_READ_WIDTH, axis_distance, axis_write_map,
                                                                    reduced_points, uniform_axes)

U, V, N = 't', 'i', symbolic.pystr_to_symbolic('N')
T, I = symbolic.pystr_to_symbolic('t'), symbolic.pystr_to_symbolic('i')
ITERS = (U, V)


def span(lo, hi):
    return (symbolic.pystr_to_symbolic(str(lo)), symbolic.pystr_to_symbolic(str(hi)))


def test_a_span_every_access_shares_is_dropped():
    """The row sweep's column axis: the same ``1:N-1`` in the write and in every read, and free of
    ``t`` and ``i``. Two iterations that touch a common cell agree on it by construction, so it
    carries nothing and the distance lives entirely in the row index."""
    write = [span(I, I), span(1, N - 2)]
    read = [span(I - 1, I + 1), span(1, N - 2)]
    assert uniform_axes([write, read], ITERS) == [1]


def test_an_axis_the_accesses_disagree_on_is_kept():
    """A range the accesses do NOT share is a real span of distances -- keeping it is what lets
    :func:`reduced_points` decide whether it can be enumerated or the nest must be refused."""
    write = [span(I, I), span(0, 2)]
    read = [span(I, I), span(1, 3)]
    assert uniform_axes([write, read], ITERS) == []


def test_a_span_that_moves_with_the_nest_is_kept():
    """``A[i, 0:i]`` spans a different set of columns each iteration, so the overlap between two
    iterations is real. Dropping it would hide exactly the dependence being looked for."""
    moving = [span(I, I), span(0, I)]
    assert uniform_axes([moving, moving], ITERS) == []


def test_a_constant_width_range_expands_into_its_points():
    """Consolidation turns the three neighbour reads into one memlet; the expansion is what puts
    the three distances back. Order is not asserted -- the caller consumes them as a set -- but
    the exact three values are."""
    read = [span(I - 1, I + 1), span(1, N - 2)]
    points = reduced_points(read, [1])
    assert points is not None
    assert sorted(str(symbolic.simplify(p[0] - I)) for p in points) == ['-1', '0', '1']
    assert all(len(p) == 1 for p in points), points


def test_a_symbolic_or_oversized_range_refuses():
    """A range whose width is not a small constant has no finite reading, so there is no set of
    distances to reason about and the nest must be refused rather than approximated."""
    assert reduced_points([span(I, I + N)], []) is None
    assert reduced_points([span(I, I + MAX_RANGE_READ_WIDTH + 1)], []) is None
    assert reduced_points([span(I, I + MAX_RANGE_READ_WIDTH)], []) is not None


def test_dropping_every_axis_refuses():
    """A carrier written wholesale each iteration has no index left to carry a distance."""
    assert reduced_points([span(1, N - 2)], [0]) is None


def test_axis_write_map_admits_only_an_outer_independent_write():
    """``A[i]`` written on every ``t`` is the admissible shape. A write that mentions ``t`` is
    excluded because it has an exact 2-D inverse and belongs on the other path; two writes that
    disagree are excluded because no single map describes them."""
    assert axis_write_map([[I]], U, V) == (1, symbolic.pystr_to_symbolic('0'))
    assert axis_write_map([[I + 2]], U, V) == (1, symbolic.pystr_to_symbolic('2'))
    assert axis_write_map([[I + T]], U, V) is None
    assert axis_write_map([[I], [I + 1]], U, V) is None
    assert axis_write_map([[2 * I]], U, V) is None, 'a non-unit coefficient has no integer inverse'


@pytest.mark.parametrize('offset, expected', [
    (-2, (0, -2)),
    (-1, (0, -1)),
    (0, (-1, 0)),
    (1, (-1, 1)),
])
def test_axis_distance_is_the_lexicographic_predecessor(offset, expected):
    """Rows below the current one were written earlier in THIS sweep, so the distance stays inside
    it (``du = 0``). The current row and any row above it are written at or after this iteration,
    so the value read is the previous sweep's (``du = -1``). Getting this backwards would let the
    time axis be handed out in parallel."""
    coeff, const = axis_write_map([[I]], U, V)
    got = axis_distance(I + offset, coeff, const, U, V)
    assert got is not None
    assert (int(got[0]), int(got[1])) == expected


def test_axis_distance_refuses_an_undecidable_offset():
    """An offset whose sign is unknown would have to guess which of the two answers applies, and
    guessing wrong mis-orders the schedule."""
    coeff, const = axis_write_map([[I]], U, V)
    assert axis_distance(I + symbolic.pystr_to_symbolic('K'), coeff, const, U, V) is None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
