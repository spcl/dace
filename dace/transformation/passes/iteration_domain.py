# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Iteration-domain helpers shared by the fusion passes: exact trip counts and unit-step alignment."""
import copy
from typing import Any, Optional, Sequence, Tuple

from dace import subsets, symbolic
from dace.sdfg import nodes
from dace.sdfg.state import LoopRegion, SDFGState
from dace.transformation.passes.analysis import loop_analysis

RangeTriple = Tuple[Any, Any, Any]


def constant_int(value: Any) -> Optional[int]:
    """``value`` as a Python int when it is an integer constant, else ``None``."""
    if symbolic.issymbolic(value):
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number == value else None


def exact_trip_count(begin: Any, end: Any, step: Any) -> Any:
    """Iteration count of the inclusive range ``begin:end:step``.

    With a positive constant ``step``, ``int_floor(end - begin, step)`` is folded exactly over integer
    symbols, so ``0:2N-1:2`` and ``0:N-1:1`` both give ``N``.
    """
    span = end - begin
    stride = constant_int(step)
    if stride is None or stride <= 0:
        return symbolic.simplify(symbolic.int_floor(span, step) + 1)
    if not symbolic.issymbolic(span):
        return int(span) // stride + 1
    quotient: Any = 0
    remainder = 0
    for term, coefficient in symbolic.simplify(span).expand().as_coefficients_dict().items():
        factor = constant_int(coefficient)
        if factor is not None and term == 1:
            remainder += factor
        elif factor is not None and factor % stride == 0:
            quotient += (factor // stride) * term
        else:
            return symbolic.simplify(symbolic.int_floor(span, step) + 1)
    return symbolic.simplify(quotient + remainder // stride + 1)


def same_trip_count(first: Any, second: Any) -> bool:
    return symbolic.simplify(first - second) == 0


def equal_trip_counts(first: Sequence[RangeTriple], second: Sequence[RangeTriple]) -> bool:
    """Whether two ranges have the same number of dimensions and the same trip count in each."""
    return len(first) == len(second) and all(
        same_trip_count(exact_trip_count(*a), exact_trip_count(*b)) for a, b in zip(first, second))


def loop_trip_count(loop: LoopRegion) -> Optional[Any]:
    """``loop``'s iteration count, or ``None`` when its header is not an affine counter."""
    start = loop_analysis.get_init_assignment(loop)
    end = loop_analysis.get_loop_end(loop)
    stride = loop_analysis.get_loop_stride(loop)
    if start is None or end is None or stride is None or constant_int(stride) == 0:
        return None
    return exact_trip_count(start, end, stride)


def step_equivalent_maps(first: nodes.MapEntry, second: nodes.MapEntry) -> bool:
    """Whether two maps differ only in step: same begins and trip counts, different ranges."""
    first_ranges = list(first.map.range.ranges)
    second_ranges = list(second.map.range.ranges)
    if first_ranges == second_ranges or len(first_ranges) != len(second_ranges):
        return False
    if any(not same_trip_count(a[0], b[0]) for a, b in zip(first_ranges, second_ranges)):
        return False
    return equal_trip_counts(first_ranges, second_ranges)


def align_maps_to_unit_step(state: SDFGState, first: nodes.MapEntry, second: nodes.MapEntry) -> bool:
    """Rewrite two step-equivalent maps onto one ``0:trip-1:1`` range so map fusion sees equal ranges.

    Each non-canonical parameter ``p`` becomes ``begin + step * p`` inside its own scope, so every
    subscript stays affine (``a[2*t]``). Returns ``False`` without touching either map when the maps
    are not step-equivalent.
    """
    if not step_equivalent_maps(first, second):
        return False
    # Deferred: the canonicalize package imports the fusion passes that import this module.
    from dace.transformation.passes.canonicalize.normalize_loops_and_maps import NormalizeLoopsAndMaps
    trips = [exact_trip_count(*rng) for rng in first.map.range.ranges]
    normalizer = NormalizeLoopsAndMaps()
    normalizer._normalize_map(state, first)
    normalizer._normalize_map(state, second)
    unit = subsets.Range([(0, symbolic.simplify(trip - 1), 1) for trip in trips])
    first.map.range = unit
    second.map.range = copy.deepcopy(unit)
    return True
