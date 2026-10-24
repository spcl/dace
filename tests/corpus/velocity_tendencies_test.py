# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""ICON's velocity tendencies is parallel at every level and in every column: what is left as a loop after lifting is
pinned here, once after ``simplify`` + ``ParallelizeLoops`` and once after the whole canonicalization."""

from collections import Counter

import pytest

from dace.sdfg.state import LoopRegion
from dace.transformation.passes.analysis import loop_analysis
from dace.transformation.passes.canonicalize.pipeline import canonicalize
from dace.transformation.passes.parallelize_loops import ParallelizeLoops
from tests.corpus.velocity_tendencies.velocity_tendencies import velocity_tendencies


def loops(sdfg) -> list[LoopRegion]:
    return [r for sd in sdfg.all_sdfgs_recursive() for r in sd.all_control_flow_regions() if isinstance(r, LoopRegion)]


def test_parallelize_loops_leaves_only_the_reductions():
    """Every level and column loop lifts. Left: the neighbour-slot accumulations (``acc += w[n] * x[nbr[n]]`` over
    3, 4 or 6 slots) and the inner loops of the ``levelmask`` any-reduction, which need a reduction, not a map."""
    sdfg = velocity_tendencies.to_sdfg(simplify=True)
    assert ParallelizeLoops().apply_pass(sdfg, {})
    left = loops(sdfg)
    assert Counter(r.loop_variable for r in left) == Counter({"n": 6, "__rd0_j0": 1, "__rd0_j2": 1}), [
        r.label for r in left
    ]


def test_canonicalize_leaves_only_short_constant_loops():
    """After canonicalization the reductions are lifted too; what stays a loop is a neighbour-slot accumulation over
    at most six slots, which canonicalization keeps sequential by design."""
    sdfg = velocity_tendencies.to_sdfg(simplify=False)
    canonicalize(sdfg)
    for loop in loops(sdfg):
        start, end = loop_analysis.get_init_assignment(loop), loop_analysis.get_loop_end(loop)
        assert start is not None and end is not None and (end - start).is_Integer and end - start < 6, loop.label


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
