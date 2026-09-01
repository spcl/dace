# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Wall-clock regression guard on simplifying the CloudSC corpus kernel.

CloudSC is the scaling case: thousands of blocks nested many levels deep, where a pass whose cost
is superlinear in nesting depth shows up as minutes rather than seconds. ConstantPropagation is the
dominant term in ``simplify`` on this input.
"""

import statistics
import time

import pytest

import dace
from tests.corpus.cloudsc.generate_data_for_cloudsc import build_cloudsc_sdfg

#: Wall-clock budget for one ``simplify()`` of CloudSC.
SIMPLIFY_BUDGET_SECONDS: float = 140.0
SIMPLIFY_REPS: int = 5


@pytest.mark.long
def test_build_cloudsc_sdfg_hands_out_private_copies():
    """The corpus memoizes the parse; every caller must still get a copy it can transform."""
    first = build_cloudsc_sdfg(simplify=False)
    second = build_cloudsc_sdfg(simplify=False)
    assert first is not second
    first.add_symbol('canary', dace.int32)
    assert 'canary' not in second.symbols


@pytest.mark.long
def test_simplify_stays_within_its_time_budget():
    """One run per copy: ``simplify`` is idempotent, so a second run on the same SDFG measures
    nothing. The median absorbs a slow rep from a loaded machine without needing a wide margin.
    """
    durations = []
    for _ in range(SIMPLIFY_REPS):
        sdfg = build_cloudsc_sdfg(simplify=False)  # deliberately outside the timer
        start = time.perf_counter()
        sdfg.simplify()
        durations.append(time.perf_counter() - start)

    median = statistics.median(durations)
    reps = ', '.join('%.1f' % d for d in durations)
    assert median < SIMPLIFY_BUDGET_SECONDS, (
        f'median simplify took {median:.1f}s, budget is {SIMPLIFY_BUDGET_SECONDS:.0f}s; reps={reps}'
    )


if __name__ == '__main__':
    test_build_cloudsc_sdfg_hands_out_private_copies()
    test_simplify_stays_within_its_time_budget()
