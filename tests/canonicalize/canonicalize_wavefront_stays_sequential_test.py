# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Rebuilding scope summaries must not buy parallelism the loop is not entitled to.

Canonicalize re-propagates the memlets before every ``LoopToMap`` stage, because an inline leaves
the enclosing scope claiming the whole array and a whole-array write is refused unconditionally.
The refusal was therefore doing two jobs at once, and tightening the summaries took one away: a
wavefront body writing ``a[i, j]`` propagates to ``a[0:i+1, 0:N-1]``, whose first dimension starts
at the literal ``0``, and ``LoopToMap`` used to read a 0-based dimension off its UPPER bound and
call it uniquely indexed by ``i``. tsvc_2_5 ``wf_diff_skew`` was lifted and produced wrong numbers.

The loop below carries ``(1, 0)`` and ``(1, -1)``: row ``i`` reads row ``i-1``. It is a genuine
wavefront and no amount of memlet precision makes it a DOALL.
"""
import numpy as np

import dace
from dace.sdfg.state import LoopRegion
from dace.transformation.passes.analysis import loop_analysis
from dace.transformation.passes.canonicalize import canonicalize

N = dace.symbol('N')


@dace.program
def wavefront_skew(a: dace.float64[N, N]):
    for i in range(1, N):
        for j in range(0, N - 1):
            a[i, j] = a[i, j] + a[i - 1, j] + a[i - 1, j + 1]


def residual_loops(sdfg: dace.SDFG):
    return [c for c in sdfg.all_control_flow_regions(recursive=True) if isinstance(c, LoopRegion)]


def test_the_carrying_loop_survives_canonicalization():
    """Exactly one loop is left, and it is the outer one -- the inner ``j`` sweep is parallel and
    must still lift, so "a loop survived" alone would not say the right thing happened."""
    sdfg = wavefront_skew.to_sdfg(simplify=True)
    sdfg.simplify()
    canonicalize(sdfg, validate_all=False)
    loops = residual_loops(sdfg)
    assert len(loops) == 1, f'expected the wavefront axis to stay sequential, got {[c.label for c in loops]}'
    start = loop_analysis.get_init_assignment(loops[0])
    end = loop_analysis.get_loop_end(loops[0])
    assert (str(start), str(end)) == ('1', 'N - 1'), f'the surviving loop is not the outer sweep: {start}..{end}'


def test_the_wavefront_keeps_its_values():
    """The dependence is diagonal, so a wrong lift is invisible on one thread; the corpus caught
    this at a max difference of 1.7e10, not at a rounding error. Run it wide."""
    n = 32
    a = np.arange(n * n, dtype=np.float64).reshape(n, n) % 7 + 1.0
    ref = a.copy()
    for i in range(1, n):
        for j in range(0, n - 1):
            ref[i, j] = ref[i, j] + ref[i - 1, j] + ref[i - 1, j + 1]

    sdfg = wavefront_skew.to_sdfg(simplify=True)
    sdfg.simplify()
    canonicalize(sdfg, validate_all=False)
    got = a.copy()
    sdfg(a=got, N=n)
    assert np.allclose(got, ref)


if __name__ == '__main__':
    test_the_carrying_loop_survives_canonicalization()
    test_the_wavefront_keeps_its_values()
