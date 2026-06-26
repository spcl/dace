# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for :class:`InitializeReductionAccumulators` and the WCR-accumulator-init bug.

The bug: a reduction ``acc[i] += f(i, j)`` into a *fresh* transient accumulator
(written only by the WCR) is never initialized to the reduction identity. Codegen
allocates the transient uninitialized and accumulates into it, so the reduction reads
garbage. It is masked when the allocation returns zeroed pages, but breaks for a
``Persistent`` accumulator reused across calls (the second call accumulates on top of
the first).
"""
import numpy as np

import dace
from dace import dtypes
from dace.sdfg import nodes
from dace.transformation.passes.initialize_reduction_accumulators import InitializeReductionAccumulators

N = dace.symbol('N')


@dace.program
def _reduce_rows(B: dace.float64[N, N], out: dace.float64[N]):
    acc = dace.define_local([N], dace.float64)

    @dace.map
    def red(i: _[0:N], j: _[0:N]):
        b << B[i, j]
        o >> acc(1, lambda x, y: x + y)[i]  # WCR sum into fresh transient
        o = b

    out[:] = acc


@dace.program
def _seeded(B: dace.float64[N, N], C: dace.float64[N]):

    @dace.map
    def seed(i: _[0:N]):
        o >> C[i]
        o = 100.0  # plain seed write

    @dace.map
    def red(i: _[0:N], j: _[0:N]):
        b << B[i, j]
        o >> C(1, lambda x, y: x + y)[i]
        o = b


def _fresh_reduction_sdfg() -> dace.SDFG:
    """``out = sum_j B[:, j]`` via a WCR into a fresh transient ``acc``."""
    return _reduce_rows.to_sdfg(simplify=False)


def _has_init_for(sdfg: dace.SDFG, arr: str) -> bool:
    """True iff some state writes the FULL ``arr`` with a non-WCR (plain) map write."""
    for state in sdfg.states():
        for node in state.nodes():
            if isinstance(node, nodes.AccessNode) and node.data == arr:
                for e in state.in_edges(node):
                    if e.data is not None and e.data.wcr is None and e.data.data == arr:
                        return True
    return False


def test_pass_adds_init_for_fresh_accumulator():
    sdfg = _fresh_reduction_sdfg()
    # The fresh accumulator transient (named via define_local) is written only by WCR.
    acc = next(n for n in sdfg.arrays if n not in ('B', 'out'))
    assert not _has_init_for(sdfg, acc), "precondition: fresh accumulator has no init"

    InitializeReductionAccumulators().apply_pass(sdfg, {})

    assert _has_init_for(sdfg, acc), "pass must insert an explicit identity-init for the fresh accumulator"
    sdfg.validate()


def test_pass_skips_seeded_accumulator():
    """An accumulator that already has a plain seed write must NOT be re-initialized."""
    sdfg = _seeded.to_sdfg(simplify=False)
    before = sum(1 for s in sdfg.states() for n in s.nodes() if isinstance(n, nodes.MapEntry))
    InitializeReductionAccumulators().apply_pass(sdfg, {})
    after = sum(1 for s in sdfg.states() for n in s.nodes() if isinstance(n, nodes.MapEntry))
    # C is already seeded (plain write) -> not "fresh" -> no init map added.
    assert after == before


def test_persistent_accumulator_reused_across_calls():
    """A Persistent fresh accumulator reused across two calls must give the same result
    both times -- with the explicit init (added by simplify, which runs
    InitializeReductionAccumulators) the second call does not accumulate onto the
    first. (Sanity / end-to-end correctness; the structural tests above are the
    regression guard, since the simple non-nested WCR form is also init'd by codegen.)"""
    sdfg = _fresh_reduction_sdfg()
    acc = next(n for n in sdfg.arrays if n not in ('B', 'out'))
    sdfg.arrays[acc].lifetime = dtypes.AllocationLifetime.Persistent
    # Apply the pass directly rather than full simplify: simplify would eliminate this
    # single-use transient (redirecting the WCR to the non-transient ``out``), which the
    # pass deliberately leaves alone. Here ``acc`` survives, exercising the init path.
    InitializeReductionAccumulators().apply_pass(sdfg, {})
    assert _has_init_for(sdfg, acc), "pass must have inserted the explicit accumulator init"

    csdfg = sdfg.compile()
    rng = np.random.default_rng(0)
    B = rng.random((8, 8))
    expected = B.sum(axis=1)

    out1 = np.zeros(8)
    csdfg(B=B, out=out1, N=8)
    out2 = np.zeros(8)
    csdfg(B=B, out=out2, N=8)
    assert np.allclose(out1, expected)
    assert np.allclose(out2, expected), "second call must not accumulate onto the first"


if __name__ == '__main__':
    test_pass_adds_init_for_fresh_accumulator()
    test_pass_skips_seeded_accumulator()
    test_persistent_accumulator_reused_across_calls()
    print('OK')
