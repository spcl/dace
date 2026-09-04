# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A carried scalar fed through SEVERAL pure stages is still rematerialized, and only then.

``LoopCarriedRotationSubstitution`` closes a delay line whose carried value was computed by cloning
the producer and evaluating it one iteration back. The producer's own inputs may themselves be
computed in the body -- the numpy-to-dace emitters routinely stage a value through an extra
assignment -- so the clone is a CHAIN whose leaves are the only reads that touch memory::

    t = 0.0
    for i in range(N):
        s = b[i] * c[i]
        u = s + 0.0
        a[i] = s + t     # t == b[i-1] * c[i-1] + 0.0
        t = u

Descending re-proves at every level that the value read is the one produced in the SAME iteration
by a pure tasklet. The negative cases below are the shapes where that proof fails deep in the chain
rather than at its head: substituting any of them would parallelize the loop and compute every
element wrong, with no error raised.
"""
import math

import numpy as np
import pytest

import dace
from dace.config import Config, set_temporary
from dace.sdfg import nodes
from dace.sdfg.state import LoopRegion
from dace.transformation.passes.canonicalize import canonicalize

N = dace.symbol('N')

#: The rewrite is exact in ARITHMETIC, so the comparisons below are bit-exact -- but only with
#: contraction off: the sequential loop rounds the carried product, while the rematerialized body
#: would contract ``b[i]*c[i] + t_remat`` into one FMA and land a ulp away.
STRICT_FP_CPU_ARGS: str = Config.get('compiler', 'cpu', 'args') + ' -fno-fast-math -ffp-contract=off'


@dace.program
def two_stage(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N]):
    """The carry is two pure tasklets away from memory."""
    t = 0.0
    for i in range(N):
        s = b[i] * c[i]
        u = s + 0.0
        a[i] = s + t
        t = u


@dace.program
def three_stage(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N]):
    """Three, which is still inside the chase limit."""
    t = 0.0
    for i in range(N):
        s = b[i] * c[i]
        u = s + 0.0
        v = u * 1.0
        a[i] = s + t
        t = v


@dace.program
def over_the_limit(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N]):
    """One stage past ``_ROTATION_CHASE_LIMIT``, so the chain is refused rather than guessed at."""
    t = 0.0
    for i in range(N):
        s = b[i] * c[i]
        u = s + 0.0
        v = u + 0.0
        w = v + 0.0
        x = w + 0.0
        a[i] = s + t
        t = x


@dace.program
def deep_producer_calls(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N]):
    """The impurity sits at the BOTTOM of the chain, where only the recursion can see it."""
    t = 0.0
    for i in range(N):
        s = math.sqrt(b[i]) * c[i]
        u = s + 0.0
        a[i] = s + t
        t = u


@dace.program
def deep_producer_accumulates(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N]):
    """``s`` reads the carry, so ``t`` accumulates -- a REDUCTION wearing a two-stage delay line."""
    t = 0.0
    for i in range(N):
        s = b[i] * c[i] + t
        u = s + 0.0
        a[i] = s * 2.0
        t = u


def oracle(program, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> None:
    """The sequential loop run in Python -- one branch per kernel above, same arithmetic, same order."""
    t = 0.0
    for i in range(a.shape[0]):
        if program is deep_producer_calls:
            s = math.sqrt(b[i]) * c[i]
        elif program is deep_producer_accumulates:
            s = b[i] * c[i] + t
        else:
            s = b[i] * c[i]
        a[i] = s * 2.0 if program is deep_producer_accumulates else s + t
        t = s * 1.0 if program is three_stage else s + 0.0


def canonicalized(program, name: str) -> dace.SDFG:
    sdfg = program.to_sdfg(simplify=True)
    sdfg.name = name
    canonicalize(sdfg, validate=True, peel_limit=4)
    return sdfg


def remat_tasklets(sdfg: dace.SDFG) -> list[str]:
    """Labels of the cloned producers the rewrite minted, if any."""
    return [
        n.label for sd in sdfg.all_sdfgs_recursive() for st in sd.states() for n in st.nodes()
        if isinstance(n, nodes.Tasklet) and n.label.endswith('_remat')
    ]


def residual_loops(sdfg: dace.SDFG) -> list[str]:
    return [
        r.label for sd in sdfg.all_sdfgs_recursive() for r in sd.all_control_flow_regions()
        if isinstance(r, LoopRegion) and r.loop_variable
    ]


def run(program, name: str) -> dace.SDFG:
    """Canonicalize, then hold the result against the sequential oracle bit for bit."""
    n = 64
    rng = np.random.default_rng(252)
    b = rng.random(n) + 1.0
    c = rng.random(n) + 1.0
    want = np.zeros(n)
    oracle(program, want, b, c)

    sdfg = canonicalized(program, name)
    got = np.zeros(n)
    with set_temporary('compiler', 'cpu', 'args', value=STRICT_FP_CPU_ARGS):
        sdfg.compile()(a=got, b=b.copy(), c=c.copy(), N=n)
    assert np.allclose(got, want, rtol=0, atol=0), 'the staged carry was rematerialized wrong'
    return sdfg


def test_a_two_stage_chain_is_rematerialized():
    """The head producer reads a body-written transient, so the clone has to go one level deeper."""
    sdfg = run(two_stage, 'remat_two_stage')
    assert len(remat_tasklets(sdfg)) == 2, 'the single read of the carry must mint one clone per stage'
    assert residual_loops(sdfg) == [], 'the carry is gone but the loop stayed sequential'


def test_a_three_stage_chain_is_rematerialized():
    """Depth is not special-cased: the same proof applies once more."""
    sdfg = run(three_stage, 'remat_three_stage')
    assert len(remat_tasklets(sdfg)) == 3, 'the single read of the carry must mint one clone per stage'
    assert residual_loops(sdfg) == [], 'the carry is gone but the loop stayed sequential'


def test_a_chain_past_the_limit_is_refused():
    """The bound is a refusal, not a truncation -- a partial chain would recompute the wrong value."""
    sdfg = run(over_the_limit, 'remat_over_the_limit')
    assert not remat_tasklets(sdfg), 'a chain past the chase limit was rematerialized anyway'
    assert residual_loops(sdfg), 'the refused chain must leave the loop sequential'


def test_a_deep_calling_producer_is_refused():
    """A call may be a dace callback; cloning the chain would run it a second time."""
    sdfg = run(deep_producer_calls, 'remat_deep_call')
    assert not remat_tasklets(sdfg), 'a call at the bottom of the chain was duplicated'
    assert residual_loops(sdfg), 'the refused chain must leave the loop sequential'


def test_a_deep_carried_producer_is_refused():
    """``s = b[i]*c[i] + t`` reads the carry, so the value is accumulated, not delayed."""
    sdfg = run(deep_producer_accumulates, 'remat_deep_carry')
    assert not remat_tasklets(sdfg), 'an accumulation was rematerialized as a delay line'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
