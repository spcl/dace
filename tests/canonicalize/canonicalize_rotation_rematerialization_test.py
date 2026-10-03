# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A carried scalar holding the PREVIOUS iteration's computed value is rematerialized, not folded.

TSVC s252's carry is not a shifted array read -- it is the previous iteration's value of a computed
transient::

    t = 0.0
    for i in range(N):
        s = b[i] * c[i]
        a[i] = s + t     # t == b[i-1] * c[i-1]
        t = s

so its closed form is the PRODUCER re-evaluated one iteration back. ``LoopCarriedRotationSubstitution``
clones the producer with its reads shifted and peels the first iteration, which makes the remainder
DOALL. Recomputation is only sound while the clone cannot duplicate a side effect and cannot re-read a
value the body itself overwrote, so the negative tests below pin those two refusals: each is a shape
where substituting would parallelize the loop and compute every element wrong, with no error raised.
"""
import math

import numpy as np
import pytest

import dace
from dace.config import Config, set_temporary
from dace.sdfg import nodes
from dace.transformation.passes.canonicalize import canonicalize

N = dace.symbol('N')

#: The rewrite is exact in ARITHMETIC, so the comparisons below are bit-exact. g++ defaults to
#: ``-ffp-contract=fast``, which would decide them instead: the sequential loop must round ``s``
#: because the next iteration carries it, while the remat body's ``b[i]*c[i] + t_remat`` contracts
#: to one FMA -- 1 ulp on 16 of 64 elements, measured. Pinning contraction off keeps the assertions
#: about the pass. Same flags the CloudSC numeric harnesses use for the same reason.
STRICT_FP_CPU_ARGS: str = Config.get('compiler', 'cpu', 'args') + ' -fno-fast-math -ffp-contract=off'


@dace.program
def remat_carry(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N]):
    t = 0.0
    for i in range(N):
        s = b[i] * c[i]
        a[i] = s + t
        t = s


@dace.program
def reduction_lookalike(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N]):
    """Surface-identical to ``remat_carry``, but the carry ACCUMULATES -- substituting is a miscompile."""
    t = 0.0
    for i in range(N):
        s = b[i] * c[i]
        a[i] = s + t
        t = t + s


@dace.program
def producer_input_overwritten(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N]):
    """``b`` is written in the body, so ``b[i-1]`` no longer holds what iteration ``i-1`` read.

    The store is fed by the CARRY, which is what keeps it in the carry's group when the body
    distributes. Store ``a[i]`` instead and the split peels it into a loop of its own; the
    producer's input then really is unwritten where the rotation looks, and rematerializing is
    legal -- the split changed the program, not the rule this pins.
    """
    t = 0.0
    for i in range(N):
        s = b[i] * c[i]
        a[i] = s + t
        b[i] = t
        t = s


@dace.program
def producer_calls_a_function(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N]):
    """A call may be a dace callback, so re-evaluating the producer may duplicate a side effect."""
    t = 0.0
    for i in range(N):
        s = math.sqrt(b[i]) * c[i]
        a[i] = s + t
        t = s


def reference(kernel: str, a, b, c):
    """The sequential loop, run in Python -- the oracle every case below is compared against."""
    t = 0.0
    for i in range(len(b)):
        s = math.sqrt(b[i]) * c[i] if kernel == 'call' else b[i] * c[i]
        a[i] = s + t
        if kernel == 'overwrite':
            b[i] = t
        t = t + s if kernel == 'reduce' else s


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


def run(program, kernel: str, name: str):
    n = 64
    rng = np.random.default_rng(252)
    b = rng.random(n) + 1.0
    c = rng.random(n) + 1.0
    want = np.zeros(n)
    want_b = b.copy()
    reference(kernel, want, want_b, c)

    sdfg = canonicalized(program, name)
    got = np.zeros(n)
    got_b = b.copy()
    with set_temporary('compiler', 'cpu', 'args', value=STRICT_FP_CPU_ARGS):
        sdfg.compile()(a=got, b=got_b, c=c.copy(), N=n)
    assert np.allclose(got, want, rtol=0, atol=0), 'the carried value was rematerialized wrong'
    assert np.allclose(got_b, want_b, rtol=0, atol=0), 'the body overwrote the wrong elements'
    return sdfg


def test_computed_carry_is_rematerialized():
    """s252: recomputing ``b[i-1] * c[i-1]`` must reproduce the sequential loop bit for bit."""
    sdfg = run(remat_carry, 'remat', 'remat_carry_canon')
    assert remat_tasklets(sdfg), 'the producer was never rematerialized'


def test_the_rematerialized_remainder_is_parallel():
    """Structural half: the carry is gone, so the peeled remainder is a Map over the whole range."""
    sdfg = canonicalized(remat_carry, 'remat_carry_struct')
    maps = [
        n.map for sd in sdfg.all_sdfgs_recursive() for st in sd.states() for n in st.nodes()
        if isinstance(n, nodes.MapEntry)
    ]
    assert maps, 'the rematerialized loop did not parallelize'
    wcrs = [
        e.data.data for sd in sdfg.all_sdfgs_recursive() for st in sd.states() for e in st.edges()
        if e.data is not None and e.data.wcr is not None
    ]
    assert not wcrs, f'a rotation has nothing to reduce, but a WCR appeared: {wcrs}'


def test_an_accumulating_carry_is_not_rematerialized():
    """``t = t + s`` reads the carry, so its producer is fed by a container the body writes."""
    sdfg = run(reduction_lookalike, 'reduce', 'reduction_lookalike_canon')
    assert not remat_tasklets(sdfg), 'a reduction was rematerialized as a delay line'


def test_a_producer_input_the_body_overwrites_is_not_rematerialized():
    """The clone reads LATE: ``b[i-1]`` has already been overwritten by iteration ``i-1``."""
    sdfg = run(producer_input_overwritten, 'overwrite', 'producer_input_overwritten_canon')
    assert not remat_tasklets(sdfg), 'the clone would re-read an overwritten element'


def test_a_calling_producer_is_not_rematerialized():
    """A second evaluation would run the call twice; the accepted producer language has no calls."""
    sdfg = run(producer_calls_a_function, 'call', 'producer_calls_a_function_canon')
    assert not remat_tasklets(sdfg), 'a call was duplicated by rematerialization'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
