# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Two TSVC carries that HAVE a closed form, at the edge of what canonicalize closes.

A loop-carried value is only a reason to run a loop sequentially when it has no closed form.
Both kernels here have one, and each was once refused for a reason one structural step short of
the shape its matcher accepts:

* ``s252`` -- ``t = s`` carries the previous iteration's PRODUCT, not an accumulation, so the
  closed form is that product re-evaluated one iteration back. ``LoopCarriedRotationSubstitution``
  does exactly that, for the corpus form and for the extra pure stage the HPCAgent-Bench
  numpy-to-dace emitter writes (``t = s + 0.0``), whose producer reads a container the body
  writes and so has to be recomputed one level deeper. This kernel is CLOSED; it stays here
  because the staged form is the shape that used to be refused.

* ``s323`` -- ``a[i] = b[i-1] + c[i]*d[i]; b[i] = a[i] + c[i]*e[i]``. There is NO ``a[i-1]``
  read, so ``a`` carries nothing; substituting ``a[i]`` leaves ``b[i] = b[i-1] + c[i]*d[i] +
  c[i]*e[i]``, a first-order recurrence with unit coefficient -- a prefix sum. DaCe lifts that
  shape today (``test_s323_becomes_a_scan_when_the_carry_skips_the_array_round_trip`` below
  proves it on the same arithmetic). What used to block the real kernel is that the carry reaches
  the ``b`` update only through a store to, and a reload of, the NON-TRANSIENT output ``a``;
  ``ForwardStoreToLoad`` forwards that same-iteration value, and the kernel is now CLOSED.

Numerics are asserted unconditionally on every case, because the outcome that matters more than
"left sequential" is "parallelized into a race". Structure is asserted separately: a value check
passes just as happily on the un-lifted loop, so it cannot tell a closed carry from an open one.
"""
import os

os.environ.setdefault("OMPI_MCA_pml", "ob1")
os.environ.setdefault("OMPI_MCA_btl", "self,vader")
os.environ.setdefault("UCX_VFS_ENABLE", "n")

import numpy as np
import pytest

import dace
from dace.sdfg import nodes as nd
from dace.sdfg.state import LoopRegion
from dace.transformation.passes.canonicalize import canonicalize

from tests.corpus.tsvc import tsvc
from tests.corpus.tsvc.tsvc_numpy import REFERENCES

N = dace.symbol('N')

#: Loose enough that an FMA contraction or a scan's reassociation cannot decide an assertion,
#: tight enough that a wrong element cannot hide behind it.
TOL = dict(rtol=1e-12, atol=1e-12)


@dace.program
def staged_delay_line(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N]):
    """s252 as the numpy-to-dace emitter spells it: the carry is one pure stage further back."""
    t = 0.0
    for i in range(N):
        s = b[i] * c[i]
        a[i] = s + t
        t = s + 0.0


@dace.program
def coupled_scan_via_transient(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N], d: dace.float64[N],
                               e: dace.float64[N]):
    """s323's arithmetic with the intra-iteration value forwarded instead of reloaded from ``a``."""
    for i in range(1, N):
        tmp = b[i - 1] + c[i] * d[i]
        a[i] = tmp
        b[i] = tmp + c[i] * e[i]


def canonicalized(program, name: str) -> dace.SDFG:
    sdfg = program.to_sdfg(simplify=True)
    sdfg.name = name
    canonicalize(sdfg, validate=True, peel_limit=4)
    return sdfg


def canonicalized_corpus(kernel_name: str, tag: str):
    """``(kernel, sdfg)`` for one corpus TSVC kernel put through the production recipe."""
    kernel = tsvc.collect(name=kernel_name)[0]
    sdfg = tsvc.to_sdfg(kernel, tag, simplify=True)
    canonicalize(sdfg, validate=True, peel_limit=4)
    return kernel, sdfg


def residual_loops(sdfg: dace.SDFG) -> list[str]:
    """Labels of the sequential ``LoopRegion`` s canonicalize did not turn into parallel work."""
    return [
        r.label for sd in sdfg.all_sdfgs_recursive() for r in sd.all_control_flow_regions()
        if isinstance(r, LoopRegion) and r.loop_variable
    ]


def maps(sdfg: dace.SDFG) -> list[nd.Map]:
    return [n.map for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nd.MapEntry)]


def libnodes(sdfg: dace.SDFG) -> list[str]:
    return sorted({type(n).__name__ for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nd.LibraryNode)})


def assert_matches_corpus_reference(kernel, sdfg: dace.SDFG) -> None:
    """The canonicalized kernel must still reproduce the numpy reference.

    Structure is read off ``sdfg`` BEFORE this runs: compiling is one-way.
    """
    arrays, call_kwargs = tsvc.make_inputs(kernel)
    want = {n: arr.copy() for n, arr in arrays.items()}
    REFERENCES[kernel.name](**want, **call_kwargs)
    got = {n: arr.copy() for n, arr in arrays.items()}
    sdfg.compile()(**got, **call_kwargs)
    for name in arrays:
        assert np.allclose(want[name], got[name], **TOL), f'{kernel.name}: value mismatch on {name}'


def s252_reference(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> None:
    t = 0.0
    for i in range(a.shape[0]):
        s = b[i] * c[i]
        a[i] = s + t
        t = s + 0.0


def s323_reference(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray, e: np.ndarray) -> None:
    for i in range(1, a.shape[0]):
        a[i] = b[i - 1] + c[i] * d[i]
        b[i] = a[i] + c[i] * e[i]


def inputs(count: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [rng.random(64) + 1.0 for _ in range(count)]


def test_s252_corpus_form_closes_the_delay_line():
    """``t = s`` -- one stage, so the producer clone reads only ``b`` and ``c``, which are unwritten.

    The baseline for the staged form below: what was once refused there is the CHAIN LENGTH, not
    the kernel, and this pins that the one-stage form is genuinely handled rather than merely
    untested.
    """
    kernel, sdfg = canonicalized_corpus('s252_d_single', 's252_corpus')
    assert residual_loops(sdfg) == [], 's252 must close its delay line and parallelize'
    assert maps(sdfg), 's252 lost its carry but never became a map'
    assert_matches_corpus_reference(kernel, sdfg)


def test_s252_staged_delay_line_is_value_preserving():
    """Whatever canonicalize decides about the staged form, it must still compute s252."""
    b, c = inputs(2, 252)
    want = np.zeros(64)
    s252_reference(want, b, c)

    sdfg = canonicalized(staged_delay_line, 'staged_delay_line_values')
    got = np.zeros(64)
    sdfg.compile()(a=got, b=b.copy(), c=c.copy(), N=64)
    assert np.allclose(want, got, **TOL), 'the staged delay line no longer computes s252'


def test_s252_staged_delay_line_should_still_close():
    """``t = s + 0.0`` is the same delay line with one extra pure stage, so it has the same closed form.

    ``t`` at iteration ``i`` is ``b[i-1]*c[i-1] + 0.0``: the update never reads ``t``, so no value
    accumulates and the dependence is a one-deep rotation, not a recurrence. Nothing about that
    changes when the stored value passes through a second pure tasklet, so
    ``rematerializable_producer`` recomputes the extra stage instead of refusing the body-written
    input it reads.
    """
    sdfg = canonicalized(staged_delay_line, 'staged_delay_line_struct')
    assert residual_loops(sdfg) == [], 'the staged delay line must close the same way the corpus form does'
    assert maps(sdfg), 'the staged delay line lost its carry but never became a map'


def test_s323_becomes_a_scan_when_the_carry_skips_the_array_round_trip():
    """Same arithmetic as s323, with ``a[i]``'s value forwarded rather than reloaded from memory.

    This is what made the corpus form below a MISS rather than a refusal: the recurrence itself is
    already in ``LoopToScan``'s contract, and ``a`` is still written on every iteration. The one
    thing that changes is that the ``b`` update reads a transient, so ``a`` is write-only in the
    body and stops being counted as a scan carrier that has to match. ``ForwardStoreToLoad`` is
    the pass that puts the corpus form into exactly this shape.
    """
    a, b, c, d, e = inputs(5, 323)
    want = [a.copy(), b.copy(), c, d, e]
    s323_reference(*want)

    sdfg = canonicalized(coupled_scan_via_transient, 'coupled_scan_via_transient')
    assert 'Scan' in libnodes(sdfg), 'the forwarded form must lift to a Scan'
    assert residual_loops(sdfg) == [], 'the forwarded form must leave no sequential loop'

    got_a, got_b = a.copy(), b.copy()
    sdfg.compile()(a=got_a, b=got_b, c=c.copy(), d=d.copy(), e=e.copy(), N=64)
    assert np.allclose(want[0], got_a, **TOL), 'the lifted scan computed `a` wrong'
    assert np.allclose(want[1], got_b, **TOL), 'the lifted scan computed `b` wrong'


def test_s323_is_value_preserving():
    """s323 must still compute s323 -- the assertion that a wrong "parallelization" would break."""
    kernel, sdfg = canonicalized_corpus('s323_d_single', 's323_values')
    assert_matches_corpus_reference(kernel, sdfg)


def test_s323_lifts_to_a_prefix_scan():
    """s323's only loop-carried dependence is a unit-coefficient first-order recurrence on ``b``.

    ``a[i] = b[i-1] + c[i]*d[i]`` and ``b[i] = a[i] + c[i]*e[i]``. The read of ``a[i]`` is the
    value written in the SAME iteration, so it is a store-to-load, not a carry: substituting it
    gives ``b[i] = b[i-1] + c[i]*d[i] + c[i]*e[i]``, whose delta depends on nothing carried. That
    is a prefix sum seeded at ``b[0]``, and ``a`` is then the exclusive scan plus ``c[i]*d[i]``.
    Nothing here forbids parallel execution; only the reassociation a scan always costs.

    ``ForwardStoreToLoad`` is what closes it: it feeds the ``b`` update from the value stored to
    ``a`` rather than from ``a`` itself (the store stays), which leaves ``b`` as the only array
    crossing between the two statements. ``SplitStatements`` then has one provable order, and the
    ``b`` loop it isolates is the scan above -- the shape
    ``test_s323_becomes_a_scan_when_the_carry_skips_the_array_round_trip`` always lifted.
    """
    _kernel, sdfg = canonicalized_corpus('s323_d_single', 's323_struct')
    assert 'Scan' in libnodes(sdfg), 's323 carries a prefix sum and must lift to a Scan'
    assert residual_loops(sdfg) == [], 's323 must leave no sequential loop once the scan is lifted'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
