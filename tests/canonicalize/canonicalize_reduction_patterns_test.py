# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Canonicalize on reduction-shaped patterns from real ICON / CLOUDSC / ECRAD kernels.

Two load-bearing shapes the other canonicalize pattern suites do not cover:

* **Masked / conditional reduction** -- ``DO i: IF cond(i): acc = acc + x(i)``.
  CLOUDSC accumulates fluxes / tendencies only where a per-element condition holds
  (e.g. only in cloudy cells). The data-dependent guard must stay per-element -- a
  hoist would change which elements contribute -- and the running sum must stay
  exact; the accumulator write must not be parallelized into a racy map.

* **Two-pass normalization** -- ``s = sum(a); DO i: b(i) = a(i) / s``. ECRAD
  normalizes cloud-fraction / spectral weights by their sum: a reduction pass
  feeds an independent elementwise pass. The second pass reads the now-fixed scalar
  and parallelizes.

Each test pins value-preservation against a numpy oracle plus a structural contract.
"""
import numpy as np

import dace
from dace.sdfg import nodes
from dace.sdfg.state import ConditionalBlock, LoopRegion
from dace.transformation.passes.canonicalize import canonicalize

N = dace.symbol('N')


def _nmaps(sdfg):
    return sum(1 for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.MapEntry))


def _wcr_edges(sdfg):
    """Edges carrying a write-conflict resolution, i.e. the accumulations that are safe to run in
    parallel. A bare write into a shared accumulator under a Map would have none."""
    return [(st.label, e.data.data) for sd in sdfg.all_sdfgs_recursive() for st in sd.states() for e in st.edges()
            if e.data is not None and e.data.wcr is not None]


def _nconds(sdfg):
    return sum(1 for r in sdfg.all_control_flow_regions(recursive=True) if isinstance(r, ConditionalBlock))


def _nloops(sdfg):
    return sum(1 for r in sdfg.all_control_flow_regions(recursive=True) if isinstance(r, LoopRegion))


@dace.program
def _masked_reduction(a: dace.float64[N], acc: dace.float64[1]):
    for i in range(N):
        if a[i] > 0.5:
            acc[0] = acc[0] + a[i]


def test_masked_conditional_reduction_value_preserving():
    """CLOUDSC-style masked accumulation ``if a[i] > 0.5: acc += a[i]`` -- the sum
    over only the elements passing a data-dependent guard. Canonicalize must keep it
    value-preserving: the guard stays per-element (hoisting it would change which
    elements contribute) and the carried accumulation stays correct."""
    a = np.linspace(0.0, 1.0, 16, dtype=np.float64)
    ref = np.zeros(1)
    _masked_reduction.to_sdfg(simplify=True)(a=a.copy(), acc=ref, N=16)

    sdfg = _masked_reduction.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True)
    got = np.zeros(1)
    sdfg(a=a.copy(), acc=got, N=16)
    assert np.allclose(got, ref)
    assert np.allclose(got[0], a[a > 0.5].sum())  # exactly the masked sum
    assert _nloops(sdfg) == 0, f'the masked reduction stayed a sequential loop: {_nloops(sdfg)}'
    assert _nmaps(sdfg) == 1, f'expected the one parallel accumulation map, got {_nmaps(sdfg)}'
    assert _nconds(sdfg) == 0, 'the per-element guard became a ConditionalBlock instead of a masked value'
    assert _wcr_edges(sdfg), 'the accumulator is written without a WCR -- a racy map, not a reduction'


@dace.program
def _two_pass_normalize(a: dace.float64[N], b: dace.float64[N]):
    s = 0.0
    for i in range(N):
        s = s + a[i]
    for i in range(N):
        b[i] = a[i] / s


def test_two_pass_normalize_value_preserving():
    """ECRAD-style normalization ``s = sum(a); b[i] = a[i] / s`` -- a reduction pass
    feeding an independent elementwise pass. Value-preserving, and the second pass
    becomes a parallel map (a pure per-element divide by the now-fixed scalar ``s``)."""
    a = np.arange(1, 17, dtype=np.float64)
    ref = np.empty(16)
    _two_pass_normalize.to_sdfg(simplify=True)(a=a.copy(), b=ref, N=16)

    sdfg = _two_pass_normalize.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True)
    got = np.empty(16)
    sdfg(a=a.copy(), b=got, N=16)
    assert np.allclose(got, ref)
    assert np.allclose(got, a / a.sum())
    assert _nmaps(sdfg) == 1, f'the elementwise second pass is the only map; got {_nmaps(sdfg)}'
    assert _nloops(sdfg) == 0, f'neither pass may stay a sequential loop, got {_nloops(sdfg)}'
    lifted = [type(n).__name__ for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.LibraryNode)]
    assert 'Reduce' in lifted, f'the summation pass did not lift to a Reduce: {lifted}'


if __name__ == '__main__':
    test_masked_conditional_reduction_value_preserving()
    test_two_pass_normalize_value_preserving()
