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
from dace.sdfg.state import LoopRegion
from dace.transformation.passes.canonicalize import canonicalize

N = dace.symbol('N')


def _nmaps(sdfg):
    return sum(1 for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.MapEntry))


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
    assert _nmaps(sdfg) >= 1  # the elementwise second pass parallelizes


if __name__ == '__main__':
    test_masked_conditional_reduction_value_preserving()
    test_two_pass_normalize_value_preserving()
