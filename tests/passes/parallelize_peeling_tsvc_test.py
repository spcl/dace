# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Loop-peeling landscape for the ``parallelize`` pipeline, grounded in TSVC.

These kernels (from the TSVC suite, expressed inner-loop-only as ``@dace.program``s
over a symbolic length ``N``) document which shapes peeling can and cannot help:

- **s291 / s2244**: TSVC presents the wrap-around / boundary kernels *already in
  peeled form* -- the special boundary element is straight-line code and the
  remainder loop reads only distinct arrays, so it is parallel as-is. These lock
  in that the pipeline maps the peeled form (and would be the *target* a peel
  produces).
- **s121** (``a[i] = a[i+1] + b[i]``): a loop-carried anti-dependence on ``a``.
  Peeling a boundary does not remove the interior anti-dependence -- this needs
  reverse + renaming, not pure peeling -- so it currently stays sequential.
- **s112** (``a[i+1] = a[i] + b[i]``): a true prefix-scan recurrence. It looks
  like s121's twin but must NEVER be parallelized; this is the correctness
  control for the peeling/reverse search.

All cases are value-preserving regardless of whether a map is produced.
"""
import numpy as np

import dace
from dace.sdfg.state import LoopRegion
from dace.sdfg import nodes
from dace.transformation.passes import parallelize

N = dace.symbol('N')


def _nloops(sdfg):
    return len([r for r in sdfg.all_control_flow_regions() if isinstance(r, LoopRegion) and r.loop_variable])


def _nmaps(sdfg):
    return sum(1 for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.MapEntry))


def test_tsvc_s291_front_peeled_form_is_parallel():
    """s291 wrap-around 2-point average, front-peeled (iter 0 split out): the
    remainder reads only ``b`` (distinct from the written ``a``) and maps."""

    @dace.program
    def s291(a: dace.float64[N], b: dace.float64[N]):
        a[0] = (b[0] + b[N - 1]) * 0.5
        for i in range(1, N):
            a[i] = (b[i] + b[i - 1]) * 0.5

    sdfg = s291.to_sdfg(simplify=True)
    parallelize(sdfg, validate=True)
    assert _nmaps(sdfg) >= 1

    a = np.zeros(8)
    b = np.arange(1, 9, dtype=np.float64)
    sdfg(a=a, b=b, N=8)
    ref = np.empty(8)
    ref[0] = (b[0] + b[7]) * 0.5
    ref[1:] = (b[1:] + b[:-1]) * 0.5
    assert np.allclose(a, ref)


def test_tsvc_s2244_back_peeled_form_is_parallel():
    """s2244 back-peeled: the tail element is straight-line and the remainder
    ``a[i] = b[i] + c[i]`` maps."""

    @dace.program
    def s2244(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N], e: dace.float64[N]):
        a[N - 1] = b[N - 2] + e[N - 2]
        for i in range(N - 1):
            a[i] = b[i] + c[i]

    sdfg = s2244.to_sdfg(simplify=True)
    parallelize(sdfg, validate=True)
    assert _nmaps(sdfg) >= 1

    a = np.zeros(8)
    b = np.arange(8, dtype=np.float64) + 1
    c = np.arange(8, dtype=np.float64) + 2
    e = np.arange(8, dtype=np.float64) + 3
    sdfg(a=a, b=b, c=c, e=e, N=8)
    ref = b + c
    ref[7] = b[6] + e[6]
    assert np.allclose(a, ref)


def test_tsvc_s121_antidependence_value_preserving():
    """s121 anti-dependence ``a[i] = a[i+1] + b[i]``: peeling cannot break the
    interior anti-dependence, so it stays a sequential loop (open: reverse +
    rename). Must remain correct."""

    @dace.program
    def s121(a: dace.float64[N], b: dace.float64[N]):
        for i in range(N - 1):
            a[i] = a[i + 1] + b[i]

    sdfg = s121.to_sdfg(simplify=True)
    parallelize(sdfg, validate=True)
    assert _nloops(sdfg) >= 1  # currently not parallelized

    a = (np.arange(8, dtype=np.float64) + 1).copy()
    b = np.arange(8, dtype=np.float64) * 0.5
    ref = a.copy()
    for i in range(7):
        ref[i] = ref[i + 1] + b[i]
    sdfg(a=a, b=b, N=8)
    assert np.allclose(a, ref)


def test_tsvc_s112_recurrence_must_not_parallelize():
    """s112 prefix-scan recurrence ``a[i+1] = a[i] + b[i]``: the correctness
    control -- peeling/reverse must NEVER turn this into a map."""

    @dace.program
    def s112(a: dace.float64[N], b: dace.float64[N]):
        for i in range(N - 2, -1, -1):
            a[i + 1] = a[i] + b[i]

    sdfg = s112.to_sdfg(simplify=True)
    parallelize(sdfg, validate=True)
    assert _nloops(sdfg) >= 1  # recurrence stays sequential

    a = (np.arange(8, dtype=np.float64) + 1).copy()
    b = np.arange(8, dtype=np.float64) * 0.5
    ref = a.copy()
    for i in range(6, -1, -1):
        ref[i + 1] = ref[i] + b[i]
    sdfg(a=a, b=b, N=8)
    assert np.allclose(a, ref)


def test_peeling_unblocks_boundary_conditional():
    """A loop whose first iteration writes a conflicting extra location
    (``if i==0: A[N-1]+=1``) is not parallel as-is, but front-peeling iteration 0
    and pruning the now-dead guard from the remainder leaves a disjoint-write
    body that LoopToMap parallelizes. The positive demonstration of peel-to-map."""

    @dace.program
    def front_conflict(A: dace.float64[N], B: dace.float64[N]):
        for i in range(N):
            A[i] = B[i] * 2.0
            if i == 0:
                A[N - 1] = A[N - 1] + 1.0

    base = front_conflict.to_sdfg(simplify=True)
    import contextlib, os
    from dace.transformation.interstate.loop_to_map import LoopToMap
    with contextlib.redirect_stdout(open(os.devnull, 'w')):
        base.apply_transformations_repeated(LoopToMap)
    assert _nmaps(base) == 0  # LoopToMap alone cannot parallelize it

    sdfg = front_conflict.to_sdfg(simplify=True)
    parallelize(sdfg, validate=True)
    assert _nmaps(sdfg) >= 1  # peel + dead-guard prune unblocks the map

    A = np.arange(1, 9, dtype=np.float64)
    B = np.arange(8, dtype=np.float64) + 0.5
    ref_A = A.copy()
    ref = front_conflict.to_sdfg(simplify=True)
    ref(A=ref_A, B=B.copy(), N=8)
    Ac = A.copy()
    sdfg(A=Ac, B=B.copy(), N=8)
    assert np.allclose(ref_A, Ac)


if __name__ == '__main__':
    test_tsvc_s291_front_peeled_form_is_parallel()
    test_tsvc_s2244_back_peeled_form_is_parallel()
    test_tsvc_s121_antidependence_value_preserving()
    test_tsvc_s112_recurrence_must_not_parallelize()
