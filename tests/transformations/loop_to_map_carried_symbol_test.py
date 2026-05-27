# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""LoopToMap and loop-carried symbols.

A loop that carries a *symbol* across iterations -- a scalar reassigned on the
loop body's interstate edge and read in the body *before* that reassignment -- is
not independently parallelizable: each iteration's read observes the previous
iteration's value. The wrap-around induction
``im = N - 1; DO i: a[i] = b[i] + b[im]; im = i`` (TSVC s291 in induction form) is
the canonical case -- ``im`` equals ``i - 1`` for every iteration except the
first, which reads ``b[N - 1]``.

* **Unpeeled**, LoopToMap must REFUSE it: turning the loop into a Map pins ``im``
  to its loop-entry value ``N - 1`` for every (now independent) iteration, silently
  computing ``b[i] + b[N - 1]`` everywhere instead of ``b[i] + b[i - 1]``.
* **Peeled** -- once the wrapping first iteration is split off and the induction
  substituted (``im -> i - 1``), the body ``a[i] = b[i] + b[i - 1]`` is affine and
  LoopToMap accepts it.
"""
import dace
from dace.sdfg.state import LoopRegion
from dace.transformation.interstate import LoopToMap

N = dace.symbol('N')


@dace.program
def _carried_symbol(a: dace.float64[N], b: dace.float64[N]):
    im = N - 1
    for i in range(N):
        a[i] = b[i] + b[im]
        im = i


@dace.program
def _peeled_affine(a: dace.float64[N], b: dace.float64[N]):
    a[0] = b[0] + b[N - 1]          # the wrapping first iteration, peeled off
    for i in range(1, N):
        a[i] = b[i] + b[i - 1]      # induction substituted: affine, parallelizable


def _loop(sdfg: dace.SDFG) -> LoopRegion:
    return next(n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, LoopRegion))


def test_loop2map_rejects_unpeeled_carried_symbol():
    """The unpeeled wrap-around induction carries ``im`` across iterations (read in
    ``b[im]`` before the body reassigns ``im = i``), so LoopToMap must refuse it: a
    Map would pin ``im`` to its loop-entry value ``N - 1`` and silently corrupt the
    result. The carried-symbol check folds in-state reads (memlet subsets), not just
    interstate-edge reads, to catch this."""
    sdfg = _carried_symbol.to_sdfg(simplify=True)
    assert not LoopToMap.can_be_applied_to(sdfg, loop=_loop(sdfg))


def test_loop2map_accepts_peeled_affine_form():
    """After peeling the wrapping first iteration and substituting the induction
    (``im -> i - 1``), the remainder ``a[i] = b[i] + b[i - 1]`` is affine and
    LoopToMap accepts it."""
    sdfg = _peeled_affine.to_sdfg(simplify=True)
    assert LoopToMap.can_be_applied_to(sdfg, loop=_loop(sdfg))


if __name__ == '__main__':
    test_loop2map_rejects_unpeeled_carried_symbol()
    test_loop2map_accepts_peeled_affine_form()
