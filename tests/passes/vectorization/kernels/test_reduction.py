# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Loop-carried scalar reduction over a tiled map (multi-dim tile-op path).

A reduction whose accumulator threads through the tiled map (`acc = acc + a[i]`)
is lowered by :class:`TileCarriedScalarReduction` to a ``(W,)`` partial-sum tile
+ a ``TileReduce`` fold (design "Option B"). These tests pin:

* e2e numerics for ``+`` and ``max`` reductions (divisible trip);
* the recognition predicate is TIGHT -- it fires on genuine reductions and is a
  strict no-op (no rewrite) on elementwise / in-place-RMW / scalar-broadcast
  kernels (the regression-safety contract).
"""
import numpy
import pytest

import dace
from dace.transformation.passes.vectorization.tile_carried_reduction import (TileCarriedScalarReduction,
                                                                             find_carried_scalar_reductions)
from tests.passes.vectorization.helpers.harness import run_vectorization_test, N

pytestmark = pytest.mark.tile_nodes


@dace.program
def vsum(a: dace.float64[N], s: dace.float64[1]):
    acc = 0.0
    for i in dace.map[0:N]:
        acc = acc + a[i]
    s[0] = acc


@dace.program
def vmax(a: dace.float64[N], s: dace.float64[1]):
    acc = -1.0e308
    for i in dace.map[0:N]:
        acc = max(acc, a[i])
    s[0] = acc


@pytest.mark.xfail(reason="carried-reduction lowering WIP: being redesigned as a preprocessing pass "
                   "(in-place body + scalar-WCR) so the existing _convert_reduction handles it",
                   strict=False)
def test_reduction_sum_e2e():
    Nv = 64  # divisible by W=8 (milestone 1: no remainder)
    a = numpy.random.random((Nv, ))
    s = numpy.zeros((1, ))
    run_vectorization_test(dace_func=vsum,
                           arrays={
                               "a": a,
                               "s": s
                           },
                           params={"N": Nv},
                           vector_width=8,
                           sdfg_name="reduction_sum",
                           vectorize_config="tile_nodes",
                           remainder_strategy="scalar")


@pytest.mark.xfail(reason="carried-reduction lowering WIP: being redesigned as a preprocessing pass "
                   "(in-place body + scalar-WCR) so the existing _convert_reduction handles it",
                   strict=False)
def test_reduction_max_e2e():
    Nv = 64
    a = numpy.random.random((Nv, ))
    s = numpy.zeros((1, ))
    run_vectorization_test(dace_func=vmax,
                           arrays={
                               "a": a,
                               "s": s
                           },
                           params={"N": Nv},
                           vector_width=8,
                           sdfg_name="reduction_max",
                           vectorize_config="tile_nodes",
                           remainder_strategy="scalar")


# --- Recognition: fires on reductions, NO-OP on false positives ------------


@dace.program
def _elementwise(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N]):
    for i in dace.map[0:N]:
        c[i] = a[i] + b[i]


@dace.program
def _inplace_rmw(a: dace.float64[N], b: dace.float64[N]):
    for i in dace.map[0:N]:
        a[i] = a[i] + b[i]


@dace.program
def _axpy(a: dace.float64[1], x: dace.float64[N], y: dace.float64[N]):
    for i in dace.map[0:N]:
        y[i] = a[0] * x[i] + y[i]


def _reductions_found(prog, Nv=64):
    """Run the multi-dim prep up to WidenAccesses and return how many carried
    reductions the predicate finds (the point the pass actually runs)."""
    import dace.transformation.passes.vectorization.widen_accesses as WA
    from dace.transformation.passes.vectorization.vectorize_cpu_multi_dim import VectorizeCPUMultiDim
    captured = {}
    orig = WA.WidenAccesses.apply_pass

    def hook(self, sdfg, res):
        captured.setdefault("n", len(find_carried_scalar_reductions(sdfg, (8, ))))
        return orig(self, sdfg, res)

    WA.WidenAccesses.apply_pass = hook
    try:
        sdfg = prog.to_sdfg()
        try:
            VectorizeCPUMultiDim(widths=(8, ), remainder_strategy="scalar_postamble",
                                 expand_tile_nodes=False).apply_pass(sdfg, {})
        except Exception:  # noqa: BLE001 -- only the recognition count matters here
            pass
    finally:
        WA.WidenAccesses.apply_pass = orig
    return captured.get("n", -1)


@pytest.mark.parametrize("prog,expected", [
    (vsum, 1),
    (vmax, 1),
    (_elementwise, 0),
    (_inplace_rmw, 0),
    (_axpy, 0),
])
def test_recognition_is_tight(prog, expected):
    """The predicate must fire ONLY on genuine carried reductions -- a strict
    no-op on elementwise, in-place RMW (write subset depends on the map param),
    and scalar-broadcast reads."""
    assert _reductions_found(prog) == expected


def test_pass_does_not_apply_to_false_positives():
    """End-to-end no-rewrite check: the pass leaves a non-reduction kernel byte-
    identical (no TileReduce / partial-sum tile introduced)."""
    from dace.transformation.interstate import LoopToMap
    sdfg = _elementwise.to_sdfg()
    sdfg.simplify()
    sdfg.apply_transformations_repeated(LoopToMap, permissive=False)
    before = sdfg.to_json()
    applied = TileCarriedScalarReduction(widths=(8, )).apply_pass(sdfg, {})
    assert applied is None, "pass must be a no-op on a non-reduction kernel"
    assert sdfg.to_json() == before, "pass mutated a non-reduction SDFG"


if __name__ == "__main__":
    # e2e tests are xfail (carry lowering WIP); run the validated recognition checks.
    for p, e in [(vsum, 1), (vmax, 1), (_elementwise, 0), (_inplace_rmw, 0), (_axpy, 0)]:
        test_recognition_is_tight(p, e)
    test_pass_does_not_apply_to_false_positives()
    print("recognition + false-positive tests passed")
