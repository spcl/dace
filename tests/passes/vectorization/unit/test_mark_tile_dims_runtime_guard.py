# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``MarkTileDims`` trip-size contract.

The K-dim vectorization pipeline does NOT require ``trip >= W`` on a tiled
inner dim. Under the unified "no-mask interior + w-mask remainder" model a
short, symbolic, or per-iteration-varying (e.g. wavefront) trip is handled by
masking -- or, under ``scalar_postamble``, by the scalar remainder loop. So:

1. A symbolic trip ``N`` is classified (a spec is recorded) and NO runtime
   guard state is planted. (Earlier designs planted a ``__builtin_trap``
   ``N >= W`` guard at SDFG entry; it traps spuriously on wavefront trips that
   depend on an outer-loop iterator -- undefined at entry -- and is unnecessary
   because the mask/remainder already handle ``trip < W`` correctly.)
2. A static trip == W is classified, no guard.
3. A static trip < W is classified (masked-tail), no guard.
4. End to end, a symbolic-trip kernel run with ``N < W`` produces CORRECT
   results (no trap) for BOTH the masking path (``full_mask``) and the scalar
   remainder path (``scalar_postamble``).
"""
import numpy as np
import pytest

import dace
from dace.transformation.passes.vectorization.config import VectorizeConfig
from dace.transformation.passes.vectorization.enums import BranchMode
from dace.transformation.passes.vectorization.mark_tile_dims import MarkTileDims
from dace.transformation.passes.vectorization.vectorize_cpu_multi_dim import VectorizeCPUMultiDim

_GUARD_STATE_LABEL = "_tile_runtime_check"  # the now-removed guard's state label


def _build_inner_map_sdfg(name: str, trip):
    """Build a single-state SDFG with one innermost map iterating ``i`` over
    ``[0, trip)``. ``trip`` may be a Python int OR a :class:`dace.symbol`."""
    sdfg = dace.SDFG(name)
    if isinstance(trip, int):
        ub = trip - 1
        sdfg.add_array('a', [trip], dtype=dace.float64, transient=False)
    else:
        N = trip
        ub = N - 1
        sdfg.add_symbol(str(N), dace.int64)
        sdfg.add_array('a', [N], dtype=dace.float64, transient=False)
    state = sdfg.add_state('main', is_start_block=True)
    state.add_mapped_tasklet(
        name='kern',
        map_ranges={'i': dace.subsets.Range([(0, ub, 1)])},
        inputs={},
        code='_out = 0.0',
        outputs={'_out': dace.memlet.Memlet('a[i]')},
        external_edges=True,
    )
    return sdfg


def _guard_states(sdfg):
    return [s for s in sdfg.nodes() if isinstance(s, dace.SDFGState) and s.label == _GUARD_STATE_LABEL]


def test_mark_tile_dims_no_guard_for_symbolic_trip():
    """Symbolic trip ``N``: a spec is recorded and NO ``__builtin_trap`` guard
    state is planted -- the mask/remainder handles ``trip < W`` at runtime."""
    N = dace.symbol('N')
    sdfg = _build_inner_map_sdfg('symbolic_trip', N)
    res = MarkTileDims(widths=(8, )).apply_pass(sdfg, {})
    assert res is not None, "MarkTileDims should classify the symbolic-trip map"
    assert not _guard_states(sdfg), 'no runtime trip guard must be planted for a symbolic trip'
    # And no tasklet anywhere calls __builtin_trap.
    for s in sdfg.states():
        for n in s.nodes():
            if isinstance(n, dace.nodes.Tasklet):
                assert '__builtin_trap' not in n.code.as_string


def test_mark_tile_dims_no_guard_for_static_trip_at_or_above_width():
    """Static trip == W: classified, no guard."""
    sdfg = _build_inner_map_sdfg('static_trip_eq_w', 8)
    res = MarkTileDims(widths=(8, )).apply_pass(sdfg, {})
    assert res is not None
    assert not _guard_states(sdfg)


def test_mark_tile_dims_specs_static_trip_below_width_for_masked_tail():
    """Static trip < W: under the unified model the dim is STILL tiled (single
    masked remainder tile, empty interior), so a spec is recorded and no guard
    is needed (the mask handles the short dim).

    Regression contract: the prior design soft-skipped ``trip < W`` and returned
    ``None``; the masked-tail model removed that skip -- a trip-4 / W-8 kernel
    vectorises correctly via the masked tail rather than dropping to sequential
    codegen."""
    sdfg = _build_inner_map_sdfg('static_trip_below_w', 4)
    res = MarkTileDims(widths=(8, )).apply_pass(sdfg, {})
    assert res is not None, "static trip < W must be spec'd (masked-tail), not soft-skipped"
    specs = list(res.values())
    assert len(specs) == 1, f"expected one spec for the single inner map, got {len(specs)}"
    assert specs[0].iter_vars == ('i', ) and specs[0].widths == (8, )
    assert not _guard_states(sdfg)


@pytest.mark.parametrize("strat,isa", [("full_mask", "AVX512"), ("scalar_postamble", "SCALAR")])
@pytest.mark.parametrize("n", [3, 5, 7])
def test_symbolic_trip_below_width_runs_correctly(strat, isa, n):
    """A symbolic-trip kernel run with ``N < W`` produces correct results -- the
    masking path (``full_mask``) and the scalar remainder path
    (``scalar_postamble``) both handle ``trip < W`` without a trap.

    This is the contract that lets the runtime ``trip >= W`` guard be removed:
    masking / scalar remainder are correct for a short trip, so the guard was
    pure (harmful) defensiveness."""
    N = dace.symbol('N')

    @dace.program
    def k(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N]):
        for i in dace.map[0:N]:
            a[i] = b[i] * 2.0 + c[i]

    sdfg = k.to_sdfg(simplify=True)
    VectorizeCPUMultiDim(
        VectorizeConfig(widths=(8, ), target_isa=isa, remainder_strategy=strat,
                        branch_mode=BranchMode.MERGE)).apply_pass(sdfg, {})
    sdfg.validate()
    rng = np.random.default_rng(n)
    b = rng.standard_normal(n)
    c = rng.standard_normal(n)
    a = np.zeros(n)
    ref = b * 2.0 + c
    sdfg.compile()(a=a, b=b.copy(), c=c.copy(), N=n)
    assert np.allclose(a, ref), f"{strat}/N={n}: max|d|={np.max(np.abs(a - ref)):.3e}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
