# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``MarkTileDims`` trip-size contract.

The K-dim vectorization pipeline does NOT require ``trip >= W`` on a tiled
inner dim. Under the unified "no-mask interior + w-mask remainder" model a
symbolic, or per-iteration-varying (e.g. wavefront) trip is handled by
masking -- or, under ``scalar_postamble``, by the scalar remainder loop. So:

1. A symbolic trip ``N`` is classified (a spec is recorded) and NO runtime
   guard state is planted. (Earlier designs planted a ``std::abort``
   ``N >= W`` guard at SDFG entry; it traps spuriously on wavefront trips that
   depend on an outer-loop iterator -- undefined at entry -- and is unnecessary
   because the mask/remainder already handle ``trip < W`` correctly.)
2. A static trip == W is classified, no guard.
3. A static trip < W is NOT classified: a constant extent below the vector
   width can never fill a tile even at runtime, so the map stays scalar
   instead of paying the interior/remainder split + mask machinery for a
   width-8 tile on an extent-3 axis (gem). ``SplitMapForTileRemainder``
   agrees: its constant-below-width class leaves the map whole and unmarked.
4. End to end, a symbolic-trip kernel run with ``N < W`` produces CORRECT
   results (no trap) for BOTH the masking path (``full_mask``) and the scalar
   remainder path (``scalar_postamble``).
"""

import numpy as np
import pytest

import dace
from dace.libraries.tileops.dispatch import detect_host_isa
from dace.transformation.passes.vectorization.config import VectorizeConfig
from dace.transformation.passes.vectorization.enums import ISA, BranchMode, RemainderStrategy
from dace.transformation.passes.vectorization.mark_tile_dims import MarkTileDims
from dace.transformation.passes.vectorization.vectorize_cpu_multi_dim import VectorizeCPUMultiDim

GUARD_STATE_LABEL = "_tile_runtime_check"  # the now-removed guard's state label
#: The host's best runnable SIMD ISA; vectorization enforces arch-native, so a hardcoded AVX-512
#: would SIGILL-refuse on an AVX2-only or ARM host.
HOST_ISA = detect_host_isa()


def _build_inner_map_sdfg(name: str, trip):
    """Build a single-state SDFG with one innermost map iterating ``i`` over
    ``[0, trip)``. ``trip`` may be a Python int OR a :class:`dace.symbol`."""
    sdfg = dace.SDFG(name)
    if isinstance(trip, int):
        ub = trip - 1
        sdfg.add_array("a", [trip], dtype=dace.float64, transient=False)
    else:
        N = trip
        ub = N - 1
        sdfg.add_symbol(str(N), dace.int64)
        sdfg.add_array("a", [N], dtype=dace.float64, transient=False)
    state = sdfg.add_state("main", is_start_block=True)
    state.add_mapped_tasklet(
        name="kern",
        map_ranges={"i": dace.subsets.Range([(0, ub, 1)])},
        inputs={},
        code="_out = 0.0",
        outputs={"_out": dace.memlet.Memlet("a[i]")},
        external_edges=True,
    )
    return sdfg


def _guard_states(sdfg):
    return [s for s in sdfg.nodes() if isinstance(s, dace.SDFGState) and s.label == GUARD_STATE_LABEL]


def test_mark_tile_dims_no_guard_for_symbolic_trip():
    """Symbolic trip ``N``: a spec is recorded and NO ``std::abort`` guard
    state is planted -- the mask/remainder handles ``trip < W`` at runtime."""
    N = dace.symbol("N")
    sdfg = _build_inner_map_sdfg("symbolic_trip", N)
    res = MarkTileDims(widths=(8,)).apply_pass(sdfg, {})
    assert res is not None, "MarkTileDims should classify the symbolic-trip map"
    assert not _guard_states(sdfg), "no runtime trip guard must be planted for a symbolic trip"
    # And no tasklet anywhere calls std::abort.
    for s in sdfg.states():
        for n in s.nodes():
            if isinstance(n, dace.nodes.Tasklet):
                assert "std::abort" not in n.code.as_string


def test_mark_tile_dims_no_guard_for_static_trip_at_or_above_width():
    """Static trip == W: classified, no guard."""
    sdfg = _build_inner_map_sdfg("static_trip_eq_w", 8)
    res = MarkTileDims(widths=(8,)).apply_pass(sdfg, {})
    assert res is not None
    assert not _guard_states(sdfg)


def test_mark_tile_dims_skips_constant_trip_below_width():
    """Static trip < W: NOT spec'd. A constant extent below the vector width can never fill a
    tile even at runtime, so the map stays scalar instead of paying the split + mask machinery
    for a width-8 tile on an extent-3 axis (gem). A symbolic trip stays tiled (the masked tail
    handles a runtime short trip); see ``test_mark_tile_dims_no_guard_for_symbolic_trip``."""
    sdfg = _build_inner_map_sdfg("static_trip_below_w", 4)
    res = MarkTileDims(widths=(8,)).apply_pass(sdfg, {})
    assert res is None, "a constant trip < W must stay scalar, not masked-tile"
    assert not _guard_states(sdfg)


def test_split_map_agrees_a_constant_trip_below_width_stays_untiled():
    """SplitMapForTileRemainder must leave the same map whole and unmarked, or the two passes
    disagree about which maps are tiled and a strided map ends up with a scalar body."""
    from dace.transformation.passes.vectorization.split_map_for_tile_remainder import (
        REGION_MARKERS,
        SplitMapForTileRemainder,
    )

    sdfg = _build_inner_map_sdfg("static_trip_below_w_split", 4)
    assert SplitMapForTileRemainder(widths=(8,), tail_mode="masked").apply_pass(sdfg, {}) is None
    (state,) = sdfg.states()
    (map_entry,) = (n for n in state.nodes() if isinstance(n, dace.nodes.MapEntry))
    assert not map_entry.map.label.endswith(REGION_MARKERS), (
        f"the below-width map must stay whole, got {map_entry.map.label!r}"
    )
    assert [(r[0], r[1], r[2]) for r in map_entry.map.range.ranges] == [(0, 3, 1)], "the map range must be untouched"


@pytest.mark.parametrize(
    "strat,isa", [(RemainderStrategy.FULL_MASK, HOST_ISA), (RemainderStrategy.SCALAR_POSTAMBLE, ISA.SCALAR)]
)
@pytest.mark.parametrize("n", [3, 5, 7])
def test_symbolic_trip_below_width_runs_correctly(strat, isa, n):
    """A symbolic-trip kernel run with ``N < W`` produces correct results -- the
    masking path (``full_mask``) and the scalar remainder path
    (``scalar_postamble``) both handle ``trip < W`` without a trap.

    This is the contract that lets the runtime ``trip >= W`` guard be removed:
    masking / scalar remainder are correct for a short trip, so the guard was
    pure (harmful) defensiveness."""
    N = dace.symbol("N")

    @dace.program
    def k(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N]):
        for i in dace.map[0:N]:
            a[i] = b[i] * 2.0 + c[i]

    sdfg = k.to_sdfg(simplify=True)
    VectorizeCPUMultiDim(
        VectorizeConfig(widths=(8,), target_isa=isa, remainder_strategy=strat, branch_mode=BranchMode.MERGE)
    ).apply_pass(sdfg, {})
    sdfg.validate()
    rng = np.random.default_rng(n)
    b = rng.standard_normal(n)
    c = rng.standard_normal(n)
    a = np.zeros(n)
    ref = b * 2.0 + c
    sdfg.compile()(a=a, b=b.copy(), c=c.copy(), N=n)
    assert np.allclose(a, ref), f"{strat}/N={n}: max|d|={np.max(np.abs(a - ref)):.3e}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
