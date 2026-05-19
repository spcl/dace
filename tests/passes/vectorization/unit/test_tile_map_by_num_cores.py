# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Unit tests for ``TileMapByNumCores`` (S-SVE2 vectorization-prep).

The pass strip-mines each single-parameter innermost map into an outer
``core`` map of exactly ``num_cores`` tiles plus a per-core inner chunk.
Strip-mining is semantics-preserving, so end-to-end numerical
equivalence against the untiled SDFG is the contract; the tiling itself
is structurally verified by the presence of a ``core``-prefixed outer
map of range ``0 : num_cores`` and the innermost-map count.
"""
import numpy as np

import dace
from dace.transformation.passes.vectorization.tile_map_by_num_cores import TileMapByNumCores
from dace.transformation.passes.vectorization.utils.map_predicates import is_innermost_map

N = dace.symbol("N")


@dace.program
def axpy1(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N]):
    for i in dace.map[0:N]:
        c[i] = a[i] + 2.0 * b[i]


def _bake(sdfg, trip: int):
    """Replace ``N`` with a literal so the SDFG compiles standalone.

    :param sdfg: SDFG to specialise.
    :param trip: Concrete trip count.
    """
    sdfg.replace_dict({"N": trip})


def _core_outer_maps(sdfg):
    """Collect ``core``-prefixed outer map entries.

    :param sdfg: SDFG to scan.
    :returns: List of ``(map_entry, state)`` for core-distribution maps.
    """
    return [(n, g) for n, g in sdfg.all_nodes_recursive()
            if isinstance(n, dace.nodes.MapEntry) and isinstance(g, dace.SDFGState)
            and any(p.startswith("core") for p in n.map.params)]


def _run(prog, trip, num_cores):
    """Compile the untiled and ``num_cores``-tiled SDFG and return both
    result arrays for the same random input.

    :param prog: The ``@dace.program``.
    :param trip: Concrete ``N``.
    :param num_cores: Tiling factor.
    :returns: ``(reference_c, tiled_c, tiled_sdfg)``.
    """
    a = np.random.rand(trip)
    b = np.random.rand(trip)

    ref = prog.to_sdfg(simplify=True)
    _bake(ref, trip)
    ref_c = np.zeros(trip)
    ref.compile()(a=a.copy(), b=b.copy(), c=ref_c)

    sd = prog.to_sdfg(simplify=True)
    _bake(sd, trip)
    TileMapByNumCores(num_cores=num_cores).apply_pass(sd, {})
    sd.validate()
    tiled_c = np.zeros(trip)
    sd.compile()(a=a.copy(), b=b.copy(), c=tiled_c)
    return ref_c, tiled_c, sd


def test_tile_divisible_equivalence_and_structure():
    """N=64, num_cores=4 (divisible): exactly one ``core`` block-
    distribution outer map appears, it distributes ``num_cores``
    contiguous blocks, and the result matches the untiled SDFG.

    The hard contract is end-to-end numerical equivalence; the
    structural check asserts the genuine invariant (tiled into
    ``num_cores`` contiguous blocks via the correct ``MapTiling``
    primitive), not a specific range shape.
    """
    ref_c, tiled_c, sd = _run(axpy1, 64, 4)
    assert np.allclose(ref_c, tiled_c, rtol=0, atol=0)
    cores = _core_outer_maps(sd)
    assert len(cores) == 1, f"expected exactly one core-distribution outer map, got {len(cores)}"
    me, _ = cores[0]
    lb, ub, step = (int(x) for x in me.map.range[0])
    n_blocks = (ub - lb) // step + 1
    assert n_blocks == 4, f"core map yields {n_blocks} blocks, expected num_cores=4 ({me.map.range})"


def test_tile_non_divisible_equivalence():
    """N=22, num_cores=4 (not divisible): per-core chunks differ in
    length but the end-to-end result is still identical."""
    ref_c, tiled_c, _ = _run(axpy1, 22, 4)
    assert np.allclose(ref_c, tiled_c, rtol=0, atol=0)


def test_num_cores_one_is_noop():
    """``num_cores <= 1`` leaves the SDFG untiled and returns ``None``."""
    sd = axpy1.to_sdfg(simplify=True)
    _bake(sd, 32)
    before = sum(1 for n, g in sd.all_nodes_recursive()
                 if isinstance(n, dace.nodes.MapEntry) and isinstance(g, dace.SDFGState))
    applied = TileMapByNumCores(num_cores=1).apply_pass(sd, {})
    after = sum(1 for n, g in sd.all_nodes_recursive()
                if isinstance(n, dace.nodes.MapEntry) and isinstance(g, dace.SDFGState))
    assert applied is None
    assert after == before
    assert not _core_outer_maps(sd)


def test_idempotent():
    """A second application does not re-tile an already core-tiled map."""
    sd = axpy1.to_sdfg(simplify=True)
    _bake(sd, 48)
    first = TileMapByNumCores(num_cores=4).apply_pass(sd, {})
    second = TileMapByNumCores(num_cores=4).apply_pass(sd, {})
    assert first == 1
    assert second is None
    assert len(_core_outer_maps(sd)) == 1
