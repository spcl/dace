# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Unit coverage for ``tile_mask_gen_dominates_consumers``.

The iteration-mask producer (:class:`TileMaskGen`) must sit in its SDFG's start
block so it dominates every masked consumer. A data-dependent ``if`` (-> TileITE)
body otherwise reads ``_tile_iter_mask`` from a non-dominating branch state
(uninitialized lanes). This pins the post-condition ``GenerateTileIterationMask``
enforces after emitting the mask in its ``_tile_mask_init`` start state.
"""
import dace
from dace.libraries.tileops import TileMaskGen
from dace.transformation.passes.vectorization.utils.pass_invariants import tile_mask_gen_dominates_consumers


def _two_state_sdfg(mask_in_start):
    """SDFG with states ``s0`` (start) -> ``s1``; place the TileMaskGen in
    ``s0`` when ``mask_in_start`` else in the non-dominating ``s1``."""
    sdfg = dace.SDFG("mask_dom")
    sdfg.add_array("m", [8], dace.bool_, storage=dace.dtypes.StorageType.Register, transient=True)
    s0 = sdfg.add_state("s0", is_start_block=True)
    s1 = sdfg.add_state("s1")
    sdfg.add_edge(s0, s1, dace.InterstateEdge())
    host = s0 if mask_in_start else s1
    node = TileMaskGen(name="mg", widths=(8, ), iter_vars=("i", ), global_ubs=("N", ))
    host.add_node(node)
    acc = host.add_access("m")
    host.add_edge(node, "_o", acc, None, dace.Memlet("m[0:8]"))
    return sdfg


def test_mask_in_start_block_passes():
    """Producer in the start block satisfies the invariant (returns None)."""
    assert tile_mask_gen_dominates_consumers(_two_state_sdfg(mask_in_start=True)) is None


def test_mask_outside_start_block_violates():
    """Producer in a non-dominating branch state is reported."""
    violation = tile_mask_gen_dominates_consumers(_two_state_sdfg(mask_in_start=False))
    assert violation is not None
    assert "start block" in violation


def test_no_mask_is_vacuously_ok():
    """An SDFG with no TileMaskGen trivially satisfies the invariant."""
    sdfg = dace.SDFG("no_mask")
    sdfg.add_state("only", is_start_block=True)
    assert tile_mask_gen_dominates_consumers(sdfg) is None
