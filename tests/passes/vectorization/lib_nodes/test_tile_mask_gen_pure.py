# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""End-to-end correctness for the ``pure`` expansion of :class:`TileMaskGen`.

The lib node materializes ``bool[widths]`` via the K-fold conjunction
``(iter_var_k + l_k < global_ub_k)``. ``iter_vars`` and ``global_ubs``
are symbols injected into the surrounding scope.
"""

import numpy as np
import pytest

import dace
from dace.libraries.tileops import TileMaskGen


def _build_maskgen_sdfg(widths, iter_vars, global_ubs, tag):
    """Build a minimal SDFG: TileMaskGen -> mask array.

    :param tag: Per-test discriminator so distinct tests building the
        same widths get unique SDFG names (no ``.dacecache`` collision).
    """
    sdfg = dace.SDFG(f"tile_mask_gen_pure_{'x'.join(str(w) for w in widths)}_{tag}")
    for iv in iter_vars:
        sdfg.add_symbol(iv, dace.int64)
    for ub in global_ubs:
        if ub not in sdfg.symbols and not ub.isdigit():
            sdfg.add_symbol(ub, dace.int64)
    sdfg.add_array("OUT", widths, dace.bool_, transient=False)

    state = sdfg.add_state("main")
    out_node = state.add_access("OUT")
    node = TileMaskGen(name="tmg", widths=widths, iter_vars=iter_vars, global_ubs=global_ubs)
    state.add_node(node)
    subset = ",".join(f"0:{w}" for w in widths)
    state.add_edge(node, "_o", out_node, None, dace.Memlet(f"OUT[{subset}]"))

    sdfg.expand_library_nodes()
    sdfg.validate()
    return sdfg


def test_tile_mask_gen_pure_1d_all_active():
    """When the trip aligns, every lane is active."""
    widths = (8,)
    sdfg = _build_maskgen_sdfg(widths, iter_vars=("i",), global_ubs=("N",), tag="all_active")
    OUT = np.zeros(widths, dtype=bool)
    sdfg(OUT=OUT, i=0, N=8)
    np.testing.assert_array_equal(OUT, np.ones(widths, dtype=bool))


def test_tile_mask_gen_pure_1d_partial_mask():
    """When ``i + l >= N``, the corresponding lane is inactive."""
    widths = (8,)
    sdfg = _build_maskgen_sdfg(widths, iter_vars=("i",), global_ubs=("N",), tag="partial")
    OUT = np.zeros(widths, dtype=bool)
    sdfg(OUT=OUT, i=0, N=5)
    ref = np.arange(8) < 5
    np.testing.assert_array_equal(OUT, ref)


def test_tile_mask_gen_pure_2d_any_dim_oob():
    """A lane is active iff (i + l_0 < M) AND (j + l_1 < N)."""
    widths = (4, 8)
    sdfg = _build_maskgen_sdfg(widths, iter_vars=("i", "j"), global_ubs=("M", "N"), tag="any_oob")
    OUT = np.zeros(widths, dtype=bool)
    sdfg(OUT=OUT, i=2, j=3, M=5, N=7)
    li = (np.arange(4) + 2)[:, None] < 5
    lj = (np.arange(8) + 3)[None, :] < 7
    ref = li & lj
    np.testing.assert_array_equal(OUT, ref)


@pytest.mark.parametrize("implementation", ["pure", "scalar"])
def test_tile_mask_gen_floor_division_bound(implementation):
    """A bound with ``//`` (``__int_floor`` in its printed form, as ``GenerateTileIterationMask`` stores it) compiles
    as C++ (dwt2d)."""
    sdfg = dace.SDFG(f"tile_mask_gen_floordiv_{implementation}")
    sdfg.add_symbol("i", dace.int64)
    sdfg.add_symbol("N", dace.int64)
    sdfg.add_array("OUT", (8,), dace.bool_)
    sdfg.add_transient("mask", (8,), dace.bool_, storage=dace.StorageType.Register)
    state = sdfg.add_state("main")
    bound = str(dace.symbolic.pystr_to_symbolic("N // 2"))
    assert "__int_floor" in bound
    node = TileMaskGen(name="tmg", widths=(8,), iter_vars=("i",), global_ubs=(bound,))
    node.implementation = implementation
    mask = state.add_access("mask")
    state.add_edge(node, "_o", mask, None, dace.Memlet("mask[0:8]"))
    state.add_edge(mask, None, state.add_access("OUT"), None, dace.Memlet("mask[0:8] -> [0:8]"))
    OUT = np.zeros(8, dtype=bool)
    sdfg(OUT=OUT, i=0, N=11)
    np.testing.assert_array_equal(OUT, np.arange(8) < 5)


def test_tile_mask_gen_rejects_length_mismatch():
    """Constructor refuses unequal widths / iter_vars / global_ubs lengths."""
    with pytest.raises(ValueError, match="lengths must agree"):
        TileMaskGen(name="bad", widths=(8, 4), iter_vars=("i",), global_ubs=("N",))
