# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""End-to-end correctness for the ``pure`` expansion of
:class:`TileBroadcastSymbol`.
"""
import numpy as np
import pytest

import dace
from dace.libraries.tileops import TileBroadcastSymbol


def _build_bsym_sdfg(widths, expr, dtype=dace.float64, free_symbols=()):
    """Build a minimal SDFG: TileBroadcastSymbol -> tile array."""
    sdfg = dace.SDFG(f"tile_bsym_pure_{'x'.join(str(w) for w in widths)}")
    for sym in free_symbols:
        sdfg.add_symbol(sym, dace.float64 if dtype == dace.float64 else dace.int64)
    sdfg.add_array("OUT", widths, dtype, transient=False)

    state = sdfg.add_state("main")
    out_node = state.add_access("OUT")
    node = TileBroadcastSymbol(name="tbs", widths=widths, expr=expr)
    state.add_node(node)
    subset = ",".join(f"0:{w}" for w in widths)
    state.add_edge(node, "_c", out_node, None, dace.Memlet(f"OUT[{subset}]"))

    sdfg.expand_library_nodes()
    sdfg.validate()
    return sdfg


@pytest.mark.parametrize("widths", [(8,), (4, 8), (2, 4, 8)])
def test_tile_broadcast_symbol_pure_literal_constant(widths):
    """Splatting a numeric literal fills every lane with that value."""
    sdfg = _build_bsym_sdfg(widths, expr="3.5")
    OUT = np.zeros(widths)
    sdfg(OUT=OUT)
    np.testing.assert_allclose(OUT, np.full(widths, 3.5), rtol=0, atol=0)


def test_tile_broadcast_symbol_pure_free_symbol():
    """Splatting a free symbol resolves to the runtime value supplied."""
    widths = (4, 8)
    sdfg = _build_bsym_sdfg(widths, expr="alpha", free_symbols=("alpha",))
    OUT = np.zeros(widths)
    sdfg(OUT=OUT, alpha=1.25)
    np.testing.assert_allclose(OUT, np.full(widths, 1.25), rtol=0, atol=0)


def test_tile_broadcast_symbol_rejects_empty_expr():
    """Constructor refuses an empty expression."""
    with pytest.raises(ValueError, match="expr must be non-empty"):
        TileBroadcastSymbol(name="bad", widths=(8,), expr="")
