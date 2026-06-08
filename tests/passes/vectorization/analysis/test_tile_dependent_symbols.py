# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Unit tests for G4 -- tile-dependent symbol classifier (design section 4.2).

Covers:

* ``_is_tile_dependent`` direct check (symbol is itself a tile iter-var).
* ``_is_tile_dependent`` one-hop transitive (sym <- i + 3 -> tile-dependent).
* ``_is_tile_dependent`` two-hop transitive (sym2 <- syma + 1, sym <- i -> tile-dep).
* ``_is_tile_dependent`` outer-scope (sym <- N + 1 -> not tile-dependent).
* ``classify_symbols`` per-symbol map shape.
* End-to-end classifier integration: ``a[2*syma + 1]`` with ``sym`` tile-dep
  classifies as GATHER instead of CONSTANT/AFFINE.
"""
import dace
from dace.subsets import Range
from dace.transformation.passes.vectorization.utils.tile_access import (
    PerDimKind,
    _is_tile_dependent,
    classify_symbols,
    classify_tile_access,
)


def _build_body_with_assign(symbol_name: str, rhs: str) -> dace.SDFG:
    """Build a body NSDFG with one interstate edge assigning ``rhs`` to
    ``symbol_name`` (so ``_is_tile_dependent`` has something to walk)."""
    sdfg = dace.SDFG("body")
    sdfg.add_symbol(symbol_name, dace.int64)
    s0 = sdfg.add_state("entry", is_start_block=True)
    s1 = sdfg.add_state("after")
    sdfg.add_edge(s0, s1, dace.InterstateEdge(assignments={symbol_name: rhs}))
    return sdfg


def test_direct_iter_var_is_tile_dependent():
    """A tile iter-var is trivially tile-dependent (no interstate walk needed)."""
    assert _is_tile_dependent("i", {"i", "j"}, inner_sdfg=None) is True


def test_outer_scope_symbol_is_not_tile_dependent():
    """A symbol with no interstate definition is outer-scope; not tile-dep."""
    sdfg = dace.SDFG("body")
    sdfg.add_state("s")
    assert _is_tile_dependent("N", {"i"}, inner_sdfg=sdfg) is False


def test_one_hop_transitive_is_tile_dependent():
    """``sym <- i + 3`` makes ``sym`` tile-dependent."""
    sdfg = _build_body_with_assign("syma", "i + 3")
    assert _is_tile_dependent("syma", {"i", "j"}, inner_sdfg=sdfg) is True


def test_two_hop_transitive_is_tile_dependent():
    """``sym2 <- syma + 1`` and ``sym <- j`` makes both transitive tile-deps."""
    sdfg = dace.SDFG("body")
    for s in ("syma", "sym2"):
        sdfg.add_symbol(s, dace.int64)
    s0 = sdfg.add_state("entry", is_start_block=True)
    s1 = sdfg.add_state("mid")
    s2 = sdfg.add_state("after")
    sdfg.add_edge(s0, s1, dace.InterstateEdge(assignments={"syma": "j"}))
    sdfg.add_edge(s1, s2, dace.InterstateEdge(assignments={"sym2": "syma + 1"}))
    assert _is_tile_dependent("syma", {"i", "j"}, inner_sdfg=sdfg) is True
    assert _is_tile_dependent("sym2", {"i", "j"}, inner_sdfg=sdfg) is True


def test_outer_scope_chain_stays_outer():
    """``sym <- N + 1`` (no tile iter-var anywhere) stays non-tile-dependent."""
    sdfg = _build_body_with_assign("syma", "N + 1")
    sdfg.add_symbol("N", dace.int64)
    assert _is_tile_dependent("syma", {"i", "j"}, inner_sdfg=sdfg) is False


def test_classify_symbols_returns_per_symbol_map():
    """``classify_symbols`` returns ``{name: bool}`` for every free symbol."""
    sdfg = _build_body_with_assign("syma", "i + 3")
    sdfg.add_symbol("N", dace.int64)
    expr = dace.symbolic.pystr_to_symbolic("2*syma + N + 1")
    out = classify_symbols(expr, iter_vars=("i", "j"), inner_sdfg=sdfg)
    assert out.get("syma") is True
    assert out.get("N") is False


def test_classifier_promotes_tile_dependent_symbol_to_gather():
    """End-to-end: ``a[2*syma + 1]`` with ``sym <- i + 3`` classifies as GATHER
    (would otherwise look CONSTANT because ``sym`` isn't a direct tile iter-var)."""
    sdfg = _build_body_with_assign("syma", "i + 3")
    subset = Range([(dace.symbolic.pystr_to_symbolic("2*syma + 1"), dace.symbolic.pystr_to_symbolic("2*syma + 1"), 1)])
    record = classify_tile_access(subset, iter_vars=("i", ), inner_sdfg=sdfg)
    assert record.per_dim_kind == (PerDimKind.GATHER, )


def test_classifier_outer_scope_symbol_does_not_force_gather():
    """``a[2*N + 1]`` with ``N`` outer stays CONSTANT (no tile-dep promotion)."""
    sdfg = dace.SDFG("body")
    sdfg.add_symbol("N", dace.int64)
    sdfg.add_state("s")
    subset = Range([(dace.symbolic.pystr_to_symbolic("2*N + 1"), dace.symbolic.pystr_to_symbolic("2*N + 1"), 1)])
    record = classify_tile_access(subset, iter_vars=("i", ), inner_sdfg=sdfg)
    assert record.per_dim_kind == (PerDimKind.CONSTANT, )
