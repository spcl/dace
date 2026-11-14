# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""One symbol name must be ONE symbol, whatever spelling it is recovered from.

Symbol identity is the name, as on main: a shape that keeps the declared dtype and a loop bound
re-parsed from its string property name the same symbol, so ``N - N`` cancels and every proof
comparing a bound against a subset sees one ``N``.
"""

from dace import dtypes, symbolic


def test_an_unscoped_parse_keeps_the_default():
    """The authority only ever overrides; with none active it declares nothing, so a name keeps the type it was
    declared with. (Not read off a parsed object: a symbol is its name, and SymPy's cache may hand back an ``N`` built
    at another dtype elsewhere.)"""
    symbolic.pystr_to_symbolic("N + 1")
    assert not symbolic.symbol_dtype_authority_active()
    assert symbolic.declared_symbol_dtype("N") is None
    assert symbolic.symbol_dtype(symbolic.symbol("N")) == symbolic.DEFAULT_SYMBOL_TYPE


def test_a_scoped_parse_cancels_against_the_declared_symbol():
    """The property that actually matters: one name, one symbol, so the difference is zero."""
    declared = symbolic.symbol("N", dtype=dtypes.int64)
    with symbolic.serialization_symbol_dtypes({"N": dtypes.int64}):
        reparsed = symbolic.pystr_to_symbolic("N - 1")
    assert symbolic.simplify(reparsed - (declared - 1)) == 0


def test_the_parse_cache_is_keyed_on_the_authority():
    """The same text under two scopes must not return the first scope's symbol."""
    with symbolic.serialization_symbol_dtypes({"M": dtypes.int32}):
        narrow = symbolic.pystr_to_symbolic("M")
    with symbolic.serialization_symbol_dtypes({"M": dtypes.int64}):
        wide = symbolic.pystr_to_symbolic("M")
    assert narrow.dtype == dtypes.int32
    assert wide.dtype == dtypes.int64, "the cache served the previous scope's symbol"


def test_one_authority_pushed_twice_is_served_from_the_parse_cache():
    with symbolic.serialization_symbol_dtypes({"K": dtypes.int64}):
        first = symbolic.pystr_to_symbolic("K + 7")
    with symbolic.serialization_symbol_dtypes({"K": dtypes.int64}):
        second = symbolic.pystr_to_symbolic("K + 7")
    assert second is first


if __name__ == "__main__":
    test_an_unscoped_parse_keeps_the_default()
    test_a_scoped_parse_cancels_against_the_declared_symbol()
    test_the_parse_cache_is_keyed_on_the_authority()
    test_one_authority_pushed_twice_is_served_from_the_parse_cache()
