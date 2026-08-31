# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""One symbol name must be ONE symbol, whatever spelling it is recovered from.

A symbol's dtype is part of its identity, so a name carried at two widths is two sympy symbols and
``N - N`` never cancels -- not even under ``simplify``. That is easy to produce by accident: an
array's shape keeps the dtype the program declared, while a loop bound lives in a string-backed
property and is RE-PARSED, which used to mint ``DEFAULT_SYMBOL_TYPE`` from the bare name. Every
proof comparing a bound against a subset then failed on a well-formed program.
"""
import dace
from dace import dtypes, symbolic


def test_an_unscoped_parse_keeps_the_default():
    """The authority only ever overrides; with none active nothing changes."""
    parsed = symbolic.pystr_to_symbolic('N + 1')
    sym = next(iter(parsed.free_symbols))
    assert sym.dtype == symbolic.DEFAULT_SYMBOL_TYPE


def test_a_scoped_parse_takes_the_declared_dtype():
    with symbolic.serialization_symbol_dtypes({'N': dtypes.int64}):
        parsed = symbolic.pystr_to_symbolic('N + 1')
    sym = next(iter(parsed.free_symbols))
    assert sym.dtype == dtypes.int64, f're-parsed symbol was stamped {sym.dtype}'


def test_a_scoped_parse_cancels_against_the_declared_symbol():
    """The property that actually matters: one name, one symbol, so the difference is zero."""
    declared = symbolic.symbol('N', dtype=dtypes.int64)
    with symbolic.serialization_symbol_dtypes({'N': dtypes.int64}):
        reparsed = symbolic.pystr_to_symbolic('N - 1')
    assert symbolic.simplify(reparsed - (declared - 1)) == 0


def test_the_parse_cache_is_keyed_on_the_authority():
    """The same text under two scopes must not return the first scope's symbol."""
    with symbolic.serialization_symbol_dtypes({'M': dtypes.int32}):
        narrow = symbolic.pystr_to_symbolic('M')
    with symbolic.serialization_symbol_dtypes({'M': dtypes.int64}):
        wide = symbolic.pystr_to_symbolic('M')
    assert narrow.dtype == dtypes.int32
    assert wide.dtype == dtypes.int64, 'the cache served the previous scope\'s symbol'


def test_a_canonicalized_graph_spells_each_name_once():
    """End to end: after canonicalization no name appears under two dtypes."""
    N = dace.symbol('N', dtype=dace.int64)

    @dace.program
    def two_dim(aa: dace.float64[N, N], bb: dace.float64[N, N]):
        for i in range(N):
            for j in range(N):
                aa[i, j] = bb[i, j] + 1.0

    sdfg = two_dim.to_sdfg(simplify=False)
    from dace.transformation.passes.canonicalize import pipeline as canon
    canon.canonicalize(sdfg)

    seen: dict = {}
    for g in sdfg.all_sdfgs_recursive():
        for desc in g.arrays.values():
            for expr in list(desc.shape) + list(desc.strides):
                for sym in getattr(expr, 'free_symbols', ()):
                    seen.setdefault(str(sym), set()).add(str(sym.dtype))
        for state in g.all_states():
            for edge in state.edges():
                if edge.data is None or edge.data.subset is None:
                    continue
                for rng in edge.data.subset.ranges:
                    for expr in rng:
                        for sym in getattr(expr, 'free_symbols', ()):
                            seen.setdefault(str(sym), set()).add(str(sym.dtype))
    split = {name: kinds for name, kinds in seen.items() if len(kinds) > 1}
    assert not split, f'names carried at more than one dtype: {split}'


if __name__ == '__main__':
    test_an_unscoped_parse_keeps_the_default()
    test_a_scoped_parse_takes_the_declared_dtype()
    test_a_scoped_parse_cancels_against_the_declared_symbol()
    test_the_parse_cache_is_keyed_on_the_authority()
    test_a_canonicalized_graph_spells_each_name_once()
