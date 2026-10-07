# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Regression tests for DaCeCodeGenerator.free_symbols's id()-keyed cache
(dace/codegen/targets/framecode.py). The cache keys entries by id(obj)
without holding a reference to obj, and callers such as
cpu.py's CPU_Persistent map codegen mutate the returned set in place
with `|=`. Both are determinism hazards: a freed object's address can be
reused by an unrelated later object, and one caller's in-place union can
corrupt what a later, unrelated caller sees for the same cached object.
"""

import dace
from dace.codegen.targets.framecode import DaCeCodeGenerator


class SymbolBearer:
    """Minimal stand-in for the short-lived objects real callers pass to
    free_symbols (e.g. a ScopeSubgraphView built fresh per map node):
    used_symbols() returns a NEW mutable set on every call, exactly like
    StateSubgraphView.used_symbols in dace/sdfg/state.py."""

    __slots__ = ("_syms",)

    def __init__(self, syms):
        self._syms = set(syms)

    def used_symbols(self, all_symbols=False):
        return set(self._syms)


def test_free_symbols_cache_inplace_union_corrupts_entry():
    # Reproduces cpu.py's CPU_Persistent scope-symbol lookup:
    #     fsyms = self._frame.free_symbols(scope)
    #     fsyms |= e.data.used_symbols(False, e)
    # The `|=` must not corrupt what a later, unrelated caller sees when it
    # looks up free_symbols for the very same scope object again.
    sdfg = dace.SDFG("framecode_fsyms_mutation")
    frame = DaCeCodeGenerator(sdfg)

    scope = SymbolBearer({"a", "b"})

    fsyms = frame.free_symbols(scope)
    assert fsyms == {"a", "b"}

    # A caller unions in symbols found on an edge into the returned set.
    fsyms |= {"edge_only_symbol"}

    # A second, unrelated lookup of the SAME object must reflect what
    # used_symbols() actually computes for it, not the first caller's union.
    refreshed = frame.free_symbols(scope)
    assert "edge_only_symbol" not in refreshed, (
        f"free_symbols cache entry for {scope!r} was mutated in place by a "
        f"caller's `|=`; a later unrelated lookup returned {refreshed!r}"
    )
    assert refreshed == {"a", "b"}


if __name__ == "__main__":
    test_free_symbols_cache_inplace_union_corrupts_entry()
