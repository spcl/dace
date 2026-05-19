# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
""" Guard placement under canonicalization (SC26 layout-artifact shape).

    ``map i: if foo(i): map j: ...`` -- a guard between two map levels whose
    condition depends on the outer map index. Canonicalization pushes the
    guard inward (exposing the i/j maps for collapsing) but must NOT hoist
    it back above ``map i``: its condition references ``i``, so it cannot
    move past the map that defines ``i``. The surviving conditional must
    therefore stay inside the map nest (no top-level ConditionalBlock) and
    its condition must still depend on the map iterator. Value-preserving
    against a pure-numpy oracle.
"""
import numpy as np
import pytest

import dace
from dace.sdfg import nodes
from dace.sdfg.state import ConditionalBlock
from dace.transformation.passes.canonicalize import canonicalize

N = dace.symbol('N')
M = dace.symbol('M')


@dace.program
def guard_between_maps(a: dace.float64[N, M], b: dace.float64[N, M]):
    for i in dace.map[0:N]:
        if i % 2 == 0:  # depends on the outer map index i
            for j in dace.map[0:M]:
                b[i, j] = a[i, j] * 2.0


def _conds(sdfg):
    return [r for r in sdfg.all_control_flow_regions(recursive=True) if isinstance(r, ConditionalBlock)]


def _map_params(sdfg):
    ps = set()
    for st in sdfg.all_states():
        for n in st.nodes():
            if isinstance(n, nodes.MapEntry):
                ps.update(str(p) for p in n.map.params)
    return ps


def test_index_dependent_guard_stays_inside_map_nest():
    n, m = 8, 6
    a = np.random.rand(n, m)
    exp = np.full((n, m), 7.0)
    for i in range(n):
        if i % 2 == 0:
            for j in range(m):
                exp[i, j] = a[i, j] * 2.0

    sdfg = guard_between_maps.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True)

    cbs = _conds(sdfg)
    assert len(cbs) >= 1, "the index-dependent guard must survive"
    # Not hoisted above the map nest: no conditional at SDFG top level.
    assert not [c for c in sdfg.nodes() if isinstance(c, ConditionalBlock)], \
        "index-dependent guard was hoisted to SDFG top level"
    # Its condition still references a map iterator -> it provably cannot be
    # lifted past the map that defines that index.
    map_params = _map_params(sdfg)
    cond_syms = set()
    for cb in cbs:
        for cond, _ in cb.branches:
            if cond is not None:
                cond_syms.update(str(s) for s in cond.get_free_symbols())
    assert cond_syms & map_params, \
        f"guard condition {cond_syms} no longer depends on a map index {map_params}"

    out = np.full((n, m), 7.0)
    sdfg(a=a.copy(), b=out, N=n, M=m)
    assert np.allclose(out, exp)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
