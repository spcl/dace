# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``LoopToMap`` in a nested SDFG forwards the parent's symbols to the new loop body only where the body uses them;
a forwarded symbol the body never reads fails validation with an "unused symbol" warning."""

import warnings

import numpy as np

import dace
from dace.transformation.interstate.loop_to_map import LoopToMap

N = dace.symbol("N")
M = dace.symbol("M")


@dace.program
def inner(a: dace.float64[N], b: dace.float64[M]):
    for i in range(N):
        a[i] = a[i] + 1.0
    b[0] = 2.0


@dace.program
def outer(a: dace.float64[N], b: dace.float64[M]):
    inner(a, b)


def test_loop_body_maps_only_the_symbols_it_uses():
    sdfg = outer.to_sdfg(simplify=False)
    nsdfg = next(n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.NestedSDFG))
    assert nsdfg.symbol_mapping.keys() == {"N", "M"}

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert nsdfg.sdfg.apply_transformations_repeated(LoopToMap) == 1
        sdfg.validate()

    body = next(n for n, _ in nsdfg.sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.NestedSDFG))
    assert "M" not in body.symbol_mapping
    assert "N" in body.symbol_mapping

    a, b = np.arange(8, dtype=np.float64), np.zeros(3)
    sdfg(a=a, b=b, N=8, M=3)
    assert np.array_equal(a, np.arange(8) + 1.0)
    assert np.array_equal(b, [2.0, 0.0, 0.0])
