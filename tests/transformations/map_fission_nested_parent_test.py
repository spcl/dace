# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.
""" Reproducer: MapFission must not corrupt an SDFG when the candidate map is
    itself nested inside another map.

    For the subgraph expression (``expr_index == 0``) ``MapFission.apply``
    rewires the fissioned scope's boundary access nodes through the
    state-level memlet paths of the map entry/exit. That assumption only holds
    for a top-level map. When the map is nested inside another map the boundary
    access nodes are left edge-less (isolated), and a later ``scope_dict`` call
    raises ``RuntimeError: Leftover nodes in queue``. The fix makes
    ``can_be_applied`` refuse the unsupported nested-parent case so the SDFG is
    never invalidated; top-level fission is unaffected.

    Kernels use the dace Python frontend only.
"""
import numpy as np
import pytest

import dace
from dace.sdfg import nodes
from dace.transformation.dataflow.map_fission import MapFission

N, M, P = (dace.symbol(s) for s in ('N', 'M', 'P'))


@dace.program
def nested_parent(x: dace.float64[N, M], y: dace.float64[N, M]):
    for j in dace.map[0:M]:
        for jj in dace.map[0:P]:
            for i in dace.map[0:N]:
                x[i, j] = 1.0
            for i in dace.map[0:N]:
                y[i, j] = 2.0


@dace.program
def toplevel_two_components(x: dace.float64[N, M], y: dace.float64[N, M]):
    for j in dace.map[0:M]:
        for i in dace.map[0:N]:
            x[i, j] = 1.0
        for i in dace.map[0:N]:
            y[i, j] = 2.0


def _find_map_entry(sdfg: dace.SDFG, param: str, nested: bool):
    for st in sdfg.all_states():
        for n in st.nodes():
            if isinstance(n, nodes.MapEntry) and n.map.params == [param]:
                if (st.entry_node(n) is not None) == nested:
                    return st, n
    return None, None


def test_mapfission_refuses_nested_parent_and_keeps_sdfg_valid():
    sdfg = nested_parent.to_sdfg(simplify=True)
    state, jj = _find_map_entry(sdfg, 'jj', nested=True)
    assert jj is not None, "expected a nested 'jj' map"

    # Guarded: the nested-parent subgraph case is refused, not corrupted.
    assert MapFission.can_be_applied_to(sdfg, map_entry=jj) is False

    # Repeated application must neither raise nor invalidate the SDFG
    # (previously this produced isolated nodes -> scope_dict RuntimeError).
    sdfg.apply_transformations_repeated(MapFission)
    sdfg.validate()


def test_mapfission_still_applies_to_toplevel_map():
    n, m = 6, 5
    sdfg = toplevel_two_components.to_sdfg(simplify=True)
    _, j = _find_map_entry(sdfg, 'j', nested=False)
    assert j is not None and MapFission.can_be_applied_to(sdfg, map_entry=j) is True

    applied = sdfg.apply_transformations_repeated(MapFission)
    assert applied >= 1
    sdfg.validate()

    x, y = np.zeros((n, m)), np.zeros((n, m))
    sdfg(x=x, y=y, N=n, M=m)
    assert np.allclose(x, 1.0) and np.allclose(y, 2.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
