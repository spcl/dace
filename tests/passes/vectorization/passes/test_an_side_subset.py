# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Unit tests for :func:`an_side_subset` (design section 3.7).

Six cases:

* ``edge.data.data == an.data`` (the simple case).
* ``edge.data.data`` points at the OTHER AN; ``other_subset`` carries the
  one we want.
* Implicit full-shape copy: ``other_subset is None``; subset reconstructed
  from descriptor.
* Multi-dim Array descriptor full-shape reconstruction.
* Refuses an edge whose endpoints don't include the AN.
* Refuses an edge with no memlet.
"""
import pytest

import dace
from dace import subsets
from dace.memlet import Memlet
from dace.transformation.passes.vectorization.utils.subsets import an_side_subset


def _build_an_to_an(shape_a=(8, ), shape_b=(8, )):
    sdfg = dace.SDFG("an_side_subset_fixture")
    sdfg.add_array("A", shape_a, dace.float64, transient=False)
    sdfg.add_array("B", shape_b, dace.float64, transient=False)
    state = sdfg.add_state("s")
    a = state.add_access("A")
    b = state.add_access("B")
    return sdfg, state, a, b


def test_data_matches_an_returns_subset():
    """edge.data.data == an.data -> return edge.data.subset."""
    sdfg, state, a, b = _build_an_to_an()
    edge = state.add_edge(a, None, b, None, Memlet("A[2:5]"))
    out = an_side_subset(edge, a, sdfg)
    assert out == subsets.Range([(2, 4, 1)])


def test_data_points_at_other_returns_other_subset():
    """edge.data.data == B.data, other_subset is A's subset -> return other_subset."""
    sdfg, state, a, b = _build_an_to_an()
    mem = Memlet(data="B", subset=subsets.Range([(0, 7, 1)]))
    mem.other_subset = subsets.Range([(2, 5, 1)])
    edge = state.add_edge(a, None, b, None, mem)
    out = an_side_subset(edge, a, sdfg)
    assert out == subsets.Range([(2, 5, 1)])


def test_no_other_subset_reconstructs_from_descriptor():
    """Implicit full-shape copy: other_subset is None -> full descriptor range."""
    sdfg, state, a, b = _build_an_to_an(shape_a=(4, ), shape_b=(4, ))
    mem = Memlet(data="B", subset=subsets.Range([(0, 3, 1)]))
    edge = state.add_edge(a, None, b, None, mem)
    out = an_side_subset(edge, a, sdfg)
    assert out == subsets.Range([(0, 3, 1)])


def test_multi_dim_full_shape_reconstruction():
    """Multi-dim descriptor reconstruction reads every shape dim."""
    sdfg, state, a, b = _build_an_to_an(shape_a=(4, 8), shape_b=(4, 8))
    mem = Memlet(data="B", subset=subsets.Range([(0, 3, 1), (0, 7, 1)]))
    edge = state.add_edge(a, None, b, None, mem)
    out = an_side_subset(edge, a, sdfg)
    assert out == subsets.Range([(0, 3, 1), (0, 7, 1)])


def test_refuses_edge_not_incident_on_an():
    """Edge that doesn't touch the AN -> ValueError."""
    sdfg, state, a, b = _build_an_to_an()
    sdfg.add_array("C", (8, ), dace.float64, transient=False)
    sdfg.add_array("D", (8, ), dace.float64, transient=False)
    c = state.add_access("C")
    d = state.add_access("D")
    edge = state.add_edge(c, None, d, None, Memlet("C[0:8]"))
    with pytest.raises(ValueError, match="not an endpoint"):
        an_side_subset(edge, a, sdfg)


def test_empty_memlet_falls_back_to_descriptor():
    """Empty Memlet on an AN-incident edge: full-shape fallback fires."""
    sdfg, state, a, b = _build_an_to_an(shape_a=(6, ), shape_b=(6, ))
    edge = state.add_edge(a, None, b, None, Memlet())
    out = an_side_subset(edge, a, sdfg)
    assert out == subsets.Range([(0, 5, 1)])


def test_infer_edge_endpoints_an_to_an():
    """``infer_edge_endpoints`` returns both src and dst data names + subsets
    for an AN -> AN edge regardless of which side ``memlet.data`` points at.
    """
    from dace.transformation.passes.vectorization.utils.subsets import infer_edge_endpoints
    sdfg, state, a, b = _build_an_to_an()
    mem = Memlet(data="A", subset=subsets.Range([(2, 5, 1)]))
    mem.other_subset = subsets.Range([(0, 3, 1)])
    edge = state.add_edge(a, None, b, None, mem)
    src_data, src_subset, dst_data, dst_subset = infer_edge_endpoints(edge, sdfg)
    assert src_data == "A"
    assert src_subset == subsets.Range([(2, 5, 1)])
    assert dst_data == "B"
    assert dst_subset == subsets.Range([(0, 3, 1)])


def test_infer_edge_endpoints_non_an_endpoint_returns_none():
    """When one endpoint is NOT an AccessNode (e.g. a Tasklet), the helper
    reports ``None`` for that side's data name + subset.
    """
    from dace.transformation.passes.vectorization.utils.subsets import infer_edge_endpoints
    sdfg, state, a, _b = _build_an_to_an()
    t = state.add_tasklet("t", inputs={"_in"}, outputs=set(), code="pass")
    edge = state.add_edge(a, None, t, "_in", Memlet("A[0:8]"))
    src_data, src_subset, dst_data, dst_subset = infer_edge_endpoints(edge, sdfg)
    assert src_data == "A"
    assert src_subset == subsets.Range([(0, 7, 1)])
    assert dst_data is None
    assert dst_subset is None
