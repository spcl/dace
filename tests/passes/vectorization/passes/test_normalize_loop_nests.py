# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Unit tests for ``normalize_loop_nests`` (the tile-pipeline loop-nest
normalization: inline loop-nesting wrapper NSDFGs, then MapCollapse).

LoopToMap wraps each parallelised loop body in an NSDFG, so nested ``for``
loops become ``j-map -> NSDFG -> i-map`` (non-adjacent maps MapCollapse cannot
fuse). ``normalize_loop_nests`` inlines the wrappers + any single-state leaf
body so the maps fuse to one multi-param map — while LEAVING an inout-connector
or multi-state leaf body (the cloudsc ``zqlhs`` RMW chain) intact for the tile
descent. These tests pin both behaviours, including value preservation on the
cloudsc inout pattern.
"""

import pytest
# [UNSKIPPED-FOR-ASSESSMENT 2026-06-14] pytestmark = pytest.mark.skip(reason="legacy K=1/K=2 descent path frozen during walker-primary migration -- this test goes through VectorizeCPUMultiDim or the harness; both depend on the legacy descent + emit infrastructure being removed. Will be revived (or replaced by walker-primary equivalents) after the new orchestrator pipeline lands end-to-end.")
import numpy
import pytest

import dace
from dace.sdfg.nodes import MapEntry, NestedSDFG
from dace.transformation.interstate import LoopToMap
from dace.transformation.passes.vectorization.vectorize_cpu_multi_dim import normalize_loop_nests

from tests.passes.vectorization.helpers.harness import _get_cloudsc_snippet_four

M = dace.symbol("M")
Nn = dace.symbol("Nn")


@dace.program
def _nested2d(a: dace.float64[M, Nn], b: dace.float64[M, Nn]):
    for j in range(M):
        for i in range(Nn):
            b[j, i] = a[j, i] + 1.0


def _maps(sdfg: dace.SDFG):
    """All map-entry param lists in ``sdfg`` (recursively)."""
    return [n.map.params for n, _ in sdfg.all_nodes_recursive() if isinstance(n, MapEntry)]


def _nsdfgs(sdfg: dace.SDFG):
    """All NestedSDFG nodes in ``sdfg`` (recursively)."""
    return [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, NestedSDFG)]


def _inout_nsdfgs(sdfg: dace.SDFG):
    """NestedSDFGs that carry an inout connector (array read AND written)."""
    return [n for n in _nsdfgs(sdfg) if set(n.in_connectors) & set(n.out_connectors)]


def test_nested_for_loops_fuse_to_multiparam_map():
    """The j-map / i-map separated by LoopToMap's wrapper NSDFG fuse into one
    ``(j, i)`` 2-param map after normalization."""
    s = _nested2d.to_sdfg(simplify=True)
    s.apply_transformations_repeated(LoopToMap, permissive=True, validate=False)
    normalize_loop_nests(s)
    s.validate()
    maps = _maps(s)
    assert any(len(p) == 2 for p in maps), f"expected a fused 2-param (j, i) map, got {maps}"


def test_simple_single_state_body_inlined_away():
    """A simple single-state leaf compute body is inlined (no NSDFG survives),
    so it tiles via the flat EmitTileOps path rather than the descent."""
    s = _nested2d.to_sdfg(simplify=True)
    s.apply_transformations_repeated(LoopToMap, permissive=True, validate=False)
    normalize_loop_nests(s)
    assert not _nsdfgs(s), f"simple single-state body should inline away; left: {[n.label for n in _nsdfgs(s)]}"


def test_cloudsc_inout_body_preserved():
    """The cloudsc ``zqlhs`` RMW body NSDFG has an inout connector; InlineSDFG
    refuses it, so normalization LEAVES it intact for the tile descent (the
    'inout connector necessity'), keeping the SDFG valid."""
    s = _get_cloudsc_snippet_four()
    assert _inout_nsdfgs(s), "fixture precondition: an inout-connector body NSDFG (zqlhs)"
    s.apply_transformations_repeated(LoopToMap, permissive=True, validate=False)
    normalize_loop_nests(s)
    s.validate()
    inout_after = _inout_nsdfgs(s)
    assert inout_after, "the inout (zqlhs RMW) body must be preserved, not flattened"
    assert any("zqlhs" in (set(n.in_connectors) & set(n.out_connectors)) for n in inout_after), \
        "zqlhs must remain an inout connector on the preserved body"


def test_cloudsc_inout_normalization_is_value_preserving():
    """Normalizing the cloudsc inout chain leaves it numerically unchanged (the
    inout body is not inlined, so the result equals the un-normalized kernel)."""
    klon = 16
    shapes = {"zfallsink": (klon, klon, 5), "zqlhs": (klon, klon, 5), "zsolqb": (klon, klon, 5)}
    arrays = {n: numpy.random.default_rng(3).random(sh).astype(numpy.float64, order="F") for n, sh in shapes.items()}
    params = {
        "kfdia": numpy.int64(klon // 2),
        "kidia": numpy.int64(1),
        "klev": numpy.int64(klon),
        "klon": numpy.int64(klon),
        "_for_it_92": numpy.int64(0),
        "_for_it_91": numpy.int64(0),
    }
    ref = _get_cloudsc_snippet_four()
    ref.name = "norm_cloudsc_ref"
    vec = _get_cloudsc_snippet_four()
    vec.name = "norm_cloudsc_vec"
    normalize_loop_nests(vec)
    vec.validate()
    ra = {k: v.copy() for k, v in arrays.items()}
    ref.compile()(**ra, **params)
    va = {k: v.copy() for k, v in arrays.items()}
    vec.compile()(**va, **params)
    for k in arrays:
        numpy.testing.assert_allclose(va[k],
                                      ra[k],
                                      rtol=1e-12,
                                      atol=1e-12,
                                      err_msg=f"normalize changed cloudsc array {k!r}")


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
