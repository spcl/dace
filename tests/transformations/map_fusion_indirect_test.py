# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
""" Map fusion across indirect (gather/scatter) accesses.

    The canonicalization pipeline fissions an indirect map into independent
    indirect maps (ConditionalComponentFission -> MapFission, carrying the
    idx[i] indirection symbols into each split) and must then be able to
    recombine them: horizontal fusion of independent gathers/scatters and
    vertical fusion of an indirect producer into its consumer. Every test
    checks the fused SDFG is a single map and numerically identical to a
    deep-copied pre-fusion run.
"""
import copy

import numpy as np
import pytest

import dace
from dace.sdfg import nodes
from dace.transformation.passes.conditional_component_fission import ConditionalComponentFission
from dace.transformation.dataflow.map_fission import MapFission
from dace.transformation.dataflow.map_fusion_vertical import MapFusionVertical
from dace.transformation.dataflow.map_fusion_horizontal import MapFusionHorizontal
from dace.transformation.passes.pattern_matching import PatternMatchAndApplyRepeated
from dace.transformation.interstate.state_fusion_with_happens_before import StateFusionExtended
from dace.transformation.interstate.sdfg_nesting import InlineSDFG

N = dace.symbol('N')


def _nmaps(sdfg):
    return len([n for st in sdfg.all_states() for n in st.nodes() if isinstance(n, nodes.MapEntry)])


def _structural_clean(sdfg):
    PatternMatchAndApplyRepeated([StateFusionExtended()]).apply_pass(sdfg, {})
    PatternMatchAndApplyRepeated([InlineSDFG()]).apply_pass(sdfg, {})


@dace.program
def two_gathers(a: dace.float64[N], idx: dace.int32[N], b: dace.float64[N], c: dace.float64[N], e: dace.float64[N]):
    for i in dace.map[0:N]:
        b[i] = a[idx[i]]
        e[i] = c[idx[i]]


@dace.program
def two_gather_stencils(a: dace.float64[N], idx: dace.int32[N], b: dace.float64[N], c: dace.float64[N],
                        e: dace.float64[N]):
    # Indirect *stencil*: gathered base with structured neighbour offsets.
    for i in dace.map[1:N - 1]:
        b[i] = a[idx[i] - 1] + a[idx[i]] + a[idx[i] + 1]
        e[i] = c[idx[i] - 1] + c[idx[i]] + c[idx[i] + 1]


@dace.program
def two_scatters(a: dace.float64[N], idx: dace.int32[N], b: dace.float64[N], c: dace.float64[N], e: dace.float64[N]):
    for i in dace.map[0:N]:
        b[idx[i]] = a[i] * 2.0
        e[idx[i]] = c[i] + 1.0


@dace.program
def indirect_producer_consumer(a: dace.float64[N], idx: dace.int32[N], b: dace.float64[N]):
    t = np.empty_like(a)
    for i in dace.map[0:N]:
        t[i] = a[idx[i]] + 1.0
    for i in dace.map[0:N]:
        b[i] = t[i] * 2.0


def _fission_then_fuse(sdfg):
    """Pipeline fuse-stage recipe: replicate the indirect NestedSDFG,
    MapFission, structural-clean, then vertical+horizontal fusion."""
    ConditionalComponentFission().apply_pass(sdfg, {})
    sdfg.apply_transformations_repeated(MapFission)
    _structural_clean(sdfg)
    return sdfg.apply_transformations_repeated([MapFusionVertical, MapFusionHorizontal])


@pytest.mark.parametrize('prog,kind,interior', [
    (two_gathers, 'gather', False),
    (two_gather_stencils, 'gather_stencil', True),
    (two_scatters, 'scatter', False),
])
def test_horizontal_fusion_recombines_indirect_maps(prog, kind, interior):
    """Two independent indirect maps fission then horizontally re-fuse into
    one map; value-preserving."""
    n = 24
    np.random.seed(5)
    a = np.random.rand(n)
    c = np.random.rand(n)
    idx = (np.random.randint(1, n - 1, size=n) if interior else
           np.random.permutation(n) if kind == 'scatter' else np.random.randint(0, n, size=n)).astype(np.int32)

    base = prog.to_sdfg(simplify=True)
    assert _nmaps(base) == 1
    ref_b, ref_e = np.zeros(n), np.zeros(n)
    copy.deepcopy(base)(a=a.copy(), idx=idx.copy(), b=ref_b, c=c.copy(), e=ref_e, N=n)

    sdfg = prog.to_sdfg(simplify=True)
    assert _nmaps(sdfg) == 1
    # Fission so there are two independent indirect maps to recombine.
    ConditionalComponentFission().apply_pass(sdfg, {})
    sdfg.apply_transformations_repeated(MapFission)
    assert _nmaps(sdfg) == 2, "indirect map must fission into two"
    _structural_clean(sdfg)
    fused = sdfg.apply_transformations_repeated([MapFusionVertical, MapFusionHorizontal])
    assert fused, "horizontal fusion must recombine the indirect maps"
    assert _nmaps(sdfg) == 1, "fused back to a single map"
    sdfg.validate()

    out_b, out_e = np.zeros(n), np.zeros(n)
    sdfg(a=a.copy(), idx=idx.copy(), b=out_b, c=c.copy(), e=out_e, N=n)
    assert np.allclose(out_b, ref_b) and np.allclose(out_e, ref_e), f"mismatch ({kind})"


def test_vertical_fusion_indirect_producer_consumer():
    """An indirect producer (`t=a[idx]+1`) vertically fuses into its
    consumer (`b=t*2`); single map, value-preserving."""
    n = 20
    np.random.seed(6)
    a = np.random.rand(n)
    idx = np.random.randint(0, n, size=n).astype(np.int32)

    base = indirect_producer_consumer.to_sdfg(simplify=True)
    assert _nmaps(base) == 2
    ref = np.zeros(n)
    copy.deepcopy(base)(a=a.copy(), idx=idx.copy(), b=ref, N=n)

    sdfg = indirect_producer_consumer.to_sdfg(simplify=True)
    _structural_clean(sdfg)
    fused = sdfg.apply_transformations_repeated([MapFusionVertical, MapFusionHorizontal])
    assert fused, "vertical fusion must fuse the indirect producer into the consumer"
    assert _nmaps(sdfg) == 1, "fused into a single map"
    sdfg.validate()

    out = np.zeros(n)
    sdfg(a=a.copy(), idx=idx.copy(), b=out, N=n)
    assert np.allclose(out, ref)
    assert np.allclose(out, (a[idx] + 1.0) * 2.0)


def test_indirect_round_trip_fission_then_fuse_is_identity():
    """fission -> fuse on an indirect map returns to one map and the exact
    pre-pass values (the pipeline round-trip the fuse stage relies on)."""
    n = 28
    np.random.seed(9)
    a = np.random.rand(n)
    c = np.random.rand(n)
    idx = np.random.randint(0, n, size=n).astype(np.int32)

    base = two_gathers.to_sdfg(simplify=True)
    ref_b, ref_e = np.zeros(n), np.zeros(n)
    copy.deepcopy(base)(a=a.copy(), idx=idx.copy(), b=ref_b, c=c.copy(), e=ref_e, N=n)

    sdfg = two_gathers.to_sdfg(simplify=True)
    _fission_then_fuse(sdfg)
    assert _nmaps(sdfg) == 1
    sdfg.validate()
    out_b, out_e = np.zeros(n), np.zeros(n)
    sdfg(a=a.copy(), idx=idx.copy(), b=out_b, c=c.copy(), e=out_e, N=n)
    assert np.allclose(out_b, ref_b) and np.allclose(out_e, ref_e)
    assert np.allclose(out_b, a[idx]) and np.allclose(out_e, c[idx])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
