# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Fusion across maps of different dimensionality: the mirrored FuseMapsAtSharedAxes split, a partial reduction whose
producer writes other data, and the parameter rename of a vertical fusion next to an inner map."""

import numpy as np

import dace
from dace.sdfg import nodes
from dace.transformation.dataflow.map_fusion_vertical import MapFusionVertical
from dace.transformation.dataflow.map_partial_reduce_fusion import MapPartialReduceFusion
from dace.transformation.passes.canonicalize.fuse_maps_at_shared_axes import FuseMapsAtSharedAxes

N = 16


def top_level_maps(sdfg: dace.SDFG) -> list[nodes.MapEntry]:
    return [n for st in sdfg.states() for n in st.nodes() if isinstance(n, nodes.MapEntry) and st.entry_node(n) is None]


def capture_sdfg() -> dace.SDFG:
    """``T[i] = a[i]`` then ``for j: for i in 0:3: out[j, i] = 10 T[j] + i``: fusing renames ``j`` to ``i``."""
    sdfg = dace.SDFG("capture")
    sdfg.add_array("a", [N], dace.float64)
    sdfg.add_array("out", [N, 3], dace.float64)
    sdfg.add_transient("T", [N], dace.float64)
    st = sdfg.add_state()
    st.add_mapped_tasklet(
        "prod",
        {"i": f"0:{N}"},
        {"__in": dace.Memlet("a[i]")},
        "__out = __in",
        {"__out": dace.Memlet("T[i]")},
        external_edges=True,
    )
    t_node = next(n for n in st.data_nodes() if n.data == "T")
    me, mx = st.add_map("cons", {"j": f"0:{N}"})
    ie, ix = st.add_map("inner", {"i": "0:3"})
    tk = st.add_tasklet("t", {"__t"}, {"__o"}, "__o = __t * 10 + i")
    st.add_memlet_path(t_node, me, ie, tk, dst_conn="__t", memlet=dace.Memlet("T[j]"))
    st.add_memlet_path(tk, ix, mx, st.add_write("out"), src_conn="__o", memlet=dace.Memlet("out[j, i]"))
    return sdfg


def test_vertical_fusion_rename_is_not_captured_by_inner_map():
    sdfg = capture_sdfg()
    assert sdfg.apply_transformations(MapFusionVertical) == 1
    a = np.arange(N, dtype=np.float64)
    out = np.zeros((N, 3))
    sdfg(a=a, out=out)
    assert np.allclose(out, a[:, None] * 10 + np.arange(3)[None, :])


def lower_dim_producer_sdfg(consumer_params: tuple[str, str]) -> dace.SDFG:
    """``t[i] = 2 a[i]`` then a map over ``consumer_params`` computing ``out[i, k] = t[i] * b[i, k]``."""
    sdfg = dace.SDFG("lower_dim_producer")
    sdfg.add_array("a", [N], dace.float64)
    sdfg.add_array("b", [N, 3], dace.float64)
    sdfg.add_array("out", [N, 3], dace.float64)
    sdfg.add_transient("t", [N], dace.float64)
    st = sdfg.add_state()
    t_node = st.add_access("t")
    st.add_mapped_tasklet(
        "prod",
        {"i": f"0:{N}"},
        {"__a": dace.Memlet("a[i]")},
        "__t = 2 * __a",
        {"__t": dace.Memlet("t[i]")},
        external_edges=True,
        output_nodes={"t": t_node},
    )
    ranges = {"i": f"0:{N}", "k": "0:3"}
    st.add_mapped_tasklet(
        "cons",
        {p: ranges[p] for p in consumer_params},
        {"__t": dace.Memlet("t[i]"), "__b": dace.Memlet("b[i, k]")},
        "__o = __t * __b",
        {"__o": dace.Memlet("out[i, k]")},
        external_edges=True,
        input_nodes={"t": t_node},
    )
    return sdfg


def run_lower_dim(sdfg: dace.SDFG) -> None:
    a, b = np.random.rand(N), np.random.rand(N, 3)
    out = np.zeros((N, 3))
    sdfg(a=a, b=b, out=out)
    assert np.allclose(out, 2 * a[:, None] * b)


def test_lower_dim_producer_fuses_into_consumer_outer_axes():
    sdfg = lower_dim_producer_sdfg(("i", "k"))
    assert FuseMapsAtSharedAxes().apply_pass(sdfg, {}) == 1
    (outer,) = top_level_maps(sdfg)
    assert outer.map.params == ["i"]
    run_lower_dim(sdfg)


def test_lower_dim_producer_keeps_consumer_loop_order():
    # The producer's axis is the consumer's inner one: fusing would make it outermost.
    sdfg = lower_dim_producer_sdfg(("k", "i"))
    assert FuseMapsAtSharedAxes().apply_pass(sdfg, {}) is None
    assert len(top_level_maps(sdfg)) == 2
    run_lower_dim(sdfg)


def partial_reduce_with_extra_output_sdfg() -> dace.SDFG:
    @dace.program
    def prog(x: dace.float64[N, 3], y: dace.float64[N, 3], rsq2: dace.float64[N], dsum: dace.float64[N, 3]):
        d = x - y
        rsq = np.sum(d * d, axis=1)
        rsq2[:] = rsq * 2
        dsum[:] = d * 3

    return prog.to_sdfg(simplify=True)


def test_partial_reduce_fusion_with_other_producer_outputs():
    sdfg = partial_reduce_with_extra_output_sdfg()
    from dace.transformation.dataflow import MapFusionVertical as MFV

    sdfg.apply_transformations_repeated(MFV)
    assert sdfg.apply_transformations(MapPartialReduceFusion) == 1
    x, y = np.random.rand(N, 3), np.random.rand(N, 3)
    rsq2, dsum = np.zeros(N), np.zeros((N, 3))
    sdfg(x=x, y=y, rsq2=rsq2, dsum=dsum)
    assert np.allclose(rsq2, 2 * np.sum((x - y) ** 2, axis=1))
    assert np.allclose(dsum, 3 * (x - y))


if __name__ == "__main__":
    test_vertical_fusion_rename_is_not_captured_by_inner_map()
    test_lower_dim_producer_fuses_into_consumer_outer_axes()
    test_lower_dim_producer_keeps_consumer_loop_order()
    test_partial_reduce_fusion_with_other_producer_outputs()
