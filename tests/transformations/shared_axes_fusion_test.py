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


def test_a_refused_split_leaves_the_boundary_memlets_as_they_were():
    """The consumer reads another iteration's row, so the split is merged back; the memlets its propagation made for
    the map's boundary must not stay."""
    size = dace.symbol("M", dace.int64)
    sdfg = dace.SDFG("refused_split")
    sdfg.add_symbol("M", dace.int64)
    sdfg.add_array("a", [size, 3], dace.float64)
    sdfg.add_array("out", [size], dace.float64)
    sdfg.add_transient("t", [size, 3], dace.float64)
    st = sdfg.add_state()
    t_node = st.add_access("t")
    st.add_mapped_tasklet(
        "prod", {"i": "0:M", "k": "0:3"}, {"__a": dace.Memlet("a[i, k]")}, "__t = 2 * __a", {"__t": dace.Memlet("t[i, k]")},
        external_edges=True, output_nodes={"t": t_node},
    )  # fmt: skip
    st.add_mapped_tasklet(
        "cons", {"i": "0:M"}, {"__t": dace.Memlet("t[M - 1 - i, 0]")}, "__o = __t", {"__o": dace.Memlet("out[i]")},
        external_edges=True, input_nodes={"t": t_node},
    )  # fmt: skip

    def boundary() -> list[dace.Memlet]:
        (entry,) = [n for n in st.nodes() if isinstance(n, nodes.MapEntry) and n.map.label.startswith("prod")]
        return [e.data for e in st.in_edges(entry) + st.out_edges(st.exit_node(entry))]

    before = boundary()
    assert FuseMapsAtSharedAxes().apply_pass(sdfg, {}) is None
    assert all(now is then for now, then in zip(boundary(), before, strict=True))
    a = np.random.rand(5, 3)
    out = np.zeros(5)
    sdfg(a=a, out=out, M=5)
    assert np.allclose(out, 2 * a[::-1, 0])


def test_lower_dim_producer_fuses_past_an_output_that_only_orders_the_consumer():
    """``t[i] = 2 a[i]; out[i, 0] = a[i]``, its ``out`` node only ordering ``for i, k in 1:3: out[i, k] = t[i] b[i, k]``
    (bout_elm_pb's ``ddt_Psi`` z=0 plane before the interior planes): the ordering is FuseMaps' pure output, so the
    producer fuses."""
    sdfg = dace.SDFG("orders_the_consumer")
    sdfg.add_array("a", [N], dace.float64)
    sdfg.add_array("b", [N, 3], dace.float64)
    sdfg.add_array("out", [N, 3], dace.float64)
    sdfg.add_transient("t", [N], dace.float64)
    st = sdfg.add_state()
    t_node, plane = st.add_access("t"), st.add_access("out")
    st.add_mapped_tasklet(
        "prod", {"i": f"0:{N}"}, {"__a": dace.Memlet("a[i]")}, "__t = 2 * __a\n__p = __a",
        {"__t": dace.Memlet("t[i]"), "__p": dace.Memlet("out[i, 0]")}, external_edges=True,
        output_nodes={"t": t_node, "out": plane},
    )  # fmt: skip
    _, cons_entry, _ = st.add_mapped_tasklet(
        "cons", {"i": f"0:{N}", "k": "1:3"}, {"__t": dace.Memlet("t[i]"), "__b": dace.Memlet("b[i, k]")},
        "__o = __t * __b", {"__o": dace.Memlet("out[i, k]")}, external_edges=True, input_nodes={"t": t_node},
    )  # fmt: skip
    st.add_nedge(plane, cons_entry, dace.Memlet())
    assert FuseMapsAtSharedAxes().apply_pass(sdfg, {}) == 1
    assert len(top_level_maps(sdfg)) == 1
    sdfg.validate()
    a, b = np.random.rand(N), np.random.rand(N, 3)
    out = np.zeros((N, 3))
    sdfg(a=a, b=b, out=out)
    assert np.allclose(out[:, 0], a) and np.allclose(out[:, 1:], 2 * a[:, None] * b[:, 1:])


if __name__ == "__main__":
    test_vertical_fusion_rename_is_not_captured_by_inner_map()
    test_lower_dim_producer_fuses_into_consumer_outer_axes()
    test_lower_dim_producer_keeps_consumer_loop_order()
    test_partial_reduce_fusion_with_other_producer_outputs()
    test_a_refused_split_leaves_the_boundary_memlets_as_they_were()
    test_lower_dim_producer_fuses_past_an_output_that_only_orders_the_consumer()
