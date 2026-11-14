# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``r[i, j] = sqrt(sum_k (p[i, k] - q[j, k])^2)``: the partial sum fuses into its consumer and leaves no array."""

import numpy as np

import dace
from dace.sdfg import nodes
from dace.transformation.dataflow.map_partial_reduce_fusion import MapPartialReduceFusion
from dace.transformation.passes import FuseMaps

N = dace.symbol("N")
M = dace.symbol("M")


def distances(name: str, block: bool, consumer: bool = True, upstream: bool = False) -> dace.SDFG:
    """Per (block of 4) row ``i``: ``sq = (p - q)^2`` over (i, j, k), ``s = sq.sum(k)``, ``r = sqrt(s)``."""
    rows = "4" if block else "N"
    sdfg = dace.SDFG(name)
    sdfg.add_array("p", (8 if block else N, 3), dace.float64)
    sdfg.add_array("q", (M, 3), dace.float64)
    sdfg.add_array("r", (8 if block else N, M), dace.float64)
    sdfg.add_transient("sq", (rows, M, 3), dace.float64)
    sdfg.add_transient("s", (rows, M), dace.float64)
    state = sdfg.add_state()
    off = "4*b + " if block else ""
    outer = state.add_map("blocks", dict(b="0:2")) if block else None
    sq, s = state.add_access("sq"), state.add_access("s")
    p, q, r = state.add_read("p"), state.add_read("q"), state.add_write("r")

    def path(*chain, memlet, **kw):
        state.add_memlet_path(*chain, memlet=memlet, **kw)

    me, mx = state.add_map("square", dict(i=f"0:{rows}", j="0:M", k="0:3"))
    if upstream:
        sdfg.add_transient("d", (rows, M, 3), dace.float64, lifetime=sdfg.arrays["sq"].lifetime)
        de, dx = state.add_map("diff", dict(i=f"0:{rows}", j="0:M", k="0:3"))
        td = state.add_tasklet("diff", {"a", "c"}, {"o"}, "o = a - c")
        path(p, *([outer[0]] if block else []), de, td, dst_conn="a", memlet=dace.Memlet(f"p[{off}i, k]"))
        path(q, *([outer[0]] if block else []), de, td, dst_conn="c", memlet=dace.Memlet("q[j, k]"))
        d = state.add_access("d")
        path(td, dx, d, src_conn="o", memlet=dace.Memlet("d[i, j, k]"))
        t = state.add_tasklet("sq", {"a"}, {"o"}, "o = a * a")
        path(d, me, t, dst_conn="a", memlet=dace.Memlet("d[i, j, k]"))
    else:
        t = state.add_tasklet("sq", {"a", "c"}, {"o"}, "o = (a - c) * (a - c)")
        path(p, *([outer[0]] if block else []), me, t, dst_conn="a", memlet=dace.Memlet(f"p[{off}i, k]"))
        path(q, *([outer[0]] if block else []), me, t, dst_conn="c", memlet=dace.Memlet("q[j, k]"))
    path(t, mx, sq, src_conn="o", memlet=dace.Memlet("sq[i, j, k]"))
    red = state.add_reduce("lambda x, y: x + y", axes=[2], identity=0)
    state.add_edge(sq, None, red, None, dace.Memlet(f"sq[0:{rows}, 0:M, 0:3]"))
    state.add_edge(red, None, s, None, dace.Memlet(f"s[0:{rows}, 0:M]"))
    if consumer:
        ce, cx = state.add_map("root", dict(i=f"0:{rows}", j="0:M"))
        t2 = state.add_tasklet("root", {"a"}, {"o"}, "o = sqrt(a)")
        path(s, ce, t2, dst_conn="a", memlet=dace.Memlet("s[i, j]"))
        path(t2, cx, *([outer[1]] if block else []), r, src_conn="o", memlet=dace.Memlet(f"r[{off}i, j]"))
    else:
        path(s, *([outer[1]] if block else []), r, memlet=dace.Memlet(f"r[{off.replace('i', '')}0:{rows}, 0:M]"))
    if block:
        for name in ("sq", "s", "d") if upstream else ("sq", "s"):
            sdfg.arrays[name].lifetime = dace.AllocationLifetime.Scope
    sdfg.validate()
    return sdfg


def check(sdfg: dace.SDFG, rows: int, take_root: bool = True):
    rng = np.random.default_rng(20261018)
    p, q = rng.random((rows, 3)), rng.random((5, 3))
    r = np.zeros((rows, 5))
    args = dict(p=p, q=q, r=r, M=5) | ({} if rows == 8 and "block" in sdfg.name else dict(N=rows))
    sdfg(**args)
    ref = ((p[:, None, :] - q[None, :, :]) ** 2).sum(axis=2)
    np.testing.assert_allclose(r, np.sqrt(ref) if take_root else ref)


def no_arrays_left(sdfg: dace.SDFG) -> bool:
    return not {"sq", "s", "d"} & {
        n.data for n, parent in sdfg.all_nodes_recursive() if isinstance(n, nodes.AccessNode)
    }


def test_partial_sum_fuses_into_its_consumer():
    sdfg = distances("partial_sum_toplevel", block=False)
    assert sdfg.apply_transformations(MapPartialReduceFusion) == 1
    sdfg.validate()
    FuseMaps().apply_pass(sdfg, {})
    sdfg.validate()
    assert no_arrays_left(sdfg)
    check(sdfg, 6)


def test_partial_sum_in_a_block_fuses_into_its_consumer():
    sdfg = distances("partial_sum_block", block=True)
    assert sdfg.apply_transformations(MapPartialReduceFusion) == 1
    FuseMaps().apply_pass(sdfg, {})
    sdfg.validate()
    assert no_arrays_left(sdfg)
    check(sdfg, 8)


def test_a_partial_sum_without_an_elementwise_consumer_is_left_to_map_reduce_fusion():
    sdfg = distances("partial_sum_no_consumer", block=False, consumer=False)
    assert sdfg.apply_transformations(MapPartialReduceFusion) == 0


def test_fuse_maps_runs_it_before_map_reduce_fusion():
    sdfg = distances("partial_sum_fuse_maps", block=True)
    FuseMaps(partial_reduce_fusion=True).apply_pass(sdfg, {})
    sdfg.validate()
    assert no_arrays_left(sdfg)
    check(sdfg, 8)


def test_fuse_maps_fuses_the_operand_producer_before_splitting():
    """``d = p - q`` is its own Map: it fuses into the squaring Map first, or the split leaves ``d`` a full array."""
    sdfg = distances("partial_sum_upstream", block=True, upstream=True)
    FuseMaps(partial_reduce_fusion=True).apply_pass(sdfg, {})
    sdfg.validate()
    assert no_arrays_left(sdfg)
    check(sdfg, 8)


if __name__ == "__main__":
    test_partial_sum_fuses_into_its_consumer()
    test_partial_sum_in_a_block_fuses_into_its_consumer()
    test_a_partial_sum_without_an_elementwise_consumer_is_left_to_map_reduce_fusion()
    test_fuse_maps_runs_it_before_map_reduce_fusion()
    test_fuse_maps_fuses_the_operand_producer_before_splitting()
