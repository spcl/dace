# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Average new blocks touched per iteration -- the layout-sensitive term the cost model pays for.

The invariant these pin: the metric must SEE a layout change. A contiguous-inner layout must cost
less than a strided one (Permute), and a contiguous tile must cost less than a scattered one even
when the reuse is on a non-innermost loop (Block/AoSoA). The second is the one the old
innermost-only fraction could not express."""
import dace
import sympy as sp

from dace.transformation.layout.cost_model.blocks_touched import average_blocks_touched
from dace.transformation.layout.cost_model.access_subsets import get_access_subsets

N = dace.symbol("N")
T = dace.symbol("T")


def _blocks_2d(strides, block_size=8):
    sdfg = dace.SDFG("bt2")
    sdfg.add_array("A", [N, N], dace.float64, strides=strides)
    sdfg.add_array("B", [N, N], dace.float64)
    st = sdfg.add_state("s", is_start_block=True)
    me, mx = st.add_map("m", {"i": "0:N", "j": "0:N"})
    t = st.add_tasklet("t", {"a"}, {"b"}, "b = a")
    st.add_memlet_path(st.add_read("A"), me, t, dst_conn="a", memlet=dace.Memlet("A[i,j]"))
    st.add_memlet_path(t, mx, st.add_write("B"), src_conn="b", memlet=dace.Memlet("B[i,j]"))
    lr = [{p: r for p, r in zip(me.map.params, me.map.range)}]
    return sp.simplify(average_blocks_touched(st, lr, get_access_subsets(st, me), block_size)["A"])


def _blocks_tiled(ii_stride, block_size=8):
    """4D tiled C[I,J,ii,jj] over loops (I,J,ii,jj). ii is a NON-innermost loop; a contiguous tile
    gives it a sub-block stride, so its reuse is only visible if the fraction is per-dimension."""
    sdfg = dace.SDFG("bt4")
    sdfg.add_array("C", [T, T, 4, 4], dace.float64, strides=(T * 16, 16, ii_stride, 1))
    sdfg.add_array("D", [T, T, 4, 4], dace.float64)
    st = sdfg.add_state("s", is_start_block=True)
    me, mx = st.add_map("m", {"I": "0:T", "J": "0:T", "ii": "0:4", "jj": "0:4"})
    t = st.add_tasklet("t", {"a"}, {"b"}, "b = a")
    st.add_memlet_path(st.add_read("C"), me, t, dst_conn="a", memlet=dace.Memlet("C[I,J,ii,jj]"))
    st.add_memlet_path(t, mx, st.add_write("D"), src_conn="b", memlet=dace.Memlet("D[I,J,ii,jj]"))
    lr = [{p: r for p, r in zip(me.map.params, me.map.range)}]
    return float(sp.simplify(average_blocks_touched(st, lr, get_access_subsets(st, me), block_size)["C"]).subs(T, 64))


def test_contiguous_inner_reaches_one_over_block_size():
    """Contiguous innermost (stride 1): 8 elements share a 64B line, so ~1/8 new block per iter."""
    value = float(_blocks_2d((N, 1)).subs(N, 4096))
    assert abs(value - 1.0 / 8.0) < 0.02


def test_permute_is_creditable_in_2d():
    """A transpose (contiguous outer instead of inner) costs strictly more -- the model sees Permute."""
    row = float(_blocks_2d((N, 1)).subs(N, 4096))  # contiguous inner
    col = float(_blocks_2d((1, N)).subs(N, 4096))  # contiguous outer
    assert col > 6 * row  # ~1.0 vs ~1/8


def test_block_size_scales_the_reuse():
    """A smaller block (GPU 32B sector = 4 fp64 elements) shares fewer elements per line, so the
    contiguous-inner cost is higher than with a 64B line (8 elements)."""
    cpu_line = float(_blocks_2d((N, 1), block_size=8).subs(N, 4096))  # 64B / 8B
    gpu_sector = float(_blocks_2d((N, 1), block_size=4).subs(N, 4096))  # 32B / 8B
    assert abs(gpu_sector - 1.0 / 4.0) < 0.02
    assert gpu_sector > cpu_line


def test_outer_tile_reuse_is_credited():
    """The fix: a contiguous tile puts a sub-block stride on the ii loop, which is NOT innermost.
    Crediting reuse only on the innermost loop (the old behaviour) would make these two equal; the
    contiguous tile must come out cheaper."""
    contiguous_tile = _blocks_tiled(4)  # ii stride 4 < block 8 -> reuse on the ii loop
    scattered_tile = _blocks_tiled(4096)  # ii stride huge -> no reuse
    assert contiguous_tile < scattered_tile


def test_extent_uses_integer_floor_not_c_division():
    """A non-unit loop step must give the right iteration count: floor((end-begin)/step)+1 via
    int_floor, not a truncating C division. Range 0:N:2 is int_floor(N,2)+1 iterations."""
    sdfg = dace.SDFG("bt_step")
    sdfg.add_array("A", [N], dace.float64)
    sdfg.add_array("B", [N], dace.float64)
    st = sdfg.add_state("s", is_start_block=True)
    me, mx = st.add_map("m", {"i": "0:N:2"})
    t = st.add_tasklet("t", {"a"}, {"b"}, "b = a")
    st.add_memlet_path(st.add_read("A"), me, t, dst_conn="a", memlet=dace.Memlet("A[i]"))
    st.add_memlet_path(t, mx, st.add_write("B"), src_conn="b", memlet=dace.Memlet("B[i]"))
    lr = [{p: r for p, r in zip(me.map.params, me.map.range)}]
    value = average_blocks_touched(st, lr, get_access_subsets(st, me), 8)["A"]
    # stride-2 access of a unit-stride array: 2 elements per step, so ~1/4 of a line per iteration.
    assert abs(float(sp.simplify(value).subs(N, 4096)) - 1.0 / 4.0) < 0.02


if __name__ == "__main__":
    test_contiguous_inner_reaches_one_over_block_size()
    test_permute_is_creditable_in_2d()
    test_block_size_scales_the_reuse()
    test_outer_tile_reuse_is_credited()
    test_extent_uses_integer_floor_not_c_division()
    print("blocks_touched tests PASS")
