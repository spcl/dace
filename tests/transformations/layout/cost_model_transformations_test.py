# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The LogP cost analysis applied to the actual layout transformations.

These pin the end-to-end signal the optimizer relies on: apply a real layout transform to an SDFG,
run the analysis on the resulting compute map, and check the predicted cost moves the right way. A
transform that fixes a strided access must lower the block-message count; one that only pads must
leave a contiguous access essentially unchanged. Pure/symbolic -- no measurement."""
import dace
import sympy as sp
import pytest
from dace import nodes

from dace.transformation.layout.permute_dimensions import PermuteDimensions
from dace.transformation.layout.pad_dimensions import PadDimensions
from dace.transformation.layout.cost_model.logp_analysis import analyze_loop_nest
from dace.transformation.layout.cost_model.loggp import LogGP, gap_from_bandwidth

N = dace.symbol("N")
P = LogGP(L=95e-9, o=0.0, g=4e-9, G=gap_from_bandwidth(100e9), line_bytes=64, bw_saturated=100e9, bw_core=40e9)


def _compute_map(sdfg):
    """The map whose scope directly contains a tasklet -- the compute nest to analyze."""
    for state in sdfg.states():
        children = state.scope_children()
        for node in state.nodes():
            if isinstance(node, nodes.MapEntry) and any(isinstance(c, nodes.Tasklet) for c in children[node]):
                return state, node
    raise AssertionError("no compute map found")


def _messages(cost, array, n=4096):
    return float(sp.simplify(cost.arrays[array].messages_per_iter).subs(N, n))


@dace.program
def transposed_add(A: dace.float64[N, N], B: dace.float64[N, N], C: dace.float64[N, N]):
    for i, j in dace.map[0:N, 0:N]:
        C[i, j] = A[i, j] + B[j, i]  # B read transposed -> strided inner access


@dace.program
def elementwise(A: dace.float64[N, N], C: dace.float64[N, N]):
    for i, j in dace.map[0:N, 0:N]:
        C[i, j] = A[i, j] * 2.0


def test_permute_relayout_fixes_a_transposed_access():
    """B is read transposed (B[j,i]): strided inner, ~1 block per iteration. PermuteDimensions relays
    B into the transposed physical order so the compute reads it contiguously, and the analysis must
    see the message count collapse from ~1 to ~1/8."""
    sdfg = transposed_add.to_sdfg(simplify=True)
    st, me = _compute_map(sdfg)
    before = analyze_loop_nest(st, me, P, block_bytes=64)
    assert _messages(before, "B") == pytest.approx(1.0, abs=0.02)  # strided, expensive

    PermuteDimensions(permute_map={"B": [1, 0]}, add_permute_maps=True).apply_pass(sdfg, {})
    st2, me2 = _compute_map(sdfg)
    after = analyze_loop_nest(st2, me2, P, block_bytes=64)
    assert "permuted_B" in after.arrays
    assert _messages(after, "permuted_B") == pytest.approx(1.0 / 8.0, abs=0.02)  # now contiguous


def test_permute_lowers_the_predicted_time():
    """The message drop must show up in the predicted time of the compute nest."""
    sdfg = transposed_add.to_sdfg(simplify=True)
    st, me = _compute_map(sdfg)
    before = analyze_loop_nest(st, me, P, block_bytes=64)
    PermuteDimensions(permute_map={"B": [1, 0]}, add_permute_maps=True).apply_pass(sdfg, {})
    st2, me2 = _compute_map(sdfg)
    after = analyze_loop_nest(st2, me2, P, block_bytes=64)
    # both nests are parallel -> bandwidth-bound; total_time is proportional to total blocks
    assert float(sp.simplify(after.total_time()).subs(N, 4096)) < \
        float(sp.simplify(before.total_time()).subs(N, 4096))


def test_pad_keeps_a_contiguous_access_cheap():
    """PadDimensions grows a dimension (here A's inner stride N -> N+3) but leaves the innermost
    stride 1, so a contiguous access stays ~1/8 block per iteration. Pad's job is alignment, not
    reducing the message count of an already-contiguous access."""
    sdfg = elementwise.to_sdfg(simplify=True)
    PadDimensions(pad_map={"A": [0, 3]}).apply_pass(sdfg, {})
    st, me = _compute_map(sdfg)
    cost = analyze_loop_nest(st, me, P, block_bytes=64)
    assert sdfg.arrays["A"].strides[0] == N + 3  # the dimension grew
    assert _messages(cost, "A") == pytest.approx(1.0 / 8.0, abs=0.02)  # still contiguous, still cheap


def test_pad_of_a_bad_stride_does_not_pretend_to_help():
    """Padding the OUTER dimension of a contiguous array does not change the inner stride, so the
    message count is unchanged -- the model does not credit a pad that moved nothing relevant."""
    sdfg = elementwise.to_sdfg(simplify=True)
    st0, me0 = _compute_map(sdfg)
    baseline = _messages(analyze_loop_nest(st0, me0, P, block_bytes=64), "A")
    PadDimensions(pad_map={"A": [4, 0]}).apply_pass(sdfg, {})  # pad outer dim only
    st, me = _compute_map(sdfg)
    cost = analyze_loop_nest(st, me, P, block_bytes=64)
    assert _messages(cost, "A") == pytest.approx(baseline, abs=0.02)


def _blocked_sdfg(ii_stride, block_bytes=64):
    """A 4D tiled access C[I,J,ii,jj] whose physical layout is what a Block transform materializes;
    ii_stride < inner-block => a contiguous tile, else a scattered one."""
    T = dace.symbol("T")
    sdfg = dace.SDFG("blocked")
    sdfg.add_array("C", [T, T, 4, 4], dace.float64, strides=(T * 16, 16, ii_stride, 1))
    sdfg.add_array("D", [T, T, 4, 4], dace.float64)
    st = sdfg.add_state("s", is_start_block=True)
    me, mx = st.add_map("m", {"I": "0:T", "J": "0:T", "ii": "0:4", "jj": "0:4"})
    t = st.add_tasklet("t", {"a"}, {"d"}, "d = a")
    st.add_memlet_path(st.add_read("C"), me, t, dst_conn="a", memlet=dace.Memlet("C[I,J,ii,jj]"))
    st.add_memlet_path(t, mx, st.add_write("D"), src_conn="d", memlet=dace.Memlet("D[I,J,ii,jj]"))
    return st, me


def test_blocked_layout_is_scored_by_tile_contiguity():
    """A contiguous tile (the point of a Block transform) is cheaper than a scattered one. Note: this
    uses the blocked STRIDES directly, because SplitDimensions rewrites the access with split-index
    (int_floor/%) subscripts that blocks_touched cannot yet reduce -- a documented limitation."""
    st_c, me_c = _blocked_sdfg(4)  # contiguous tile
    st_s, me_s = _blocked_sdfg(4096)  # scattered tile
    contig = analyze_loop_nest(st_c, me_c, P, block_bytes=64)
    scattered = analyze_loop_nest(st_s, me_s, P, block_bytes=64)
    cm = float(sp.simplify(contig.arrays["C"].messages_per_iter).subs(dace.symbol("T"), 64))
    sm = float(sp.simplify(scattered.arrays["C"].messages_per_iter).subs(dace.symbol("T"), 64))
    assert cm < sm


def test_gpu_sector_makes_a_transposed_access_relatively_worse():
    """On a 32B GPU sector a transposed access wastes more of each transfer than on a 64B line, so
    the analysis penalises it more at sector granularity -- the device the model targets first."""
    sdfg = transposed_add.to_sdfg(simplify=True)
    st, me = _compute_map(sdfg)
    line = analyze_loop_nest(st, me, P, block_bytes=64)
    sector = analyze_loop_nest(st, me, P, block_bytes=32)
    # contiguous A: fewer elements per 32B sector -> more messages than on a 64B line
    assert _messages(sector, "A") > _messages(line, "A")
    # transposed B is a whole block per element either way (~1), so it stays saturated
    assert _messages(sector, "B") == pytest.approx(_messages(line, "B"), abs=0.02)


if __name__ == "__main__":
    test_permute_relayout_fixes_a_transposed_access()
    test_permute_lowers_the_predicted_time()
    test_pad_keeps_a_contiguous_access_cheap()
    test_pad_of_a_bad_stride_does_not_pretend_to_help()
    test_blocked_layout_is_scored_by_tile_contiguity()
    test_gpu_sector_makes_a_transposed_access_relatively_worse()
    print("cost_model transformation tests PASS")
