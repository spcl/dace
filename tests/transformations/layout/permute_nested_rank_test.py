# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""PermuteDimensions must not silently skip a nested SDFG it cannot permute.

How much of a permuted array a nested SDFG receives decides what the body needs:

* FULL RANK -- the outer permutation applies verbatim. This is what ``prepare_for_layout`` leaves
  behind, because ``ExpandNestedSDFGInputs`` widens narrowed nested in/out subsets to the full
  outer array and mirrors the outer shape onto the inner descriptor.
* ONE DIMENSION -- a scalar or a single surviving axis. A permutation reorders axes relative to
  each other, so with one axis there is nothing to reorder.
* IN BETWEEN -- a partial slice (``A[0:a, j, 0:c] -> nA[a, c]``), whose induced permutation depends
  on which axes survive the squeeze. The pass does not compute it and must REFUSE.

The refusal is the point. The code used to fall through this case silently, leaving the nested body
reading the OLD layout while the outer array was relaid. Nothing downstream catches that when the
surviving extents are the same symbol (``A[N, N, N]``): the shapes still line up, so it is a silent
transpose, the same class as a permuted copy that squares away to an identity. These tests pin the
loud failure and pin that the real pipeline never reaches it.
"""
import numpy
import pytest
import dace

from dace.transformation.layout.permute_dimensions import PermuteDimensions
from dace.transformation.layout.prepare import prepare_for_layout

N = dace.symbol("N")


def _plane_body(inner_shape):
    """A nested SDFG that doubles a plane, with ``a``/``out`` at ``inner_shape``."""
    inner = dace.SDFG(f"plane_{len(inner_shape)}d")
    inner.add_array("a", list(inner_shape), dace.float64)
    inner.add_array("out", list(inner_shape), dace.float64)
    st = inner.add_state("s", is_start_block=True)
    params = {f"i{d}": f"0:{s}" for d, s in enumerate(inner_shape)}
    sub = ", ".join(params)
    me, mx = st.add_map("m", params)
    t = st.add_tasklet("t", {"x"}, {"y"}, "y = x * 2.0")
    st.add_memlet_path(st.add_read("a"), me, t, dst_conn="x", memlet=dace.Memlet(f"a[{sub}]"))
    st.add_memlet_path(t, mx, st.add_write("out"), src_conn="y", memlet=dace.Memlet(f"out[{sub}]"))
    return inner


def _cube_calling(inner_shape, in_subset, out_subset):
    """A[N,N,N] -> nested SDFG (inner rank = len(inner_shape)) -> B[N,N,N]."""
    sdfg = dace.SDFG("cube")
    sdfg.add_array("A", [N, N, N], dace.float64)
    sdfg.add_array("B", [N, N, N], dace.float64)
    st = sdfg.add_state("s", is_start_block=True)
    nsdfg = st.add_nested_sdfg(_plane_body(inner_shape), {"a"}, {"out"}, {"N": N})
    st.add_edge(st.add_read("A"), None, nsdfg, "a", dace.Memlet(f"A[{in_subset}]"))
    st.add_edge(nsdfg, "out", st.add_write("B"), None, dace.Memlet(f"B[{out_subset}]"))
    return sdfg


def test_partial_slice_into_nested_sdfg_is_refused():
    """rank 3 outer -> rank 2 inner: the induced permutation is not computed, so refuse."""
    sdfg = _cube_calling((N, N), f"0, 0:{N}, 0:{N}", f"0, 0:{N}, 0:{N}")
    with pytest.raises(NotImplementedError, match="partial slice"):
        PermuteDimensions(permute_map={"A": [2, 1, 0]}, add_permute_maps=True).apply_pass(sdfg, {})


def test_single_surviving_axis_needs_no_inner_permutation():
    """rank 3 outer -> rank 1 inner: one axis cannot be reordered, so this is NOT refused."""
    sdfg = _cube_calling((N, ), f"0, 0, 0:{N}", f"0, 0, 0:{N}")
    PermuteDimensions(permute_map={"A": [2, 1, 0]}, add_permute_maps=True).apply_pass(sdfg, {})
    nested = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.NestedSDFG)]
    assert len(nested) == 1
    assert tuple(nested[0].sdfg.arrays["a"].shape) == (N, )  # inner descriptor untouched


def _mixed_rank_cube():
    """A[N,N,N] arrives at FULL rank; B leaves through a rank-collapsing partial slice."""
    inner = dace.SDFG("plane_mixed")
    inner.add_array("a", [N, N, N], dace.float64)
    inner.add_array("out", [N, N], dace.float64)
    st = inner.add_state("s", is_start_block=True)
    me, mx = st.add_map("m", {"i0": f"0:{N}", "i1": f"0:{N}"})
    t = st.add_tasklet("t", {"x"}, {"y"}, "y = x * 2.0")
    st.add_memlet_path(st.add_read("a"), me, t, dst_conn="x", memlet=dace.Memlet("a[0, i0, i1]"))
    st.add_memlet_path(t, mx, st.add_write("out"), src_conn="y", memlet=dace.Memlet("out[i0, i1]"))

    sdfg = dace.SDFG("cube_mixed")
    sdfg.add_array("A", [N, N, N], dace.float64)
    sdfg.add_array("B", [N, N, N], dace.float64)
    state = sdfg.add_state("s", is_start_block=True)
    nsdfg = state.add_nested_sdfg(inner, {"a"}, {"out"}, {"N": N})
    state.add_edge(state.add_read("A"), None, nsdfg, "a", dace.Memlet(f"A[0:{N}, 0:{N}, 0:{N}]"))
    state.add_edge(nsdfg, "out", state.add_write("B"), None, dace.Memlet(f"B[0, 0:{N}, 0:{N}]"))
    return sdfg


def test_refusal_keys_on_the_permuted_array_only():
    """A partial slice of an array NOBODY permutes is not the pass's problem: B leaves through a
    rank-collapsing slice, but only A is permuted (and A arrives at full rank), so no refusal."""
    sdfg = _mixed_rank_cube()
    PermuteDimensions(permute_map={"A": [2, 1, 0]}, add_permute_maps=True).apply_pass(sdfg, {})


@pytest.mark.parametrize("permutation", [[2, 1, 0], [1, 0, 2], [0, 2, 1]])
def test_prepare_for_layout_widens_nested_inputs_so_the_pass_never_refuses(permutation):
    """The in-contract path: prepare_for_layout widens the nested input to full rank, so the
    partial-slice refusal is unreachable and the permuted program stays bit-exact vs numpy."""
    sdfg = _cube_calling((N, N), f"0, 0:{N}, 0:{N}", f"0, 0:{N}, 0:{N}")
    prepare_for_layout(sdfg, validate=False)

    for state in sdfg.all_states():
        for node in state.nodes():
            if isinstance(node, dace.nodes.NestedSDFG):
                for e in state.in_edges(node):
                    if e.dst_conn is not None and e.data.data is not None:
                        assert len(node.sdfg.arrays[e.dst_conn].shape) == len(sdfg.arrays[e.data.data].shape)

    PermuteDimensions(permute_map={"A": permutation}, add_permute_maps=True).apply_pass(sdfg, {})

    n = 4
    A = numpy.random.default_rng(0).random((n, n, n))
    expect = numpy.zeros((n, n, n))
    expect[0, :, :] = A[0, :, :] * 2.0
    B = numpy.zeros((n, n, n))
    sdfg(A=A.copy(), B=B, N=n)
    assert numpy.array_equal(B, expect)


if __name__ == "__main__":
    test_partial_slice_into_nested_sdfg_is_refused()
    test_single_surviving_axis_needs_no_inner_permutation()
    test_unpermuted_array_through_a_partial_slice_is_not_refused()
    for p in ([2, 1, 0], [1, 0, 2], [0, 2, 1]):
        test_prepare_for_layout_widens_nested_inputs_so_the_pass_never_refuses(p)
    print("permute nested-rank tests PASS")
