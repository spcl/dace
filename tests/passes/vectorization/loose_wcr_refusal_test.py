# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A loose WCR in one map's body refuses that map, not the kernel: the rest of the kernel still tiles."""

import pytest

import dace
from dace.transformation.passes.vectorization.utils.errors import VectorizeUnsupported
from dace.transformation.passes.vectorization.utils.map_predicates import mark_maps_no_vectorize
from dace.transformation.passes.vectorization.vectorize_multi_dim import _AssertNoBodyWCR

N = 64


def two_maps_one_loose_wcr() -> dace.SDFG:
    """``loose``: its body accumulates into a transient through a WCR edge that stays inside the map (no lifted
    reduction boundary). ``clean``: an elementwise map beside it."""
    sdfg = dace.SDFG("loose_wcr_two_maps")
    for name in ("a", "b", "c"):
        sdfg.add_array(name, [N], dace.float64)
    sdfg.add_scalar("acc", dace.float64, transient=True)
    state = sdfg.add_state()

    entry, exit_ = state.add_map("loose", dict(i=f"0:{N}"))
    add = state.add_tasklet("add", {"x"}, {"y"}, "y = x")
    acc = state.add_access("acc")
    copy = state.add_tasklet("copy", {"v"}, {"w"}, "w = v")
    state.add_memlet_path(state.add_read("a"), entry, add, dst_conn="x", memlet=dace.Memlet("a[i]"))
    state.add_edge(add, "y", acc, None, dace.Memlet("acc[0]", wcr="lambda p, q: p + q"))
    state.add_edge(acc, None, copy, "v", dace.Memlet("acc[0]"))
    state.add_memlet_path(copy, exit_, state.add_write("b"), src_conn="w", memlet=dace.Memlet("b[i]"))

    state.add_mapped_tasklet(
        "clean",
        dict(j=f"0:{N}"),
        {"x": dace.Memlet("a[j]")},
        "y = 2 * x",
        {"y": dace.Memlet("c[j]")},
        external_edges=True,
    )
    sdfg.validate()
    return sdfg


def test_a_loose_wcr_refuses_only_the_map_whose_body_holds_it():
    sdfg = two_maps_one_loose_wcr()
    with pytest.raises(VectorizeUnsupported) as refused:
        _AssertNoBodyWCR().apply_pass(sdfg, {})
    assert refused.value.maps == ("loose",)


def test_a_map_marked_scalar_no_longer_refuses():
    """The retry after the refusal marks the map scalar; its body is then outside the region to be tiled."""
    sdfg = two_maps_one_loose_wcr()
    assert mark_maps_no_vectorize(sdfg, ("loose",)) == {"loose"}
    assert _AssertNoBodyWCR().apply_pass(sdfg, {}) is None


if __name__ == "__main__":
    test_a_loose_wcr_refuses_only_the_map_whose_body_holds_it()
    test_a_map_marked_scalar_no_longer_refuses()
