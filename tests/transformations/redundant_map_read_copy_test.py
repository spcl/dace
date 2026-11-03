# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A read-only staging copy made inside a map scope is folded into its readers: icon_scatter's per-(jk, n)
``x = nbr_idx[n, :, :].copy()`` read through ``out[x[i, j], ...]``."""

import numpy as np

import dace
from dace.transformation.dataflow.redundant_array import RedundantMapReadCopy

K = dace.symbol("K")
M = dace.symbol("M")


def staged_gather(write_source: bool = False) -> dace.SDFG:
    """``for n: t = src[n, :]; for i: out[n, i] = t[idx[i]]`` with ``t`` copied inside the outer map's scope.
    ``write_source`` also writes ``src`` from the inner map, which makes the copy a real snapshot."""
    sdfg = dace.SDFG(f"staged_gather_{write_source}")
    sdfg.add_array("src", [K, M], dace.float64)
    sdfg.add_array("idx", [M], dace.int64)
    sdfg.add_array("out", [K, M], dace.float64)
    sdfg.add_transient("t", [M], dace.float64)
    state = sdfg.add_state()
    outer_entry, outer_exit = state.add_map("rows", {"n": "0:K"})
    inner_entry, inner_exit = state.add_map("cols", {"i": "0:M"})
    t = state.add_access("t")
    state.add_memlet_path(state.add_read("src"), outer_entry, t, memlet=dace.Memlet("src[n, 0:M]"))
    tasklet = state.add_tasklet("gather", {"row", "k"}, {"o"}, "o = row[k]")
    state.add_memlet_path(t, inner_entry, tasklet, dst_conn="row", memlet=dace.Memlet("t[0:M]", dynamic=True))
    state.add_memlet_path(
        state.add_read("idx"), outer_entry, inner_entry, tasklet, dst_conn="k", memlet=dace.Memlet("idx[i]")
    )
    state.add_memlet_path(
        tasklet, inner_exit, outer_exit, state.add_write("out"), src_conn="o", memlet=dace.Memlet("out[n, i]")
    )
    if write_source:
        clobber = state.add_tasklet("clobber", {}, {"s"}, "s = 0.0")
        state.add_nedge(inner_entry, clobber, dace.Memlet())
        state.add_memlet_path(
            clobber, inner_exit, outer_exit, state.add_write("src"), src_conn="s", memlet=dace.Memlet("src[n, i]")
        )
    sdfg.validate()
    return sdfg


def staged_index_through_nested() -> dace.SDFG:
    """``for n: t = cols[n, :]; out[n] = vals[t[0]] + t[1]`` with ``t`` read by a nested SDFG, ``t[0]`` in an
    interstate assignment (icon_scatter's ``out_slice = __sc0_x0[i, j]``) and ``t[1]`` through a memlet."""
    sdfg = dace.SDFG("staged_index_through_nested")
    sdfg.add_array("cols", [K, M], dace.int64)
    sdfg.add_array("vals", [M], dace.float64)
    sdfg.add_array("out", [K], dace.float64)
    sdfg.add_transient("t", [M], dace.int64)
    body = dace.SDFG("body")
    body.add_array("t", [M], dace.int64)
    body.add_array("vals", [M], dace.float64)
    body.add_array("out", [K], dace.float64)
    body.add_symbol("n", dace.int64)
    body.add_symbol("j", dace.int64)
    first = body.add_state("first", is_start_block=True)
    second = body.add_state("second")
    body.add_edge(first, second, dace.InterstateEdge(assignments={"j": "t[0]"}))
    tasklet = second.add_tasklet("pick", {"v", "w"}, {"o"}, "o = v + w")
    second.add_edge(second.add_read("vals"), None, tasklet, "v", dace.Memlet("vals[j]"))
    second.add_edge(second.add_read("t"), None, tasklet, "w", dace.Memlet("t[1]"))
    second.add_edge(tasklet, "o", second.add_write("out"), None, dace.Memlet("out[n]"))
    state = sdfg.add_state()
    entry, exit_ = state.add_map("rows", {"n": "0:K"})
    t = state.add_access("t")
    state.add_memlet_path(state.add_read("cols"), entry, t, memlet=dace.Memlet("cols[n, 0:M]"))
    nested = state.add_nested_sdfg(body, {"t", "vals"}, {"out"}, {"n": "n", "M": "M", "K": "K"})
    state.add_edge(t, None, nested, "t", dace.Memlet("t[0:M]"))
    state.add_memlet_path(state.add_read("vals"), entry, nested, dst_conn="vals", memlet=dace.Memlet("vals[0:M]"))
    state.add_memlet_path(nested, exit_, state.add_write("out"), src_conn="out", memlet=dace.Memlet("out[n]"))
    sdfg.validate()
    return sdfg


def test_a_staging_copy_read_by_a_nested_sdfg_is_folded():
    sdfg = staged_index_through_nested()
    assert sdfg.apply_transformations_repeated(RedundantMapReadCopy) == 1
    assert "t" not in sdfg.arrays
    sdfg.validate()
    rng = np.random.default_rng(1)
    cols, vals, out = rng.integers(0, 5, (3, 5)), rng.random(5), np.zeros(3)
    sdfg(cols=cols, vals=vals, out=out, K=3, M=5)
    assert np.allclose(out, vals[cols[:, 0]] + cols[:, 1])


def test_a_read_only_staging_copy_in_a_map_is_folded():
    sdfg = staged_gather()
    assert sdfg.apply_transformations_repeated(RedundantMapReadCopy) == 1
    assert "t" not in sdfg.arrays
    sdfg.validate()
    rng = np.random.default_rng(0)
    src, idx, out = rng.random((3, 7)), rng.integers(0, 7, 7), np.zeros((3, 7))
    sdfg(src=src, idx=idx, out=out, K=3, M=7)
    assert np.allclose(out, src[:, idx])


def test_a_copy_of_a_container_the_scope_writes_is_kept():
    sdfg = staged_gather(write_source=True)
    assert sdfg.apply_transformations_repeated(RedundantMapReadCopy) == 0
    assert "t" in sdfg.arrays


if __name__ == "__main__":
    test_a_read_only_staging_copy_in_a_map_is_folded()
    test_a_copy_of_a_container_the_scope_writes_is_kept()
    test_a_staging_copy_read_by_a_nested_sdfg_is_folded()
