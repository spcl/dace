# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``MapFusionVertical`` and a read or write that every iteration makes over the whole array.

A body that indexes through another array (``offs[k]``) leaves its memlet at the whole array. Equal read and write
subsets of that kind are not point wise: every iteration touches all of it, so fusing a first Map that reads ``A``
through a shifted index with a second Map that writes ``A[i]`` lets one iteration overwrite what another reads. Found
on ``mg_vcycle`` (NestForge#84), the smoother ``th = stencil(uh); uh = th`` of each level.
"""

import numpy as np
import pytest

import dace
from dace.sdfg import nodes
from dace.transformation.passes.fuse_maps import FuseMaps

N = dace.symbol("N")


def smoother(shifted: bool) -> dace.SDFG:
    """``tmp[i] = 0.5 * A[(i + 1) % N]`` (``A[i]`` when not ``shifted``), then ``A[i] = tmp[i]``. Shifted, both Maps
    move ``A`` through a memlet of the whole array, as ``mg_vcycle``'s body does through ``offs[k]``."""
    sdfg = dace.SDFG("shifted_smoother" if shifted else "pointwise_smoother")
    sdfg.add_array("A", [N], dace.float64)
    sdfg.add_transient("tmp", [N], dace.float64)
    state = sdfg.add_state()
    first_entry, first_exit = state.add_map("first", {"i": "0:N"})
    first = state.add_tasklet("first", {"a"}, {"o"}, "o = 0.5 * a[(i + 1) % N]" if shifted else "o = 0.5 * a")
    state.add_memlet_path(
        state.add_read("A"),
        first_entry,
        first,
        dst_conn="a",
        memlet=dace.Memlet("A[0:N]" if shifted else "A[i]"),
    )
    tmp = state.add_access("tmp")
    state.add_memlet_path(first, first_exit, tmp, src_conn="o", memlet=dace.Memlet("tmp[i]"))
    second_entry, second_exit = state.add_map("second", {"i": "0:N"})
    second = state.add_tasklet("second", {"a"}, {"o"}, "o[i] = a" if shifted else "o = a")
    state.add_memlet_path(tmp, second_entry, second, dst_conn="a", memlet=dace.Memlet("tmp[i]"))
    state.add_memlet_path(
        second,
        second_exit,
        state.add_write("A"),
        src_conn="o",
        memlet=dace.Memlet("A[0:N]" if shifted else "A[i]"),
    )
    sdfg.validate()
    return sdfg


def count_maps(sdfg: dace.SDFG) -> int:
    return sum(isinstance(found[0], nodes.MapEntry) for found in sdfg.all_nodes_recursive())


def fuse(sdfg: dace.SDFG, relaxed_dataflow: bool) -> None:
    FuseMaps(
        allow_intermediates_with_symbolic_size=True, partial_reduce_fusion=True, relaxed_dataflow=relaxed_dataflow
    ).apply_pass(sdfg, {})


@pytest.mark.parametrize("relaxed_dataflow", [True, False])
def test_whole_array_read_is_not_fused_with_a_write_back(relaxed_dataflow: bool):
    sdfg = smoother(shifted=True)
    fuse(sdfg, relaxed_dataflow)
    assert count_maps(sdfg) == 2


@pytest.mark.parametrize("relaxed_dataflow", [True, False])
def test_pointwise_read_is_still_fused_with_a_write_back(relaxed_dataflow: bool):
    sdfg = smoother(shifted=False)
    fuse(sdfg, relaxed_dataflow)
    assert count_maps(sdfg) == 1


def test_shifted_smoother_result_is_unchanged_by_fusion():
    sdfg = smoother(shifted=True)
    fuse(sdfg, relaxed_dataflow=True)
    size = 64
    rng = np.random.default_rng(0)
    data = rng.random(size)
    want = 0.5 * np.roll(data, -1)
    sdfg(A=data, N=size)
    assert np.allclose(data, want)
