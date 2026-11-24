# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A scratch array sized ``Max(0, n)`` that each iteration fills over ``0:n`` before reading it is private to the
iteration: the two extents differ only when the array is empty (ICON velocity_tendencies: the clip-count mask of the
CFL clipping, sized by the column window, kept the level loop sequential)."""

import numpy as np

import dace
from dace import symbolic
from dace.libraries.standard.nodes.reduce import Reduce
from dace.sdfg.state import LoopRegion
from dace.transformation.passes.parallelization.parallelize_loops import ParallelizeLoops

K = dace.symbol("K")
N = dace.symbol("N")


def row_counts_sdfg() -> dace.SDFG:
    sdfg = dace.SDFG("clamped_scratch_row_counts")
    sdfg.add_array("a", [K, N], dace.float64)
    sdfg.add_array("count", [K], dace.int64)
    sdfg.add_array("mask", [symbolic.pystr_to_symbolic("Max(0, N)")], dace.int64, transient=True)
    loop = LoopRegion("rows", "k < K", "k", "k = 0", "k = k + 1")
    sdfg.add_node(loop, is_start_block=True)
    body = loop.add_state("body", is_start_block=True)
    mask = body.add_access("mask")
    body.add_mapped_tasklet(
        "positive",
        {"c": "0:N"},
        {"x": dace.Memlet("a[k, c]")},
        "m = 1 if x > 0 else 0",
        {"m": dace.Memlet("mask[c]")},
        external_edges=True,
        output_nodes={"mask": mask},
    )
    reduce = Reduce("count_positive", "lambda x, y: x + y", identity=0)
    body.add_node(reduce)
    # The reduction reads the whole mask, clamped extent and all
    body.add_edge(mask, None, reduce, None, dace.Memlet.from_array("mask", sdfg.arrays["mask"]))
    body.add_edge(reduce, None, body.add_write("count"), None, dace.Memlet("count[k]"))
    return sdfg


def test_a_scratch_array_filled_over_its_clamped_extent_is_private_per_iteration():
    sdfg = row_counts_sdfg()
    loop = next(r for r in sdfg.all_control_flow_regions() if isinstance(r, LoopRegion))
    assert ParallelizeLoops().accepts(loop)
    ParallelizeLoops().apply_pass(sdfg, {})
    assert not any(isinstance(r, LoopRegion) for r in sdfg.all_control_flow_regions(recursive=True))
    a = np.random.default_rng(5).normal(size=(6, 11))
    count = np.zeros(6, dtype=np.int64)
    sdfg(a=a, count=count, K=6, N=11)
    assert np.array_equal(count, (a > 0).sum(axis=1))


if __name__ == "__main__":
    test_a_scratch_array_filled_over_its_clamped_extent_is_private_per_iteration()
