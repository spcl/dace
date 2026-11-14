# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The walker-primary ``VectorizeCPUMultiDim`` pipeline on minimal kernels: it leaves an empty SDFG alone,
vectorizes 1-D and 2-D copies into programs that still copy, and refuses tile ranks it does not support."""

import numpy as np
import pytest

import dace
from dace.transformation.passes.vectorization.config import VectorizeConfig
from dace.transformation.passes.vectorization.enums import ISA, BranchMode, RemainderStrategy
from dace.transformation.passes.vectorization.vectorize_cpu_multi_dim import VectorizeCPUMultiDim


def copy_kernel(name: str, shape: tuple[int, ...]) -> dace.SDFG:
    """``B[idx] = A[idx]`` over one map spanning ``shape``."""
    sdfg = dace.SDFG(name)
    sdfg.add_array("A", shape, dace.float64)
    sdfg.add_array("B", shape, dace.float64)
    state = sdfg.add_state("s")
    params = [f"i{d}" for d in range(len(shape))]
    me, mx = state.add_map("k", {p: f"0:{n}" for p, n in zip(params, shape)})
    tasklet = state.add_tasklet("body", {"_a"}, {"_b"}, "_b = _a")
    index = ", ".join(params)
    state.add_memlet_path(state.add_read("A"), me, tasklet, dst_conn="_a", memlet=dace.Memlet(f"A[{index}]"))
    state.add_memlet_path(tasklet, mx, state.add_write("B"), src_conn="_b", memlet=dace.Memlet(f"B[{index}]"))
    return sdfg


def test_an_empty_sdfg_is_left_unchanged_and_reported_untiled():
    sdfg = dace.SDFG("vectorize_empty")
    sdfg.add_state("s")
    with pytest.warns(UserWarning, match="tiled nothing"):
        VectorizeCPUMultiDim(VectorizeConfig(widths=(8,), target_isa=ISA.SCALAR)).apply_pass(sdfg, {})
    assert [len(state.nodes()) for state in sdfg.states()] == [0]


@pytest.mark.parametrize("shape,widths", [((16,), (8,)), ((16, 32), (4, 8))], ids=["k1", "k2"])
def test_a_vectorized_copy_still_copies(shape, widths):
    sdfg = copy_kernel(f"vectorize_copy_k{len(widths)}", shape)
    VectorizeCPUMultiDim(VectorizeConfig(widths=widths, target_isa=ISA.SCALAR)).apply_pass(sdfg, {})
    a = np.random.default_rng(0).random(shape)
    b = np.zeros(shape)
    sdfg(A=a, B=b)
    assert np.array_equal(b, a)


@pytest.mark.parametrize("widths,rank", [((), 0), ((8, 8, 8, 8), 4)], ids=["k0", "k4"])
def test_a_tile_rank_outside_one_to_three_is_refused(widths, rank):
    with pytest.raises(NotImplementedError, match=f"K={rank} not in"):
        VectorizeCPUMultiDim(VectorizeConfig(widths=widths, target_isa=ISA.SCALAR))


@pytest.mark.parametrize(
    "branch_mode,remainder",
    [
        (BranchMode.MERGE, RemainderStrategy.SCALAR_POSTAMBLE),
        (BranchMode.FP_FACTOR, RemainderStrategy.SCALAR_POSTAMBLE),
    ],
    ids=["merge", "fp_factor"],
)
def test_every_branch_mode_runs_on_an_empty_sdfg(branch_mode, remainder):
    """``fp_factor`` needs K=1 and a scalar postamble; ``merge`` accepts any combination."""
    sdfg = dace.SDFG(f"vectorize_branch_{branch_mode.name.lower()}")
    sdfg.add_state("s")
    config = VectorizeConfig(widths=(8,), target_isa=ISA.SCALAR, branch_mode=branch_mode, remainder_strategy=remainder)
    with pytest.warns(UserWarning, match="tiled nothing"):
        VectorizeCPUMultiDim(config).apply_pass(sdfg, {})
    assert config.branch_mode is branch_mode


if __name__ == "__main__":
    test_an_empty_sdfg_is_left_unchanged_and_reported_untiled()
    test_a_vectorized_copy_still_copies((16,), (8,))
    test_a_vectorized_copy_still_copies((16, 32), (4, 8))
    test_a_tile_rank_outside_one_to_three_is_refused((), 0)
    test_a_tile_rank_outside_one_to_three_is_refused((8, 8, 8, 8), 4)
    test_every_branch_mode_runs_on_an_empty_sdfg(BranchMode.MERGE, RemainderStrategy.SCALAR_POSTAMBLE)
    test_every_branch_mode_runs_on_an_empty_sdfg(BranchMode.FP_FACTOR, RemainderStrategy.SCALAR_POSTAMBLE)
