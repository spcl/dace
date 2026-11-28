# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""DropImpliedGuards: ICON velocity_tendencies' ``IF (clip_count == 0) CYCLE`` over a count of flags."""

import numpy as np
import pytest

import dace
from dace.sdfg import nodes
from dace.transformation.passes.canonicalize import canonicalize
from dace.transformation.passes.canonicalize.pipeline import stage_labels
from dace.transformation.passes.drop_implied_guards import DropImpliedGuards, implied_guards

K, N = dace.symbol("K"), dace.symbol("N")
SIZES = {"K": 6, "N": 16}


@dace.program
def clip(z: dace.float64[K, N], lim: dace.float64, flag: dace.bool_[K, N], out: dace.float64[K, N]):
    for k in range(K):
        count = 0
        for i in range(N):
            flag[k, i] = abs(z[k, i]) > lim
            if flag[k, i]:
                count += 1
        if count == 0:
            continue
        for i in range(N):
            if flag[k, i]:
                out[k, i] = z[k, i] * 2.0


@dace.program
def clip_other_flag(
    z: dace.float64[K, N], lim: dace.float64, flag: dace.bool_[K, N], other: dace.bool_[K, N], out: dace.float64[K, N]
):
    for k in range(K):
        count = 0
        for i in range(N):
            flag[k, i] = abs(z[k, i]) > lim
            if flag[k, i]:
                count += 1
        if count == 0:
            continue
        for i in range(N):
            if other[k, i]:
                out[k, i] = z[k, i] * 2.0


@dace.program
def clip_count_down(z: dace.float64[K, N], lim: dace.float64, flag: dace.bool_[K, N], out: dace.float64[K, N]):
    for k in range(K):
        count = 0
        for i in range(N):
            flag[k, i] = abs(z[k, i]) > lim
            if flag[k, i]:
                count -= 1
        if count == 0:
            continue
        for i in range(N):
            if flag[k, i]:
                out[k, i] = z[k, i] * 2.0


def reference(z: np.ndarray, lim: float, flag: np.ndarray, out: np.ndarray) -> None:
    for k in range(z.shape[0]):
        flag[k] = np.abs(z[k]) > lim
        if flag[k].any():
            out[k][flag[k]] = z[k][flag[k]] * 2.0


def canonical(program, fused: bool) -> dace.SDFG:
    sdfg = program.to_sdfg()
    labels = stage_labels("cpu")
    canonicalize(sdfg, target="cpu", stages=labels if fused else labels[: labels.index("fuse")])
    return sdfg


def guard_reads_count(sdfg: dace.SDFG) -> bool:
    return any(
        "count" in str(condition.as_string)
        for block in sdfg.all_control_flow_blocks(recursive=True)
        if isinstance(block, dace.sdfg.state.ConditionalBlock)
        for condition, body in block.branches
        if condition is not None
    )


def clipping_inputs() -> tuple[np.ndarray, float]:
    rng = np.random.default_rng(7)
    z = rng.standard_normal((SIZES["K"], SIZES["N"]))
    z[1] = 0.0  # a level that clips nothing: the early exit's case
    z[4] *= 10.0  # a level that clips almost everything
    return z, 1.0


@pytest.mark.parametrize("fused", [True, False], ids=["merged_guard", "nested_guard"])
def test_count_guard_implied_by_the_flag_is_dropped_and_the_result_is_unchanged(fused: bool) -> None:
    sut = canonical(clip, fused)
    assert guard_reads_count(sut)
    assert len(implied_guards_everywhere(sut)) == 1

    DropImpliedGuards().apply_pass(sut, {})
    sut.validate()

    assert not guard_reads_count(sut)
    z, lim = clipping_inputs()
    flag, out = np.zeros(z.shape, bool), np.full(z.shape, -1.0)
    ref_flag, ref_out = flag.copy(), out.copy()
    sut(z=z, lim=lim, flag=flag, out=out, **SIZES)
    reference(z, lim, ref_flag, ref_out)
    assert np.array_equal(flag, ref_flag) and np.array_equal(out, ref_out)


def test_the_count_is_dead_once_its_guard_is_gone() -> None:
    sut = canonical(clip, True)
    DropImpliedGuards().apply_pass(sut, {})
    sut.simplify()
    assert not any(isinstance(n, nodes.AccessNode) and n.data == "count" for n, parent in sut.all_nodes_recursive()), (
        "nothing reads count, so it goes"
    )


def fused_map_count(drop: bool) -> int:
    sdfg = canonical(clip, False)
    if drop:
        DropImpliedGuards().apply_pass(sdfg, {})
    labels = stage_labels("cpu")
    canonicalize(sdfg, target="cpu", stages=labels[labels.index("fuse") :])
    return sum(isinstance(n, nodes.MapEntry) for n, parent in sdfg.all_nodes_recursive())


def test_without_the_guard_the_flag_map_and_the_clipping_map_fuse() -> None:
    assert fused_map_count(drop=False) == 3, "the count's guard keeps the clipping map apart"
    assert fused_map_count(drop=True) == 2, "the k map and one fused i map, as in the OpenACC version"


def test_a_guard_over_a_different_flag_is_kept() -> None:
    sut = canonical(clip_other_flag, True)
    assert implied_guards_everywhere(sut) == []


def test_a_count_that_can_go_negative_keeps_its_guard() -> None:
    sut = canonical(clip_count_down, True)
    assert implied_guards_everywhere(sut) == []


def implied_guards_everywhere(sdfg: dace.SDFG) -> list:
    return [g for nested in sdfg.all_sdfgs_recursive() for g in implied_guards(nested)]


if __name__ == "__main__":
    test_count_guard_implied_by_the_flag_is_dropped_and_the_result_is_unchanged(True)
    test_count_guard_implied_by_the_flag_is_dropped_and_the_result_is_unchanged(False)
    test_the_count_is_dead_once_its_guard_is_gone()
    test_without_the_guard_the_flag_map_and_the_clipping_map_fuse()
    test_a_guard_over_a_different_flag_is_kept()
    test_a_count_that_can_go_negative_keeps_its_guard()
