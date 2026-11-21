# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``canonicalize(wavefront=False)``: a nest that carries a dependence along both axes stays a sequential nest, for a
caller that decides the skew itself; the default still skews it. ``WavefrontSkew.plan_or_reason`` says why a loop is
no wavefront."""

import numpy as np

import dace
from dace.sdfg import nodes
from dace.sdfg.state import LoopRegion
from dace.transformation.passes.canonicalize.pipeline import canonicalize
from dace.transformation.passes.parallelization.wavefront.skew import Plan, WavefrontSkew
from dace.transformation.passes.parallelization.wavefront.statements import occurrence_index

N = dace.symbol("N")


@dace.program
def wavefront_2d(aa: dace.float64[N, N]):
    """TSVC s2111: each element reads its left and its upper neighbor."""
    for i in range(1, N):
        for j in range(1, N):
            aa[i, j] = (aa[i, j - 1] + aa[i - 1, j]) / 1.9


@dace.program
def outer_carried(aa: dace.float64[N, N], bb: dace.float64[N, N]):
    """TSVC s1119: only the outer axis carries, so the inner one is parallel already."""
    for i in range(1, N):
        for j in range(N):
            aa[i, j] = aa[i - 1, j] + bb[i, j]


def loops_and_maps(sdfg: dace.SDFG) -> tuple[int, int]:
    loops = [r for r in sdfg.all_control_flow_regions(recursive=True) if isinstance(r, LoopRegion)]
    maps = [node for node, parent in sdfg.all_nodes_recursive() if isinstance(node, nodes.MapEntry)]
    return len(loops), len(maps)


def canonical(wavefront: bool) -> dace.SDFG:
    sdfg = wavefront_2d.to_sdfg(simplify=True)
    canonicalize(sdfg, wavefront=wavefront)
    sdfg.validate()
    return sdfg


def reference(aa: np.ndarray) -> np.ndarray:
    out = aa.copy()
    for i in range(1, out.shape[0]):
        for j in range(1, out.shape[1]):
            out[i, j] = (out[i, j - 1] + out[i - 1, j]) / 1.9
    return out


def test_without_wavefronts_the_nest_stays_two_sequential_loops():
    assert loops_and_maps(canonical(wavefront=False)) == (2, 0)


def test_by_default_the_nest_becomes_a_wavefront_with_a_doall_per_diagonal():
    loops, maps = loops_and_maps(canonical(wavefront=True))

    assert maps == 1, "the default skews the nest: one doall runs each diagonal (of tiles, on a CPU)"
    assert loops >= 1


def test_both_forms_compute_the_same_values():
    n = 24
    aa = np.random.default_rng(2111).standard_normal((n, n))
    for wavefront in (False, True):
        got = aa.copy()
        canonical(wavefront)(aa=got, N=n)
        assert np.allclose(got, reference(aa)), f"wavefront={wavefront}"


def test_a_loop_whose_inner_axis_is_parallel_is_no_wavefront_and_the_reason_says_so():
    sdfg = outer_carried.to_sdfg(simplify=True)
    outer = next(r for r in sdfg.all_control_flow_regions(recursive=True) if isinstance(r, LoopRegion))

    reason = WavefrontSkew().plan_or_reason(outer, sdfg, occurrence_index(sdfg))

    assert reason == "an axis is already parallel at a fixed outer iteration: LoopToMap reaches it"


def test_plan_and_plan_or_reason_agree_on_a_wavefront():
    sdfg = wavefront_2d.to_sdfg(simplify=True)
    outer = next(r for r in sdfg.all_control_flow_regions(recursive=True) if isinstance(r, LoopRegion))
    sut = WavefrontSkew()

    assert isinstance(sut.plan_or_reason(outer, sdfg, occurrence_index(sdfg)), Plan)
    assert isinstance(sut.plan(outer, sdfg, occurrence_index(sdfg)), Plan)


if __name__ == "__main__":
    test_without_wavefronts_the_nest_stays_two_sequential_loops()
    test_by_default_the_nest_becomes_a_wavefront_with_a_doall_per_diagonal()
    test_both_forms_compute_the_same_values()
    test_a_loop_whose_inner_axis_is_parallel_is_no_wavefront_and_the_reason_says_so()
    test_plan_and_plan_or_reason_agree_on_a_wavefront()
