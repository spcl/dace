# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for the WCR -> OpenMP ``reduction(...)`` codegen extension.

The CPU codegen emits ``#pragma omp parallel for reduction(op:var)`` for WCR
write-edges that target a true ``Scalar`` descriptor outside a parallel map,
*and* skips the per-edge ``reduce_atomic`` emission for the same target --
the OMP runtime privatizes the variable per thread and tree-reduces at the
end, so an extra atomic add is strictly wasted work.
"""
import os
import shutil

import numpy as np
import pytest

import dace
from dace.sdfg import nodes

N = dace.symbol("N")


def _build_wcr_scalar_sum() -> dace.SDFG:
    """Hand-built SDFG: scalar 'acc' accumulated via WCR from a parallel map.

    Done by hand (rather than from a @dace.program) so the accumulator is a
    true Scalar descriptor -- the shape the OMP reduction clause requires.
    """
    sdfg = dace.SDFG("wcr_scalar_sum")
    sdfg.add_array("src", [N], dace.float64)
    sdfg.add_array("out", [1], dace.float64)
    sdfg.add_scalar("acc", dace.float64, transient=True)

    # init: acc = 0
    init = sdfg.add_state("init", is_start_block=True)
    t0 = init.add_tasklet("seed", {}, {"o"}, "o = 0.0")
    init.add_edge(t0, "o", init.add_write("acc"), None, dace.Memlet("acc[0]"))

    # map: parallel for i: WCR-+= src[i] -> acc
    map_state = sdfg.add_state("map_state")
    sdfg.add_edge(init, map_state, dace.InterstateEdge())
    me, mx = map_state.add_map("m", dict(i="0:N"), schedule=dace.ScheduleType.CPU_Multicore)
    t = map_state.add_tasklet("acc", {"v"}, {"r"}, "r = v")
    src_an = map_state.add_read("src")
    acc_an = map_state.add_write("acc")
    map_state.add_memlet_path(src_an, me, t, dst_conn="v", memlet=dace.Memlet("src[i]"))
    map_state.add_memlet_path(t, mx, acc_an, src_conn="r", memlet=dace.Memlet("acc[0]", wcr="lambda a, b: a + b"))

    # writeback: out[0] = acc
    post = sdfg.add_state("post")
    sdfg.add_edge(map_state, post, dace.InterstateEdge())
    t1 = post.add_tasklet("wb", {"i"}, {"o"}, "o = i")
    post.add_edge(post.add_read("acc"), None, t1, "i", dace.Memlet("acc[0]"))
    post.add_edge(t1, "o", post.add_write("out"), None, dace.Memlet("out[0]"))

    sdfg.validate()
    return sdfg


def _compile_and_read_src(sdfg: dace.SDFG):
    """Compile the SDFG and return ``(csdfg, generated_cpp_source)``."""
    if os.path.exists(sdfg.build_folder):
        shutil.rmtree(sdfg.build_folder)
    csdfg = sdfg.compile()
    src_path = os.path.join(sdfg.build_folder, "src", "cpu", sdfg.name + ".cpp")
    assert os.path.exists(src_path), src_path
    return csdfg, open(src_path).read()


def test_scalar_wcr_emits_omp_reduction_clause():
    """The OMP pragma should include ``reduction(+:acc)`` for the scalar accumulator."""
    sdfg = _build_wcr_scalar_sum()
    csdfg, src = _compile_and_read_src(sdfg)

    # The OMP pragma must include the reduction clause for the scalar accumulator.
    pragma_lines = [l for l in src.splitlines() if "#pragma omp parallel for" in l]
    assert any("reduction(+:" in l for l in pragma_lines), \
        "expected reduction(+:...) clause in the OMP pragma -- got:\n" + "\n".join(pragma_lines)


def test_scalar_wcr_does_not_emit_atomic_for_covered_target():
    """The body must NOT call ``reduce_atomic`` for the WCR target covered by
    the OMP ``reduction(...)`` clause -- the OMP runtime handles privatization +
    final tree-reduce, an extra per-iter atomic is strictly wasted work."""
    sdfg = _build_wcr_scalar_sum()
    csdfg, src = _compile_and_read_src(sdfg)

    # Some ``reduce_atomic`` call may legitimately exist elsewhere in the
    # generated TU (other WCR writes), but specifically NOT on ``acc``.
    bad = [l for l in src.splitlines() if "reduce_atomic" in l and "acc" in l]
    assert not bad, "found per-iter atomic on the OMP-reduction-covered target:\n" + "\n".join(bad)


def test_scalar_wcr_numerically_correct():
    """End-to-end: the OMP-reduction code path computes the sum correctly."""
    sdfg = _build_wcr_scalar_sum()
    csdfg, _ = _compile_and_read_src(sdfg)

    n = 1024
    rng = np.random.default_rng(0)
    src = rng.random(n)
    out = np.array([0.0])
    csdfg(src=src, out=out, N=n)
    assert np.isclose(float(out[0]), float(src.sum()))


def test_multiple_omp_reducible_targets_fall_back_to_atomic():
    """With 2+ scalar WCR targets in one map, we refuse the OMP reduction clause
    (single-target limit) and the atomic path emits as before."""
    sdfg = dace.SDFG("wcr_two_scalars")
    sdfg.add_array("src", [N], dace.float64)
    sdfg.add_scalar("acc1", dace.float64, transient=True)
    sdfg.add_scalar("acc2", dace.float64, transient=True)
    sdfg.add_array("out1", [1], dace.float64)
    sdfg.add_array("out2", [1], dace.float64)

    init = sdfg.add_state("init", is_start_block=True)
    init.add_edge(init.add_tasklet("z1", {}, {"o"}, "o = 0.0"), "o", init.add_write("acc1"), None, dace.Memlet("acc1[0]"))
    init.add_edge(init.add_tasklet("z2", {}, {"o"}, "o = 0.0"), "o", init.add_write("acc2"), None, dace.Memlet("acc2[0]"))

    ms = sdfg.add_state("ms")
    sdfg.add_edge(init, ms, dace.InterstateEdge())
    me, mx = ms.add_map("m", dict(i="0:N"), schedule=dace.ScheduleType.CPU_Multicore)
    t = ms.add_tasklet("two", {"v"}, {"r1", "r2"}, "r1 = v; r2 = 2.0 * v")
    ms.add_memlet_path(ms.add_read("src"), me, t, dst_conn="v", memlet=dace.Memlet("src[i]"))
    ms.add_memlet_path(t, mx, ms.add_write("acc1"), src_conn="r1",
                       memlet=dace.Memlet("acc1[0]", wcr="lambda a, b: a + b"))
    ms.add_memlet_path(t, mx, ms.add_write("acc2"), src_conn="r2",
                       memlet=dace.Memlet("acc2[0]", wcr="lambda a, b: a + b"))

    post = sdfg.add_state("post")
    sdfg.add_edge(ms, post, dace.InterstateEdge())
    for tag in ("1", "2"):
        wb = post.add_tasklet("wb" + tag, {"i"}, {"o"}, "o = i")
        post.add_edge(post.add_read("acc" + tag), None, wb, "i", dace.Memlet("acc" + tag + "[0]"))
        post.add_edge(wb, "o", post.add_write("out" + tag), None, dace.Memlet("out" + tag + "[0]"))

    sdfg.validate()
    csdfg, src = _compile_and_read_src(sdfg)
    pragma_lines = [l for l in src.splitlines() if "#pragma omp parallel for" in l]
    assert not any("reduction(" in l for l in pragma_lines), \
        "expected no reduction(...) clause when 2+ targets qualify -- got:\n" + "\n".join(pragma_lines)


if __name__ == "__main__":
    test_scalar_wcr_emits_omp_reduction_clause()
    test_scalar_wcr_does_not_emit_atomic_for_covered_target()
    test_scalar_wcr_numerically_correct()
    test_multiple_omp_reducible_targets_fall_back_to_atomic()
