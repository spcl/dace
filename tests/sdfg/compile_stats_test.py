# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``SDFG.compile(stats=...)`` splits the wall-clock time of a compilation into code generation, the build and the
load, so a caller can tell them apart without wrapping DaCe's internals."""

import numpy as np

import dace

N = dace.symbol("N")


@dace.program
def doubled(a: dace.float64[N], b: dace.float64[N]):
    for i in dace.map[0:N]:
        b[i] = 2.0 * a[i]


def test_a_fresh_compilation_reports_each_step():
    sdfg = doubled.to_sdfg(simplify=True)
    sdfg.name = "compile_stats_fresh"
    with dace.config.set_temporary("compiler", "use_cache", value=False):
        stats = {}
        compiled = sdfg.compile(stats=stats)
    assert not stats["cached"]
    assert stats["codegen_seconds"] > 0 and stats["compile_seconds"] > 0 and stats["load_seconds"] > 0, stats
    a, b = np.arange(5.0), np.zeros(5)
    compiled(a=a, b=b, N=5)
    assert np.allclose(b, 2.0 * a)


def test_a_cached_library_reports_no_generation_or_build():
    sdfg = doubled.to_sdfg(simplify=True)
    sdfg.name = "compile_stats_cached"
    sdfg.compile()
    stats = {}
    with dace.config.set_temporary("compiler", "use_cache", value=True):
        sdfg.compile(stats=stats)
    assert stats["cached"] and stats["codegen_seconds"] == 0.0 and stats["compile_seconds"] == 0.0, stats


if __name__ == "__main__":
    test_a_fresh_compilation_reports_each_step()
    test_a_cached_library_reports_no_generation_or_build()
