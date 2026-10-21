# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``RangePrivatizeScatterReductions``: a parallel scatter ``out[idx[i]] (op)= v`` keeps a private copy of
only the rows its thread's chunk writes, instead of an OpenMP ``reduction(op:out[0:M])`` copy of all of ``out``.

Each test runs canonicalize + finalize for the CPU and asserts the emitted shape (the pre-pass and private
span, no array-section clause) and the values against the sequential numpy reference.
"""

import os

import numpy as np
import pytest

import dace
from dace import nodes
from dace.config import set_temporary
from dace.transformation.passes.canonicalize.finalize import finalize_for_target
from dace.transformation.passes.canonicalize.pipeline import canonicalize
from dace.transformation.passes.cpu_specialization.range_privatize_scatter_reductions import (
    RangePrivatizeScatterReductions,
)

N = dace.symbol("N", dace.int64, positive=True)
M = dace.symbol("M", dace.int64, positive=True)
T = dace.symbol("T", dace.int64, positive=True)

NV, MV = 4099, 61


@dace.program
def scatter_sum(idx: dace.int64[N], val: dace.float64[N], out: dace.float64[M]):
    for i in dace.map[0:N]:
        out[idx[i]] += val[i]


@dace.program
def scatter_complex_sum(idx: dace.int64[N], val: dace.complex128[N], out: dace.complex128[M]):
    for i in dace.map[0:N]:
        out[idx[i]] += val[i]


@dace.program
def scatter_count(idx: dace.int64[N], out: dace.int64[M]):
    for i in dace.map[0:N]:
        out[idx[i]] += 1


@dace.program
def scatter_min(idx: dace.int64[N], val: dace.float64[N], out: dace.float64[M]):
    for i in dace.map[0:N]:
        out[idx[i]] = min(out[idx[i]], val[i])


@dace.program
def scatter_max(idx: dace.int64[N], val: dace.float64[N], out: dace.float64[M]):
    for i in dace.map[0:N]:
        out[idx[i]] = max(out[idx[i]], val[i])


@dace.program
def scatter_rows(idx: dace.int64[N], val: dace.float64[N, 3], out: dace.float64[M, 3]):
    for i in dace.map[0:N]:
        for c in range(3):
            out[idx[i], c] += val[i, c]


@dace.program
def scatter_corners(nodelist: dace.int64[N, 8], val: dace.float64[N, 8], out: dace.float64[M]):
    for e in dace.map[0:N]:
        for k in range(8):
            out[nodelist[e, k]] += val[e, k]


@dace.program
def scatter_in_time_loop(idx: dace.int64[N], val: dace.float64[N], out: dace.float64[M]):
    for t in range(T):
        for i in dace.map[0:N]:
            out[idx[i]] += val[i] * (t + 1)


@dace.program
def elementwise_add(val: dace.float64[N], out: dace.float64[N]):
    for i in dace.map[0:N]:
        out[i] += val[i]


@dace.program
def gather_sum(idx: dace.int64[N], val: dace.float64[M], out: dace.float64[N]):
    for i in dace.map[0:N]:
        out[i] += val[idx[i]]


def _specialize(program, tag: str) -> tuple[dace.SDFG, str]:
    """``program`` canonicalized and specialized for the CPU, and its generated code."""
    sdfg = program.to_sdfg(simplify=True)
    canonicalize(sdfg, target="cpu")
    finalize_for_target(sdfg, "cpu")
    sdfg.build_folder = os.path.join(sdfg.build_folder + f"_{tag}_{os.getpid()}")
    with set_temporary("compiler", "cpu", "implementation", value="experimental_readable"):
        code = "\n".join(obj.clean_code for obj in sdfg.generate_code())
    return sdfg, code


def _marked(sdfg: dace.SDFG) -> int:
    return sum(1 for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.MapEntry) and n.map.omp_range_privatize)


def _assert_range_privatized(sdfg: dace.SDFG, code: str) -> None:
    assert _marked(sdfg) == 1, "the scatter map must be range-privatized"
    assert "#pragma omp for schedule(static) nowait" in code and "__rp0_priv_out" in code, code
    assert "reduction(" not in code, "no thread may hold an array-section copy of the whole accumulator"


def _indices(pattern: str, shape, m: int) -> np.ndarray:
    rng = np.random.default_rng(20261021)
    idx = rng.integers(0, m, shape)
    return np.sort(idx, axis=None).reshape(shape) if pattern == "sorted" else idx


def _run(sdfg: dace.SDFG, **kwargs) -> None:
    with set_temporary("compiler", "cpu", "implementation", value="experimental_readable"):
        sdfg(**kwargs)


@pytest.mark.parametrize("pattern", ["sorted", "random"])
def test_float_sum_matches_add_at(pattern):
    sdfg, code = _specialize(scatter_sum, f"sum_{pattern}")
    _assert_range_privatized(sdfg, code)
    idx = _indices(pattern, NV, MV)
    val = np.random.default_rng(1).random(NV)
    out = np.random.default_rng(2).random(MV)
    ref = out.copy()
    np.add.at(ref, idx, val)
    _run(sdfg, idx=idx, val=val, out=out, N=NV, M=MV)
    assert np.allclose(out, ref, rtol=1e-12, atol=1e-12)


def test_complex_sum_matches_add_at():
    sdfg, code = _specialize(scatter_complex_sum, "csum")
    _assert_range_privatized(sdfg, code)
    assert "declare reduction" not in code
    rng = np.random.default_rng(3)
    idx = _indices("sorted", NV, MV)
    val = rng.random(NV) + 1j * rng.random(NV)
    out = rng.random(MV) + 1j * rng.random(MV)
    ref = out.copy()
    np.add.at(ref, idx, val)
    _run(sdfg, idx=idx, val=val, out=out, N=NV, M=MV)
    assert np.allclose(out, ref, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("pattern", ["sorted", "random"])
def test_integer_histogram_is_exact(pattern):
    sdfg, code = _specialize(scatter_count, f"count_{pattern}")
    _assert_range_privatized(sdfg, code)
    idx = _indices(pattern, NV, MV)
    out = np.arange(MV, dtype=np.int64)
    ref = out + np.bincount(idx, minlength=MV)
    _run(sdfg, idx=idx, out=out, N=NV, M=MV)
    assert np.array_equal(out, ref)


@pytest.mark.parametrize("program, combine", [(scatter_min, np.minimum), (scatter_max, np.maximum)])
def test_min_max_are_exact(program, combine):
    sdfg, code = _specialize(program, program.name)
    _assert_range_privatized(sdfg, code)
    assert "numeric_limits<double>::infinity()" in code, "the private span starts at the operator's identity"
    idx = _indices("random", NV, MV)
    val = np.random.default_rng(4).standard_normal(NV)
    out = np.random.default_rng(5).standard_normal(MV)
    ref = out.copy()
    combine.at(ref, idx, val)
    _run(sdfg, idx=idx, val=val, out=out, N=NV, M=MV)
    assert np.array_equal(out, ref)


def test_multidimensional_rows_match_add_at():
    sdfg, code = _specialize(scatter_rows, "rows")
    _assert_range_privatized(sdfg, code)
    idx = _indices("sorted", NV, MV)
    val = np.random.default_rng(6).random((NV, 3))
    out = np.zeros((MV, 3))
    ref = out.copy()
    np.add.at(ref, idx, val)
    _run(sdfg, idx=idx, val=val, out=out, N=NV, M=MV)
    assert np.allclose(out, ref, rtol=1e-12, atol=1e-12)


def test_element_corners_lulesh_shape():
    """Eight corner indices per element, read from a 2-D node list: the pre-pass walks the inner loop too."""
    sdfg, code = _specialize(scatter_corners, "corners")
    _assert_range_privatized(sdfg, code)
    nodelist = _indices("sorted", (NV, 8), MV)
    val = np.random.default_rng(7).random((NV, 8))
    out = np.zeros(MV)
    ref = out.copy()
    np.add.at(ref, nodelist.ravel(), val.ravel())
    _run(sdfg, nodelist=nodelist, val=val, out=out, N=NV, M=MV)
    assert np.allclose(out, ref, rtol=1e-12, atol=1e-12)


def test_scatter_under_a_sequential_loop():
    """The map over ``(t, i)`` keeps ``i`` as an inner loop, which takes no ``simd``: its writes into the
    private span are plain read-modify-writes."""
    sdfg, code = _specialize(scatter_in_time_loop, "timeloop")
    _assert_range_privatized(sdfg, code)
    assert "omp simd" not in code
    idx = _indices("random", NV, MV)
    val = np.random.default_rng(8).random(NV)
    out = np.zeros(MV)
    ref = out.copy()
    for t in range(3):
        np.add.at(ref, idx, val * (t + 1))
    _run(sdfg, idx=idx, val=val, out=out, N=NV, M=MV, T=3)
    assert np.allclose(out, ref, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("program", [elementwise_add, gather_sum])
def test_a_map_that_does_not_scatter_is_left_alone(program):
    sdfg, code = _specialize(program, program.name)
    assert _marked(sdfg) == 0
    assert "__rp0" not in code
    idx = _indices("random", NV, MV)
    val = np.random.default_rng(9).random(MV if program is gather_sum else NV)
    out = np.random.default_rng(10).random(NV)
    ref = out + (val[idx] if program is gather_sum else val)
    kwargs = dict(idx=idx, M=MV) if program is gather_sum else {}
    _run(sdfg, val=val, out=out, N=NV, **kwargs)
    assert np.allclose(out, ref)


def test_pass_marks_once():
    sdfg = scatter_sum.to_sdfg(simplify=True)
    canonicalize(sdfg, target="cpu")
    finalize_for_target(sdfg, "cpu")
    assert _marked(sdfg) == 1
    assert RangePrivatizeScatterReductions().apply_pass(sdfg, {}) is None, "a marked map is not marked again"


if __name__ == "__main__":
    for p in ("sorted", "random"):
        test_float_sum_matches_add_at(p)
        test_integer_histogram_is_exact(p)
    test_complex_sum_matches_add_at()
    test_min_max_are_exact(scatter_min, np.minimum)
    test_min_max_are_exact(scatter_max, np.maximum)
    test_multidimensional_rows_match_add_at()
    test_element_corners_lulesh_shape()
    test_scatter_under_a_sequential_loop()
    test_a_map_that_does_not_scatter_is_left_alone(elementwise_add)
    test_a_map_that_does_not_scatter_is_left_alone(gather_sum)
    test_pass_marks_once()
