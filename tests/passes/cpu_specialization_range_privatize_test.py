# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``RangePrivatizeScatterReductions``: a parallel scatter ``out[f(i, idx)] (op)= v`` keeps a private copy of only
the rows its thread's chunk writes, instead of an OpenMP ``reduction(op:out[0:M])`` copy of all of ``out``.

Each test runs canonicalize + finalize for the CPU and asserts the emitted shape (the pre-pass and private span,
no array-section clause) and the values against numpy. The scatter maps run on a fixed team of four threads, so
the combine of overlapping spans and the atomic fallback run even when the suite pins one OpenMP thread.
"""

import os

import numpy as np
import pytest

import dace
from dace import nodes
from dace.config import set_temporary
from dace.frontend import operations
from dace.transformation.passes.canonicalize.finalize import finalize_for_target
from dace.transformation.passes.canonicalize.pipeline import canonicalize
from dace.transformation.passes.cpu_specialization.range_privatize_scatter_reductions import (
    RangePrivatizeScatterReductions,
    range_private_accumulators,
)

N = dace.symbol("N", dace.int64, positive=True)
M = dace.symbol("M", dace.int64, positive=True)
T = dace.symbol("T", dace.int64, positive=True)
K = dace.symbol("K", dace.int64, positive=True)
L = dace.symbol("L", dace.int64)

NV, MV, THREADS = 4099, 61, 4


@dace.program
def scatter_sum(idx: dace.int64[N], val: dace.float64[N], out: dace.float64[M]):
    for i in dace.map[0:N]:
        out[idx[i]] += val[i]


@dace.program
def scatter_any_length(idx: dace.int64[L], val: dace.float64[L], out: dace.float64[M]):
    for i in dace.map[0:L]:
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
def scatter_product(idx: dace.int64[N], val: dace.float64[N], out: dace.float64[M]):
    for i in dace.map[0:N]:
        out[idx[i]] *= val[i]


@dace.program
def scatter_bitwise(idx: dace.int64[N], bits: dace.int64[N], o: dace.int64[M], x: dace.int64[M], a: dace.int64[M]):
    for i in dace.map[0:N]:
        o[idx[i]] |= bits[i]
        x[idx[i]] ^= bits[i]
        a[idx[i]] &= bits[i]


@dace.program
def scatter_two_dtypes(idx: dace.int64[N], val: dace.float64[N], out: dace.float64[M], cnt: dace.int32[M]):
    for i in dace.map[0:N]:
        out[idx[i]] += val[i]
        cnt[idx[i]] += 1


@dace.program
def scatter_two_index_sets(
    a: dace.int64[N], b: dace.int64[N], val: dace.float64[N], out: dace.float64[M], cnt: dace.int64[M]
):
    for i in dace.map[0:N]:
        out[a[i]] += val[i]
        cnt[b[i]] += 1


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
def clamped_histogram(r: dace.float64[N], out: dace.int64[M]):
    for i in dace.map[0:N]:
        k = min(max(int(r[i] * M), 0), M - 1)
        out[k] += 1


@dace.program
def scatter_hashed(idx: dace.int64[N], val: dace.float64[N], out: dace.float64[M]):
    for i in dace.map[0:N]:
        out[(idx[i] * 7 + 3) % M] += val[i]


@dace.program
def scatter_guarded(idx: dace.int64[N], val: dace.float64[N], out: dace.float64[M]):
    for i in dace.map[0:N]:
        if val[i] > 0.5:
            out[idx[i]] += val[i]


@dace.program
def csr_transpose_spmv(
    ptr: dace.int64[N + 1], col: dace.int64[4 * N], val: dace.float64[4 * N], x: dace.float64[N], out: dace.float64[M]
):
    for r in dace.map[0:N]:
        for k in range(ptr[r], ptr[r + 1]):
            out[col[k]] += val[k] * x[r]


@dace.program
def scatter_twice(idx: dace.int64[N], val: dace.float64[N], out: dace.float64[M]):
    for i in dace.map[0:N]:
        out[idx[i]] += val[i]
    for i in dace.map[0:N]:
        out[idx[i]] += 2 * val[i]


@dace.program
def scatter_own_row(idx: dace.int64[K, N], val: dace.float64[K, N], out: dace.float64[K, M]):
    for j in dace.map[0:K]:
        for i in dace.map[0:N]:
            out[j, idx[j, i]] += val[j, i]


@dace.program
def triangular_packed_scatter(bb: dace.float64[N, N], flat: dace.float64[N * N]):
    for i in range(N):
        k = (i + 1) * i // 2 + i
        for j in range(i, N):
            flat[k] = flat[k] + bb[j, i]
            k = k + j + 1


@dace.program
def elementwise_add(val: dace.float64[N], out: dace.float64[N]):
    for i in dace.map[0:N]:
        out[i] += val[i]


@dace.program
def scatter_block_loop(idx: dace.int64[N], val: dace.float64[N, K, K], out: dace.float64[M, K, K]):
    for k in range(N):
        for a in range(K):
            for b in range(K):
                out[idx[k], a, b] += val[k, a, b]


@dace.program
def gather_sum(idx: dace.int64[N], val: dace.float64[M], out: dace.float64[N]):
    for i in dace.map[0:N]:
        out[i] += val[idx[i]]


def _canonical(program) -> dace.SDFG:
    sdfg = program.to_sdfg(simplify=True)
    canonicalize(sdfg, target="cpu")
    finalize_for_target(sdfg, "cpu")
    return sdfg


def _marked_maps(sdfg: dace.SDFG) -> list[nodes.Map]:
    return [n.map for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.MapEntry) and n.map.omp_range_privatize]


def _specialize(program, tag: str, threads: int = THREADS) -> tuple[dace.SDFG, str]:
    """``program`` specialized for the CPU, its range-privatized maps on ``threads`` threads, and its code."""
    sdfg = _canonical(program)
    for m in _marked_maps(sdfg):
        m.omp_num_threads = threads
    sdfg.name = f"{sdfg.name}_{tag}"
    sdfg.build_folder = os.path.join(sdfg.build_folder + f"_{os.getpid()}")
    with set_temporary("compiler", "cpu", "implementation", value="experimental_readable"):
        code = "\n".join(obj.clean_code for obj in sdfg.generate_code())
    return sdfg, code


def _assert_range_privatized(sdfg: dace.SDFG, code: str, *names: str, maps: int = 1) -> None:
    assert len(_marked_maps(sdfg)) == maps, "the scatter map must be range-privatized"
    for name in names or ("out",):
        assert f"__rp0_{name}.open(" in code, code
    assert "reduction(" not in code, "no thread may hold an array-section copy of the whole accumulator"


def _indices(pattern: str, shape, m: int) -> np.ndarray:
    rng = np.random.default_rng(20261021)
    if pattern == "equal":
        return np.full(shape, m // 2, dtype=np.int64)
    if pattern == "extremes":
        return (np.arange(np.prod(shape)) % 2 * (m - 1)).reshape(shape).astype(np.int64)
    idx = rng.integers(0, m, shape)
    if pattern == "sorted":
        return np.sort(idx, axis=None).reshape(shape)
    if pattern == "reversed":
        return np.sort(idx, axis=None)[::-1].reshape(shape).copy()
    return idx


def _run(sdfg: dace.SDFG, **kwargs) -> None:
    with set_temporary("compiler", "cpu", "implementation", value="experimental_readable"):
        sdfg(**kwargs)


@pytest.mark.parametrize(
    "pattern, m",
    [
        ("sorted", MV),
        ("random", MV),
        ("reversed", MV),
        ("equal", MV),
        ("extremes", MV),
        ("random", 100003),
    ],
)
def test_float_sum_matches_add_at(pattern, m):
    """Equal indices contend on one row, extremes and a sparse random scatter span the whole array with few
    updates per row, so a thread there updates the shared rows atomically instead of privatizing."""
    sdfg, code = _specialize(scatter_sum, f"sum_{pattern}_{m}")
    _assert_range_privatized(sdfg, code)
    assert "__rp0_atomic_out ? (void)dace::wcr_fixed<dace::ReductionType::Sum, double>::reduce_atomic(" in code
    idx = _indices(pattern, NV, m)
    val = np.random.default_rng(1).random(NV)
    out = np.random.default_rng(2).random(m)
    ref = out.copy()
    np.add.at(ref, idx, val)
    _run(sdfg, idx=idx, val=val, out=out, N=NV, M=m)
    assert np.allclose(out, ref, rtol=1e-12, atol=1e-12)


def test_iterations_sorted_by_row_match_add_at_and_repeat_bitwise():
    """A random scatter over an array larger than a thread's cache share: every chunk spans all rows, so the
    iterations are sorted by row and each thread privatizes about an eighth of them; the fold is in thread order."""
    sdfg, code = _specialize(scatter_sum, "sum_sorted", threads=8)
    _assert_range_privatized(sdfg, code)
    n, m = 1 << 21, 1 << 18
    idx = _indices("random", n, m)
    val = np.random.default_rng(5).random(n)
    out = np.random.default_rng(6).random(m)
    ref = out.copy()
    np.add.at(ref, idx, val)
    first, second = out.copy(), out.copy()
    _run(sdfg, idx=idx, val=val, out=first, N=n, M=m)
    _run(sdfg, idx=idx, val=val, out=second, N=n, M=m)
    assert np.allclose(first, ref, rtol=1e-12, atol=1e-12)
    assert np.array_equal(first, second)


def test_block_rows_split_by_no_chunk_are_written_in_place_bitwise():
    """A blocked-CSR scatter: block row ``r``'s nine products land on ``3r .. 3r + 2`` in any order, and the even
    chunk boundaries cut block rows. Each boundary moves to the next block row, so every thread alone writes its
    rows, in iteration order, straight into the shared array: bitwise ``np.add.at``."""
    sdfg, code = _specialize(scatter_sum, "sum_owned", threads=8)
    _assert_range_privatized(sdfg, code)
    rows = NV // 9
    idx = np.repeat(np.arange(rows), 9) * 3 + np.random.default_rng(9).integers(0, 3, 9 * rows)
    val = np.random.default_rng(7).random(idx.size)
    out = np.random.default_rng(8).random(3 * rows)
    ref = out.copy()
    np.add.at(ref, idx, val)
    _run(sdfg, idx=idx, val=val, out=out, N=idx.size, M=3 * rows)
    assert np.array_equal(out, ref)


@pytest.mark.parametrize("n", [0, 1, 3])
def test_fewer_iterations_than_threads(n):
    """Threads with an empty chunk hold no span and take no part in the combine."""
    sdfg, code = _specialize(scatter_any_length, f"short_{n}", threads=8)
    _assert_range_privatized(sdfg, code)
    idx = _indices("random", n, MV)
    val = np.random.default_rng(3).random(n)
    out = np.random.default_rng(4).random(MV)
    ref = out.copy()
    np.add.at(ref, idx, val)
    _run(sdfg, idx=idx, val=val, out=out, L=n, M=MV)
    assert np.array_equal(out, ref)


def test_complex_sum_matches_add_at():
    sdfg, code = _specialize(scatter_complex_sum, "csum")
    _assert_range_privatized(sdfg, code)
    assert "Accumulator<dace::ReductionType::Sum, dace::complex128, false>" in code, "no host atomic on complex"
    assert "declare reduction" not in code
    rng = np.random.default_rng(3)
    idx = _indices("random", NV, MV)
    val = rng.random(NV) + 1j * rng.random(NV)
    out = rng.random(MV) + 1j * rng.random(MV)
    ref = out.copy()
    np.add.at(ref, idx, val)
    _run(sdfg, idx=idx, val=val, out=out, N=NV, M=MV)
    assert np.allclose(out, ref, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("pattern", ["sorted", "random", "equal"])
def test_integer_histogram_is_exact(pattern):
    sdfg, code = _specialize(scatter_count, f"count_{pattern}")
    _assert_range_privatized(sdfg, code)
    idx = _indices(pattern, NV, MV)
    out = np.arange(MV, dtype=np.int64)
    ref = out + np.bincount(idx, minlength=MV)
    _run(sdfg, idx=idx, out=out, N=NV, M=MV)
    assert np.array_equal(out, ref)


@pytest.mark.parametrize("program, combine", [(scatter_min, np.minimum), (scatter_max, np.maximum)])
def test_min_max_start_at_the_identity_and_are_exact(program, combine):
    """min / max have no lock-free host atomic, so every thread privatizes, whatever its span."""
    sdfg, code = _specialize(program, program.name)
    _assert_range_privatized(sdfg, code)
    op = "Min" if program is scatter_min else "Max"
    assert f"Accumulator<dace::ReductionType::{op}, double, false>" in code
    idx = _indices("random", NV, MV)
    val = np.random.default_rng(4).standard_normal(NV)
    out = np.random.default_rng(5).standard_normal(MV)
    ref = out.copy()
    combine.at(ref, idx, val)
    _run(sdfg, idx=idx, val=val, out=out, N=NV, M=MV)
    assert np.array_equal(out, ref)


def test_product_starts_at_one():
    sdfg, code = _specialize(scatter_product, "prod")
    _assert_range_privatized(sdfg, code)
    idx = _indices("random", NV, MV)
    val = np.random.default_rng(6).uniform(0.9, 1.1, NV)
    out = np.random.default_rng(7).uniform(0.5, 2.0, MV)
    ref = out.copy()
    np.multiply.at(ref, idx, val)
    _run(sdfg, idx=idx, val=val, out=out, N=NV, M=MV)
    assert np.allclose(out, ref, rtol=1e-12, atol=0)


def test_bitwise_identities_are_exact():
    sdfg, code = _specialize(scatter_bitwise, "bits")
    _assert_range_privatized(sdfg, code, "o", "x", "a")
    idx = _indices("random", NV, MV)
    bits = np.random.default_rng(8).integers(0, 1 << 40, NV)
    o, x, a = (np.random.default_rng(s).integers(0, 1 << 40, MV) for s in (9, 10, 11))
    refs = [o.copy(), x.copy(), a.copy()]
    for ufunc, ref in zip((np.bitwise_or, np.bitwise_xor, np.bitwise_and), refs):
        ufunc.at(ref, idx, bits)
    _run(sdfg, idx=idx, bits=bits, o=o, x=x, a=a, N=NV, M=MV)
    for got, ref in zip((o, x, a), refs):
        assert np.array_equal(got, ref)


def test_two_accumulators_of_different_types_share_one_pre_pass():
    sdfg, code = _specialize(scatter_two_dtypes, "twotypes")
    _assert_range_privatized(sdfg, code, "out", "cnt")
    assert "Accumulator<dace::ReductionType::Sum, int, true>" in code
    assert "__rp0_lo[0] = " in code and "__rp0_lo[1] = " not in code, "the same rows need one pre-pass"
    idx = _indices("random", NV, MV)
    val = np.random.default_rng(12).random(NV)
    out = np.zeros(MV)
    cnt = np.zeros(MV, dtype=np.int32)
    ref = np.zeros(MV)
    np.add.at(ref, idx, val)
    _run(sdfg, idx=idx, val=val, out=out, cnt=cnt, N=NV, M=MV)
    assert np.allclose(out, ref, rtol=1e-12, atol=1e-12)
    assert np.array_equal(cnt, np.bincount(idx, minlength=MV))


def test_two_accumulators_with_different_indices_have_one_span_each():
    sdfg, code = _specialize(scatter_two_index_sets, "twoindex")
    _assert_range_privatized(sdfg, code, "out", "cnt")
    assert "__rp0_lo[1] = " in code
    a = _indices("sorted", NV, MV)
    b = _indices("random", NV, MV)
    val = np.random.default_rng(19).random(NV)
    out = np.zeros(MV)
    cnt = np.zeros(MV, dtype=np.int64)
    ref = np.zeros(MV)
    np.add.at(ref, a, val)
    _run(sdfg, a=a, b=b, val=val, out=out, cnt=cnt, N=NV, M=MV)
    assert np.allclose(out, ref, rtol=1e-12, atol=1e-12)
    assert np.array_equal(cnt, np.bincount(b, minlength=MV))


@pytest.mark.parametrize("pattern", ["sorted", "random"])
def test_multidimensional_rows_match_add_at(pattern):
    sdfg, code = _specialize(scatter_rows, f"rows_{pattern}")
    _assert_range_privatized(sdfg, code)
    idx = _indices(pattern, NV, MV)
    val = np.random.default_rng(6).random((NV, 3))
    out = np.random.default_rng(13).random((MV, 3))
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


def test_a_block_row_read_inside_the_block_loops_is_noted_once():
    """The row ``idx[k]`` is bound inside the ``a, b`` block loops, but reads neither and the loops run (``K`` is
    positive): the pre-pass reads it once per ``k`` and counts the ``K * K`` writes, with no loop of its own."""
    sdfg, code = _specialize(scatter_block_loop, "block_loop")
    _assert_range_privatized(sdfg, code)
    prepass = code[code.index("__rp0_lo[0] = std") : code.index("__rp0_order.note(")]
    assert "for (" not in prepass, prepass
    k = 3
    idx = _indices("sorted", NV, MV)
    val = np.random.default_rng(11).random((NV, k, k))
    out = np.random.default_rng(12).random((MV, k, k))
    ref = out.copy()
    np.add.at(ref, idx, val)
    _run(sdfg, idx=idx, val=val, out=out, N=NV, M=MV, K=k)
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


def test_clamped_computed_index_is_re_evaluated():
    """The bin ``min(max(int(r * M), 0), M - 1)`` is computed by tasklets and an interstate assignment; the
    pre-pass evaluates the same chain."""
    sdfg, code = _specialize(clamped_histogram, "hist")
    _assert_range_privatized(sdfg, code)
    r = np.random.default_rng(14).uniform(-0.2, 1.2, NV)
    out = np.zeros(MV, dtype=np.int64)
    ref = np.bincount(np.clip((r * MV).astype(np.int64), 0, MV - 1), minlength=MV)
    _run(sdfg, r=r, out=out, N=NV, M=MV)
    assert np.array_equal(out, ref)


def test_arithmetic_index_of_an_index_array():
    sdfg, code = _specialize(scatter_hashed, "hashed")
    _assert_range_privatized(sdfg, code)
    idx = _indices("random", NV, 10 * MV)
    val = np.random.default_rng(15).random(NV)
    out = np.zeros(MV)
    ref = out.copy()
    np.add.at(ref, (idx * 7 + 3) % MV, val)
    _run(sdfg, idx=idx, val=val, out=out, N=NV, M=MV)
    assert np.allclose(out, ref, rtol=1e-12, atol=1e-12)


def test_guarded_write_is_counted_only_where_it_runs():
    sdfg, code = _specialize(scatter_guarded, "guarded")
    _assert_range_privatized(sdfg, code)
    assert "if ((__rp_val_index > 0.5))" in code, "the pre-pass evaluates the guard"
    idx = _indices("random", NV, MV)
    val = np.random.default_rng(16).random(NV)
    out = np.zeros(MV)
    ref = out.copy()
    taken = val > 0.5
    np.add.at(ref, idx[taken], val[taken])
    _run(sdfg, idx=idx, val=val, out=out, N=NV, M=MV)
    assert np.allclose(out, ref, rtol=1e-12, atol=1e-12)


def test_index_inside_a_data_dependent_loop():
    """CSR transpose: the row's nonzeros ``ptr[r]:ptr[r + 1]`` bound a loop the pre-pass replays."""
    sdfg, code = _specialize(csr_transpose_spmv, "csr")
    _assert_range_privatized(sdfg, code)
    rng = np.random.default_rng(17)
    ptr = np.concatenate([[0], np.cumsum(rng.integers(0, 5, NV))]).astype(np.int64)
    col = np.zeros(4 * NV, dtype=np.int64)
    col[: ptr[-1]] = rng.integers(0, MV, ptr[-1])
    val = rng.random(4 * NV)
    x = rng.random(NV)
    out = np.zeros(MV)
    rows = np.repeat(np.arange(NV), np.diff(ptr))
    ref = np.zeros(MV)
    np.add.at(ref, col[: ptr[-1]], val[: ptr[-1]] * x[rows])
    _run(sdfg, ptr=ptr, col=col, val=val, x=x, out=out, N=NV, M=MV)
    assert np.allclose(out, ref, rtol=1e-12, atol=1e-12)


def test_a_privatized_body_is_not_shared_with_an_unprivatized_twin():
    """Two maps run the same body; only the first is range-privatized, so its nest (which takes the atomic
    flag) must not be reused by the second."""
    sdfg = _canonical(scatter_twice)
    marked = _marked_maps(sdfg)
    assert len(marked) == 2
    marked[0].omp_num_threads = THREADS
    marked[1].omp_range_privatize = False
    sdfg.name = f"{sdfg.name}_twins"
    with set_temporary("compiler", "cpu", "implementation", value="experimental_readable"):
        code = "\n".join(obj.clean_code for obj in sdfg.generate_code())
    assert code.count("__rp0_out.open(") == 1 and "__rp1" not in code
    idx = _indices("random", NV, MV)
    val = np.random.default_rng(18).random(NV)
    out = np.zeros(MV)
    ref = np.zeros(MV)
    np.add.at(ref, idx, 3 * val)
    _run(sdfg, idx=idx, val=val, out=out, N=NV, M=MV)
    assert np.allclose(out, ref, rtol=1e-12, atol=1e-12)


def test_a_row_owned_by_the_map_variable_is_left_alone():
    """``out[j, idx[j, i]]``: each iteration of the parallel ``j`` map writes its own row, no reduction."""
    sdfg = _canonical(scatter_own_row)
    assert not _marked_maps(sdfg)


def test_a_scatter_inside_another_parallel_scope_is_refused():
    sdfg = _canonical(scatter_own_row)
    state, outer, inner = next(
        (st, st.entry_node(n), n)
        for n, st in sdfg.all_nodes_recursive()
        if isinstance(n, nodes.MapEntry) and st.entry_node(n) is not None
    )
    inner.map.schedule = dace.ScheduleType.CPU_Multicore
    edge = next(e for e in state.out_edges(state.exit_node(inner)) if e.data.wcr is not None)
    assert operations.detect_reduction_type(edge.data.wcr) == dace.ReductionType.Sum
    assert outer.map.schedule == dace.ScheduleType.CPU_Multicore
    assert range_private_accumulators(state, inner) == {}
    outer.map.schedule = dace.ScheduleType.Sequential
    assert set(range_private_accumulators(state, inner)) == {"out"}


def test_the_analysis_leaves_the_sdfg_unchanged():
    """tsvc s141: the analysis walks the header of the inner loop before refusing the carried ``k``; it must
    not rewrite the loop's code while renaming its variable."""
    sdfg = triangular_packed_scatter.to_sdfg(simplify=True)
    canonicalize(sdfg, target="cpu")
    finalize_for_target(sdfg, "cpu")
    before = sdfg.hash_sdfg()
    for node, state in sdfg.all_nodes_recursive():
        if isinstance(node, nodes.MapEntry):
            range_private_accumulators(state, node)
    assert sdfg.hash_sdfg() == before
    sdfg.validate()
    bb = np.random.default_rng(20).random((11, 11))
    flat = np.zeros(121)
    ref = np.zeros(121)
    for i in range(11):
        k = i * (i + 1) // 2 + i
        for j in range(i, 11):
            ref[k] += bb[j, i]
            k += j + 1
    _run(sdfg, bb=bb, flat=flat, N=11)
    assert np.allclose(flat, ref, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("program", [elementwise_add, gather_sum])
def test_a_map_that_does_not_scatter_is_left_alone(program):
    sdfg, code = _specialize(program, program.name)
    assert not _marked_maps(sdfg)
    assert "__rp0" not in code
    idx = _indices("random", NV, MV)
    val = np.random.default_rng(9).random(MV if program is gather_sum else NV)
    out = np.random.default_rng(10).random(NV)
    ref = out + (val[idx] if program is gather_sum else val)
    kwargs = dict(idx=idx, M=MV) if program is gather_sum else {}
    _run(sdfg, val=val, out=out, N=NV, **kwargs)
    assert np.allclose(out, ref)


def test_pass_marks_once():
    sdfg = _canonical(scatter_sum)
    assert len(_marked_maps(sdfg)) == 1
    assert RangePrivatizeScatterReductions().apply_pass(sdfg, {}) is None, "a marked map is not marked again"


def test_a_row_stride_spelled_as_a_power_is_contiguous():
    """cp2k_trs4's ``sink[3 * n + 1, b, b]`` reaches the check with the row stride ``ipow(b, 2)``: the same
    ``b * b`` as the extents, so the scatter into it is range-privatized, not left to an atomic per element."""
    from dace.data import Array
    from dace.symbolic import pystr_to_symbolic
    from dace.transformation.passes.cpu_specialization.range_privatize_scatter_reductions import _contiguous_rows

    desc = Array(dace.float64, [M, K, K], strides=[pystr_to_symbolic("ipow(K, 2)"), K, 1])
    assert _contiguous_rows(desc)


if __name__ == "__main__":
    for p, m in [("sorted", MV), ("random", MV), ("reversed", MV), ("equal", MV), ("extremes", MV), ("random", 100003)]:
        test_float_sum_matches_add_at(p, m)
    test_block_rows_split_by_no_chunk_are_written_in_place_bitwise()
    for n in (0, 1, 3):
        test_fewer_iterations_than_threads(n)
    test_complex_sum_matches_add_at()
    for p in ("sorted", "random", "equal"):
        test_integer_histogram_is_exact(p)
    test_min_max_start_at_the_identity_and_are_exact(scatter_min, np.minimum)
    test_min_max_start_at_the_identity_and_are_exact(scatter_max, np.maximum)
    test_product_starts_at_one()
    test_bitwise_identities_are_exact()
    test_two_accumulators_of_different_types_share_one_pre_pass()
    test_two_accumulators_with_different_indices_have_one_span_each()
    test_multidimensional_rows_match_add_at("sorted")
    test_multidimensional_rows_match_add_at("random")
    test_element_corners_lulesh_shape()
    test_a_block_row_read_inside_the_block_loops_is_noted_once()
    test_scatter_under_a_sequential_loop()
    test_clamped_computed_index_is_re_evaluated()
    test_arithmetic_index_of_an_index_array()
    test_guarded_write_is_counted_only_where_it_runs()
    test_index_inside_a_data_dependent_loop()
    test_a_privatized_body_is_not_shared_with_an_unprivatized_twin()
    test_a_row_owned_by_the_map_variable_is_left_alone()
    test_a_scatter_inside_another_parallel_scope_is_refused()
    test_the_analysis_leaves_the_sdfg_unchanged()
    test_a_map_that_does_not_scatter_is_left_alone(elementwise_add)
    test_a_map_that_does_not_scatter_is_left_alone(gather_sum)
    test_pass_marks_once()
    test_a_row_stride_spelled_as_a_power_is_contiguous()
