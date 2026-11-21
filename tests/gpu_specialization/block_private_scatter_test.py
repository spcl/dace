# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A GPU kernel's scatter reduction folds into a private copy per thread block, which commits once per element.

A copy of at most ``compiler.cuda.block_private_shared_elements`` elements lives in shared memory; a larger one, of at
most ``compiler.cuda.block_private_global_elements``, is the block's slice of a global buffer ``PrivatizeScatterPerBlock``
adds to a kernel it grid-strides. Each path is forced through the shared capacity. The structural checks need no GPU;
the numeric ones run on the device, at sizes that leave the last block partial.
"""

import contextlib
import re

import numpy as np
import pytest

import dace
from dace.transformation.passes.canonicalize.finalize import finalize_for_target, offload_to_gpu
from dace.transformation.passes.canonicalize.pipeline import canonicalize

N, M = dace.symbol("N"), dace.symbol("M")
BINS = 100  # not a multiple of the 256-thread block
SIZE = 1000  # writes: not a multiple of the block either, and ten per bin


@dace.program
def scatter_add(idx: dace.int64[N], w: dace.float64[N], h: dace.float64[BINS]):
    for i in dace.map[0:N]:
        h[idx[i]] += w[i]


@dace.program
def scatter_add_symbolic(idx: dace.int64[N], w: dace.float64[N], h: dace.float64[M]):
    # The kernel reads ``M``, so it receives the size its copies are checked against
    for i in dace.map[0:N]:
        h[idx[i] % M] += w[i]


@dace.program
def scatter_rows(idx: dace.int64[N], w: dace.float64[N, 3], f: dace.float64[BINS, 3]):
    for i in dace.map[0:N]:
        f[idx[i], :] += w[i, :]


@dace.program
def scatter_min(idx: dace.int64[N], w: dace.float64[N], h: dace.float64[BINS]):
    for i in dace.map[0:N]:
        with dace.tasklet:
            ix << idx[i]
            v << w[i]
            out >> h(1, lambda a, b: min(a, b))[:]
            out[ix] = v


@dace.program
def scatter_max(idx: dace.int64[N], w: dace.float64[N], h: dace.float64[BINS]):
    for i in dace.map[0:N]:
        with dace.tasklet:
            ix << idx[i]
            v << w[i]
            out >> h(1, lambda a, b: max(a, b))[:]
            out[ix] = v


@dace.program
def scatter_sub(idx: dace.int64[N], w: dace.float64[N], h: dace.float64[BINS]):
    for i in dace.map[0:N]:
        with dace.tasklet:
            ix << idx[i]
            v << w[i]
            out >> h(1, lambda a, b: a - b)[:]
            out[ix] = v


@dace.program
def scatter_write(idx: dace.int64[N], w: dace.float64[N], h: dace.float64[BINS]):
    for i in dace.map[0:N]:
        h[idx[i]] = w[i]


@dace.program
def scatter_reading_target(idx: dace.int64[N], w: dace.float64[N], h: dace.float64[BINS]):
    for i in dace.map[0:N]:
        h[idx[i]] += w[i] * h[0]


@dace.program
def scatter_per_iteration(idx: dace.int64[N], w: dace.float64[N], h: dace.float64[N]):
    for i in dace.map[0:N]:
        h[idx[i]] += w[i]


@dace.program
def scatter_huge(idx: dace.int64[N], w: dace.float64[N], h: dace.float64[1000000]):
    for i in dace.map[0:N]:
        h[idx[i]] += w[i]


#: The shared capacity that sends a ``BINS``-element target to each copy.
SHARED_CAPACITY = {"shared": 2048, "global": 16}


@contextlib.contextmanager
def copies_in(path: str):
    with dace.config.set_temporary("compiler", "cuda", "block_private_shared_elements", value=SHARED_CAPACITY[path]):
        yield


def canonical_gpu(program, name: str) -> dace.SDFG:
    sdfg = program.to_sdfg(simplify=True)
    sdfg.name = name
    canonicalize(sdfg, target="gpu")
    offload_to_gpu(sdfg)
    finalize_for_target(sdfg, "gpu")
    return sdfg


def kernel_code(program, name: str, path: str = "shared") -> tuple[dace.SDFG, str]:
    sdfg = canonical_gpu(program, name)
    with copies_in(path):
        code = "\n".join(obj.clean_code for obj in sdfg.generate_code())
    return sdfg, code


def kernel_bodies(code: str) -> str:
    """The ``__global__`` functions of ``code``: where the kernel's own names (not a nested SDFG's) are in scope."""
    return "\n".join(re.findall(r"__global__ void .*?\n}\n", code, re.DOTALL))


def atomics_into(code: str, array: str) -> list[str]:
    """The pointer argument of every atomic update of ``array`` itself (not of a private copy) in the kernels."""
    pointers = re.findall(r"reduce_atomic\(\s*([^,]*),", kernel_bodies(code))
    return [p for p in pointers if re.search(rf"(^|[^\w]){array}\b", p) and "__bpart" not in p]


def bins(rng, n: int, m: int) -> np.ndarray:
    """``n`` bin indices below ``m``, every bin hit, most several times."""
    return np.concatenate([np.arange(m), rng.integers(0, m, n - m)]).astype(np.int64)


def run_on_device(program, name: str, path: str, target: str, **arrays) -> np.ndarray:
    """``program`` built with the copies of ``path``, run on device copies of ``arrays``; ``target`` after it."""
    cupy = pytest.importorskip("cupy")
    sdfg = canonical_gpu(program, name)
    with copies_in(path):
        compiled = sdfg.compile()
    device = {k: cupy.asarray(v) for k, v in arrays.items()}
    compiled(**device, N=SIZE)
    return cupy.asnumpy(device[target])


def test_a_scatter_into_a_small_array_folds_into_shared_memory_and_commits_once_per_element():
    _, code = kernel_code(scatter_add, "bps_shared_code")
    copy = re.search(rf"__shared__ double (__bpart_\w+)\[{BINS}\];", code).group(1)
    # The lanes update the block's copy, the body's nested SDFG through it; the array itself only takes the commit
    assert f"&{copy}[0]" in kernel_bodies(code)
    assert atomics_into(code, "h") and all("__bk_" in p for p in atomics_into(code, "h"))


@pytest.mark.parametrize("backend", ["hip", "cuda"])
def test_a_scatter_too_large_for_shared_memory_is_a_slice_of_a_global_buffer_per_block(backend):
    with dace.config.set_temporary("compiler", "cuda", "backend", value=backend):
        _, code = kernel_code(scatter_add, f"bps_global_code_{backend}", "global")
    assert "__shared__ double __bpart" not in code
    # The block's own slice of the buffer; the lanes update the array itself only when the grid outgrows the buffer
    slice_ = re.search(rf"double \* const (__bpart_\w+_g) = \w+ \+ \(size_t\)\(.*\) \* {BINS};", code).group(1)
    commits = [p for p in atomics_into(code, "h") if "__bk_" in p]
    fallbacks = [p for p in atomics_into(code, "h") if "__bk_" not in p]
    assert commits and len(fallbacks) <= 1
    taken = rf"\(\({slice_}_on\) \? \((&{slice_}\[0\]|dace::wcr_fixed<[^>]*>::reduce_atomic\(&{slice_}\[)"
    assert re.search(taken, kernel_bodies(code))


def test_a_symbolic_extent_gets_a_shared_copy_and_a_global_slice():
    _, code = kernel_code(scatter_add_symbolic, "bps_symbolic_code")
    shared = re.search(r"__shared__ double (__bpart_\w+)\[2048\];", code).group(1)
    assert f"const bool {shared}_on = (M <= 2048);" in code
    assert re.search(rf"const bool {shared}_g_on = !{shared}_on && \(M <= 8192\) && ", code)


@pytest.mark.parametrize(
    "program",
    [scatter_write, scatter_reading_target, scatter_sub, scatter_huge, scatter_per_iteration],
    ids=["plain_write", "target_read", "non_associative", "huge_extent", "extent_grows_with_the_work"],
)
@pytest.mark.parametrize("path", ["shared", "global"])
def test_a_scatter_that_is_not_a_privatizable_reduction_keeps_its_atomics(program, path):
    _, code = kernel_code(program, f"bps_refuse_{program.name}_{path}", path)
    assert "__bpart" not in code


@pytest.mark.gpu
@pytest.mark.parametrize("path", ["shared", "global"])
def test_a_scatter_add_with_repeated_bins_matches_numpy(path):
    rng = np.random.default_rng(1)
    idx, w = bins(rng, SIZE, BINS), rng.random(SIZE)
    h0 = rng.random(BINS)
    expected = h0.copy()
    np.add.at(expected, idx, w)
    got = run_on_device(scatter_add, f"bps_add_{path}", path, "h", idx=idx, w=w, h=h0)
    np.testing.assert_allclose(got, expected, rtol=1e-12)


@pytest.mark.gpu
@pytest.mark.parametrize("path", ["shared", "global"])
def test_a_scatter_of_rows_matches_numpy(path):
    rng = np.random.default_rng(2)
    idx, w = bins(rng, SIZE, BINS), rng.random((SIZE, 3))
    f0 = rng.random((BINS, 3))
    expected = f0.copy()
    np.add.at(expected, idx, w)
    got = run_on_device(scatter_rows, f"bps_rows_{path}", path, "f", idx=idx, w=w, f=f0)
    np.testing.assert_allclose(got, expected, rtol=1e-12)


@pytest.mark.gpu
@pytest.mark.parametrize("path", ["shared", "global"])
@pytest.mark.parametrize("program, ufunc", [(scatter_min, np.minimum), (scatter_max, np.maximum)], ids=["min", "max"])
def test_a_min_or_max_scatter_matches_numpy(program, ufunc, path):
    rng = np.random.default_rng(3)
    idx, w = bins(rng, SIZE, BINS), rng.standard_normal(SIZE)
    h0 = rng.standard_normal(BINS)
    expected = h0.copy()
    ufunc.at(expected, idx, w)
    got = run_on_device(program, f"bps_{program.name}_{path}", path, "h", idx=idx, w=w, h=h0)
    np.testing.assert_array_equal(got, expected)


@pytest.mark.gpu
def test_a_symbolic_extent_uses_the_copy_it_fits_at_run_time():
    """One build: 100 bins fit shared memory, 3000 the global slice, 20000 neither (the atomics)."""
    cupy = pytest.importorskip("cupy")
    sdfg = canonical_gpu(scatter_add_symbolic, "bps_symbolic")
    compiled = sdfg.compile()
    rng = np.random.default_rng(4)
    for m in (100, 3000, 20000):
        n = 4 * m + 37
        idx, w, h0 = bins(rng, n, m), rng.random(n), rng.random(m)
        expected = h0.copy()
        np.add.at(expected, idx, w)
        h = cupy.asarray(h0)
        compiled(idx=cupy.asarray(idx), w=cupy.asarray(w), h=h, N=n, M=m)
        np.testing.assert_allclose(cupy.asnumpy(h), expected, rtol=1e-12, err_msg=f"M={m}")
