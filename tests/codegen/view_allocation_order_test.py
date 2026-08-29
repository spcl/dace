# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A view must be bound only after the container it points into has been allocated."""
import re

import numpy as np

import dace

N = dace.symbol('N', dtype=dace.int64)
levels = dace.symbol('levels', dtype=dace.int64)


@dace.program
def strided_view_of_loop_local(image: dace.float64[N, N], out: dace.float64[N, N]):
    """A Daubechies-4 DWT: the per-level transients are sized by the loop variable, and the strided
    slices taken of them are views bound into those allocations."""
    out[:] = image
    r = np.sqrt(3.0)
    d = 4.0 * np.sqrt(2.0)
    h0, h1 = (1.0 + r) / d, (3.0 + r) / d
    h2, h3 = (3.0 - r) / d, (1.0 - r) / d
    for level in range(levels):
        size = N >> level
        half = size // 2
        block = out[:size, :size]
        even, odd = block[:, 0:2 * half:2], block[:, 1:2 * half:2]
        even1, odd1 = np.roll(even, -1, axis=1), np.roll(odd, -1, axis=1)
        band_lo = h0 * even + h1 * odd + h2 * even1 + h3 * odd1
        band_hi = h3 * even + -h2 * odd + h1 * even1 + -h0 * odd1
        # The rebind is what the emitter produces, and what makes lo/hi containers to view.
        lo = band_lo
        hi = band_hi
        lo_even, lo_odd = lo[0:2 * half:2, :], lo[1:2 * half:2, :]
        lo_even1, lo_odd1 = np.roll(lo_even, -1, axis=0), np.roll(lo_odd, -1, axis=0)
        out[:half, :half] = h0 * lo_even + h1 * lo_odd + h2 * lo_even1 + h3 * lo_odd1
        out[half:2 * half, :half] = h3 * lo_even + -h2 * lo_odd + h1 * lo_even1 + -h0 * lo_odd1
        hi_even, hi_odd = hi[0:2 * half:2, :], hi[1:2 * half:2, :]
        hi_even1, hi_odd1 = np.roll(hi_even, -1, axis=0), np.roll(hi_odd, -1, axis=0)
        out[:half, half:2 * half] = h0 * hi_even + h1 * hi_odd + h2 * hi_even1 + h3 * hi_odd1
        out[half:2 * half, half:2 * half] = h3 * hi_even + -h2 * hi_odd + h1 * hi_even1 + -h0 * hi_odd1


def binding_precedes_nothing(code: str) -> list:
    """Every ``view = &base[...]`` that the generated code emits before allocating ``base``."""
    allocated_at = {}
    for index, line in enumerate(code.splitlines()):
        allocation = re.search(r'^\s*(\w+) = new \(std::align_val_t', line)
        if allocation is not None:
            allocated_at.setdefault(allocation.group(1), index)

    early = []
    for index, line in enumerate(code.splitlines()):
        binding = re.search(r'^\s*(\w+) = &(\w+)\[', line)
        if binding is None:
            continue
        base = binding.group(2)
        if base in allocated_at and allocated_at[base] > index:
            early.append((binding.group(1), base, index, allocated_at[base]))
    return early


def test_view_is_bound_after_its_source_is_allocated():
    sdfg = strided_view_of_loop_local.to_sdfg(simplify=True)
    code = '\n'.join(c.clean_code for c in sdfg.generate_code())

    early = binding_precedes_nothing(code)
    assert not early, ('a view is bound before the buffer it points into exists, so it reads a null '
                       f'or freed pointer: {early}')


def test_strided_view_of_loop_local_transient_is_value_exact():
    """The wild read this ordering used to produce was a SIGSEGV, not a wrong number."""
    extent, depth = 32, 2
    rng = np.random.default_rng(0)
    image = rng.random((extent, extent))
    out = np.zeros((extent, extent))

    expected = image.copy()
    r = np.sqrt(3.0)
    d = 4.0 * np.sqrt(2.0)
    h0, h1, h2, h3 = (1.0 + r) / d, (3.0 + r) / d, (3.0 - r) / d, (1.0 - r) / d
    for level in range(depth):
        size = extent >> level
        half = size // 2
        block = expected[:size, :size]
        even, odd = block[:, 0:2 * half:2], block[:, 1:2 * half:2]
        even1, odd1 = np.roll(even, -1, axis=1), np.roll(odd, -1, axis=1)
        lo = h0 * even + h1 * odd + h2 * even1 + h3 * odd1
        hi = h3 * even + -h2 * odd + h1 * even1 + -h0 * odd1
        for source, column in ((lo, 0), (hi, half)):
            e, o = source[0:2 * half:2, :], source[1:2 * half:2, :]
            e1, o1 = np.roll(e, -1, axis=0), np.roll(o, -1, axis=0)
            expected[:half, column:column + half] = h0 * e + h1 * o + h2 * e1 + h3 * o1
            expected[half:2 * half, column:column + half] = h3 * e + -h2 * o + h1 * e1 + -h0 * o1

    strided_view_of_loop_local(image=image.copy(), out=out, N=extent, levels=depth)
    assert np.allclose(out, expected)


if __name__ == '__main__':
    test_view_is_bound_after_its_source_is_allocated()
    test_strided_view_of_loop_local_transient_is_value_exact()
