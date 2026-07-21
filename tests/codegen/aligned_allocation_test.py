# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for heap-transient allocation alignment in the generated CPU code.

An alignment attribute on the element type of a ``new`` expression never
affected the allocation and newer GCC rejects it outright for constant array
bounds, so heap arrays are allocated with C++17 aligned ``operator new`` (when
``compiler.cpp_standard`` >= 17) or with no annotation at all (below 17).
"""
import numpy as np
import re

import dace
from dace.config import set_temporary


def _heap_transient_sdfg(name: str) -> dace.SDFG:
    """An SDFG with a constant-size CPU_Heap transient (A -> tmp -> B copy)."""
    sdfg = dace.SDFG(name)
    sdfg.add_array('A', [2], dace.float64)
    sdfg.add_array('B', [2], dace.float64)
    sdfg.add_transient('tmp', [2], dace.float64, storage=dace.StorageType.CPU_Heap)
    state = sdfg.add_state()
    tmp = state.add_access('tmp')
    state.add_edge(state.add_read('A'), None, tmp, None, dace.Memlet('A[0:2]'))
    state.add_edge(tmp, None, state.add_write('B'), None, dace.Memlet('tmp[0:2]'))
    return sdfg


def test_aligned_allocation_property():
    """Checks if the `.alignment` property is honored."""
    new_code = r'new\s*\(std::align_val_t\({alignment}\)\)\s*double\s*\[2\]\s*;'
    del_code = r'::operator\s+delete\[\]\(tmp,\s*std::align_val_t\({alignment}\)\)\s*;'
    for alignment in [-1, 0, 64, 128]:
        name_suffix = str(alignment) if alignment >= 0 else f"m{str(abs(alignment))}"
        sdfg = _heap_transient_sdfg(f"sdfg_allocation_{name_suffix}")
        array_desc = sdfg.arrays["tmp"]
        array_desc.alignment = alignment
        code = sdfg.generate_code()[0].clean_code

        if alignment < 0:
            assert re.search(r'tmp\s+=\s*new\s+double\s*\[2\]\s*;', code)
            assert re.search(r'delete\[\]\s+tmp\s*;', code)

        elif alignment == 0:
            assert re.search(new_code.format(alignment=64), code)
            assert re.search(del_code.format(alignment=64), code)

        else:
            assert re.search(new_code.format(alignment=alignment), code)
            assert re.search(del_code.format(alignment=alignment), code)


def test_heap_allocation_aligned_new_cpp17():
    """With cpp_standard >= 17 (the default), heap arrays use aligned operator new/delete."""
    code = _heap_transient_sdfg('aligned_new_probe').generate_code()[0].clean_code
    assert 'new (std::align_val_t(64)) double[2]' in code
    assert '::operator delete[](tmp, std::align_val_t(64));' in code
    assert 'DACE_ALIGN(64)[' not in code  # the attribute is invalid in a new expression
    assert 'delete[] tmp' not in code  # would pair the unaligned deallocation function
    # The direct operator call skips destructors (and assumes no array cookie);
    # a compile-time guard enforces the trivial-destructibility this relies on.
    assert 'static_assert(std::is_trivially_destructible<double>::value' in code


def test_heap_allocation_plain_new_below_cpp17():
    """Below C++17 there is no aligned operator new; emit no annotation at all."""
    with set_temporary('compiler', 'cpp_standard', value='14'):
        code = _heap_transient_sdfg('plain_new_probe').generate_code()[0].clean_code
    assert 'new double[2]' in code
    assert 'delete[] tmp' in code
    assert 'align_val_t' not in code
    assert 'DACE_ALIGN(64)[' not in code


def test_heap_transient_end_to_end():
    """A constant-size heap transient compiles and runs (hard error on GCC >= 16 before)."""

    @dace.program
    def inner(a: dace.float64[2], b: dace.float64[2]) -> dace.float64[2]:
        mid = (a + b) / 2.0
        return mid + 1.0

    @dace.program
    def aligned_alloc_e2e(p1: dace.float64[2], p2: dace.float64[2]):
        return inner(p1, p2)

    p1 = np.random.rand(2)
    p2 = np.random.rand(2)
    result = aligned_alloc_e2e(p1, p2)
    assert np.allclose(result, (p1 + p2) / 2.0 + 1.0)


if __name__ == '__main__':
    test_heap_allocation_aligned_new_cpp17()
    test_heap_allocation_plain_new_below_cpp17()
    test_heap_transient_end_to_end()
