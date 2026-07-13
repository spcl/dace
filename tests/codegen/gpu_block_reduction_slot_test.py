# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Unit tests for :func:`gpu_block_reduction_write_slot`, which decides whether a write to a GPU
thread-block tree-reduction accumulator can fold into the per-thread register partial (returning its
slot) or must keep the plain atomic WCR (returning ``None``)."""
import dace
from dace.subsets import Range
from dace.codegen.targets.cpu import gpu_block_reduction_write_slot


def test_scalar_write_maps_to_slot_zero():
    """A single-element scalar accumulator (base 0, one slot) folds into slot 0."""
    assert gpu_block_reduction_write_slot(Range([(0, 0, 1)]), base=0, length=1) == 0


def test_in_range_subset_element_maps_to_offset():
    """An element inside a length-m subset folds into ``offset - base``."""
    assert gpu_block_reduction_write_slot(Range([(3, 3, 1)]), base=0, length=8) == 3
    assert gpu_block_reduction_write_slot(Range([(5, 5, 1)]), base=2, length=8) == 3


def test_multidimensional_write_keeps_atomic():
    """A multi-dimensional subset cannot be indexed by ranges[0][0]; keep the atomic (return None)."""
    assert gpu_block_reduction_write_slot(Range([(0, 0, 1), (2, 2, 1)]), base=0, length=4) is None


def test_missing_subset_keeps_atomic():
    assert gpu_block_reduction_write_slot(None, base=0, length=4) is None


def test_constant_offset_outside_span_keeps_atomic():
    """A constant offset at or beyond the reduced span (e.g. a second reduction edge over the same
    array) would overflow the register partial; keep the atomic instead."""
    assert gpu_block_reduction_write_slot(Range([(5, 5, 1)]), base=0, length=4) is None
    assert gpu_block_reduction_write_slot(Range([(0, 0, 1)]), base=2, length=4) is None


def test_symbolic_offset_is_trusted():
    """A symbolic (per-element) offset cannot be bounds-checked at compile time; it is a valid tile
    index within the span, so return it rather than falling back to the atomic."""
    k = dace.symbolic.symbol('k')
    slot = gpu_block_reduction_write_slot(Range([(k, k, 1)]), base=0, length=8)
    assert slot is not None
    assert str(slot) == 'k'


if __name__ == '__main__':
    test_scalar_write_maps_to_slot_zero()
    test_in_range_subset_element_maps_to_offset()
    test_multidimensional_write_keeps_atomic()
    test_missing_subset_keeps_atomic()
    test_constant_offset_outside_span_keeps_atomic()
    test_symbolic_offset_is_trusted()
