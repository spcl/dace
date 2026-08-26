# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Type promotion in :func:`dace.dtypes.result_type_of`."""
import functools
import itertools

import numpy as np
import pytest

import dace
from dace import dtypes

SCALARS = [
    dace.bool_, dace.int8, dace.uint8, dace.int16, dace.uint16, dace.int32, dace.uint32, dace.int64, dace.uint64,
    dace.float16, dace.float32, dace.float64, dace.complex64, dace.complex128
]


def test_result_type_of_is_commutative():
    """Callers fold this over unordered containers, so an answer that depends on argument order is
    an answer that depends on iteration order."""
    for lhs, rhs in itertools.permutations(SCALARS, 2):
        assert str(dtypes.result_type_of(lhs, rhs)) == str(dtypes.result_type_of(rhs, lhs)), \
            f'{lhs} and {rhs} promote differently depending on the order'


def test_result_type_of_is_order_independent_when_folded():
    """The commutativity above is not enough on its own: a fold visits the types pairwise, so the
    property that matters to a caller is that every permutation reaches the same type."""
    for triple in itertools.combinations(SCALARS, 3):
        promoted = {str(functools.reduce(dtypes.result_type_of, order)) for order in itertools.permutations(triple)}
        assert len(promoted) == 1, f'{[str(t) for t in triple]} folds to {sorted(promoted)} depending on the order'


@pytest.mark.parametrize('real,complex_', [(dace.float16, dace.complex64), (dace.float32, dace.complex64),
                                           (dace.float64, dace.complex64), (dace.float64, dace.complex128),
                                           (dace.float32, dace.complex128)])
def test_mixing_a_float_and_a_complex_type_keeps_both_parts(real, complex_):
    """A complex type carries its real part in HALF its width, so a byte-width comparison ties
    complex64 with float64. That tie used to be settled by argument order: one way round the result
    was ``double`` and the imaginary part was gone. The result must be complex, and wide enough for
    the real side -- which is what numpy promotes these to."""
    expected = np.promote_types(real.as_numpy_dtype(), complex_.as_numpy_dtype())
    for got in (dtypes.result_type_of(real, complex_), dtypes.result_type_of(complex_, real)):
        assert got.as_numpy_dtype() == expected, f'{real} with {complex_} promoted to {got}, expected {expected}'


def test_float64_with_complex64_is_the_regression():
    """The reported case, spelled out: neither order may answer ``double`` (imaginary part dropped)
    nor ``complex64`` (half the real precision dropped)."""
    assert dtypes.result_type_of(dace.float64, dace.complex64) == dace.complex128
    assert dtypes.result_type_of(dace.complex64, dace.float64) == dace.complex128


if __name__ == '__main__':
    pytest.main([__file__])
