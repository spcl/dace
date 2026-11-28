# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A runtime ``assert`` over a program's symbols (``assert H % 16 == 0``, ``assert K <= N``) is a symbol relation:
the extent proofs use it, and the call checks it. A U-Net pools ``H`` by 16 and upsamples by 2 to meet ``H // 8``,
which is the same extent only when 16 divides ``H``."""

import warnings

import numpy as np
import pytest

import dace
from dace import symbolic

H = dace.symbol("H")
K = dace.symbol("K")
N = dace.symbol("N")


@dace.program
def upsampled_pool(x: dace.float64[H], y: dace.float64[H // 8]):
    assert H % 16 == 0
    y[:] = np.repeat(x[: H // 16], 2)


@dace.program
def upsampled_pool_unasserted(x: dace.float64[H], y: dace.float64[H // 8]):
    y[:] = np.repeat(x[: H // 16], 2)


@dace.program
def bounded_prefix(x: dace.float64[N], y: dace.float64[K]):
    assert K <= N and 0 < K
    y[:] = x[:K]


@dace.program
def asserted_on_data(x: dace.float64[N]):
    assert x[0] > 0
    x[:] = 1.0


def relations(sdfg: dace.SDFG) -> set[symbolic.Relation]:
    return set(sdfg.facts().relations)


def test_a_divisibility_assert_proves_the_extent():
    sdfg = upsampled_pool.to_sdfg(simplify=False)
    assert symbolic.Relation(symbolic.RelationKind.DIVIDES, 16, H) in relations(sdfg)
    with pytest.raises(IndexError):
        upsampled_pool_unasserted.to_sdfg(simplify=False)


def test_a_divisibility_assert_computes_what_numpy_computes():
    x = np.arange(32.0)
    y = np.zeros(4)
    upsampled_pool(x=x, y=y, H=32)
    assert np.array_equal(y, np.repeat(x[:2], 2)), y


def test_comparisons_joined_by_and_are_each_a_relation():
    facts = bounded_prefix.to_sdfg(simplify=False).facts()
    assert symbolic.holds(symbolic.Relation(symbolic.RelationKind.LE, K, N), facts)
    assert symbolic.holds(symbolic.Relation(symbolic.RelationKind.LT, 0, K), facts)


def test_an_assert_on_data_is_still_skipped():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        asserted_on_data.to_sdfg(simplify=False)
    assert any("Runtime assertion" in str(w.message) for w in caught)


if __name__ == "__main__":
    test_a_divisibility_assert_proves_the_extent()
    test_a_divisibility_assert_computes_what_numpy_computes()
    test_comparisons_joined_by_and_are_each_a_relation()
    test_an_assert_on_data_is_still_skipped()
