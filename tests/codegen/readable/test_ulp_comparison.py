# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The CPU comparison against the legacy generator accepts a difference of 1 ULP per element and nothing more."""
import numpy as np
import pytest

from tests.codegen.readable.conftest import assert_max_one_ulp, assert_outputs_equivalent


def test_one_ulp_is_accepted():
    value = np.array([1.0, -3.5, 1e-300])
    assert_max_one_ulp(value, np.nextafter(value, np.inf))


def test_two_ulp_is_rejected():
    value = np.array([1.0])
    with pytest.raises(AssertionError):
        assert_max_one_ulp(value, np.nextafter(np.nextafter(value, np.inf), np.inf))


def test_complex_parts_are_compared_separately():
    value = np.array([1.0 + 2.0j])
    assert_max_one_ulp(value, np.array([np.nextafter(1.0, 2.0) + np.nextafter(2.0, 3.0) * 1j]))
    with pytest.raises(AssertionError):
        assert_max_one_ulp(value, np.array([1.0 + 2.1j]))


def test_nan_and_inf_positions_must_match():
    value = np.array([1.0, np.nan, np.inf, -np.inf])
    assert_max_one_ulp(value, value.copy())
    for other in ([1.0, 0.0, np.inf, -np.inf], [1.0, np.nan, np.nan, -np.inf], [1.0, np.nan, np.inf, np.inf]):
        with pytest.raises(AssertionError):
            assert_max_one_ulp(value, np.array(other))


def test_integers_stay_exact():
    outputs = {"out": np.arange(4)}
    assert_outputs_equivalent(outputs, {"out": np.arange(4)}, "cpu")
    with pytest.raises(AssertionError):
        assert_outputs_equivalent(outputs, {"out": np.arange(4) + 1}, "cpu")


if __name__ == "__main__":
    test_one_ulp_is_accepted()
    test_two_ulp_is_rejected()
    test_complex_parts_are_compared_separately()
    test_nan_and_inf_positions_must_match()
    test_integers_stay_exact()
