# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Unit tests for the dace cast-stripping rewrites used before vectorization.

``DaceCastRemover`` strips ``dace.intNN(...)`` / ``dace.floatNN(...)`` casts,
keeping the cast value. It must match only true cast names — a loose
``startswith`` prefix match also caught builtins such as ``int_floor`` /
``int_ceil`` and replaced them with their first argument, silently dropping
the divisor (TSVC s276's ``int_floor(LEN_1D, 2)`` lane condition became
``LEN_1D``).
"""
import pytest

from dace.transformation.passes.vectorization.tasklet_preprocessing_passes import (
    _remove_dace_int_casts,
    _remove_dace_float_casts,
)


@pytest.mark.parametrize(
    "src,expected",
    [
        # True int casts ARE stripped to the cast value.
        ("x = int32(a)", "x = a"),
        ("x = int64(a + 1)", "x = a + 1"),
        ("x = dace.int32(a)", "x = a"),
        # int_floor / int_ceil are NOT casts — they keep all their arguments.
        ("x = int_floor(LEN_1D, 2)", "x = int_floor(LEN_1D, 2)"),
        ("x = int_ceil(n, 4)", "x = int_ceil(n, 4)"),
        ("x = dace.int_floor(n, 2)", "x = dace.int_floor(n, 2)"),
    ])
def test_remove_int_casts_keeps_int_floor(src: str, expected: str):
    assert _remove_dace_int_casts(src).strip() == expected


@pytest.mark.parametrize(
    "src,expected",
    [
        ("x = float32(a)", "x = a"),
        ("x = float64(a * b)", "x = a * b"),
        ("x = dace.float64(a)", "x = a"),
        # A hypothetical ``float...`` builtin keeps its arguments.
        ("x = floor(a)", "x = floor(a)"),
    ])
def test_remove_float_casts_keeps_non_cast_calls(src: str, expected: str):
    assert _remove_dace_float_casts(src).strip() == expected
