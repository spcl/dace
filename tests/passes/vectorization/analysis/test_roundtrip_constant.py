# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The vectorized-tasklet constant must be emitted VERBATIM.

``_roundtrip_constant`` feeds the ``{constant}`` slot of a C++ template, so the
exact source literal has to survive: a ``float()`` round-trip would turn
``"Infinity"`` into ``"inf"`` (invalid C++), drop sympy ``"oo"``, re-precision a
clean ``"0.1"``, and rewrite ``"2"`` to ``"2.0"``. ``_is_number`` must still
recognise every numeric form (including infinity) so a constant is never
mistaken for a symbol.
"""
import pytest

from dace.transformation.passes.vectorization.utils.tasklets import (
    _is_number,
    _roundtrip_constant,
)


@pytest.mark.parametrize(
    "s", ["0.1", "2", "2.0", "5.5", "-1e30", "1e-5", "Infinity", "-Infinity", "inf", "oo", "-oo", "~0", "0", None])
def test_roundtrip_constant_is_verbatim(s):
    """Every input is returned exactly as given (no float mangling)."""
    assert _roundtrip_constant(s) is s


@pytest.mark.parametrize("s,expected", [
    ("0.1", True),
    ("2", True),
    ("-1e30", True),
    ("inf", True),
    ("Infinity", True),
    ("-inf", True),
    ("oo", True),
    ("-oo", True),
    ("+oo", True),
    (2, True),
    (2.5, True),
    ("x", False),
    ("a1", False),
    ("idx_index", False),
    (None, False),
])
def test_is_number_recognises_numeric_and_infinity(s, expected):
    assert _is_number(s) is expected
