# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Closures, guards, and structured arguments of custom SDFG-convertible objects: arrays that a convertible only reports
after conversion, guards that invalidate a program's cache, and list/dict arguments.
"""

import numpy as np

import dace
from dace.frontend.python.common import SDFGConvertible

N = dace.symbol("N")
M = dace.symbol("M")

#: Global array that a convertible reads without the program referring to it
HIDDEN = np.full((5,), 3.0)


class _StructuredSum(SDFGConvertible):
    """Computes ``xs[0] + xs[1] * d['w']``; receives a list and a dict of arrays."""

    def __init__(self):
        self.received = None

    def __sdfg__(self, xs, d):
        self.received = (xs, d)

        @dace.program
        def structured_sum(xs_0: dace.float64[N], xs_1: dace.float64[N], d_w: dace.float64[N]):
            return xs_0 + xs_1 * d_w

        return structured_sum.to_sdfg()

    def __sdfg_signature__(self):
        return ["xs", "d"], []

    def __sdfg_closure__(self, reevaluate=None):
        return {}


class _HiddenGlobal(SDFGConvertible):
    """Adds the sum of the module-level array ``HIDDEN``, which it only reports as its closure once converted."""

    def __init__(self):
        self.conversions = 0

    def __sdfg__(self, x):
        self.conversions += 1
        dtype = dace.dtypes.dtype_to_typeclass(HIDDEN.dtype.type)

        @dace.program
        def add_hidden(x: dace.float64[N], hidden: dtype[M]):
            return x + np.sum(hidden)

        return add_hidden.to_sdfg()

    def __sdfg_signature__(self):
        return ["x"], []

    def __sdfg_closure__(self, reevaluate=None):
        return {"hidden": HIDDEN}


class _Guarded(SDFGConvertible):
    """Multiplies by an attribute that is baked into the SDFG as a constant and reported as a guard."""

    def __init__(self, factor: float):
        self.factor = factor
        self.conversions = 0

    def __sdfg__(self, x):
        self.conversions += 1
        factor = self.factor

        @dace.program
        def scale(x: dace.float64[N]):
            return x * factor

        return scale.to_sdfg()

    def __sdfg_signature__(self):
        return ["x"], []

    def __sdfg_closure__(self, reevaluate=None):
        return {}

    def __sdfg_guards__(self):
        return {"__test_guarded_factor": lambda: self.factor}


def test_structured_arguments_are_descriptors():
    """Lists and dicts of arrays reach ``__sdfg__`` as data descriptors, and their elements bind to the SDFG."""
    summer = _StructuredSum()

    @dace.program
    def program(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N]):
        return summer([a, b], {"w": c})

    a, b, c = np.random.rand(7), np.random.rand(7), np.random.rand(7)
    result = program(a, b, c)
    xs, d = summer.received
    assert all(isinstance(x, dace.data.Array) for x in xs)
    assert list(d.keys()) == ["w"] and isinstance(d["w"], dace.data.Array)
    assert np.allclose(result, a + b * c)


WEIGHTS = np.random.rand(6)


def test_structured_arguments_with_global_array():
    """A global array inside a dict argument is resolved as a closure array of the program."""
    summer = _StructuredSum()

    @dace.program
    def program(a, b):
        return summer((a, b), {"w": WEIGHTS})

    a, b = np.random.rand(6), np.random.rand(6)
    assert np.allclose(program(a, b), a + b * WEIGHTS)


def test_closure_array_reported_after_conversion():
    """
    A closure array that a convertible only reports after ``__sdfg__`` is passed on every call and re-evaluated (so
    reassigning it is visible), and its descriptor is part of the cache key (a new shape or dtype parses again).
    """
    global HIDDEN
    adder = _HiddenGlobal()

    @dace.program
    def program(x: dace.float64[N]):
        return adder(x)

    original = HIDDEN
    try:
        x = np.random.rand(5)
        assert np.allclose(program(x), x + 15.0)
        HIDDEN = np.full((5,), -1.0)  # Same descriptor: re-evaluated, not parsed again
        assert np.allclose(program(x), x - 5.0)
        assert adder.conversions == 1
        HIDDEN = np.full((3,), 2.0)  # New shape
        assert np.allclose(program(x), x + 6.0)
        assert adder.conversions == 2
        HIDDEN = np.full((4,), 0.5, dtype=np.float32)  # New dtype
        assert np.allclose(program(x), x + 2.0)
        assert adder.conversions == 3
    finally:
        HIDDEN = original


def test_guards_invalidate_program_cache():
    """A change in a value reported by ``__sdfg_guards__`` parses the program again; an unchanged one does not."""
    guarded = _Guarded(2.0)

    @dace.program
    def program(x: dace.float64[N]):
        return guarded(x)

    x = np.random.rand(4)
    assert np.allclose(program(x), x * 2.0)
    assert np.allclose(program(x), x * 2.0)
    assert guarded.conversions == 1
    guarded.factor = 5.0
    assert np.allclose(program(x), x * 5.0)
    assert guarded.conversions == 2


if __name__ == "__main__":
    test_structured_arguments_are_descriptors()
    test_structured_arguments_with_global_array()
    test_closure_array_reported_after_conversion()
    test_guards_invalidate_program_cache()
