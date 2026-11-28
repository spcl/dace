# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Stores whose target or value shape NumPy reads differently from a plain subscript.

``out[:, :, :H][k] = h`` writes through the view ``out[:, :, :H]`` (a bidirectional RNN's half of
the output), and ``cat[:, 0:1, :] = cls`` stores a ``(1, 1, D)`` value into a ``(B, 1, D)`` slot,
since NumPy strips a stored value's leading size-1 axes (a vision transformer's class token).
"""

import numpy as np

import dace

S, B, H, D = (dace.symbol(name) for name in "SBHD")


@dace.program
def store_through_sliced_subscript(out: dace.float64[S, B, 2 * H], hn: dace.float64[2, B, H]):
    for k in range(S):
        out[:, :, :H][k] = hn[0]
        out[:, :, H:][S - k - 1] = hn[1] + k


def test_a_store_through_a_sliced_subscript_writes_only_the_slice():
    """The chained target used to resolve to ``out[k]`` whole, sized ``2 * H`` against the value's ``H``."""
    s, b, h = 3, 2, 2
    hn = np.arange(2.0 * b * h).reshape(2, b, h).copy()
    out = np.full((s, b, 2 * h), -1.0)
    store_through_sliced_subscript(out=out, hn=hn, S=s, B=b, H=h)
    expected = np.full((s, b, 2 * h), -1.0)
    for k in range(s):
        expected[:, :, :h][k] = hn[0]
        expected[:, :, h:][s - k - 1] = hn[1] + k
    assert np.array_equal(out, expected), out


@dace.program
def store_through_plain_indices(a: dace.float64[S, B], v: dace.float64):
    for k in range(S):
        a[k][1] = v


def test_a_store_through_plain_indices_writes_the_element():
    s, b = 3, 4
    a = np.zeros((s, b))
    store_through_plain_indices(a=a, v=2.5, S=s, B=b)
    expected = np.zeros((s, b))
    expected[:, 1] = 2.5
    assert np.array_equal(a, expected), a


@dace.program
def store_with_leading_unit_axes(cat: dace.float64[B, 3, D], cls: dace.float64[1, 1, D]):
    cat[:, 0:1, :] = cls


def test_a_stored_value_loses_its_leading_unit_axes():
    """``(1, 1, D)`` into ``(B, 1, D)``: the store squeezed the target to ``(B, D)`` and refused the value's rank."""
    b, d = 2, 4
    cat = np.zeros((b, 3, d))
    cls = np.arange(1.0, d + 1).reshape(1, 1, d).copy()
    store_with_leading_unit_axes(cat=cat, cls=cls, B=b, D=d)
    expected = np.zeros((b, 3, d))
    expected[:, 0:1, :] = cls
    assert np.array_equal(cat, expected), cat


if __name__ == "__main__":
    test_a_store_through_a_sliced_subscript_writes_only_the_slice()
    test_a_store_through_plain_indices_writes_the_element()
    test_a_stored_value_loses_its_leading_unit_axes()
