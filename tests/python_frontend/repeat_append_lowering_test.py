# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``np.repeat`` and ``np.append`` lower to dataflow, not to a Python callback.

Neither had a replacement, so the frontend fell back to the pyobject callback path: the SDFG
carried a ``__pystate`` array and a tasklet reading ``numpy_repeat(...)``. The numbers came out
right and everything else about the kernel was lost -- the call re-enters CPython, so it holds the
GIL, cannot be scheduled on a GPU, and blocks every downstream map fusion.

``repeat`` is a ``Broadcast`` (Fortran ``SPREAD``): insert an axis of length ``repeats`` after the
repeated one and merge the two back. ``append`` is ``concatenate``, which already lowers natively.
The assertions are structural first -- a numeric check alone passes just as well through a
callback, which is exactly how this went unnoticed.
"""
import numpy as np
import pytest

import dace
from dace.sdfg import nodes as nd

N = dace.symbol('N', dtype=dace.int64)


def callback_tasklets(sdfg: dace.SDFG) -> list:
    """Tasklets whose code calls back into the Python interpreter."""
    return [
        n.label for _, state in enumerate(sdfg.states()) for n in state.nodes()
        if isinstance(n, nd.Tasklet) and 'numpy_' in n.code.as_string
    ]


def libnode_names(sdfg: dace.SDFG) -> list:
    return sorted({type(n).__name__ for state in sdfg.states() for n in state.nodes() if isinstance(n, nd.LibraryNode)})


def test_repeat_lowers_to_a_broadcast_library_node():

    @dace.program
    def prog(a: dace.float64[N], out: dace.float64[3 * N]):
        out[:] = np.repeat(a, 3)

    sdfg = prog.to_sdfg(simplify=False)
    assert callback_tasklets(sdfg) == []
    assert '__pystate' not in sdfg.arrays
    assert 'Broadcast' in libnode_names(sdfg)


def test_repeat_matches_numpy():

    @dace.program
    def prog(a: dace.float64[N], out: dace.float64[3 * N]):
        out[:] = np.repeat(a, 3)

    a = np.random.rand(7)
    out = np.zeros(21)
    prog(a=a, out=out, N=7)
    assert np.allclose(out, np.repeat(a, 3))


def test_repeat_along_an_axis_matches_numpy():

    @dace.program
    def prog(a: dace.float64[N, 4], out: dace.float64[N, 8]):
        out[:] = np.repeat(a, 2, axis=1)

    a = np.random.rand(5, 4)
    out = np.zeros((5, 8))
    prog(a=a, out=out, N=5)
    assert np.allclose(out, np.repeat(a, 2, axis=1))


def test_repeat_on_the_leading_axis_matches_numpy():

    @dace.program
    def prog(a: dace.float64[N, 4], out: dace.float64[2 * N, 4]):
        out[:] = np.repeat(a, 2, axis=0)

    a = np.random.rand(5, 4)
    out = np.zeros((10, 4))
    prog(a=a, out=out, N=5)
    assert np.allclose(out, np.repeat(a, 2, axis=0))


def test_append_lowers_without_a_callback():

    @dace.program
    def prog(a: dace.float64[N], b: dace.float64[N], out: dace.float64[2 * N]):
        out[:] = np.append(a, b)

    sdfg = prog.to_sdfg(simplify=False)
    assert callback_tasklets(sdfg) == []
    assert '__pystate' not in sdfg.arrays


def test_append_matches_numpy():

    @dace.program
    def prog(a: dace.float64[N], b: dace.float64[N], out: dace.float64[2 * N]):
        out[:] = np.append(a, b)

    a, b = np.random.rand(6), np.random.rand(6)
    out = np.zeros(12)
    prog(a=a, b=b, out=out, N=6)
    assert np.allclose(out, np.append(a, b))


def test_append_along_an_axis_matches_numpy():

    @dace.program
    def prog(a: dace.float64[N, 3], b: dace.float64[N, 3], out: dace.float64[2 * N, 3]):
        out[:] = np.append(a, b, axis=0)

    a, b = np.random.rand(4, 3), np.random.rand(4, 3)
    out = np.zeros((8, 3))
    prog(a=a, b=b, out=out, N=4)
    assert np.allclose(out, np.append(a, b, axis=0))


def test_repeat_refuses_a_per_element_repeat_count():
    """A repeats ARRAY makes the output extent data-dependent; refuse it by name."""

    @dace.program
    def prog(a: dace.float64[N], r: dace.int64[N], out: dace.float64[2 * N]):
        out[:] = np.repeat(a, r)

    with pytest.raises(Exception, match='repeat'):
        prog.to_sdfg(simplify=False)


if __name__ == '__main__':
    test_repeat_lowers_to_a_broadcast_library_node()
    test_repeat_matches_numpy()
    test_repeat_along_an_axis_matches_numpy()
    test_repeat_on_the_leading_axis_matches_numpy()
    test_append_lowers_without_a_callback()
    test_append_matches_numpy()
    test_append_along_an_axis_matches_numpy()
    test_repeat_refuses_a_per_element_repeat_count()
