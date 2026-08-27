# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Python floor division and modulo compute on the DEVICE what they compute on the host.

The failure this pins is silent rather than loud. ``int_floor_ni`` once reached for ``std::div``,
which is host-only; nvcc answers a host call from device code with warning #20011 instead of an
error, and then deletes the guarded region that holds the call. The kernel compiles, launches, and
returns success having stored NOTHING, so the program reads back whatever the buffer already held
-- tsvc ``s315``, whose ``a[i] = (7 * i) % LEN`` left the array at its input values and whose argmax
was then taken over the wrong data.

Numeric assertions, because the number is where the deletion shows: a codegen-text check would pass
on the broken build, which emitted the store and lost it in the device compiler.
"""
import numpy as np
import pytest

import dace

N = dace.symbol('N', dtype=dace.int64)


@dace.program
def python_modulo_on_device(out: dace.int64[N]):
    for i in dace.map[0:N]:
        out[i] = (7 * i) % N


@dace.program
def python_floor_division_on_device(numerator: dace.int64[N], out: dace.int64[N]):
    for i in dace.map[0:N]:
        out[i] = numerator[i] // 7


@pytest.mark.gpu
def test_python_modulo_stores_on_the_device():
    sdfg = python_modulo_on_device.to_sdfg()
    sdfg.apply_gpu_transformations()
    out = np.zeros(64, dtype=np.int64)
    sdfg(out=out, N=64)
    assert np.array_equal(out, (7 * np.arange(64)) % 64)


@pytest.mark.parametrize('target', [pytest.param('host'), pytest.param('device', marks=pytest.mark.gpu)])
def test_python_floor_division_rounds_towards_negative_infinity(target):
    """``//`` is Python's floor division, and a negative numerator is where that is visible.

    ``-32 // 7`` is ``-5``: the quotient rounds toward negative infinity, so the remainder takes
    the divisor's sign. C's ``/`` rounds toward zero and answers ``-4`` -- the value the emitter
    produced while it wrote ``ifloor(a / b)``, where the integer division had already truncated and
    flooring an integer changed nothing.

    Both targets, one assertion: the correction branch is also the branch that held the host-only
    ``std::div`` call, which nvcc answered by deleting the region around it.
    """
    numerator = np.arange(-32, 32, dtype=np.int64)
    sdfg = python_floor_division_on_device.to_sdfg()
    if target == 'device':
        sdfg.apply_gpu_transformations()
    out = np.zeros(64, dtype=np.int64)
    sdfg(numerator=numerator, out=out, N=64)
    assert np.array_equal(out, numerator // 7)


if __name__ == '__main__':
    test_python_modulo_stores_on_the_device()
    test_python_floor_division_rounds_towards_negative_infinity('host')
    test_python_floor_division_rounds_towards_negative_infinity('device')
