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


@pytest.mark.gpu
def test_python_floor_division_agrees_between_the_host_and_the_device():
    """Signed operands take the correction branch, which is where the host-only call sat.

    The comparison is against the CPU lowering of the SAME program, not against numpy: this tree
    lowers ``//`` to a truncating division on both targets, and that divergence is its own
    question. What has to hold here is that the two targets compute the same thing.
    """
    numerator = np.arange(-32, 32, dtype=np.int64)

    host = np.zeros(64, dtype=np.int64)
    python_floor_division_on_device.to_sdfg()(numerator=numerator, out=host, N=64)

    sdfg = python_floor_division_on_device.to_sdfg()
    sdfg.apply_gpu_transformations()
    device = np.zeros(64, dtype=np.int64)
    sdfg(numerator=numerator, out=device, N=64)

    assert np.array_equal(device, host)
    assert np.any(device != 0), 'an empty kernel also leaves the output at zero'


if __name__ == '__main__':
    test_python_modulo_stores_on_the_device()
    test_python_floor_division_agrees_between_the_host_and_the_device()
