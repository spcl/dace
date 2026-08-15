# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" A symbolically-sized Register array has no fixed-size C++ declaration, so it either
    becomes a VLA or falls back to the heap, selected by ``compiler.vla_symbolic_registers``.
    Both paths must compute the same result; only the heap path warns.
"""
import dace
import numpy as np
import pytest
import warnings

from dace.config import set_temporary

N = dace.symbol('N')


def build(name: str) -> dace.SDFG:
    sdfg = dace.SDFG(name)
    sdfg.add_array('A', [N], dace.float32)
    sdfg.add_transient('tmp', [N], dace.float32, storage=dace.StorageType.Register)
    sdfg.add_array('B', [N], dace.float32)
    state = sdfg.add_state()
    A = state.add_read('A')
    tmp = state.add_access('tmp')
    B = state.add_write('B')
    state.add_nedge(A, tmp, dace.Memlet.simple('A', '0:N'))
    state.add_nedge(tmp, B, dace.Memlet.simple('tmp', '0:N'))
    return sdfg


@pytest.mark.parametrize('vla', [False, True])
def test_symbolic_register_array(vla: bool) -> None:
    sdfg = build(f'vla_test_{"vla" if vla else "heap"}')
    A = np.random.rand(12).astype(np.float32)
    B = np.random.rand(12).astype(np.float32)

    with set_temporary('compiler', 'vla_symbolic_registers', value=vla):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            sdfg(A=A, B=B, N=12)
            spilled = any('Variable-length' in str(warn.message) for warn in w)

    assert spilled is not vla
    assert np.linalg.norm(A - B) < 1e-5


if __name__ == "__main__":
    test_symbolic_register_array(False)
    test_symbolic_register_array(True)
