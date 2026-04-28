# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Reservation enforcement for framework-owned descriptor names.

The gpu_specialization pipeline takes ownership of ``gpu_streams`` and
uses its presence as the canonical "lowering applied" signal.
``SDFG.add_datadesc`` rejects user-driven additions of names in
``SDFG.RESERVED_NAMES``; the pipeline itself bypasses the guard via
``add_datadesc(..., _internal_use=True)``.
"""
import pytest

import dace


def test_user_add_array_with_reserved_name_raises():
    sdfg = dace.SDFG('reserved_user')
    with pytest.raises(NameError, match='reserved'):
        sdfg.add_array('gpu_streams', [4], dace.int64)


def test_user_add_datadesc_with_reserved_name_raises():
    sdfg = dace.SDFG('reserved_datadesc')
    desc = dace.data.Array(dtype=dace.int64, shape=(4, ))
    with pytest.raises(NameError, match='reserved'):
        sdfg.add_datadesc('gpu_streams', desc)


def test_internal_use_bypasses_reservation():
    sdfg = dace.SDFG('reserved_internal')
    desc = dace.data.Array(dtype=dace.dtypes.gpuStream_t, shape=(4, ))
    name = sdfg.add_datadesc('gpu_streams', desc, _internal_use=True)
    assert name == 'gpu_streams'
    assert 'gpu_streams' in sdfg.arrays


if __name__ == '__main__':
    test_user_add_array_with_reserved_name_raises()
    test_user_add_datadesc_with_reserved_name_raises()
    test_internal_use_bypasses_reservation()
