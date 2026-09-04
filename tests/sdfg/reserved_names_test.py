# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``SDFG.add_datadesc`` rejects user additions of ``SDFG.RESERVED_NAMES`` (e.g. ``gpu_streams``),
while ``_internal_use=True`` bypasses the guard for the pipeline itself."""
import pytest

import dace


def test_user_add_array_with_reserved_name_raises():
    """``SDFG.add_array`` with a reserved name raises ``NameError``."""
    sdfg = dace.SDFG('reserved_user')
    with pytest.raises(NameError, match='reserved'):
        sdfg.add_array('gpu_streams', [4], dace.int64)


def test_user_add_datadesc_with_reserved_name_raises():
    """``SDFG.add_datadesc`` with a reserved name raises ``NameError``."""
    sdfg = dace.SDFG('reserved_datadesc')
    desc = dace.data.Array(dtype=dace.int64, shape=(4, ))
    with pytest.raises(NameError, match='reserved'):
        sdfg.add_datadesc('gpu_streams', desc)


def test_internal_use_bypasses_reservation():
    """``add_datadesc`` with ``_internal_use=True`` accepts a reserved name."""
    sdfg = dace.SDFG('reserved_internal')
    desc = dace.data.Array(dtype=dace.dtypes.gpuStream_t, shape=(4, ))
    name = sdfg.add_datadesc('gpu_streams', desc, _internal_use=True)
    assert name == 'gpu_streams'
    assert 'gpu_streams' in sdfg.arrays


if __name__ == '__main__':
    test_user_add_array_with_reserved_name_raises()
    test_user_add_datadesc_with_reserved_name_raises()
    test_internal_use_bypasses_reservation()
