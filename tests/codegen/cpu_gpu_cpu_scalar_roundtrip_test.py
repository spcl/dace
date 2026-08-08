# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Asserts a CPU scalar -> GPU scalar transient -> CPU array round-trip runs and preserves the value."""
import numpy as np
import pytest

import dace
from dace import StorageType


@pytest.mark.gpu
def test_cpu_gpu_cpu_scalar_roundtrip():
    """A scalar copied host -> GPU transient -> host array yields the original value at ``output[0]``."""
    sdfg = dace.SDFG('h2d_d2h_scalar')
    sdfg.add_scalar('scal_in', dace.float32)
    sdfg.add_scalar('gpu_scal', dace.float32, StorageType.GPU_Global, transient=True)
    sdfg.add_array('output', [1], dace.float32)

    state = sdfg.add_state()
    state.add_nedge(state.add_read('scal_in'), state.add_access('gpu_scal'), dace.Memlet('scal_in'))
    state.add_nedge(state.add_access('gpu_scal'), state.add_write('output'), dace.Memlet('gpu_scal'))

    out = np.zeros(1, dtype=np.float32)
    sdfg(scal_in=np.float32(2), output=out)
    assert out[0] == 2.0
