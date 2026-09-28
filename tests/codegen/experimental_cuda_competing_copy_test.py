# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A device-to-device copy sharing its destination with a kernel whose write region may overlap.

``InsertExplicitCopies`` leaves a copy implicit when another edge may write the same region, and
the experimental CUDA codegen then handed that copy to the CPU codegen: a host ``CopyND`` that
dereferences device pointers. Reduced from GT4Py ``concat_where`` over a dynamic domain, where the
two write regions are disjoint but bounded by ``Min``/``Max`` expressions ``intersects`` cannot
decide.

Codegen-only: no GPU and no nvcc required.
"""
import pytest

import dace
from dace import dtypes
from dace.codegen.exceptions import CodegenError
from dace.transformation.passes import insert_explicit_copies


def _kernel_and_copy_into_one_array_sdfg() -> dace.SDFG:
    """``C[0:Min(K, N)]`` written by a kernel, ``C[Max(K, 0):N]`` by a device-to-device copy from ``B``."""
    GPU = dtypes.StorageType.GPU_Global
    sdfg = dace.SDFG('kernel_and_copy_into_one_array')
    for name in 'ABC':
        sdfg.add_array(name, ['N'], dace.float64, storage=GPU)
    sdfg.add_symbol('K', dace.int64)

    state = sdfg.add_state('main')
    c = state.add_write('C')
    entry, exit_node = state.add_map('k', dict(i='0:Min(K, N)'), schedule=dtypes.ScheduleType.GPU_Device)
    tasklet = state.add_tasklet('double', {'a'}, {'o'}, 'o = a * 2.0')
    state.add_memlet_path(state.add_read('A'), entry, tasklet, dst_conn='a', memlet=dace.Memlet('A[i]'))
    state.add_memlet_path(tasklet, exit_node, c, src_conn='o', memlet=dace.Memlet('C[i]'))
    state.add_nedge(state.add_read('B'), c, dace.Memlet('B[Max(K, 0):N] -> [Max(K, 0):N]'))
    sdfg.validate()
    return sdfg


def _generate_code(sdfg: dace.SDFG):
    with dace.config.set_temporary('compiler', 'cuda', 'implementation', value='experimental'):
        return sdfg.generate_code()


def test_a_device_copy_with_an_undecidable_competing_write_is_not_emitted_on_the_host():
    sdfg = _kernel_and_copy_into_one_array_sdfg()
    frame = [code for code in _generate_code(sdfg) if code.title == 'Frame']
    assert len(frame) == 1, 'expected exactly one frame code object'
    code = frame[0].clean_code

    assert 'CopyND' not in code, 'the device-to-device copy was emitted as a host copy'
    assert 'cudaMemcpyAsync' in code, 'the device-to-device copy was not emitted as a GPU copy'


def test_a_device_copy_left_implicit_is_an_error_rather_than_a_host_copy(monkeypatch):
    """Should a device copy still reach the codegen as a plain edge, it has no correct host lowering."""
    monkeypatch.setattr(insert_explicit_copies, '_consumers_to_precede', lambda state, edge: None)
    with pytest.raises(CodegenError, match='involves GPU memory'):
        _generate_code(_kernel_and_copy_into_one_array_sdfg())


if __name__ == '__main__':
    test_a_device_copy_with_an_undecidable_competing_write_is_not_emitted_on_the_host()
