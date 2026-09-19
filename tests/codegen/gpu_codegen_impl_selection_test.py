# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests that the ``compiler.cuda.implementation`` config selects the active GPU
code generator at build time.

Both ``CUDACodeGen`` (legacy) and ``ExperimentalCUDACodeGen`` register under
distinct names, and code generation instantiates only the configured one. The
selection is read per ``generate_code`` call, so flipping the config switches the
active codegen within the same process (only code generation is exercised, so no
GPU is required).
"""
import dace
from dace.codegen.target import TargetCodeGenerator
from dace.codegen.targets.cuda import CUDACodeGen
from dace.codegen.targets.experimental_cuda import ExperimentalCUDACodeGen


def _build_gpu_sdfg():
    """Build a small SDFG with a single ``GPU_Device``-scheduled map."""
    sdfg = dace.SDFG('gpu_codegen_impl_selection')
    sdfg.add_array('A', (16, ), dace.float64, storage=dace.StorageType.GPU_Global)
    sdfg.add_array('B', (16, ), dace.float64, storage=dace.StorageType.GPU_Global)
    state = sdfg.add_state()
    rd = state.add_read('A')
    wr = state.add_write('B')
    me, mx = state.add_map('m', dict(i='0:16'), schedule=dace.ScheduleType.GPU_Device)
    tasklet = state.add_tasklet('double', {'inp'}, {'out'}, 'out = inp * 2.0')
    state.add_memlet_path(rd, me, tasklet, dst_conn='inp', memlet=dace.Memlet('A[i]'))
    state.add_memlet_path(tasklet, mx, wr, src_conn='out', memlet=dace.Memlet('B[i]'))
    sdfg.validate()
    return sdfg


def _gpu_codegen_classes(sdfg):
    """Return the set of GPU TargetCodeGenerator classes that emitted code."""
    return {
        code_object.target
        for code_object in sdfg.generate_code() if code_object.target.target_name in ('cuda', 'experimental_cuda')
    }


def test_both_gpu_codegens_are_registered():
    """Both CUDA code generators are registered simultaneously."""
    registered = {v['name'] for v in TargetCodeGenerator.extensions().values()}
    assert 'cuda' in registered
    assert 'experimental_cuda' in registered


def test_config_selects_active_gpu_codegen_at_runtime():
    """The configured implementation drives which GPU codegen is triggered, and
    the choice tracks the config when it is changed within a single process."""
    # Legacy selected -> only the legacy codegen is triggered.
    with dace.config.set_temporary('compiler', 'cuda', 'implementation', value='legacy'):
        used = _gpu_codegen_classes(_build_gpu_sdfg())
    assert used == {CUDACodeGen}

    # Switch to experimental -> only the experimental codegen is triggered.
    with dace.config.set_temporary('compiler', 'cuda', 'implementation', value='experimental'):
        used = _gpu_codegen_classes(_build_gpu_sdfg())
    assert used == {ExperimentalCUDACodeGen}

    # Switch back to legacy -> the legacy codegen is triggered again.
    with dace.config.set_temporary('compiler', 'cuda', 'implementation', value='legacy'):
        used = _gpu_codegen_classes(_build_gpu_sdfg())
    assert used == {CUDACodeGen}


if __name__ == '__main__':
    test_both_gpu_codegens_are_registered()
    test_config_selects_active_gpu_codegen_at_runtime()
