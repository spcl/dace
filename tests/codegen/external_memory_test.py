# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
"""
Tests external memory allocation.
"""
import dace
import numpy as np
import pytest


@pytest.mark.parametrize('symbolic', (False, True))
def test_external_mem(symbolic):
    N = dace.symbol('N') if symbolic else 20

    @dace.program
    def tester(a: dace.float64[N]):
        workspace = dace.ndarray([N], dace.float64, lifetime=dace.AllocationLifetime.External)

        workspace[:] = a
        workspace += 1
        a[:] = workspace

    sdfg = tester.to_sdfg()

    # Test that there is no allocation
    code = sdfg.generate_code()[0].clean_code
    # No heap allocation in either form (plain or C++ >= 17 aligned).
    assert 'new double' not in code
    assert 'new (std::align_val_t(64)) double' not in code
    assert 'delete[]' not in code
    assert 'set_external_memory' in code

    a = np.random.rand(20)

    if symbolic:
        extra_args = dict(N=20)
    else:
        extra_args = {}

    # The nanobind interface stores no symbol values between calls (by
    # design), so the workspace queries take them per call; ctypes reuses
    # the values passed to initialize() and accepts no arguments here.
    ws_args = extra_args if dace.Config.get('compiler', 'interface') == 'nanobind' else {}

    # Test workspace size
    csdfg = sdfg.compile()
    csdfg.initialize(a, **extra_args)
    sizes = csdfg.get_workspace_sizes(**ws_args)
    assert sizes == {dace.StorageType.CPU_Heap: 20 * 8}

    # Test setting the workspace
    wsp = np.random.rand(20)
    csdfg.set_workspace(dace.StorageType.CPU_Heap, wsp, **ws_args)

    ref = a + 1

    csdfg(a, **extra_args)

    assert np.allclose(a, ref)
    assert np.allclose(wsp, ref)


def test_external_twobuffers():
    N = dace.symbol('N')

    @dace.program
    def tester(a: dace.float64[N]):
        workspace = dace.ndarray([N], dace.float64, lifetime=dace.AllocationLifetime.External)
        workspace2 = dace.ndarray([2], dace.float64, lifetime=dace.AllocationLifetime.External)

        workspace[:] = a
        workspace += 1
        workspace2[0] = np.sum(workspace)
        workspace2[1] = np.mean(workspace)
        a[0] = workspace2[0] + workspace2[1]

    sdfg = tester.to_sdfg()
    csdfg = sdfg.compile()

    # See test_external_mem: nanobind takes the symbols per call.
    ws_args = dict(N=20) if dace.Config.get('compiler', 'interface') == 'nanobind' else {}

    # Test workspace size
    a = np.random.rand(20)
    csdfg.initialize(a=a, N=20)
    sizes = csdfg.get_workspace_sizes(**ws_args)
    assert sizes == {dace.StorageType.CPU_Heap: 22 * 8}

    # Test setting the workspace
    wsp = np.random.rand(22)
    csdfg.set_workspace(dace.StorageType.CPU_Heap, wsp, **ws_args)

    ref = a + 1
    ref2 = np.copy(a)
    s, m = np.sum(ref), np.mean(ref)
    ref2[0] = s + m

    csdfg(a=a, N=20)

    assert np.allclose(a, ref2)
    assert np.allclose(wsp[:-2], ref)
    assert np.allclose(wsp[-2], s)
    assert np.allclose(wsp[-1], m)


def test_external_memory_detection_with_gpu_arrays():
    """Regression test for [PR#2461](https://github.com/spcl/dace/pull/2461)"""
    from dace.codegen import compiler
    from dace.codegen.compiled_sdfg import CompiledSDFG, ReloadableDLL
    from dace.config import set_temporary

    # This probes the CTYPES CompiledSDFG scan and needs the ctypes build
    # layout (stub library + program library); the nanobind interface builds
    # an extension module without the stub, so the interface is pinned.
    # The env var overrides set_temporary, so it must be dropped first.
    with pytest.MonkeyPatch.context() as mp:
        mp.delenv('DACE_compiler_interface', raising=False)
        with set_temporary('compiler', 'interface', value='ctypes'):
            _external_memory_detection_with_gpu_arrays_body(CompiledSDFG, ReloadableDLL, compiler)


def _external_memory_detection_with_gpu_arrays_body(CompiledSDFG, ReloadableDLL, compiler):

    # NOT named `tester` like the rest of the file: under `cache=unique` the
    # build folder derives from name+pid, so a shared name would alias this
    # ctypes build with the other tests' builds in one folder.
    @dace.program
    def extmem_detection_tester(a: dace.float64[20]):
        workspace = dace.ndarray([20], dace.float64, lifetime=dace.AllocationLifetime.External)
        workspace[:] = a
        workspace += 1
        a[:] = workspace

    sdfg = extmem_detection_tester.to_sdfg()
    csdfg = sdfg.compile()
    assert csdfg.has_gpu_code == False
    binary_name = str(compiler.get_binary_name(sdfg.build_folder, sdfg.name))
    # ONE shared DLL handle for both probes: a second ReloadableDLL over the
    # same file triggers the stub's already-loaded file-renaming fallback.
    lib = ReloadableDLL(binary_name)

    def make_probe(gpu_storage: dace.StorageType) -> CompiledSDFG:
        # ``arrays_recursive()`` yields in insertion order: the GPU array
        #  comes FIRST, before the external-lifetime arrays - the order the
        #  broken scan dropped.
        crafted = dace.SDFG(sdfg.name)
        crafted.add_array('gpu_scratch', [2], dace.float64, storage=gpu_storage, transient=True)
        crafted.add_array('ws_heap', [2],
                          dace.float64,
                          storage=dace.StorageType.CPU_Heap,
                          transient=True,
                          lifetime=dace.AllocationLifetime.External)
        crafted.add_array('ws_pinned', [2],
                          dace.float64,
                          storage=dace.StorageType.CPU_Pinned,
                          transient=True,
                          lifetime=dace.AllocationLifetime.External)
        crafted.add_state()
        return CompiledSDFG(crafted, lib)

    external = {dace.StorageType.CPU_Heap, dace.StorageType.CPU_Pinned}

    # Defect 1: a GPU_Global array must be detected as GPU code.
    probe = make_probe(dace.StorageType.GPU_Global)
    assert probe.has_gpu_code
    assert probe.external_memory_types == external

    # Defect 2: a leading GPU-storage array (GPU_Shared was the ONLY member
    #  of GPU_STORAGES, i.e. the storage that took the old ``break``) must
    #  not swallow the external arrays behind it.
    probe = make_probe(dace.StorageType.GPU_Shared)
    assert probe.external_memory_types == external
    assert not probe.has_gpu_code  # No GPU_Global array, no GPU-scheduled node.


if __name__ == '__main__':
    test_external_mem(False)
    test_external_mem(True)
    test_external_twobuffers()
    test_external_memory_detection_with_gpu_arrays()
