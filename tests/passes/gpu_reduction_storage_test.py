# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Structural tests for storage-aware GPU reduction lowering + Custom (ITE) WCR handling.

Storage-aware lowering (the GPU ``Reduce`` expansions inspect their OUTPUT descriptor's storage):

* a device-writable output (``GPU_Global`` / ``CPU_Pinned``) lowers straight to the output --
  CUB ``DeviceReduce`` / the device-map reduction writes the device pointer directly, with NO extra
  transient/copy;
* a host output (``CPU_Heap`` / ``Register`` / ``Default`` / ...) does NOT fall back to the slow pure
  loop -- the expansion reduces into a fresh ``GPU_Global`` scratch and copies it to the real output
  (a device->host copy DaCe emits at the storage boundary).

ITE (Custom) WCR: an ``a if a > b else b`` (argmax/argmin-style) reduction is a ``Custom`` reduction
type; every GPU expansion must lower it as a device functor with a C++ ternary, never drop it or fall
back to pure. ``ExpandReduceGPUAuto`` cannot express a Custom op through its ``dace::warpReduce``
primitive, so it delegates to the CUB ``DeviceReduce`` functor path (carrying the CUB environments).

All tests are structural (build/expand/generate-code, inspect) -- none need a GPU.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MPI4PY_RC_INITIALIZE", "0")
os.environ.setdefault("OMPI_MCA_pml", "ob1")
os.environ.setdefault("OMPI_MCA_btl", "self,vader")
os.environ.setdefault("UCX_VFS_ENABLE", "n")

import pytest

import dace
from dace import dtypes
from dace.libraries.standard.nodes.reduce import GPU_REDUCE_DEVICE_WRITABLE_STORAGE

N = dace.symbol('N')

#: The GPU reduce expansions that must be storage-aware.
GPU_REDUCE_IMPLS = ['GPUAuto', 'CUDA (device)']


def build_reduce(out_storage, impl, wcr='lambda a, b: a + b', identity=0.0):
    """A single top-level ``Reduce`` (kernel-mode): ``A`` (GPU_Global) -> reduce -> ``out``.

    ``out`` is non-transient with the requested storage; a non-device ``out`` also feeds a
    ``sink`` so the reduction result is used.
    """
    sdfg = dace.SDFG(f'red_{impl.replace(" ", "_").replace("(", "").replace(")", "")}_{out_storage.name}')
    sdfg.add_array('A', [N], dace.float64, storage=dtypes.StorageType.GPU_Global)
    device_out = out_storage in GPU_REDUCE_DEVICE_WRITABLE_STORAGE
    sdfg.add_array('out', [1], dace.float64, storage=out_storage, transient=False)
    if not device_out:
        sdfg.add_array('sink', [1], dace.float64, storage=dtypes.StorageType.GPU_Global)
    st = sdfg.add_state()
    red = st.add_reduce(wcr, axes=None, identity=identity)
    red.implementation = impl
    red.schedule = dtypes.ScheduleType.GPU_Device
    st.add_nedge(st.add_read('A'), red, dace.Memlet('A[0:N]'))
    w = st.add_write('out')
    st.add_nedge(red, w, dace.Memlet('out[0]'))
    if not device_out:
        st.add_nedge(w, st.add_write('sink'), dace.Memlet('sink[0]'))
    sdfg.validate()
    return sdfg


def gpu_global_scratch_count(sdfg):
    """Number of transient ``GPU_Global`` scratch arrays introduced by the host-output lowering."""
    return sum(1 for _, name, d in sdfg.arrays_recursive()
               if d.transient and d.storage == dtypes.StorageType.GPU_Global and '_out_gpu' in name)


@pytest.mark.parametrize('impl', GPU_REDUCE_IMPLS)
def test_device_output_lowers_without_scratch(impl):
    """A device-writable (GPU_Global) reduce output is written directly -- no GPU scratch/copy."""
    sdfg = build_reduce(dtypes.StorageType.GPU_Global, impl)
    sdfg.expand_library_nodes()
    sdfg.validate()
    assert gpu_global_scratch_count(sdfg) == 0, \
        f"{impl}: device output must lower with no GPU scratch"


@pytest.mark.parametrize('impl', GPU_REDUCE_IMPLS)
@pytest.mark.parametrize('host_storage', [dtypes.StorageType.Register, dtypes.StorageType.CPU_Heap])
def test_host_output_lowers_via_gpu_scratch_and_copy(impl, host_storage):
    """A host reduce output lowers to a GPU_Global scratch + a device->host copy (never pure)."""
    sdfg = build_reduce(host_storage, impl)
    sdfg.expand_library_nodes()
    sdfg.validate()
    # Exactly one GPU_Global scratch, and a copy state whose source is that scratch and destination is
    # the host output -- i.e. the reduction never writes host memory directly.
    assert gpu_global_scratch_count(sdfg) == 1, \
        f"{impl}/{host_storage.name}: host output must lower via a GPU_Global scratch + copy"
    found_copy = False
    for nsdfg in sdfg.all_sdfgs_recursive():
        for state in nsdfg.states():
            for e in state.edges():
                src, dst = e.src, e.dst
                if (isinstance(src, dace.nodes.AccessNode) and isinstance(dst, dace.nodes.AccessNode)
                        and '_out_gpu' in src.data):
                    dst_desc = nsdfg.arrays[dst.data]
                    if dst_desc.storage not in GPU_REDUCE_DEVICE_WRITABLE_STORAGE:
                        found_copy = True
    assert found_copy, f"{impl}/{host_storage.name}: expected a scratch->host-output copy edge"


@pytest.mark.parametrize('impl', GPU_REDUCE_IMPLS)
def test_ite_custom_wcr_lowers_to_device_functor_ternary(impl):
    """An ITE (Custom) WCR ``a if a > b else b`` lowers to a device functor with a C++ ternary,
    with no NotImplementedError and no fall-back to a pure reduction map."""
    sdfg = build_reduce(dtypes.StorageType.GPU_Global, impl, wcr='lambda a, b: a if a > b else b', identity=-1e300)
    sdfg.expand_library_nodes()
    sdfg.validate()
    code = "\n".join(c.clean_code for c in sdfg.generate_code())
    assert 'struct __reduce_' in code, f"{impl}: ITE reduction must emit a device functor struct"
    assert '? ' in code and ' : ' in code, f"{impl}: ITE functor must contain a C++ ternary"
    # The custom reduction must be a CUB DeviceReduce (functor), not the pure element-wise WCR loop.
    assert 'cub::DeviceReduce' in code or 'DeviceReduce' in code, \
        f"{impl}: ITE reduction must lower to a CUB DeviceReduce functor (no pure fallback)"


def test_gpuauto_delegates_custom_and_resets_environment():
    """GPUAuto carries the CUB scratch environment ONLY when delegating a Custom WCR; a subsequent
    non-Custom GPUAuto expansion must not leak that (128 MB) environment."""
    from dace.libraries.standard.nodes.reduce import ExpandReduceGPUAuto
    # Custom -> delegates, GPUAuto env includes CUB ReduceScratch.
    custom = build_reduce(dtypes.StorageType.GPU_Global,
                           'GPUAuto',
                           wcr='lambda a, b: a if a > b else b',
                           identity=-1e300)
    custom.expand_library_nodes()
    env_paths = {e if isinstance(e, str) else e.full_class_path() for e in ExpandReduceGPUAuto.environments}
    assert any('ReduceScratch' in p for p in env_paths), "Custom delegation must carry the CUB ReduceScratch env"
    # Non-Custom -> inline device maps, env reset to empty (no leaked 128 MB scratch pool).
    plain = build_reduce(dtypes.StorageType.GPU_Global, 'GPUAuto')
    plain.expand_library_nodes()
    assert list(ExpandReduceGPUAuto.environments) == [], "non-Custom GPUAuto must reset environments to empty"


if __name__ == '__main__':
    for impl in GPU_REDUCE_IMPLS:
        test_device_output_lowers_without_scratch(impl)
        for hs in [dtypes.StorageType.Register, dtypes.StorageType.CPU_Heap]:
            test_host_output_lowers_via_gpu_scratch_and_copy(impl, hs)
        test_ite_custom_wcr_lowers_to_device_functor_ternary(impl)
    test_gpuauto_delegates_custom_and_resets_environment()
    print('OK')
