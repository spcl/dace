# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Tests the placement of GPU shared memory in static or dynamic shared memory.

Shared memory containers are placed as their ``StorageType.GPU_Shared(dynamic=...)`` attribute says, or, if left to the
code generator, statically while they fit in ``compiler.cuda.max_static_shared_memory`` (48 KiB on CUDA, 64 KiB on HIP)
and dynamically otherwise. Dynamic containers become views of a flat buffer bound to the kernel's dynamic shared memory,
and a kernel that needs more than the limit requests it from the device at launch.

The tests that only inspect the generated code run without a GPU; the ones marked ``gpu`` run the programs.
"""

import json
import re
import warnings
from typing import List, Optional

import numpy as np
import pytest

import dace
from dace import data as dt, nodes
from dace.codegen import common
from dace.transformation.passes import gpu_shared_memory

N = dace.symbol("N")
M = dace.symbol("M")
S = dace.StorageType

# 16 and 32 KiB of doubles: of the 48 KiB of static shared memory CUDA allows, two 16 KiB containers fit, and only one
# of 32 KiB
KIB16 = 2048
KIB32 = 4096


def _cuda_code(sdfg: dace.SDFG, backend: str = "cuda", **config) -> str:
    """Generates the GPU code of ``sdfg`` for the given backend, with the given ``compiler.cuda`` entries set."""
    with dace.config.temporary_config():
        dace.config.Config.set("compiler", "cuda", "backend", value=backend)
        dace.config.Config.set("compiler", "cuda", "default_block_size", value="32,1,1")
        for key, value in config.items():
            dace.config.Config.set("compiler", "cuda", key, value=value)
        # The backend is cached for the whole process; clear it before and after, so that the backend set above reaches
        # the code generator and does not leak into other tests
        common.get_gpu_backend.cache_clear()
        try:
            return next(c for c in sdfg.generate_code() if c.name == f"{sdfg.name}_cuda").clean_code
        finally:
            common.get_gpu_backend.cache_clear()


def _placement(sdfg: dace.SDFG, name: str) -> Optional[bool]:
    """Returns where ``name`` was placed: True for dynamic, False for static shared memory."""
    for nsdfg in sdfg.all_sdfgs_recursive():
        if name in nsdfg.arrays:
            desc = nsdfg.arrays[name]
            if isinstance(desc, dt.View):
                return True
            return dace.dtypes.is_dynamic_shared(desc.storage)
    raise KeyError(name)


def _shared_warnings(record: List[warnings.WarningMessage]) -> List[str]:
    return [str(w.message) for w in record if "placed in dynamic shared memory" in str(w.message)]


def _generate(sdfg: dace.SDFG, backend: str = "cuda", **config):
    """
    Generates the GPU code of ``sdfg``, and returns it with the placement warnings. Code generation works on a copy, so
    the shared memory passes are then also run on ``sdfg`` itself, for the tests to inspect their decisions.
    """
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        code = _cuda_code(sdfg, backend, **config)
    with warnings.catch_warnings(), dace.config.temporary_config():
        warnings.simplefilter("ignore")
        dace.config.Config.set("compiler", "cuda", "backend", value=backend)
        for key, value in config.items():
            dace.config.Config.set("compiler", "cuda", key, value=value)
        common.get_gpu_backend.cache_clear()
        try:
            gpu_shared_memory.plan_gpu_shared_memory(sdfg)
        finally:
            common.get_gpu_backend.cache_clear()
    return code, _shared_warnings(record)


@dace.program
def two_arrays(A: dace.float64[N] @ S.GPU_Global, B: dace.float64[N] @ S.GPU_Global):
    for i in dace.map[0:N:32] @ dace.ScheduleType.GPU_Device:
        s1 = dace.define_local([KIB16], dace.float64, storage=S.GPU_Shared)
        s2 = dace.define_local([KIB16], dace.float64, storage=S.GPU_Shared)
        for j in dace.map[0:32] @ dace.ScheduleType.GPU_ThreadBlock:
            s1[j] = A[i + j]
            s2[j] = s1[j] * 2
            B[i + j] = s2[j] + 1


@dace.program
def three_arrays(A: dace.float64[N] @ S.GPU_Global, B: dace.float64[N] @ S.GPU_Global):
    for i in dace.map[0:N:32] @ dace.ScheduleType.GPU_Device:
        s1 = dace.define_local([KIB32], dace.float64, storage=S.GPU_Shared)
        s2 = dace.define_local([KIB32], dace.float64, storage=S.GPU_Shared)
        s3 = dace.define_local([KIB32], dace.float64, storage=S.GPU_Shared)
        for j in dace.map[0:32] @ dace.ScheduleType.GPU_ThreadBlock:
            s1[j] = A[i + j]
            s2[j] = s1[j] * 2
            s3[j] = s2[j] + 1
            B[i + j] = s3[j]


@dace.program
def symbolic_size(A: dace.float64[N] @ S.GPU_Global, B: dace.float64[N] @ S.GPU_Global):
    for i in dace.map[0:N:32] @ dace.ScheduleType.GPU_Device:
        s = dace.define_local([M], dace.float64, storage=S.GPU_Shared)
        for j in dace.map[0:32] @ dace.ScheduleType.GPU_ThreadBlock:
            s[j] = A[i + j]
            B[i + j] = s[j] + 1


def _set_storage(sdfg: dace.SDFG, name: str, storage: S) -> dace.SDFG:
    for nsdfg in sdfg.all_sdfgs_recursive():
        if name in nsdfg.arrays:
            nsdfg.arrays[name].storage = storage
    return sdfg


def _two_level_sdfg(setzero: bool = False) -> dace.SDFG:
    """
    A kernel with a shared container of symbolic size ``a`` in its own SDFG, and a nested SDFG with a dynamically
    placed container ``b``: the nested SDFG's part of dynamic shared memory starts after ``a``, at a symbolic offset.
    """
    inner = dace.SDFG("two_level_inner")
    inner.add_array("x", [1], dace.float64)
    inner.add_array("y", [1], dace.float64)
    inner.add_array("b", [64], dace.float64, storage=S.GPU_Shared(dynamic=True), transient=True)
    inner_state = inner.add_state()
    b = inner_state.add_access("b")
    b.setzero = setzero
    inner_state.add_mapped_tasklet(
        "store",
        {"k": "0:1"},
        {"v": dace.Memlet("x[0]")},
        "w = v",
        {"w": dace.Memlet("b[k]")},
        external_edges=True,
        output_nodes={"b": b},
    )
    inner_state.add_mapped_tasklet(
        "load",
        {"k": "0:1"},
        {"v": dace.Memlet("b[k]")},
        "w = v + 1",
        {"w": dace.Memlet("y[0]")},
        external_edges=True,
        input_nodes={"b": b},
    )

    sdfg = dace.SDFG("two_level_shared")
    sdfg.add_array("A", [N], dace.float64, storage=S.GPU_Global)
    sdfg.add_array("B", [N], dace.float64, storage=S.GPU_Global)
    sdfg.add_array("a", [M], dace.float64, storage=S.GPU_Shared, transient=True)
    state = sdfg.add_state()
    kernel_entry, kernel_exit = state.add_map("kernel", {"i": "0:N:32"}, schedule=dace.ScheduleType.GPU_Device)
    block_entry, block_exit = state.add_map("block", {"j": "0:32"}, schedule=dace.ScheduleType.GPU_ThreadBlock)
    copy = state.add_tasklet("copy", {"v"}, {"w"}, "w = v")
    a = state.add_access("a")
    nsdfg = state.add_nested_sdfg(inner, {"x"}, {"y"})
    state.add_memlet_path(
        state.add_read("A"), kernel_entry, block_entry, copy, dst_conn="v", memlet=dace.Memlet("A[i + j]")
    )
    state.add_edge(copy, "w", a, None, dace.Memlet("a[j]"))
    state.add_edge(a, None, nsdfg, "x", dace.Memlet("a[j]"))
    state.add_memlet_path(
        nsdfg, block_exit, kernel_exit, state.add_write("B"), src_conn="y", memlet=dace.Memlet("B[i + j]")
    )
    nsdfg.integrate_into_parent()
    sdfg.validate()
    return sdfg


def _two_kernel_sdfg(first=("s",), second=("s",), storage: S = S.GPU_Shared) -> dace.SDFG:
    """
    Two kernels in one state: the first copies ``A`` to ``B`` and the second ``B`` to ``C``, each through the shared
    containers it names, in order. Containers named by both kernels are one data descriptor of the SDFG.
    """
    sdfg = dace.SDFG("two_kernels")
    for name in "ABC":
        sdfg.add_array(name, [N], dace.float64, storage=S.GPU_Global)
    for name in sorted(set(first) | set(second)):
        sdfg.add_array(name, [KIB32], dace.float64, storage=storage, transient=True)
    state = sdfg.add_state()
    source = state.add_read("A")
    for label, names, src, dst in (("first", first, "A", "B"), ("second", second, "B", "C")):
        kernel_entry, kernel_exit = state.add_map(label, {"i": "0:N:32"}, schedule=dace.ScheduleType.GPU_Device)
        block_entry, block_exit = state.add_map(
            f"{label}_block", {"j": "0:32"}, schedule=dace.ScheduleType.GPU_ThreadBlock
        )
        last = state.add_tasklet(f"{label}_load", {"v"}, {"w"}, "w = v")
        state.add_memlet_path(
            source, kernel_entry, block_entry, last, dst_conn="v", memlet=dace.Memlet(f"{src}[i + j]")
        )
        for k, name in enumerate(names):
            container = state.add_access(name)
            state.add_edge(last, "w", container, None, dace.Memlet(f"{name}[j]"))
            last = state.add_tasklet(f"{label}_{k}", {"v"}, {"w"}, "w = v")
            state.add_edge(container, None, last, "v", dace.Memlet(f"{name}[j]"))
        source = state.add_access(dst)
        state.add_memlet_path(last, block_exit, kernel_exit, source, src_conn="w", memlet=dace.Memlet(f"{dst}[i + j]"))
    sdfg.validate()
    return sdfg


# Storage type #########################################################################################################


def test_storage_type_attribute():
    dynamic = S.GPU_Shared(dynamic=True)
    assert dynamic == S.GPU_Shared and S.GPU_Shared == dynamic and dynamic != S.GPU_Global
    assert {S.GPU_Shared: "found"}[dynamic] == "found"
    assert dace.dtypes.is_dynamic_shared(S.GPU_Shared) is None
    assert dace.dtypes.is_dynamic_shared(S.GPU_Shared()) is None
    assert dace.dtypes.is_dynamic_shared(dynamic) is True
    assert dace.dtypes.is_dynamic_shared(S.GPU_Shared(dynamic=False)) is False
    with pytest.raises(ValueError):
        dace.dtypes.is_dynamic_shared(S.GPU_Global)


@pytest.mark.parametrize("storage", [S.GPU_Shared, S.GPU_Shared(dynamic=True), S.GPU_Shared(dynamic=False)])
def test_storage_type_serialization(storage: S):
    sdfg = dace.SDFG("storage_serialization")
    sdfg.add_array("a", [4], dace.float32, storage=storage, transient=True)
    sdfg.add_state()
    serialized = sdfg.to_json()
    stored = serialized["attributes"]["_arrays"]["a"]["attributes"]["storage"]
    if storage._is_template:
        # Stored as before the storage type had attributes
        assert stored == "GPU_Shared"
    restored = dace.SDFG.from_json(json.loads(json.dumps(serialized))).arrays["a"].storage
    assert restored is storage


# Placement ############################################################################################################


def test_fitting_containers_stay_static():
    sdfg = two_arrays.to_sdfg(simplify=False)
    code, shared_warnings = _generate(sdfg)
    assert not shared_warnings
    assert _placement(sdfg, "s1") is False and _placement(sdfg, "s2") is False
    assert "__shared__ double s1[2048];" in code and "__shared__ double s2[2048];" in code
    assert "extern __shared__" not in code
    assert "DACE_KERNEL_REQUEST_DYNAMIC_SHARED_MEMORY" not in code


def test_overflow_is_placed_dynamically():
    sdfg = three_arrays.to_sdfg(simplify=False)
    code, shared_warnings = _generate(sdfg)
    assert len(shared_warnings) == 2 and all("does not fit in the static shared memory" in w for w in shared_warnings)
    assert _placement(sdfg, "s1") is False
    assert _placement(sdfg, "s2") is True and _placement(sdfg, "s3") is True
    assert "__shared__ double s1[4096];" in code
    assert "extern __shared__ __align__(16) uint8_t __dace_dynsmem_extern[];" in code
    assert re.search(r"s2 = \(double\*\)\(&__dace_dynsmem\w*\[0\]\);", code)
    assert re.search(r"s3 = \(double\*\)\(&__dace_dynsmem\w*\[32768\]\);", code)
    # 64 KiB of dynamic shared memory is beyond what CUDA grants without opting in
    assert re.search(r'DACE_KERNEL_REQUEST_DYNAMIC_SHARED_MEMORY\((\w+), "\1", 65536\);', code)
    assert re.search(r"LaunchKernel\(.*, 65536, ", code)
    sdfg.validate()


def test_explicit_placement():
    sdfg = two_arrays.to_sdfg(simplify=False)
    _set_storage(sdfg, "s1", S.GPU_Shared(dynamic=True))
    _set_storage(sdfg, "s2", S.GPU_Shared(dynamic=False))
    code, shared_warnings = _generate(sdfg)
    assert not shared_warnings
    assert _placement(sdfg, "s1") is True and _placement(sdfg, "s2") is False
    assert "__shared__ double s2[2048];" in code
    assert re.search(r"s1 = \(double\*\)\(&__dace_dynsmem\w*\[0\]\);", code)
    # Within the limit, no request is needed
    assert "DACE_KERNEL_REQUEST_DYNAMIC_SHARED_MEMORY" not in code
    assert re.search(r"LaunchKernel\(.*, 16384, ", code)


def test_static_placement_of_symbolic_size_raises():
    sdfg = _set_storage(symbolic_size.to_sdfg(simplify=False), "s", S.GPU_Shared(dynamic=False))
    with pytest.raises(ValueError, match="requires a constant size"):
        _cuda_code(sdfg)


def test_symbolic_size_is_placed_dynamically():
    sdfg = symbolic_size.to_sdfg(simplify=False)
    code, shared_warnings = _generate(sdfg)
    assert not shared_warnings
    assert _placement(sdfg, "s") is True
    assert re.search(r"LaunchKernel\(.*, \(8 \* M\), ", code)
    # Whether the request is needed is only known at launch
    assert re.search(
        r'if \(\(\(8 \* M\)\) > 49152\) \{\s*DACE_KERNEL_REQUEST_DYNAMIC_SHARED_MEMORY\(\w+, "\w+", '
        r"\(8 \* M\)\);",
        code,
    )


def test_size_known_only_in_kernel_raises():
    sdfg = dace.SDFG("kernel_local_size")
    sdfg.add_array("A", [N], dace.float64, storage=S.GPU_Global)
    sdfg.add_array("s", ["i + 1"], dace.float64, storage=S.GPU_Shared, transient=True)
    state = sdfg.add_state()
    entry, exit = state.add_map("kernel", {"i": "0:N"}, schedule=dace.ScheduleType.GPU_Device)
    store = state.add_tasklet("store", {}, {"w"}, "w = 1")
    load = state.add_tasklet("load", {"v"}, {"w"}, "w = v")
    s = state.add_access("s")
    state.add_nedge(entry, store, dace.Memlet())
    state.add_edge(store, "w", s, None, dace.Memlet("s[0]"))
    state.add_edge(s, None, load, "v", dace.Memlet("s[0]"))
    state.add_memlet_path(load, exit, state.add_write("A"), src_conn="w", memlet=dace.Memlet("A[i]"))
    with pytest.raises(ValueError, match="not known when the kernel is launched"):
        _cuda_code(sdfg)


def test_nested_sdfg_offset_is_passed_as_symbol():
    sdfg = _two_level_sdfg(setzero=True)
    code, _ = _generate(sdfg)
    nsdfg = next(n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.NestedSDFG))
    base = gpu_shared_memory.DYNAMIC_SHARED_MEMORY_BASE
    assert base in nsdfg.sdfg.symbols
    assert str(nsdfg.symbol_mapping[base]) == "16*int_ceil(8*M, 16)"
    assert _placement(sdfg, "a") is True and _placement(sdfg, "b") is True
    assert re.search(rf"b = \(double\*\)\(&__dace_dynsmem\w*\[{base}\]\);", code)
    # A container placed dynamically is still zeroed where it is allocated
    assert "dace::ResetShared<double, 32, 1, 1, 64, 1, false>::Reset(b);" in code
    # The kernel is launched with both parts
    assert re.search(r"LaunchKernel\(.*, \(\(16 \* int_ceil\(\(8 \* M\), 16\)\) \+ 512\), ", code)
    sdfg.validate()


def test_kernels_do_not_share_containers():
    """A container two kernels access is a separate container, with its own place in shared memory, in each."""
    sdfg = _two_kernel_sdfg(storage=S.GPU_Shared(dynamic=True))
    code, _ = _generate(sdfg)
    assert _placement(sdfg, "s") is True and _placement(sdfg, "s_0") is True
    kernels = re.findall(r"__global__ void .*? (first|second)_\w+\((.*?)\) \{", code)
    assert len(kernels) == 2
    # Shared memory is not passed to either kernel, which would mean it was allocated outside of them
    assert all("__dace_dynsmem" not in args and " s" not in args for _, args in kernels)
    assert re.search(r"\bs = \(double\*\)\(&__dace_dynsmem\w*\[0\]\);", code)
    assert re.search(r"\bs_0 = \(double\*\)\(&__dace_dynsmem\w*\[0\]\);", code)
    assert len(re.findall(r"LaunchKernel\(.*, 32768, ", code)) == 2


def test_placement_is_per_kernel():
    """``s`` fits in the static shared memory of the first kernel, but not in the second, after ``a``."""
    sdfg = _two_kernel_sdfg(first=("s",), second=("a", "s"))
    code, shared_warnings = _generate(sdfg)
    assert len(shared_warnings) == 1 and '"s_0"' in shared_warnings[0]
    assert _placement(sdfg, "s") is False and _placement(sdfg, "a") is False and _placement(sdfg, "s_0") is True
    assert "__shared__ double s[4096];" in code and "__shared__ double a[4096];" in code
    assert re.search(r"\bs_0 = \(double\*\)\(&__dace_dynsmem\w*\[0\]\);", code)


@pytest.mark.parametrize("backend,warp_size", [("cuda", 32), ("hip", 64)])
def test_dynamic_map_state_follows_the_warp_size(backend: str, warp_size: int):
    """The fine-grained scheduling state holds two arrays of ``WARP_SIZE`` squared indices per warp."""
    with dace.config.temporary_config():
        dace.config.Config.set("compiler", "cuda", "backend", value=backend)
        common.get_gpu_backend.cache_clear()
        try:
            assert common.gpu_warp_size() == warp_size
            assert gpu_shared_memory.dynamic_map_state_elements(True, 128) == 2 * (128 // warp_size) * warp_size**2
            assert gpu_shared_memory.dynamic_map_state_elements(False, 128) == 4
        finally:
            common.get_gpu_backend.cache_clear()


def test_hip_limit():
    """HIP allows 64 KiB of static shared memory, so two of the three containers are static and none is requested."""
    sdfg = three_arrays.to_sdfg(simplify=False)
    code, shared_warnings = _generate(sdfg, backend="hip")
    assert len(shared_warnings) == 1
    assert _placement(sdfg, "s1") is False and _placement(sdfg, "s2") is False and _placement(sdfg, "s3") is True
    assert "DACE_KERNEL_REQUEST_DYNAMIC_SHARED_MEMORY" not in code
    assert re.search(r"LaunchKernel\(.*, 32768, ", code)


def test_configured_limit():
    sdfg = two_arrays.to_sdfg(simplify=False)
    code, shared_warnings = _generate(sdfg, max_static_shared_memory=8192)
    assert len(shared_warnings) == 2
    assert _placement(sdfg, "s1") is True and _placement(sdfg, "s2") is True
    # Beyond the configured limit, the 32 KiB of dynamic shared memory are requested
    assert re.search(r'DACE_KERNEL_REQUEST_DYNAMIC_SHARED_MEMORY\((\w+), "\1", 32768\);', code)


# End-to-end ###########################################################################################################


def _run(program, simplify: bool = False, size: int = 1024, **symbols) -> np.ndarray:
    sdfg = program.to_sdfg(simplify=simplify) if not isinstance(program, dace.SDFG) else program
    import cupy

    A = cupy.arange(size, dtype=cupy.float64)
    B = cupy.zeros(size, dtype=cupy.float64)
    sdfg(A=A, B=B, N=size, **symbols)
    return cupy.asnumpy(B)


@pytest.mark.gpu
def test_dynamic_shared_memory_beyond_the_default_limit():
    B = _run(three_arrays)
    assert np.array_equal(B, np.arange(1024, dtype=np.float64) * 2 + 1)


@pytest.mark.gpu
def test_symbolic_size():
    B = _run(symbolic_size, M=32)
    assert np.array_equal(B, np.arange(1024, dtype=np.float64) + 1)


@pytest.mark.gpu
def test_nested_sdfg_offset():
    B = _run(_two_level_sdfg(), M=40)
    assert np.array_equal(B, np.arange(1024, dtype=np.float64) + 1)


@pytest.mark.gpu
def test_kernels_with_separate_containers():
    import cupy

    sdfg = _two_kernel_sdfg(first=("s",), second=("a", "s"))
    A = cupy.arange(1024, dtype=cupy.float64)
    B = cupy.zeros(1024, dtype=cupy.float64)
    C = cupy.zeros(1024, dtype=cupy.float64)
    sdfg(A=A, B=B, C=C, N=1024)
    assert np.array_equal(cupy.asnumpy(C), np.arange(1024, dtype=np.float64))


@pytest.mark.gpu
def test_request_beyond_the_device_raises(capfd):
    """No GPU grants 16 MiB of shared memory per thread-block; the launch fails with the request and the limit."""
    with pytest.raises(RuntimeError, match="symbolic_size"):
        _run(symbolic_size, M=2 * 1024 * 1024)
    assert re.search(
        r"requests 16777216 bytes of dynamic shared memory, but device \d+ allows at most \d+ bytes",
        capfd.readouterr().out,
    )


if __name__ == "__main__":
    test_storage_type_attribute()
    for storage in (S.GPU_Shared, S.GPU_Shared(dynamic=True), S.GPU_Shared(dynamic=False)):
        test_storage_type_serialization(storage)
    test_fitting_containers_stay_static()
    test_overflow_is_placed_dynamically()
    test_explicit_placement()
    test_static_placement_of_symbolic_size_raises()
    test_symbolic_size_is_placed_dynamically()
    test_size_known_only_in_kernel_raises()
    test_nested_sdfg_offset_is_passed_as_symbol()
    test_kernels_do_not_share_containers()
    test_placement_is_per_kernel()
    test_dynamic_map_state_follows_the_warp_size("cuda", 32)
    test_dynamic_map_state_follows_the_warp_size("hip", 64)
    test_hip_limit()
    test_configured_limit()
    test_dynamic_shared_memory_beyond_the_default_limit()
    test_symbolic_size()
    test_nested_sdfg_offset()
    test_kernels_with_separate_containers()
    # test_request_beyond_the_device_raises needs pytest's ``capfd`` fixture
