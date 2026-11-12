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

import numpy as np
import pytest

import dace
from dace import data as dt
from dace import nodes
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


def _placement(sdfg: dace.SDFG, name: str) -> bool | None:
    """Returns where ``name`` was placed: True for dynamic, False for static shared memory."""
    for nsdfg in sdfg.all_sdfgs_recursive():
        if name in nsdfg.arrays:
            desc = nsdfg.arrays[name]
            if isinstance(desc, dt.View):
                return True
            return desc.storage.dynamic
    raise KeyError(name)


def _shared_warnings(record: list[warnings.WarningMessage]) -> list[str]:
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
def static_and_dynamic(A: dace.float64[N] @ S.GPU_Global, B: dace.float64[N] @ S.GPU_Global):
    for i in dace.map[0:N:32] @ dace.ScheduleType.GPU_Device:
        s1 = dace.define_local([KIB32], dace.float64, storage=S.GPU_Shared)
        s2 = dace.define_local([KIB32], dace.float64, storage=S.GPU_Shared)
        for j in dace.map[0:32] @ dace.ScheduleType.GPU_ThreadBlock:
            s1[j] = A[i + j]
            s2[j] = s1[j] * 2
            B[i + j] = s2[j] + 1


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


def _two_level_sdfg(setzero: bool = False, inner_size=64) -> dace.SDFG:
    """
    A kernel with a shared container of symbolic size ``a`` in its own SDFG, and a nested SDFG with a dynamically
    placed container ``b`` of ``inner_size`` elements, which conflicts with ``a``: it is placed before ``a`` if its size
    is a constant, after it (at a symbolic offset) otherwise.
    """
    inner = dace.SDFG("two_level_inner")
    inner.add_array("x", [1], dace.float64)
    inner.add_array("y", [1], dace.float64)
    if dace.symbolic.issymbolic(inner_size):
        inner.add_symbol("M", M.dtype)
    inner.add_array("b", [inner_size], dace.float64, storage=S.GPU_Shared(dynamic=True), transient=True)
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
    mapping = {"M": "M"} if "M" in inner.symbols else None
    nsdfg = state.add_nested_sdfg(inner, {"x"}, {"y"}, symbol_mapping=mapping)
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
    assert S.GPU_Shared.dynamic is None
    assert S.GPU_Shared().dynamic is None
    assert dynamic.dynamic is True
    assert S.GPU_Shared(dynamic=False).dynamic is False
    with pytest.raises(AttributeError):
        S.GPU_Global.dynamic


@pytest.mark.parametrize(
    "storage",
    [
        S.GPU_Shared,
        S.GPU_Shared(dynamic=True),
        S.GPU_Shared(dynamic=False),
    ],
)
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
    assert "__shared__ __align__(16) double s1[2048];" in code and "__shared__ __align__(16) double s2[2048];" in code
    assert "extern __shared__" not in code
    assert "DACE_KERNEL_REQUEST_DYNAMIC_SHARED_MEMORY" not in code


def test_overflow_is_placed_dynamically():
    sdfg = three_arrays.to_sdfg(simplify=False)
    code, shared_warnings = _generate(sdfg)
    assert len(shared_warnings) == 2 and all("does not fit in the static shared memory" in w for w in shared_warnings)
    assert _placement(sdfg, "s1") is False
    assert _placement(sdfg, "s2") is True and _placement(sdfg, "s3") is True
    assert "__shared__ __align__(16) double s1[4096];" in code
    assert "extern __shared__ __align__(16) uint8_t __dace_dynsmem_extern[];" in code
    assert re.search(r"s2 = \(double\*\)\(&__dace_dynsmem\w*\[0\]\);", code)
    assert re.search(r"s3 = \(double\*\)\(&__dace_dynsmem\w*\[32768\]\);", code)
    # 64 KiB of dynamic shared memory is beyond what CUDA grants without opting in
    assert re.search(r'DACE_KERNEL_REQUEST_DYNAMIC_SHARED_MEMORY\((\w+), "\1", 65536\);', code)
    assert re.search(r"LaunchKernel\(.*, 65536, ", code)
    sdfg.validate()


def test_static_and_dynamic_beyond_the_default_limit():
    """32 KiB of static and 32 KiB of dynamic shared memory: each fits in the default limit, together they do not."""
    sdfg = static_and_dynamic.to_sdfg(simplify=False)
    code, _ = _generate(sdfg)
    assert _placement(sdfg, "s1") is False and _placement(sdfg, "s2") is True
    assert re.search(r'DACE_KERNEL_REQUEST_DYNAMIC_SHARED_MEMORY\((\w+), "\1", 32768\);', code)


def test_static_containers_are_aligned():
    """Static containers are aligned as dynamic ones, for vector copies (here, two bytes would suffice)."""
    sdfg = two_arrays.to_sdfg(simplify=False)
    for nsdfg in sdfg.all_sdfgs_recursive():
        for desc in nsdfg.arrays.values():
            if desc.storage == S.GPU_Shared:
                desc.dtype = dace.int16
    code, _ = _generate(sdfg)
    assert "__shared__ __align__(16) int16_t s1[2048];" in code


def test_explicit_placement():
    sdfg = two_arrays.to_sdfg(simplify=False)
    _set_storage(sdfg, "s1", S.GPU_Shared(dynamic=True))
    _set_storage(sdfg, "s2", S.GPU_Shared(dynamic=False))
    code, shared_warnings = _generate(sdfg)
    assert not shared_warnings
    assert _placement(sdfg, "s1") is True and _placement(sdfg, "s2") is False
    assert "__shared__ __align__(16) double s2[2048];" in code
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
    """A nested container after one of symbolic size receives its offset as a symbol."""
    sdfg = _two_level_sdfg(inner_size=M)
    code, _ = _generate(sdfg)
    nsdfg = next(n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.NestedSDFG))
    base = gpu_shared_memory.DYNAMIC_SHARED_MEMORY_BASE
    assert base in nsdfg.sdfg.symbols
    assert str(nsdfg.symbol_mapping[base]) == "16*int_ceil(8*M, 16)"
    assert _placement(sdfg, "a") is True and _placement(sdfg, "b") is True
    assert re.search(r"\ba = \(double\*\)\(&__dace_dynsmem\w*\[0\]\);", code)
    assert re.search(rf"\bb = \(double\*\)\(&__dace_dynsmem\w*\[{base}\]\);", code)
    # The kernel is launched with both parts
    assert re.search(r"LaunchKernel\(.*, \(\(8 \* M\) \+ \(16 \* int_ceil\(\(8 \* M\), 16\)\)\), ", code)
    sdfg.validate()


def test_constant_sizes_are_placed_first():
    """A nested container of constant size is placed before one of symbolic size it conflicts with: at a constant
    offset, without a symbol."""
    sdfg = _two_level_sdfg(setzero=True)
    code, _ = _generate(sdfg)
    nsdfg = next(n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.NestedSDFG))
    assert gpu_shared_memory.DYNAMIC_SHARED_MEMORY_BASE not in nsdfg.sdfg.symbols
    assert re.search(r"\bb = \(double\*\)\(&__dace_dynsmem\w*\[0\]\);", code)
    assert re.search(r"\ba = \(double\*\)\(&__dace_dynsmem\w*\[512\]\);", code)
    # A container placed dynamically is still zeroed where it is allocated
    assert "dace::ResetShared<double, 32, 1, 1, 64, 1, false>::Reset(b);" in code
    assert re.search(r"LaunchKernel\(.*, \(\(8 \* M\) \+ 512\), ", code)
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
    assert "__shared__ __align__(16) double s[4096];" in code and "__shared__ __align__(16) double a[4096];" in code
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
    """HIP allows 64 KiB of static shared memory, so two of the three containers are static. With the third, the kernel
    uses more than that, and requests its dynamic shared memory."""
    sdfg = three_arrays.to_sdfg(simplify=False)
    code, shared_warnings = _generate(sdfg, backend="hip")
    assert len(shared_warnings) == 1
    assert _placement(sdfg, "s1") is False and _placement(sdfg, "s2") is False and _placement(sdfg, "s3") is True
    assert re.search(r'DACE_KERNEL_REQUEST_DYNAMIC_SHARED_MEMORY\((\w+), "\1", 32768\);', code)
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
def test_static_and_dynamic_shared_memory_beyond_the_default_limit():
    B = _run(static_and_dynamic)
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


def test_inline_storage_constructor():
    """A storage type can be instantiated inside a program, without binding it to a global first."""

    @dace.program
    def prog(A: dace.float64[32] @ S.GPU_Global):
        for _ in dace.map[0:1] @ dace.ScheduleType.GPU_Device:
            s = dace.ndarray([32], dtype=dace.float64, storage=dace.StorageType.GPU_Shared(dynamic=True))
            s[:] = 1
            A[:] = s

    sdfg = prog.to_sdfg()
    assert _placement(sdfg, "s") is True
    assert sdfg.arrays["s"].storage == S.GPU_Shared(dynamic=True)


def _task(name: str, value: int) -> dace.SDFG:
    """A thread block's task: its threads fill a 16 KiB shared container, and then read it in reverse (a value another
    thread wrote) into ``y``."""
    task = dace.SDFG(name)
    task.add_array("y", [32], dace.float64)
    task.add_array("s", [KIB16], dace.float64, storage=S.GPU_Shared(dynamic=True), transient=True)
    tb = dace.ScheduleType.GPU_ThreadBlock
    fill = task.add_state("fill")
    fill.add_mapped_tasklet(
        "fill", {"j": "0:32"}, {}, f"w = {value} + j", {"w": dace.Memlet("s[j]")}, schedule=tb, external_edges=True
    )
    read = task.add_state_after(fill, "read")
    read.add_mapped_tasklet(
        "read",
        {"j": "0:32"},
        {"v": dace.Memlet("s[31 - j]")},
        "w = v",
        {"w": dace.Memlet("y[j]")},
        schedule=tb,
        external_edges=True,
    )
    return task


def _tasks_sdfg(same_state: bool) -> dace.SDFG:
    """A kernel whose thread blocks run two tasks with shared containers of their own, in two states of the block's
    code (or in one)."""
    body = dace.SDFG("tasks_body")
    body.add_array("b", [2, 32], dace.float64, storage=S.GPU_Global)
    first = body.add_state("first")
    second = first if same_state else body.add_state_after(first, "second")
    for row, (state, value) in enumerate(((first, 100), (second, 200))):
        call = state.add_nested_sdfg(_task(f"task{row}", value), set(), {"y"})
        state.add_edge(call, "y", state.add_write("b"), None, dace.Memlet(f"b[{row}, 0:32]"))
        call.integrate_into_parent()

    sdfg = dace.SDFG("tasks_same_state" if same_state else "tasks_in_states")
    sdfg.add_array("B", [N, 2, 32], dace.float64, storage=S.GPU_Global)
    state = sdfg.add_state()
    entry, exit_ = state.add_map("kernel", {"i": "0:N"}, schedule=dace.ScheduleType.GPU_Device)
    call = state.add_nested_sdfg(body, set(), {"b"})
    state.add_nedge(entry, call, dace.Memlet())
    state.add_memlet_path(call, exit_, state.add_write("B"), src_conn="b", memlet=dace.Memlet("B[i, 0:2, 0:32]"))
    call.integrate_into_parent()
    sdfg.validate()
    return sdfg


def test_tasks_in_different_states_share_shared_memory():
    """The tasks of different states of a block's code never run at the same time: their containers start at the same
    offset of dynamic shared memory. In one state too, as each task ends with a barrier."""
    for same_state, size in ((False, KIB16 * 8), (True, KIB16 * 8)):
        sdfg = _tasks_sdfg(same_state)
        code, _ = _generate(sdfg)
        assert re.search(rf"LaunchKernel\(.*, {size}, ", code), same_state
        offsets = re.findall(r"\bs = \(double\*\)\(&__dace_dynsmem\w*\[(\w+)\]\);", code)
        assert len(offsets) == 2, offsets


@pytest.mark.gpu
def test_tasks_in_different_states_run():
    import cupy

    sdfg = _tasks_sdfg(same_state=False)
    B = cupy.zeros((3, 2, 32))
    sdfg(B=B, N=3)
    expected = np.stack([100 + np.arange(32)[::-1], 200 + np.arange(32)[::-1]])
    assert np.array_equal(cupy.asnumpy(B), np.broadcast_to(expected, B.shape))


# The containers that may share memory (``SharedMemoryInterference``)

TB = dace.ScheduleType.GPU_ThreadBlock


def _conflicts(sdfg: dace.SDFG, reuse: bool | None = True) -> gpu_shared_memory.SharedMemoryConflicts:
    """The conflicts of the only kernel of an SDFG (the passes before the analysis run first)."""
    gpu_shared_memory.PrivatizeKernelSharedMemory().apply_pass(sdfg, {})
    gpu_shared_memory.LowerDynamicMapState().apply_pass(sdfg, {})
    (result,) = gpu_shared_memory.SharedMemoryInterference(reuse).apply_pass(sdfg, {}).values()
    return result


def _shares(conflicts: gpu_shared_memory.SharedMemoryConflicts, first: str, second: str) -> bool:
    """Whether two containers, given as ``sdfg:name``, may share memory."""
    found = {f"{sdfg.name}:{name}": (sdfg, name) for sdfg, name in conflicts.containers}
    return not conflicts.conflict(found[first], found[second])


def _kernel(body: dace.SDFG, rows: int) -> dace.SDFG:
    """A kernel whose thread blocks run ``body``, which writes rows of 32 elements of ``b`` (of ``B`` for each block)."""
    sdfg = dace.SDFG(f"{body.name}_kernel")
    sdfg.add_array("B", [N, rows, 32], dace.float64, storage=S.GPU_Global)
    state = sdfg.add_state()
    entry, exit_ = state.add_map("kernel", {"i": "0:N"}, schedule=dace.ScheduleType.GPU_Device)
    mapping = {"i": "i"} if "i" in body.symbols else None
    call = state.add_nested_sdfg(body, set(), {"b"}, symbol_mapping=mapping)
    state.add_nedge(entry, call, dace.Memlet())
    state.add_memlet_path(call, exit_, state.add_write("B"), src_conn="b", memlet=dace.Memlet(f"B[i, 0:{rows}, 0:32]"))
    call.integrate_into_parent()
    sdfg.validate()
    return sdfg


def _body(name: str, rows: int, shared: tuple[str, ...]) -> dace.SDFG:
    """A thread block's code with 16 KiB shared containers ``shared``, which writes ``rows`` rows of ``b``."""
    body = dace.SDFG(name)
    body.add_array("b", [rows, 32], dace.float64, storage=S.GPU_Global)
    for container in shared:
        body.add_array(container, [KIB16], dace.float64, storage=S.GPU_Shared, transient=True)
    return body


def _fill(state: dace.SDFGState, name: str, value: int = 1) -> nodes.AccessNode:
    """The threads of the block write elements of a shared container."""
    state.add_mapped_tasklet(
        f"fill_{name}", {"j": "0:32"}, {}, f"w = {value} + j", {"w": dace.Memlet(f"{name}[j]")}, schedule=TB,
        external_edges=True,
    )  # fmt: skip
    return next(n for n in state.sink_nodes() if isinstance(n, nodes.AccessNode) and n.data == name)


def _drain(state: dace.SDFGState, name: str, row: int, source: nodes.AccessNode | None = None) -> None:
    """The threads of the block copy elements of a shared container that other threads wrote into a row of ``b``."""
    state.add_mapped_tasklet(
        f"drain_{name}", {"j": "0:32"}, {"v": dace.Memlet(f"{name}[31 - j]")}, "w = v",
        {"w": dace.Memlet(f"b[{row}, j]")}, schedule=TB, external_edges=True,
        input_nodes={name: source} if source is not None else None,
    )  # fmt: skip


def _phased_producer(value: int, phases: int = 2, loop: bool = False) -> dace.SDFG:
    """
    A thread block's computation of ``out`` (a shared container of the caller) with a 16 KiB scratch ``s``: its threads
    fill the scratch and read it in reverse, then write ``out``. In two phases (states, a value of every thread in a
    register in between), the scratch is no longer used when ``out`` is written; in one, a C++ tasklet does all. With
    ``loop``, the two phases repeat.
    """
    producer = dace.SDFG(f"producer{value}")
    producer.add_array("out", [32], dace.float64, storage=S.GPU_Shared)
    producer.add_array("s", [KIB16], dace.float64, storage=S.GPU_Shared, transient=True)
    producer.add_array("acc", [1], dace.float64, storage=S.Register, transient=True)
    region = producer
    if loop:
        region = dace.sdfg.state.LoopRegion("repeat", "k < 2", "k", "k = 0", "k = k + 1", sdfg=producer)
        producer.add_node(region, is_start_block=True)
    compute = region.add_state("compute", is_start_block=True)
    entry, exit_ = compute.add_map("threads", {"j": "0:32"}, schedule=TB)
    fill = f"scratch[j] = {value} + j;\n__syncthreads();\n"
    if phases == 1:
        code = fill + "double v = scratch[31 - j];\n__syncthreads();\no = v;\n"
        tasklet = compute.add_tasklet("produce", {}, {"scratch", "o"}, code, language=dace.Language.CPP)
        compute.add_nedge(entry, tasklet, dace.Memlet())
        compute.add_memlet_path(tasklet, exit_, compute.add_write("out"), src_conn="o", memlet=dace.Memlet("out[j]"))
    else:
        code = fill + "a = scratch[31 - j];\n"
        tasklet = compute.add_tasklet("compute", {}, {"scratch", "a"}, code, language=dace.Language.CPP)
        compute.add_nedge(entry, tasklet, dace.Memlet())
        compute.add_memlet_path(tasklet, exit_, compute.add_write("acc"), src_conn="a", memlet=dace.Memlet("acc[0]"))
        store = region.add_state_after(compute, "store")
        store.add_mapped_tasklet(
            "store", {"j": "0:32"}, {"v": dace.Memlet("acc[0]")}, "w = v", {"w": dace.Memlet("out[j]")}, schedule=TB,
            external_edges=True,
        )  # fmt: skip
    scratch = dace.Memlet(f"s[0:{KIB16}]", dynamic=True)
    compute.add_memlet_path(tasklet, exit_, compute.add_write("s"), src_conn="scratch", memlet=scratch)
    return producer


def _producer_body(producer: dace.SDFG, written_before: bool = False) -> dace.SDFG:
    """A thread block's code that produces ``t`` and copies it out, or, in the other branch, runs a task with a shared
    container of its own. With ``written_before``, ``t`` is written first (and the producer writes part of it)."""
    body = _body("producer_body", 2, ("t",))
    body.add_symbol("i", dace.int64)
    start = body.add_state("start", is_start_block=True)
    if written_before:
        _fill(start, "t")
    produce, other = body.add_state("produce"), body.add_state("other")
    body.add_edge(start, produce, dace.InterstateEdge("i % 2 == 0"))
    body.add_edge(start, other, dace.InterstateEdge("i % 2 == 1"))
    call = produce.add_nested_sdfg(producer, set(), {"out"})
    t = produce.add_access("t")
    produce.add_edge(call, "out", t, None, dace.Memlet("t[0:32]"))
    _drain(produce, "t", 0, t)
    call.integrate_into_parent()
    task = other.add_nested_sdfg(_task("task", 200), set(), {"y"})
    other.add_edge(task, "y", other.add_write("b"), None, dace.Memlet("b[1, 0:32]"))
    task.integrate_into_parent()
    return body


def test_workspace_dead_before_the_output_shares_its_memory():
    """A two-phase producer's scratch is dead once the output is written: they share memory. Not so if one C++ tasklet
    writes both, or if the phases repeat in a loop."""
    conflicts = _conflicts(_kernel(_producer_body(_phased_producer(100)), 2))
    assert not conflicts.pinned
    assert _shares(conflicts, "producer100:s", "producer_body:t")
    assert _shares(conflicts, "producer100:s", "task:s") and _shares(conflicts, "producer_body:t", "task:s")
    for producer in (_phased_producer(100, phases=1), _phased_producer(100, loop=True)):
        conflicts = _conflicts(_kernel(_producer_body(producer), 2))
        assert not _shares(conflicts, "producer100:s", "producer_body:t")
        assert _shares(conflicts, "producer100:s", "task:s")


def test_scratch_overlaps_an_output_that_holds_no_values():
    """``t`` holds no values until the producer writes it, whose scratch is dead by then: the scratch, ``t``, and the
    task's container in the other branch, where ``t`` holds no values either, start at the same offset. Written before
    (and read after), ``t`` conflicts with the scratch, but shares its memory with the task's container: the scratch,
    which shares memory with nothing, stays static."""
    code, _ = _generate(_kernel(_producer_body(_phased_producer(100)), 2))
    assert re.search(rf"LaunchKernel\(.*, {KIB16 * 8}, ", code) and "__shared__ __align__(16) double" not in code
    offsets = re.findall(r"\b[st] = \(double\*\)\(&__dace_dynsmem\w*\[(\w+)\]\);", code)
    assert len(offsets) == 3 and set(offsets) == {"0"}, offsets

    code, _ = _generate(_kernel(_producer_body(_phased_producer(100), written_before=True), 2))
    assert re.search(rf"LaunchKernel\(.*, {KIB16 * 8}, ", code)
    assert re.search(rf"__shared__ __align__\(16\) double s\[{KIB16}\]", code)
    offsets = re.findall(r"\b[st] = \(double\*\)\(&__dace_dynsmem\w*\[(\w+)\]\);", code)
    assert len(offsets) == 2 and set(offsets) == {"0"}, offsets


@pytest.mark.gpu
@pytest.mark.parametrize("written_before", [False, True])
def test_scratch_overlapping_an_output_runs(written_before: bool):
    import cupy

    sdfg = _kernel(_producer_body(_phased_producer(100), written_before), 2)
    B = cupy.zeros((4, 2, 32))
    sdfg(B=B, N=4)
    expected = np.zeros((4, 2, 32))
    expected[0::2, 0] = 100 + np.arange(32)  # Even blocks produce (read in reverse twice), odd ones run the task
    expected[1::2, 1] = 200 + np.arange(32)[::-1]
    assert np.array_equal(cupy.asnumpy(B), expected)


@pytest.mark.gpu
def test_tasks_in_one_state_run():
    import cupy

    sdfg = _tasks_sdfg(same_state=True)
    B = cupy.zeros((3, 2, 32))
    sdfg(B=B, N=3)
    expected = np.stack([100 + np.arange(32)[::-1], 200 + np.arange(32)[::-1]])
    assert np.array_equal(cupy.asnumpy(B), np.broadcast_to(expected, B.shape))


def test_value_live_across_a_nested_sdfg():
    """``t`` holds values written before the producer, read after it: it conflicts with the producer's scratch. In the
    other branch it is never read again: that dead value does not block the task's container."""
    conflicts = _conflicts(_kernel(_producer_body(_phased_producer(100), written_before=True), 2))
    assert not _shares(conflicts, "producer100:s", "producer_body:t")
    assert _shares(conflicts, "producer_body:t", "task:s")


def test_tasks_in_one_state_share_memory():
    """Two nested SDFGs of one state, each ending with a barrier: their containers share memory."""
    conflicts = _conflicts(_tasks_sdfg(same_state=True))
    assert not conflicts.pinned and _shares(conflicts, "task0:s", "task1:s")


def test_containers_of_exclusive_branches_and_consecutive_states():
    body = _body("branches", 3, ("x", "y", "z"))
    body.add_symbol("i", dace.int64)
    start = body.add_state("start", is_start_block=True)
    left, right = body.add_state("left"), body.add_state("right")
    body.add_edge(start, left, dace.InterstateEdge("i % 2 == 0"))
    body.add_edge(start, right, dace.InterstateEdge("i % 2 == 1"))
    _drain(left, "x", 0, _fill(left, "x"))
    _drain(right, "y", 1, _fill(right, "y"))
    # After the branches: z is written while x (written in the left branch) is read later
    then = body.add_state("then")
    body.add_edge(left, then, dace.InterstateEdge())
    body.add_edge(right, then, dace.InterstateEdge())
    _fill(then, "z")
    last = body.add_state_after(then, "last")
    _drain(last, "z", 2)
    conflicts = _conflicts(_kernel(body, 3))
    assert _shares(conflicts, "branches:x", "branches:y")
    assert _shares(conflicts, "branches:x", "branches:z") and _shares(conflicts, "branches:y", "branches:z")

    body = _body("consecutive", 2, ("x", "y"))
    first = body.add_state("first")
    _fill(first, "x")
    second = body.add_state_after(first, "second")
    _drain(second, "y", 0, _fill(second, "y"))
    third = body.add_state_after(second, "third")
    _drain(third, "x", 1)
    conflicts = _conflicts(_kernel(body, 2))
    assert not _shares(conflicts, "consecutive:x", "consecutive:y")


def test_value_read_in_a_loop_is_live_around_it():
    """``x`` is written before a loop and read in it: it holds values while ``y`` is used in the loop."""
    body = _body("loop", 2, ("x", "y"))
    first = body.add_state("first", is_start_block=True)
    _fill(first, "x")
    loop = dace.sdfg.state.LoopRegion("repeat", "k < 2", "k", "k = 0", "k = k + 1", sdfg=body)
    body.add_node(loop)
    body.add_edge(first, loop, dace.InterstateEdge())
    read = loop.add_state("read", is_start_block=True)
    _drain(read, "x", 0)
    use = loop.add_state_after(read, "use")
    _drain(use, "y", 1, _fill(use, "y"))
    conflicts = _conflicts(_kernel(body, 2))
    assert not _shares(conflicts, "loop:x", "loop:y")


def _hot_producer(barrier_at_end: bool) -> dace.SDFG:
    """A nested SDFG whose last state writes its scratch ``s`` with one tasklet that all threads run (no thread-block
    map, so no barrier at its end), or with a thread-block map."""
    producer = dace.SDFG("hot")
    producer.add_array("out", [32], dace.float64, storage=S.GPU_Global)
    producer.add_array("s", [KIB16], dace.float64, storage=S.GPU_Shared, transient=True)
    state = producer.add_state("write")
    outputs = {"w": dace.Memlet("s[j]"), "o": dace.Memlet("out[j]")}
    if barrier_at_end:
        state.add_mapped_tasklet("write", {"j": "0:32"}, {}, "w = 1\no = 1", outputs, schedule=TB, external_edges=True)
    else:
        tasklet = state.add_tasklet("write", {}, {"w", "o"}, "w = 1\no = 1")
        state.add_edge(tasklet, "w", state.add_write("s"), None, dace.Memlet("s[0]"))
        state.add_edge(tasklet, "o", state.add_write("out"), None, dace.Memlet("out[0]"))
    return producer


@pytest.mark.parametrize("barrier_at_end", [False, True])
def test_hot_containers_of_a_nested_sdfg(barrier_at_end: bool):
    """A nested SDFG that ends without a barrier leaves its scratch in use (other threads may still write it): a
    thread-block map that follows in the same state may not use the same memory for ``u``."""
    body = _body("after_hot", 2, ("u",))
    body.add_array("g", [32], dace.float64, storage=S.GPU_Global, transient=True)
    state = body.add_state("state")
    call = state.add_nested_sdfg(_hot_producer(barrier_at_end), set(), {"out"})
    g = state.add_access("g")
    state.add_edge(call, "out", g, None, dace.Memlet("g[0:32]"))
    state.add_mapped_tasklet(
        "use", {"j": "0:32"}, {"v": dace.Memlet("g[j]")}, "w = v", {"w": dace.Memlet("u[j]")}, schedule=TB,
        external_edges=True, input_nodes={"g": g},
    )  # fmt: skip
    u = next(n for n in state.sink_nodes() if isinstance(n, nodes.AccessNode) and n.data == "u")
    _drain(state, "u", 0, u)
    call.integrate_into_parent()
    conflicts = _conflicts(_kernel(body, 2))
    assert _shares(conflicts, "hot:s", "after_hot:u") == barrier_at_end


@pytest.mark.parametrize("read_by_condition", [False, True])
def test_container_read_by_control_flow(read_by_condition: bool):
    """A container that a condition reads holds a value until then."""
    body = _body("condition", 2, ("x", "y"))
    first = body.add_state("first", is_start_block=True)
    _fill(first, "x")
    second, done = body.add_state("second"), body.add_state("done")
    body.add_edge(first, second, dace.InterstateEdge())
    _drain(second, "y", 0, _fill(second, "y"))
    body.add_edge(second, done, dace.InterstateEdge("x[0] > 0" if read_by_condition else "1 > 0"))
    _drain(done, "y", 1)
    conflicts = _conflicts(_kernel(body, 2))
    assert _shares(conflicts, "condition:x", "condition:y") != read_by_condition


def test_zeroed_dynamic_map_state_and_thread_level_containers():
    # A container zeroed where it is allocated holds values in its whole SDFG
    for setzero in (False, True):
        body = _body("zeroed", 2, ("x", "y"))
        first = body.add_state("first")
        x = _fill(first, "x")
        x.setzero = setzero
        _drain(first, "x", 0, x)
        second = body.add_state_after(first, "second")
        _drain(second, "y", 1, _fill(second, "y"))
        assert _shares(_conflicts(_kernel(body, 2)), "zeroed:x", "zeroed:y") != setzero

    # The scheduling state of a dynamic thread-block map is in use with the containers of the map
    body = _body("dynamic", 2, ("x", "y"))
    first = body.add_state("first")
    first.add_mapped_tasklet(
        "fill", {"j": "0:32"}, {}, "w = j", {"w": dace.Memlet("x[j]")},
        schedule=dace.ScheduleType.GPU_ThreadBlock_Dynamic, external_edges=True,
    )  # fmt: skip
    second = body.add_state_after(first, "second")
    _drain(second, "x", 0)
    _drain(second, "y", 1, _fill(second, "y"))
    conflicts = _conflicts(_kernel(body, 2))
    (dynmap,) = [f"{s.name}:{n}" for s, n in conflicts.containers if n.startswith("__dace_dynmap_state")]
    assert not _shares(conflicts, dynmap, "dynamic:x") and _shares(conflicts, dynmap, "dynamic:y")

    # The containers of a nested SDFG in a thread-block map are its threads' own: they conflict with every other
    inner = dace.SDFG("thread_level")
    inner.add_array("o", [1], dace.float64)
    inner.add_array("z", [32], dace.float64, storage=S.GPU_Shared, transient=True)
    istate = inner.add_state()
    write = istate.add_tasklet("write", {}, {"w"}, "w = 1")
    read = istate.add_tasklet("read", {"v"}, {"w"}, "w = v")
    z = istate.add_access("z")
    istate.add_edge(write, "w", z, None, dace.Memlet("z[0]"))
    istate.add_edge(z, None, read, "v", dace.Memlet("z[0]"))
    istate.add_edge(read, "w", istate.add_write("o"), None, dace.Memlet("o[0]"))
    body = _body("threads", 2, ("y",))
    body.add_array("g", [32], dace.float64, storage=S.GPU_Global, transient=True)
    state = body.add_state()
    entry, exit_ = state.add_map("threads", {"j": "0:32"}, schedule=TB)
    call = state.add_nested_sdfg(inner, set(), {"o"})
    state.add_nedge(entry, call, dace.Memlet())
    state.add_memlet_path(call, exit_, state.add_write("g"), src_conn="o", memlet=dace.Memlet("g[j]"))
    call.integrate_into_parent()
    second = body.add_state_after(state, "second")
    _drain(second, "y", 1, _fill(second, "y"))
    conflicts = _conflicts(_kernel(body, 2))
    assert ("thread_level", "z") in {(s.name, n) for s, n in conflicts.pinned}
    assert not _shares(conflicts, "thread_level:z", "threads:y")


def test_kernels_the_analysis_does_not_cover():
    # Reuse disabled, and a persistent kernel: every container conflicts with every other
    assert _shares(_conflicts(_tasks_sdfg(same_state=False)), "task0:s", "task1:s")
    assert not _shares(_conflicts(_tasks_sdfg(same_state=False), reuse=False), "task0:s", "task1:s")
    with dace.config.set_temporary("compiler", "cuda", "shared_memory_reuse", value=False):
        assert not _shares(_conflicts(_tasks_sdfg(same_state=False), reuse=None), "task0:s", "task1:s")
    sdfg = _tasks_sdfg(same_state=False)
    next(
        n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.MapEntry)
    ).map.schedule = dace.ScheduleType.GPU_Persistent
    assert not _shares(_conflicts(sdfg), "task0:s", "task1:s")

    # A collaborative copy (an edge between two shared containers in the block's code)
    body = _body("collaborative", 2, ("x", "y", "z"))
    first = body.add_state("first")
    x = _fill(first, "x")
    y = first.add_access("y")
    first.add_nedge(x, y, dace.Memlet("x[0:32]"))
    _drain(first, "y", 0, y)
    second = body.add_state_after(first, "second")
    _drain(second, "z", 1, _fill(second, "z"))
    conflicts = _conflicts(_kernel(body, 2))
    assert conflicts.pinned == set(conflicts.containers)


@pytest.mark.parametrize("per_thread", [False, True])
def test_branches_on_values_of_threads(per_thread: bool):
    """A branch condition that threads of a thread-block map wrote in registers may differ between the threads of a
    block: the kernel's containers do not share memory. Written to shared memory, it is the same for all."""
    body = _body("divergent", 2, ("x", "y"))
    storage = S.Register if per_thread else S.GPU_Shared
    body.add_array("flag", [1], dace.int32, storage=storage, transient=True)
    start = body.add_state("start", is_start_block=True)
    start.add_mapped_tasklet(
        "flag", {"j": "0:1"}, {}, "w = j", {"w": dace.Memlet("flag[0]")}, schedule=TB, external_edges=True
    )
    left, right = body.add_state("left"), body.add_state("right")
    body.add_edge(start, left, dace.InterstateEdge("flag[0] > 0"))
    body.add_edge(start, right, dace.InterstateEdge("flag[0] <= 0"))
    _drain(left, "x", 0, _fill(left, "x"))
    _drain(right, "y", 1, _fill(right, "y"))
    conflicts = _conflicts(_kernel(body, 2))
    assert _shares(conflicts, "divergent:x", "divergent:y") != per_thread


if __name__ == "__main__":
    test_scratch_overlaps_an_output_that_holds_no_values()
    test_scratch_overlapping_an_output_runs(False)
    test_scratch_overlapping_an_output_runs(True)
    test_tasks_in_different_states_share_shared_memory()
    test_tasks_in_different_states_run()
    test_storage_type_attribute()
    for storage in (S.GPU_Shared, S.GPU_Shared(dynamic=True), S.GPU_Shared(dynamic=False)):
        test_storage_type_serialization(storage)
    test_fitting_containers_stay_static()
    test_overflow_is_placed_dynamically()
    test_static_and_dynamic_beyond_the_default_limit()
    test_static_containers_are_aligned()
    test_explicit_placement()
    test_inline_storage_constructor()
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
    test_static_and_dynamic_shared_memory_beyond_the_default_limit()
    test_symbolic_size()
    test_nested_sdfg_offset()
    test_kernels_with_separate_containers()
    # test_request_beyond_the_device_raises needs pytest's ``capfd`` fixture
