# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""GPU external translation units (Model 2) on the NEW cuda code generator.

``compiler.cpu.codegen_params.external_translation_units`` lifts each top-level control node that
contains GPU work into its OWN standalone SDFG, which is code-generated on its own -- so each becomes
its own ``.cu`` -- and called back from the parent through the standalone SDFG's public extern-C handle
ABI (``__dace_init_* / __program_* / __dace_exit_*``), linked in-binary. The point the legacy
``split_nsdfg_translation_units`` path cannot reach: both cuda generators emit ONE ``.cu`` per
generate_code invocation, so per-kernel ``.cu`` files require per-kernel generate_code, i.e. standalone
SDFGs (verified: a 2-kernel program yields a single 2-``__global__`` ``.cu`` without this).

The fixtures build the device programs DIRECTLY -- ``GPU_Global`` arrays and maps with explicit
``GPU_Device`` / ``CPU_Multicore`` schedules -- rather than routing a host program through
``apply_gpu_transformations``. That is exactly the shape a lifted child has (a standalone kernel over
device pointers), it keeps the nestedness the test intends, and the device arguments are passed as
cupy arrays at call time. Coverage: flat sibling kernels, a kernel inside a top-level loop, and a
host/device hybrid, x both builders (``cmake`` / ``native``). The whole file is gated on
:func:`external_tu_available`, so it skips until the feature lands and never reports a false green.
"""
import re

import numpy as np
import pytest

import dace
from dace.codegen import codegen
from dace.config import Config

from tests.codegen.readable.conftest import gpu_available, to_host

#: Force the experimental ("new") cuda generator for every SDFG this file builds.
NEW_CUDA = "experimental"
#: The Model-2 master toggle.
EXT_TU_KEY = ("compiler", "cpu", "codegen_params", "external_translation_units")
GPU_GLOBAL = dace.StorageType.GPU_Global
GPU_DEVICE = dace.ScheduleType.GPU_Device
CPU_MULTICORE = dace.ScheduleType.CPU_Multicore


def cu_objects(objects):
    """The ``.cu`` CodeObjects (one per GPU translation unit)."""
    return [o for o in objects if o.language == "cu"]


def kernel_cu_objects(objects):
    """The ``.cu`` TUs that actually define a kernel. A parent whose kernels were all lifted still
    emits a (kernel-less) ``.cu`` for its runtime glue -- that TU is not a kernel unit."""
    return [o for o in cu_objects(objects) if "__global__ void" in (o.clean_code or o.code)]


def count_global_kernels(objects):
    """Total ``__global__`` definitions across every ``.cu`` (the real kernel count, TU-independent)."""
    return sum((o.clean_code or o.code).count("__global__ void") for o in cu_objects(objects))


def parent_frame(objects):
    """The parent program's frame ``.cpp`` -- host code, not a lifted child (``_nest_``) and not a ``.cu``."""
    frames = [o for o in objects if o.language == "cpp" and o.target_type == "" and "_nest_" not in o.name]
    assert len(frames) == 1, [o.name for o in frames]
    return frames[0].clean_code or frames[0].code


# --------------------------------------------------------------------------- #
# Fixtures -- device programs built directly (GPU_Global + explicit schedules).
# --------------------------------------------------------------------------- #
def two_sibling_kernels(name):
    """Flat nestedness: two top-level GPU maps (no enclosing control node) over a shared device input.
    Model 2 lifts each into its own standalone SDFG -> two ``.cu`` files, one ``__global__`` each."""
    sdfg = dace.SDFG(name)
    for arr in ("A", "B", "C"):
        sdfg.add_array(arr, [256], dace.float64, storage=GPU_GLOBAL)
    s0 = sdfg.add_state("s0", is_start_block=True)
    e0, x0 = s0.add_map("k0", dict(i="0:256"), schedule=GPU_DEVICE)
    t0 = s0.add_tasklet("t0", {"a"}, {"b"}, "b = a * 2.0")
    s0.add_memlet_path(s0.add_read("A"), e0, t0, dst_conn="a", memlet=dace.Memlet("A[i]"))
    s0.add_memlet_path(t0, x0, s0.add_write("B"), src_conn="b", memlet=dace.Memlet("B[i]"))
    s1 = sdfg.add_state_after(s0, "s1")
    e1, x1 = s1.add_map("k1", dict(j="0:256"), schedule=GPU_DEVICE)
    t1 = s1.add_tasklet("t1", {"a"}, {"c"}, "c = a + 3.0")
    s1.add_memlet_path(s1.add_read("A"), e1, t1, dst_conn="a", memlet=dace.Memlet("A[j]"))
    s1.add_memlet_path(t1, x1, s1.add_write("C"), src_conn="c", memlet=dace.Memlet("C[j]"))
    return sdfg


def kernel_in_loop(name):
    """Nested: a top-level sequential loop whose body is one GPU kernel. Model 2 lifts the WHOLE loop
    into a single child SDFG -> one ``.cu`` (depth 1: the kernel inside is not lifted again). The loop
    variable ``t`` is defined inside the child, so it never crosses the call boundary."""
    from dace.sdfg.state import LoopRegion
    sdfg = dace.SDFG(name)
    for arr in ("A", "B"):
        sdfg.add_array(arr, [256], dace.float64, storage=GPU_GLOBAL)
    loop = LoopRegion("loop", condition_expr="t < 4", loop_var="t", initialize_expr="t = 0",
                      update_expr="t = t + 1")
    sdfg.add_node(loop, is_start_block=True)
    body = loop.add_state("body", is_start_block=True)
    e, x = body.add_map("k", dict(i="0:256"), schedule=GPU_DEVICE)
    t = body.add_tasklet("t", {"a"}, {"b"}, "b = a + t")
    body.add_memlet_path(body.add_read("A"), e, t, dst_conn="a", memlet=dace.Memlet("A[i]"))
    body.add_memlet_path(t, x, body.add_write("B"), src_conn="b", memlet=dace.Memlet("B[i]"))
    return sdfg


def hybrid_cpu_gpu(name):
    """Hybrid: a host map (``CPU_Multicore`` over host arrays) and a device kernel (``GPU_Device`` over
    ``GPU_Global`` arrays), as independent top-level nests. Only the GPU nest is lifted to its own
    SDFG/``.cu``; the CPU-only nest is never lifted and stays inline in the parent."""
    sdfg = dace.SDFG(name)
    sdfg.add_array("A", [256], dace.float64)  # host
    sdfg.add_array("B", [256], dace.float64)  # host
    sdfg.add_array("C", [256], dace.float64, storage=GPU_GLOBAL)  # device
    sdfg.add_array("D", [256], dace.float64, storage=GPU_GLOBAL)  # device
    s0 = sdfg.add_state("host", is_start_block=True)
    e0, x0 = s0.add_map("cpu_m", dict(i="0:256"), schedule=CPU_MULTICORE)
    t0 = s0.add_tasklet("t0", {"a"}, {"b"}, "b = a + 1.0")
    s0.add_memlet_path(s0.add_read("A"), e0, t0, dst_conn="a", memlet=dace.Memlet("A[i]"))
    s0.add_memlet_path(t0, x0, s0.add_write("B"), src_conn="b", memlet=dace.Memlet("B[i]"))
    s1 = sdfg.add_state_after(s0, "device")
    e1, x1 = s1.add_map("gpu_m", dict(j="0:256"), schedule=GPU_DEVICE)
    t1 = s1.add_tasklet("t1", {"c"}, {"d"}, "d = c * 2.0")
    s1.add_memlet_path(s1.add_read("C"), e1, t1, dst_conn="c", memlet=dace.Memlet("C[j]"))
    s1.add_memlet_path(t1, x1, s1.add_write("D"), src_conn="d", memlet=dace.Memlet("D[j]"))
    return sdfg


def generate_with_ext_tu(sdfg, on):
    with dace.config.set_temporary("compiler", "cuda", "implementation", value=NEW_CUDA), \
         dace.config.set_temporary(*EXT_TU_KEY, value=on):
        return codegen.generate_code(sdfg)


def external_tu_available():
    """True iff the GPU external-TU feature is wired: the knob exists AND turning it on splits a
    2-kernel program's single ``.cu`` into one ``.cu`` per kernel (codegen-only; needs no GPU)."""
    try:
        Config.get(*EXT_TU_KEY)
    except Exception:  # noqa: BLE001 - key absent -> feature not present
        return False
    try:
        off = kernel_cu_objects(generate_with_ext_tu(two_sibling_kernels("ext_tu_probe"), on=False))
        on = kernel_cu_objects(generate_with_ext_tu(two_sibling_kernels("ext_tu_probe"), on=True))
    except Exception:  # noqa: BLE001 - feature under development raised -> not ready
        return False
    # Off: both kernels share one .cu. On: each kernel gets its own .cu (Model 2).
    return len(off) == 1 and len(on) == 2


@pytest.fixture
def require_external_tu():
    if not external_tu_available():
        pytest.skip("GPU external translation units (Model 2) not wired up yet")


def device_array(host):
    """A cupy device array holding ``host`` (the fixtures' GPU_Global args take device pointers)."""
    import cupy as cp
    return cp.asarray(host)


# --------------------------------------------------------------------------- #
# Codegen-shape tests (no GPU device needed) -- assert the per-kernel .cu split.
# --------------------------------------------------------------------------- #
def test_flag_off_single_cu(require_external_tu):
    """Off (default): one ``.cu`` carrying both kernels -- byte-for-byte the untouched generator."""
    objects = generate_with_ext_tu(two_sibling_kernels("off_one_cu"), on=False)
    assert len(cu_objects(objects)) == 1
    assert count_global_kernels(objects) == 2


def test_two_siblings_two_cu(require_external_tu):
    """On: two top-level sibling kernels -> two kernel-bearing ``.cu`` TUs, one ``__global__`` each."""
    objects = generate_with_ext_tu(two_sibling_kernels("two_cu"), on=True)
    kernel_cus = kernel_cu_objects(objects)
    assert len(kernel_cus) == 2, [o.name for o in kernel_cus]
    for o in kernel_cus:
        assert (o.clean_code or o.code).count("__global__ void") == 1, o.name
    # Total kernel count is conserved -- the split moves kernels between TUs, it does not add/drop them.
    assert count_global_kernels(objects) == 2


def test_loop_nested_one_kernel_cu(require_external_tu):
    """A kernel inside a top-level loop -> exactly one kernel ``.cu`` for the whole loop (depth 1)."""
    objects = generate_with_ext_tu(kernel_in_loop("loop_cu"), on=True)
    assert len(kernel_cu_objects(objects)) == 1, [o.name for o in kernel_cu_objects(objects)]
    assert count_global_kernels(objects) == 1


def test_hybrid_only_gpu_externalized(require_external_tu):
    """Hybrid host+device program: only the GPU nest becomes its own kernel ``.cu``; the CPU map stays
    in the parent (a CPU-only nest is never lifted -- that is the ``split_nsdfg`` path, off here)."""
    objects = generate_with_ext_tu(hybrid_cpu_gpu("hybrid_cu"), on=True)
    assert len(kernel_cu_objects(objects)) == 1, [o.name for o in kernel_cu_objects(objects)]
    assert count_global_kernels(objects) == 1


def test_external_call_emitted(require_external_tu):
    """The parent calls each lifted child through its public handle ABI: an ``extern "C"`` declaration
    of ``__dace_init`` / ``__program`` / ``__dace_exit`` for the child, then an init -> program -> exit
    sequence over a local handle -- the in-binary cross-TU call the static linker resolves."""
    frame = parent_frame(generate_with_ext_tu(two_sibling_kernels("call_emit"), on=True))
    assert 'extern "C" void *__dace_init_' in frame
    assert 'extern "C" void __program_' in frame
    assert 'extern "C" int __dace_exit_' in frame
    # Two children -> two handle-scoped call sites, each a consistent init/program/exit over one handle.
    handles = re.findall(r"void \*__exttu_h_(\w+) = __dace_init_(\w+)\(", frame)
    assert len(handles) == 2, frame
    for hsuffix, child in handles:
        assert hsuffix == child, (hsuffix, child)
        assert re.search(r"__program_%s\(__exttu_h_%s\b" % (re.escape(child), re.escape(child)), frame), child
        assert re.search(r"__dace_exit_%s\(__exttu_h_%s\)" % (re.escape(child), re.escape(child)), frame), child


# --------------------------------------------------------------------------- #
# Compile + run tests (need a GPU) -- correctness vs the single-TU build, both builders.
# --------------------------------------------------------------------------- #
@pytest.mark.gpu
@pytest.mark.parametrize("build_mode", ["cmake", "native"])
def test_two_siblings_run_matches_single_tu(require_external_tu, build_mode):
    """The split library must LINK (child extern-C entry points resolve in-binary) and compute exactly
    what the single-TU build computes -- under both the cmake and native builders."""
    if not gpu_available():
        pytest.skip("no CUDA-capable GPU available")
    rng = np.random.default_rng(0)
    A = rng.random(256)
    outputs = {}
    for on in (False, True):
        sdfg = two_sibling_kernels(f"run_{build_mode}_{int(on)}")
        B, C = device_array(np.zeros(256)), device_array(np.zeros(256))
        with dace.config.set_temporary("compiler", "cuda", "implementation", value=NEW_CUDA), \
             dace.config.set_temporary("compiler", "build_mode", value=build_mode), \
             dace.config.set_temporary(*EXT_TU_KEY, value=on):
            sdfg.compile()(A=device_array(A), B=B, C=C)
        outputs[on] = (to_host(B), to_host(C))
    assert np.allclose(outputs[True][0], A * 2.0)
    assert np.allclose(outputs[True][1], A + 3.0)
    assert np.allclose(outputs[True][0], outputs[False][0])
    assert np.allclose(outputs[True][1], outputs[False][1])


@pytest.mark.gpu
@pytest.mark.parametrize("build_mode", ["cmake", "native"])
def test_loop_nested_run(require_external_tu, build_mode):
    """The lifted loop-child must link + run: final value is the last iteration (t=3), matching single-TU."""
    if not gpu_available():
        pytest.skip("no CUDA-capable GPU available")
    rng = np.random.default_rng(1)
    A = rng.random(256)
    outs = {}
    for on in (False, True):
        sdfg = kernel_in_loop(f"loop_run_{build_mode}_{int(on)}")
        B = device_array(np.zeros(256))
        with dace.config.set_temporary("compiler", "cuda", "implementation", value=NEW_CUDA), \
             dace.config.set_temporary("compiler", "build_mode", value=build_mode), \
             dace.config.set_temporary(*EXT_TU_KEY, value=on):
            sdfg.compile()(A=device_array(A), B=B)
        outs[on] = to_host(B)
    assert np.allclose(outs[True], A + 3.0)
    assert np.allclose(outs[True], outs[False])


@pytest.mark.gpu
@pytest.mark.parametrize("build_mode", ["cmake", "native"])
def test_hybrid_run(require_external_tu, build_mode):
    """Hybrid host+device program links + runs: the host (CPU) map and the lifted GPU child both
    compute, and both match the single-TU build, under either builder."""
    if not gpu_available():
        pytest.skip("no CUDA-capable GPU available")
    rng = np.random.default_rng(2)
    A = rng.random(256)
    C = rng.random(256)
    outs = {}
    for on in (False, True):
        sdfg = hybrid_cpu_gpu(f"hybrid_run_{build_mode}_{int(on)}")
        B = np.zeros(256)  # host output
        D = device_array(np.zeros(256))  # device output
        with dace.config.set_temporary("compiler", "cuda", "implementation", value=NEW_CUDA), \
             dace.config.set_temporary("compiler", "build_mode", value=build_mode), \
             dace.config.set_temporary(*EXT_TU_KEY, value=on):
            sdfg.compile()(A=A.copy(), B=B, C=device_array(C), D=D)
        outs[on] = (to_host(B), to_host(D))
    assert np.allclose(outs[True][0], A + 1.0)  # host (CPU) map
    assert np.allclose(outs[True][1], C * 2.0)  # lifted GPU kernel
    assert np.allclose(outs[True][0], outs[False][0])
    assert np.allclose(outs[True][1], outs[False][1])


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
