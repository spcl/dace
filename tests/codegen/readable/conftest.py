# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Shared fixtures + skip gates for the experimental "readable" CPU code generator.

CPU kernels are compared against the legacy generator to within 1 ULP per element and run in a forked child, so that
a crashing kernel cannot take down pytest. GPU kernels run in-process (CUDA does not survive a fork) and are compared
with a tolerance, as their reduction and atomic order is not reproducible.
"""

import functools
import os
import signal
import tempfile

# Steer Open MPI off UCX before dace's lazy MPI import can stall; setdefault defers to external config.
os.environ.setdefault("OMPI_MCA_pml", "ob1")
os.environ.setdefault("OMPI_MCA_btl", "self,vader")
os.environ.setdefault("UCX_VFS_ENABLE", "n")
# Under mpirun the MPI tests need mpi4py's automatic MPI_Init; only a plain session skips it.
if "OMPI_COMM_WORLD_SIZE" not in os.environ:
    os.environ.setdefault("MPI4PY_RC_INITIALIZE", "0")

# No thread pin here. The compare is not bit-exact any more -- ``assert_outputs_equivalent`` grades
# at the dtype tolerance, sized for the reassociation a threaded WCR costs -- so forcing the whole
# worker to one thread buys nothing and grades a build nobody ships. ``run_isolated`` still pins its
# CHILD, where the pin is load-bearing: libgomp caches OMP_NUM_THREADS in its initialiser, which a
# full-tree collection has long since run, so only the child's direct ``set_openmp_thread_count``
# actually takes. The root conftest's deliberate multi-thread count stands for everything else.

import numpy as np
import pytest

import dace
from dace.config import Config, set_temporary

#: The two CPU code generators under test.
LEGACY = "legacy"
EXPERIMENTAL = "experimental_readable"
#: Config path selecting the CPU generator implementation.
IMPLEMENTATION_KEY = ("compiler", "cpu", "implementation")

#: Config path of the CPU compiler flags.
CPU_ARGS_KEY = ("compiler", "cpu", "args")


def use_implementation(implementation):
    """Context manager pinning ``compiler.cpu.implementation`` for a codegen run."""
    return set_temporary(*IMPLEMENTATION_KEY, value=implementation)


def trivial_elementwise_sdfg(name):
    """A tiny ``b[i] = a[i] + 1`` map SDFG, built directly with the low-level API (not ``@dace.program``)."""
    sdfg = dace.SDFG(name)
    sdfg.add_array("a", [8], dace.float64)
    sdfg.add_array("b", [8], dace.float64)
    state = sdfg.add_state("main")
    read, write = state.add_read("a"), state.add_write("b")
    entry, exit_node = state.add_map("m", {"i": "0:8"})
    tasklet = state.add_tasklet("t", {"inp"}, {"out"}, "out = inp + 1.0")
    state.add_memlet_path(read, entry, tasklet, dst_conn="inp", memlet=dace.Memlet("a[i]"))
    state.add_memlet_path(tasklet, exit_node, write, src_conn="out", memlet=dace.Memlet("b[i]"))
    return sdfg


def generated_code(sdfg):
    """Concatenated generated C++ for ``sdfg`` (codegen only, no compile)."""
    return "\n".join((obj.clean_code or obj.code) for obj in sdfg.generate_code())


@functools.lru_cache(maxsize=1, typed=True)
def experimental_available():
    """True iff the readable CPU generator is wired up and its output differs from legacy.

    Nothing is swallowed here: a raising probe means the generator regressed, and that must surface
    as an error rather than as silent skips.
    """
    Config.get(*IMPLEMENTATION_KEY)
    # Same SDFG object under both configs -- avoids spurious divergence from DaCe deduplicating
    # two separately-built SDFGs' names.
    sdfg = trivial_elementwise_sdfg("readable_probe")
    with use_implementation(LEGACY):
        legacy_code = generated_code(sdfg)
    with use_implementation(EXPERIMENTAL):
        experimental_code = generated_code(sdfg)
    return experimental_code != legacy_code


def without_fma_contraction():
    """Builds without fused multiply-add contraction. The two generators nest the same computation differently, so
    the compiler contracts different multiply-adds and an accumulating kernel drifts by far more than 1 ULP."""
    return set_temporary(*CPU_ARGS_KEY, value=f"{Config.get(*CPU_ARGS_KEY)} -ffp-contract=off")


def without_simd():
    """Builds without OpenMP ``simd`` clauses: the generators place them on different loops, and a vectorized
    reduction reassociates its sum."""
    return set_temporary("compiler", "cpu", "simd_maps", value=False)


def to_host(value):
    """Return a host numpy array for ``value`` (handles cupy device arrays)."""
    if type(value).__module__.split(".")[0] == "cupy":
        import cupy

        return cupy.asnumpy(value)
    return np.asarray(value)


def waitpid_with_timeout(pid, timeout):
    """``os.waitpid`` with a SIGALRM deadline; SIGKILL the child on timeout."""

    def on_alarm(signum, frame):
        raise TimeoutError

    previous = signal.signal(signal.SIGALRM, on_alarm)
    signal.alarm(int(timeout))
    try:
        _, status = os.waitpid(pid, 0)
    except TimeoutError:
        os.kill(pid, signal.SIGKILL)
        os.waitpid(pid, 0)
        raise RuntimeError(f"isolated kernel run timed out after {timeout}s")
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)
    if not (os.WIFEXITED(status) and os.WEXITSTATUS(status) == 0):
        raise RuntimeError(f"isolated kernel run failed (wait status={status})")


def run_isolated(build_and_run, timeout=300):
    """Run ``build_and_run() -> Dict[str, ndarray]`` in a forked child process (CPU only -- CUDA and
    ``os.fork`` don't mix). A crash or timeout surfaces as a ``RuntimeError`` in the parent.

    A CPU kernel that already ran in THIS process leaves a live OpenMP thread team behind, and a
    fork from that state deadlocks the child on libgomp's team barrier (the top-level conftest
    guards ``os.fork`` against exactly this). Pausing the pools first is semantically transparent
    -- the next parallel region rebuilds the team -- and makes the fork safe.

    The child pins the thread count to 1 before it builds anything. The module-level
    ``OMP_NUM_THREADS`` write cannot do that on its own: libgomp reads the variable once, in its
    initialiser, and a full-tree collection maps libgomp long before this directory's conftest runs
    -- which is how a 4-thread team survived the pin in CI and let the two generators accumulate the
    dot products of ``polybench/lu`` in different orders. Pinning in the child rather than the
    parent keeps the root conftest's deliberate multi-thread count for every other test in the
    worker.
    """
    from dace.transformation.layout.isolation import pause_openmp_pools, set_openmp_thread_count

    pause_openmp_pools()
    handle, path = tempfile.mkstemp(suffix=".npz")
    os.close(handle)
    pid = os.fork()
    if pid == 0:  # child
        try:
            set_openmp_thread_count(1)
            outputs = build_and_run()
            np.savez(path, **{name: to_host(value) for name, value in outputs.items()})
            os._exit(0)
        except BaseException:  # noqa: BLE001 - report and exit non-zero, never raise past fork
            import traceback

            traceback.print_exc()
            os._exit(17)
    try:
        waitpid_with_timeout(pid, timeout)
        with np.load(path) as data:
            return {name: data[name] for name in data.files}
    finally:
        if os.path.exists(path):
            os.remove(path)


def tolerance_for(dtype):
    """``(rtol, atol)`` matched to precision: fp64 tight, fp32 relaxed, ints exact."""
    dt = np.dtype(dtype)
    if dt.kind in "iub":
        return 0.0, 0.0
    single = (dt.kind == "f" and dt.itemsize <= 4) or (dt.kind == "c" and dt.itemsize <= 8)
    return (1e-5, 1e-6) if single else (1e-9, 1e-11)


def max_abs_diff(legacy, experimental):
    """Max |legacy - experimental| for an error message (best effort)."""
    try:
        return float(np.nanmax(np.abs(legacy.astype(np.complex128) - experimental.astype(np.complex128))))
    except Exception:  # noqa: BLE001
        return float("nan")


def assert_max_one_ulp(legacy, experimental):
    """Raises ``AssertionError`` unless every element is within 1 ULP, NaN and inf positions included; a complex
    array is compared by its real and imaginary parts."""
    if legacy.dtype.kind == "c":
        legacy = np.stack([legacy.real, legacy.imag])
        experimental = np.stack([experimental.real, experimental.imag])
    np.testing.assert_array_max_ulp(legacy, experimental, maxulp=1)


def assert_outputs_equivalent(legacy, experimental, target, label=""):
    """Asserts the readable outputs equal the legacy ones: integers exactly and floats within 1 ULP on CPU, within
    a dtype tolerance on GPU."""
    legacy = {name: to_host(value) for name, value in legacy.items()}
    experimental = {name: to_host(value) for name, value in experimental.items()}
    assert set(legacy) == set(experimental), f"{label}: output-key mismatch {sorted(legacy)} vs {sorted(experimental)}"
    for name, lv in legacy.items():
        ev = experimental[name]
        assert lv.shape == ev.shape, f"{label}/{name}: shape {lv.shape} vs {ev.shape}"
        if target == "cpu":
            if lv.dtype.kind in "fc":
                assert_max_one_ulp(lv, ev)
            else:
                assert np.array_equal(lv, ev), f"{label}/{name}: experimental CPU codegen differs from legacy"
        else:
            rtol, atol = tolerance_for(lv.dtype)
            assert np.allclose(lv, ev, rtol=rtol, atol=atol, equal_nan=True), (
                f"{label}/{name}: experimental GPU codegen diverges from legacy, max|diff|={max_abs_diff(lv, ev):.3e}"
            )


# #
# Fixtures
# #
@pytest.fixture
def require_experimental():
    """Assert the readable generator is wired up; it is required, not optional."""
    assert experimental_available(), (
        "the readable CPU generator produced byte-identical output to legacy -- it is not wired up"
    )


@pytest.fixture(
    params=[
        pytest.param("cpu", id="cpu"),
        pytest.param("gpu", id="gpu", marks=pytest.mark.gpu),
    ]
)
def target(request):
    """Codegen target. The GPU variant carries ``@pytest.mark.gpu`` (select with ``-m gpu``)."""
    return request.param


@pytest.fixture(params=[LEGACY, EXPERIMENTAL])
def codegen_variant(request):
    """A single CPU generator implementation (unused by the equivalence tests here, which drive both)."""
    return request.param
