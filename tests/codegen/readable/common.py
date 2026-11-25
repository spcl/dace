# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Helpers shared by the tests of the readable CPU code generator: pinning a generator, running a build in a
process of its own, and comparing its outputs with the legacy generator's."""

import ctypes
import os
import signal
import tempfile

# dace lazily ``from mpi4py import MPI`` inside ``to_sdfg`` (auto-calls
# ``MPI_Init``); steer Open MPI off UCX before that import so it cannot stall.
# ``setdefault`` defers to any externally-provided configuration.
os.environ.setdefault("OMPI_MCA_pml", "ob1")
os.environ.setdefault("OMPI_MCA_btl", "self,vader")
os.environ.setdefault("UCX_VFS_ENABLE", "n")
# Under mpirun the MPI tests need mpi4py's automatic MPI_Init; only a plain session skips it.
if "OMPI_COMM_WORLD_SIZE" not in os.environ:
    os.environ.setdefault("MPI4PY_RC_INITIALIZE", "0")

# Pin a single OpenMP thread so parallel reductions accumulate in a deterministic order. The
# suite compares the legacy and experimental generators BIT-EXACTLY on CPU; with more than one
# thread a reduction's summation order is non-deterministic and the two runs (both multi-threaded)
# would differ by rounding on FP-heavy kernels -- a thread-scheduling artifact, not a code-generator
# difference. ``setdefault`` defers to an externally-provided value.
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pytest

from dace.config import set_temporary

#: The two CPU code generators under test.
LEGACY = "legacy"
EXPERIMENTAL = "experimental_readable"
#: Config path selecting the CPU generator implementation.
IMPLEMENTATION_KEY = ("compiler", "cpu", "implementation")
#: Codegen targets of the equivalence tests; the GPU one needs ``-m gpu``.
TARGETS = ["cpu", pytest.param("gpu", marks=pytest.mark.gpu)]


def use_implementation(implementation):
    """Context manager pinning ``compiler.cpu.implementation`` for a codegen run."""
    return set_temporary(*IMPLEMENTATION_KEY, value=implementation)


def generated_code(sdfg):
    """Concatenated generated C++ for ``sdfg`` (codegen only, no compile)."""
    return "\n".join((obj.clean_code or obj.code) for obj in sdfg.generate_code())


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


def pause_openmp_pools():
    """Tear down the thread pool of every OpenMP runtime already loaded: a team the parent started does not survive a
    fork, so the child's first parallel region would wait on it forever."""
    for soname in ("libgomp.so.1", "libomp.so.5", "libomp.so", "libiomp5.so"):
        try:
            runtime = ctypes.CDLL(soname, mode=os.RTLD_NOLOAD)
        except OSError:
            continue
        runtime.omp_pause_resource_all(1)  # omp_pause_soft


def run_isolated(build_and_run, timeout=300):
    """Run ``build_and_run() -> Dict[str, ndarray]`` in a forked child process.

    Repo rule: always fork when running compiled kernels -- an experimental
    kernel that segfaults must not take down the pytest process. The child
    marshals its output arrays through a temporary ``.npz``; a crash or timeout
    surfaces as a ``RuntimeError`` in the parent.

    Reserved for CPU runs. CUDA and ``os.fork`` are incompatible (a CUDA context
    initialized in the parent is unusable after fork), so GPU cases run in-process
    (see the test drivers).
    """
    handle, path = tempfile.mkstemp(suffix=".npz")
    os.close(handle)
    pause_openmp_pools()
    pid = os.fork()
    if pid == 0:  # child
        try:
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


def assert_outputs_equivalent(legacy, experimental, target, label=""):
    """Assert the readable generator reproduced the legacy outputs.

    On CPU the two runs use the same SDFG, host compiler and inputs, so a
    deterministic legacy result must be reproduced BIT-EXACTLY (the repo rule:
    treat any discrepancy as a real bug, not a tolerance question). On GPU, where
    reduction/atomic ordering is not reproducible, compare with a tight dtype-
    aware tolerance.
    """
    legacy = {name: to_host(value) for name, value in legacy.items()}
    experimental = {name: to_host(value) for name, value in experimental.items()}
    assert set(legacy) == set(experimental), f"{label}: output-key mismatch {sorted(legacy)} vs {sorted(experimental)}"
    for name, lv in legacy.items():
        ev = experimental[name]
        assert lv.shape == ev.shape, f"{label}/{name}: shape {lv.shape} vs {ev.shape}"
        if target == "cpu":
            equal = np.array_equal(lv, ev, equal_nan=True) if lv.dtype.kind == "f" else np.array_equal(lv, ev)
            assert equal, (
                f"{label}/{name}: experimental CPU codegen is not bit-exact vs legacy, "
                f"max|diff|={max_abs_diff(lv, ev):.3e}"
            )
        else:
            rtol, atol = tolerance_for(lv.dtype)
            assert np.allclose(lv, ev, rtol=rtol, atol=atol, equal_nan=True), (
                f"{label}/{name}: experimental GPU codegen diverges from legacy, max|diff|={max_abs_diff(lv, ev):.3e}"
            )
