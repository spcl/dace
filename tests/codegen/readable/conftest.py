# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Shared fixtures + skip gates for the experimental "readable" CPU code generator.

This suite proves that the experimental generator
(``compiler.cpu.implementation = experimental``) is numerically equivalent to the
``legacy`` generator across the npbench, polybench, tsvc and tsvc_2_5 corpora, on
CPU and GPU.

The readable generator is developed in a parallel task and is not ready yet.
:func:`experimental_available` gates the whole suite: until the generator is
wired up it returns ``False`` and every test skips, so
``pytest tests/codegen/readable`` is green today and flips on automatically once
the feature lands.

The gate is deliberately stronger than "did it raise": today the code generator
ignores the ``implementation`` key and silently falls back to legacy (byte-
identical output), so an exception-only probe would wrongly report the feature as
ready. The probe therefore also requires the experimental output to *differ* from
legacy on a trivial SDFG (the readable form emits per-array ``_idx`` index
functions and ``const``/``constexpr`` initialization, which legacy never does).
"""
import functools
import os
import shutil
import signal
import subprocess
import tempfile

# dace lazily ``from mpi4py import MPI`` inside ``to_sdfg`` (auto-calls
# ``MPI_Init``); steer Open MPI off UCX before that import so it cannot stall.
# ``setdefault`` defers to any externally-provided configuration.
os.environ.setdefault("OMPI_MCA_pml", "ob1")
os.environ.setdefault("OMPI_MCA_btl", "self,vader")
os.environ.setdefault("UCX_VFS_ENABLE", "n")
os.environ.setdefault("MPI4PY_RC_INITIALIZE", "0")

# Pin a single OpenMP thread so parallel reductions accumulate in a deterministic order. The
# suite compares the legacy and experimental generators BIT-EXACTLY on CPU; with more than one
# thread a reduction's summation order is non-deterministic and the two runs (both multi-threaded)
# would differ by rounding on FP-heavy kernels -- a thread-scheduling artifact, not a code-generator
# difference. ``setdefault`` defers to an externally-provided value.
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pytest

import dace
from dace.config import Config, set_temporary

#: The two CPU code generators under test.
LEGACY = "legacy"
EXPERIMENTAL = "experimental_readable"
#: Config path selecting the CPU generator implementation.
IMPLEMENTATION_KEY = ("compiler", "cpu", "implementation")


def use_implementation(implementation):
    """Context manager pinning ``compiler.cpu.implementation`` for a codegen run."""
    return set_temporary(*IMPLEMENTATION_KEY, value=implementation)


def trivial_elementwise_sdfg(name):
    """A tiny ``b[i] = a[i] + 1`` map SDFG, built with the low-level API.

    Constructed directly (not via ``@dace.program``) so the probe is self-
    contained and works from any interpreter.
    """
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


@functools.lru_cache(maxsize=1)
def experimental_available():
    """True iff the experimental readable CPU generator is wired up and active.

    All of the following must hold, else the feature is "not ready" and the whole
    suite skips:

    1. the ``compiler.cpu.implementation`` config key exists;
    2. code generation under ``experimental`` does not raise on a trivial SDFG;
    3. that generated code DIFFERS from the ``legacy`` output for the same SDFG
       (a silent legacy fallback -- today's state -- produces identical bytes).
    """
    try:
        Config.get(*IMPLEMENTATION_KEY)
    except Exception:  # noqa: BLE001 - key absent -> feature not present
        return False
    try:
        # One SDFG object generated under each config: the ONLY difference is the
        # generator, so this is deterministic (unlike two separately-built SDFGs,
        # whose names DaCe may deduplicate, spuriously diverging the output).
        sdfg = trivial_elementwise_sdfg("readable_probe")
        with use_implementation(LEGACY):
            legacy_code = generated_code(sdfg)
        with use_implementation(EXPERIMENTAL):
            experimental_code = generated_code(sdfg)
    except Exception:  # noqa: BLE001 - generator under development raised -> not ready
        return False
    return experimental_code != legacy_code


@functools.lru_cache(maxsize=1)
def gpu_available():
    """True iff a CUDA device is usable (cupy device count, else ``nvidia-smi -L``)."""
    try:
        import cupy
        return cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:  # noqa: BLE001 - cupy missing / no driver
        pass
    smi = shutil.which("nvidia-smi")
    if not smi:
        return False
    try:
        proc = subprocess.run([smi, "-L"], capture_output=True, text=True, timeout=15)
        return proc.returncode == 0 and "GPU" in proc.stdout
    except Exception:  # noqa: BLE001
        return False


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
    assert set(legacy) == set(experimental), (f"{label}: output-key mismatch "
                                              f"{sorted(legacy)} vs {sorted(experimental)}")
    for name, lv in legacy.items():
        ev = experimental[name]
        assert lv.shape == ev.shape, f"{label}/{name}: shape {lv.shape} vs {ev.shape}"
        if target == "cpu":
            equal = np.array_equal(lv, ev, equal_nan=True) if lv.dtype.kind == "f" else np.array_equal(lv, ev)
            assert equal, (f"{label}/{name}: experimental CPU codegen is not bit-exact vs legacy, "
                           f"max|diff|={max_abs_diff(lv, ev):.3e}")
        else:
            rtol, atol = tolerance_for(lv.dtype)
            assert np.allclose(lv, ev, rtol=rtol, atol=atol,
                               equal_nan=True), (f"{label}/{name}: experimental GPU codegen diverges from legacy, "
                                                 f"max|diff|={max_abs_diff(lv, ev):.3e}")


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture
def require_experimental():
    """Skip the test unless the readable generator is wired up (see the gate)."""
    if not experimental_available():
        pytest.skip("experimental readable codegen not ready")


@pytest.fixture
def require_gpu():
    """Skip the test unless a CUDA device is present."""
    if not gpu_available():
        pytest.skip("no CUDA-capable GPU available")


@pytest.fixture(params=[
    pytest.param("cpu", id="cpu"),
    pytest.param("gpu", id="gpu", marks=pytest.mark.gpu),
])
def target(request):
    """Codegen target. The GPU variant carries ``@pytest.mark.gpu`` (select with
    ``-m gpu``) and additionally skips when no CUDA device is available."""
    if request.param == "gpu" and not gpu_available():
        pytest.skip("no CUDA-capable GPU available")
    return request.param


@pytest.fixture(params=[LEGACY, EXPERIMENTAL])
def codegen_variant(request):
    """A single CPU generator implementation. Available for future single-variant
    tests; the equivalence tests here drive both variants within one test and
    compare, so they do not consume this fixture."""
    return request.param
