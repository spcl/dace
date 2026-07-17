# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Shared fixtures + skip gates for the experimental "readable" CPU code generator.

Skips the whole suite until the generator is wired up and its output differs from legacy.
"""
import functools
import os
import shutil
import signal
import subprocess
import tempfile

# Steer Open MPI off UCX before dace's lazy MPI import can stall; setdefault defers to external config.
os.environ.setdefault("OMPI_MCA_pml", "ob1")
os.environ.setdefault("OMPI_MCA_btl", "self,vader")
os.environ.setdefault("UCX_VFS_ENABLE", "n")
os.environ.setdefault("MPI4PY_RC_INITIALIZE", "0")

# Pin 1 OpenMP thread: reduction order must be deterministic for the bit-exact legacy-vs-experimental compare.
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


@functools.lru_cache(maxsize=1)
def experimental_available():
    """True iff the experimental readable CPU generator is wired up and its output differs from legacy."""
    try:
        Config.get(*IMPLEMENTATION_KEY)
    except Exception:  # noqa: BLE001 - key absent -> feature not present
        return False
    try:
        # Same SDFG object under both configs -- avoids spurious divergence from DaCe deduplicating
        # two separately-built SDFGs' names.
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
    """Run ``build_and_run() -> Dict[str, ndarray]`` in a forked child process (CPU only -- CUDA and
    ``os.fork`` don't mix). A crash or timeout surfaces as a ``RuntimeError`` in the parent."""
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
    """Assert the readable generator reproduced the legacy outputs (dtype-aware tolerance; exact for ints)."""
    legacy = {name: to_host(value) for name, value in legacy.items()}
    experimental = {name: to_host(value) for name, value in experimental.items()}
    assert set(legacy) == set(experimental), (f"{label}: output-key mismatch "
                                              f"{sorted(legacy)} vs {sorted(experimental)}")
    for name, lv in legacy.items():
        ev = experimental[name]
        assert lv.shape == ev.shape, f"{label}/{name}: shape {lv.shape} vs {ev.shape}"
        rtol, atol = tolerance_for(lv.dtype)
        assert np.allclose(lv, ev, rtol=rtol, atol=atol,
                           equal_nan=True), (f"{label}/{name}: experimental {target} codegen diverges from legacy, "
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
    """Codegen target ("cpu"/"gpu"); the GPU variant carries ``@pytest.mark.gpu`` and skips without CUDA."""
    if request.param == "gpu" and not gpu_available():
        pytest.skip("no CUDA-capable GPU available")
    return request.param


@pytest.fixture(params=[LEGACY, EXPERIMENTAL])
def codegen_variant(request):
    """A single CPU generator implementation (unused by the equivalence tests here, which drive both)."""
    return request.param
