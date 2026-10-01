# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Shared helpers for the readable CPU code generator tests.

CPU kernels are compared bit-exactly against the legacy generator and run in a forked child, so that a crashing
kernel cannot take down pytest. GPU kernels run in-process (CUDA does not survive a fork) and are compared with a
tolerance, as their reduction and atomic order is not reproducible.
"""
import copy
import functools
import os
import shutil
import signal
import subprocess
import tempfile

# dace imports mpi4py lazily, which calls MPI_Init; keep Open MPI off UCX so that it cannot stall
os.environ.setdefault("OMPI_MCA_pml", "ob1")
os.environ.setdefault("OMPI_MCA_btl", "self,vader")
os.environ.setdefault("UCX_VFS_ENABLE", "n")
os.environ.setdefault("MPI4PY_RC_INITIALIZE", "0")

# One thread keeps the summation order of reductions fixed, which the bit-exact comparison needs
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pytest

import dace

from dace.config import set_temporary

#: The two CPU code generators under test.
LEGACY = "legacy"
EXPERIMENTAL = "experimental_readable"
#: Config path selecting the CPU generator implementation.
IMPLEMENTATION_KEY = ("compiler", "cpu", "implementation")


def use_implementation(implementation):
    """Pins ``compiler.cpu.implementation`` for a code generation run."""
    return set_temporary(*IMPLEMENTATION_KEY, value=implementation)


@functools.lru_cache(maxsize=1, typed=True)
def gpu_available():
    """Whether a CUDA device is usable."""
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
    """A host array for ``value``, which may be a cupy array."""
    if type(value).__module__.split(".")[0] == "cupy":
        import cupy
        return cupy.asnumpy(value)
    return np.asarray(value)


def waitpid_with_timeout(pid, timeout):
    """``os.waitpid`` that kills the child after ``timeout`` seconds."""

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
    """Runs ``build_and_run() -> dict[str, ndarray]`` in a forked child, which returns its arrays through a
    temporary ``.npz``. A crash or timeout raises ``RuntimeError``."""
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
    """``(rtol, atol)`` for a dtype."""
    dt = np.dtype(dtype)
    if dt.kind in "iub":
        return 0.0, 0.0
    single = (dt.kind == "f" and dt.itemsize <= 4) or (dt.kind == "c" and dt.itemsize <= 8)
    return (1e-5, 1e-6) if single else (1e-9, 1e-11)


def max_abs_diff(legacy, experimental):
    """The largest absolute difference, for the failure message."""
    try:
        return float(np.nanmax(np.abs(legacy.astype(np.complex128) - experimental.astype(np.complex128))))
    except Exception:  # noqa: BLE001
        return float("nan")


def assert_outputs_equivalent(legacy, experimental, target, label=""):
    """Asserts the readable outputs equal the legacy ones: exactly on CPU, within a dtype tolerance on GPU."""
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
    """The code generation target; the GPU variant is marked ``gpu`` and skipped without a device."""
    if request.param == "gpu" and not gpu_available():
        pytest.skip("no CUDA-capable GPU available")
    return request.param


def heap_pipeline_1d(name, shape, rng, alignment=0):
    """``T[i] = A[i] + 1`` then ``B[i] = T[i] * 2`` over ``rng``, with ``T`` a heap transient of ``shape``."""
    sdfg = dace.SDFG(name)
    sdfg.add_array('A', [shape], dace.float64)
    sdfg.add_array('B', [shape], dace.float64)
    sdfg.add_transient('T', [shape], dace.float64, storage=dace.StorageType.CPU_Heap, alignment=alignment)

    write = sdfg.add_state('write')
    read_a, write_t = write.add_read('A'), write.add_write('T')
    entry, exit_node = write.add_map('write_map', {'i': rng})
    add = write.add_tasklet('add_one', {'a'}, {'o'}, 'o = a + 1.0')
    write.add_memlet_path(read_a, entry, add, dst_conn='a', memlet=dace.Memlet('A[i]'))
    write.add_memlet_path(add, exit_node, write_t, src_conn='o', memlet=dace.Memlet('T[i]'))

    read = sdfg.add_state_after(write, 'read')
    read_t, write_b = read.add_read('T'), read.add_write('B')
    entry2, exit2 = read.add_map('read_map', {'i': rng})
    mul = read.add_tasklet('mul_two', {'t'}, {'o'}, 'o = t * 2.0')
    read.add_memlet_path(read_t, entry2, mul, dst_conn='t', memlet=dace.Memlet('T[i]'))
    read.add_memlet_path(mul, exit2, write_b, src_conn='o', memlet=dace.Memlet('B[i]'))

    sdfg.validate()
    return sdfg


def run_variant(build, name, implementation, base, target='cpu'):
    """Builds and runs one variant on a copy of ``base`` and returns its arrays. CPU runs in a forked child, GPU
    in-process."""

    def work():
        sdfg = build(name)
        if target == 'gpu':
            sdfg.apply_gpu_transformations()
        arrays = copy.deepcopy(base)
        sdfg.compile()(**arrays)
        return {key: value for key, value in arrays.items() if isinstance(value, np.ndarray)}

    with use_implementation(implementation):
        return work() if target == 'gpu' else run_isolated(work)


def assert_bit_exact(build, base_name, base):
    """Asserts that both generators give bit-identical outputs and returns them as (legacy, readable)."""
    legacy = run_variant(build, base_name + '_legacy', LEGACY, base)
    experimental = run_variant(build, base_name + '_experimental', EXPERIMENTAL, base)
    assert set(legacy) == set(experimental)
    for key in legacy:
        assert np.array_equal(legacy[key], experimental[key]), f'{base_name}: output {key} is not bit-exact'
    return legacy, experimental


def experimental_code(build, name):
    """The generated C++ of the host code object under the readable generator."""
    with use_implementation(EXPERIMENTAL):
        return build(name).generate_code()[0].clean_code


def generated_for(build, name, implementation, gpu=False):
    """All generated code of ``build`` under ``implementation``."""
    with use_implementation(implementation):
        sdfg = build(name)
        if gpu:
            sdfg.apply_gpu_transformations()
        return '\n'.join(obj.clean_code or obj.code for obj in sdfg.generate_code())
