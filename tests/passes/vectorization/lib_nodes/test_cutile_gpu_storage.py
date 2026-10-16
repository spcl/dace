# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for ``VectorizeCuTile(use_gpu_storage=True)``.

The knob marks non-transient arrays ``GPU_Global`` (via ``auto_optimize``'s
``apply_gpu_storage``) before GPU scheduling, so ``GPUTransformSDFG`` creates
no ``*_gpu`` transient clones and no copy-in/copy-out states. The compiled
Python-backend SDFG then takes device (cupy) arrays directly — the calling
convention NPBench uses to keep H2D/D2H transfers out of the timed region.

Structural tests need no GPU; runtime tests (``@pytest.mark.gpu``) compile,
run with cupy arrays, and compare against NumPy references.
"""
from typing import Iterator, Tuple

import numpy as np
import pytest

import dace
from dace import data, dtypes
from dace.sdfg import SDFG, nodes
from dace.transformation.passes.vectorization.vectorize_cutile import VectorizeCuTile

N = dace.symbol("N")
M = dace.symbol("M")


@dace.program
def _axpy(a: dace.float64, x: dace.float64[N], y: dace.float64[N]):
    y[:] = a * x + y


@dace.program
def _scale_add_2d(A: dace.float64[M, N], B: dace.float64[M, N]):
    B[:] = A * 2 + B


@dace.program
def _axpy_scalar_out(a: dace.float64, x: dace.float64[N], y: dace.float64[N], s: dace.float64):
    y[:] = a * x + y
    with dace.tasklet:
        inp << y[0]
        out >> s
        out = inp


def _axpy_sdfg(name: str, use_gpu_storage: bool) -> SDFG:
    """Lower the axpy program through the cuTile front door.

    :param name: Unique SDFG name.
    :param use_gpu_storage: Forwarded to :class:`VectorizeCuTile`.
    :returns: The lowered SDFG.
    """
    sdfg = _axpy.to_sdfg(simplify=False)
    sdfg.name = name
    VectorizeCuTile(widths=(32, ), use_gpu_storage=use_gpu_storage).apply_pass(sdfg, {})
    return sdfg


def _nontransient_arrays(sdfg: SDFG) -> Iterator[Tuple[str, data.Array]]:
    """Yield ``(name, desc)`` for non-transient (argument) arrays, scalars excluded."""
    for name, desc in sdfg.arrays.items():
        if not desc.transient and isinstance(desc, data.Array):
            yield name, desc


def _has_host_device_copy_edge(sdfg: SDFG) -> bool:
    """True if any state copies between a host array and a GPU_Global transient.

    :param sdfg: The SDFG to inspect.
    :returns: Whether a copy-in or copy-out edge exists.
    """
    host = (dtypes.StorageType.CPU_Heap, dtypes.StorageType.Default)
    for state in sdfg.states():
        for edge in state.edges():
            if not (isinstance(edge.src, nodes.AccessNode) and isinstance(edge.dst, nodes.AccessNode)):
                continue
            src = sdfg.arrays.get(edge.src.data)
            dst = sdfg.arrays.get(edge.dst.data)
            if src is None or dst is None:
                continue
            if (src.storage in host and dst.storage == dtypes.StorageType.GPU_Global and dst.transient):
                return True
            if (src.storage == dtypes.StorageType.GPU_Global and src.transient and dst.storage in host):
                return True
    return False


# ============================================================
# Structural tests (no GPU)
# ============================================================


def test_gpu_storage_marks_args_and_skips_clones():
    """With the knob on, argument arrays are GPU_Global and GPUTransformSDFG
    creates neither ``*_gpu`` clones nor copy edges."""
    sdfg = _axpy_sdfg("gs_knob_on", use_gpu_storage=True)

    args = dict(_nontransient_arrays(sdfg))
    assert set(args) == {"x", "y"}
    for name, desc in args.items():
        assert desc.storage == dtypes.StorageType.GPU_Global, f"'{name}' storage is {desc.storage}"
        assert f"{name}_gpu" not in sdfg.arrays, f"unexpected clone gpu_{name}"
    assert not _has_host_device_copy_edge(sdfg)
    sdfg.validate()


def test_default_keeps_host_arrays():
    """Regression guard: with the knob off (default), argument arrays stay on
    host and the ``*_gpu`` clones + copy edges DO exist."""
    sdfg = _axpy_sdfg("gs_knob_off", use_gpu_storage=False)

    for name, desc in _nontransient_arrays(sdfg):
        assert desc.storage != dtypes.StorageType.GPU_Global, f"'{name}' unexpectedly on GPU"
        clone = sdfg.arrays.get(f"{name}_gpu")
        assert clone is not None, f"missing clone gpu_{name}"
        assert clone.transient and clone.storage == dtypes.StorageType.GPU_Global
    assert _has_host_device_copy_edge(sdfg)
    sdfg.validate()


def test_gpu_storage_read_only_scalar_stays_host():
    """The read-only scalar ``a`` keeps host storage (``apply_gpu_storage``
    skips unwritten non-transient scalars; GPUTransformSDFG later stamps
    remaining host data CPU_Heap)."""
    sdfg = _axpy_sdfg("gs_scalar_host", use_gpu_storage=True)

    desc = sdfg.arrays["a"]
    assert not desc.transient
    assert desc.storage in (dtypes.StorageType.Default, dtypes.StorageType.CPU_Heap)


def test_gpu_storage_written_scalar_stays_host():
    """Scalar outputs keep the host ABI and are copied back by offloading."""
    sdfg = _axpy_scalar_out.to_sdfg(simplify=False)
    sdfg.name = "gs_written_scalar"
    VectorizeCuTile(widths=(32, ), use_gpu_storage=True).apply_pass(sdfg, {})

    desc = sdfg.arrays["s"]
    assert not desc.transient
    assert desc.storage == dtypes.StorageType.CPU_Heap
    assert _has_host_device_copy_edge(sdfg)


def test_gpu_storage_warns_when_ineffective():
    """On an SDFG already GPU-transformed WITHOUT gpu storage (existing
    ``*_gpu`` clones + copy states), the knob cannot remove the copies and
    emits a UserWarning."""
    sdfg = _axpy.to_sdfg(simplify=False)
    sdfg.name = "gs_warn_ineffective"
    VectorizeCuTile.canonicalize_for_cutile(sdfg)
    sdfg.apply_gpu_transformations( simplify=False)

    with pytest.warns(UserWarning, match="use_gpu_storage had no effect"):
        VectorizeCuTile(widths=(32, ), run_canonicalize=False, use_gpu_storage=True).apply_pass(sdfg, {})


def test_gpu_storage_pre_scheduled_noop():
    """On an already GPU-scheduled SDFG (autoopt_gpu track:
    ``auto_optimize(GPU, use_gpu_storage=True, expand=False)``), the knob is a
    no-op and the pipeline still lowers without clones."""
    from dace.transformation.auto import auto_optimize as opt

    sdfg = _axpy.to_sdfg(simplify=False)
    sdfg.name = "gs_pre_scheduled"
    VectorizeCuTile.canonicalize_for_cutile(sdfg)
    opt.auto_optimize(sdfg, dtypes.DeviceType.GPU, use_gpu_storage=True, expand=False)

    num_kernels = VectorizeCuTile(widths=(32, ), run_canonicalize=False, use_gpu_storage=True).apply_pass(sdfg, {})
    assert num_kernels == 1

    for name, desc in _nontransient_arrays(sdfg):
        assert desc.storage == dtypes.StorageType.GPU_Global
        assert f"{name}_gpu" not in sdfg.arrays
    assert not _has_host_device_copy_edge(sdfg)
    sdfg.validate()


# ============================================================
# Runtime tests (GPU)
# ============================================================


@pytest.mark.gpu
@pytest.mark.parametrize("n", [64, 70])
def test_axpy_gpu_storage_runtime(n: int):
    """Compile+run axpy with cupy arrays (divisible n=64 and remainder n=70);
    the result lands in-place in the caller's cupy array."""
    import cupy as cp

    sdfg = _axpy_sdfg(f"gs_rt_axpy_{n}", use_gpu_storage=True)
    csdfg = sdfg.compile()

    rng = np.random.default_rng(7)
    a = 2.5
    x_host = rng.random(n)
    y_host = rng.random(n)
    ref = a * x_host + y_host

    x_gpu = cp.asarray(x_host)
    y_gpu = cp.asarray(y_host)
    csdfg(a=a, x=x_gpu, y=y_gpu, N=n)

    # In-place: the caller's cupy array holds the result (no hidden copy-back).
    np.testing.assert_allclose(cp.asnumpy(y_gpu), ref, rtol=1e-14)


@pytest.mark.gpu
def test_2d_gpu_storage_runtime():
    """2-D ``B = A * 2 + B`` with widths (8, 8), shape (20, 28) — both dims hit
    the remainder path — symbolic M, N."""
    import cupy as cp

    sdfg = _scale_add_2d.to_sdfg(simplify=False)
    sdfg.name = "gs_rt_2d"
    VectorizeCuTile(widths=(8, 8), use_gpu_storage=True).apply_pass(sdfg, {})
    csdfg = sdfg.compile()

    m, n = 20, 28
    rng = np.random.default_rng(8)
    A_host = rng.random((m, n))
    B_host = rng.random((m, n))
    ref = A_host * 2 + B_host

    A_gpu = cp.asarray(A_host)
    B_gpu = cp.asarray(B_host)
    csdfg(A=A_gpu, B=B_gpu, M=m, N=n)

    np.testing.assert_allclose(cp.asnumpy(B_gpu), ref, rtol=1e-14)


@pytest.mark.gpu
def test_gpu_storage_rejects_host_arrays():
    """Contract change: a knob-compiled SDFG called with numpy (host) arrays
    fails instead of silently copying."""
    sdfg = _axpy_sdfg("gs_rt_reject_host", use_gpu_storage=True)
    csdfg = sdfg.compile()

    n = 64
    rng = np.random.default_rng(9)
    x = rng.random(n)
    y = rng.random(n)

    # Observed error: `RuntimeError: NumPy only supports stream=None` from
    # cuda.tile/cupy when handed host arrays.
    with pytest.raises(RuntimeError, match="NumPy"):
        csdfg(a=2.5, x=x, y=y, N=n)


if __name__ == "__main__":
    test_gpu_storage_marks_args_and_skips_clones()
    test_default_keeps_host_arrays()
    test_gpu_storage_read_only_scalar_stays_host()
    test_gpu_storage_written_scalar_stays_host()
    test_gpu_storage_warns_when_ineffective()
    test_gpu_storage_pre_scheduled_noop()
    test_axpy_gpu_storage_runtime(64)
    test_axpy_gpu_storage_runtime(70)
    test_2d_gpu_storage_runtime()
    test_gpu_storage_rejects_host_arrays()
