# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Reduction over View inputs. A View is ``source_ptr + offset`` with its own
strides/step (``view[i]`` addresses ``i * stride``), so a reduction must work
directly on a strided / sliced / reshaped View -- on the GPU via the efficient
``GPUAuto`` device schedule (a strided reduction), never falling back to the pure
expansion.
"""
import warnings

import numpy as np
import pytest

import dace
from dace.transformation.auto.auto_optimize import set_fast_implementations


def gpu_available() -> bool:
    """True if a CUDA/HIP device is usable for compilation + execution."""
    try:
        import cupy  # noqa: F401  # a working cupy implies a usable device

        return cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def run_on_gpu(program, inputs: dict) -> tuple:
    """Build ``program`` for the GPU (cuSOLVER/cuBLAS/GPUAuto selection) and run
    it on a copy of ``inputs``; return (outputs, warnings)."""
    sdfg = program.to_sdfg(simplify=True)
    sdfg.apply_gpu_transformations()
    set_fast_implementations(sdfg, dace.DeviceType.GPU)
    call = {k: v.copy() for k, v in inputs.items()}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        sdfg(**call)
    return call, caught


def view_pure_fallback(caught) -> bool:
    """Whether a 'View -> Pure expansion' fallback warning was emitted."""
    return any('View' in str(w.message) and 'Pure' in str(w.message) for w in caught)


@dace.program
def reduce_strided_view(x: dace.float64[8, 16], y: dace.float64[8]):
    y[:] = np.sum(x[:, ::2], axis=1)


@dace.program
def reduce_sliced_view(x: dace.float64[8, 16], y: dace.float64[7]):
    y[:] = np.sum(x[1:8, 2:14], axis=1)


@dace.program
def reduce_strided_view_full(x: dace.float64[8, 16], s: dace.float64[1]):
    s[0] = np.sum(x[:, ::2])


def test_reduce_strided_view_cpu():
    x = np.random.rand(8, 16)
    y = np.zeros(8)
    reduce_strided_view(x=x.copy(), y=y)
    assert np.allclose(y, np.sum(x[:, ::2], axis=1))


@pytest.mark.gpu
def test_reduce_strided_view_gpu():
    if not gpu_available():
        pytest.skip('no CUDA/HIP device')
    x = np.random.rand(8, 16)
    out, caught = run_on_gpu(reduce_strided_view, dict(x=x, y=np.zeros(8)))
    assert not view_pure_fallback(caught), 'GPUAuto reduction fell back to Pure on a View input'
    assert np.allclose(out['y'], np.sum(x[:, ::2], axis=1))


@pytest.mark.gpu
def test_reduce_sliced_view_gpu():
    if not gpu_available():
        pytest.skip('no CUDA/HIP device')
    x = np.random.rand(8, 16)
    out, caught = run_on_gpu(reduce_sliced_view, dict(x=x, y=np.zeros(7)))
    assert not view_pure_fallback(caught)
    assert np.allclose(out['y'], np.sum(x[1:8, 2:14], axis=1))


@pytest.mark.gpu
def test_reduce_strided_view_full_gpu():
    if not gpu_available():
        pytest.skip('no CUDA/HIP device')
    x = np.random.rand(8, 16)
    out, caught = run_on_gpu(reduce_strided_view_full, dict(x=x, s=np.zeros(1)))
    assert not view_pure_fallback(caught)
    assert np.allclose(out['s'][0], np.sum(x[:, ::2]))


if __name__ == '__main__':
    test_reduce_strided_view_cpu()
    test_reduce_strided_view_gpu()
    test_reduce_sliced_view_gpu()
    test_reduce_strided_view_full_gpu()
    print('ok')
