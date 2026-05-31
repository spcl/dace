# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Minimal reproducers for four GPU-offloading bugs found in the npbench sweep.

Each test is a tight kernel extracted from the failure pattern of a real
npbench kernel. The kernel exercises a single failure mode end-to-end:

- :func:`test_reduce_on_view`       -- lenet maxpool sliding-window View
- :func:`test_host_iedge_read`      -- mandelbrot2 ``if I[j]:`` host-side
- :func:`test_map_straddle`         -- mandelbrot2 follow-on (host/GPU on one map)
- :func:`test_int_pow_used_as_index`-- stockham_fft ``pow(R, K)`` as index

Each test pre-validates the SDFG after :func:`auto_optimize` with
``device=GPU`` -- the contract is that GPU offloading produces a valid SDFG.
Numerical correctness is asserted where the kernel is small and tractable.
"""
import numpy as np
import pytest

import dace
from dace.transformation.auto.auto_optimize import auto_optimize

N = dace.symbol('N')


# ---------------------------------------------------------------------------
# Bug 1: Reduce expansion with a View input (lenet maxpool pattern).
#
# ``np.max(x[:, 2*i:2*i+2, 2*j:2*j+2, :], axis=(1, 2))`` creates a sliding-window
# View into ``x`` and feeds it into a ``Reduce`` node. ``ExpandReduceGPUAuto``
# copies the input descriptor into its nested SDFG; if that descriptor is a
# View the viewed-source array is left behind and ``_in`` becomes dangling.
# Validation rejects the nested SDFG on the next ``MapCollapse`` apply.
# ---------------------------------------------------------------------------


@dace.program
def _reduce_view_kernel(x: dace.float64[N, 4, 4, 8], out: dace.float64[N, 2, 2, 8]):
    """One sliding-window max-pool over a 4x4 input -> 2x2 output.

    Python ``for`` loops (not ``dace.map``) so the Reduce nodes stay at the
    SDFG top-level after auto-optimize -- their View inputs keep
    ``GPU_Global`` storage and pick the ``GPUAuto`` expansion path (the one
    that copies the input descriptor into a nested SDFG).
    """
    for i in range(2):
        for j in range(2):
            out[:, i, j, :] = np.max(x[:, 2 * i:2 * i + 2, 2 * j:2 * j + 2, :], axis=(1, 2))


@pytest.mark.gpu
def test_reduce_on_view():
    """Structural: the SDFG must validate after GPU offloading. The bug
    this reproducer exercises is purely structural -- a dangling View in
    ``ExpandReduceGPUAuto``'s nested SDFG. Numerical correctness for the
    sliding-window max-pool pattern depends on the unrelated auto-optimize
    behaviour around Reduce nodes and is asserted by lenet's own test."""
    sdfg = _reduce_view_kernel.to_sdfg()
    sdfg = auto_optimize(sdfg, dace.dtypes.DeviceType.GPU)
    sdfg.validate()


# ---------------------------------------------------------------------------
# Bug 2: host interstate-edge read of a GPU-resident container
# (mandelbrot2 ``I[:length] = ...; for j: if I[j]: ...``).
#
# A bool transient is written by a GPU-eligible map, then the kernel branches
# on individual entries inside a host loop. The branch lowers to an interstate
# edge with assignment ``i_tmp = I[j]``; if ``I`` was promoted to GPU storage
# the validator rejects the SDFG ('reading inaccessible GPU_Global container
# in host code interstate edge').
# ---------------------------------------------------------------------------


@dace.program
def _host_iedge_read_kernel(data: dace.float64[N], out: dace.int32[1]):
    # ``np.greater`` materialises a bool array DaCe can lower to a GPU map.
    I = np.greater(data, 0.5)
    count: dace.int32 = 0
    for j in range(N):
        if I[j]:
            count = count + 1
    out[0] = count


@pytest.mark.gpu
def test_host_iedge_read():
    sdfg = _host_iedge_read_kernel.to_sdfg()
    sdfg = auto_optimize(sdfg, dace.dtypes.DeviceType.GPU)
    sdfg.validate()

    rng = np.random.default_rng(1)
    data = rng.uniform(0, 1, size=20)
    out = np.zeros(1, dtype=np.int32)
    sdfg(data=data, out=out, N=20)
    assert int(out[0]) == int(np.sum(data > 0.5))


# ---------------------------------------------------------------------------
# Bug 3: a map straddling host (host_data) input and GPU-promoted output.
#
# Once bug 2 is fixed and ``I`` is pinned to host, the negation map
# ``I[:] = np.logical_not(I[:])`` reads host ``I`` and writes a top-level
# intermediate, which the existing storage-promotion step would still bump to
# GPU_Global. The map ends up with one input on CPU_Heap and one output on
# GPU_Global, and ``infer_types.set_default_schedule_and_storage_types``
# raises on the multi-constraint conflict.
# ---------------------------------------------------------------------------


@dace.program
def _map_straddle_kernel(data: dace.float64[N], out: dace.bool_[N]):
    I = np.greater(data, 0.5)

    # Force I host via an iedge read.
    keep_alive: dace.int32 = 0
    for j in range(N):
        if I[j]:
            keep_alive = keep_alive + 1

    # Negation: reads host I, writes a transient that would otherwise be
    # GPU-promoted. The map's host_data membership must propagate.
    not_I = np.logical_not(I)
    for k in dace.map[0:N]:
        out[k] = not_I[k]
    # Touch ``keep_alive`` so the iedge read isn't dead-code-eliminated.
    if keep_alive < 0:
        out[0] = False


@pytest.mark.gpu
def test_map_straddle():
    sdfg = _map_straddle_kernel.to_sdfg()
    sdfg = auto_optimize(sdfg, dace.dtypes.DeviceType.GPU)
    sdfg.validate()

    rng = np.random.default_rng(2)
    data = rng.uniform(0, 1, size=15)
    out = np.zeros(15, dtype=np.bool_)
    sdfg(data=data, out=out, N=15)
    ref = np.logical_not(data > 0.5)
    assert np.array_equal(out, ref)


# ---------------------------------------------------------------------------
# Bug 4: integer-typed ``pow(R, K)`` used as an array index.
#
# Stockham FFT uses ``R ** K`` where ``R`` and ``K`` are ``int64_t`` symbols,
# and the result indexes a transient. ``dace::math::pow(int64_t, int64_t)``
# previously fell through to ``std::pow`` returning ``double`` -- illegal as
# an array subscript or ``cudaLaunchKernel`` grid-dim.
#
# Per the corrected design: integer pow is only safe when the exponent type
# is *unsigned* (otherwise a negative exponent produces a fractional value).
# Stockham_fft itself needs to declare its radix exponents as unsigned for
# the integer overload to apply; the runtime contract is checked here.
# ---------------------------------------------------------------------------

R = dace.symbol('R', dace.int64)
K = dace.symbol('K', dace.uint32)  # unsigned -> integer pow is well-defined


@dace.program
def _int_pow_index_kernel(out: dace.float64[R**K]):
    for i in dace.map[0:R**K]:
        out[i] = out[i] + 1.0


@pytest.mark.gpu
def test_int_pow_used_as_index():
    sdfg = _int_pow_index_kernel.to_sdfg()
    sdfg = auto_optimize(sdfg, dace.dtypes.DeviceType.GPU)
    sdfg.validate()

    R_v, K_v = 2, 6  # 2**6 = 64
    out = np.zeros(R_v**K_v, dtype=np.float64)
    sdfg(out=out, R=R_v, K=K_v)
    assert np.array_equal(out, np.ones(R_v**K_v, dtype=np.float64))


if __name__ == "__main__":
    test_reduce_on_view()
    test_host_iedge_read()
    test_map_straddle()
    test_int_pow_used_as_index()
