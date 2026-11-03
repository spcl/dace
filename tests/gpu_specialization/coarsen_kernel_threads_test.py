# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""CoarsenKernelThreads gives every thread of a kernel several consecutive elements of one axis.

The structural checks need no GPU; the numeric one runs on the device, at sizes the factor does not divide.
"""

import numpy as np
import pytest

import dace
from dace import dtypes
from dace.transformation.passes.canonicalize.finalize import finalize_for_target, offload_to_gpu
from dace.transformation.passes.canonicalize.pipeline import canonicalize
from dace.transformation.passes.gpu_specialization.coarsen_kernel_threads import CoarsenKernelThreads

M, N = (dace.symbol(s) for s in "MN")


@dace.program
def scaled_columns(w: dace.float64[N], A: dace.float64[M, N], B: dace.float64[M, N]):
    for i, j in dace.map[0:M, 0:N]:
        B[i, j] = w[j] * A[i, j]


def canonical_gpu(name: str) -> dace.SDFG:
    sdfg = scaled_columns.to_sdfg(simplify=True)
    sdfg.name = name
    canonicalize(sdfg, target="gpu")
    offload_to_gpu(sdfg)
    # The pass under test takes the kernel as the offload leaves it, without the coalesced tiling.
    with dace.config.set_temporary("optimizer", "canonicalization", "gpu_coarsen_factor", value=1):
        finalize_for_target(sdfg, "gpu")
    return sdfg


def test_the_outer_axis_becomes_a_serial_loop_of_each_thread():
    """By default the outermost parameter (not threadIdx.x) is tiled; the grid keeps the launch properties and the
    tile is a Sequential map."""
    sdfg = canonical_gpu("coarsen_default")
    (kernel,) = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.MapEntry)]
    kernel.map.instrument = dtypes.InstrumentationType.GPU_Events
    block = list(kernel.map.gpu_block_size or [])
    assert CoarsenKernelThreads(factor=4).apply_pass(sdfg, {}) == {kernel.map.label: {"i": 4}}
    entries = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.MapEntry)]
    grid = next(e for e in entries if e.map.schedule == dtypes.ScheduleType.GPU_Device)
    tile = next(e for e in entries if e.map.params == ["i"])
    assert grid.map.params == ["c_i", "j"] and str(grid.map.range[0][2]) == "4"
    assert tile.map.schedule == dtypes.ScheduleType.Sequential
    assert list(grid.map.gpu_block_size or []) == block
    assert grid.map.instrument == dtypes.InstrumentationType.GPU_Events
    assert tile.map.instrument == dtypes.InstrumentationType.No_Instrumentation
    sdfg.validate()


def test_a_named_parameter_is_coarsened_and_factor_one_changes_nothing():
    sdfg = canonical_gpu("coarsen_named")
    assert CoarsenKernelThreads(factor=1).apply_pass(sdfg, {}) is None
    assert CoarsenKernelThreads(factor=4, parameter="x").apply_pass(sdfg, {}) is None
    assert list(CoarsenKernelThreads(factor=4, parameter="j").apply_pass(sdfg, {}).values()) == [{"j": 4}]


def test_two_parameters_are_coarsened_into_a_tile_per_thread():
    sdfg = canonical_gpu("coarsen_2d")
    assert list(CoarsenKernelThreads(factors={"i": 3, "j": 2}).apply_pass(sdfg, {}).values()) == [{"i": 3, "j": 2}]
    entries = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.MapEntry)]
    grid = next(e for e in entries if e.map.schedule == dtypes.ScheduleType.GPU_Device)
    tile = next(e for e in entries if e.map.schedule == dtypes.ScheduleType.Sequential)
    assert grid.map.params == ["c_i", "c_j"] and tile.map.params == ["i", "j"]
    sdfg.validate()


@pytest.mark.gpu
@pytest.mark.parametrize("factors", [{"i": 6}, {"j": 6}, {"i": 6, "j": 2}])
def test_coarsened_kernels_compute_what_numpy_does(factors):
    import cupy as cp

    m, n = 37, 1013
    sdfg = canonical_gpu("coarsen_numeric_" + "_".join(f"{p}{f}" for p, f in factors.items()))
    CoarsenKernelThreads(factors=factors).apply_pass(sdfg, {})
    rng = np.random.default_rng(0)
    w, a = rng.random(n), rng.random((m, n))
    b = cp.zeros((m, n))
    sdfg(w=cp.asarray(w), A=cp.asarray(a), B=b, M=m, N=n)
    np.testing.assert_allclose(cp.asnumpy(b), w * a)


if __name__ == "__main__":
    test_the_outer_axis_becomes_a_serial_loop_of_each_thread()
    test_a_named_parameter_is_coarsened_and_factor_one_changes_nothing()
    test_two_parameters_are_coarsened_into_a_tile_per_thread()
