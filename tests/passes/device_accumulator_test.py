# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A scalar accumulating what device code computes lives on the device, and only a host map writing device memory
becomes the kernel: a kernel takes a host Scalar by value, and its accumulation would never leave it
(bdf_newton_krylov's ``h_pk``, an rvalue handed to ``reduce_atomic``)."""

import numpy as np
import pytest

import dace
from dace.libraries.blas import Dot, MatMul
from dace.transformation.passes.canonicalize.finalize import finalize_for_target, offload_to_gpu
from dace.transformation.passes.gpu_specialization.promote_host_maps_to_kernels import PromoteHostMapsToKernels

N = dace.symbol("N")
#: The connectors ``x``, ``y`` and the result of each library node a row's dot product can be.
CONNECTORS = {MatMul: ("_a", "_b", "_c"), Dot: ("_x", "_y", "_result")}


def row_dots(
    name: str, node: type = MatMul, storage: dace.StorageType = dace.StorageType.Default
) -> tuple[dace.SDFG, dace.nodes.MapEntry]:
    """bdf_newton_krylov's projection after canonicalization: ``h = 0; for i in map: h += q[i, :] @ w[i, :]``, and
    ``out`` receives ``h``. A device library node per row keeps the map over the rows on the host at the offload.

    :param node: the library node of the dot product (canonicalization lifts ``@`` to ``MatMul``, which the GPU
                 specialization makes a ``Dot``).
    :param storage: where ``h`` lives.
    :returns: the SDFG and the map over the rows.
    """
    sdfg = dace.SDFG(name)
    sdfg.add_array("q", [N, N], dace.float64)
    sdfg.add_array("w", [N, N], dace.float64)
    sdfg.add_array("out", [1], dace.float64)
    sdfg.add_scalar("t", dace.float64, transient=True)
    sdfg.add_scalar("h", dace.float64, transient=True, storage=storage)
    state = sdfg.add_state("rows")
    zero = state.add_tasklet("zero", {}, {"o"}, "o = 0.0")
    h_init = state.add_access("h")
    state.add_edge(zero, "o", h_init, None, dace.Memlet("h[0]"))
    entry, exit_ = state.add_map("rows", {"i": "0:N"})
    state.add_nedge(h_init, entry, dace.Memlet())
    x, y, result = CONNECTORS[node]
    dot = node("dot")
    state.add_node(dot)
    for conn, array in ((x, "q"), (y, "w")):
        state.add_memlet_path(state.add_read(array), entry, dot, dst_conn=conn, memlet=dace.Memlet(f"{array}[i, 0:N]"))
    t = state.add_access("t")
    state.add_edge(dot, result, t, None, dace.Memlet("t[0]"))
    h = state.add_access("h")
    state.add_memlet_path(
        t, exit_, h, memlet=dace.Memlet(data="h", subset="0", other_subset="0", wcr="lambda x, y: x + y")
    )
    store = state.add_tasklet("store", {"v"}, {"o"}, "o = v")
    state.add_edge(h, None, store, "v", dace.Memlet("h[0]"))
    state.add_edge(store, "o", state.add_write("out"), None, dace.Memlet("out[0]"))
    sdfg.validate()
    return sdfg, entry


def test_the_accumulator_of_a_device_result_is_placed_on_the_device():
    """``t`` is a device library node's result; the accumulation into ``h`` is device work, not a transfer."""
    sdfg, _ = row_dots("row_dots_offload")
    sdfg.apply_gpu_transformations(validate=True, simplify=False)
    assert sdfg.arrays["h"].storage == dace.StorageType.GPU_Global


def test_a_host_map_accumulating_into_a_host_scalar_stays_on_the_host():
    """The map writes ``h`` outside it: in host memory, it cannot be the kernel."""
    sdfg, rows = row_dots("row_dots_host", Dot, dace.StorageType.CPU_Heap)
    for name in ("q", "w", "out", "t"):
        sdfg.arrays[name].storage = dace.StorageType.GPU_Global
    rows.map.schedule = dace.ScheduleType.Sequential
    next(n for n in sdfg.start_block.nodes() if isinstance(n, Dot)).schedule = dace.ScheduleType.GPU_Device
    assert PromoteHostMapsToKernels().apply_pass(sdfg, {}) is None
    assert rows.map.schedule == dace.ScheduleType.Sequential
    sdfg.arrays["h"].storage = dace.StorageType.GPU_Global
    assert PromoteHostMapsToKernels().apply_pass(sdfg, {}) == 1
    assert rows.map.schedule == dace.ScheduleType.GPU_Device


@pytest.mark.gpu
def test_the_row_dots_match_numpy_on_the_device():
    import cupy  # Only present on GPU runners.

    sdfg, _ = row_dots("row_dots_run")
    offload_to_gpu(sdfg)
    finalize_for_target(sdfg, target="gpu")
    rng = np.random.default_rng(0)
    q, w = rng.random((300, 300)), rng.random((300, 300))
    out = cupy.zeros(1)
    with (
        dace.config.set_temporary("compiler", "cuda", "implementation", value="experimental"),
        dace.config.set_temporary("compiler", "emit_tree_reductions", value=True),
    ):
        sdfg(q=cupy.asarray(q), w=cupy.asarray(w), out=out, N=300)
    np.testing.assert_allclose(cupy.asnumpy(out)[0], (q * w).sum(), rtol=1e-12, atol=0)


if __name__ == "__main__":
    test_the_accumulator_of_a_device_result_is_placed_on_the_device()
    test_a_host_map_accumulating_into_a_host_scalar_stays_on_the_host()
    test_the_row_dots_match_numpy_on_the_device()
