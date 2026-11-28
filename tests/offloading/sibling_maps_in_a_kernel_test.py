# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A kernel whose body holds maps side by side becomes one kernel per map, each over both axes.

ICON VT fuses ``z_ekinh`` (levels ``1:nlev``) and ``z_w_concorr_mc`` (levels ``nflatlev:nlev``) into one map over
``jc`` that holds both level maps and a tasklet. As one kernel over ``jc``, each thread looped over the levels.
"""

import numpy as np
import pytest

import dace
from dace import dtypes
from dace.sdfg import nodes
from dace.transformation import pass_pipeline as ppl
from dace.transformation.passes.offloading import OffloadToAccelerator

N, L, F = (dace.symbol(name) for name in ("N", "L", "F"))


def sibling_maps_sdfg() -> dace.SDFG:
    """``A[j, k] = 2 B[j, k]`` over ``k < L`` and ``C[j, k] = B[j, k] + 1`` over ``F <= k < L`` in two maps, and
    ``D[j] = 0``, all under one map over ``j``."""
    sdfg = dace.SDFG("sibling_maps_in_a_kernel")
    for name in ("A", "B", "C"):
        sdfg.add_array(name, [N, L], dace.float64)
    sdfg.add_array("D", [N], dace.float64)
    sdfg.add_symbol("F", dace.int64)
    state = sdfg.add_state("compute")
    outer_entry, outer_exit = state.add_map("outer", {"j": "0:N"})
    b = state.add_read("B")
    for name, low, code in (("A", "0", "y = 2.0 * x"), ("C", "F", "y = x + 1.0")):
        entry, exit_ = state.add_map(f"inner_{name}", {"k": f"{low}:L"})
        tasklet = state.add_tasklet(f"t_{name}", {"x"}, {"y"}, code)
        state.add_memlet_path(b, outer_entry, entry, tasklet, dst_conn="x", memlet=dace.Memlet("B[j, k]"))
        state.add_memlet_path(
            tasklet, exit_, outer_exit, state.add_write(name), src_conn="y", memlet=dace.Memlet(f"{name}[j, k]")
        )
    zero = state.add_tasklet("zero", {}, {"y"}, "y = 0.0")
    state.add_nedge(outer_entry, zero, dace.Memlet())
    state.add_memlet_path(zero, outer_exit, state.add_write("D"), src_conn="y", memlet=dace.Memlet("D[j]"))
    sdfg.validate()
    return sdfg


def test_sibling_maps_in_a_kernel_become_kernels_over_both_axes():
    sdfg = sibling_maps_sdfg()
    ppl.Pipeline([OffloadToAccelerator()]).apply_pass(sdfg, {})
    sdfg.validate()
    maps = [node.map for node, _ in sdfg.all_nodes_recursive() if isinstance(node, nodes.MapEntry)]
    assert sorted(len(m.params) for m in maps) == [1, 2, 2], [(m.params, m.schedule) for m in maps]
    assert all(m.schedule == dtypes.ScheduleType.GPU_Device for m in maps), [(m.params, m.schedule) for m in maps]


@pytest.mark.gpu
def test_sibling_maps_split_into_kernels_compute_what_the_fused_map_did():
    rng = np.random.default_rng(0)
    B = rng.random((6, 5))
    expected = {"A": 2.0 * B, "C": np.zeros((6, 5)), "D": np.zeros(6)}
    expected["C"][:, 2:] = B[:, 2:] + 1.0
    sdfg = sibling_maps_sdfg()
    sdfg.apply_gpu_transformations()
    A, C, D = np.zeros((6, 5)), np.zeros((6, 5)), np.ones(6)
    sdfg(A=A, B=B, C=C, D=D, N=6, L=5, F=2)
    assert np.allclose(A, expected["A"]) and np.allclose(C, expected["C"]) and np.allclose(D, expected["D"])


if __name__ == "__main__":
    test_sibling_maps_in_a_kernel_become_kernels_over_both_axes()
    test_sibling_maps_split_into_kernels_compute_what_the_fused_map_did()
