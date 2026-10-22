# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The LIBXSMM GEMM: a small real product inside a parallel region lowers to ``libxsmm_?gemm`` and matches NumPy."""

import numpy as np
import pytest

import dace
from dace import dtypes
from dace.libraries.blas import Gemm
from dace.libraries.blas.environments.libxsmm import LIBXSMM
from dace.transformation.passes.canonicalize.finalize import canonicalize_set_fast_implementations

pytestmark = pytest.mark.skipif(not LIBXSMM.is_installed(), reason="LIBXSMM is not installed (LIBXSMM_ROOT)")

B = dace.symbol("B")


def batch_of_gemms(dtype, trans_a: bool, beta: float) -> dace.SDFG:
    """``C[b] = A[b]^(T) @ X[b] + beta * C[b]`` in place for every ``b`` of a parallel map, one 9x81x5 GEMM per
    iteration."""
    sdfg = dace.SDFG(f"libxsmm_batch_{dtype.to_string()}_{trans_a}_{int(beta)}")
    a_shape = [B, 5, 9] if trans_a else [B, 9, 5]
    sdfg.add_array("A", a_shape, dtype)
    sdfg.add_array("X", [B, 5, 81], dtype)
    sdfg.add_array("C", [B, 9, 81], dtype)
    state = sdfg.add_state()
    entry, exit_ = state.add_map("batch", {"b": "0:B"}, schedule=dtypes.ScheduleType.CPU_Multicore)
    gemm = Gemm("gemm", transA=trans_a, alpha=1.0, beta=beta, cin=False)
    state.add_node(gemm)
    a_sub = "b, 0:5, 0:9" if trans_a else "b, 0:9, 0:5"
    state.add_memlet_path(state.add_read("A"), entry, gemm, dst_conn="_a", memlet=dace.Memlet(f"A[{a_sub}]"))
    state.add_memlet_path(state.add_read("X"), entry, gemm, dst_conn="_b", memlet=dace.Memlet("X[b, 0:5, 0:81]"))
    state.add_memlet_path(gemm, exit_, state.add_write("C"), src_conn="_c", memlet=dace.Memlet("C[b, 0:9, 0:81]"))
    sdfg.validate()
    return sdfg


@pytest.mark.parametrize("dtype", [dace.float32, dace.float64])
@pytest.mark.parametrize("trans_a", [False, True])
@pytest.mark.parametrize("beta", [0.0, 1.0])
def test_a_gemm_in_a_parallel_map_lowers_to_libxsmm(dtype, trans_a, beta):
    sdfg = batch_of_gemms(dtype, trans_a, beta)
    canonicalize_set_fast_implementations(sdfg, dtypes.DeviceType.CPU)
    gemm = next(n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, Gemm))
    assert gemm.implementation == "LIBXSMM"
    rng = np.random.default_rng(0)
    b, npdt = 37, dtype.type
    a = rng.random((b, 5, 9) if trans_a else (b, 9, 5)).astype(npdt)
    x = rng.random((b, 5, 81)).astype(npdt)
    c = rng.random((b, 9, 81)).astype(npdt)
    expected = (np.swapaxes(a, 1, 2) if trans_a else a) @ x + beta * c
    sdfg(A=a, X=x, C=c, B=b)
    assert np.allclose(c, expected, rtol=1e-5 if dtype == dace.float32 else 1e-12)


def test_a_top_level_gemm_keeps_openblas():
    """A GEMM no parallel region encloses is a threaded call: OpenBLAS with its shape plan, not LIBXSMM."""
    sdfg = dace.SDFG("libxsmm_top_level")
    sdfg.add_array("A", [64, 64], dace.float64)
    sdfg.add_array("X", [64, 64], dace.float64)
    sdfg.add_array("C", [64, 64], dace.float64)
    state = sdfg.add_state()
    gemm = Gemm("gemm")
    state.add_node(gemm)
    state.add_edge(state.add_read("A"), None, gemm, "_a", dace.Memlet("A[0:64, 0:64]"))
    state.add_edge(state.add_read("X"), None, gemm, "_b", dace.Memlet("X[0:64, 0:64]"))
    state.add_edge(gemm, "_c", state.add_write("C"), None, dace.Memlet("C[0:64, 0:64]"))
    canonicalize_set_fast_implementations(sdfg, dtypes.DeviceType.CPU)
    assert gemm.implementation == "Planned"


if __name__ == "__main__":
    for dtype in [dace.float32, dace.float64]:
        for trans_a in [False, True]:
            for beta in [0.0, 1.0]:
                test_a_gemm_in_a_parallel_map_lowers_to_libxsmm(dtype, trans_a, beta)
    test_a_top_level_gemm_keeps_openblas()
