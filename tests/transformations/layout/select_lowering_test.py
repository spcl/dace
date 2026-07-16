# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The layout transforms must NOT hardcode a library lowering: they insert relayout library nodes
(TensorTranspose / TensorDot) with ``implementation is None``, and the device-driven
``select_layout_lowering`` step chooses ``pure`` (CPU) or ``cuTENSOR`` (GPU) just before compile.
These tests pin that contract: the passes leave lowering unset, the step selects it, an explicit
choice is preserved, and a bad device is rejected."""
import numpy
import pytest
import dace

from dace.transformation.layout.rewrite_libnodes import GemmToTensorDot, RewriteCopyForLayout
from dace.transformation.layout.select_lowering import select_layout_lowering
from dace.libraries.blas.nodes.gemm import Gemm
from dace.libraries.linalg import TensorTranspose, TensorDot


def _gemm_sdfg():
    M, K, Nn = 4, 5, 6
    sdfg = dace.SDFG("gemm_sel")
    sdfg.add_array("A", [M, K], dace.float64)
    sdfg.add_array("B", [K, Nn], dace.float64)
    sdfg.add_array("C", [M, Nn], dace.float64)
    st = sdfg.add_state("s", is_start_block=True)
    g = Gemm("gemm")
    st.add_node(g)
    st.add_edge(st.add_read("A"), None, g, "_a", dace.Memlet.from_array("A", sdfg.arrays["A"]))
    st.add_edge(st.add_read("B"), None, g, "_b", dace.Memlet.from_array("B", sdfg.arrays["B"]))
    st.add_edge(g, "_c", st.add_write("C"), None, dace.Memlet.from_array("C", sdfg.arrays["C"]))
    return sdfg, st


def _tds(sdfg):
    return [n for n in sdfg.all_nodes_recursive() if isinstance(n[0], TensorDot)]


def test_gemm_to_tensordot_leaves_lowering_unset():
    """The transform is device-agnostic: the inserted TensorDot has no implementation."""
    sdfg, _ = _gemm_sdfg()
    assert GemmToTensorDot().apply_pass(sdfg, {}) == 1
    td = _tds(sdfg)[0][0]
    assert td.implementation is None


def test_select_cpu_sets_pure():
    sdfg, _ = _gemm_sdfg()
    GemmToTensorDot().apply_pass(sdfg, {})
    assert select_layout_lowering(sdfg, "cpu") == 1
    assert _tds(sdfg)[0][0].implementation == "pure"


def test_select_gpu_sets_cutensor():
    sdfg, _ = _gemm_sdfg()
    GemmToTensorDot().apply_pass(sdfg, {})
    assert select_layout_lowering(sdfg, "gpu") == 1
    assert _tds(sdfg)[0][0].implementation == "cuTENSOR"


def test_select_preserves_explicit_choice():
    """A lowering the caller pinned is never overwritten (only ``None`` nodes are touched)."""
    sdfg = dace.SDFG("tt_pinned")
    sdfg.add_array("X", [4, 3], dace.float64)
    sdfg.add_array("Y", [3, 4], dace.float64)
    st = sdfg.add_state("s", is_start_block=True)
    tt = TensorTranspose("t", axes=[1, 0])
    tt.implementation = "HPTT"
    st.add_node(tt)
    st.add_edge(st.add_read("X"), None, tt, "_inp_tensor", dace.Memlet.from_array("X", sdfg.arrays["X"]))
    st.add_edge(tt, "_out_tensor", st.add_write("Y"), None, dace.Memlet.from_array("Y", sdfg.arrays["Y"]))
    assert select_layout_lowering(sdfg, "cpu") == 0  # pinned node skipped
    assert tt.implementation == "HPTT"


def test_select_bad_device_rejected():
    sdfg, _ = _gemm_sdfg()
    with pytest.raises(ValueError):
        select_layout_lowering(sdfg, "tpu")


def test_selected_gemm_runs_bitexact_cpu():
    """End to end: transform (no lowering) -> select CPU -> compile -> matches numpy."""
    sdfg, _ = _gemm_sdfg()
    GemmToTensorDot().apply_pass(sdfg, {})
    select_layout_lowering(sdfg, "cpu")
    A = numpy.random.rand(4, 5)
    B = numpy.random.rand(5, 6)
    C = numpy.zeros((4, 6))
    sdfg(A=A, B=B, C=C)
    assert numpy.allclose(C, A @ B)


if __name__ == "__main__":
    test_gemm_to_tensordot_leaves_lowering_unset()
    test_select_cpu_sets_pure()
    test_select_gpu_sets_cutensor()
    test_select_preserves_explicit_choice()
    test_select_bad_device_rejected()
    test_selected_gemm_runs_bitexact_cpu()
    print("select_lowering tests PASS")
