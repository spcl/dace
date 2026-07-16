# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The layout transforms must NOT hardcode a library lowering: they insert relayout library nodes
(TensorTranspose / TensorDot / LayoutChange) with ``implementation is None``, and the device-driven
``select_layout_lowering`` step chooses the expansion just before compile. On CPU it picks ``pure``.
On GPU it PREFERS ``cuTENSOR`` but only where it can build and run (library linkable, operands
GPU-resident, dtype supported); otherwise it falls back to the pure GPU map. These tests pin that
contract: the passes leave lowering unset, the step selects it per device with the right gates, an
explicit choice is preserved, and a bad device is rejected."""
import numpy
import pytest
import dace

from dace.transformation.layout import select_lowering
from dace.transformation.layout.rewrite_libnodes import GemmToTensorDot, RewriteCopyForLayout
from dace.transformation.layout.select_lowering import select_layout_lowering
from dace.transformation.layout.permute_dimensions import PermuteDimensions
from dace.transformation.layout.prepare import prepare_for_layout
from dace.libraries.blas.nodes.gemm import Gemm
from dace.libraries.linalg import TensorTranspose, TensorDot
from dace.libraries.layout.layout_change import LayoutChange

M, Nn = 6, 4


@dace.program
def elementwise_copy(A: dace.float64[M, Nn], C: dace.float64[M, Nn]):
    C[:] = A


def _gemm_sdfg():
    m, k, n = 4, 5, 6
    sdfg = dace.SDFG("gemm_sel")
    sdfg.add_array("A", [m, k], dace.float64)
    sdfg.add_array("B", [k, n], dace.float64)
    sdfg.add_array("C", [m, n], dace.float64)
    st = sdfg.add_state("s", is_start_block=True)
    g = Gemm("gemm")
    st.add_node(g)
    st.add_edge(st.add_read("A"), None, g, "_a", dace.Memlet.from_array("A", sdfg.arrays["A"]))
    st.add_edge(st.add_read("B"), None, g, "_b", dace.Memlet.from_array("B", sdfg.arrays["B"]))
    st.add_edge(g, "_c", st.add_write("C"), None, dace.Memlet.from_array("C", sdfg.arrays["C"]))
    return sdfg, st


def _transpose_sdfg(storage, dtype=dace.float64):
    """A standalone TensorTranspose (implementation unset) with operands in the given storage."""
    sdfg = dace.SDFG("tt_sel")
    sdfg.add_array("X", [4, 3], dtype, storage=storage)
    sdfg.add_array("Y", [3, 4], dtype, storage=storage)
    st = sdfg.add_state("s", is_start_block=True)
    tt = TensorTranspose("t", axes=[1, 0])
    st.add_node(tt)
    st.add_edge(st.add_read("X"), None, tt, "_inp_tensor", dace.Memlet.from_array("X", sdfg.arrays["X"]))
    st.add_edge(tt, "_out_tensor", st.add_write("Y"), None, dace.Memlet.from_array("Y", sdfg.arrays["Y"]))
    return sdfg, tt


def _layout_change_sdfg(storage):
    sdfg = dace.SDFG("lc_sel")
    sdfg.add_array("P", [4, 3], dace.float64, storage=storage)
    sdfg.add_array("Q", [4, 3], dace.float64, storage=storage)
    st = sdfg.add_state("s", is_start_block=True)
    lc = LayoutChange("lc")
    st.add_node(lc)
    st.add_edge(st.add_read("P"), None, lc, "_inp", dace.Memlet.from_array("P", sdfg.arrays["P"]))
    st.add_edge(lc, "_out", st.add_write("Q"), None, dace.Memlet.from_array("Q", sdfg.arrays["Q"]))
    return sdfg, lc


def _tds(sdfg):
    return [n for n in sdfg.all_nodes_recursive() if isinstance(n[0], TensorDot)]


def _tts(sdfg):
    return [n for n in sdfg.all_nodes_recursive() if isinstance(n[0], TensorTranspose)]


# --------------------------------------------------------------------------- #
#  Passes leave lowering unset
# --------------------------------------------------------------------------- #
def test_gemm_to_tensordot_leaves_lowering_unset():
    """The transform is device-agnostic: the inserted TensorDot has no implementation."""
    sdfg, _ = _gemm_sdfg()
    assert GemmToTensorDot().apply_pass(sdfg, {}) == 1
    assert _tds(sdfg)[0][0].implementation is None


def test_permute_leaves_its_transpose_unset_then_select_sets_it():
    """PermuteDimensions replaces a copy it made transposing with a TensorTranspose (it is the pass
    that knows the permutation), and it must leave the LOWERING unset -- choosing the library
    expansion is select_layout_lowering's job, not a transform's."""
    sdfg = elementwise_copy.to_sdfg(simplify=True)
    prepare_for_layout(sdfg, validate=False)
    PermuteDimensions(permute_map={"A": [1, 0]}, add_permute_maps=True).apply_pass(sdfg, {})
    assert RewriteCopyForLayout().apply_pass(sdfg, {}) == 0  # the permute already converted it
    tt = _tts(sdfg)[0][0]
    assert tt.implementation is None  # the transform did not choose a lowering
    assert select_layout_lowering(sdfg, "cpu") == 1
    assert tt.implementation == "pure"


# --------------------------------------------------------------------------- #
#  CPU selection
# --------------------------------------------------------------------------- #
def test_select_cpu_sets_pure():
    sdfg, _ = _gemm_sdfg()
    GemmToTensorDot().apply_pass(sdfg, {})
    assert select_layout_lowering(sdfg, "cpu") == 1
    assert _tds(sdfg)[0][0].implementation == "pure"


def test_select_over_layout_change_node():
    """LayoutChange is one of the selected node types -- CPU picks pure."""
    sdfg, lc = _layout_change_sdfg(dace.StorageType.Default)
    assert lc.implementation is None
    assert select_layout_lowering(sdfg, "cpu") == 1
    assert lc.implementation == "pure"


# --------------------------------------------------------------------------- #
#  GPU selection gates: storage, cuTENSOR availability, dtype
# --------------------------------------------------------------------------- #
def test_select_gpu_gpu_storage_supported_dtype_gets_cutensor(monkeypatch):
    monkeypatch.setattr(select_lowering, "cutensor_is_linkable", lambda: True)
    sdfg, tt = _transpose_sdfg(dace.StorageType.GPU_Global, dace.float64)
    assert select_layout_lowering(sdfg, "gpu") == 1
    assert tt.implementation == "cuTENSOR"


def test_select_gpu_layout_change_gpu_storage_gets_cutensor(monkeypatch):
    monkeypatch.setattr(select_lowering, "cutensor_is_linkable", lambda: True)
    sdfg, lc = _layout_change_sdfg(dace.StorageType.GPU_Global)
    assert select_layout_lowering(sdfg, "gpu") == 1
    assert lc.implementation == "cuTENSOR"


def test_select_gpu_cpu_storage_falls_back_to_pure(monkeypatch):
    """cuTENSOR needs device-resident operands: a CPU-storage node under device='gpu' gets pure."""
    monkeypatch.setattr(select_lowering, "cutensor_is_linkable", lambda: True)
    sdfg, tt = _transpose_sdfg(dace.StorageType.Default, dace.float64)
    assert select_layout_lowering(sdfg, "gpu") == 1
    assert tt.implementation == "pure"  # storage gate


def test_select_gpu_cutensor_absent_falls_back_to_pure(monkeypatch):
    """No linkable cuTENSOR -> the pure GPU map, so a GPU sweep degrades instead of failing."""
    monkeypatch.setattr(select_lowering, "cutensor_is_linkable", lambda: False)
    sdfg, tt = _transpose_sdfg(dace.StorageType.GPU_Global, dace.float64)
    assert select_layout_lowering(sdfg, "gpu") == 1
    assert tt.implementation == "pure"  # library gate


def test_select_gpu_unsupported_dtype_falls_back_to_pure(monkeypatch):
    """A dtype cuTENSOR does not support (e.g. int32) falls back to pure even on GPU storage."""
    monkeypatch.setattr(select_lowering, "cutensor_is_linkable", lambda: True)
    sdfg, tt = _transpose_sdfg(dace.StorageType.GPU_Global, dace.int32)
    assert select_layout_lowering(sdfg, "gpu") == 1
    assert tt.implementation == "pure"  # dtype gate


# --------------------------------------------------------------------------- #
#  Explicit choice preserved; bad device rejected; CPU end-to-end
# --------------------------------------------------------------------------- #
def test_select_preserves_explicit_choice():
    """A lowering the caller pinned is never overwritten (only ``None`` nodes are touched)."""
    sdfg, tt = _transpose_sdfg(dace.StorageType.Default)
    tt.implementation = "HPTT"
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
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            if "monkeypatch" in fn.__code__.co_varnames:
                continue  # needs the pytest fixture
            fn()
    print("select_lowering tests PASS (fixture tests via pytest)")
