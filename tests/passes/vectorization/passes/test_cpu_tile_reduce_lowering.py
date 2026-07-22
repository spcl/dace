# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Lowering + end-to-end tests for the CPU (scalar / avx512 / avx2 / neon / sve)
K=1 ``TileReduce`` backend.

The in-map / per-tile reduction (``acc = sum/prod/min/max over the tile``) lowers
to ``dace::tileops::tile_reduce<T, VLEN, Op>`` (``dace/tile_ops/<isa>.h``), which
delegates to the shared ``horizontal_reduce_<op>`` primitive -- the ISA one-shot
reduce where the hardware has it, the portable log-depth tree otherwise. These
assert the lowering SELECTION, the EMITTED intrinsic call, and (on the host ISA)
the compiled NUMERIC result against a plain NumPy reduction.
"""
import numpy as np
import pytest

import dace
from dace.libraries.tileops import _dispatch
from dace.libraries.tileops.nodes.tile_reduce import (
    ExpandTileReduceAVX512,
    ExpandTileReduceScalar,
    TileReduce,
)

_CPU_ISAS = ["SCALAR", "AVX512", "AVX2", "ARM_NEON", "ARM_SVE"]


@pytest.mark.parametrize("isa", _CPU_ISAS)
def test_reduce_selects_isa_for_full_k1(isa):
    # A full (axis=None), unmasked, K=1 TileReduce lowers to the CPU ISA
    # ``tile_reduce`` intrinsic (the CPU ISA headers now ship it).
    n = TileReduce("t", op="+", widths=(8, ))
    n.target_isa = isa
    impl = _dispatch._ISA_TO_IMPL[isa]
    assert impl in n.implementations
    assert _dispatch.select_tile_implementation(n) == impl


def test_reduce_kge2_falls_back_to_pure_on_cpu():
    # K>=2 always lowers to 'pure' regardless of ISA (the intrinsic is a K==1
    # horizontal fold; the selector routes K>=2 to the pure per-lane expansion).
    n = TileReduce("t", op="+", widths=(4, 8))
    n.target_isa = "AVX512"
    assert _dispatch.select_tile_implementation(n) == "pure"


def _reduce_sdfg(dtype, W, op="+", axis=None, mask=False):
    """A minimal SDFG holding one ``TileReduce`` (Register tiles), for expansion tests."""
    shape = (W, ) if axis is None else (2, W)
    dst_shape = (1, ) if axis is None else (2, )
    op_tag = {"+": "add", "*": "mul", "min": "min", "max": "max"}[op]
    sdfg = dace.SDFG(f"trcpu_{dtype.to_string()}_{W}_{op_tag}_{'full' if axis is None else f'ax{axis}'}_{int(mask)}")
    sdfg.add_array("src", shape, dtype, storage=dace.StorageType.Register, transient=True)
    sdfg.add_array("dst", dst_shape, dtype, storage=dace.StorageType.Register, transient=True)
    st = sdfg.add_state()
    widths = (W, ) if axis is None else (2, W)
    n = TileReduce("tr", widths=widths, op=op, axis=axis, has_mask=mask)
    s, d = st.add_access("src"), st.add_access("dst")
    st.add_edge(s, None, n, "_src", dace.Memlet.from_array("src", sdfg.arrays["src"]))
    st.add_edge(n, "_dst", d, None,
                dace.Memlet("dst[0]") if axis is None else dace.Memlet.from_array("dst", sdfg.arrays["dst"]))
    if mask:
        sdfg.add_array("msk", shape, dace.bool_, storage=dace.StorageType.Register, transient=True)
        m = st.add_access("msk")
        st.add_edge(m, None, n, "_mask", dace.Memlet.from_array("msk", sdfg.arrays["msk"]))
    return sdfg, st, n


def test_reduce_cpu_emits_intrinsic_for_full_k1():
    # A full (axis=None), unmasked, K=1 reduce lowers to the tile_reduce intrinsic
    # on every CPU ISA (the header delegates to horizontal_reduce_<op>).
    sdfg, st, n = _reduce_sdfg(dace.float64, 8, "+")
    assert "dace::tileops::tile_reduce<double, 8, '+'>(_src)" in ExpandTileReduceScalar.expansion(n, st,
                                                                                                  sdfg).code.as_string
    sdfg, st, n = _reduce_sdfg(dace.float32, 16, "+")
    assert "tile_reduce<float, 16, '+'>" in ExpandTileReduceAVX512.expansion(n, st, sdfg).code.as_string
    # ``max`` maps to the 'M' op char (same legend as tile_binop).
    sdfg, st, n = _reduce_sdfg(dace.float64, 8, "max")
    assert "tile_reduce<double, 8, 'M'>" in ExpandTileReduceAVX512.expansion(n, st, sdfg).code.as_string


def test_reduce_cpu_falls_back_to_pure_for_masked_and_axis():
    # The tile_reduce intrinsic takes no mask and only does a full reduction, so a
    # masked reduce or a single-axis reduce delegates to the pure per-lane expansion.
    sdfg, st, n = _reduce_sdfg(dace.float64, 8, "+", mask=True)
    assert "tile_reduce" not in ExpandTileReduceAVX512.expansion(n, st, sdfg).code.as_string
    sdfg, st, n = _reduce_sdfg(dace.float64, 8, "+", axis=1)
    assert "tile_reduce" not in ExpandTileReduceScalar.expansion(n, st, sdfg).code.as_string


def _build_host_reduce_sdfg(W, op, dtype=dace.float64):
    """SRC tile -> host-ISA TileReduce -> DST scalar; compiles + runs on this host."""
    op_tag = {"+": "add", "*": "mul", "min": "min", "max": "max"}[op]
    sdfg = dace.SDFG(f"tile_reduce_hostisa_{W}_{op_tag}")
    sdfg.add_array("SRC", (W, ), dtype, transient=False)
    sdfg.add_array("DST", (1, ), dtype, transient=False)
    state = sdfg.add_state("main")
    node = TileReduce(name="tr", widths=(W, ), op=op, axis=None, has_mask=False)
    state.add_node(node)
    state.add_edge(state.add_access("SRC"), None, node, "_src", dace.Memlet(f"SRC[0:{W}]"))
    state.add_edge(node, "_dst", state.add_access("DST"), None, dace.Memlet("DST[0]"))
    # Force the concrete host ISA backend (not the default 'pure').
    node.target_isa = _dispatch.detect_host_isa()
    node.implementation = _dispatch.select_tile_implementation(node)
    sdfg.expand_library_nodes()
    sdfg.validate()
    return sdfg


@pytest.mark.parametrize("op,ref", [
    ("+", np.sum),
    ("*", np.prod),
    ("min", np.min),
    ("max", np.max),
])
@pytest.mark.parametrize("W", [8, 16, 17])
def test_reduce_cpu_end_to_end_numeric(op, ref, W):
    # The host-ISA horizontal reduce is bit-close to the NumPy reference; reduction
    # reassociation only reorders the fp fold (matching the vectorized-Reduce path's
    # atol), so a tight tolerance suffices.
    sdfg = _build_host_reduce_sdfg(W, op)
    rng = np.random.default_rng(seed=W * 7 + ord(op[0]))
    SRC = (rng.random(W) + 0.5)  # all > 0 so prod stays well-conditioned
    DST = np.zeros(1)
    sdfg(SRC=SRC, DST=DST)
    np.testing.assert_allclose(DST[0], ref(SRC), rtol=1e-12, atol=1e-12)


if __name__ == "__main__":
    for _isa in _CPU_ISAS:
        test_reduce_selects_isa_for_full_k1(_isa)
    test_reduce_kge2_falls_back_to_pure_on_cpu()
    test_reduce_cpu_emits_intrinsic_for_full_k1()
    test_reduce_cpu_falls_back_to_pure_for_masked_and_axis()
    for _op, _ref in [("+", np.sum), ("*", np.prod), ("min", np.min), ("max", np.max)]:
        for _W in [8, 16, 17]:
            test_reduce_cpu_end_to_end_numeric(_op, _ref, _W)
    print("CPU tile-reduce lowering tests passed")
