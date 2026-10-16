# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for the divisible-offset ALIGNED fast path (bug 16 performance).

A nonzero begin offset used to force the per-element ``ct.gather`` /
``ct.scatter`` path even when the offset was a multiple of the tile width
(block-aligned). The fast path keeps the aligned ``ct.load`` / ``ct.store``
with the block index shifted by ``offset // width`` whenever the offset is
provably a nonnegative multiple of the width; anything unprovable (odd
constants, opaque symbolic offsets) still takes the gather/scatter path.

Tail safety: the iteration mask is generated for the unshifted iteration
space; a shifted aligned ``ct.load`` pads partial out-of-bounds tiles
(``PaddingMode``) and ``ct.where(_mask, ...)`` discards the padded lanes, so
non-divisible array sizes remain correct. Every block of a valid program
contains at least one in-bounds element (the memlet subset is in-bounds), so
the undefined fully-out-of-bounds tile case cannot occur; nonnegativity of the
shift is required for the fast path.

Codegen-assertion tests need no GPU; numerics tests are marked ``gpu``.
"""
import numpy as np
import pytest

import dace
from dace.dtypes import ScheduleType
from dace.libraries.tileops import TileScatter
from dace.transformation.passes.vectorization.vectorize_cutile import VectorizeCuTile

N = dace.symbol("N")


@dace.program
def _off_w(A: dace.float64[N], B: dace.float64[N]):
    B[8:] = 2.0 * A[8:]


@dace.program
def _off_2w(A: dace.float64[N], B: dace.float64[N]):
    B[16:] = 2.0 * A[16:]


@dace.program
def _off_1(A: dace.float64[N], B: dace.float64[N]):
    B[1:] = 2.0 * A[1:]


@dace.program
def _off_load_only(A: dace.float64[N], B: dace.float64[N]):
    B[:-8] = 2.0 * A[8:]


@dace.program
def _off_store_only(A: dace.float64[N], B: dace.float64[N]):
    B[8:] = 2.0 * A[:-8]


def _lower(prog, widths=(8, )):
    """Lower a program through ``VectorizeCuTile``; return ``(sdfg, code)``."""
    sdfg = prog.to_sdfg(simplify=True)
    VectorizeCuTile(widths=widths).apply_pass(sdfg, {})
    code = "".join(c.clean_code for c in sdfg.generate_code())
    return sdfg, code


# ---------------------------------------------------------------------------
# Generated-code assertions (no GPU required)
# ---------------------------------------------------------------------------


def test_load_offset_width_keeps_aligned_load():
    """Offset == width: block load with the index shifted by 1, no gather."""
    _, code = _lower(_off_w)
    assert "ct.load(_src" in code or "ct.load(_cpy_in" in code
    assert "__pid0 + 1" in code
    assert "ct.gather" not in code


def test_load_offset_two_widths_keeps_aligned_load():
    """Offset == 2*width: block load with the index shifted by 2."""
    _, code = _lower(_off_2w)
    assert "ct.load(_src" in code or "ct.load(_cpy_in" in code
    assert "__pid0 + 2" in code
    assert "ct.gather" not in code


def test_load_offset_one_takes_gather():
    """Offset == 1 (non-divisible): the load must stay on the gather path."""
    _, code = _lower(_off_1)
    assert "ct.gather" in code
    assert "ct.load(_src" not in code


def _expand_store_in_cutile_map(dst_subset: str):
    """Expand an unmasked ``TileScatter`` inside a CuTile map with the given
    destination memlet subset; return the tasklet body.

    :param dst_subset: The ``_dst`` memlet subset (e.g. ``"__i0 + 8:__i0 + 16"``).
    :returns: The cutile-expansion tasklet body string.
    """
    sdfg = dace.SDFG(f"store_fastpath_{abs(hash(dst_subset)) % 10**8}")
    state = sdfg.add_state("main")
    sdfg.add_array("dst", (128, ), dace.float64)
    sdfg.add_array("src_tile", (8, ), dace.float64, transient=True)
    me, mx = state.add_map("m", {"__i0": "0:64:8"}, schedule=ScheduleType.CuTile)
    node = TileScatter(name="S", widths=(8, ))
    state.add_node(node)
    src = state.add_access("src_tile")
    state.add_edge(me, None, src, None, dace.Memlet())
    state.add_edge(src, None, node, "_src", dace.Memlet.from_array("src_tile", sdfg.arrays["src_tile"]))
    state.add_memlet_path(node,
                          mx,
                          state.add_write("dst"),
                          src_conn="_dst",
                          memlet=dace.Memlet(data="dst", subset=dst_subset))
    sdfg.fill_scope_connectors()
    cls = node.implementations["cutile"]
    return cls.expansion(node, state, sdfg).code.as_string


def test_store_offset_width_keeps_aligned_store():
    """Unmasked store with offset == width: ``ct.store`` with a +1 block shift."""
    body = _expand_store_in_cutile_map("__i0 + 8:__i0 + 16")
    assert "ct.store(_dst" in body
    assert "__pid0 + 1" in body
    assert "ct.scatter" not in body


def test_store_offset_two_widths_keeps_aligned_store():
    """Unmasked store with offset == 2*width: ``ct.store`` with a +2 block shift."""
    body = _expand_store_in_cutile_map("__i0 + 16:__i0 + 24")
    assert "ct.store(_dst" in body
    assert "__pid0 + 2" in body
    assert "ct.scatter" not in body


def test_store_offset_nondivisible_takes_scatter():
    """Unmasked store with offset == 3 (non-divisible): scatter with +3 indices."""
    body = _expand_store_in_cutile_map("__i0 + 3:__i0 + 11")
    assert "ct.store" not in body
    assert "ct.scatter" in body
    assert "+ 3" in body


# ---------------------------------------------------------------------------
# End-to-end GPU numerics (fast path AND gather fallback must both be correct)
# ---------------------------------------------------------------------------


def _run_and_check(prog, n, np_slice_out, np_slice_in, seed):
    """Compile ``prog``, run on random data, compare against NumPy.

    :param prog: The ``@dace.program`` under test.
    :param n: Array length.
    :param np_slice_out: Output slice mirroring the program's write.
    :param np_slice_in: Input slice mirroring the program's read.
    :param seed: RNG seed.
    """
    sdfg = prog.to_sdfg(simplify=True)
    VectorizeCuTile(widths=(8, )).apply_pass(sdfg, {})
    csdfg = sdfg.compile()
    rng = np.random.default_rng(seed)
    A = rng.random(n)
    B = np.full(n, -5.0)
    csdfg(A=A, B=B, N=n)
    expected = np.full(n, -5.0)
    expected[np_slice_out] = 2.0 * A[np_slice_in]
    np.testing.assert_allclose(B, expected, rtol=1e-14)


@pytest.mark.gpu
@pytest.mark.parametrize("n", [40, 43, 64, 71])
def test_e2e_offset_width(n):
    """Offset == width, divisible and non-divisible sizes (tail masking)."""
    _run_and_check(_off_w, n, np.s_[8:], np.s_[8:], n)


@pytest.mark.gpu
@pytest.mark.parametrize("n", [48, 51])
def test_e2e_offset_two_widths(n):
    """Offset == 2*width, divisible and non-divisible sizes."""
    _run_and_check(_off_2w, n, np.s_[16:], np.s_[16:], n + 1)


@pytest.mark.gpu
@pytest.mark.parametrize("n", [40, 43])
def test_e2e_offset_one_gather(n):
    """Offset == 1 (gather fallback) stays numerically correct."""
    _run_and_check(_off_1, n, np.s_[1:], np.s_[1:], n + 2)


@pytest.mark.gpu
@pytest.mark.parametrize("n", [40, 43])
def test_e2e_shifted_load_unshifted_store(n):
    """``B[:-8] = 2*A[8:]``: only the load is block-shifted."""
    _run_and_check(_off_load_only, n, np.s_[:-8], np.s_[8:], n + 3)


@pytest.mark.gpu
@pytest.mark.parametrize("n", [40, 43])
def test_e2e_unshifted_load_shifted_store(n):
    """``B[8:] = 2*A[:-8]``: only the store carries the +width offset."""
    _run_and_check(_off_store_only, n, np.s_[8:], np.s_[:-8], n + 4)


if __name__ == "__main__":
    test_load_offset_width_keeps_aligned_load()
    test_load_offset_two_widths_keeps_aligned_load()
    test_load_offset_one_takes_gather()
    test_store_offset_width_keeps_aligned_store()
    test_store_offset_two_widths_keeps_aligned_store()
    test_store_offset_nondivisible_takes_scatter()
    for nn in (40, 43):
        test_e2e_offset_width(nn)
        test_e2e_offset_one_gather(nn)
        test_e2e_shifted_load_unshifted_store(nn)
        test_e2e_unshifted_load_shifted_store(nn)
    test_e2e_offset_two_widths(48)
    print("divisible-offset fast path tests passed")
