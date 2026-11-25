# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A per-iteration buffer read inside its fill only because an enclosing loop bounds the fill's extent.

QE ``vexx_bp_k_gpu``'s ``newdxx_g``: each ``na`` iteration fills ``aux2[0:Min(256, ngm - 256*iblock + 256)]`` and
reads ``aux2(1)``. The read is inside the fill because ``iblock`` stays within the block count, a range the coverage
test never asked: ``aux2`` was "read before written", carried, and the ``na`` and ``iblock`` loops stayed unproven.
"""

import numpy as np

import dace
from dace.sdfg.state import LoopRegion
from dace.transformation.interstate.loop_to_map import read_before_written

H, Q = dace.symbol("H"), dace.symbol("Q")


def block_buffer_sdfg() -> tuple[dace.SDFG, LoopRegion]:
    """``for p in 1..ceil(H/256): for q in 0..Q-1: buf[0:Min(256, H - 256*p + 256)] = B[..]; out[p - 1, q] = buf[0]``."""
    sdfg = dace.SDFG("block_buffer_cover")
    sdfg.add_array("B", [H], dace.float64)
    sdfg.add_array("out", [(H + 255) // 256, Q], dace.float64)
    sdfg.add_transient("buf", [256], dace.float64)
    outer = LoopRegion("blocks", "p < (H + 255) // 256 + 1", "p", "p = 1", "p = p + 1")
    sdfg.add_node(outer, is_start_block=True)
    inner = LoopRegion("rows", "q < Q", "q", "q = 0", "q = q + 1")
    outer.add_node(inner, is_start_block=True)
    fill = inner.add_state("fill", is_start_block=True)
    fill.add_mapped_tasklet(
        "fill",
        {"r": "0:Min(256, H - 256*p + 256)"},
        {"b": dace.Memlet("B[256*p - 256 + r]")},
        "o = b",
        {"o": dace.Memlet("buf[r]")},
        external_edges=True,
    )
    use = inner.add_state_after(fill, "use")
    take = use.add_tasklet("take", {"x"}, {"y"}, "y = x")
    use.add_edge(use.add_read("buf"), None, take, "x", dace.Memlet("buf[0]"))
    use.add_edge(take, "y", use.add_write("out"), None, dace.Memlet("out[p - 1, q]"))
    return sdfg, inner


def test_a_buffer_read_inside_a_fill_an_enclosing_loop_bounds_is_written_first():
    sdfg, inner = block_buffer_sdfg()
    sdfg.validate()
    assert read_before_written(inner, {"buf"}) == set()
    h, q = 600, 3
    b = np.random.default_rng(0).random(h)
    out = np.zeros(((h + 255) // 256, q))
    sdfg(B=b, out=out, H=h, Q=q)
    assert np.array_equal(out, np.repeat(b[::256][:, None], q, axis=1))


if __name__ == "__main__":
    test_a_buffer_read_inside_a_fill_an_enclosing_loop_bounds_is_written_first()
