# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A block buffer every iteration refills from element 0 stays private to the iteration once the loop is a map.

QE ``vexx_bp_k_gpu``'s ``addusxx_g``: each ``iblock`` iteration clears ``aux1[0:256]`` and fills
``aux1[0:Min(256, ngm - 256*iblock)]``. The extent names the iterator, which LoopToMap took for "each iteration owns
its own part": it left the buffer shared by the map, and the blocks raced on it.
"""

import numpy as np

import dace
from dace.sdfg import nodes
from dace.sdfg.state import LoopRegion
from dace.transformation.interstate import LoopToMap

G = dace.symbol("G")


def block_buffer_sdfg() -> tuple[dace.SDFG, LoopRegion]:
    """``for b in 0..ceil(G/4): buf[0:4] = 0; buf[0:n] += a[4b : 4b + n]; out[4b : 4b + n] = 2 * buf[0:n]``."""
    sdfg = dace.SDFG("block_buffer_private")
    sdfg.add_array("a", [G], dace.float64)
    sdfg.add_array("out", [G], dace.float64)
    sdfg.add_transient("buf", [4], dace.float64)
    blocks = LoopRegion("blocks", "b < (G + 3) // 4", "b", "b = 0", "b = b + 1")
    sdfg.add_node(blocks, is_start_block=True)
    clear = blocks.add_state("clear", is_start_block=True)
    clear.add_mapped_tasklet("clear", {"r": "0:4"}, {}, "o = 0.0", {"o": dace.Memlet("buf[r]")}, external_edges=True)
    fill = blocks.add_state_after(clear, "fill")
    fill.add_mapped_tasklet(
        "fill",
        {"r": "0:Min(4, G - 4*b)"},
        {"x": dace.Memlet("a[4*b + r]"), "y": dace.Memlet("buf[r]")},
        "o = x + y",
        {"o": dace.Memlet("buf[r]")},
        external_edges=True,
    )
    use = blocks.add_state_after(fill, "use")
    use.add_mapped_tasklet(
        "use",
        {"r": "0:Min(4, G - 4*b)"},
        {"x": dace.Memlet("buf[r]")},
        "o = 2 * x",
        {"o": dace.Memlet("out[4*b + r]")},
        external_edges=True,
    )
    return sdfg, blocks


def test_a_block_buffer_refilled_from_zero_is_private_to_each_block():
    sdfg, blocks = block_buffer_sdfg()
    LoopToMap.apply_to(sdfg, loop=blocks)
    (body,) = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.NestedSDFG)]
    assert "buf" not in body.in_connectors and "buf" not in body.out_connectors
    assert body.sdfg.arrays["buf"].transient

    g = 4001
    a = np.random.default_rng(0).random(g)
    out = np.zeros(g)
    sdfg(a=a, out=out, G=g)
    assert np.array_equal(out, 2 * a)


if __name__ == "__main__":
    test_a_block_buffer_refilled_from_zero_is_private_to_each_block()
