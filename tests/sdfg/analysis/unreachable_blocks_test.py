# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Control flow analyses must stay total on a CFG holding blocks unreachable from its start block.

Unreachability has two causes that produce the *same* graph shape: dead code the frontend emitted,
which must be tolerated, and a region a transformation severed by mistake, which must not be. No
analysis local to the CFG can tell them apart, so the analyses are total and codegen -- which knows
that by then every remaining block has to be emitted -- is what stays loud. Both halves are pinned
here, because making the analyses total is only safe for as long as the second half holds.
"""

import numpy as np
import pytest

import dace
from dace import SDFG
from dace.sdfg import InterstateEdge
from dace.sdfg.analysis import cfg as cfg_analysis
from dace.sdfg.state import ControlFlowRegion

N = 8


def build_sdfg_with_dead_chain():
    """SDFG computing ``a = b * 2`` alongside a two-block chain nothing branches into.

    The dead chain is wired to itself, so neither block is isolated and ``validate`` accepts the
    SDFG -- the same shape a severed region leaves behind.
    """
    sdfg = SDFG("dead_chain")
    sdfg.add_array("a", [N], dace.float64)
    sdfg.add_array("b", [N], dace.float64)
    sdfg.add_symbol("f", dace.float64)

    entry = sdfg.add_state("entry", is_start_block=True)
    live = sdfg.add_state("live")
    sdfg.add_edge(entry, live, InterstateEdge(assignments={"f": "2.0"}))
    me, mx = live.add_map("scale", {"i": f"0:{N}"})
    t = live.add_tasklet("scale", {"x"}, {"y"}, "y = x * f")
    live.add_memlet_path(live.add_read("b"), me, t, dst_conn="x", memlet=dace.Memlet("b[i]"))
    live.add_memlet_path(t, mx, live.add_write("a"), src_conn="y", memlet=dace.Memlet("a[i]"))

    dead_head = sdfg.add_state("dead_head")
    dead_tail = sdfg.add_state("dead_tail")
    sdfg.add_edge(dead_head, dead_tail, InterstateEdge(assignments={"f": "99.0"}))
    dt = dead_tail.add_tasklet("poison", {}, {"y"}, "y = 999.0")
    dead_tail.add_edge(dt, "y", dead_tail.add_write("a"), None, dace.Memlet("a[0]"))
    return sdfg, dead_head, dead_tail


def test_block_parent_tree_covers_unreachable_blocks():
    """``block_parent_tree`` must map every block, not only those reachable from the start."""
    sdfg, dead_head, dead_tail = build_sdfg_with_dead_chain()
    sdfg.validate()

    ptree = cfg_analysis.block_parent_tree(sdfg)

    assert set(ptree.keys()) == set(sdfg.nodes())
    # A block nothing reaches sits inside no structured control flow, so it roots the tree.
    assert ptree[dead_head] is None
    assert ptree[dead_tail] is None


def test_dominator_analyses_cover_unreachable_blocks():
    """``block_immediate_dominators``/``all_dominators`` feed ``back_edges``; all of them must stay total."""
    sdfg, dead_head, dead_tail = build_sdfg_with_dead_chain()

    idom = cfg_analysis.block_immediate_dominators(sdfg)
    alldoms = cfg_analysis.all_dominators(sdfg)

    assert set(idom.keys()) == set(sdfg.nodes())
    assert idom[dead_head] is dead_head  # Dominated by nothing, hence its own immediate dominator.
    assert set(alldoms.keys()) == set(sdfg.nodes())
    assert alldoms[dead_head] == set()
    assert cfg_analysis.back_edges(sdfg) == []
    assert cfg_analysis.branch_merges(sdfg) == {}


def test_dead_code_neither_breaks_simplify_nor_changes_the_result():
    """Dead code must survive the whole pipeline gracefully and leave the computation alone."""
    sdfg, _, _ = build_sdfg_with_dead_chain()

    sdfg.simplify()

    b = np.arange(N, dtype=np.float64) + 1.0
    a = np.zeros(N, dtype=np.float64)
    sdfg(a=a, b=b)
    assert np.allclose(a, b * 2.0)


def test_codegen_still_refuses_to_drop_a_severed_block():
    """The guard that makes totalizing the analyses safe: codegen must never silently omit a block.

    If this ever starts passing silently, a transformation that disconnects part of a region stops
    being an error and starts being a wrong program, which is precisely what the analyses can no
    longer catch on their own.
    """
    sdfg = SDFG("severed_region")
    sdfg.add_array("a", [1], dace.float64)

    region = ControlFlowRegion("region", sdfg)
    sdfg.add_node(region, is_start_block=True)
    head = region.add_state("head", is_start_block=True)
    ht = head.add_tasklet("head", {}, {"y"}, "y = 1.0")
    head.add_edge(ht, "y", head.add_write("a"), None, dace.Memlet("a[0]"))
    orphan = region.add_state("orphan")
    ot = orphan.add_tasklet("orphan", {}, {"y"}, "y = 2.0")
    orphan.add_edge(ot, "y", orphan.add_write("a"), None, dace.Memlet("a[0]"))
    # No edge into ``orphan``: the shape a transformation leaves when it forgets to reconnect.
    sdfg.validate()

    # The analyses no longer raise on it ...
    assert cfg_analysis.block_parent_tree(region)[orphan] is None

    # ... so codegen is the one that has to refuse, and it has to say which block went missing.
    with pytest.raises(RuntimeError, match="Not all states were generated") as excinfo:
        sdfg.compile()
    assert "orphan" in str(excinfo.value)


if __name__ == "__main__":
    test_block_parent_tree_covers_unreachable_blocks()
    test_dominator_analyses_cover_unreachable_blocks()
    test_dead_code_neither_breaks_simplify_nor_changes_the_result()
    test_codegen_still_refuses_to_drop_a_severed_block()
