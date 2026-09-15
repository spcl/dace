# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Control flow analyses must stay total on a CFG holding blocks unreachable from its start block.

An unreachable block never runs, whether the frontend emitted it as dead code or a transformation
severed it, so the analyses are total and ControlFlowRaising deletes such blocks before codegen.
"""

import numpy as np

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


def test_codegen_drops_a_block_no_path_reaches():
    """ControlFlowRaising runs before codegen and deletes a block unreachable from its region's start block, since
    that block never runs: its write is neither emitted nor executed."""
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

    # ... and codegen drops it instead of refusing.
    code = "".join(obj.clean_code for obj in sdfg.generate_code())
    assert "= 1.0;" in code and "= 2.0;" not in code
    a = np.zeros(1)
    sdfg(a=a)
    assert a[0] == 1.0


if __name__ == "__main__":
    test_block_parent_tree_covers_unreachable_blocks()
    test_dominator_analyses_cover_unreachable_blocks()
    test_dead_code_neither_breaks_simplify_nor_changes_the_result()
    test_codegen_drops_a_block_no_path_reaches()
