# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Unit tests for :class:`FlattenBranches`.

``ConditionalBlock`` with >=3 branches (or two-arm ``if/elif`` w/o ``else``)
survives ``SameWriteSetIfElseToITECFG`` / ``BranchNormalization`` and reaches the
tiled body as a scalar per-tile guard. ``FlattenBranches`` desugars it into a chain
of single-arm blocks with accumulated-negation conds (``if/elif/else`` ->
sequential-``if``); single-arm ITE lowering -> per-lane masks.
"""
import dace
from dace.sdfg.state import ConditionalBlock
from dace.transformation.passes.vectorization.flatten_branches import FlattenBranches


def _count_conditional_blocks(sdfg):
    return [b for b in sdfg.all_control_flow_blocks() if isinstance(b, ConditionalBlock)]


def _build_three_way(sdfg, arr="A"):
    """A 3-way ConditionalBlock: ``if c0: .. elif c1: .. elif c2: ..`` inside ``sdfg``."""
    pre = sdfg.add_state("pre", is_start_block=True)
    cb = ConditionalBlock(label="cb", sdfg=sdfg, parent=sdfg)
    sdfg.add_node(cb)
    sdfg.add_edge(pre, cb, dace.InterstateEdge())
    post = sdfg.add_state("post")
    sdfg.add_edge(cb, post, dace.InterstateEdge())
    conds = ["c0 > 0", "c1 > 0", "c2 > 0"]
    for i, c in enumerate(conds):
        body = dace.sdfg.state.ControlFlowRegion(f"arm{i}", sdfg=sdfg)
        st = body.add_state(f"arm{i}_s", is_start_block=True)
        t = st.add_tasklet(f"w{i}", set(), {"o"}, f"o = {i}")
        an = st.add_access(arr)
        st.add_edge(t, "o", an, None, dace.Memlet(f"{arr}[{i}]"))
        cb.add_branch(dace.properties.CodeBlock(c), body)
    return cb


def test_three_way_becomes_three_single_arm_blocks():
    sdfg = dace.SDFG("flat3")
    sdfg.add_array("A", [4], dace.float64)
    for s in ("c0", "c1", "c2"):
        sdfg.add_symbol(s, dace.float64)
    cb = _build_three_way(sdfg)
    assert len(cb.branches) == 3

    n = FlattenBranches().apply_pass(sdfg, {})
    assert n == 1

    blocks = _count_conditional_blocks(sdfg)
    # Three single-arm blocks replace the one 3-way block.
    assert len(blocks) == 3
    for b in blocks:
        assert len(b.branches) == 1

    # Accumulated-negation conditions preserve if/elif first-match semantics.
    conds = sorted(b.branches[0][0].as_string for b in blocks)
    assert any(c == "(c0 > 0)" for c in conds)
    assert any("not (c0 > 0)" in c and "(c1 > 0)" in c and "not (c1 > 0)" not in c for c in conds)
    assert any("not (c0 > 0)" in c and "not (c1 > 0)" in c and "(c2 > 0)" in c for c in conds)
    sdfg.validate()


def test_two_arm_if_else_is_left_untouched():
    """Two-arm ``if/else`` (second arm bare) -> left for SameWriteSetIfElseToITECFG /
    BranchNormalization; FlattenBranches must not fire."""
    sdfg = dace.SDFG("ifelse")
    sdfg.add_array("A", [4], dace.float64)
    sdfg.add_symbol("c0", dace.float64)
    pre = sdfg.add_state("pre", is_start_block=True)
    cb = ConditionalBlock(label="cb", sdfg=sdfg, parent=sdfg)
    sdfg.add_node(cb)
    sdfg.add_edge(pre, cb, dace.InterstateEdge())
    for i, c in enumerate(["c0 > 0", None]):
        body = dace.sdfg.state.ControlFlowRegion(f"arm{i}", sdfg=sdfg)
        st = body.add_state(f"arm{i}_s", is_start_block=True)
        t = st.add_tasklet(f"w{i}", set(), {"o"}, f"o = {i}")
        an = st.add_access("A")
        st.add_edge(t, "o", an, None, dace.Memlet(f"A[{i}]"))
        cb.add_branch(dace.properties.CodeBlock(c) if c else None, body)

    n = FlattenBranches().apply_pass(sdfg, {})
    assert n is None  # no-op
    assert len(_count_conditional_blocks(sdfg)) == 1


if __name__ == "__main__":
    test_three_way_becomes_three_single_arm_blocks()
    test_two_arm_if_else_is_left_untouched()
    print("ok")
