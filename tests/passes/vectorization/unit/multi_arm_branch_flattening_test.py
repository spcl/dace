# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A ``ConditionalBlock`` with >=3 branches (or ``if/elif`` without ``else``) becomes a chain of single-arm blocks."""
import itertools

import numpy as np

import dace
from dace.sdfg.state import ConditionalBlock
from dace.transformation.passes.vectorization.branch_normalization import BranchNormalization


def conditional_blocks(sdfg):
    return [b for b in sdfg.all_control_flow_blocks() if isinstance(b, ConditionalBlock)]


def build_three_way(sdfg, arr="A"):
    """A 3-way ConditionalBlock: ``if c0: A[0] = 0 elif c1: A[1] = 1 elif c2: A[2] = 2`` inside ``sdfg``."""
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


def three_way_sdfg(name):
    sdfg = dace.SDFG(name)
    sdfg.add_array("A", [4], dace.float64)
    for s in ("c0", "c1", "c2"):
        sdfg.add_symbol(s, dace.float64)
    return sdfg, build_three_way(sdfg)


def test_three_way_becomes_three_single_arm_blocks():
    sdfg, cb = three_way_sdfg("flat3")
    assert len(cb.branches) == 3

    n = BranchNormalization().flatten_multi_arm_blocks(sdfg)
    assert n == 1

    blocks = conditional_blocks(sdfg)
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
    """Two-arm ``if/else`` (second arm bare) is left for the two-arm rewrites; flattening must not fire."""
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

    n = BranchNormalization().flatten_multi_arm_blocks(sdfg)
    assert n == 0
    assert len(conditional_blocks(sdfg)) == 1


def test_three_way_block_is_lowered_to_first_match_ite_writes():
    sdfg = three_way_sdfg("lowered3")[0]

    BranchNormalization().apply_pass(sdfg, {})

    assert conditional_blocks(sdfg) == []
    sdfg.validate()
    compiled = sdfg.compile()
    for c0, c1, c2 in itertools.product((-1.0, 1.0), repeat=3):
        got = np.full(4, 9.0)
        compiled(A=got, c0=c0, c1=c1, c2=c2)
        want = np.full(4, 9.0)
        first = next((i for i, c in enumerate((c0, c1, c2)) if c > 0), None)
        if first is not None:
            want[first] = first
        np.testing.assert_array_equal(got, want, err_msg=f"c0={c0}, c1={c1}, c2={c2}")
