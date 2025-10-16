# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.

import dace
from dace.memlet import Memlet
from dace.properties import CodeBlock
from dace.sdfg.sdfg import InterstateEdge
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion, LoopRegion
from dace.transformation.interstate.loop_unroll import LoopUnroll


def _get_sdfg(add_state_before: bool, l: int) -> dace.SDFG:
    sdfg = dace.SDFG("size_5_loop_sdfg")

    for_cfg = LoopRegion(
        label="size_5_loop",
        condition_expr=CodeBlock(f"i < {l}"),
        loop_var="i",
        initialize_expr=CodeBlock("i = 0"),
        update_expr=CodeBlock("i = i + 1"),
        sdfg=sdfg
    )

    if add_state_before:
        _ps = sdfg.add_state(label="pre_s", is_start_block=True)
        sdfg.add_node(for_cfg, is_start_block=False)
        sdfg.add_edge(_ps, for_cfg, InterstateEdge(assignments={}))
    else:
        sdfg.add_node(for_cfg, is_start_block=True)

    body = ControlFlowRegion(label="for_body", sdfg=sdfg, parent=for_cfg)
    for_cfg.add_node(body, is_start_block=True)

    s1 = body.add_state(label="s1", is_start_block=True)

    c1 = ConditionalBlock(label="cond1", sdfg=sdfg, parent=body)
    c_body = ControlFlowRegion(label="if_body", sdfg=sdfg, parent=c1)

    c1.add_branch(condition=CodeBlock("a_sym > 0.0"), branch=c_body)

    body.add_node(c1, is_start_block=False)
    body.add_edge(s1, c1, InterstateEdge(assignments={"a_sym": "A[i]"}))

    s2 = c_body.add_state(label="s2", is_start_block=True)

    b_an = s2.add_access("B")
    a_an = s2.add_access("A")
    t = s2.add_tasklet(
        name="assign",
        inputs={"_in"},
        outputs={"_out"},
        code="_out = _in"
    )
    s2.add_edge(t, "_out", b_an, None, Memlet(expr="B[i]"))
    s2.add_edge(a_an, None, t, "_in", Memlet(expr="A[i]"))

    sdfg.add_array("A", shape=(5,), dtype=dace.float64)
    sdfg.add_array("B" , shape=(5,), dtype=dace.float64)

    sdfg.validate()
    return sdfg

def test_if_block_inside_for():
    sdfg = _get_sdfg(add_state_before=True, l=5)

    sdfg.apply_transformations_repeated(LoopUnroll, validate_all=True)

    loops = {n for n in sdfg.all_control_flow_regions() if isinstance(n, LoopRegion)}
    assert len(loops) == 0

def test_top_level_for():
    sdfg = _get_sdfg(add_state_before=False, l=5)

    sdfg.apply_transformations_repeated(LoopUnroll, validate_all=True)

    loops = {n for n in sdfg.all_control_flow_regions() if isinstance(n, LoopRegion)}
    assert len(loops) == 0

def test_empty_loop():
    sdfg = _get_sdfg(add_state_before=False, l=0)

    sdfg.apply_transformations_repeated(LoopUnroll, validate_all=True)

    loops = {n for n in sdfg.all_control_flow_regions() if isinstance(n, LoopRegion)}
    assert len(loops) == 0

if __name__ == "__main__":
    test_if_block_inside_for()
    test_empty_loop()
    test_top_level_for()