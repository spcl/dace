# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The shared renaming helpers of ``scalar_fission``: the forward must-definition engine and its access views."""

import dace
from dace import subsets
from dace.properties import CodeBlock
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion, LoopRegion
from dace.transformation.passes.scalar_fission import (
    Accesses,
    AccessIndex,
    MustDefinition,
    SubsetAccesses,
    subsets_cover,
)


def write(state, name, subset="0"):
    tasklet = state.add_tasklet("w", {}, {"o"}, "o = 1.0")
    state.add_edge(tasklet, "o", state.add_write(name), None, dace.Memlet(f"{name}[{subset}]"))


def read(state, name, subset="0", out="out"):
    tasklet = state.add_tasklet("r", {"i"}, {"o"}, "o = i")
    state.add_edge(state.add_read(name), None, tasklet, "i", dace.Memlet(f"{name}[{subset}]"))
    state.add_edge(tasklet, "o", state.add_write(out), None, dace.Memlet(f"{out}[0]"))


def loop_sdfg(name: str, shape=(1,)):
    sdfg = dace.SDFG(name)
    sdfg.add_array("t", shape, dace.float64, transient=True)
    sdfg.add_array("out", [1], dace.float64)
    loop = LoopRegion("loop", "i < 4", "i", "i = 0", "i = i + 1")
    sdfg.add_node(loop, is_start_block=True)
    return sdfg, loop


def exposed(sdfg, loop, accesses) -> bool:
    return MustDefinition(accesses, AccessIndex(sdfg)).exposed(loop)


def branch(cond_block: ConditionalBlock, condition: str | None, label: str):
    body = ControlFlowRegion(label)
    state = body.add_state(label + "_s", is_start_block=True)
    cond_block.add_branch(CodeBlock(condition) if condition else None, body)
    return state


def test_a_conditional_defines_only_when_every_branch_and_an_else_write():
    for with_else in (True, False):
        sdfg, loop = loop_sdfg(f"exhaustive_{with_else}")
        cond = ConditionalBlock("if")
        loop.add_node(cond, is_start_block=True)
        write(branch(cond, "i < 2", "then"), "t")
        if with_else:
            write(branch(cond, None, "else"), "t")
        after = loop.add_state("r")
        loop.add_edge(cond, after, dace.InterstateEdge())
        read(after, "t")
        assert exposed(sdfg, loop, Accesses("t")) is not with_else


def test_adjacent_writes_together_cover_a_read():
    sdfg, loop = loop_sdfg("tiles", shape=(8,))
    first = loop.add_state("w", is_start_block=True)
    write(first, "t", "0:4")
    write(first, "t", "4:8")
    second = loop.add_state_after(first, "r")
    read(second, "t", "0:8")
    assert not exposed(sdfg, loop, SubsetAccesses("t", sdfg.facts()))
    whole = subsets.Range.from_string("0:8")
    assert subsets_cover([subsets.Range.from_string("0:4"), subsets.Range.from_string("4:8")], whole, sdfg.facts())
    assert not subsets_cover([subsets.Range.from_string("0:4"), subsets.Range.from_string("5:8")], whole, sdfg.facts())


if __name__ == "__main__":
    test_a_conditional_defines_only_when_every_branch_and_an_else_write()
    test_adjacent_writes_together_cover_a_read()
