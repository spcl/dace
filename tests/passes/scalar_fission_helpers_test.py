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
    SlotAccesses,
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


def test_a_read_after_a_write_in_the_same_iteration_is_covered():
    sdfg, loop = loop_sdfg("covered")
    first = loop.add_state("w", is_start_block=True)
    write(first, "t")
    second = loop.add_state_after(first, "r")
    read(second, "t")
    assert not exposed(sdfg, loop, Accesses("t"))


def test_a_read_before_the_write_is_exposed():
    sdfg, loop = loop_sdfg("carried")
    first = loop.add_state("r", is_start_block=True)
    read(first, "t")
    second = loop.add_state_after(first, "w")
    write(second, "t")
    assert exposed(sdfg, loop, Accesses("t"))


def test_a_write_before_a_conditional_covers_a_read_inside_its_branch():
    sdfg, loop = loop_sdfg("branch_read")
    first = loop.add_state("w", is_start_block=True)
    write(first, "t")
    cond = ConditionalBlock("if")
    loop.add_node(cond)
    loop.add_edge(first, cond, dace.InterstateEdge())
    read(branch(cond, "i < 2", "then"), "t")
    assert not exposed(sdfg, loop, Accesses("t"))


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


def test_a_write_ordered_before_a_read_in_the_same_state_covers_it():
    sdfg, loop = loop_sdfg("ordered")
    state = loop.add_state("s", is_start_block=True)
    tasklet = state.add_tasklet("w", {}, {"o"}, "o = 1.0")
    written = state.add_write("t")
    state.add_edge(tasklet, "o", written, None, dace.Memlet("t[0]"))
    reader = state.add_read("t")
    state.add_nedge(written, reader, dace.Memlet())
    sink = state.add_tasklet("r", {"i"}, {"o"}, "o = i")
    state.add_edge(reader, None, sink, "i", dace.Memlet("t[0]"))
    state.add_edge(sink, "o", state.add_write("out"), None, dace.Memlet("out[0]"))
    assert not exposed(sdfg, loop, Accesses("t"))


def test_a_slot_write_covers_only_its_own_element():
    sdfg, loop = loop_sdfg("slots", shape=(4,))
    first = loop.add_state("w", is_start_block=True)
    write(first, "t", "1")
    second = loop.add_state_after(first, "r")
    read(second, "t", "1")
    read(second, "t", "2", out="out")
    assert not exposed(sdfg, loop, SlotAccesses("t", subsets.Range.from_string("1")))
    assert exposed(sdfg, loop, SlotAccesses("t", subsets.Range.from_string("2")))


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


def test_an_interstate_edge_reading_the_container_before_any_write_is_exposed():
    sdfg, loop = loop_sdfg("edge_read")
    sdfg.add_symbol("c", dace.float64)
    first = loop.add_state("s", is_start_block=True)
    second = loop.add_state("w")
    loop.add_edge(first, second, dace.InterstateEdge(assignments={"c": "t[0]"}))
    write(second, "t")
    assert exposed(sdfg, loop, Accesses("t"))


if __name__ == "__main__":
    test_a_read_after_a_write_in_the_same_iteration_is_covered()
    test_a_read_before_the_write_is_exposed()
    test_a_write_before_a_conditional_covers_a_read_inside_its_branch()
    test_a_conditional_defines_only_when_every_branch_and_an_else_write()
    test_a_write_ordered_before_a_read_in_the_same_state_covers_it()
    test_a_slot_write_covers_only_its_own_element()
    test_adjacent_writes_together_cover_a_read()
    test_an_interstate_edge_reading_the_container_before_any_write_is_exposed()
