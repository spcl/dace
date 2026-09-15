# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Flattening an N-arm branch must not re-test a guard that an earlier arm has mutated.

``if c0: A0 elif c1: A1 else: A2`` flattened to ``if c0: A0`` / ``if not c0 and c1: A1`` /
``if not c0 and not c1: A2`` re-reads the guard data after ``A0`` wrote it, so a lane whose update
flips the guards takes a second arm as well.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np

import dace
from dace import symbolic
from dace.properties import CodeBlock
from dace.sdfg import nodes
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion
from dace.transformation.passes.vectorization.branch_normalization import BranchNormalization
from dace.transformation.passes.vectorization.same_write_set_if_else_to_ite_cfg import SameWriteSetIfElseToITECFG

LENGTH = 61


def lower_branches(sdfg: dace.SDFG) -> None:
    for lowering in (SameWriteSetIfElseToITECFG(), BranchNormalization()):
        lowering.apply_pass(sdfg, {})


def add_chain(body: dace.SDFG, guards: tuple[str | None, ...]) -> tuple[ConditionalBlock, list[dace.SDFGState]]:
    """``entry -> chain -> done`` with one empty single-state arm per guard; returns the chain and arm states."""
    entry = body.add_state("entry", is_start_block=True)
    chain = ConditionalBlock("chain", sdfg=body, parent=body)
    body.add_node(chain)
    body.add_edge(entry, chain, dace.InterstateEdge())
    body.add_edge(chain, body.add_state("done"), dace.InterstateEdge())
    states = []
    for index, guard in enumerate(guards):
        arm = ControlFlowRegion(f"arm{index}", sdfg=body)
        states.append(arm.add_state(f"arm{index}_s", is_start_block=True))
        chain.add_branch(None if guard is None else CodeBlock(guard), arm)
    return chain, states


def compute(state: dace.SDFGState, source: str | None, source_subset: str, target: str, target_subset: str,
            expression: str) -> None:
    tasklet = state.add_tasklet(f"{target}_from_{source}", {"x": None} if source else {}, {"y": None},
                                f"y = {expression}")
    if source:
        state.add_edge(state.add_read(source), None, tasklet, "x", dace.Memlet(f"{source}[{source_subset}]"))
    state.add_edge(tasklet, "y", state.add_write(target), None, dace.Memlet(f"{target}[{target_subset}]"))


def lanes_program(name: str, body: dace.SDFG, inputs: tuple[str, ...], outputs: tuple[str, ...]) -> dace.SDFG:
    """A parallel map over ``i`` whose nested body is ``body``."""
    sdfg = dace.SDFG(name)
    for array in dict.fromkeys(inputs + outputs):
        sdfg.add_array(array, [LENGTH], dace.float64)
    state = sdfg.add_state("main", is_start_block=True)
    map_entry, map_exit = state.add_map("lanes", {"i": f"0:{LENGTH}"})
    nsdfg = state.add_nested_sdfg(body, dict.fromkeys(inputs), dict.fromkeys(outputs), {"i": "i"})
    for array in inputs:
        state.add_memlet_path(state.add_read(array),
                              map_entry,
                              nsdfg,
                              dst_conn=array,
                              memlet=dace.Memlet(f"{array}[0:{LENGTH}]"))
    for array in outputs:
        state.add_memlet_path(nsdfg,
                              map_exit,
                              state.add_write(array),
                              src_conn=array,
                              memlet=dace.Memlet(f"{array}[0:{LENGTH}]"))
    sdfg.validate()
    return sdfg


def array_guard_program() -> tuple[dace.SDFG, dace.SDFG]:
    """``if a[i] > 0.5: a[i] -= 0.5  elif a[i] > 0.0: b[i] = 2 a[i]  else: b[i] = -1``; returns (program, body)."""
    body = dace.SDFG("array_guard_body")
    for array in ("a", "b"):
        body.add_array(array, [LENGTH], dace.float64)
    body.add_symbol("i", dace.int64)
    chain_states = add_chain(body, ("a[i] > 0.5", "a[i] > 0.0", None))[1]
    compute(chain_states[0], "a", "i", "a", "i", "x - 0.5")
    compute(chain_states[1], "a", "i", "b", "i", "x * 2.0")
    compute(chain_states[2], None, "", "b", "i", "-1.0")
    return lanes_program("array_guard_chain", body, ("a", "b"), ("a", "b")), body


def array_guard_reference(a: np.ndarray, b: np.ndarray) -> None:
    for i in range(LENGTH):
        if a[i] > 0.5:
            a[i] = a[i] - 0.5
        elif a[i] > 0.0:
            b[i] = a[i] * 2.0
        else:
            b[i] = -1.0


def test_every_flattened_guard_reads_a_snapshot_taken_before_the_arms():
    sdfg, body = array_guard_program()

    BranchNormalization().flatten_multi_arm_blocks(sdfg)

    blocks = [block for block in body.all_control_flow_blocks() if isinstance(block, ConditionalBlock)]
    assert len(blocks) == 3 and all(len(block.branches) == 1 for block in blocks)
    guards = [block.branches[0][0].as_string for block in blocks]
    guard_arrays = set().union(*(symbolic.symbols_in_code(guard, potential_symbols=set(body.arrays))
                                 for guard in guards))
    assert guard_arrays, guards
    assert all(body.arrays[name].transient and body.arrays[name].dtype == dace.bool_ for name in guard_arrays), guards
    snapshots = [
        block for block in body.nodes() if isinstance(block, dace.SDFGState) and any(
            isinstance(node, nodes.AccessNode) and node.data == "a" for node in block.nodes())
    ]
    assert len(snapshots) == 2, [block.label for block in snapshots]


def test_array_guard_chain_keeps_first_match_semantics_when_an_arm_mutates_a_later_guard():
    rng = np.random.default_rng(7)
    a, b = rng.uniform(-1.0, 1.0, LENGTH), rng.uniform(-1.0, 1.0, LENGTH)
    want_a, want_b = a.copy(), b.copy()
    array_guard_reference(want_a, want_b)
    sdfg = array_guard_program()[0]

    lower_branches(sdfg)

    sdfg.validate()
    got_a, got_b = a.copy(), b.copy()
    sdfg(a=got_a, b=got_b)
    np.testing.assert_array_equal(got_a, want_a)
    np.testing.assert_array_equal(got_b, want_b)


def test_branch_normalization_alone_keeps_first_match_semantics_when_an_arm_mutates_a_later_guard():
    rng = np.random.default_rng(13)
    a, b = rng.uniform(-1.0, 1.0, LENGTH), rng.uniform(-1.0, 1.0, LENGTH)
    want_a, want_b = a.copy(), b.copy()
    array_guard_reference(want_a, want_b)
    sdfg, body = array_guard_program()

    BranchNormalization().apply_pass(sdfg, {})

    assert not [block for block in body.all_control_flow_blocks() if isinstance(block, ConditionalBlock)]
    sdfg.validate()
    got_a, got_b = a.copy(), b.copy()
    sdfg(a=got_a, b=got_b)
    np.testing.assert_array_equal(got_a, want_a)
    np.testing.assert_array_equal(got_b, want_b)
