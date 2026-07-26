# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``SinkStateIntoLoop`` must only sink a between-loops state when replicating it is value-preserving."""
import copy

import numpy as np
import pytest

import dace
from dace.sdfg.sdfg import InterstateEdge
from dace.sdfg.state import LoopRegion, SDFGState
from dace.transformation.passes.canonicalize.loop_fusion import LoopFusion
from dace.transformation.passes.canonicalize.sink_state_into_loop import SinkStateIntoLoop

N = 8


def _loop(label: str, start: str = "1", end: str = f"{N}") -> LoopRegion:
    return LoopRegion(label,
                      condition_expr=f"i < {end}",
                      loop_var="i",
                      initialize_expr=f"i = {start}",
                      update_expr="i = i + 1")


def _build(middle_writes_what_loop2_writes: bool = False,
           middle_reads_what_loop2_writes: bool = False,
           second_end: str = f"{N}") -> dace.SDFG:
    """`loop1{T[i]=T[i-1]+A[i]}` ; `middle{S=3}` ; `loop2{B[i]=B[i-1]+T[i]*S}`.

    Both flags turn the middle state into something the second loop interferes with, so replicating
    it would no longer be idempotent.
    """
    sdfg = dace.SDFG("sink_state")
    for name in ("A", "B", "T"):
        sdfg.add_array(name, [N], dace.float64, transient=(name == "T"))
    sdfg.add_scalar("S", dace.float64, transient=True)
    sdfg.add_symbol("i", dace.int64)

    first = _loop("loop1")
    body1 = first.add_state("body1", is_start_block=True)
    acc = body1.add_tasklet("acc", {"prev", "a"}, {"o"}, "o = prev + a")
    body1.add_edge(body1.add_access("T"), None, acc, "prev", dace.Memlet("T[i - 1]"))
    body1.add_edge(body1.add_access("A"), None, acc, "a", dace.Memlet("A[i]"))
    body1.add_edge(acc, "o", body1.add_access("T"), None, dace.Memlet("T[i]"))

    middle = SDFGState("middle")
    source = "B" if middle_reads_what_loop2_writes else "A"
    seed = middle.add_tasklet("seed", {"v"}, {"o"}, "o = v + 3.0")
    middle.add_edge(middle.add_access(source), None, seed, "v", dace.Memlet(f"{source}[0]"))
    middle.add_edge(seed, "o", middle.add_access("S"), None, dace.Memlet("S[0]"))

    second = _loop("loop2", end=second_end)
    body2 = second.add_state("body2", is_start_block=True)
    step = body2.add_tasklet("step", {"prev", "t", "s"}, {"o"}, "o = prev + t * s")
    body2.add_edge(body2.add_access("B"), None, step, "prev", dace.Memlet("B[i - 1]"))
    body2.add_edge(body2.add_access("T"), None, step, "t", dace.Memlet("T[i]"))
    body2.add_edge(body2.add_access("S"), None, step, "s", dace.Memlet("S[0]"))
    body2.add_edge(step, "o", body2.add_access("B"), None, dace.Memlet("B[i]"))
    if middle_writes_what_loop2_writes:
        clobber = body2.add_tasklet("clobber", {}, {"o"}, "o = 1.0")
        body2.add_edge(clobber, "o", body2.add_access("S"), None, dace.Memlet("S[0]"))

    sdfg.add_node(first, is_start_block=True)
    sdfg.add_node(middle)
    sdfg.add_node(second)
    sdfg.add_edge(first, middle, InterstateEdge())
    sdfg.add_edge(middle, second, InterstateEdge())
    sdfg.validate()
    return sdfg


def _top_level(sdfg: dace.SDFG):
    return [type(b).__name__ for b in sdfg.nodes()]


def _run(sdfg: dace.SDFG) -> dict:
    args = {
        "A": np.arange(1, N + 1, dtype=np.float64),
        "B": np.zeros(N, dtype=np.float64),
    }
    sdfg(**args)
    return args


def test_replicable_middle_state_is_sunk_and_the_loops_then_fuse():
    """The state only reads `A` (which loop2 never writes), so every replica computes the same `S`."""
    sdfg = _build()
    oracle = _run(copy.deepcopy(sdfg))

    assert SinkStateIntoLoop().apply_pass(sdfg, {}) == 1
    assert _top_level(sdfg) == ["LoopRegion", "LoopRegion"], _top_level(sdfg)
    sdfg.validate()

    got = _run(copy.deepcopy(sdfg))
    assert np.array_equal(got["B"], oracle["B"]), f"sinking changed B: {got['B']} != {oracle['B']}"

    # Adjacency was the only thing blocking fusion.
    assert LoopFusion().apply_pass(sdfg, {}) == 1
    assert len([b for b in sdfg.nodes() if isinstance(b, LoopRegion)]) == 1
    assert np.array_equal(_run(sdfg)["B"], oracle["B"]), "fusing after the sink changed B"


@pytest.mark.parametrize("writes,reads", [(True, False), (False, True)])
def test_a_middle_state_the_loop_interferes_with_is_not_sunk(writes: bool, reads: bool):
    """If the loop writes what the state writes (WAW) or reads (stale inputs), replication is unsound."""
    sdfg = _build(middle_writes_what_loop2_writes=writes, middle_reads_what_loop2_writes=reads)
    before = _top_level(sdfg)

    assert SinkStateIntoLoop().apply_pass(sdfg, {}) is None
    assert _top_level(sdfg) == before


def test_a_possibly_empty_loop_does_not_swallow_the_state():
    """Sinking into a loop that may run zero times would drop the state's effect entirely."""
    sdfg = _build(second_end="M")
    sdfg.add_symbol("M", dace.int64)

    assert SinkStateIntoLoop().apply_pass(sdfg, {}) is None
    assert _top_level(sdfg) == ["LoopRegion", "SDFGState", "LoopRegion"]


if __name__ == "__main__":
    pytest.main([__file__])
