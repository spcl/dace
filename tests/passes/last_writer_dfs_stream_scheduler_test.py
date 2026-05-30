# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for ``LastWriterDFSStreamScheduler`` (v1 stream scheduler).

The worked example: ``State1 -> for cfgh { State2 } -> State3`` where
every state contains two independent GPU kernels (``A`` writes ``X``,
``B`` writes ``Y``). The scheduler must:

  - Assign the ``X``-chain (``A1, A2, A3``) to one stream and the
    ``Y``-chain (``B1, B2, B3``) to a different stream.
  - Insert zero cross-stream events (no consumer reads from a
    different stream).
  - Insert zero interstate host-syncs (no GPU-resident array appears
    in the loop condition or any other interstate-edge expression).
"""
import dace
import pytest

from dace import nodes
from dace.transformation.passes.gpu_specialization.stream_scheduling import (LastWriterDFSStreamScheduler,
                                                                             StreamEventToken)
from dace.transformation.passes.gpu_specialization.stream_scheduling.scheduler import (stream_signatures_match,
                                                                                       lastwriter_stream_join)

# ----------------------------------------------------------------------------
# Pure helpers (LastWriter, lattice join, signature comparison).
# ----------------------------------------------------------------------------


def test_last_writer_signature_match_ignores_event_id():
    a = {"X": StreamEventToken(0, 7), "Y": StreamEventToken(1, 9)}
    b = {"X": StreamEventToken(0, 42), "Y": StreamEventToken(1, 99)}
    assert stream_signatures_match(a, b)


def test_last_writer_signature_mismatch_on_stream_change():
    a = {"X": StreamEventToken(0, 7)}
    b = {"X": StreamEventToken(1, 7)}
    assert not stream_signatures_match(a, b)


def test_lattice_join_keeps_agreeing_stream():
    a = {"X": StreamEventToken(0, 7)}
    b = {"X": StreamEventToken(0, 11)}
    joined = lastwriter_stream_join(a, b)
    assert joined["X"].stream_id == 0
    assert joined["X"].event_id is None  # widened


def test_lattice_join_widens_on_disagreement():
    a = {"X": StreamEventToken(0, 7)}
    b = {"X": StreamEventToken(1, 11)}
    joined = lastwriter_stream_join(a, b)
    assert joined["X"].stream_id == -1
    assert joined["X"].event_id is None


# ----------------------------------------------------------------------------
# Worked example: State1 -> for { State2 } -> State3.
# ----------------------------------------------------------------------------


def _build_worked_example_sdfg():
    """Construct the SDFG used by the chain-of-two test.

    Layout:
        State1   contains two independent kernels (``A1``, ``B1``).
                 A1 writes X, B1 writes Y.
        LoopRegion (for k in range(cfgh)) wrapping
            State2   ``A2`` reads X, writes X; ``B2`` reads Y, writes Y.
        State3   ``A3`` reads X; ``B3`` reads Y. (No writes.)
    """
    sdfg = dace.SDFG("worked_example")
    N = dace.symbol("N", dtype=dace.int32)
    cfgh = dace.symbol("cfgh", dtype=dace.int32)
    sdfg.add_array("X", [N], dace.float64, storage=dace.StorageType.GPU_Global)
    sdfg.add_array("Y", [N], dace.float64, storage=dace.StorageType.GPU_Global)

    def _add_mapped_kernel(state, name, out_arr):
        state.add_mapped_tasklet(
            name,
            map_ranges={"i": f"0:{N}"},
            inputs={},
            code="o = 0.0",
            outputs={"o": dace.Memlet(f"{out_arr}[i]")},
            external_edges=True,
            schedule=dace.ScheduleType.GPU_Device,
        )

    # State 1.
    state1 = sdfg.add_state("state1", is_start_block=True)
    _add_mapped_kernel(state1, "A1", "X")
    _add_mapped_kernel(state1, "B1", "Y")

    # Loop containing State 2. Add the loop to the SDFG *before* its
    # internal state -- otherwise the state's owning-SDFG chain is None
    # and memlet propagation fails on construction.
    loop = dace.sdfg.state.LoopRegion(label="for_cfgh",
                                      condition_expr=f"k < {cfgh}",
                                      loop_var="k",
                                      initialize_expr="k = 0",
                                      update_expr="k = k + 1")
    sdfg.add_node(loop)
    state2 = loop.add_state("state2", is_start_block=True)

    def _add_mapped_inplace(state, name, arr):
        state.add_mapped_tasklet(
            name,
            map_ranges={"i": f"0:{N}"},
            inputs={"v": dace.Memlet(f"{arr}[i]")},
            code="o = v + 1.0",
            outputs={"o": dace.Memlet(f"{arr}[i]")},
            external_edges=True,
            schedule=dace.ScheduleType.GPU_Device,
        )

    _add_mapped_inplace(state2, "A2", "X")
    _add_mapped_inplace(state2, "B2", "Y")

    # State 3 (read-only).
    state3 = sdfg.add_state("state3")

    def _add_mapped_consumer(state, name, arr):
        # Use a fresh "scratch" GPU array as the sink so the body is non-empty.
        scratch_name = f"_scratch_{name}"
        if scratch_name not in sdfg.arrays:
            sdfg.add_array(scratch_name, [1], dace.float64, transient=True, storage=dace.StorageType.GPU_Global)
        state.add_mapped_tasklet(
            name,
            map_ranges={"i": f"0:{N}"},
            inputs={"v": dace.Memlet(f"{arr}[i]")},
            code="o = v",
            outputs={"o": dace.Memlet(f"{scratch_name}[0]", wcr="lambda a, b: a + b")},
            external_edges=True,
            schedule=dace.ScheduleType.GPU_Device,
        )

    _add_mapped_consumer(state3, "A3", "X")
    _add_mapped_consumer(state3, "B3", "Y")

    sdfg.add_edge(state1, loop, dace.InterstateEdge())
    sdfg.add_edge(loop, state3, dace.InterstateEdge())
    return sdfg


def _scheduled_nodes_by_label(sdfg, prefix):
    """Return all ``MapEntry`` nodes whose map name starts with ``prefix``.

    ``add_mapped_tasklet`` names the Map ``<prefix>_map``; the scheduler
    assigns streams to the MapEntry, so we look there.
    """
    matches = []
    for nsdfg in sdfg.all_sdfgs_recursive():
        for state in nsdfg.states():
            for n in state.nodes():
                if isinstance(n, nodes.MapEntry) and n.map.label.startswith(prefix):
                    matches.append(n)
    return matches


def test_worked_example_two_streams_zero_events():
    sdfg = _build_worked_example_sdfg()
    scheduler = LastWriterDFSStreamScheduler()
    assignments = scheduler.assign_streams(sdfg)
    ctx = scheduler._last_ctx

    # Each pair (A_i, B_i) must land on different streams; the chains
    # (A_*) must share a single stream and the (B_*) chain another.
    a_streams = {assignments[n] for n in _scheduled_nodes_by_label(sdfg, "A")}
    b_streams = {assignments[n] for n in _scheduled_nodes_by_label(sdfg, "B")}
    assert len(a_streams) == 1, f"A-chain should share one stream; got {a_streams}"
    assert len(b_streams) == 1, f"B-chain should share one stream; got {b_streams}"
    assert a_streams.isdisjoint(b_streams), \
        f"A and B chains must be on different streams; got A={a_streams}, B={b_streams}"

    # Zero cross-stream events expected: the worked example has no
    # consumer reading from a stream other than its inherited one.
    assert ctx.cross_stream_edges == [], (
        f"Expected no cross-stream events; got {len(ctx.cross_stream_edges)}: {ctx.cross_stream_edges}")

    # Zero interstate host-syncs expected: no GPU-resident array
    # appears in the loop condition (only the symbol ``cfgh``).
    assert ctx.interstate_host_reads == [], (
        f"Expected no interstate host-reads; got {len(ctx.interstate_host_reads)}: {ctx.interstate_host_reads}")

    # No loop fell back to per-iteration syncs.
    assert ctx.per_iteration_loops == set(), (
        f"Expected loop to reach fixed point; got per-iter: {ctx.per_iteration_loops}")


# ----------------------------------------------------------------------------
# Cross-stream event placement (fork case).
# ----------------------------------------------------------------------------


def _build_fork_join_sdfg():
    """``A`` writes ``X``; ``B1`` and ``B2`` both read ``X`` in the same
    state; ``C`` reads from both.

    The scheduler should put ``A``, ``B1``, ``C`` on one chain and
    ``B2`` on a second stream (the fork), then emit a cross-stream
    event from ``B2`` to ``C``.
    """
    sdfg = dace.SDFG("fork_join")
    N = dace.symbol("N", dtype=dace.int32)
    for name in ("X", "Y", "Z", "W"):
        sdfg.add_array(name, [N], dace.float64, storage=dace.StorageType.GPU_Global)
    state = sdfg.add_state("s", is_start_block=True)

    def _kernel(name, in_arr, out_arr):
        ins = {"v": dace.Memlet(f"{in_arr}[i]")} if in_arr else {}
        code = "o = v + 1.0" if in_arr else "o = 0.0"
        state.add_mapped_tasklet(name,
                                 map_ranges={"i": f"0:{N}"},
                                 inputs=ins,
                                 code=code,
                                 outputs={"o": dace.Memlet(f"{out_arr}[i]")},
                                 external_edges=True,
                                 schedule=dace.ScheduleType.GPU_Device)

    _kernel("A", None, "X")
    _kernel("B1", "X", "Y")
    _kernel("B2", "X", "Z")
    state.add_mapped_tasklet("C",
                             map_ranges={"i": f"0:{N}"},
                             inputs={
                                 "y": dace.Memlet("Y[i]"),
                                 "z": dace.Memlet("Z[i]"),
                             },
                             code="o = y + z",
                             outputs={"o": dace.Memlet("W[i]")},
                             external_edges=True,
                             schedule=dace.ScheduleType.GPU_Device)
    return sdfg


def test_fork_join_uses_two_streams_with_events():
    """DFS chain inheritance on a diamond: A forks → B1 stays on A's
    stream, B2 takes a fresh stream → C re-joins on B1's stream.

    Expected event pattern:
        A (s0) ──→ B1 (s0) ──→ C (s0)        (inherited, no events)
                 └→ B2 (s1) ──→ C            (fork; one event in, one out)

    So exactly 2 cross-stream events: (A → B2) and (B2 → C).
    """
    sdfg = _build_fork_join_sdfg()
    scheduler = LastWriterDFSStreamScheduler()
    scheduler.assign_streams(sdfg)
    ctx = scheduler._last_ctx

    # Two distinct streams used.
    streams = set(ctx.assignments.values())
    assert len(streams) == 2, f"Expected exactly 2 streams; got {streams}"

    # Two cross-stream events.
    assert len(
        ctx.cross_stream_edges) == 2, (f"Expected exactly two cross-stream events; got {len(ctx.cross_stream_edges)}: "
                                       f"{ctx.cross_stream_edges}")


# ----------------------------------------------------------------------------
# Conditional branch-merge.
# ----------------------------------------------------------------------------


def _build_conditional_disagreement_sdfg():
    """Conditional whose branches write the same array on different streams.

    The scheduler must pick a join stream and emit a cross-stream event
    for whichever branch disagrees.
    """
    sdfg = dace.SDFG("cond_disagreement")
    N = dace.symbol("N", dtype=dace.int32)
    for name in ("X", "Y"):
        sdfg.add_array(name, [N], dace.float64, storage=dace.StorageType.GPU_Global)
    sdfg.add_scalar("flag", dace.int32, transient=False)
    start = sdfg.add_state("start", is_start_block=True)
    end = sdfg.add_state("end")
    # Seed both branches with different LastWriter streams by writing
    # X and Y from independent kernels in the start state.
    start.add_mapped_tasklet("seed_X",
                             map_ranges={"i": f"0:{N}"},
                             inputs={},
                             code="o = 0.0",
                             outputs={"o": dace.Memlet("X[i]")},
                             external_edges=True,
                             schedule=dace.ScheduleType.GPU_Device)
    start.add_mapped_tasklet("seed_Y",
                             map_ranges={"i": f"0:{N}"},
                             inputs={},
                             code="o = 0.0",
                             outputs={"o": dace.Memlet("Y[i]")},
                             external_edges=True,
                             schedule=dace.ScheduleType.GPU_Device)

    cond = dace.sdfg.state.ConditionalBlock(label="if_flag")
    sdfg.add_node(cond)

    # then-branch: write Z by reading X (inherits X's stream)
    then_cfr = dace.sdfg.state.ControlFlowRegion(label="then_branch")
    cond.add_branch(dace.properties.CodeBlock("flag > 0"), then_cfr)
    then_state = then_cfr.add_state("then_state", is_start_block=True)
    sdfg.add_array("Z", [N], dace.float64, storage=dace.StorageType.GPU_Global)
    then_state.add_mapped_tasklet("then_kernel",
                                  map_ranges={"i": f"0:{N}"},
                                  inputs={"v": dace.Memlet("X[i]")},
                                  code="o = v",
                                  outputs={"o": dace.Memlet("Z[i]")},
                                  external_edges=True,
                                  schedule=dace.ScheduleType.GPU_Device)

    # else-branch: write Z by reading Y (inherits Y's stream)
    else_cfr = dace.sdfg.state.ControlFlowRegion(label="else_branch")
    cond.add_branch(None, else_cfr)
    else_state = else_cfr.add_state("else_state", is_start_block=True)
    else_state.add_mapped_tasklet("else_kernel",
                                  map_ranges={"i": f"0:{N}"},
                                  inputs={"v": dace.Memlet("Y[i]")},
                                  code="o = v",
                                  outputs={"o": dace.Memlet("Z[i]")},
                                  external_edges=True,
                                  schedule=dace.ScheduleType.GPU_Device)

    sdfg.add_edge(start, cond, dace.InterstateEdge())
    sdfg.add_edge(cond, end, dace.InterstateEdge())
    return sdfg


def test_conditional_branches_on_different_streams_emit_event():
    sdfg = _build_conditional_disagreement_sdfg()
    scheduler = LastWriterDFSStreamScheduler()
    scheduler.assign_streams(sdfg)
    ctx = scheduler._last_ctx
    # Branch streams disagree on Z; at least one cross-stream event
    # must be recorded for the join.
    assert ctx.cross_stream_edges, "Expected cross-stream event at conditional join"


if __name__ == "__main__":
    pytest.main([__file__])
