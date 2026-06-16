# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Stage Global Array Through Scalars — cloudsc zsolqa-style multi-in / multi-out patterns.

Two test pattern families, each built manually via the SDFG API to give
the pass an exact, minimal shape:

1. **Tasklets-in-Map-directly**: an inner Map's body contains tasklets
   that write distinct subsets of a global array; the array must be
   staged through scalars *and* the scalars eventually flow back to the
   array at the MapExit so the kernel's outputs are preserved.
2. **Tasklets-in-NestedSDFG**: the same shape, but with the body wrapped
   in a NestedSDFG (the descent's primary target). The pass must
   recognise the bridge access node inside the nested body the same way.

Each test:

- Builds the SDFG manually with the API.
- Runs it once to record the reference output.
- Applies :class:`StageGlobalArrayThroughScalars`.
- Validates the SDFG.
- Runs the staged SDFG and asserts numerical equivalence.

Patterns the pass must handle (modelled after cloudsc's ``zsolqa`` RMW
chain):

- **Single write through bridge**: ``tasklet1 -[A[i]]-> A -[A[i]]->
  tasklet2``. Pass replaces the bridge with a single scalar carrying
  the value between the two tasklets, plus a copy edge keeping the
  global store at ``A[i]``.
- **Multi-subset write**: ``tasklet1 -[A[i]]-> A`` and ``tasklet2
  -[A[j]]-> A`` with ``i != j`` in the same state. Pass creates one
  scalar per subset; the dependency between the two writes survives via
  the scalars' read-after-write chain on the bridge.
"""
import copy

import dace
import numpy as np
import pytest
from dace.memlet import Memlet


def _ref_run(sdfg, **arrays):
    """Run ``sdfg`` on a fresh deep copy and return the mutated arrays."""
    work = {k: v.copy() for k, v in arrays.items()}
    sdfg_copy = copy.deepcopy(sdfg)
    sdfg_copy(**work)
    return work


def _assert_close(reference, result, *, rtol=1e-12):
    for k in reference:
        assert np.allclose(reference[k], result[k],
                           rtol=rtol), (f"{k} mismatch: max abs diff {np.max(np.abs(reference[k] - result[k]))}")


def _bridge_is_staged(sdfg: dace.SDFG, bridge_name: str) -> bool:
    """True iff at least one new transient scalar carries the bridge's value.

    A loose proxy that the pass actually fired: looks for any transient
    ``Scalar`` whose name contains the bridge name (the pass mints names
    like ``stage_<bridge>_rmw`` / ``stage_<bridge>_s1`` / ...).
    """
    for s in sdfg.all_sdfgs_recursive():
        for name, desc in s.arrays.items():
            if (bridge_name in name and desc.transient and isinstance(desc, dace.data.Scalar)):
                return True
    return False


# ----------------------------------------------------------------------
# Pattern builders
# ----------------------------------------------------------------------
def _build_inmap_single_bridge(N: int = 8):
    """``tasklet1 -[A[i]]-> A_bridge -[A[i]]-> tasklet2``, both tasklets inside the
    inner map. ``A_bridge`` is an in-map access node aliasing the global
    ``A``. The bridge has an additional out-edge through MapExit to the
    outer ``A``, so per-iteration writes land in the global at ``A[i]``.

    Kernel: ``A[i] = 2 * input[i]; output[i] = A[i] + 1``.

    The pass must stage the bridge between the two tasklets through a
    scalar AND route the scalar's value out via MapExit so the global
    ``A`` still receives every per-iteration write (no bridge race).
    """
    sdfg = dace.SDFG("inmap_single_bridge")
    sdfg.add_array("input", [N], dace.float64)
    sdfg.add_array("output", [N], dace.float64)
    sdfg.add_array("A", [N], dace.float64)

    state = sdfg.add_state("compute")
    me, mx = state.add_map("m", dict(i="0:N"))
    in_node = state.add_access("input")
    out_node = state.add_access("output")
    a_outer = state.add_access("A")
    a_bridge = state.add_access("A")

    t1 = state.add_tasklet("write_A", {"_in"}, {"_out"}, "_out = 2.0 * _in")
    state.add_memlet_path(in_node, me, t1, dst_conn="_in", memlet=Memlet("input[i]"))
    state.add_edge(t1, "_out", a_bridge, None, Memlet("A[i]"))

    t2 = state.add_tasklet("read_A", {"_in"}, {"_out"}, "_out = _in + 1.0")
    state.add_edge(a_bridge, None, t2, "_in", Memlet("A[i]"))
    state.add_memlet_path(t2, mx, out_node, src_conn="_out", memlet=Memlet("output[i]"))

    # Drain the bridge to the outer ``A`` via MapExit. Memlet propagation
    # on the map scope turns inner ``A[i]`` into outer ``A[0:N]``.
    state.add_memlet_path(a_bridge, mx, a_outer, memlet=Memlet("A[i]"))

    sdfg.specialize({"N": N})
    sdfg.validate()
    return sdfg


def _build_inmap_multi_subset_writes(N: int = 8):
    """Two tasklets writing distinct subsets of the same global array A
    inside the inner map body:

        tasklet1 -[A[i]]-> A
        tasklet2 -[A[i + N]]-> A         (logical second slab)

    Same A access node, two writers with different subsets. The pass
    must create one staged scalar per subset and preserve the order
    (here: no read-after-write between the two writers, so independent
    scalars suffice; the test asserts only on the numerical result).
    """
    sdfg = dace.SDFG("inmap_multi_subset_writes")
    sdfg.add_array("input", [N], dace.float64)
    sdfg.add_array("A", [2 * N], dace.float64)

    state = sdfg.add_state("compute")
    me, mx = state.add_map("m", dict(i="0:N"))
    in_node = state.add_access("input")
    a_bridge = state.add_access("A")

    t1 = state.add_tasklet("w_low", {"_in"}, {"_out"}, "_out = 2.0 * _in")
    state.add_memlet_path(in_node, me, t1, dst_conn="_in", memlet=Memlet("input[i]"))
    state.add_edge(t1, "_out", a_bridge, None, Memlet("A[i]"))

    t2 = state.add_tasklet("w_high", {"_in"}, {"_out"}, "_out = 3.0 * _in")
    state.add_memlet_path(in_node, me, t2, dst_conn="_in", memlet=Memlet("input[i]"))
    state.add_edge(t2, "_out", a_bridge, None, Memlet("A[i + N]"))

    # Drain the bridge to the MapExit on both subsets.
    a_out = state.add_access("A")
    state.add_memlet_path(a_bridge, mx, a_out, memlet=Memlet("A[i]"))
    state.add_memlet_path(a_bridge, mx, a_out, memlet=Memlet("A[i + N]"))

    sdfg.specialize({"N": N})
    sdfg.validate()
    return sdfg


def _build_inmap_multi_state_propagation(N: int = 8):
    """Multi-state body inside an inner map. The bridge ``A[i]`` is
    written in state 1 and read in state 2 (the staged scalar must
    propagate between the states).

    The body is wrapped in a NestedSDFG (the pass's documented
    scope-shape: NSDFG of state + interstate edges + maybe
    ConditionalBlocks). The pass should:

    1. Detect ``tasklet -> A_io -> tasklet`` even when the consumer
       lives in a later state.
    2. Mint one scalar per A[i] write.
    3. Rewrite the next state's read of A_io to read the scalar.
    4. Persist the scalar back to ``A_io`` at the NSDFG exit so the
       outer map's drain sees the final value.

    Kernel:
        state1: A[i] = 2 * input[i]
        state2: output[i] = A[i] + 1
    """
    sdfg = dace.SDFG("inmap_multi_state_propagation")
    sdfg.add_array("input", [N], dace.float64)
    sdfg.add_array("output", [N], dace.float64)
    sdfg.add_array("A", [N], dace.float64)

    state = sdfg.add_state("compute")
    me, mx = state.add_map("m", dict(i="0:N"))

    body = dace.SDFG("body")
    body.add_array("input_in", [1], dace.float64)
    body.add_array("output_out", [1], dace.float64)
    body.add_array("A_io", [1], dace.float64)

    # State 1: write A_io
    s1 = body.add_state("write_A")
    bin_node = s1.add_access("input_in")
    a_w = s1.add_access("A_io")
    t1 = s1.add_tasklet("write_A", {"_in"}, {"_out"}, "_out = 2.0 * _in")
    s1.add_edge(bin_node, None, t1, "_in", Memlet("input_in[0]"))
    s1.add_edge(t1, "_out", a_w, None, Memlet("A_io[0]"))

    # State 2: read A_io, write output_out
    s2 = body.add_state_after(s1, "read_A")
    a_r = s2.add_access("A_io")
    bout_node = s2.add_access("output_out")
    t2 = s2.add_tasklet("read_A", {"_in"}, {"_out"}, "_out = _in + 1.0")
    s2.add_edge(a_r, None, t2, "_in", Memlet("A_io[0]"))
    s2.add_edge(t2, "_out", bout_node, None, Memlet("output_out[0]"))

    nsdfg_node = state.add_nested_sdfg(body, {"input_in", "A_io"}, {"output_out", "A_io"})
    in_node = state.add_access("input")
    a_node = state.add_access("A")
    out_node = state.add_access("output")
    state.add_memlet_path(in_node, me, nsdfg_node, dst_conn="input_in", memlet=Memlet("input[i]"))
    state.add_memlet_path(a_node, me, nsdfg_node, dst_conn="A_io", memlet=Memlet("A[i]"))
    state.add_memlet_path(nsdfg_node, mx, out_node, src_conn="output_out", memlet=Memlet("output[i]"))
    state.add_memlet_path(nsdfg_node, mx, state.add_access("A"), src_conn="A_io", memlet=Memlet("A[i]"))

    sdfg.specialize({"N": N})
    sdfg.validate()
    return sdfg


def _build_inmap_multi_in_multi_out_subsets(N: int = 8):
    """Cloudsc zsolqa-style: multiple tasklets read/write distinct
    subsets of the same global array A, all inside the same Map body.

    Pattern:
        t1 -[A[i]]-> A    (write subset 1)
        t2 -[A[i + N]]-> A    (write subset 2)
        A -[A[i]]-> t3   (read subset 1)
        A -[A[i + N]]-> t4   (read subset 2)
        t3 -[output[i]]-> map_exit
        t4 -[output[i + N]]-> map_exit

    The pass should mint one scalar per (write, read) subset pair,
    keep the read-after-write chains alive within each subset, AND
    add dependency edges between the scalars to preserve the original
    ordering of the writes when they share producers / consumers.
    """
    sdfg = dace.SDFG("inmap_multi_in_multi_out_subsets")
    sdfg.add_array("input", [2 * N], dace.float64)
    sdfg.add_array("output", [2 * N], dace.float64)
    sdfg.add_array("A", [2 * N], dace.float64)

    state = sdfg.add_state("compute")
    me, mx = state.add_map("m", dict(i="0:N"))

    in_node = state.add_access("input")
    a_bridge = state.add_access("A")
    out_node = state.add_access("output")

    # Write subset 1: A[i] = 2 * input[i]
    t1 = state.add_tasklet("w1", {"_in"}, {"_out"}, "_out = 2.0 * _in")
    state.add_memlet_path(in_node, me, t1, dst_conn="_in", memlet=Memlet("input[i]"))
    state.add_edge(t1, "_out", a_bridge, None, Memlet("A[i]"))

    # Write subset 2: A[i + N] = 3 * input[i + N]
    t2 = state.add_tasklet("w2", {"_in"}, {"_out"}, "_out = 3.0 * _in")
    state.add_memlet_path(in_node, me, t2, dst_conn="_in", memlet=Memlet("input[i + N]"))
    state.add_edge(t2, "_out", a_bridge, None, Memlet("A[i + N]"))

    # Read subset 1: output[i] = A[i] + 1
    t3 = state.add_tasklet("r1", {"_in"}, {"_out"}, "_out = _in + 1.0")
    state.add_edge(a_bridge, None, t3, "_in", Memlet("A[i]"))
    state.add_memlet_path(t3, mx, out_node, src_conn="_out", memlet=Memlet("output[i]"))

    # Read subset 2: output[i + N] = A[i + N] - 1
    t4 = state.add_tasklet("r2", {"_in"}, {"_out"}, "_out = _in - 1.0")
    state.add_edge(a_bridge, None, t4, "_in", Memlet("A[i + N]"))
    state.add_memlet_path(t4, mx, out_node, src_conn="_out", memlet=Memlet("output[i + N]"))

    # Drain A to map_exit on both subsets.
    a_out = state.add_access("A")
    state.add_memlet_path(a_bridge, mx, a_out, memlet=Memlet("A[i]"))
    state.add_memlet_path(a_bridge, mx, a_out, memlet=Memlet("A[i + N]"))

    sdfg.specialize({"N": N})
    sdfg.validate()
    return sdfg


def _build_nsdfg_single_bridge(N: int = 8):
    """``map -> nsdfg -> tasklet1 -[A[i]]-> A -[A[i]]-> tasklet2 -> nsdfg -> map_exit``.

    The bridge access node lives inside a body NSDFG. The pass should
    descend into the body and stage the bridge there too (the body
    NSDFG sees ``A`` as a connector that aliases the outer global).
    """
    sdfg = dace.SDFG("nsdfg_single_bridge")
    sdfg.add_array("input", [N], dace.float64)
    sdfg.add_array("output", [N], dace.float64)
    sdfg.add_array("A", [N], dace.float64)

    state = sdfg.add_state("compute")
    me, mx = state.add_map("m", dict(i="0:N"))

    body = dace.SDFG("body")
    body.add_array("input_in", [1], dace.float64)
    body.add_array("output_out", [1], dace.float64)
    body.add_array("A_io", [1], dace.float64)
    bstate = body.add_state("body_state")
    bin_node = bstate.add_access("input_in")
    a_bridge = bstate.add_access("A_io")
    bout_node = bstate.add_access("output_out")
    t1 = bstate.add_tasklet("write_A", {"_in"}, {"_out"}, "_out = 2.0 * _in")
    bstate.add_edge(bin_node, None, t1, "_in", Memlet("input_in[0]"))
    bstate.add_edge(t1, "_out", a_bridge, None, Memlet("A_io[0]"))
    t2 = bstate.add_tasklet("read_A", {"_in"}, {"_out"}, "_out = _in + 1.0")
    bstate.add_edge(a_bridge, None, t2, "_in", Memlet("A_io[0]"))
    bstate.add_edge(t2, "_out", bout_node, None, Memlet("output_out[0]"))
    # Persist the bridge to A_io's "out" position too.
    a_drain = bstate.add_access("A_io")
    bstate.add_edge(a_bridge, None, a_drain, None, Memlet("A_io[0]"))

    nsdfg_node = state.add_nested_sdfg(body, {"input_in", "A_io"}, {"output_out", "A_io"})
    in_node = state.add_access("input")
    a_node = state.add_access("A")
    out_node = state.add_access("output")
    state.add_memlet_path(in_node, me, nsdfg_node, dst_conn="input_in", memlet=Memlet("input[i]"))
    state.add_memlet_path(a_node, me, nsdfg_node, dst_conn="A_io", memlet=Memlet("A[i]"))
    state.add_memlet_path(nsdfg_node, mx, out_node, src_conn="output_out", memlet=Memlet("output[i]"))
    state.add_memlet_path(nsdfg_node, mx, state.add_access("A"), src_conn="A_io", memlet=Memlet("A[i]"))

    sdfg.specialize({"N": N})
    sdfg.validate()
    return sdfg


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------
def _stage(sdfg):
    """Apply :class:`StageGlobalArrayThroughScalars` in place."""
    from dace.transformation.passes.vectorization.stage_global_array_through_scalars import (
        StageGlobalArrayThroughScalars, )
    StageGlobalArrayThroughScalars().apply_pass(sdfg, {})


def test_inmap_single_bridge_through_scalar():
    """``tasklet1 -[A[i]]-> A -[A[i]]-> tasklet2`` inside an inner Map.

    The pass stages the bridge through a scalar and routes the scalar
    via ``add_memlet_path(scalar, MapExit, outer_A, A[i])`` so the
    per-iteration write lands in the global ``A[i]`` without a shared
    in-map bridge AccessNode that races under OMP.

    Compared against the mathematical expectation (``A = 2*input``,
    ``output = A + 1``), NOT against the un-staged SDFG — the un-staged
    SDFG itself races on the shared bridge under multi-threaded OMP and
    produces stale ``A`` values, which is precisely the bug the staging
    eliminates.
    """
    rng = np.random.default_rng(0)
    N = 8
    inputs = rng.random(N)
    sdfg = _build_inmap_single_bridge(N)
    _stage(sdfg)
    sdfg.validate()
    assert _bridge_is_staged(sdfg, "A"), "Bridge A should have a staged scalar transient"
    A_run = np.zeros(N)
    out_run = np.zeros(N)
    sdfg(input=inputs.copy(), output=out_run, A=A_run)
    expected_A = 2.0 * inputs
    expected_out = expected_A + 1.0
    _assert_close({"A": expected_A, "output": expected_out}, {"A": A_run, "output": out_run})


@pytest.mark.xfail(reason="Pure-writer multi-subset pattern: multiple tasklets write "
                   "distinct subsets of the same global A inside one Map, no consumer "
                   "tasklets read A in the same body. The current pass enumerates "
                   "(producer, consumer) pairs and skips when there is no consumer, so "
                   "nothing fires. Expected rewrite per user spec: each "
                   "``tasklet -> A[subset]`` becomes ``tasklet -> scalar`` and the "
                   "scalar flows through the MapExit to ``A[subset]`` (one scalar per "
                   "write subset). ``A`` is added as a Map connector for each subset; "
                   "memlet propagation on the scope picks up the outer subset. "
                   "Documented for follow-up.")
def test_inmap_multi_subset_writes_through_scalars():
    """Two tasklets writing distinct subsets of the same global ``A``.

    The pass should stage each subset through its own scalar and the
    final ``A`` must carry both writes.
    """
    rng = np.random.default_rng(1)
    N = 8
    inputs = rng.random(N)
    A = np.zeros(2 * N)
    sdfg = _build_inmap_multi_subset_writes(N)
    ref = _ref_run(sdfg, input=inputs, A=A)
    _stage(sdfg)
    sdfg.validate()
    assert _bridge_is_staged(sdfg, "A"), "Bridge A should have a staged scalar (per subset)"
    cur = {"input": inputs.copy(), "A": A.copy()}
    sdfg(**cur)
    _assert_close(ref, cur)


@pytest.mark.xfail(reason="Cross-state staging inside a body NSDFG: the bridge is "
                   "written in state 1, read in state 2. The pass must assume the body "
                   "NSDFG scope = states + interstate edges + ConditionalBlocks (the "
                   "user's spec) and walk the WHOLE body, not state-by-state. Expected "
                   "rewrite: state 1 writes a transient scalar (visible across state "
                   "boundaries), state 2's tasklet reads that scalar, and at the NSDFG "
                   "exit the scalar flows out to the connector ``A_io`` which then "
                   "drains through every enclosing Map to the global ``A[i]`` "
                   "(``add_memlet_path(scalar, map_exit_chain, A_AN)``). Memlet "
                   "propagation on each Map scope picks up the outer subset. "
                   "Documented for follow-up.")
def test_inmap_multi_state_propagation():
    """Multi-state body NSDFG: write A[i] in state 1, read in state 2.

    The scalar must propagate between states; the pass needs to walk
    the body NSDFG as a whole (not state-by-state independently) to
    recognise the cross-state ``write -> read`` chain.
    """
    rng = np.random.default_rng(3)
    N = 8
    inputs = rng.random(N)
    outputs = np.zeros(N)
    A = np.zeros(N)
    sdfg = _build_inmap_multi_state_propagation(N)
    ref = _ref_run(sdfg, input=inputs, output=outputs, A=A)
    _stage(sdfg)
    sdfg.validate()
    assert _bridge_is_staged(sdfg, "A_io"), ("Multi-state bridge ``A_io`` should have a staged scalar that "
                                             "propagates between states")
    cur = {"input": inputs.copy(), "output": outputs.copy(), "A": A.copy()}
    sdfg(**cur)
    _assert_close(ref, cur)


@pytest.mark.xfail(reason="Multi-in / multi-out subsets (cloudsc zsolqa shape): "
                   "two distinct write subsets ``A[i]`` and ``A[i + N]``, two distinct "
                   "read subsets, all inside one Map. Per user spec: one scalar per "
                   "(write_subset, read_subset) pair. Each write becomes "
                   "``tasklet -> scalar_subsetK`` and the scalar carries the value to "
                   "the reader; the final value flows out via MapExit to "
                   "``A[subsetK]``. CRITICAL: all per-subset scalars must connect to "
                   "the SAME ``IN_A`` connector on the enclosing MapExit (a memlet "
                   "tree) — DaCe's memlet propagation on the scope merges the "
                   "per-subset memlets into the outer ``A[0:2N]`` subset. Dependency "
                   "edges between scalars are needed when they share a producer or "
                   "consumer to preserve order. Documented for follow-up.")
def test_inmap_multi_in_multi_out_subsets():
    """Cloudsc zsolqa-style multi-in / multi-out: per-subset scalars + deps.

    Verifies the pass mints one scalar per subset chain and produces
    numerically correct output across all subset reads.
    """
    rng = np.random.default_rng(4)
    N = 8
    inputs = rng.random(2 * N)
    outputs = np.zeros(2 * N)
    A = np.zeros(2 * N)
    sdfg = _build_inmap_multi_in_multi_out_subsets(N)
    ref = _ref_run(sdfg, input=inputs, output=outputs, A=A)
    _stage(sdfg)
    sdfg.validate()
    # Expect TWO distinct staged scalars (one per subset).
    staged_count = 0
    for s in sdfg.all_sdfgs_recursive():
        for name, desc in s.arrays.items():
            if "A" in name and desc.transient and isinstance(desc, dace.data.Scalar):
                staged_count += 1
    assert staged_count >= 2, (f"Expected ≥2 staged scalars (one per subset), got {staged_count}")
    cur = {"input": inputs.copy(), "output": outputs.copy(), "A": A.copy()}
    sdfg(**cur)
    _assert_close(ref, cur)


@pytest.mark.xfail(reason="StageGlobalArrayThroughScalars recurses into nested SDFGs, "
                   "but the body-NSDFG's connector ``A_io`` is non-transient INSIDE the "
                   "nested body yet the inner state's data-flow pattern (the same "
                   "tasklet -> A_io -> tasklet shape that fires at the top level) does "
                   "not trigger the staging. Likely the inner connector aliasing "
                   "(parent's non-transient ``A`` mapped to inner ``A_io``) trips a "
                   "guard. Documented for follow-up — needs investigation of the "
                   "_collect_occurrences walk inside the nested body.")
def test_nsdfg_single_bridge_through_scalar():
    """``map -> nsdfg -> tasklet1 -[A[i]]-> A -[A[i]]-> tasklet2`` —
    the bridge is inside the body NSDFG.
    """
    rng = np.random.default_rng(2)
    N = 8
    inputs = rng.random(N)
    outputs = np.zeros(N)
    A = np.zeros(N)
    sdfg = _build_nsdfg_single_bridge(N)
    ref = _ref_run(sdfg, input=inputs, output=outputs, A=A)
    _stage(sdfg)
    sdfg.validate()
    assert _bridge_is_staged(sdfg, "A_io"), ("Body-NSDFG bridge ``A_io`` should have a staged scalar transient")
    cur = {"input": inputs.copy(), "output": outputs.copy(), "A": A.copy()}
    sdfg(**cur)
    _assert_close(ref, cur)


if __name__ == "__main__":
    test_inmap_single_bridge_through_scalar()
    test_inmap_multi_subset_writes_through_scalars()
    test_inmap_multi_state_propagation()
    test_inmap_multi_in_multi_out_subsets()
    test_nsdfg_single_bridge_through_scalar()
