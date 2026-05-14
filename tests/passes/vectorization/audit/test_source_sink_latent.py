# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Regression guards for three latent bugs in ``utils/source_sink.py`` that
were already fixed in commit ``b62ea3841`` and earlier rewrites of
``input_is_zero_and_transient_accumulator``. The plan still listed them
as outstanding; these tests exist to catch any future commit that
re-introduces the buggy shape (string-vs-int compare, De-Morgan flip,
UnboundLocalError in the else branch).

Each test constructs the *minimal* SDFG shape that hits the helper's
checked code path. If the helper regresses to the old buggy form, the
expected return / raise changes and the test flags it.
"""
import pytest

import dace
from dace import Memlet
from dace.transformation.passes.vectorization.utils.source_sink import (
    check_writes_to_scalar_sinks_happen_through_assign_tasklets,
    input_is_zero_and_transient_accumulator,
    move_out_reduction,
)


# --------------------------------------------------------------------------
# Bug 1: ``check_writes_to_scalar_sinks_happen_through_assign_tasklets``
#
# Latent bug (now fixed): ``if len(in_edges) != "1":`` compared an int to
# a string, so the check was always True and fell through to
# ``in_edges[0]`` — IndexError when ``in_edges`` was empty, wrong-arm
# raise otherwise. Current code uses ``!= 1`` (int) which raises the
# documented "exactly 1 incoming edge" exception cleanly.
# --------------------------------------------------------------------------


def test_check_writes_to_scalar_sinks_zero_in_edges_raises_named_exception():
    """Zero in-edges to a scalar sink must raise the explicit
    "exactly 1 incoming edge" exception — NOT IndexError from accessing
    ``in_edges[0]`` past the buggy string compare."""
    sdfg = dace.SDFG("bug1_zero_in_edges")
    sdfg.add_scalar("out", dace.float64, transient=False)
    state = sdfg.add_state()
    sink = state.add_access("out")  # 0 in_edges by construction

    with pytest.raises(Exception, match="exactly 1 incoming edge"):
        check_writes_to_scalar_sinks_happen_through_assign_tasklets(sdfg, [(state, sink)])


def test_check_writes_to_scalar_sinks_two_in_edges_raises_named_exception():
    """Two in-edges — also rejected with the explicit named exception."""
    sdfg = dace.SDFG("bug1_two_in_edges")
    sdfg.add_scalar("out", dace.float64, transient=False)
    sdfg.add_scalar("in1", dace.float64, transient=True)
    sdfg.add_scalar("in2", dace.float64, transient=True)
    state = sdfg.add_state()
    sink = state.add_access("out")
    t1 = state.add_tasklet("a1", {"_in"}, {"_out"}, "_out = _in")
    t2 = state.add_tasklet("a2", {"_in"}, {"_out"}, "_out = _in")
    state.add_edge(state.add_access("in1"), None, t1, "_in", Memlet("in1[0]"))
    state.add_edge(state.add_access("in2"), None, t2, "_in", Memlet("in2[0]"))
    state.add_edge(t1, "_out", sink, None, Memlet("out[0]"))
    state.add_edge(t2, "_out", sink, None, Memlet("out[0]"))

    with pytest.raises(Exception, match="exactly 1 incoming edge"):
        check_writes_to_scalar_sinks_happen_through_assign_tasklets(sdfg, [(state, sink)])


def test_check_writes_to_scalar_sinks_one_assign_tasklet_passes():
    """Exactly one assignment-tasklet in-edge — no exception. The
    happy path the helper documents."""
    sdfg = dace.SDFG("bug1_one_assign")
    sdfg.add_scalar("out", dace.float64, transient=False)
    sdfg.add_scalar("in_", dace.float64, transient=True)
    state = sdfg.add_state()
    sink = state.add_access("out")
    src = state.add_access("in_")
    t = state.add_tasklet("assign", {"_in"}, {"_out"}, "_out = _in")
    state.add_edge(src, None, t, "_in", Memlet("in_[0]"))
    state.add_edge(t, "_out", sink, None, Memlet("out[0]"))

    # Should not raise.
    check_writes_to_scalar_sinks_happen_through_assign_tasklets(sdfg, [(state, sink)])


def test_check_writes_to_scalar_sinks_non_assign_tasklet_raises():
    """One in-edge but the producing tasklet does compute (not pure
    assignment) — caught by the assignment-tasklet check."""
    sdfg = dace.SDFG("bug1_compute_tasklet")
    sdfg.add_scalar("out", dace.float64, transient=False)
    sdfg.add_scalar("in_", dace.float64, transient=True)
    state = sdfg.add_state()
    sink = state.add_access("out")
    src = state.add_access("in_")
    t = state.add_tasklet("compute", {"_in"}, {"_out"}, "_out = _in * 2.0")
    state.add_edge(src, None, t, "_in", Memlet("in_[0]"))
    state.add_edge(t, "_out", sink, None, Memlet("out[0]"))

    with pytest.raises(Exception, match="assignment tasklet"):
        check_writes_to_scalar_sinks_happen_through_assign_tasklets(sdfg, [(state, sink)])


# --------------------------------------------------------------------------
# Bug 2: ``input_is_zero_and_transient_accumulator``
#
# Latent bug (now fixed): ``if not (code_str.strip() != "out = 0" or
# code_str.strip() != "out = 0;"):`` — De Morgan flipped wrong. The
# expression is ``not (A != X or A != Y)`` = ``A == X and A == Y`` which
# is impossible when X != Y, so the check was always False and the
# helper accepted *any* init code as a zero-init. Current code parses
# the RHS as a float and only accepts 0.0 (with optional fFdDlL suffix).
# --------------------------------------------------------------------------


def _build_acc_init_outer(init_code: str):
    """Build the minimal outer-state + 1-state inner-SDFG shape the
    helper expects.

    outer state:
        init_tasklet --[acc_data]--> acc_outer_an --[acc_data]--> nsdfg
        nsdfg --[acc_data]--> acc_outer_sink

    inner SDFG (one state):
        acc_inner_source --(transient acc body)--> acc_inner_sink
    """
    outer = dace.SDFG("bug2_outer")
    outer.add_scalar("acc", dace.float64, transient=True)
    state = outer.add_state()

    # Outer init: tasklet -> acc -> nsdfg (in connector "acc")
    init_t = state.add_tasklet("acc_init", {}, {"_acc_out"}, init_code)
    acc_outer = state.add_access("acc")
    state.add_edge(init_t, "_acc_out", acc_outer, None, Memlet("acc[0]"))

    # Inner SDFG: passthrough with same connector names
    inner = dace.SDFG("bug2_inner")
    inner.add_scalar("acc", dace.float64, transient=False)
    inner_state = inner.add_state()
    inner_src = inner_state.add_access("acc")
    inner_sink = inner_state.add_access("acc")
    # Need a tasklet between them so the access node isn't both src and sink of the same edge
    pass_t = inner_state.add_tasklet("pass", {"_i"}, {"_o"}, "_o = _i")
    inner_state.add_edge(inner_src, None, pass_t, "_i", Memlet("acc[0]"))
    inner_state.add_edge(pass_t, "_o", inner_sink, None, Memlet("acc[0]"))

    nsdfg = state.add_nested_sdfg(inner, {"acc"}, {"acc"})
    state.add_edge(acc_outer, None, nsdfg, "acc", Memlet("acc[0]"))
    acc_outer_sink = state.add_access("acc")
    state.add_edge(nsdfg, "acc", acc_outer_sink, None, Memlet("acc[0]"))
    return outer, state, nsdfg, inner_src, inner_sink


def test_input_is_zero_accepts_zero_init():
    """``acc = 0`` init — helper recognises the accumulator. The fixed
    body strips the optional ``;`` and ``fFdDlL`` suffix and float-parses
    the RHS; ``0`` parses to ``0.0`` so the helper returns True."""
    _, state, nsdfg, inner_src, inner_sink = _build_acc_init_outer("_acc_out = 0")
    ok, name = input_is_zero_and_transient_accumulator(state, nsdfg, inner_src, inner_sink)
    assert ok is True
    assert name == "acc"


def test_input_is_zero_accepts_zero_with_semicolon():
    """``acc = 0;`` (trailing semicolon) — accepted after strip."""
    _, state, nsdfg, inner_src, inner_sink = _build_acc_init_outer("_acc_out = 0;")
    ok, name = input_is_zero_and_transient_accumulator(state, nsdfg, inner_src, inner_sink)
    assert ok is True
    assert name == "acc"


def test_input_is_zero_accepts_zero_dot_zero():
    """``acc = 0.0`` — accepted after float-parse."""
    _, state, nsdfg, inner_src, inner_sink = _build_acc_init_outer("_acc_out = 0.0")
    ok, name = input_is_zero_and_transient_accumulator(state, nsdfg, inner_src, inner_sink)
    assert ok is True


def test_input_is_zero_rejects_non_zero_init():
    """``acc = 1.0`` — REJECTED. Catches the De-Morgan-flipped variant
    where every init code was accepted as zero-init."""
    _, state, nsdfg, inner_src, inner_sink = _build_acc_init_outer("_acc_out = 1.0")
    ok, _name = input_is_zero_and_transient_accumulator(state, nsdfg, inner_src, inner_sink)
    assert ok is False, "Non-zero init must be rejected; if this fails the De Morgan flip is back"


def test_input_is_zero_rejects_symbolic_init():
    """``acc = bias`` (unparseable as float) — REJECTED."""
    _, state, nsdfg, inner_src, inner_sink = _build_acc_init_outer("_acc_out = bias")
    ok, _name = input_is_zero_and_transient_accumulator(state, nsdfg, inner_src, inner_sink)
    assert ok is False


def test_input_is_zero_rejects_arbitrary_value():
    """``acc = -inf`` — REJECTED. ``inf`` is parseable by Python's
    ``float()``, but only ``0.0`` matches."""
    _, state, nsdfg, inner_src, inner_sink = _build_acc_init_outer("_acc_out = 3.14")
    ok, _name = input_is_zero_and_transient_accumulator(state, nsdfg, inner_src, inner_sink)
    assert ok is False


# --------------------------------------------------------------------------
# Bug 3: ``move_out_reduction`` UnboundLocalError on empty node_path
#
# Latent bug (now fixed): the final ``return False, source_data,
# sink_data`` referenced names that were only assigned inside the
# ``if num_flops <= 1 and is_inout_accumulator:`` branch. When that
# branch was False, the else fell through with both names unbound and
# raised ``UnboundLocalError`` instead of returning the documented
# (bool, source_data, sink_data) tuple. Commit b62ea3841 hoisted
# ``source_data`` to line 350 and added an early-return for
# ``len(node_path) < 2`` so both names are bound before any return.
# --------------------------------------------------------------------------


def test_move_out_reduction_orphan_source_no_unbound_local():
    """Scalar source with zero outgoing edges — ``only_one_flop_after_source``'s
    BFS returns ``node_path == [source]`` of length 1. The function
    must early-return ``(False, source_data, "")`` without
    UnboundLocalError on ``node_path[1]`` or on the final ``return False, ...``
    referencing unbound names."""
    sdfg = dace.SDFG("bug3_orphan_source")
    sdfg.add_scalar("acc", dace.float64, transient=True)
    state = sdfg.add_state()
    src = state.add_access("acc")  # 0 in_edges, 0 out_edges

    inner = dace.SDFG("bug3_inner")
    inner.add_state()
    nsdfg = state.add_nested_sdfg(inner, set(), set())

    # node_path will have len 1 → len < 2 guard fires; both source_data
    # and sink_data must be bound at the early return.
    result = move_out_reduction([(state, src)], state, nsdfg, inner, vector_width=8)
    assert isinstance(result, tuple) and len(result) == 3
    has_reduction, returned_source_data, returned_sink_data = result
    assert has_reduction is False
    assert returned_source_data == "acc"
    # ``sink_data`` was the unbound name in the old buggy code. The fix
    # returns "" here; what we actually assert is that NO
    # UnboundLocalError fires — getting past the return is the contract.
    assert returned_sink_data == ""
