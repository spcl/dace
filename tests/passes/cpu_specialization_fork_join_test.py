# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The CPU fork/join cost model and the transfer specialization that follows it.

Pins the decisions of
:class:`~dace.transformation.passes.cpu_specialization.sequentialize_unprofitable_parallel_scopes.SequentializeUnprofitableParallelScopes`
and
:class:`~dace.transformation.passes.cpu_specialization.specialize_cpu_transfers.SpecializeCpuTransfers`
-- the single home of "make it sequential again" on CPU. The canonical form these run on is the
maximally parallel one, so every assertion here is about what the SPECIALIZATION takes away, never
about what canonicalization produced.

Kernel shapes referenced by name below (all measured on a 72-thread node, LEN_2D=768):
``s115``   ``for j: map i in j+1:LEN_2D``  -- 768 regions per call, ~100% team startup.
``s119``   ``for i: map j in 1:LEN_2D``    -- same shape, 767 regions.
jacobi/heat ``for t: map[i, j] over N x N`` -- must KEEP its map: the work per entry outgrows the
number of entries, which is exactly what separates it from the two above.
"""

import contextlib

import numpy as np
import pytest

import dace
from dace import dtypes
from dace.libraries.standard.nodes.copy import CopyLibraryNode
from dace.libraries.standard.nodes.fill import FillLibraryNode
from dace.libraries.standard.nodes.reduce import Reduce
from dace.properties import CodeBlock
from dace.sdfg import nodes
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion, LoopRegion
from dace.transformation.passes.cpu_specialization import (
    SequentializeUnprofitableParallelScopes,
    SpecializeCpuTransfers,
)
from dace.transformation.passes.cpu_specialization.sequentialize_unprofitable_parallel_scopes import (
    min_work_per_region,
    release_one_trip_maps,
)

N = dace.symbol("N")
M = dace.symbol("M")

#: The break-even the size tests pin, in iterations. Deliberately NOT the schema default:
#: ``CalibrateCpuThresholds`` leaves a key alone only while it still differs from the default, so
#: pinning a non-default value is also what stops a ``canonicalize`` call inside the block from
#: recalibrating the threshold out from under the assertion.
PINNED_MIN_WORK_PER_REGION = 128

#: Iteration counts either side of that pin, far enough out that no rounding decides the comparison.
BELOW_BREAK_EVEN = PINNED_MIN_WORK_PER_REGION // 4
ABOVE_BREAK_EVEN = PINNED_MIN_WORK_PER_REGION * 16


@contextlib.contextmanager
def pinned_break_even():
    """Pin the fork/join threshold for the duration of the block, and restore it on the way out.

    The break-even is a property of the machine: ``CalibrateCpuThresholds`` rescales it to the
    host's physical core count -- 64 on a two-core runner, 2304 on a 72-core node -- and writes it
    to the PROCESS config, where it outlives both the pass and the test that triggered it. So a size
    test that hardcoded one machine's number asserts the wrong side of the comparison on every other
    machine, and one that reads the live value instead asserts whatever an earlier test in the same
    worker happened to leave behind. Pinning both sides leaves the cost model as the only variable.
    """
    with dace.config.set_temporary("compiler", "cpu", "parallel_min_work_per_region", value=PINNED_MIN_WORK_PER_REGION):
        assert min_work_per_region() == PINNED_MIN_WORK_PER_REGION, (
            "DACE_compiler_cpu_parallel_min_work_per_region overrides the pin, so these sizes no "
            "longer straddle the threshold; unset it to run the size tests"
        )
        yield


def map_state(container, sdfg, label, ranges, schedule=dtypes.ScheduleType.Default):
    """A state whose only content is one mapped ``a[...] = 1.0`` tasklet over ``ranges``."""
    state = container.add_state(label, is_start_block=len(container.nodes()) == 0)
    index = ", ".join(ranges)
    state.add_mapped_tasklet(
        label,
        {f"__i{i}": r for i, r in enumerate(ranges)},
        {},
        "out = 1.0",
        {"out": dace.Memlet(f"a[{','.join(f'__i{i}' for i in range(len(ranges)))}]")},
        schedule=schedule,
        external_edges=True,
    )
    assert index  # ranges must be non-empty
    return state


def loop(container, label, end, var="it"):
    """A ``for var in 0:end`` LoopRegion added to ``container``."""
    region = LoopRegion(
        label,
        initialize_expr=f"{var} = 0",
        condition_expr=f"{var} < {end}",
        update_expr=f"{var} = {var} + 1",
        loop_var=var,
    )
    container.add_node(region, is_start_block=len(container.nodes()) == 0)
    return region


def one_map_sdfg(name, shape, ranges, loop_end=None, schedule=dtypes.ScheduleType.Default):
    """An SDFG holding one map, optionally wrapped in a ``for it in 0:loop_end`` loop."""
    sdfg = dace.SDFG(name)
    sdfg.add_array("a", shape, dace.float64)
    if loop_end is not None and str(loop_end) not in sdfg.symbols and not str(loop_end).isdigit():
        sdfg.add_symbol(str(loop_end), dace.int64)
    container = sdfg if loop_end is None else loop(sdfg, "outer", loop_end)
    map_state(container, sdfg, "body", ranges, schedule)
    sdfg.validate()
    return sdfg


def schedules(sdfg):
    """Every map label -> schedule in ``sdfg``."""
    return {n.map.label: n.map.schedule for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.MapEntry)}


def test_map_below_break_even_goes_sequential():
    """A constant trip count under the threshold cannot pay for a fork/join, at any nesting."""
    sdfg = one_map_sdfg("small_top_level", [BELOW_BREAK_EVEN], [f"0:{BELOW_BREAK_EVEN}"])
    with pinned_break_even():
        SequentializeUnprofitableParallelScopes().apply_pass(sdfg, {})
    assert schedules(sdfg)["body_map"] == dtypes.ScheduleType.Sequential


def test_map_above_break_even_stays_parallel():
    """A constant trip count above the threshold keeps its region."""
    sdfg = one_map_sdfg("large_top_level", [ABOVE_BREAK_EVEN], [f"0:{ABOVE_BREAK_EVEN}"])
    with pinned_break_even():
        SequentializeUnprofitableParallelScopes().apply_pass(sdfg, {})
    assert schedules(sdfg)["body_map"] == dtypes.ScheduleType.Default


def test_symbolic_extent_stays_parallel():
    """A symbolic extent not provably below break-even keeps its parallel map: a symbol is assumed big."""
    sdfg = one_map_sdfg("symbolic_top_level", [N], ["0:N-1"])
    with pinned_break_even():
        SequentializeUnprofitableParallelScopes().apply_pass(sdfg, {})
    entry = next(n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.MapEntry))
    assert entry.map.schedule == dtypes.ScheduleType.Default


def test_map_re_entered_by_symbolic_loop_stays_parallel():
    """An enclosing loop is not a reason to sequentialize: a region entered ``E`` times costs
    ``E * (fork + work/P)`` against ``E * work``, so ``E`` multiplies both sides and cancels.

    The extent is symbolic, so the map keeps its region -- the same ruling as
    :func:`test_symbolic_extent_stays_parallel`, which the enclosing loop does not overturn. This is npbench ``cavity_flow``'s shape, whose 1,044,484-element pressure map
    was pinned ``Sequential`` by the old work-vs-entries comparison."""
    sdfg = one_map_sdfg("map_in_long_loop", [N], ["0:N"], loop_end="N")
    assert SequentializeUnprofitableParallelScopes().apply_pass(sdfg, {}) is None
    assert schedules(sdfg)["body_map"] == dtypes.ScheduleType.Default


def csr_segment_sdfg(name, in_loop):
    """A map over ``b:e`` with both ends read from ``indptr`` (a CSR row), in a ``for it in 0:N`` loop or at top level."""
    sdfg = dace.SDFG(name)
    sdfg.add_array("indptr", [N + 1], dace.int64)
    sdfg.add_array("a", [M], dace.float64)
    for bound in ("b", "e"):
        sdfg.add_symbol(bound, dace.int64)
    container = loop(sdfg, "rows", "N") if in_loop else sdfg
    row = "it" if in_loop else "0"
    read = container.add_state("read_row", is_start_block=True)
    body = map_state(container, sdfg, "body", ["b:e"])
    container.add_edge(read, body, dace.InterstateEdge(assignments={"b": f"indptr[{row}]", "e": f"indptr[{row} + 1]"}))
    sdfg.validate()
    return sdfg


@pytest.mark.parametrize("in_loop", [True, False])
def test_a_data_segment_map_in_a_sequential_loop_goes_sequential(in_loop):
    """ilu0's shape: inside a sequential loop, a map over a range read from data forks once per outer iteration over a
    few elements (1.5 s against numba's 11 ms). At top level the same map is one region and keeps it."""
    sdfg = csr_segment_sdfg(f"csr_segment_{in_loop}", in_loop)
    SequentializeUnprofitableParallelScopes().apply_pass(sdfg, {})
    entry = next(n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.MapEntry))
    if in_loop:
        assert entry.map.schedule == dtypes.ScheduleType.Sequential
        assert "run sequentially on CPU -- its range b:e - 1 is read from data" in entry.specialization_hint
    else:
        assert entry.map.schedule == dtypes.ScheduleType.Default


def test_map_inside_a_conditional_is_decided():
    """A ConditionalBlock is not a ControlFlowRegion, so its branches need their own descent.

    Without it every map an ``if`` guards was invisible to the cost model -- amg_setup puts its
    whole body under one, and its three stream-compaction loops kept a region per row.
    """
    sdfg = dace.SDFG("small_map_under_if")
    sdfg.add_symbol("N", dace.int64)
    sdfg.add_array("a", [BELOW_BREAK_EVEN], dace.float64)
    body = ControlFlowRegion("body", sdfg=sdfg)
    map_state(body, sdfg, "body", [f"0:{BELOW_BREAK_EVEN}"])
    branch = ConditionalBlock("guard")
    branch.add_branch(CodeBlock("N > 0"), body)
    sdfg.add_node(branch, is_start_block=True)
    sdfg.validate()

    with pinned_break_even():
        SequentializeUnprofitableParallelScopes().apply_pass(sdfg, {})
    assert schedules(sdfg)["body_map"] == dtypes.ScheduleType.Sequential


def test_one_trip_map_releases_the_maps_in_its_body():
    """A peeled first trip ``14:Min(14, N - 1) + 1`` is at most one iteration: it runs sequentially, and the
    map in its body keeps its region through schedule inference and the cost model instead of being pinned
    under it (``cloudsc_monolith``'s first vertical level, 3.7 s serial at 24 threads)."""
    from dace.sdfg import infer_types

    inner = dace.SDFG("one_trip_body")
    inner.add_array("a", [N], dace.float64)
    map_state(inner, inner, "plane", ["0:N"])
    sdfg = dace.SDFG("one_trip_wrapper")
    sdfg.add_array("a", [N], dace.float64)
    state = sdfg.add_state("peeled", is_start_block=True)
    entry, exit_node = state.add_map("peeled", {"k": "14:Min(14, N - 1) + 1"})
    nsdfg = state.add_nested_sdfg(inner, {}, {"a"}, symbol_mapping={"N": N})
    state.add_nedge(entry, nsdfg, dace.Memlet())
    state.add_memlet_path(nsdfg, exit_node, state.add_write("a"), src_conn="a", memlet=dace.Memlet("a[0:N]"))
    sdfg.validate()
    with pinned_break_even():
        assert release_one_trip_maps(sdfg) == 1
        infer_types.set_default_schedule_and_storage_types(sdfg, None)
        SequentializeUnprofitableParallelScopes().apply_pass(sdfg, {})
    assert schedules(sdfg) == {"peeled": dtypes.ScheduleType.Sequential, "plane_map": dtypes.ScheduleType.CPU_Multicore}


def test_map_of_at_most_one_trip_by_the_sdfg_facts_is_released():
    """``K <= 1`` is a registry relation, not a numeric bound of the extent."""
    inner = dace.SDFG("facts_one_trip_body")
    inner.add_array("a", [N], dace.float64)
    map_state(inner, inner, "plane", ["0:N"])
    sdfg = dace.SDFG("facts_one_trip_wrapper")
    sdfg.add_symbol("K", dace.int64)
    sdfg.add_array("a", [N], dace.float64)
    k = dace.symbolic.pystr_to_symbolic("K")
    sdfg.add_symbol_relation(
        dace.symbolic.Relation(dace.symbolic.RelationKind.LE, k, dace.symbolic.pystr_to_symbolic("1"))
    )
    state = sdfg.add_state("peeled", is_start_block=True)
    entry, exit_node = state.add_map("peeled", {"k": "0:K"})
    nsdfg = state.add_nested_sdfg(inner, {}, {"a"}, symbol_mapping={"N": N})
    state.add_nedge(entry, nsdfg, dace.Memlet())
    state.add_memlet_path(nsdfg, exit_node, state.add_write("a"), src_conn="a", memlet=dace.Memlet("a[0:N]"))
    sdfg.validate()
    assert release_one_trip_maps(sdfg) == 1
    assert schedules(sdfg)["peeled"] == dtypes.ScheduleType.Sequential


def staged_map_state(container, label, out):
    """A state with one map over ``0:N`` that computes ``out[i]`` through the scalar transient ``t`` inside it."""
    state = container.add_state(label, is_start_block=len(container.nodes()) == 0)
    entry, exit_node = state.add_map(label, {"i": "0:N"})
    first = state.add_tasklet(f"{label}_first", {}, {"o"}, "o = 2.0")
    second = state.add_tasklet(f"{label}_second", {"x"}, {"o"}, "o = x + 1.0")
    staged = state.add_access("t")
    state.add_nedge(entry, first, dace.Memlet())
    state.add_edge(first, "o", staged, None, dace.Memlet("t[0]"))
    state.add_edge(staged, None, second, "x", dace.Memlet("t[0]"))
    state.add_memlet_path(second, exit_node, state.add_write(out), src_conn="o", memlet=dace.Memlet(f"{out}[i]"))
    return state


def test_released_maps_get_their_own_staged_scalars():
    """Two maps of a one-trip body stage a value through one scalar transient ``t``: code generation declares it
    once outside both, shared by every thread once they run in parallel. Each released map gets its own copy."""
    inner = dace.SDFG("staged_body")
    inner.add_array("a", [N], dace.float64)
    inner.add_array("b", [N], dace.float64)
    inner.add_scalar("t", dace.float64, transient=True)
    first = staged_map_state(inner, "first", "a")
    inner.add_edge(first, staged_map_state(inner, "second", "b"), dace.InterstateEdge())
    sdfg = dace.SDFG("staged_wrapper")
    sdfg.add_array("a", [N], dace.float64)
    sdfg.add_array("b", [N], dace.float64)
    state = sdfg.add_state("peeled", is_start_block=True)
    entry, exit_node = state.add_map("peeled", {"k": "14:Min(14, N - 1) + 1"})
    nsdfg = state.add_nested_sdfg(inner, {}, {"a", "b"}, symbol_mapping={"N": N})
    state.add_nedge(entry, nsdfg, dace.Memlet())
    for name in ("a", "b"):
        state.add_memlet_path(
            nsdfg, exit_node, state.add_write(name), src_conn=name, memlet=dace.Memlet(f"{name}[0:N]")
        )
    sdfg.validate()
    assert release_one_trip_maps(sdfg) == 1
    sdfg.validate()
    assert schedules(sdfg) == {
        "peeled": dtypes.ScheduleType.Sequential,
        "first": dtypes.ScheduleType.CPU_Multicore,
        "second": dtypes.ScheduleType.CPU_Multicore,
    }
    staged = {st.label: {n.data for n in st.data_nodes() if inner.arrays[n.data].transient} for st in inner.states()}
    assert len(staged["first"]) == len(staged["second"]) == 1 and staged["first"] != staged["second"]


def test_a_map_reading_a_shared_scalar_first_stays_under_its_parent():
    """A map whose scope reads the scalar ``t`` another state wrote, without writing it first, would read that
    state's value through a private copy: it is not released."""
    inner = dace.SDFG("carried_body")
    inner.add_array("a", [N], dace.float64)
    inner.add_scalar("t", dace.float64, transient=True)
    init = inner.add_state("init", is_start_block=True)
    init.add_edge(init.add_tasklet("seed", {}, {"o"}, "o = 3.0"), "o", init.add_write("t"), None, dace.Memlet("t[0]"))
    use = inner.add_state("use")
    inner.add_edge(init, use, dace.InterstateEdge())
    reader, reader_exit = use.add_map("reader", {"i": "0:N"})
    carried = use.add_access("t")
    copy_out = use.add_tasklet("copy_out", {"x"}, {"o"}, "o = x")
    use.add_nedge(reader, carried, dace.Memlet())
    use.add_edge(carried, None, copy_out, "x", dace.Memlet("t[0]"))
    use.add_memlet_path(copy_out, reader_exit, use.add_write("a"), src_conn="o", memlet=dace.Memlet("a[i]"))
    inner.validate()
    sdfg = dace.SDFG("carried_wrapper")
    sdfg.add_array("a", [N], dace.float64)
    state = sdfg.add_state("peeled", is_start_block=True)
    entry, exit_node = state.add_map("peeled", {"k": "14:Min(14, N - 1) + 1"})
    nsdfg = state.add_nested_sdfg(inner, {}, {"a"}, symbol_mapping={"N": N})
    state.add_nedge(entry, nsdfg, dace.Memlet())
    state.add_memlet_path(nsdfg, exit_node, state.add_write("a"), src_conn="a", memlet=dace.Memlet("a[0:N]"))
    assert release_one_trip_maps(sdfg) == 1
    assert reader.map.schedule == dtypes.ScheduleType.Default


def test_triangular_map_in_loop_stays_parallel():
    """s115 exactly: ``for j in 0:N { map i in j+1:N }``. The inner map IS data-parallel -- that is
    why it is a Map -- and its extent ``N - j - 1`` is symbolic, so it keeps its region. Reading it at the last iteration (empty) is an artifact of substituting the sweep variable with
    the same value as ``N``, not a property of the nest, so it must not pin the schedule."""
    sdfg = dace.SDFG("triangular")
    sdfg.add_array("a", [N], dace.float64)
    outer = loop(sdfg, "outer", "N", var="j")
    map_state(outer, sdfg, "body", ["j+1:N"])
    sdfg.validate()
    assert SequentializeUnprofitableParallelScopes().apply_pass(sdfg, {}) is None
    assert schedules(sdfg)["body_map"] == dtypes.ScheduleType.Default


def test_wide_map_in_time_loop_stays_parallel():
    """jacobi / heat: ``for t: map[i, j]`` keeps its map. Symbolic work, so above break-even."""
    sdfg = one_map_sdfg("wide_map_in_loop", [N, M], ["0:N", "0:M"], loop_end="N")
    SequentializeUnprofitableParallelScopes().apply_pass(sdfg, {})
    assert schedules(sdfg)["body_map"] == dtypes.ScheduleType.Default


def test_provably_short_loop_is_not_re_entry():
    """A provably short loop around real work keeps the map parallel -- the trip-count awareness
    that ``libnode_is_sequential`` lacks (it counts ANY enclosing loop as re-entry)."""
    sdfg = one_map_sdfg("map_in_short_loop", [N], ["0:N"], loop_end="4")
    SequentializeUnprofitableParallelScopes().apply_pass(sdfg, {})
    assert schedules(sdfg)["body_map"] == dtypes.ScheduleType.Default


def test_nested_parallel_map_is_always_sequentialized():
    """The rule inherited from the canonicalize finalize tail: an explicit ``CPU_Multicore`` map
    inside a parallel map forks a team per outer iteration, at any size."""
    sdfg = dace.SDFG("nested_parallel")
    sdfg.add_array("a", [N, N], dace.float64)
    state = sdfg.add_state()
    outer_entry, outer_exit = state.add_map("outer", dict(i="0:N"), schedule=dtypes.ScheduleType.CPU_Multicore)
    inner_entry, inner_exit = state.add_map("inner", dict(j="0:N"), schedule=dtypes.ScheduleType.CPU_Multicore)
    tasklet = state.add_tasklet("t", {}, {"out"}, "out = 1.0")
    access = state.add_access("a")
    state.add_edge(outer_entry, None, inner_entry, None, dace.Memlet())
    state.add_edge(inner_entry, None, tasklet, None, dace.Memlet())
    state.add_memlet_path(tasklet, inner_exit, outer_exit, access, src_conn="out", memlet=dace.Memlet("a[i, j]"))
    sdfg.validate()

    SequentializeUnprofitableParallelScopes().apply_pass(sdfg, {})

    assert schedules(sdfg)["outer"] == dtypes.ScheduleType.CPU_Multicore
    assert schedules(sdfg)["inner"] == dtypes.ScheduleType.Sequential


def test_nested_parallel_canonical_form_is_valid():
    """Schedules are labels: canonical form may leave parallel maps nested and still validate, so
    resolving the nesting is a specialization and not a correctness repair."""
    sdfg = dace.SDFG("nested_parallel_valid")
    sdfg.add_array("a", [N, N], dace.float64)
    state = sdfg.add_state()
    outer_entry, outer_exit = state.add_map("outer", dict(i="0:N"), schedule=dtypes.ScheduleType.CPU_Multicore)
    inner_entry, inner_exit = state.add_map("inner", dict(j="0:N"), schedule=dtypes.ScheduleType.CPU_Multicore)
    tasklet = state.add_tasklet("t", {}, {"out"}, "out = 1.0")
    access = state.add_access("a")
    state.add_edge(outer_entry, None, inner_entry, None, dace.Memlet())
    state.add_edge(inner_entry, None, tasklet, None, dace.Memlet())
    state.add_memlet_path(tasklet, inner_exit, outer_exit, access, src_conn="out", memlet=dace.Memlet("a[i, j]"))

    sdfg.validate()


def test_threshold_zero_disables_the_cost_model():
    """``parallel_min_work_per_region = 0`` is the A/B lever: only the nested-parallelism rule
    survives, so a re-entered map keeps its region."""
    sdfg = one_map_sdfg("ab_off", [N], ["0:N"], loop_end="N")
    with dace.config.set_temporary("compiler", "cpu", "parallel_min_work_per_region", value=0):
        SequentializeUnprofitableParallelScopes().apply_pass(sdfg, {})
    assert schedules(sdfg)["body_map"] == dtypes.ScheduleType.Default


def reduce_in_loop(name, loop_end):
    """An SDFG with one ``Reduce`` library node inside a ``for it in 0:loop_end`` loop."""
    sdfg = dace.SDFG(name)
    sdfg.add_array("row", [N], dace.float64)
    sdfg.add_array("acc", [1], dace.float64)
    outer = loop(sdfg, "outer", loop_end)
    state = outer.add_state("body", is_start_block=True)
    node = Reduce("reduce_sum", wcr="lambda a, b: a + b", axes=None, identity=0.0)
    node.schedule = dtypes.ScheduleType.CPU_Multicore
    state.add_node(node)
    state.add_edge(state.add_access("row"), None, node, "_in", dace.Memlet("row[0:N]"))
    state.add_edge(node, "_out", state.add_access("acc"), None, dace.Memlet("acc[0]"))
    sdfg.validate()
    return sdfg, node


def test_library_node_in_long_loop_goes_sequential():
    """A library node opens its own region per entry, so a long loop around it is a hazard."""
    sdfg, node = reduce_in_loop("reduce_long_loop", "N")
    SequentializeUnprofitableParallelScopes().apply_pass(sdfg, {})
    assert node.schedule == dtypes.ScheduleType.Sequential


def test_library_node_in_short_loop_stays_parallel():
    """...but a PROVABLY short loop is not re-entry -- the trip-count-aware half of the rule."""
    sdfg, node = reduce_in_loop("reduce_short_loop", "4")
    SequentializeUnprofitableParallelScopes().apply_pass(sdfg, {})
    assert node.schedule == dtypes.ScheduleType.CPU_Multicore


def transfer_sdfg(name, loop_end, kind, subset="0:N"):
    """An SDFG with one copy / memset library node inside a ``for it in 0:loop_end`` loop."""
    sdfg = dace.SDFG(name)
    sdfg.add_array("src", [N], dace.float64)
    sdfg.add_array("dst", [N], dace.float64)
    container = sdfg if loop_end is None else loop(sdfg, "outer", loop_end)
    state = container.add_state("body", is_start_block=True)
    if kind == "copy":
        node = CopyLibraryNode(name="cpy")
        state.add_edge(
            state.add_access("src"), None, node, CopyLibraryNode.INPUT_CONNECTOR_NAME, dace.Memlet(f"src[{subset}]")
        )
        state.add_edge(
            node, CopyLibraryNode.OUTPUT_CONNECTOR_NAME, state.add_access("dst"), None, dace.Memlet(f"dst[{subset}]")
        )
    else:
        node = FillLibraryNode(name="mset")
        state.add_edge(
            node, FillLibraryNode.OUTPUT_CONNECTOR_NAME, state.add_access("dst"), None, dace.Memlet(f"dst[{subset}]")
        )
    sdfg.validate()
    return sdfg, node


@pytest.mark.parametrize("kind,implementation", [("copy", "MemcpyCPU"), ("memset", "CPU")])
def test_re_entered_transfer_collapses_to_libc(kind, implementation):
    """stockham_fft's inner copy: re-entered at a size the loop changes -> Sequential, and a contiguous
    sequential transfer is one libc call, not a serial element loop."""
    sdfg, node = transfer_sdfg(f"{kind}_in_loop", "N", kind, subset="0:it + 1")
    SpecializeCpuTransfers().apply_pass(sdfg, {})
    assert node.schedule == dtypes.ScheduleType.Sequential
    assert node.implementation == implementation


@pytest.mark.parametrize("kind", ["copy", "memset"])
def test_a_loop_that_keeps_the_transfer_size_does_not_sequentialize_it(kind):
    """bicgstab's per-iteration ``N``-element memsets: every entry is the same size, so the entry count cancels
    out of the fork-or-not decision and only the size decides -- not the loop around it."""
    sdfg, node = transfer_sdfg(f"{kind}_invariant_in_loop", "N", kind)
    SpecializeCpuTransfers().apply_pass(sdfg, {})
    assert node.schedule == dtypes.ScheduleType.Default
    assert node.implementation in (None, "Auto")


@pytest.mark.parametrize("kind", ["copy", "memset"])
def test_top_level_transfer_keeps_the_parallel_element_map(kind):
    """A bulk transfer nobody re-enters keeps the canonical parallel element map, and no
    implementation is forced on it -- ``memcpy`` is not the default."""
    sdfg, node = transfer_sdfg(f"{kind}_top_level", None, kind)
    SpecializeCpuTransfers().apply_pass(sdfg, {})
    assert node.schedule == dtypes.ScheduleType.Default
    assert node.implementation in (None, "Auto")


def test_passes_are_idempotent():
    """Both passes run in the canonicalize band AND in the ``finalize_for_target`` tail, so a
    second application must find nothing left to do."""
    sdfg = one_map_sdfg("idempotent_map", [BELOW_BREAK_EVEN], [f"0:{BELOW_BREAK_EVEN}"], loop_end="N")
    with pinned_break_even():
        assert SequentializeUnprofitableParallelScopes().apply_pass(sdfg, {}) == 1
        assert SequentializeUnprofitableParallelScopes().apply_pass(sdfg, {}) is None

    transfers, _node = transfer_sdfg("idempotent_copy", "N", "copy", subset="0:it + 1")
    assert SpecializeCpuTransfers().apply_pass(transfers, {}) == 2
    assert SpecializeCpuTransfers().apply_pass(transfers, {}) is None


def test_strided_re_entered_copy_is_sequential_but_not_memcpy():
    """A non-contiguous copy has no single-``memcpy`` form: it is still sequentialized, but the
    implementation is left to the library node's own selector."""
    sdfg, node = transfer_sdfg("strided_copy_in_loop", "N", "copy", subset="0:it + 1:2")
    SpecializeCpuTransfers().apply_pass(sdfg, {})
    assert node.schedule == dtypes.ScheduleType.Sequential
    assert node.implementation in (None, "Auto")


def test_converting_re_entered_copy_is_not_memcpy():
    """A converting copy is a cast, so a ``memcpy`` cannot do it: the values must be converted."""
    sdfg, _ = transfer_sdfg("converting_copy_in_loop", "N", "copy")
    sdfg.arrays["dst"].dtype = dace.float32
    SpecializeCpuTransfers().apply_pass(sdfg, {})

    src = np.arange(1, 9, dtype=np.float64)
    dst = np.zeros(8, dtype=np.float32)
    sdfg(src=src, dst=dst, N=8)
    np.testing.assert_array_equal(dst, src)


def test_each_distinct_trip_count_is_asked_once():
    """Maps sharing a trip count share one sympy verdict, and every map still gets its own decision."""
    from unittest import mock

    from dace import symbolic

    sdfg = dace.SDFG("shared_trip_counts")
    sdfg.add_symbol("N", dace.int64)
    sdfg.add_array("a", [ABOVE_BREAK_EVEN], dace.float64)
    extents = [BELOW_BREAK_EVEN, "N", BELOW_BREAK_EVEN, "N", ABOVE_BREAK_EVEN, BELOW_BREAK_EVEN]
    for index, extent in enumerate(extents):
        map_state(sdfg, sdfg, f"m{index}", [f"0:{extent}"])
    for first, second in zip(sdfg.nodes(), sdfg.nodes()[1:]):
        sdfg.add_edge(first, second, dace.InterstateEdge())
    sdfg.validate()
    with pinned_break_even():
        with mock.patch.object(symbolic, "ask", side_effect=symbolic.ask) as spy:
            SequentializeUnprofitableParallelScopes().apply_pass(sdfg, {})
    assert spy.call_count == 3, spy.call_args_list
    seq, par = dtypes.ScheduleType.Sequential, dtypes.ScheduleType.Default
    assert schedules(sdfg) == {f"m{i}_map": seq if e == BELOW_BREAK_EVEN else par for i, e in enumerate(extents)}


if __name__ == "__main__":
    test_map_of_at_most_one_trip_by_the_sdfg_facts_is_released()
    test_map_below_break_even_goes_sequential()
    test_map_above_break_even_stays_parallel()
    test_symbolic_extent_stays_parallel()
    test_map_re_entered_by_symbolic_loop_stays_parallel()
    for in_loop in [True, False]:
        test_a_data_segment_map_in_a_sequential_loop_goes_sequential(in_loop)
    test_map_inside_a_conditional_is_decided()
    test_one_trip_map_releases_the_maps_in_its_body()
    test_released_maps_get_their_own_staged_scalars()
    test_a_map_reading_a_shared_scalar_first_stays_under_its_parent()
    test_triangular_map_in_loop_stays_parallel()
    test_wide_map_in_time_loop_stays_parallel()
    test_provably_short_loop_is_not_re_entry()
    test_nested_parallel_map_is_always_sequentialized()
    test_nested_parallel_canonical_form_is_valid()
    test_threshold_zero_disables_the_cost_model()
    test_library_node_in_long_loop_goes_sequential()
    test_library_node_in_short_loop_stays_parallel()
    for kind, implementation in [("copy", "MemcpyCPU"), ("memset", "CPU")]:
        test_re_entered_transfer_collapses_to_libc(kind, implementation)
    for kind in ["copy", "memset"]:
        test_a_loop_that_keeps_the_transfer_size_does_not_sequentialize_it("copy")
    test_a_loop_that_keeps_the_transfer_size_does_not_sequentialize_it("memset")
    test_top_level_transfer_keeps_the_parallel_element_map(kind)
    test_passes_are_idempotent()
    test_strided_re_entered_copy_is_sequential_but_not_memcpy()
    test_converting_re_entered_copy_is_not_memcpy()
    test_each_distinct_trip_count_is_asked_once()
