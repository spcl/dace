# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""One OpenMP team for a whole sequential loop, asserted on the pragmas that reach the compiler.

:class:`~dace.transformation.passes.cpu_specialization.hoist_parallel_region.HoistParallelRegion`
is a rewrite whose entire effect is in the emitted form: the computation, the iteration-to-thread
assignment and every barrier stay exactly as they were, and only the fork/join per trip of the
enclosing loop goes away. So the assertions here are on the emitted C++ -- ONE ``#pragma omp
parallel`` where there used to be one ``#pragma omp parallel for`` per trip -- paired with a
numeric check, because "it still computes the right answer" passes just as happily on the
un-hoisted form and is not the property at stake.

The refusal tests are the other half, and the more important one. Every statement inside a parallel
region that is not inside a worksharing construct runs once per THREAD, so a loop body carrying one
is a wrong answer waiting to happen; the pass has to leave those graphs untouched, byte for byte.
"""

import os

os.environ.setdefault("OMPI_MCA_pml", "ob1")
os.environ.setdefault("OMPI_MCA_btl", "self,vader")
os.environ.setdefault("UCX_VFS_ENABLE", "n")

import ctypes
import ctypes.util
import re

import numpy as np
import pytest

import dace
from dace import dtypes
from dace.properties import CodeBlock
from dace.sdfg import nodes as nd
from dace.sdfg.dealias import convert_legacy_nested_sdfgs
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion, LoopRegion
from dace.transformation.passes.canonicalize.finalize import finalize_for_target
from dace.transformation.passes.canonicalize.pipeline import canonicalize
from dace.transformation.passes.canonicalize.supply_num_threads import SupplyNumThreads
from dace.transformation.passes.cpu_specialization.band_carried_loops import BandCarriedLoops
from dace.transformation.passes.cpu_specialization.hoist_parallel_region import HoistParallelRegion
from tests.cfg_tree import assert_tree_matches_a_reset, spy_on_resets
from tests.corpus.tsvc import tsvc
from tests.corpus.tsvc.tsvc_numpy import REFERENCES

N = dace.symbol("N")

#: The three TSVC kernels whose canonical form is exactly "sequential loop around one parallel
#: map", i.e. the shape this pass exists for. ``s115`` additionally carries an anti-dependence
#: snapshot at the state's top level, which is the branch that has to be worksharing-wrapped.
#: Kernels the team hoist takes. ``s233`` is NOT among them any more: every one of its
#: loop-carried dependences is at distance zero in the map parameter, so ``BandCarriedLoops``
#: claims it first and gives each thread a whole band, which is the faster shape and the one this
#: file's ``test_the_canonical_form_keeps_every_barrier`` predicted a specialization stage would
#: take. Its numerics and its barrier policy are still checked, over ``TSVC_KERNELS`` below.
HOISTED_KERNELS = ["s115_d_single", "s119_d_single"]

#: Kernels banded instead of hoisted -- one region for the nest either way, but the worksharing
#: construct sits OUTSIDE the carry rather than inside it.
BANDED_KERNELS = ["s233_d_single"]

#: Every TSVC kernel this file finalizes, whichever of the two rewrites claims it.
TSVC_KERNELS = HOISTED_KERNELS + BANDED_KERNELS


def finalized(name, tag):
    """``(kernel, sdfg)`` for one TSVC kernel put through canonicalize and the CPU perf tail."""
    kernel = tsvc.collect(name=name)[0]
    sdfg = tsvc.to_sdfg(kernel, tag, simplify=True)
    canonicalize(sdfg, validate=True)
    finalize_for_target(sdfg, "cpu")
    return kernel, sdfg


def pragmas(sdfg):
    """``(teams, per_trip_regions, worksharing_loops)`` counted in ``sdfg``'s emitted C++.

    A bare ``#pragma omp parallel`` opens a team; ``#pragma omp parallel for`` opens one AND
    distributes, which is the per-trip form this pass replaces; ``#pragma omp for`` distributes
    inside a team already open.
    """
    code = sdfg.generate_code()[0].clean_code
    return (
        len(re.findall(r"#pragma omp parallel(?! for)", code)),
        len(re.findall(r"#pragma omp parallel for", code)),
        len(re.findall(r"#pragma omp for", code)),
    )


def assert_matches_reference(kernel, sdfg):
    """The finalized kernel must reproduce the numpy reference element for element."""
    arrays, call_kwargs = tsvc.make_inputs(kernel)
    ref = {n: a.copy() for n, a in arrays.items()}
    REFERENCES[kernel.name](**ref, **call_kwargs)
    got = {n: a.copy() for n, a in arrays.items()}
    sdfg.compile()(**got, **call_kwargs)
    for n, arr in arrays.items():
        if np.issubdtype(arr.dtype, np.integer):
            continue
        assert np.allclose(ref[n], got[n], equal_nan=True), f"{kernel.name}: value mismatch on {n}"


def loop_region(container, label, end, var="it"):
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


def mapped_state(container, label, target, expr, ranges, schedule=dtypes.ScheduleType.CPU_Multicore, inputs=None):
    """A state whose only content is one mapped ``target[i] = expr`` tasklet over ``ranges``."""
    state = container.add_state(label, is_start_block=len(container.nodes()) == 0)
    params = {f"__i{i}": r for i, r in enumerate(ranges)}
    index = ",".join(params)
    state.add_mapped_tasklet(
        label,
        params,
        {f"in_{name}": dace.Memlet(f"{name}[{index}]") for name in (inputs or ())},
        f"out = {expr}",
        {"out": dace.Memlet(f"{target}[{index}]")},
        schedule=schedule,
        external_edges=True,
    )
    return state


def loop_over_map_sdfg(name):
    """``for it in 0:N { map i in 0:N: a[i] = 1.0 }`` -- the minimal hoistable shape."""
    sdfg = dace.SDFG(name)
    sdfg.add_array("a", [N], dace.float64)
    mapped_state(loop_region(sdfg, "outer", "N"), "body", "a", "1.0", ["0:N"])
    sdfg.validate()
    return sdfg


def teams(sdfg):
    """How many ``CPU_Persistent`` map scopes the pass left in ``sdfg``."""
    return sum(
        1
        for n, _ in sdfg.all_nodes_recursive()
        if isinstance(n, nd.MapEntry) and n.map.schedule == dtypes.ScheduleType.CPU_Persistent
    )


def assert_declined(sdfg, why):
    """The pass must refuse ``sdfg`` AND leave it bit-identical."""
    before = sdfg.to_json()
    assert HoistParallelRegion().apply_pass(sdfg, {}) is None, why
    assert sdfg.to_json() == before, f"declining must not mutate the SDFG ({why})"


@pytest.mark.parametrize("name", HOISTED_KERNELS)
def test_one_team_replaces_the_per_trip_region(name):
    """The whole point: one ``#pragma omp parallel``, no ``#pragma omp parallel for`` left."""
    _kernel, sdfg = finalized(name, "hoist_" + name)
    team_count, per_trip, worksharing = pragmas(sdfg)
    assert team_count == 1, f"{name} must open exactly one team, got {team_count}"
    assert per_trip == 0, f"{name} must not open a region per trip, got {per_trip}"
    assert worksharing >= 1, f"{name} must still distribute its map, got {worksharing} omp-for"


@pytest.mark.parametrize("name", TSVC_KERNELS)
def test_finalized_kernel_matches_the_numpy_reference(name):
    """Neither rewrite reorders anything, so the values must be the reference's."""
    kernel, sdfg = finalized(name, "num_" + name)
    assert teams(sdfg) == (1 if name in HOISTED_KERNELS else 0), f"{name} took the wrong rewrite"
    assert_matches_reference(kernel, sdfg)


def tasklets_outside_the_map(sdfg):
    """Tasklets sitting beside a map rather than inside it, in every state that has one.

    Inside a hoisted team such a tasklet runs on EVERY thread. It is the hazard the snapshot
    wrapping existed to avoid, and it stays checkable however many worksharing loops the state
    ends up with. States with no map cannot replicate anything and are skipped -- the symbol
    assumption checks live in one.
    """
    loose = []
    for sd in sdfg.all_sdfgs_recursive():
        for state in sd.states():
            if not any(isinstance(n, nd.MapEntry) for n in state.nodes()):
                continue
            scope = state.scope_dict()
            loose += [
                f"{state.label}:{n.label}" for n in state.nodes() if isinstance(n, nd.Tasklet) and scope[n] is None
            ]
    return loose


def test_s115_keeps_every_store_inside_a_worksharing_map():
    """``s115`` no longer needs a snapshot beside its sweep, so the team distributes ONE loop.

    It used to be two. The read that looked like a write-after-read has no writer at all, so
    8d0f36f77 stopped ``break_anti_dependence`` splitting the sweep, and with no separate scalar
    store there is nothing left to wrap in a one-iteration map. That commit measured the pass's
    own suite and s115's numerics, not this count, which is how the pin came to disagree.

    Counting to two pinned the workaround rather than the property, so the count is stated as the
    shape it now has and the HAZARD is asserted directly: nothing may sit beside the map, where a
    hoisted team would run it per thread. That hazard check and the value check both hold with
    8d0f36f77 reverted as well -- only the count moved -- and the values are asserted here so a
    lost store cannot pass as a simplification.
    """
    kernel, sdfg = finalized("s115_d_single", "snapshot_s115")
    team_count, per_trip, worksharing = pragmas(sdfg)
    assert (team_count, per_trip) == (1, 0)
    assert worksharing == 1, "the sweep no longer carries a snapshot region beside it"
    assert not tasklets_outside_the_map(sdfg), "a store beside the map runs on every thread"
    assert_matches_reference(kernel, sdfg)


def test_wavefront_reaching_into_the_neighbouring_band_is_still_correct():
    """``a[i, j] = a[i, j] + a[i-1, j] + a[i-1, j+1]``: the case a ``nowait`` would silently break.

    Every band reads one element of the band to its right, written on the previous trip. The team
    hoist keeps the barrier that makes that read safe, so this must agree with numpy exactly -- it
    is the kernel that catches a barrier removed one stage too early.
    """

    @dace.program
    def wf_diff_skew(a: dace.float64[N, N]):
        for i in range(1, N):
            for j in range(0, N - 1):
                a[i, j] = a[i, j] + a[i - 1, j] + a[i - 1, j + 1]

    sdfg = wf_diff_skew.to_sdfg(simplify=False)
    canonicalize(sdfg, validate=True)
    finalize_for_target(sdfg, "cpu")
    assert teams(sdfg) == 1, "the wavefront must hoist its team"
    assert pragmas(sdfg)[1] == 0, "no per-trip region may survive"

    rng = np.random.default_rng(1234)
    a = rng.random((37, 37))
    ref = a.copy()
    for i in range(1, 37):
        for j in range(0, 36):
            ref[i, j] = ref[i, j] + ref[i - 1, j] + ref[i - 1, j + 1]
    got = a.copy()
    sdfg.compile()(a=got, N=37)
    assert np.allclose(ref, got)


def test_minimal_loop_over_map_hoists():
    """The predicate's positive control: the refusal tests below differ from this by one node."""
    sdfg = loop_over_map_sdfg("hoistable")
    assert HoistParallelRegion().apply_pass(sdfg, {}) == 1
    assert teams(sdfg) == 1
    sdfg.validate()


def test_second_run_adds_no_second_team():
    """Idempotent: the hoisted loop now sits inside a map scope, which the walk never descends into."""
    sdfg = loop_over_map_sdfg("idempotent")
    HoistParallelRegion().apply_pass(sdfg, {})
    assert HoistParallelRegion().apply_pass(sdfg, {}) is None
    assert teams(sdfg) == 1


def test_nested_sdfg_inside_the_map_keeps_its_parent_pointers(monkeypatch):
    """Outlining moves states into a NEW SDFG, and a nested SDFG that rode along must name the new one.
    Validation reads that pointer -- the wavefront kernels, whose skewed body is a nested SDFG under the
    map, are the ones that found this. ``add_node`` re-homes it, without a whole-tree rebuild."""
    sdfg = dace.SDFG("nested_in_map")
    sdfg.add_array("a", [N], dace.float64)
    body = loop_region(sdfg, "outer", "N").add_state("body", is_start_block=True)
    inner = dace.SDFG("inner_nested_sdfg_inside_the_map_keeps_its_parent_pointers")
    inner.add_array("o", [1], dace.float64)
    inner_state = inner.add_state("set", is_start_block=True)
    tasklet = inner_state.add_tasklet("set", {}, {"out"}, "out = 1.0")
    inner_state.add_edge(tasklet, "out", inner_state.add_access("o"), None, dace.Memlet("o[0]"))
    nsdfg = body.add_nested_sdfg(inner, {}, {"o"})
    entry, exit_node = body.add_map("m", {"i": "0:N"}, schedule=dtypes.ScheduleType.CPU_Multicore)
    body.add_nedge(entry, nsdfg, dace.Memlet())
    body.add_memlet_path(nsdfg, exit_node, body.add_access("a"), src_conn="o", memlet=dace.Memlet("a[i]"))
    convert_legacy_nested_sdfgs(sdfg)
    sdfg.validate()

    resets = spy_on_resets(monkeypatch)
    assert HoistParallelRegion().apply_pass(sdfg, {}) == 1
    monkeypatch.undo()
    assert resets == []
    assert_tree_matches_a_reset(sdfg)
    sdfg.validate()


def worksharing_maps(sdfg):
    """Labels of every ``CPU_Multicore`` map in ``sdfg``, nested SDFGs included."""
    return sorted(
        n.map.label
        for n, _ in sdfg.all_nodes_recursive()
        if isinstance(n, nd.MapEntry) and n.map.schedule == dtypes.ScheduleType.CPU_Multicore
    )


def test_top_level_sequential_map_in_the_loop_is_shared_out():
    """A ``Sequential`` map beside the parallel one would run P times in the team, so it becomes an ``omp for``
    too: the cost model sequentialized it to save a fork, and inside the team it costs a barrier instead."""
    sdfg = dace.SDFG("sequential_neighbour")
    sdfg.add_array("a", [N], dace.float64)
    sdfg.add_array("b", [N], dace.float64)
    outer = loop_region(sdfg, "outer", "N")
    mapped_state(outer, "par", "a", "1.0", ["0:N"])
    mapped_state(outer, "seq", "b", "2.0", ["0:N"], schedule=dtypes.ScheduleType.Sequential)
    outer.add_edge(outer.nodes()[0], outer.nodes()[1], dace.InterstateEdge())
    sdfg.validate()
    assert HoistParallelRegion().apply_pass(sdfg, {}) == 1
    assert worksharing_maps(sdfg) == ["par_map", "seq_map"]
    a, b = np.zeros(5), np.zeros(5)
    sdfg(a=a, b=b, N=5)
    assert np.allclose(a, 1.0) and np.allclose(b, 2.0)


def test_sequential_map_around_a_parallel_map_is_refused():
    """A ``Sequential`` map with a parallel map inside is a loop nest of its own, not a statement to share
    out: the team would run its inner region P times over."""
    sdfg = dace.SDFG("sequential_around_parallel")
    sdfg.add_array("a", [N], dace.float64)
    sdfg.add_array("b", [N, N], dace.float64)
    outer = loop_region(sdfg, "outer", "N")
    mapped_state(outer, "par", "a", "1.0", ["0:N"])
    nest = outer.add_state("nest")
    seq_entry, seq_exit = nest.add_map("seq", {"j": "0:N"}, schedule=dtypes.ScheduleType.Sequential)
    par_entry, par_exit = nest.add_map("inner", {"k": "0:N"}, schedule=dtypes.ScheduleType.CPU_Multicore)
    tasklet = nest.add_tasklet("set", {}, {"out"}, "out = 2.0")
    nest.add_nedge(seq_entry, par_entry, dace.Memlet())
    nest.add_nedge(par_entry, tasklet, dace.Memlet())
    nest.add_memlet_path(
        tasklet, par_exit, seq_exit, nest.add_access("b"), src_conn="out", memlet=dace.Memlet("b[j, k]")
    )
    outer.add_edge(outer.nodes()[0], nest, dace.InterstateEdge())
    sdfg.validate()
    assert_declined(sdfg, "a Sequential map with a parallel map inside breaks replication-freedom (H)")


def test_top_level_fill_in_the_loop_is_expanded_and_shared_out():
    """A fill expands to one map over the elements it writes, so it is shared out like any other map."""
    from dace.libraries.standard.nodes.fill import FillLibraryNode

    sdfg = dace.SDFG("library_neighbour")
    sdfg.add_array("a", [N], dace.float64)
    sdfg.add_array("b", [N], dace.float64)
    outer = loop_region(sdfg, "outer", "N")
    mapped_state(outer, "par", "a", "1.0", ["0:N"])
    fill_state = outer.add_state("fill")
    node = FillLibraryNode("fill_b", value=3.0)
    fill_state.add_node(node)
    fill_state.add_edge(
        node, FillLibraryNode.OUTPUT_CONNECTOR_NAME, fill_state.add_access("b"), None, dace.Memlet("b[0:N]")
    )
    outer.add_edge(outer.nodes()[0], fill_state, dace.InterstateEdge())
    sdfg.validate()
    assert HoistParallelRegion().apply_pass(sdfg, {}) == 1
    assert not any(isinstance(n, nd.LibraryNode) for n, _ in sdfg.all_nodes_recursive())
    assert len(worksharing_maps(sdfg)) == 2
    a, b = np.zeros(5), np.zeros(5)
    sdfg(a=a, b=b, N=5)
    assert np.allclose(a, 1.0) and np.allclose(b, 3.0)


def test_top_level_nested_sdfg_in_the_loop_is_refused():
    """A nested SDFG beside the maps is a statement nothing can share out. Refuse."""
    sdfg = dace.SDFG("nested_neighbour")
    sdfg.add_array("a", [N], dace.float64)
    sdfg.add_array("b", [1], dace.float64)
    outer = loop_region(sdfg, "outer", "N")
    mapped_state(outer, "par", "a", "1.0", ["0:N"])
    inner = dace.SDFG("nested_neighbour_body")
    inner.add_array("o", [1], dace.float64)
    inner_state = inner.add_state("set", is_start_block=True)
    tasklet = inner_state.add_tasklet("set", {}, {"out"}, "out = 1.0")
    inner_state.add_edge(tasklet, "out", inner_state.add_access("o"), None, dace.Memlet("o[0]"))
    nest_state = outer.add_state("nest")
    nsdfg = nest_state.add_nested_sdfg(inner, {}, {"o"})
    nest_state.add_edge(nsdfg, "o", nest_state.add_access("b"), None, dace.Memlet("b[0]"))
    outer.add_edge(outer.nodes()[0], nest_state, dace.InterstateEdge())
    sdfg.validate()
    assert_declined(sdfg, "a top-level nested SDFG breaks replication-freedom (H)")


def test_scalar_recurrence_in_an_inner_loop_is_not_worth_a_team():
    """``seidel_2d``'s shape: a worksharing map per outer trip, and an inner loop of scalar statements. In a
    team each statement becomes a one-iteration ``omp for`` -- a barrier per element of the inner loop
    against one fork saved per outer trip -- so the loop is left alone, copies and all."""
    sdfg = dace.SDFG("inner_scalar_recurrence")
    sdfg.add_array("a", [N], dace.float64)
    sdfg.add_array("b", [N], dace.float64)
    outer = loop_region(sdfg, "outer", "N")
    par = mapped_state(outer, "par", "a", "1.0", ["0:N"])
    par.add_nedge(par.add_access("a"), par.add_access("b"), dace.Memlet("a[0:N] -> [0:N]"))
    inner = loop_region(outer, "inner", "N", var="jt")
    body = inner.add_state("body", is_start_block=True)
    tasklet = body.add_tasklet("step", {"x"}, {"y"}, "y = x + 1.0")
    body.add_edge(body.add_read("b"), None, tasklet, "x", dace.Memlet("b[jt]"))
    body.add_edge(tasklet, "y", body.add_write("b"), None, dace.Memlet("b[jt]"))
    outer.add_edge(par, inner, dace.InterstateEdge())
    sdfg.validate()
    assert_declined(sdfg, "a repair inside an inner loop with no worksharing map costs a barrier per trip")


def test_loop_local_transient_crossing_two_map_scopes_stays_outside_the_nest():
    """A scope-lifetime transient would move INTO the outlined nest, one copy per thread -- condition (T).

    The first map fills ``t`` and the second reads it, so a private copy would hand the second ``omp for``
    whatever its own thread happened to write. ``t`` stays in the enclosing SDFG instead, allocated
    before the region opens and shared by the team.
    """
    sdfg = dace.SDFG("privatized_transient")
    sdfg.add_array("a", [N], dace.float64)
    sdfg.add_transient("t", [N], dace.float64)
    outer = loop_region(sdfg, "outer", "N")
    producer = mapped_state(outer, "produce", "t", "1.0", ["0:N"])
    consumer = mapped_state(outer, "consume", "a", "in_t + 1.0", ["0:N"], inputs=["t"])
    outer.add_edge(producer, consumer, dace.InterstateEdge())
    sdfg.arrays["t"].lifetime = dtypes.AllocationLifetime.Scope
    sdfg.validate()
    assert HoistParallelRegion().apply_pass(sdfg, {}) == 1
    assert "t" in sdfg.arrays, "the hand-off transient must stay in the enclosing SDFG"
    (nest,) = [n for n in sdfg.all_nodes_recursive() if isinstance(n[0], nd.NestedSDFG)]
    assert "t" not in nest[0].sdfg.arrays or not nest[0].sdfg.arrays["t"].transient
    a = np.zeros(6)
    sdfg(a=a, N=6)
    assert np.allclose(a, 2.0)


def bulk_copy_in_loop_sdfg(name, copy):
    """``for it { map i: b[i] = 1.0; a = b }`` with the copy given as a memlet string."""
    sdfg = dace.SDFG(name)
    sdfg.add_array("a", [N], dace.float64)
    sdfg.add_array("b", [N], dace.float64)
    outer = loop_region(sdfg, "outer", "N")
    body = mapped_state(outer, "par", "b", "1.0", ["0:N"])
    body.add_nedge(body.add_access("b"), body.add_access("a"), dace.Memlet(copy))
    sdfg.validate()
    return sdfg


def test_bulk_copy_between_access_nodes_in_the_loop_is_shared_out_as_a_worksharing_map():
    """``jacobi_2d``'s ``A[1:N-1, 1:N-1] = B[1:N-1, 1:N-1]``: two access nodes and a memlet.

    Replicated across the team it would be a data race with no barrier before the next trip reads it, so
    the copy becomes an ``omp for`` over the copied elements, with the barrier of any other map.
    """
    sdfg = bulk_copy_in_loop_sdfg("bulk_copy_neighbour", "b[0:N] -> [0:N]")
    assert HoistParallelRegion().apply_pass(sdfg, {}) == 1
    sdfg.validate()
    assert not any(
        isinstance(e.src, nd.AccessNode) and isinstance(e.dst, nd.AccessNode) and not e.data.is_empty()
        for e, _ in sdfg.all_edges_recursive()
    ), "the copy must leave as a worksharing map, not as a copy every thread replicates"
    a, b = np.zeros(7), np.zeros(7)
    sdfg(a=a, b=b, N=7)
    assert np.allclose(a, 1.0) and np.allclose(b, 1.0)


def test_relinearizing_bulk_copy_in_the_loop_is_refused():
    """A copy that reshapes has no element-for-element map, so nothing shares it out."""
    sdfg = dace.SDFG("bulk_copy_reshape")
    sdfg.add_array("a", [N, 2], dace.float64)
    sdfg.add_array("b", [2 * N], dace.float64)
    outer = loop_region(sdfg, "outer", "N")
    body = mapped_state(outer, "par", "b", "1.0", ["0:2*N"])
    body.add_nedge(body.add_access("b"), body.add_access("a"), dace.Memlet("b[0:2*N] -> [0:N, 0:2]"))
    sdfg.validate()
    assert_declined(sdfg, "a copy without an element-for-element map breaks (H)")


def test_loop_without_a_parallel_map_is_refused():
    """Nothing to distribute, so a team would only replicate work."""
    sdfg = dace.SDFG("all_sequential")
    sdfg.add_array("a", [N], dace.float64)
    mapped_state(loop_region(sdfg, "outer", "N"), "body", "a", "1.0", ["0:N"], schedule=dtypes.ScheduleType.Sequential)
    sdfg.validate()
    assert_declined(sdfg, "a loop with no CPU_Multicore map has nothing to hoist")


def test_break_in_the_loop_is_refused():
    """A team must encounter the same worksharing constructs on every thread; a break is not worth
    proving that about, so the pass declines the shape outright."""
    from dace.sdfg.state import BreakBlock

    sdfg = dace.SDFG("with_break")
    sdfg.add_array("a", [N], dace.float64)
    outer = loop_region(sdfg, "outer", "N")
    body = mapped_state(outer, "body", "a", "1.0", ["0:N"])
    stop = BreakBlock("stop")
    outer.add_node(stop)
    outer.add_edge(body, stop, dace.InterstateEdge(condition="it > 3"))
    sdfg.validate()
    assert_declined(sdfg, "a BreakBlock inside the loop is refused")


def banded(sdfg):
    """Run the band pass the way ``finalize_for_target`` does, with the thread-count symbol declared."""
    SupplyNumThreads().apply_pass(sdfg, {})
    return BandCarriedLoops().apply_pass(sdfg, {})


def sweep_state(container, label, target, code, ranges, inputs):
    """A state with one ``CPU_Multicore`` mapped tasklet; ``target``/``inputs`` are memlet strings."""
    state = container.add_state(label, is_start_block=len(container.nodes()) == 0)
    state.add_mapped_tasklet(
        label,
        ranges,
        {f"in{k}": dace.Memlet(m) for k, m in enumerate(inputs)},
        code,
        {"out": dace.Memlet(target)},
        schedule=dtypes.ScheduleType.CPU_Multicore,
        external_edges=True,
    )
    return state


def chain(container, *states):
    """Connect ``states`` in order inside ``container``."""
    for first, second in zip(states, states[1:]):
        container.add_edge(first, second, dace.InterstateEdge())


#: Team sizes every banded kernel is checked at. The band count follows the team, so an odd extent puts
#: band edges mid-dependence at each of them; 24 bands over the 37 columns below leave some bands one wide.
THREAD_COUNTS = (1, 4, 24)


def matches_at_every_team_size(sdfg, reference, **arguments):
    """Run the compiled ``sdfg`` once per entry of :data:`THREAD_COUNTS` on fresh copies of the array
    ``arguments`` and compare every array to ``reference`` (name -> expected array)."""
    compiled = sdfg.compile()
    gomp = ctypes.CDLL(ctypes.util.find_library("gomp"))
    for threads in THREAD_COUNTS:
        gomp.omp_set_num_threads(threads)
        fresh = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in arguments.items()}
        compiled(**fresh)
        for name, expected in reference.items():
            assert np.allclose(fresh[name], expected), f"{sdfg.name}: {name} differs at {threads} threads"


def sweep_reference(a, step):
    """``a[it + 1, cols] = step(a[it, cols])`` for every ``it``, on a copy."""
    ref = a.copy()
    for it in range(a.shape[0] - 1):
        ref[it + 1] = step(ref[it], ref[it + 1])
    return ref


def test_band_reads_two_maps_by_their_offsets():
    """``t[i]`` over ``0:N-1`` and ``t[j - 1]`` over ``1:N`` are one position of one band: the band cuts each
    map's offsets from its begin, so the test compares offsets, not the parameters as spelled."""
    sdfg = dace.SDFG("band_offsets")
    sdfg.add_array("a", [N, N], dace.float64)
    sdfg.add_transient("t", [N - 1], dace.float64)
    outer = loop_region(sdfg, "outer", "N - 1")
    first = sweep_state(outer, "first", "t[i]", "out = 2.0 * in0", {"i": "0:N-1"}, ["a[it, i + 1]"])
    second = sweep_state(outer, "second", "a[it + 1, j]", "out = in0 + 1.0", {"j": "1:N"}, ["t[j - 1]"])
    chain(outer, first, second)
    sdfg.validate()
    assert banded(sdfg) == 1
    a = np.random.default_rng(7).random((37, 37))
    ref = sweep_reference(a, lambda prev, cur: np.concatenate([cur[:1], 2.0 * prev[1:] + 1.0]))
    matches_at_every_team_size(sdfg, {"a": ref}, a=a, N=37)


def test_band_runs_a_scalar_statement_in_every_band():
    """A tasklet beside the maps writing a loop-local scalar is run by every band on its own copy."""
    sdfg = dace.SDFG("band_scalar_statement")
    sdfg.add_array("a", [N, N], dace.float64)
    sdfg.add_array("c", [1], dace.float64)
    sdfg.add_transient("s", [1], dace.float64)
    outer = loop_region(sdfg, "outer", "N - 1")
    prologue = outer.add_state("prologue", is_start_block=True)
    tasklet = prologue.add_tasklet("double", {"x"}, {"y"}, "y = 2.0 * x")
    prologue.add_edge(prologue.add_read("c"), None, tasklet, "x", dace.Memlet("c[0]"))
    prologue.add_edge(tasklet, "y", prologue.add_write("s"), None, dace.Memlet("s[0]"))
    sweep = sweep_state(outer, "sweep", "a[it + 1, i]", "out = in0 + in1", {"i": "0:N"}, ["a[it, i]", "s[0]"])
    chain(outer, prologue, sweep)
    sdfg.validate()
    assert banded(sdfg) == 1
    a, c = np.random.default_rng(8).random((37, 37)), np.array([0.25])
    ref = sweep_reference(a, lambda prev, cur: prev + 0.5)
    matches_at_every_team_size(sdfg, {"a": ref}, a=a, c=c, N=37)


def test_band_refuses_a_scalar_statement_reading_what_another_band_writes():
    """``s = a[it, 0]`` reads the column band 0 wrote on the previous trip -- ``s115``'s shape. The band
    refuses, and the team hoist, which keeps the barrier, takes the loop."""
    sdfg = dace.SDFG("band_cross_scalar")
    sdfg.add_array("a", [N, N], dace.float64)
    sdfg.add_transient("s", [1], dace.float64)
    outer = loop_region(sdfg, "outer", "N - 1")
    prologue = outer.add_state("prologue", is_start_block=True)
    tasklet = prologue.add_tasklet("pick", {"x"}, {"y"}, "y = x")
    prologue.add_edge(prologue.add_read("a"), None, tasklet, "x", dace.Memlet("a[it, 0]"))
    prologue.add_edge(tasklet, "y", prologue.add_write("s"), None, dace.Memlet("s[0]"))
    sweep = sweep_state(outer, "sweep", "a[it + 1, i]", "out = in0 + in1", {"i": "0:N"}, ["a[it, i]", "s[0]"])
    chain(outer, prologue, sweep)
    sdfg.validate()
    assert banded(sdfg) is None
    assert HoistParallelRegion().apply_pass(sdfg, {}) == 1
    a = np.random.default_rng(9).random((37, 37))
    ref = sweep_reference(a, lambda prev, cur: prev + prev[0])
    matches_at_every_team_size(sdfg, {"a": ref}, a=a, N=37)


def test_band_narrows_a_whole_fill_to_what_is_read():
    """``t[:] = 0`` beside maps over ``1:N`` has a different extent, so the band could not cut it the same
    way; nothing reads ``t[0]``, so the fill shrinks to ``t[1:N]`` and the loop bands."""
    from dace.libraries.standard.nodes.fill import FillLibraryNode

    sdfg = dace.SDFG("band_fill_hull")
    sdfg.add_array("a", [N, N], dace.float64)
    sdfg.add_transient("t", [N], dace.float64)
    outer = loop_region(sdfg, "outer", "N - 1")
    clear = outer.add_state("clear", is_start_block=True)
    fill = FillLibraryNode("clear_t", value=0.0)
    clear.add_node(fill)
    clear.add_edge(fill, FillLibraryNode.OUTPUT_CONNECTOR_NAME, clear.add_write("t"), None, dace.Memlet("t[0:N]"))
    gather = sweep_state(outer, "gather", "t[i]", "out = in0 + in1", {"i": "1:N"}, ["t[i]", "a[it, i]"])
    store = sweep_state(outer, "store", "a[it + 1, i]", "out = in0", {"i": "1:N"}, ["t[i]"])
    chain(outer, clear, gather, store)
    sdfg.validate()
    assert banded(sdfg) == 1
    a = np.random.default_rng(10).random((37, 37))
    ref = sweep_reference(a, lambda prev, cur: np.concatenate([cur[:1], prev[1:]]))
    matches_at_every_team_size(sdfg, {"a": ref}, a=a, N=37)


def test_band_takes_a_row_fill_loop_as_one_map():
    """``SpecializeCpuTransfers``' shape: a sequential row loop around a row fill. It is rebuilt as one map
    over rows and columns, so its column axis is cut like every other map's."""
    from dace.libraries.standard.nodes.fill import FillLibraryNode

    sdfg = dace.SDFG("band_row_fill")
    sdfg.add_array("a", [N, N], dace.float64)
    sdfg.add_transient("z", [3, N], dace.float64)
    outer = loop_region(sdfg, "outer", "N - 1")
    clear = outer.add_state("clear", is_start_block=True)
    entry, exit_node = clear.add_map("rows", {"r": "0:3"}, schedule=dtypes.ScheduleType.Sequential)
    fill = FillLibraryNode("clear_row", value=1.5)
    clear.add_node(fill)
    clear.add_nedge(entry, fill, dace.Memlet())
    clear.add_memlet_path(
        fill,
        exit_node,
        clear.add_write("z"),
        src_conn=FillLibraryNode.OUTPUT_CONNECTOR_NAME,
        memlet=dace.Memlet("z[r, 0:N]"),
    )
    sweep = sweep_state(outer, "sweep", "a[it + 1, i]", "out = in0 + in1", {"i": "0:N"}, ["a[it, i]", "z[1, i]"])
    chain(outer, clear, sweep)
    sdfg.validate()
    assert banded(sdfg) == 1
    assert not any(isinstance(n, nd.LibraryNode) for n, _ in sdfg.all_nodes_recursive())
    a = np.random.default_rng(11).random((37, 37))
    ref = sweep_reference(a, lambda prev, cur: prev + 1.5)
    matches_at_every_team_size(sdfg, {"a": ref}, a=a, N=37)


def test_band_refuses_a_neighbour_read_between_two_maps():
    """``t[i]`` written by one map and ``t[i + 1]`` read by the next: the read crosses a band edge within
    one trip, which only the barrier between the two ``omp for`` orders. The team hoist takes it."""
    sdfg = dace.SDFG("band_neighbour_read")
    sdfg.add_array("a", [N, N], dace.float64)
    sdfg.add_transient("t", [N], dace.float64)
    outer = loop_region(sdfg, "outer", "N - 1")
    first = sweep_state(outer, "first", "t[i]", "out = in0", {"i": "0:N"}, ["a[it, i]"])
    second = sweep_state(outer, "second", "a[it + 1, i]", "out = in0", {"i": "0:N-1"}, ["t[i + 1]"])
    chain(outer, first, second)
    sdfg.validate()
    assert banded(sdfg) is None
    assert HoistParallelRegion().apply_pass(sdfg, {}) == 1
    a = np.random.default_rng(12).random((37, 37))
    ref = sweep_reference(a, lambda prev, cur: np.concatenate([prev[1:], cur[-1:]]))
    matches_at_every_team_size(sdfg, {"a": ref}, a=a, N=37)


def test_band_refuses_maps_of_different_extents():
    """A map over ``0:N`` and one over ``0:N-1`` are cut at different offsets, so the same position lands in
    different bands even where both index it alike."""
    sdfg = dace.SDFG("band_extents")
    sdfg.add_array("a", [N, N], dace.float64)
    sdfg.add_transient("t", [N], dace.float64)
    outer = loop_region(sdfg, "outer", "N - 1")
    first = sweep_state(outer, "first", "t[i]", "out = in0 + 1.0", {"i": "0:N"}, ["a[it, i]"])
    second = sweep_state(outer, "second", "a[it + 1, i]", "out = in0", {"i": "0:N-1"}, ["t[i]"])
    chain(outer, first, second)
    sdfg.validate()
    assert banded(sdfg) is None


def test_band_refuses_an_accumulation_into_one_location():
    """``s += a[it, i]`` over the map writes one location from every band. Refused."""
    sdfg = dace.SDFG("band_accumulation")
    sdfg.add_array("a", [N, N], dace.float64)
    sdfg.add_array("s", [1], dace.float64)
    outer = loop_region(sdfg, "outer", "N")
    body = outer.add_state("body", is_start_block=True)
    body.add_mapped_tasklet(
        "acc",
        {"i": "0:N"},
        {"v": dace.Memlet("a[it, i]")},
        "w = v",
        {"w": dace.Memlet("s[0]", wcr="lambda x, y: x + y")},
        schedule=dtypes.ScheduleType.CPU_Multicore,
        external_edges=True,
    )
    sdfg.validate()
    assert banded(sdfg) is None


def test_band_takes_a_map_inside_a_conditional():
    """A trip that runs its map only on even ``it`` still keeps every column in its band."""
    sdfg = dace.SDFG("band_conditional")
    sdfg.add_array("a", [N, N], dace.float64)
    outer = loop_region(sdfg, "outer", "N - 1")
    branch = ConditionalBlock("even")
    outer.add_node(branch, is_start_block=True)
    region = ControlFlowRegion("even_body")
    branch.add_branch(CodeBlock("it % 2 == 0"), region)
    sweep_state(region, "sweep", "a[it + 1, i]", "out = in0 * 2.0", {"i": "0:N"}, ["a[it, i]"])
    sdfg.validate()
    assert banded(sdfg) == 1
    a = np.random.default_rng(13).random((37, 37))
    ref = a.copy()
    for it in range(36):
        if it % 2 == 0:
            ref[it + 1] = 2.0 * ref[it]
    matches_at_every_team_size(sdfg, {"a": ref}, a=a, N=37)


@pytest.mark.parametrize("name", TSVC_KERNELS)
def test_the_canonical_form_keeps_every_barrier(name):
    """No ``omp for`` may carry ``nowait``, and this is policy rather than an unfinished feature.

    Dropping the exit barrier is legal only when every thread's reads at trip ``k+1`` fall inside
    the band it wrote at trip ``k`` -- formally, for a partition ``B_1..B_P`` of the map's index
    space, ``R_p(k+1) & (U_q W_q(k)) subset W_p(k)``. ``s233``, ``s231`` and ``s235`` satisfy it;
    ``s119`` and ``wf_diff_skew`` miss it by ONE element at the band boundary, and ``s115`` reads a
    scalar that only one band writes. Measured out of tree by patching the pragma into the emitted
    source: ``nowait`` leaves ``s233`` bit-identical and 2.4x faster, and turns ``s119`` into a
    wrong answer (max relative error 4.8e7). A verdict that sharp is a specialization decision --
    it belongs to a stage that can afford the dependence analysis, not to the canonical starting
    point, which has to be right for every shape it is handed.
    """
    _kernel, sdfg = finalized(name, "barrier_" + name)
    assert not re.search(r"#pragma omp for[^\n]*nowait", sdfg.generate_code()[0].clean_code), (
        "canonicalization must not remove a barrier: that verdict belongs to a later stage"
    )


def test_a_loop_nested_inside_another_region_is_hoisted_in_its_own_graph():
    """The team is outlined from the graph that holds the loop, not from the SDFG (warpx_boris_push's shape)."""
    sdfg = dace.SDFG("nested_hoistable")
    sdfg.add_array("a", [N], dace.float64)
    sdfg.add_array("b", [N], dace.float64)
    outer = loop_region(sdfg, "outer", "N")

    # A top-level nested SDFG in the outer body: the outer loop is refused here.
    prologue = outer.add_state("prologue", is_start_block=True)
    stamp = dace.SDFG("nested_hoistable_prologue")
    stamp.add_array("o", [N], dace.float64)
    stamp_state = stamp.add_state("set", is_start_block=True)
    tasklet = stamp_state.add_tasklet("set", {}, {"out"}, "out = 2.0")
    stamp_state.add_edge(tasklet, "out", stamp_state.add_access("o"), None, dace.Memlet("o[0]"))
    stamp_node = prologue.add_nested_sdfg(stamp, {}, {"o"})
    prologue.add_edge(stamp_node, "o", prologue.add_access("b"), None, dace.Memlet("b[0:N]"))

    inner = LoopRegion(
        "inner", initialize_expr="jt = 0", condition_expr="jt < N", update_expr="jt = jt + 1", loop_var="jt"
    )
    outer.add_node(inner)
    outer.add_edge(prologue, inner, dace.InterstateEdge())
    mapped_state(inner, "inner_body", "a", "1.0", ["0:N"])
    sdfg.validate()

    assert HoistParallelRegion().apply_pass(sdfg, {}) == 1, "the inner worksharing loop must be hoisted"
    sdfg.validate()
    assert teams(sdfg) == 1, "exactly one persistent team, around the inner loop"
    assert not any(isinstance(b, LoopRegion) and b.label == "inner" for b in sdfg.nodes()), (
        "the inner loop must stay inside the outer region, not be lifted to the SDFG"
    )


if __name__ == "__main__":
    for kernel_name in HOISTED_KERNELS:
        test_one_team_replaces_the_per_trip_region(kernel_name)
    for kernel_name in TSVC_KERNELS:
        test_finalized_kernel_matches_the_numpy_reference(kernel_name)
    test_s115_keeps_every_store_inside_a_worksharing_map()
    test_wavefront_reaching_into_the_neighbouring_band_is_still_correct()
    test_minimal_loop_over_map_hoists()
    test_second_run_adds_no_second_team()
    test_nested_sdfg_inside_the_map_keeps_its_parent_pointers(pytest.MonkeyPatch())
    test_top_level_sequential_map_in_the_loop_is_shared_out()
    test_sequential_map_around_a_parallel_map_is_refused()
    test_top_level_fill_in_the_loop_is_expanded_and_shared_out()
    test_top_level_nested_sdfg_in_the_loop_is_refused()
    test_scalar_recurrence_in_an_inner_loop_is_not_worth_a_team()
    test_loop_local_transient_crossing_two_map_scopes_stays_outside_the_nest()
    test_bulk_copy_between_access_nodes_in_the_loop_is_shared_out_as_a_worksharing_map()
    test_relinearizing_bulk_copy_in_the_loop_is_refused()
    test_loop_without_a_parallel_map_is_refused()
    test_break_in_the_loop_is_refused()
    test_band_reads_two_maps_by_their_offsets()
    test_band_runs_a_scalar_statement_in_every_band()
    test_band_refuses_a_scalar_statement_reading_what_another_band_writes()
    test_band_narrows_a_whole_fill_to_what_is_read()
    test_band_takes_a_row_fill_loop_as_one_map()
    test_band_refuses_a_neighbour_read_between_two_maps()
    test_band_refuses_maps_of_different_extents()
    test_band_refuses_an_accumulation_into_one_location()
    test_band_takes_a_map_inside_a_conditional()
    for kernel_name in TSVC_KERNELS:
        test_the_canonical_form_keeps_every_barrier(kernel_name)
    test_a_loop_nested_inside_another_region_is_hoisted_in_its_own_graph()
