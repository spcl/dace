# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The ``OpenMP`` implementation of the ``Sync`` library node: ``#pragma omp barrier`` for the team of a
``CPU_Persistent`` region, nothing where one thread runs, and an error inside a worksharing map."""

import numpy as np
import pytest

import dace
from dace.libraries.standard.nodes.sync import Sync, SyncScope

NTHREADS = 4


def phases_sdfg(name: str, schedule: dace.ScheduleType, scope: SyncScope, implementation: str = "OpenMP") -> dace.SDFG:
    """Thread ``t`` writes ``A[t]``, waits at the barrier, then reads its neighbor's ``A[(t + 1) % 4]``."""
    inner = dace.SDFG(f"{name}_phases")
    inner.add_symbol("t", dace.int64)
    inner.add_array("A", (NTHREADS,), dace.float64)
    inner.add_array("B", (NTHREADS,), dace.float64)
    write = inner.add_state("write", is_start_block=True)
    task = write.add_tasklet("write", {}, {"a"}, "a = t + 1")
    write.add_edge(task, "a", write.add_write("A"), None, dace.Memlet("A[t]"))
    wait = inner.add_state_after(write, "wait")
    barrier = Sync("barrier", scope=scope)
    barrier.implementation = implementation
    wait.add_node(barrier)
    read = inner.add_state_after(wait, "read")
    task = read.add_tasklet("read", {"a"}, {"b"}, "b = a")
    read.add_edge(read.add_read("A"), None, task, "a", dace.Memlet(f"A[(t + 1) % {NTHREADS}]"))
    read.add_edge(task, "b", read.add_write("B"), None, dace.Memlet("B[t]"))

    sdfg = dace.SDFG(name)
    sdfg.add_array("A", (NTHREADS,), dace.float64)
    sdfg.add_array("B", (NTHREADS,), dace.float64)
    state = sdfg.add_state()
    entry, exit_ = state.add_map("team", {"t": f"0:{NTHREADS}"}, schedule=schedule)
    entry.map.omp_num_threads = NTHREADS
    nsdfg = state.add_nested_sdfg(inner, {"A": None}, {"A": None, "B": None}, symbol_mapping={"t": "t"})
    state.add_memlet_path(state.add_read("A"), entry, nsdfg, dst_conn="A", memlet=dace.Memlet(f"A[0:{NTHREADS}]"))
    for out in ("A", "B"):
        state.add_memlet_path(
            nsdfg, exit_, state.add_write(out), src_conn=out, memlet=dace.Memlet(f"{out}[0:{NTHREADS}]")
        )
    return sdfg


def test_team_barrier_orders_the_phases():
    sdfg = phases_sdfg("omp_team_barrier", dace.ScheduleType.CPU_Persistent, SyncScope.BLOCK)
    code = sdfg.generate_code()[0].clean_code
    assert "#pragma omp parallel num_threads(4)" in code
    assert code.count("#pragma omp barrier") == 1

    a, b = np.zeros(NTHREADS), np.zeros(NTHREADS)
    sdfg(A=a, B=b)
    assert np.array_equal(a, np.arange(1, NTHREADS + 1))
    assert np.array_equal(b, np.roll(np.arange(1, NTHREADS + 1), -1))


def test_warp_barrier_on_a_cpu_waits_for_nobody():
    sdfg = phases_sdfg("omp_warp_barrier", dace.ScheduleType.CPU_Persistent, SyncScope.WARP)
    assert "#pragma omp barrier" not in sdfg.generate_code()[0].clean_code


def test_one_thread_needs_no_barrier():
    sdfg = phases_sdfg("omp_sequential_barrier", dace.ScheduleType.Sequential, SyncScope.BLOCK)
    assert "#pragma omp barrier" not in sdfg.generate_code()[0].clean_code
    sdfg.compile()


def test_worksharing_map_cannot_hold_a_barrier():
    sdfg = phases_sdfg("omp_worksharing_barrier", dace.ScheduleType.CPU_Multicore, SyncScope.BLOCK)
    with pytest.raises(ValueError, match="worksharing"):
        sdfg.expand_library_nodes()


def test_grid_barrier_in_a_team_is_a_team_barrier():
    sdfg = phases_sdfg("omp_grid_barrier", dace.ScheduleType.CPU_Persistent, SyncScope.GRID)
    assert sdfg.generate_code()[0].clean_code.count("#pragma omp barrier") == 1


def test_an_automatic_barrier_in_a_team_is_a_team_barrier():
    """``Auto`` picks the OpenMP barrier inside a CPU team rather than the GPU grid barrier."""
    sdfg = phases_sdfg("omp_auto_barrier", dace.ScheduleType.CPU_Persistent, SyncScope.GRID, implementation="Auto")
    code = sdfg.generate_code()[0].clean_code
    assert code.count("#pragma omp barrier") == 1 and "cooperative_groups" not in code


if __name__ == "__main__":
    test_team_barrier_orders_the_phases()
    test_warp_barrier_on_a_cpu_waits_for_nobody()
    test_one_thread_needs_no_barrier()
    test_worksharing_map_cannot_hold_a_barrier()
    test_grid_barrier_in_a_team_is_a_team_barrier()
    test_an_automatic_barrier_in_a_team_is_a_team_barrier()
