# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The CPU library-node parallel-lowering rule in
:func:`~dace.transformation.auto.auto_optimize.apply_cpu_library_parallelism`, both halves of it.

A library node opens its own OpenMP region only when nothing re-enters it
(:func:`~dace.transformation.auto.auto_optimize.libnode_is_sequential`) AND its own schedule would
run as an OpenMP team (:func:`~dace.transformation.auto.auto_optimize.libnode_runs_multicore`) AND
it carries enough work to pay for the region
(:func:`~dace.transformation.auto.auto_optimize.libnode_work_is_below_break_even`). The scope half
is covered by ``canonicalize_finalize_nested_sdfg_schedule_test``; the two halves below are the ones
that decide a TOP-LEVEL node, where the scope half alone says "parallel".

The work threshold is host-derived (``CalibrateCpuThresholds`` scales it by physical core count), so
every case here pins it -- otherwise the same source would assert different things on an 8-core
development box and on a 72-core node.
"""
import dace
import pytest
from dace import dtypes
from dace.libraries.standard.nodes.reduce import Reduce
from dace.libraries.standard.nodes.scan import Scan, ScanOp, INPUT_CONNECTOR_NAME, OUTPUT_CONNECTOR_NAME
from dace.transformation.auto.auto_optimize import apply_cpu_library_parallelism

THRESHOLD = 256


def build_toplevel_reduce(n, schedule):
    """A ``Reduce`` in a top-level state -- no enclosing map, no enclosing loop, so the SCOPE half of
    the rule says "parallel" and whatever the assertions observe comes from the other halves."""
    sdfg = dace.SDFG('toplevel_reduce')
    sdfg.add_array('A', [n], dace.float64)
    sdfg.add_array('out', [1], dace.float64)
    state = sdfg.add_state()
    node = Reduce('reduce_sum', wcr='lambda a, b: a + b', axes=None, identity=0.0)
    node.schedule = schedule
    state.add_node(node)
    state.add_edge(state.add_access('A'), None, node, '_in', dace.Memlet(f'A[0:{n}]'))
    state.add_edge(node, '_out', state.add_access('out'), None, dace.Memlet('out[0]'))
    sdfg.validate()
    return sdfg, state, node


def build_toplevel_scan(extent):
    """A top-level inclusive-sum ``Scan`` over ``extent`` (an int or a symbolic expression)."""
    sdfg = dace.SDFG('toplevel_scan')
    sdfg.add_array('A', [extent], dace.float64)
    sdfg.add_array('B', [extent], dace.float64)
    state = sdfg.add_state()
    node = Scan('scan_sum', op=ScanOp.SUM, exclusive=False, identity=0.0)
    node.schedule = dtypes.ScheduleType.CPU_Multicore
    state.add_node(node)
    state.add_edge(state.add_access('A'), None, node, INPUT_CONNECTOR_NAME, dace.Memlet(f'A[0:{extent}]'))
    state.add_edge(node, OUTPUT_CONNECTOR_NAME, state.add_access('B'), None, dace.Memlet(f'B[0:{extent}]'))
    sdfg.validate()
    return sdfg, state, node


@pytest.mark.parametrize('schedule', [dtypes.ScheduleType.CPU_Multicore, dtypes.ScheduleType.Default])
def test_toplevel_openmp_team_schedule_opens_a_region(schedule):
    """``CPU_Multicore`` is the plain case. ``Default`` must decide the same way: the enum documents
    it as the scope-default PARALLEL schedule and ``SCOPEDEFAULT_SCHEDULE[None]`` resolves a
    top-level scope to ``CPU_Multicore``, so a node introduced before type inference has run would
    otherwise be silently single-threaded for the rest of the pipeline."""
    sdfg, state, node = build_toplevel_reduce(4096, schedule)
    with dace.config.set_temporary('compiler', 'cpu', 'parallel_min_work_per_region', value=THRESHOLD):
        assert apply_cpu_library_parallelism(node, state, sdfg) is True
    assert node.implementation == 'OpenMP'


def test_toplevel_non_team_schedule_takes_the_single_core_expansion():
    """The half this rule adds. A top-level node passes the scope test, but a schedule that is not an
    OpenMP team (here ``SVE_Map``, an Arm vector map) names an execution context that cannot host a
    ``#pragma omp parallel`` -- selecting ``OpenMP`` for it emits a team inside a vector map."""
    sdfg, state, node = build_toplevel_reduce(4096, dtypes.ScheduleType.SVE_Map)
    with dace.config.set_temporary('compiler', 'cpu', 'parallel_min_work_per_region', value=THRESHOLD):
        assert apply_cpu_library_parallelism(node, state, sdfg) is True
    assert node.implementation == 'pure'


def test_provably_small_toplevel_scan_stays_sequential():
    """The size half, replacing the ``PARALLEL_MIN_ELEMENTS_CONTIGUOUS`` gate that ``scan.hpp`` used
    to re-test on every call. Eight elements cannot pay for a fork/join."""
    sdfg, state, node = build_toplevel_scan(8)
    with dace.config.set_temporary('compiler', 'cpu', 'parallel_min_work_per_region', value=THRESHOLD):
        assert apply_cpu_library_parallelism(node, state, sdfg) is True
    assert node.implementation == 'pure'


def test_symbolic_extent_scan_is_assumed_big_and_stays_parallel():
    """A symbolic extent is assumed BIG. Reading "unknown" as "small" would single-thread every
    dynamically sized scan and reduction in the program -- the opposite of the canonical parallel
    form, and unrecoverable later because the sequential choice is what reaches codegen."""
    n = dace.symbol('N', dtype=dace.int64)
    sdfg, state, node = build_toplevel_scan(n)
    with dace.config.set_temporary('compiler', 'cpu', 'parallel_min_work_per_region', value=THRESHOLD):
        assert apply_cpu_library_parallelism(node, state, sdfg) is True
    assert node.implementation == 'CPU'


def test_large_toplevel_scan_is_parallel():
    """The counterpart of the small case on the same node type, so the ``pure`` verdict above is
    attributable to the element count and not to something else about a top-level ``Scan``."""
    sdfg, state, node = build_toplevel_scan(4096)
    with dace.config.set_temporary('compiler', 'cpu', 'parallel_min_work_per_region', value=THRESHOLD):
        assert apply_cpu_library_parallelism(node, state, sdfg) is True
    assert node.implementation == 'CPU'


if __name__ == '__main__':
    test_toplevel_openmp_team_schedule_opens_a_region(dtypes.ScheduleType.CPU_Multicore)
    test_toplevel_openmp_team_schedule_opens_a_region(dtypes.ScheduleType.Default)
    test_toplevel_non_team_schedule_takes_the_single_core_expansion()
    test_provably_small_toplevel_scan_stays_sequential()
    test_symbolic_extent_scan_is_assumed_big_and_stays_parallel()
    test_large_toplevel_scan_is_parallel()
