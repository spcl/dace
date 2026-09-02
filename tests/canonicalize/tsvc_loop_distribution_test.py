# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Loop DISTRIBUTION on the TSVC kernels it frees, asserted structurally and numerically.

Every kernel here was one sequential loop until the split learned something: to order more than
two groups instead of flipping one bit, to follow a producer chain across states, to keep a
carried scalar's reader and writer in one group, to read an in-place ``a[i] = a[i] + x[i]`` as
data-parallel rather than as a recurrence, and to drop the dead staging the per-group pruning
leaves behind. The assertions are on what ``canonicalize`` PRODUCES -- how many sequential loops
survive and which arrays each still writes -- because a numeric comparison alone passes just as
happily on the un-split loop, and "it parallelized" is the property at stake.
"""
import os

os.environ.setdefault("OMPI_MCA_pml", "ob1")
os.environ.setdefault("OMPI_MCA_btl", "self,vader")
os.environ.setdefault("UCX_VFS_ENABLE", "n")

import numpy as np
import pytest

from dace.sdfg import nodes as nd
from dace.sdfg import utils as sdutil
from dace.sdfg.state import LoopRegion
from dace.transformation.passes.canonicalize.pipeline import _build_stages, canonicalize
from dace.transformation.passes.canonicalize.split_statements import SplitStatements

from tests.corpus.tsvc import tsvc
from tests.corpus.tsvc.tsvc_numpy import REFERENCES


def canonicalized(name):
    """``(kernel, sdfg)`` for one TSVC kernel put through the production canonicalize recipe."""
    kernel = tsvc.collect(name=name)[0]
    sdfg = tsvc.to_sdfg(kernel, 'dist_' + name, simplify=True)
    canonicalize(sdfg, validate=True, peel_limit=4)
    return kernel, sdfg


def distributed(name):
    """``(kernel, sdfg)`` stopped right after ``SplitStatements``, where the split's own output shows.

    The full recipe turns the split's loops into maps and re-fuses what belongs together, so the
    number and ORDER of the loops the split emitted is only readable here.
    """
    kernel = tsvc.collect(name=name)[0]
    sdfg = tsvc.to_sdfg(kernel, 'split_' + name, simplify=True)
    for _label, unit in _build_stages():
        unit.apply_pass(sdfg, {})
        if isinstance(unit, SplitStatements):
            break
    sdfg.reset_cfg_list()
    return kernel, sdfg


def split_loops_in_order(sdfg):
    """The arrays each residual loop writes, in the order the SDFG runs the loops."""
    loops = residual_loops(sdfg)
    regions = {id(loop.parent_graph): loop.parent_graph for loop in loops}
    assert len(regions) == 1, 'the split must leave its loops in one region'
    graph = next(iter(regions.values()))
    return [
        written_arrays(blk) for blk in sdutil.dfs_topological_sort(graph, [graph.start_block])
        if isinstance(blk, LoopRegion) and blk.loop_variable
    ]


def reaches(state, src, dst):
    """Whether the dataflow FORCES ``src`` to run before ``dst`` -- a path between them.

    Two scopes in one state with no path between them are UNORDERED, so a topological listing
    would report an order the graph does not actually promise. Only a path does.
    """
    seen, stack = {id(src): None}, [src]
    while stack:
        node = stack.pop()
        if node is dst:
            return True
        for edge in state.out_edges(node):
            if id(edge.dst) not in seen:
                seen[id(edge.dst)] = None
                stack.append(edge.dst)
    return False


def map_writing(sdfg, name):
    """``(state, entry)`` of the one top-level map storing to ``name``."""
    found = [(state, entry) for state in sdfg.all_states() for entry in state.nodes() if isinstance(entry, nd.MapEntry)
             and state.entry_node(entry) is None and name in written_arrays_of_scope(state, entry)]
    assert len(found) == 1, f'expected exactly one top-level map writing {name}, got {len(found)}'
    return found[0]


def written_arrays_of_scope(state, entry):
    """The non-transient arrays the map under ``entry`` stores to, sorted."""
    return sorted({
        e.data.data
        for e in state.in_edges(state.exit_node(entry))
        if e.data is not None and e.data.data is not None and not state.sdfg.arrays[e.data.data].transient
    })


def residual_loops(sdfg):
    """The sequential ``LoopRegion`` s canonicalize did not turn into parallel work."""
    return [r for r in sdfg.all_control_flow_regions() if isinstance(r, LoopRegion) and r.loop_variable]


def num_maps(sdfg):
    return sum(1 for node, _ in sdfg.all_nodes_recursive() if isinstance(node, nd.MapEntry))


def written_arrays(loop):
    """The non-transient arrays ``loop``'s body stores to, sorted.

    An EMPTY in-memlet is an ordering edge and moves no value, so it must not read as a store --
    ``s243`` holds such an edge on ``d`` and would otherwise report ``d`` among its outputs.
    """
    out = set()
    for state in loop.all_states():
        for node in state.data_nodes():
            stores = [e for e in state.in_edges(node) if e.data is not None and not e.data.is_empty()]
            if stores and not state.sdfg.arrays[node.data].transient:
                out.add(node.data)
    return sorted(out)


def assert_matches_reference(kernel, sdfg):
    """The canonicalized kernel must reproduce the numpy reference element for element.

    Structure is read off ``sdfg`` BEFORE this runs: compiling is one-way, and the SDFG a
    ``CompiledSDFG`` still points at is not the object to inspect afterwards.
    """
    arrays, call_kwargs = tsvc.make_inputs(kernel)
    ref = {n: a.copy() for n, a in arrays.items()}
    REFERENCES[kernel.name](**ref, **call_kwargs)
    got = {n: a.copy() for n, a in arrays.items()}
    csdfg = sdfg.compile()
    csdfg(**got, **call_kwargs)
    for n, arr in arrays.items():
        if np.issubdtype(arr.dtype, np.integer):
            continue
        assert np.allclose(ref[n], got[n], equal_nan=True), f'{kernel.name}: value mismatch on {n}'


def test_s211_reader_and_writer_of_b_become_two_maps():
    """``a[i] = b[i-1] + c[i]*d[i]; b[i] = b[i+1] - e[i]*d[i]``.

    ``b`` is read one behind by one statement and one ahead by the other, so no single loop is
    parallel -- but the ``b`` writer read AHEAD of its own store, which puts it first, and ``a``
    then reads the finished ``b``. Two loops, both data-parallel.
    """
    kernel, sdfg = canonicalized('s211_d_single')
    assert residual_loops(sdfg) == [], 's211 must distribute into two parallel maps'
    assert num_maps(sdfg) >= 2
    assert_matches_reference(kernel, sdfg)


def test_s261_dead_staging_does_not_keep_the_clone_sequential():
    """``t = a[i]+b[i]; a[i] = t + c[i-1]; c[i] = c[i]*d[i]``.

    The split fires either way; what used to cost the ``c`` clone its map is the dead read of
    ``c[i-1]`` the pruning left behind, staged into a transient nothing reads and held alive by
    ordering edges. With the staging gone the clone reads and writes only ``c[i]``.
    """
    kernel, sdfg = canonicalized('s261_d_single')
    assert residual_loops(sdfg) == [], 's261 must distribute into two parallel maps'
    assert num_maps(sdfg) >= 2
    assert_matches_reference(kernel, sdfg)


def test_s261_runs_the_writer_of_c_before_its_reader():
    """The ORDER the s261 split has to pick, asserted on both forms it passes through.

    ``a[i] = (a[i] + b[i]) + c[i - 1]`` reads the ``c`` that ``c[i] = c[i] * d[i]`` stored one
    iteration earlier -- a true carried dependence at distance 1 -- so only one order of the two
    loops reproduces the fused loop: all of ``c``, then all of ``a``. Run them the other way round
    and the reader sees the ORIGINAL ``c``, which still validates and still parallelizes. Counting
    loops or maps cannot tell the two apart, so the order is what has to be pinned.
    """
    _kernel, split = distributed('s261_d_single')
    assert split_loops_in_order(split) == [['c'], ['a']], 'the c loop must be emitted before the a loop'

    kernel, sdfg = canonicalized('s261_d_single')
    assert num_maps(sdfg) == 2, 's261 must end up as exactly two maps'
    writer_state, writer = map_writing(sdfg, 'c')
    reader_state, reader = map_writing(sdfg, 'a')
    assert writer_state is reader_state, 'both maps land in one state, joined by the c they share'
    assert reaches(writer_state, writer_state.exit_node(writer), reader), \
        'the a map must read the c the other map wrote, not the original'
    assert_matches_reference(kernel, sdfg)


def test_s241_raw_dependence_cycle_is_refused():
    """``a[i] = b[i]*c[i]*d[i]; b[i] = a[i]*a[i+1]*d[i]`` -- the split must NOT fire on this shape.

    S1 feeds S2 through ``a[i]`` in the same iteration, and S2 reads ``a[i + 1]`` before a later
    iteration of S1 overwrites it. The first constraint puts S1 first, the second puts S2 first,
    and no order of the two loops satisfies both -- which is why TSVC files this kernel under node
    splitting rather than distribution. Splitting it either way silently changes the values.

    Pinned on the RAW shape, which is what this refusal protects. Later in the recipe
    ``BreakAntiDependence(forward_reads=True)`` snapshots ``a``, so the ``a[i + 1]`` read no longer
    names the array S1 writes; with the cycle broken the distribution IS legal and the recipe does
    take it. Turning that pass off leaves this kernel one sequential loop, which is the property
    the refusal is responsible for.
    """
    _kernel, split = distributed('s241_d_single')
    assert split_loops_in_order(split) == [['a', 'b']], 's241 is a dependence cycle and must stay one loop'


def test_s243_dependence_cycle_is_refused():
    """``a[i] = ..; b[i] = a[i] + ..; a[i] = b[i] + a[i+1]*d[i]`` -- the same cycle, three statements.

    ``a`` and ``b`` feed each other within the iteration and ``a[i + 1]`` reaches back across it,
    so the groups are one strongly connected component however they are cut. Unlike ``s241`` this
    one is never distributed at all: the split declines it at every point in the recipe.
    """
    _kernel, split = distributed('s243_d_single')
    assert split_loops_in_order(split) == [['a', 'b']], 's243 is a dependence cycle and must stay one loop'


def test_s3251_three_groups_need_a_sorted_order():
    """``a[i+1] = b[i]+c[i]; b[i] = c[i]*e[i]; d[i] = a[i]*e[i]`` -- three groups, two constraints.

    ``d`` reads ``a`` before ``a``'s own store reaches that element, and ``a``'s statement reads the
    ``b`` the third group overwrites. One order satisfies both, and it is not expressible as "which
    of two goes first". The body is also three STATES, so the producer cones have to cross them to
    see the groups apart at all.
    """
    kernel, sdfg = canonicalized('s3251_d_single')
    assert residual_loops(sdfg) == [], 's3251 must distribute into parallel maps'
    assert num_maps(sdfg) >= 2
    assert_matches_reference(kernel, sdfg)


def test_s2251_carried_scalar_stays_with_its_reader_and_then_rotates():
    """``a[i] = s*e[i]; s = b[i]+c[i]; b[i] = a[i]+d[i]`` -- ``s`` holds ``b[i-1]+c[i-1]``.

    No order of ``s``'s reader against its writer reproduces the fused loop, so they stay in one
    group and only the ``b`` statement is peeled off. That is what makes the rotation legal: in the
    peeled group ``b`` is no longer written, so ``s``'s producer can be re-evaluated one iteration
    back and the carry disappears.
    """
    kernel, sdfg = canonicalized('s2251_d_single')
    assert residual_loops(sdfg) == [], 's2251 must lose its carried scalar and fully parallelize'
    assert num_maps(sdfg) >= 2
    assert_matches_reference(kernel, sdfg)


def test_s222_elementwise_statement_is_peeled_off_the_recurrence():
    """``a[i] += b[i]*c[i]; e[i] = e[i-1]*e[i-1]; a[i] -= b[i]*c[i]``.

    Both arrays are read and written by the loop. Only ``e`` reads an element other than the one it
    writes, so only ``e`` keeps a sequential loop; the ``a`` statements are per-element and become a
    map. Reading ``a`` as carried too leaves no free group and refuses the split outright.
    """
    kernel, sdfg = canonicalized('s222_d_single')
    loops = residual_loops(sdfg)
    assert len(loops) == 1, f's222 must keep exactly the e recurrence sequential, got {len(loops)}'
    assert written_arrays(loops[0]) == ['e'], 'the surviving loop must be the e recurrence alone'
    assert num_maps(sdfg) >= 1, 'the a statements must come out as a map'
    assert_matches_reference(kernel, sdfg)


if __name__ == '__main__':
    pytest.main([__file__])
