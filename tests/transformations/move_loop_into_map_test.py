# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.

import dace
from dace.properties import CodeBlock
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion, LoopRegion
from dace.libraries.standard.nodes.fill.node import FillLibraryNode
from dace.transformation.interstate import MoveLoopIntoMap
from dace.transformation.interstate.move_loop_into_map import analyze_lanes
from dace.transformation.passes.canonicalize.move_loop_into_map_gated import MoveLoopIntoMapGated
import copy
import json
import numpy as np

I = dace.symbol("I")
J = dace.symbol("J")
K = dace.symbol("K")
N = dace.symbol("N")


# forward loop with loop carried dependency
@dace.program
def forward_loop(data: dace.float64[I, J]):
    for i in range(4, I):
        for j in dace.map[0:J]:
            data[i, j] = data[i - 1, j]


# backward loop with loop carried dependency
@dace.program
def backward_loop(data: dace.float64[I, J]):
    for i in range(I - 2, 3, -1):
        for j in dace.map[0:J]:
            data[i, j] = data[i + 1, j]


@dace.program
def multiple_edges(data: dace.float64[I, J]):
    for i in range(4, I):
        for j in dace.map[1:J]:
            data[i, j] = data[i - 1, j] + data[i - 2, j]


@dace.program
def should_not_apply_1():
    for i in range(20):
        a = np.zeros([i])


@dace.program
def should_not_apply_2():
    for i in range(2, 20):
        a = np.ndarray([i], np.float64)
        a[0:2] = 0


@dace.program
def should_not_apply_3():
    a = np.ndarray((2, 10), np.float64)
    for i in range(20):
        for j in dace.map[10]:
            a[i % 2, j] = a[(i + 1) % 2, j]


@dace.program
def apply_multiple_times(A: dace.float64[10, 10, 10]):
    for i in range(10):
        for j in range(10):
            for k in dace.map[0:10]:
                A[k, i, j] = i * 100 + j * 10 + k


@dace.program
def apply_multiple_times_1(A: dace.float64[10, 10, 10, 10]):
    l = 5
    for i in range(l, 10):
        for j in range(l, 10):
            for k in dace.map[0:10]:
                A[k, i, j, l] = k * 1000 + i * 100 + j * 10 + l


def _semantic_eq(program):
    A1 = np.random.rand(16, 16)
    A2 = np.copy(A1)

    sdfg = program.to_sdfg(simplify=True)
    sdfg(A1, I=A1.shape[0], J=A1.shape[1])

    count = sdfg.apply_transformations(MoveLoopIntoMap)
    assert count > 0
    sdfg(A2, I=A2.shape[0], J=A2.shape[1])

    assert np.allclose(A1, A2)


def test_forward_loops_semantic_eq():
    _semantic_eq(forward_loop)


def test_backward_loops_semantic_eq():
    _semantic_eq(backward_loop)


def test_multiple_edges():
    _semantic_eq(multiple_edges)


def test_itervar_in_map_range():
    sdfg = should_not_apply_1.to_sdfg(simplify=True)
    count = sdfg.apply_transformations(MoveLoopIntoMap)
    assert count == 0


def test_itervar_in_data():
    sdfg = should_not_apply_2.to_sdfg(simplify=True)
    count = sdfg.apply_transformations(MoveLoopIntoMap)
    assert count == 0


def test_non_injective_index():
    sdfg = should_not_apply_3.to_sdfg(simplify=True)
    count = sdfg.apply_transformations(MoveLoopIntoMap)
    assert count == 0


def test_apply_multiple_times():
    sdfg = apply_multiple_times.to_sdfg(simplify=True)
    overall = 0
    count = 1
    while (count > 0):
        count = sdfg.apply_transformations_repeated(MoveLoopIntoMap, permissive=True)
        overall += count
        sdfg.simplify()

    assert overall == 2

    val = np.zeros((10, 10, 10), dtype=np.float64)
    ref = val.copy()

    sdfg(A=val)
    apply_multiple_times.f(ref)

    assert np.allclose(val, ref)


def test_apply_multiple_times_1():
    sdfg = apply_multiple_times_1.to_sdfg(simplify=True)
    overall = 0
    count = 1
    while (count > 0):
        count = sdfg.apply_transformations_repeated(MoveLoopIntoMap, permissive=True)
        overall += count
        sdfg.simplify()

    assert overall == 2

    val = np.zeros((10, 10, 10, 10), dtype=np.float64)
    ref = val.copy()

    sdfg(A=val)
    apply_multiple_times_1.f(ref)

    assert np.allclose(val, ref)


def test_more_than_a_map():
    """ `out` is read and written indirectly by the MapExit, potentially leading to a RW dependency.

    Note that there is actually no dependency, however, the transformation, because it relies
    on `SDFGState.read_and_write_sets()` it can not detect this and can thus not be applied.
    """
    sdfg = dace.SDFG('more_than_a_map')
    _, aarr = sdfg.add_array('A', (3, 3), dace.float64)
    _, barr = sdfg.add_array('B', (3, 3), dace.float64)
    _, oarr = sdfg.add_array('out', (3, 3), dace.float64)
    _, tarr = sdfg.add_array('tmp', (3, 3), dace.float64, transient=True)
    loop = LoopRegion('myloop', '_ < 10', '_', '_ = 0', '_ = _ + 1')
    sdfg.add_node(loop)
    body = loop.add_state('map_state')
    aread = body.add_access('A')
    oread = body.add_access('out')
    bread = body.add_access('B')
    twrite = body.add_access('tmp')
    owrite = body.add_access('out')
    body.add_mapped_tasklet('op',
                            dict(i='0:3', j='0:3'),
                            dict(__in1=dace.Memlet('out[i, j]'), __in2=dace.Memlet('B[i, j]')),
                            '__out = __in1 - __in2',
                            dict(__out=dace.Memlet('tmp[i, j]')),
                            external_edges=True,
                            input_nodes=dict(out=oread, B=bread),
                            output_nodes=dict(tmp=twrite))
    body.add_nedge(aread, oread, dace.Memlet.from_array('A', aarr))
    body.add_nedge(twrite, owrite, dace.Memlet.from_array('out', oarr))
    count = sdfg.apply_transformations(MoveLoopIntoMap)
    assert count == 0


def test_more_than_a_map_1():
    """
    `out` is written indirectly by the MapExit but is not read and, therefore, does not create a RW dependency.
    """
    sdfg = dace.SDFG('more_than_a_map_1')
    _, aarr = sdfg.add_array('A', (3, 3), dace.float64)
    _, barr = sdfg.add_array('B', (3, 3), dace.float64)
    _, oarr = sdfg.add_array('out', (3, 3), dace.float64)
    _, tarr = sdfg.add_array('tmp', (3, 3), dace.float64, transient=True)
    loop = LoopRegion('myloop', '_ < 10', '_', '_ = 0', '_ = _ + 1')
    sdfg.add_node(loop)
    body = loop.add_state('map_state')
    aread = body.add_access('A')
    bread = body.add_access('B')
    twrite = body.add_access('tmp')
    owrite = body.add_access('out')
    body.add_mapped_tasklet('op',
                            dict(i='0:3', j='0:3'),
                            dict(__in1=dace.Memlet('A[i, j]'), __in2=dace.Memlet('B[i, j]')),
                            '__out = __in1 - __in2',
                            dict(__out=dace.Memlet('tmp[i, j]')),
                            external_edges=True,
                            input_nodes=dict(A=aread, B=bread),
                            output_nodes=dict(tmp=twrite))
    body.add_nedge(twrite, owrite, dace.Memlet.from_array('out', oarr))
    count = sdfg.apply_transformations(MoveLoopIntoMap)
    assert count > 0

    A = np.arange(9, dtype=np.float64).reshape(3, 3).copy()
    B = np.arange(9, 18, dtype=np.float64).reshape(3, 3).copy()
    val = np.empty((3, 3), dtype=np.float64)
    sdfg(A=A, B=B, out=val)

    def reference(A, B):
        for i in range(10):
            tmp = A - B
            out = tmp
        return out

    ref = reference(A, B)
    assert np.allclose(val, ref)


def test_more_than_a_map_2():
    """ `out` is written indirectly by the MapExit with a subset dependent on the loop variable. This creates a RW
        dependency.
    """
    sdfg = dace.SDFG('more_than_a_map_2')
    _, aarr = sdfg.add_array('A', (3, 3), dace.float64)
    _, barr = sdfg.add_array('B', (3, 3), dace.float64)
    _, oarr = sdfg.add_array('out', (3, 3), dace.float64)
    _, tarr = sdfg.add_array('tmp', (3, 3), dace.float64, transient=True)
    loop = LoopRegion('myloop', 'k < 10', 'k', 'k = 0', 'k = k + 1')
    sdfg.add_node(loop)
    body = loop.add_state('map_state')
    aread = body.add_access('A')
    bread = body.add_access('B')
    twrite = body.add_access('tmp')
    owrite = body.add_access('out')
    body.add_mapped_tasklet('op',
                            dict(i='0:3', j='0:3'),
                            dict(__in1=dace.Memlet('A[i, j]'), __in2=dace.Memlet('B[i, j]')),
                            '__out = __in1 - __in2',
                            dict(__out=dace.Memlet('tmp[i, j]')),
                            external_edges=True,
                            input_nodes=dict(A=aread, B=bread),
                            output_nodes=dict(tmp=twrite))
    body.add_nedge(twrite, owrite, dace.Memlet('out[k%3, (k+1)%3]', other_subset='(k+1)%3, k%3'))
    count = sdfg.apply_transformations(MoveLoopIntoMap)
    assert count == 0


def test_more_than_a_map_3():
    """ There are more than one connected components in the loop body. The transformation should not apply. """
    sdfg = dace.SDFG('more_than_a_map_3')
    _, aarr = sdfg.add_array('A', (3, 3), dace.float64)
    _, barr = sdfg.add_array('B', (3, 3), dace.float64)
    _, oarr = sdfg.add_array('out', (3, 3), dace.float64)
    _, tarr = sdfg.add_array('tmp', (3, 3), dace.float64, transient=True)
    loop = LoopRegion('myloop', '_ < 10', '_', '_ = 0', '_ = _ + 1')
    sdfg.add_node(loop)
    body = loop.add_state('map_state')
    aread = body.add_access('A')
    bread = body.add_access('B')
    twrite = body.add_access('tmp')
    owrite = body.add_access('out')
    body.add_mapped_tasklet('op',
                            dict(i='0:3', j='0:3'),
                            dict(__in1=dace.Memlet('A[i, j]'), __in2=dace.Memlet('B[i, j]')),
                            '__out = __in1 - __in2',
                            dict(__out=dace.Memlet('tmp[i, j]')),
                            external_edges=True,
                            input_nodes=dict(A=aread, B=bread),
                            output_nodes=dict(tmp=twrite))
    body.add_nedge(twrite, owrite, dace.Memlet.from_array('out', oarr))
    aread2 = body.add_access('A')
    owrite2 = body.add_access('out')
    body.add_nedge(aread2, owrite2, dace.Memlet.from_array('out', oarr))
    count = sdfg.apply_transformations(MoveLoopIntoMap)
    assert count == 0


def test_more_than_a_map_4():
    """
    The test is very similar to `test_more_than_a_map()`. But a memlet is different
    which leads to a RW dependency, which blocks the transformation.
    """
    sdfg = dace.SDFG('more_than_a_map')
    _, aarr = sdfg.add_array('A', (3, 3), dace.float64)
    _, barr = sdfg.add_array('B', (3, 3), dace.float64)
    _, oarr = sdfg.add_array('out', (3, 3), dace.float64)
    _, tarr = sdfg.add_array('tmp', (3, 3), dace.float64, transient=True)
    body = sdfg.add_state('map_state')
    aread = body.add_access('A')
    oread = body.add_access('out')
    bread = body.add_access('B')
    twrite = body.add_access('tmp')
    owrite = body.add_access('out')
    body.add_mapped_tasklet('op',
                            dict(i='0:3', j='0:3'),
                            dict(__in1=dace.Memlet('out[i, j]'), __in2=dace.Memlet('B[i, j]')),
                            '__out = __in1 - __in2',
                            dict(__out=dace.Memlet('tmp[i, j]')),
                            external_edges=True,
                            input_nodes=dict(out=oread, B=bread),
                            output_nodes=dict(tmp=twrite))
    body.add_nedge(aread, oread, dace.Memlet('A[Mod(_, 3), 0:3] -> [Mod(_ + 1, 3), 0:3]', aarr))
    body.add_nedge(twrite, owrite, dace.Memlet.from_array('out', oarr))
    sdfg.add_loop(None, body, None, '_', '0', '_ < 10', '_ + 1')

    sdfg_args_ref = {
        "A": np.array(np.random.rand(3, 3), dtype=np.float64),
        "B": np.array(np.random.rand(3, 3), dtype=np.float64),
        "out": np.array(np.random.rand(3, 3), dtype=np.float64),
    }
    sdfg_args_res = copy.deepcopy(sdfg_args_ref)

    # Perform the reference execution
    sdfg(**sdfg_args_ref)

    # Apply the transformation and execute the SDFG again.
    count = sdfg.apply_transformations(MoveLoopIntoMap, validate_all=True, validate=True)
    sdfg(**sdfg_args_res)

    for name in sdfg_args_ref.keys():
        assert np.allclose(sdfg_args_ref[name], sdfg_args_res[name])
    assert count == 0


# ``for k { map i; if c[k]: map i }``: every map writes only its own lane ``i``, so the loop can run per lane.
@dace.program
def branching_column(a: dace.float64[K, N], b: dace.float64[N], w: dace.float64[K], c: dace.float64[K]):
    for k in range(1, K):
        s = w[k] * 2.0
        for i in dace.map[0:N]:
            a[k, i] = a[k - 1, i] + b[i] * s
        if c[k] > 0.5:
            for i in dace.map[0:N]:
                b[i] = b[i] * 0.5 + a[k, i]


# Lane ``i`` reads the row lane ``i + 1`` wrote one trip earlier: a dependence between lanes.
@dace.program
def lane_shift(a: dace.float64[K, N], b: dace.float64[N], c: dace.float64[K]):
    for k in range(1, K):
        for i in dace.map[0:N - 1]:
            a[k, i] = a[k - 1, i + 1] + b[i]
        if c[k] > 0.5:
            for i in dace.map[0:N - 1]:
                b[i] = b[i] * 0.5 + a[k, i]


# The branch condition reads what lane 0 wrote this trip, so every lane depends on lane 0.
@dace.program
def lane_guard(a: dace.float64[K, N], b: dace.float64[N]):
    for k in range(1, K):
        for i in dace.map[0:N]:
            a[k, i] = a[k - 1, i] + b[i]
        if a[k, 0] > 0.5:
            for i in dace.map[0:N]:
                b[i] = b[i] * 0.5


# The branch inside the first map puts its body in a nested SDFG, bound per lane by its connector memlets.
@dace.program
def nested_column(a: dace.float64[K, N], b: dace.float64[N], c: dace.float64[K]):
    for k in range(1, K):
        for i in dace.map[0:N]:
            if b[i] > 0.5:
                a[k, i] = a[k - 1, i] + b[i]
            else:
                a[k, i] = a[k - 1, i] * 0.5
        if c[k] > 0.5:
            for i in dace.map[0:N]:
                b[i] = b[i] * 0.75 + a[k, i] * 0.125


def lane_loop(prog):
    sdfg = prog.to_sdfg(simplify=True)
    return sdfg, next(r for r in sdfg.all_control_flow_regions() if isinstance(r, LoopRegion))


def branching_inputs():
    rng = np.random.default_rng(0)
    return dict(a=rng.random((7, 13)), b=rng.random(13), w=rng.random(7), c=rng.random(7), K=7, N=13)


def graph_digest(sdfg):
    drop = {'guid', 'cfg_list_id', 'hash', 'transformation_hist', 'orig_sdfg'}

    def strip(o):
        if isinstance(o, dict):
            return {k: strip(v) for k, v in o.items() if k not in drop}
        return [strip(v) for v in o] if isinstance(o, list) else o

    return json.dumps(strip(sdfg.to_json()), sort_keys=True)


def test_a_loop_over_branching_maps_moves_into_one_map_over_their_lanes():
    sdfg, loop = lane_loop(branching_column)
    assert any(isinstance(r, ConditionalBlock) for r in loop.all_control_flow_regions())
    MoveLoopIntoMap.apply_to(sdfg, options={'cfg_body': True}, loop=loop)
    sdfg.validate()

    assert not any(isinstance(b, LoopRegion) for b in sdfg.nodes())
    outer = [n for s in sdfg.states() for n in s.nodes() if isinstance(n, dace.nodes.MapEntry)]
    assert len(outer) == 1 and str(outer[0].map.range) == '0:N', outer
    lane = outer[0].map.params[0]
    inner = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.MapEntry) and n is not outer[0]]
    assert len(inner) == 2 and all(str(m.map.range) == lane for m in inner), [str(m.map.range) for m in inner]
    nested = [r for r in sdfg.all_control_flow_regions(recursive=True) if isinstance(r, LoopRegion)]
    assert len(nested) == 1 and nested[0].sdfg is not sdfg


def test_a_loop_moved_into_its_lanes_computes_what_the_loop_did():
    sdfg, loop = lane_loop(branching_column)
    want = branching_inputs()
    got = copy.deepcopy(want)
    copy.deepcopy(sdfg)(**want)
    MoveLoopIntoMap.apply_to(sdfg, options={'cfg_body': True}, loop=loop)
    sdfg(**got)
    for name in ('a', 'b'):
        np.testing.assert_allclose(got[name], want[name], rtol=1e-13, atol=0, err_msg=name)


def test_a_map_body_behind_a_nested_sdfg_moves_per_lane():
    sdfg, loop = lane_loop(nested_column)
    assert any(isinstance(n, dace.nodes.NestedSDFG) for s in loop.all_states() for n in s.nodes())
    args = branching_inputs()
    del args['w']
    want, got = copy.deepcopy(args), copy.deepcopy(args)
    copy.deepcopy(sdfg)(**want)
    MoveLoopIntoMap.apply_to(sdfg, options={'cfg_body': True}, loop=loop)
    assert not any(isinstance(b, LoopRegion) for b in sdfg.nodes())
    sdfg(**got)
    for name in ('a', 'b'):
        np.testing.assert_allclose(got[name], want[name], rtol=1e-13, atol=0, err_msg=name)


def wide_fill_loop(name: str, read_after: bool) -> tuple:
    """``tmp[:] = 1; for k { tmp[0:16] = 0; map i in 1:15 { tmp[i] += a[k-1, i] }; map i { a[k, i] = tmp[i] } }``,
    optionally followed by a read of ``tmp[0]`` -- an element of the fill no lane owns."""
    sdfg = dace.SDFG(name)
    sdfg.add_array('a', (7, 16), dace.float64)
    sdfg.add_array('out', (1, ), dace.float64)
    sdfg.add_array('tmp', (16, ), dace.float64, transient=True)
    init = sdfg.add_state('init', is_start_block=True)
    init.add_edge(FillLibraryNode('init_tmp', value=1.0), FillLibraryNode.OUTPUT_CONNECTOR_NAME, init.add_write('tmp'),
                  None, dace.Memlet('tmp[0:16]'))
    loop = LoopRegion('kloop', 'k < 7', 'k', 'k = 1', 'k = k + 1')
    sdfg.add_node(loop)
    sdfg.add_edge(init, loop, dace.InterstateEdge())
    first = loop.add_state('accumulate', is_start_block=True)
    fill = FillLibraryNode('fill_tmp', value=0.0)
    filled = first.add_access('tmp')
    first.add_edge(fill, FillLibraryNode.OUTPUT_CONNECTOR_NAME, filled, None, dace.Memlet('tmp[0:16]'))
    first.add_mapped_tasklet('accumulate', {'i': '1:15'}, {
        '__t': dace.Memlet('tmp[i]'),
        '__a': dace.Memlet('a[k - 1, i]')
    },
                             '__out = __t + 2.0 * __a', {'__out': dace.Memlet('tmp[i]')},
                             input_nodes={'tmp': filled},
                             external_edges=True)
    second = loop.add_state('store')
    loop.add_edge(first, second, dace.InterstateEdge())
    second.add_mapped_tasklet('store', {'i': '1:15'}, {'__t': dace.Memlet('tmp[i]')},
                              '__out = __t + 1.0', {'__out': dace.Memlet('a[k, i]')},
                              external_edges=True)
    if read_after:
        after = sdfg.add_state_after(loop, 'read_after')
        after.add_nedge(after.add_read('tmp'), after.add_write('out'), dace.Memlet('tmp[0] -> [0]'))
    sdfg.validate()
    return sdfg, loop


def test_a_fill_wider_than_the_lanes_shrinks_to_each_lane_when_the_rest_is_dead():
    sdfg, loop = wide_fill_loop('wide_fill_dead_rest', read_after=False)
    rng = np.random.default_rng(1)
    want = dict(a=rng.random((7, 16)), out=np.zeros(1))
    got = copy.deepcopy(want)
    copy.deepcopy(sdfg)(**want)
    MoveLoopIntoMap.apply_to(sdfg, options={'cfg_body': True}, loop=loop)
    sdfg.validate()
    fills = [
        e.data.subset for n, s in sdfg.all_nodes_recursive() if isinstance(n, FillLibraryNode) and n.label == 'fill_tmp'
        for e in s.out_edges(n)
    ]
    assert len(fills) == 1 and fills[0].num_elements() == 1, fills
    sdfg(**got)
    np.testing.assert_allclose(got['a'], want['a'], rtol=1e-13, atol=0)


def test_a_fill_wider_than_the_lanes_is_refused_when_its_rest_is_read():
    sdfg, loop = wide_fill_loop('wide_fill_read_rest', read_after=True)
    assert analyze_lanes(loop, sdfg).refusal == 'tmp, filled beyond the lanes, is read outside them in read_after'


def test_a_dependence_between_lanes_refuses_the_interchange():
    sdfg, loop = lane_loop(lane_shift)
    before = graph_digest(sdfg)
    assert analyze_lanes(loop, sdfg).refusal == 'a is accessed across lanes'
    assert sdfg.apply_transformations(MoveLoopIntoMap, options={'cfg_body': True}) == 0
    assert graph_digest(sdfg) == before


def test_a_branch_reading_one_lanes_result_refuses_the_interchange():
    sdfg, loop = lane_loop(lane_guard)
    assert analyze_lanes(loop, sdfg).refusal == 'lane-independent code reads a, which the maps write per lane'
    assert sdfg.apply_transformations(MoveLoopIntoMap, options={'cfg_body': True}) == 0


def test_a_branch_condition_naming_lane_data_refuses_the_interchange():
    """The condition reads ``a`` by name, with no access node anywhere to show the read."""
    sdfg = dace.SDFG('lane_condition_read')
    sdfg.add_array('a', (7, 13), dace.float64)
    sdfg.add_array('b', (13, ), dace.float64)
    loop = LoopRegion('kloop', 'k < 7', 'k', 'k = 1', 'k = k + 1')
    sdfg.add_node(loop, is_start_block=True)
    first = loop.add_state('shift', is_start_block=True)
    first.add_mapped_tasklet('shift',
                             dict(i='0:13'),
                             dict(__in=dace.Memlet('a[k - 1, i]')),
                             '__out = __in + 1.0',
                             dict(__out=dace.Memlet('a[k, i]')),
                             external_edges=True)
    guard = ConditionalBlock('guard', sdfg, loop)
    loop.add_node(guard)
    loop.add_edge(first, guard, dace.InterstateEdge())
    body = ControlFlowRegion('guard_body', sdfg, guard)
    body.add_state('halve', is_start_block=True).add_mapped_tasklet('halve',
                                                                    dict(i='0:13'),
                                                                    dict(__in=dace.Memlet('b[i]')),
                                                                    '__out = __in * 0.5',
                                                                    dict(__out=dace.Memlet('b[i]')),
                                                                    external_edges=True)
    guard.add_branch(CodeBlock('a[k, 0] > 0.5'), body)
    sdfg.validate()
    assert analyze_lanes(loop, sdfg).refusal == 'lane-independent code reads a, which the maps write per lane'


def test_cfg_body_leaves_the_single_map_interchange_byte_identical():
    classic, generalized = forward_loop.to_sdfg(simplify=True), forward_loop.to_sdfg(simplify=True)
    assert classic.apply_transformations(MoveLoopIntoMap) == 1
    assert generalized.apply_transformations(MoveLoopIntoMap, options={'cfg_body': True}) == 1
    assert graph_digest(generalized) == graph_digest(classic)


def test_the_cpu_interchange_gate_leaves_a_branching_body_alone():
    """The CPU gate only ever sees the single-map shape; the generalized body is a GPU decision."""
    sdfg, loop = lane_loop(branching_column)
    before = graph_digest(sdfg)
    assert MoveLoopIntoMapGated(target='cpu').apply_pass(sdfg, {}) is None
    assert graph_digest(sdfg) == before


if __name__ == '__main__':
    test_forward_loops_semantic_eq()
    test_backward_loops_semantic_eq()
    test_multiple_edges()
    test_itervar_in_map_range()
    test_itervar_in_data()
    test_non_injective_index()
    test_apply_multiple_times()
    test_apply_multiple_times_1()
    test_more_than_a_map()
    test_more_than_a_map_1()
    test_more_than_a_map_2()
    test_more_than_a_map_3()
    test_more_than_a_map_4()
    test_a_loop_over_branching_maps_moves_into_one_map_over_their_lanes()
    test_a_loop_moved_into_its_lanes_computes_what_the_loop_did()
    test_a_map_body_behind_a_nested_sdfg_moves_per_lane()
    test_a_fill_wider_than_the_lanes_shrinks_to_each_lane_when_the_rest_is_dead()
    test_a_fill_wider_than_the_lanes_is_refused_when_its_rest_is_read()
    test_a_dependence_between_lanes_refuses_the_interchange()
    test_a_branch_reading_one_lanes_result_refuses_the_interchange()
    test_a_branch_condition_naming_lane_data_refuses_the_interchange()
    test_cfg_body_leaves_the_single_map_interchange_byte_identical()
    test_the_cpu_interchange_gate_leaves_a_branching_body_alone()
