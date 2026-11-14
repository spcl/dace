# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.

import copy
import json

import numpy as np

import dace
from dace.libraries.standard.nodes.fill.node import FillLibraryNode
from dace.properties import CodeBlock
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion, LoopRegion
from dace.transformation.interstate import MoveLoopIntoMap
from dace.transformation.interstate.move_loop_into_map import analyze_lanes
from dace.transformation.passes.canonicalize.move_loop_into_map_gated import MoveLoopIntoMapGated
from tests.sdfg.cfg_list_in_place_test import assert_tree_consistent


def assert_cfg_list_matches_reset(sdfg: dace.SDFG) -> None:
    """The kept CFG list and ids equal a fresh copy's after ``reset_cfg_list``, and every parent pointer holds."""
    fresh = copy.deepcopy(sdfg)
    fresh.reset_cfg_list()
    kept = [(type(r).__name__, r.label, r.cfg_id) for r in sdfg.cfg_list]
    assert kept == [(type(r).__name__, r.label, r.cfg_id) for r in fresh.cfg_list]
    assert_tree_consistent(sdfg)


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
    assert_cfg_list_matches_reset(sdfg)
    sdfg(A2, I=A2.shape[0], J=A2.shape[1])

    assert np.allclose(A1, A2)


def test_forward_loops_semantic_eq():
    _semantic_eq(forward_loop)


def test_backward_loops_semantic_eq():
    _semantic_eq(backward_loop)


def test_multiple_edges():
    _semantic_eq(multiple_edges)


K = dace.symbol("K")


@dace.program
def loop_bound_only_in_the_loop(data: dace.float64[I, J]):
    for i in range(1, K):
        for j in dace.map[0:J]:
            data[i, j] = data[i - 1, j]


def test_loop_bound_symbols_are_declared_in_the_nested_sdfg():
    sdfg = loop_bound_only_in_the_loop.to_sdfg(simplify=True)
    assert sdfg.apply_transformations(MoveLoopIntoMap) == 1
    for node, _ in sdfg.all_nodes_recursive():
        if isinstance(node, dace.nodes.NestedSDFG):
            assert node.symbol_mapping.keys() <= node.sdfg.symbols.keys()


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


def test_scaled_index_is_injective_for_a_nonzero_multiplier():
    """``data[S * i, j]`` names a different row in each iteration only if ``S`` is nonzero, as declared for ``P``."""
    S = dace.symbol("S")
    P = dace.symbol("P", positive=True)

    @dace.program
    def unknown_multiplier(data: dace.float64[I, J]):
        for i in range(I):
            for j in dace.map[0:J]:
                data[S * i, j] = data[S * i, j] + 1

    @dace.program
    def positive_multiplier(data: dace.float64[I, J]):
        for i in range(I):
            for j in dace.map[0:J]:
                data[P * i, j] = data[P * i, j] + 1

    assert unknown_multiplier.to_sdfg(simplify=True).apply_transformations(MoveLoopIntoMap) == 0
    assert positive_multiplier.to_sdfg(simplify=True).apply_transformations(MoveLoopIntoMap) == 1


def test_apply_multiple_times():
    sdfg = apply_multiple_times.to_sdfg(simplify=True)
    overall = 0
    count = 1
    while count > 0:
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
    while count > 0:
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
    """`out` is read and written indirectly by the MapExit, potentially leading to a RW dependency.

    Note that there is actually no dependency, however, the transformation, because it relies
    on `SDFGState.read_and_write_sets()` it can not detect this and can thus not be applied.
    """
    sdfg = dace.SDFG("more_than_a_map")
    _, aarr = sdfg.add_array("A", (3, 3), dace.float64)
    _, barr = sdfg.add_array("B", (3, 3), dace.float64)
    _, oarr = sdfg.add_array("out", (3, 3), dace.float64)
    _, tarr = sdfg.add_array("tmp", (3, 3), dace.float64, transient=True)
    loop = LoopRegion("myloop", "_ < 10", "_", "_ = 0", "_ = _ + 1")
    sdfg.add_node(loop)
    body = loop.add_state("map_state")
    aread = body.add_access("A")
    oread = body.add_access("out")
    bread = body.add_access("B")
    twrite = body.add_access("tmp")
    owrite = body.add_access("out")
    body.add_mapped_tasklet(
        "op",
        dict(i="0:3", j="0:3"),
        dict(__in1=dace.Memlet("out[i, j]"), __in2=dace.Memlet("B[i, j]")),
        "__out = __in1 - __in2",
        dict(__out=dace.Memlet("tmp[i, j]")),
        external_edges=True,
        input_nodes=dict(out=oread, B=bread),
        output_nodes=dict(tmp=twrite),
    )
    body.add_nedge(aread, oread, dace.Memlet.from_array("A", aarr))
    body.add_nedge(twrite, owrite, dace.Memlet.from_array("out", oarr))
    count = sdfg.apply_transformations(MoveLoopIntoMap)
    assert count == 0


def test_more_than_a_map_1():
    """
    `out` is written indirectly by the MapExit but is not read and, therefore, does not create a RW dependency.
    """
    sdfg = dace.SDFG("more_than_a_map_1")
    _, aarr = sdfg.add_array("A", (3, 3), dace.float64)
    _, barr = sdfg.add_array("B", (3, 3), dace.float64)
    _, oarr = sdfg.add_array("out", (3, 3), dace.float64)
    _, tarr = sdfg.add_array("tmp", (3, 3), dace.float64, transient=True)
    loop = LoopRegion("myloop", "_ < 10", "_", "_ = 0", "_ = _ + 1")
    sdfg.add_node(loop)
    body = loop.add_state("map_state")
    aread = body.add_access("A")
    bread = body.add_access("B")
    twrite = body.add_access("tmp")
    owrite = body.add_access("out")
    body.add_mapped_tasklet(
        "op",
        dict(i="0:3", j="0:3"),
        dict(__in1=dace.Memlet("A[i, j]"), __in2=dace.Memlet("B[i, j]")),
        "__out = __in1 - __in2",
        dict(__out=dace.Memlet("tmp[i, j]")),
        external_edges=True,
        input_nodes=dict(A=aread, B=bread),
        output_nodes=dict(tmp=twrite),
    )
    body.add_nedge(twrite, owrite, dace.Memlet.from_array("out", oarr))
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
    """`out` is written indirectly by the MapExit with a subset dependent on the loop variable. This creates a RW
    dependency.
    """
    sdfg = dace.SDFG("more_than_a_map_2")
    _, aarr = sdfg.add_array("A", (3, 3), dace.float64)
    _, barr = sdfg.add_array("B", (3, 3), dace.float64)
    _, oarr = sdfg.add_array("out", (3, 3), dace.float64)
    _, tarr = sdfg.add_array("tmp", (3, 3), dace.float64, transient=True)
    loop = LoopRegion("myloop", "k < 10", "k", "k = 0", "k = k + 1")
    sdfg.add_node(loop)
    body = loop.add_state("map_state")
    aread = body.add_access("A")
    bread = body.add_access("B")
    twrite = body.add_access("tmp")
    owrite = body.add_access("out")
    body.add_mapped_tasklet(
        "op",
        dict(i="0:3", j="0:3"),
        dict(__in1=dace.Memlet("A[i, j]"), __in2=dace.Memlet("B[i, j]")),
        "__out = __in1 - __in2",
        dict(__out=dace.Memlet("tmp[i, j]")),
        external_edges=True,
        input_nodes=dict(A=aread, B=bread),
        output_nodes=dict(tmp=twrite),
    )
    body.add_nedge(twrite, owrite, dace.Memlet("out[k%3, (k+1)%3]", other_subset="(k+1)%3, k%3"))
    count = sdfg.apply_transformations(MoveLoopIntoMap)
    assert count == 0


def test_more_than_a_map_3():
    """There are more than one connected components in the loop body. The transformation should not apply."""
    sdfg = dace.SDFG("more_than_a_map_3")
    _, aarr = sdfg.add_array("A", (3, 3), dace.float64)
    _, barr = sdfg.add_array("B", (3, 3), dace.float64)
    _, oarr = sdfg.add_array("out", (3, 3), dace.float64)
    _, tarr = sdfg.add_array("tmp", (3, 3), dace.float64, transient=True)
    loop = LoopRegion("myloop", "_ < 10", "_", "_ = 0", "_ = _ + 1")
    sdfg.add_node(loop)
    body = loop.add_state("map_state")
    aread = body.add_access("A")
    bread = body.add_access("B")
    twrite = body.add_access("tmp")
    owrite = body.add_access("out")
    body.add_mapped_tasklet(
        "op",
        dict(i="0:3", j="0:3"),
        dict(__in1=dace.Memlet("A[i, j]"), __in2=dace.Memlet("B[i, j]")),
        "__out = __in1 - __in2",
        dict(__out=dace.Memlet("tmp[i, j]")),
        external_edges=True,
        input_nodes=dict(A=aread, B=bread),
        output_nodes=dict(tmp=twrite),
    )
    body.add_nedge(twrite, owrite, dace.Memlet.from_array("out", oarr))
    aread2 = body.add_access("A")
    owrite2 = body.add_access("out")
    body.add_nedge(aread2, owrite2, dace.Memlet.from_array("out", oarr))
    count = sdfg.apply_transformations(MoveLoopIntoMap)
    assert count == 0


def test_more_than_a_map_4():
    """
    The test is very similar to `test_more_than_a_map()`. But a memlet is different
    which leads to a RW dependency, which blocks the transformation.
    """
    sdfg = dace.SDFG("more_than_a_map")
    _, aarr = sdfg.add_array("A", (3, 3), dace.float64)
    _, barr = sdfg.add_array("B", (3, 3), dace.float64)
    _, oarr = sdfg.add_array("out", (3, 3), dace.float64)
    _, tarr = sdfg.add_array("tmp", (3, 3), dace.float64, transient=True)
    body = sdfg.add_state("map_state")
    aread = body.add_access("A")
    oread = body.add_access("out")
    bread = body.add_access("B")
    twrite = body.add_access("tmp")
    owrite = body.add_access("out")
    body.add_mapped_tasklet(
        "op",
        dict(i="0:3", j="0:3"),
        dict(__in1=dace.Memlet("out[i, j]"), __in2=dace.Memlet("B[i, j]")),
        "__out = __in1 - __in2",
        dict(__out=dace.Memlet("tmp[i, j]")),
        external_edges=True,
        input_nodes=dict(out=oread, B=bread),
        output_nodes=dict(tmp=twrite),
    )
    body.add_nedge(aread, oread, dace.Memlet("A[Mod(_, 3), 0:3] -> [Mod(_ + 1, 3), 0:3]", aarr))
    body.add_nedge(twrite, owrite, dace.Memlet.from_array("out", oarr))
    sdfg.add_loop(None, body, None, "_", "0", "_ < 10", "_ + 1")

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
        for i in dace.map[0 : N - 1]:
            a[k, i] = a[k - 1, i + 1] + b[i]
        if c[k] > 0.5:
            for i in dace.map[0 : N - 1]:
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
    drop = {"guid", "cfg_list_id", "hash", "transformation_hist", "orig_sdfg"}

    def strip(o):
        if isinstance(o, dict):
            return {k: strip(v) for k, v in o.items() if k not in drop}
        return [strip(v) for v in o] if isinstance(o, list) else o

    return json.dumps(strip(sdfg.to_json()), sort_keys=True)


def test_a_loop_over_branching_maps_moves_into_one_map_over_their_lanes():
    sdfg, loop = lane_loop(branching_column)
    assert any(isinstance(r, ConditionalBlock) for r in loop.all_control_flow_regions())
    MoveLoopIntoMap.apply_to(sdfg, options={"cfg_body": True}, loop=loop)
    sdfg.validate()

    assert not any(isinstance(b, LoopRegion) for b in sdfg.nodes())
    outer = [n for s in sdfg.states() for n in s.nodes() if isinstance(n, dace.nodes.MapEntry)]
    assert len(outer) == 1 and str(outer[0].map.range) == "0:N", outer
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
    MoveLoopIntoMap.apply_to(sdfg, options={"cfg_body": True}, loop=loop)
    sdfg(**got)
    for name in ("a", "b"):
        np.testing.assert_allclose(got[name], want[name], rtol=1e-13, atol=0, err_msg=name)


def test_a_map_body_behind_a_nested_sdfg_moves_per_lane():
    sdfg, loop = lane_loop(nested_column)
    assert any(isinstance(n, dace.nodes.NestedSDFG) for s in loop.states() for n in s.nodes())
    args = branching_inputs()
    del args["w"]
    want, got = copy.deepcopy(args), copy.deepcopy(args)
    copy.deepcopy(sdfg)(**want)
    MoveLoopIntoMap.apply_to(sdfg, options={"cfg_body": True}, loop=loop)
    assert not any(isinstance(b, LoopRegion) for b in sdfg.nodes())
    sdfg(**got)
    for name in ("a", "b"):
        np.testing.assert_allclose(got[name], want[name], rtol=1e-13, atol=0, err_msg=name)


def wide_fill_loop(name: str, read_after: bool) -> tuple:
    """``tmp[:] = 1; for k { tmp[0:16] = 0; map i in 1:15 { tmp[i] += a[k-1, i] }; map i { a[k, i] = tmp[i] } }``,
    optionally followed by a read of ``tmp[0]`` -- an element of the fill no lane owns."""
    sdfg = dace.SDFG(name)
    sdfg.add_array("a", (7, 16), dace.float64)
    sdfg.add_array("out", (1,), dace.float64)
    sdfg.add_array("tmp", (16,), dace.float64, transient=True)
    init = sdfg.add_state("init", is_start_block=True)
    init.add_edge(
        FillLibraryNode("init_tmp", value=1.0),
        FillLibraryNode.OUTPUT_CONNECTOR_NAME,
        init.add_write("tmp"),
        None,
        dace.Memlet("tmp[0:16]"),
    )
    loop = LoopRegion("kloop", "k < 7", "k", "k = 1", "k = k + 1")
    sdfg.add_node(loop)
    sdfg.add_edge(init, loop, dace.InterstateEdge())
    first = loop.add_state("accumulate", is_start_block=True)
    fill = FillLibraryNode("fill_tmp", value=0.0)
    filled = first.add_access("tmp")
    first.add_edge(fill, FillLibraryNode.OUTPUT_CONNECTOR_NAME, filled, None, dace.Memlet("tmp[0:16]"))
    first.add_mapped_tasklet(
        "accumulate",
        {"i": "1:15"},
        {"__t": dace.Memlet("tmp[i]"), "__a": dace.Memlet("a[k - 1, i]")},
        "__out = __t + 2.0 * __a",
        {"__out": dace.Memlet("tmp[i]")},
        input_nodes={"tmp": filled},
        external_edges=True,
    )
    second = loop.add_state("store")
    loop.add_edge(first, second, dace.InterstateEdge())
    second.add_mapped_tasklet(
        "store",
        {"i": "1:15"},
        {"__t": dace.Memlet("tmp[i]")},
        "__out = __t + 1.0",
        {"__out": dace.Memlet("a[k, i]")},
        external_edges=True,
    )
    if read_after:
        after = sdfg.add_state_after(loop, "read_after")
        after.add_nedge(after.add_read("tmp"), after.add_write("out"), dace.Memlet("tmp[0] -> [0]"))
    sdfg.validate()
    return sdfg, loop


def test_a_fill_wider_than_the_lanes_shrinks_to_each_lane_when_the_rest_is_dead():
    sdfg, loop = wide_fill_loop("wide_fill_dead_rest", read_after=False)
    rng = np.random.default_rng(1)
    want = dict(a=rng.random((7, 16)), out=np.zeros(1))
    got = copy.deepcopy(want)
    copy.deepcopy(sdfg)(**want)
    MoveLoopIntoMap.apply_to(sdfg, options={"cfg_body": True}, loop=loop)
    sdfg.validate()
    fills = [
        e.data.subset
        for n, s in sdfg.all_nodes_recursive()
        if isinstance(n, FillLibraryNode) and n.label == "fill_tmp"
        for e in s.out_edges(n)
    ]
    assert len(fills) == 1 and fills[0].num_elements() == 1, fills
    sdfg(**got)
    np.testing.assert_allclose(got["a"], want["a"], rtol=1e-13, atol=0)


def test_a_fill_wider_than_the_lanes_is_refused_when_its_rest_is_read():
    sdfg, loop = wide_fill_loop("wide_fill_read_rest", read_after=True)
    assert analyze_lanes(loop, sdfg).refusal == "tmp, filled beyond the lanes, is read outside them in read_after"


def twice_nested_loop(name: str) -> tuple:
    """``for k { map i { nest { nest { a[k, i] = a[k-1, i] + 1 } } }; map i { b[i] = a[k, i] } }`` where both nested
    SDFG boundaries carry whole-array memlets, as they do before memlet propagation reaches them."""
    inner = dace.SDFG(f"{name}_inner")
    inner.add_array("a", (7, 13), dace.float64)
    inner.add_symbol("i", dace.int64)
    inner.add_symbol("k", dace.int64)
    istate = inner.add_state("add", is_start_block=True)
    tasklet = istate.add_tasklet("add", {"x"}, {"y"}, "y = x + 1.0")
    istate.add_edge(istate.add_read("a"), None, tasklet, "x", dace.Memlet("a[k - 1, i]"))
    istate.add_edge(tasklet, "y", istate.add_write("a"), None, dace.Memlet("a[k, i]"))
    middle = dace.SDFG(f"{name}_middle")
    middle.add_array("a", (7, 13), dace.float64)
    middle.add_symbol("i", dace.int64)
    middle.add_symbol("k", dace.int64)
    mstate = middle.add_state("nest", is_start_block=True)
    node = mstate.add_nested_sdfg(inner, {"a": None}, {"a": None}, {"i": "i", "k": "k"})
    mstate.add_edge(mstate.add_read("a"), None, node, "a", dace.Memlet("a[0:7, 0:13]"))
    mstate.add_edge(node, "a", mstate.add_write("a"), None, dace.Memlet("a[0:7, 0:13]"))
    sdfg = dace.SDFG(name)
    sdfg.add_array("a", (7, 13), dace.float64)
    sdfg.add_array("b", (13,), dace.float64)
    loop = LoopRegion("kloop", "k < 7", "k", "k = 1", "k = k + 1")
    sdfg.add_node(loop, is_start_block=True)
    first = loop.add_state("step", is_start_block=True)
    entry, exit_ = first.add_map("step", {"i": "0:13"})
    node = first.add_nested_sdfg(middle, {"a": None}, {"a": None}, {"i": "i", "k": "k"})
    first.add_memlet_path(first.add_read("a"), entry, node, dst_conn="a", memlet=dace.Memlet("a[0:7, 0:13]"))
    first.add_memlet_path(node, exit_, first.add_write("a"), src_conn="a", memlet=dace.Memlet("a[0:7, 0:13]"))
    second = loop.add_state("store")
    loop.add_edge(first, second, dace.InterstateEdge())
    second.add_mapped_tasklet(
        "store", {"i": "0:13"}, {"x": dace.Memlet("a[k, i]")}, "y = x", {"y": dace.Memlet("b[i]")}, external_edges=True
    )
    sdfg.validate()
    return sdfg, loop


def test_lane_accesses_behind_two_nested_sdfgs_are_read_at_the_innermost_memlet():
    """A whole-array connector memlet says nothing about lanes, so the analysis reads through every nesting level."""
    sdfg, loop = twice_nested_loop("twice_nested_lanes")
    assert analyze_lanes(loop, sdfg).refusal is None
    rng = np.random.default_rng(2)
    want = dict(a=rng.random((7, 13)), b=np.zeros(13))
    got = copy.deepcopy(want)
    copy.deepcopy(sdfg)(**want)
    MoveLoopIntoMap.apply_to(sdfg, options={"cfg_body": True}, loop=loop)
    assert not any(isinstance(b, LoopRegion) for b in sdfg.nodes())
    sdfg(**got)
    for name in ("a", "b"):
        np.testing.assert_allclose(got[name], want[name], rtol=1e-13, atol=0, err_msg=name)


def viewed_write_loop(name: str) -> tuple:
    """``for k { map i { view = a; view[k, i] = a[k-1, i] + 1 }; map i { b[i] = a[k, i] } }``: the first map writes
    through a whole-array view whose binding edge carries the whole array."""
    sdfg = dace.SDFG(name)
    sdfg.add_array("a", (7, 13), dace.float64)
    sdfg.add_array("b", (13,), dace.float64)
    sdfg.add_view("a_view", (7, 13), dace.float64)
    loop = LoopRegion("kloop", "k < 7", "k", "k = 1", "k = k + 1")
    sdfg.add_node(loop, is_start_block=True)
    first = loop.add_state("step", is_start_block=True)
    entry, exit_ = first.add_map("step", {"i": "0:13"})
    tasklet = first.add_tasklet("add", {"x"}, {"y"}, "y = x + 1.0")
    view = first.add_access("a_view")
    first.add_memlet_path(first.add_read("a"), entry, tasklet, dst_conn="x", memlet=dace.Memlet("a[k - 1, i]"))
    first.add_edge(tasklet, "y", view, None, dace.Memlet("a_view[k, i]"))
    first.add_memlet_path(view, exit_, first.add_write("a"), memlet=dace.Memlet("a[0:7, 0:13]"))
    second = loop.add_state("store")
    loop.add_edge(first, second, dace.InterstateEdge())
    second.add_mapped_tasklet(
        "store", {"i": "0:13"}, {"x": dace.Memlet("a[k, i]")}, "y = x", {"y": dace.Memlet("b[i]")}, external_edges=True
    )
    sdfg.validate()
    return sdfg, loop


def test_a_lane_write_through_a_whole_array_view_counts_at_the_view_index():
    """The view's binding edge carries the whole array; the lane index is on the write into the view."""
    sdfg, loop = viewed_write_loop("viewed_lane_write")
    assert analyze_lanes(loop, sdfg).refusal is None
    rng = np.random.default_rng(3)
    want = dict(a=rng.random((7, 13)), b=np.zeros(13))
    got = copy.deepcopy(want)
    copy.deepcopy(sdfg)(**want)
    MoveLoopIntoMap.apply_to(sdfg, options={"cfg_body": True}, loop=loop)
    assert not any(isinstance(b, LoopRegion) for b in sdfg.nodes())
    sdfg(**got)
    for name in ("a", "b"):
        np.testing.assert_allclose(got[name], want[name], rtol=1e-13, atol=0, err_msg=name)


def test_a_dependence_between_lanes_refuses_the_interchange():
    sdfg, loop = lane_loop(lane_shift)
    before = graph_digest(sdfg)
    assert analyze_lanes(loop, sdfg).refusal == "a is accessed across lanes"
    assert sdfg.apply_transformations(MoveLoopIntoMap, options={"cfg_body": True}) == 0
    assert graph_digest(sdfg) == before


def test_a_branch_reading_one_lanes_result_refuses_the_interchange():
    sdfg, loop = lane_loop(lane_guard)
    assert analyze_lanes(loop, sdfg).refusal == "lane-independent code reads a, which the maps write per lane"
    assert sdfg.apply_transformations(MoveLoopIntoMap, options={"cfg_body": True}) == 0


def test_a_branch_condition_naming_lane_data_refuses_the_interchange():
    """The condition reads ``a`` by name, with no access node anywhere to show the read."""
    sdfg = dace.SDFG("lane_condition_read")
    sdfg.add_array("a", (7, 13), dace.float64)
    sdfg.add_array("b", (13,), dace.float64)
    loop = LoopRegion("kloop", "k < 7", "k", "k = 1", "k = k + 1")
    sdfg.add_node(loop, is_start_block=True)
    first = loop.add_state("shift", is_start_block=True)
    first.add_mapped_tasklet(
        "shift",
        dict(i="0:13"),
        dict(__in=dace.Memlet("a[k - 1, i]")),
        "__out = __in + 1.0",
        dict(__out=dace.Memlet("a[k, i]")),
        external_edges=True,
    )
    guard = ConditionalBlock("guard", sdfg, loop)
    loop.add_node(guard)
    loop.add_edge(first, guard, dace.InterstateEdge())
    body = ControlFlowRegion("guard_body", sdfg, guard)
    body.add_state("halve", is_start_block=True).add_mapped_tasklet(
        "halve",
        dict(i="0:13"),
        dict(__in=dace.Memlet("b[i]")),
        "__out = __in * 0.5",
        dict(__out=dace.Memlet("b[i]")),
        external_edges=True,
    )
    guard.add_branch(CodeBlock("a[k, 0] > 0.5"), body)
    sdfg.validate()
    assert analyze_lanes(loop, sdfg).refusal == "lane-independent code reads a, which the maps write per lane"


def test_cfg_body_leaves_the_single_map_interchange_byte_identical():
    classic, generalized = forward_loop.to_sdfg(simplify=True), forward_loop.to_sdfg(simplify=True)
    assert classic.apply_transformations(MoveLoopIntoMap) == 1
    assert generalized.apply_transformations(MoveLoopIntoMap, options={"cfg_body": True}) == 1
    assert graph_digest(generalized) == graph_digest(classic)


def test_the_cpu_interchange_gate_leaves_a_branching_body_alone():
    """The CPU gate only ever sees the single-map shape; the generalized body is a GPU decision."""
    sdfg, loop = lane_loop(branching_column)
    before = graph_digest(sdfg)
    assert MoveLoopIntoMapGated(target="cpu").apply_pass(sdfg, {}) is None
    assert graph_digest(sdfg) == before


def test_an_update_ahead_of_the_map_keeps_the_loop_outside():
    sdfg = dace.SDFG("update_ahead_of_the_map")
    sdfg.add_array("x", [4], dace.float64)
    sdfg.add_array("y", [4], dace.float64)
    loop = LoopRegion("outer", "t < 2", "t", "t = 0", "t = t + 1")
    sdfg.add_node(loop, is_start_block=True)
    body = loop.add_state("body", is_start_block=True)
    update = body.add_tasklet("update", {"a"}, {"b"}, "b = a + 1")
    updated = body.add_access("x")
    body.add_edge(body.add_read("x"), None, update, "a", dace.Memlet("x[0]"))
    body.add_edge(update, "b", updated, None, dace.Memlet("x[0]"))
    body.add_mapped_tasklet(
        "reader",
        {"i": "0:4"},
        {"a": dace.Memlet("x[i]")},
        "b = a",
        {"b": dace.Memlet("y[i]")},
        input_nodes={"x": updated},
        external_edges=True,
    )

    assert sdfg.apply_transformations_repeated(MoveLoopIntoMap) == 0


@dace.program
def plane_column_sweep(q: dace.float64[I + 2, J, K], f: dace.float64[I + 2, J, K]):
    """Steps over stencils along ``i`` only, each over its own ``i`` range: the plane columns never interact."""
    for t in range(N):
        for i, j, k in dace.map[1 : I + 2, 0:J, 0:K]:
            f[i, j, k] = q[i - 1, j, k] - q[i, j, k]
        for i, j, k in dace.map[1 : I + 1, 0:J, 0:K]:
            q[i, j, k] = q[i, j, k] + 0.5 * (f[i + 1, j, k] - f[i, j, k])


def test_shared_lane_axes_with_a_tile_sweep_a_tile_of_columns_per_lane():
    """The lanes are every axis the maps share, ``(j, k)``; with ``lane_tile`` each lane steps four ``k`` and sweeps
    them inside the loop, the last tile stopping at ``K``."""
    sdfg = plane_column_sweep.to_sdfg(simplify=True)
    loop = next(b for b in sdfg.nodes() if isinstance(b, LoopRegion))
    options = {"cfg_body": True, "lane_axis": True, "shared_axes": True, "lane_tile": 4}
    assert MoveLoopIntoMap.can_be_applied_to(sdfg, options=options, loop=loop)
    MoveLoopIntoMap.apply_to(sdfg, options=options, loop=loop)
    sdfg.validate()
    (lanes,) = [n for n in sdfg.start_block.nodes() if isinstance(n, dace.nodes.MapEntry)]
    assert str(lanes.map.range) == "0:J, 0:K:4"
    inner = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.MapEntry) and n is not lanes]
    assert sorted(str(e.map.range) for e in inner) == [
        "1:I + 1, j_lane, k_lane:Min(K - 1, k_lane + 3) + 1",
        "1:I + 2, j_lane, k_lane:Min(K - 1, k_lane + 3) + 1",
    ]
    rng = np.random.default_rng(5)
    q, f = rng.random((9, 3, 6)), np.zeros((9, 3, 6))
    ref_q, ref_f = q.copy(), f.copy()
    for _ in range(3):
        ref_f[1:] = ref_q[:-1] - ref_q[1:]
        ref_q[1:-1] = ref_q[1:-1] + 0.5 * (ref_f[2:] - ref_f[1:-1])
    sdfg(q=q, f=f, I=7, J=3, K=6, N=3)
    assert np.allclose(q, ref_q) and np.allclose(f, ref_f)


if __name__ == "__main__":
    test_forward_loops_semantic_eq()
    test_backward_loops_semantic_eq()
    test_multiple_edges()
    test_itervar_in_map_range()
    test_itervar_in_data()
    test_non_injective_index()
    test_scaled_index_is_injective_for_a_nonzero_multiplier()
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
    test_lane_accesses_behind_two_nested_sdfgs_are_read_at_the_innermost_memlet()
    test_a_lane_write_through_a_whole_array_view_counts_at_the_view_index()
    test_a_dependence_between_lanes_refuses_the_interchange()
    test_a_branch_reading_one_lanes_result_refuses_the_interchange()
    test_a_branch_condition_naming_lane_data_refuses_the_interchange()
    test_cfg_body_leaves_the_single_map_interchange_byte_identical()
    test_the_cpu_interchange_gate_leaves_a_branching_body_alone()
    test_loop_bound_symbols_are_declared_in_the_nested_sdfg()
    test_an_update_ahead_of_the_map_keeps_the_loop_outside()
    test_shared_lane_axes_with_a_tile_sweep_a_tile_of_columns_per_lane()


def window_scan(name: str, window: str) -> tuple[dace.SDFG, LoopRegion]:
    """``for k in 1:7: map i: a[k, i] = 0.5 * a[k - 1, i] + a[k, i]``, the body a nested SDFG whose read connector is
    the one memlet ``window`` (``a[k - 1:k + 1, i]`` for the scan shape, as a fused scan body gets it)."""
    inner = dace.SDFG(f"{name}_body")
    inner.add_array("a", (7, 13), dace.float64)
    inner.add_symbol("i", dace.int64)
    inner.add_symbol("k", dace.int64)
    istate = inner.add_state("level", is_start_block=True)
    tasklet = istate.add_tasklet("level", {"prev", "cur"}, {"y"}, "y = 0.5 * prev + cur")
    read = istate.add_read("a")
    istate.add_edge(read, None, tasklet, "prev", dace.Memlet("a[k - 1, i]"))
    istate.add_edge(read, None, tasklet, "cur", dace.Memlet("a[k, i]"))
    istate.add_edge(tasklet, "y", istate.add_write("a"), None, dace.Memlet("a[k, i]"))
    sdfg = dace.SDFG(name)
    sdfg.add_array("a", (7, 13), dace.float64)
    loop = LoopRegion("kloop", "k < 7", "k", "k = 1", "k = k + 1")
    sdfg.add_node(loop, is_start_block=True)
    state = loop.add_state("scan", is_start_block=True)
    entry, exit_ = state.add_map("scan", {"i": "0:13"})
    node = state.add_nested_sdfg(inner, {"a"}, {"a"}, {"i": "i", "k": "k"})
    state.add_memlet_path(state.add_read("a"), entry, node, dst_conn="a", memlet=dace.Memlet(window))
    state.add_memlet_path(node, exit_, state.add_write("a"), src_conn="a", memlet=dace.Memlet("a[k, i]"))
    sdfg.validate()
    return sdfg, loop


def test_a_read_window_that_slides_with_the_loop_moves_into_the_map():
    """The scan reads ``a[k - 1:k + 1, i]`` through one memlet and writes ``a[k, i]``: the window is not an index, but
    it starts where the loop variable says and keeps its width, so the loop moves into the map per lane."""
    sdfg, loop = window_scan("sliding_window_scan", "a[k - 1:k + 1, i]")
    assert MoveLoopIntoMap.can_be_applied_to(sdfg, loop=loop)
    rng = np.random.default_rng(3)
    want = dict(a=rng.random((7, 13)))
    got = copy.deepcopy(want)
    copy.deepcopy(sdfg)(**want)
    MoveLoopIntoMap.apply_to(sdfg, loop=loop)
    assert not any(isinstance(b, LoopRegion) for b in sdfg.nodes())
    sdfg(**got)
    assert np.allclose(got["a"], want["a"])


def test_a_read_window_that_grows_with_the_loop_is_refused():
    """``a[0:k + 1, i]`` widens with ``k``: it is no window, so the index rule still refuses the interchange."""
    sdfg, loop = window_scan("growing_window_scan", "a[0:k + 1, i]")
    assert not MoveLoopIntoMap.can_be_applied_to(sdfg, loop=loop)
