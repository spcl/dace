# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The vectorizer tiles a map whose body loops only in step over the lanes.

A sequential loop whose trip count and symbols are the same for every lane stays scalar control
flow in the tile body, and the body is widened per lane: a container the loop carries becomes a
tile, one carry per lane (polybench adi's per-row sweeps, deriche's IIR passes over rows). A loop
bounded by the tiled param (lu's triangular update) or whose head or symbols read data (pointer
chasing) has a per-lane trip or symbol no scalar can hold, so ``map_body_has_lane_varying_loop``
refuses it.

A CONDITIONAL is a different story and is deliberately still allowed: a branch lowers by MASKING
(evaluate both sides, select per lane), which is what the tile mask machinery is for. Refusing
conditionals here would cost 18 TSVC kernels (s271, s441, vif, ...) their vectorization for no
soundness gain -- the genuinely unmaskable ones are refused precisely by
``map_body_has_tiled_param_dependent_branch``.
"""

import copy

import pytest

import dace
import tests.corpus.measure_parallelization as mp
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion, LoopRegion
from dace.transformation.passes.vectorization.utils.map_predicates import (
    is_vectorizable_map,
    map_body_has_lane_varying_loop,
    map_body_has_param_dependent_loop,
    map_body_has_tiled_param_dependent_branch,
)
from tests.passes.vectorization.tile_assertions import TILE_NODE_TYPES

N = 16


def _map_over_body(inner: dace.SDFG, arrays):
    """Wrap ``inner`` in a ``row`` map reading and writing each of ``arrays``."""
    sdfg = dace.SDFG("wrapper")
    for a in arrays:
        sdfg.add_array(a, [N, N], dace.float64)
    st = sdfg.add_state("main", is_start_block=True)
    me, mx = st.add_map("row", dict(i=f"0:{N}"))
    # Every symbol the body declares rides in under its own name. ``i`` is the map param -- the one
    # striding rebinds to the tile base; anything else is a free symbol of the wrapper and therefore
    # LANE-UNIFORM inside the tile.
    for sym, stype in inner.symbols.items():
        if sym != "i":
            sdfg.add_symbol(sym, stype)
    ns = st.add_nested_sdfg(inner, set(arrays), set(arrays), symbol_mapping={s: s for s in inner.symbols})
    for a in arrays:
        st.add_memlet_path(st.add_access(a), me, ns, dst_conn=a, memlet=dace.Memlet(f"{a}[0:{N}, 0:{N}]"))
        st.add_memlet_path(ns, mx, st.add_access(a), src_conn=a, memlet=dace.Memlet(f"{a}[0:{N}, 0:{N}]"))
    return sdfg, st, me


def _elementwise_state(container, arr: str, label: str, index: str):
    """``arr[i, <index>] = arr[i, <index>] * 2`` as one state of ``container``."""
    body = container.add_state(label, is_start_block=True)
    t = body.add_tasklet(f"{label}_t", {"cur"}, {"out"}, "out = cur * 2")
    body.add_edge(body.add_access(arr), None, t, "cur", dace.Memlet(f"{arr}[i, {index}]"))
    body.add_edge(t, "out", body.add_access(arr), None, dace.Memlet(f"{arr}[i, {index}]"))
    return body


def _flat_body():
    """A body NSDFG that is pure dataflow -- the tileable shape."""
    inner = dace.SDFG("flat_flat_body")
    inner.add_symbol("i", dace.int64)
    inner.add_array("p", [N, N], dace.float64)
    _elementwise_state(inner, "p", "body", "0")
    return inner


def _loop_body(arrays, bound: str = f"j < {N}", chase: bool = False):
    """A body NSDFG that sweeps each of ``arrays`` along j in a sequential loop under ``bound``; with ``chase``,
    the sweep reads row ``k`` and an assignment inside the loop reads the next ``k`` from the data."""
    inner = dace.SDFG("loops")
    inner.add_symbol("i", dace.int64)
    inner.add_symbol("k", dace.int64)
    for a in arrays:
        inner.add_array(a, [N, N], dace.float64)
    prev = None
    for a in arrays:
        loop = LoopRegion(f"sweep_{a}", bound, "j", "j = 1", "j = j + 1", sdfg=inner)
        inner.add_node(loop)
        body = loop.add_state("body", is_start_block=True)
        r, w = body.add_access(a), body.add_access(a)
        t = body.add_tasklet(f"{a}_t", {"cur", "prev"}, {"out"}, "out = cur + prev")
        body.add_edge(r, None, t, "cur", dace.Memlet(f"{a}[i, j]"))
        body.add_edge(r, None, t, "prev", dace.Memlet(f"{a}[i, {'k' if chase else 'j - 1'}]"))
        body.add_edge(t, "out", w, None, dace.Memlet(f"{a}[i, j]"))
        if chase:
            step = loop.add_state("step")
            loop.add_edge(body, step, dace.InterstateEdge(assignments={"k": f"int({a}[i, j])"}))
        if prev is None:
            inner.start_block = inner.node_id(loop)
        else:
            inner.add_edge(prev, loop, dace.InterstateEdge())
        prev = loop
    return inner


def _conditional_body(guard: str = "lim > 0"):
    """A body NSDFG whose work sits under a ``ConditionalBlock`` guard.

    The default guard is over ``lim``, a symbol the tile never strides, so it holds the SAME answer
    for all W lanes -- the maskable shape this gate must let through. Pass ``'i > 0'`` for the
    opposite case: a guard over the param striding rebinds to the tile base.
    """
    inner = dace.SDFG("guarded_conditional_body")
    inner.add_symbol("i", dace.int64)
    inner.add_symbol("lim", dace.int64)
    inner.add_array("p", [N, N], dace.float64)
    cond = ConditionalBlock("guard", sdfg=inner)
    inner.add_node(cond, is_start_block=True)
    branch = ControlFlowRegion("then", sdfg=inner)
    cond.add_branch(dace.properties.CodeBlock(guard), branch)
    _elementwise_state(branch, "p", "then_body", "0")
    return inner


def test_flat_body_is_accepted():
    """The gate is narrow: a pure-dataflow body stays vectorizable."""
    sdfg, state, map_entry = _map_over_body(_flat_body(), ["p"])
    assert map_body_has_lane_varying_loop(state, map_entry) is False
    assert is_vectorizable_map(state, map_entry, 1) is True


@pytest.mark.parametrize("arrays", [["p"], ["p", "q"], ["p", "q", "v"]])
def test_a_lane_uniform_loop_body_is_accepted(arrays):
    """A sweep along j carries ``p[i, j - 1]`` in each lane's own row: one carried sweep as much as three (adi)."""
    sdfg, state, map_entry = _map_over_body(_loop_body(arrays), arrays)
    assert map_body_has_lane_varying_loop(state, map_entry) is False
    assert is_vectorizable_map(state, map_entry, 1) is True


def test_a_loop_that_reads_its_next_symbol_from_the_data_is_refused():
    """``k = p[i, j]`` gives every lane its own ``k``, which one scalar symbol cannot hold."""
    sdfg, state, map_entry = _map_over_body(_loop_body(["p"], chase=True), ["p"])
    assert map_body_has_lane_varying_loop(state, map_entry) is True
    assert is_vectorizable_map(state, map_entry, 1) is False


def test_a_loop_bounded_by_the_tiled_param_is_refused():
    """``j < i`` is a per-lane trip count (lu's triangular update), which both loop gates see."""
    sdfg, state, map_entry = _map_over_body(_loop_body(["p"], bound="j < i"), ["p"])
    assert map_body_has_lane_varying_loop(state, map_entry) is True
    assert map_body_has_param_dependent_loop(state, map_entry) is True
    assert is_vectorizable_map(state, map_entry, 1) is False


def test_conditional_body_is_still_allowed():
    """A lane-uniform guard is MASKABLE, so this gate must let it through -- refusing every
    conditional would strip 18 TSVC kernels of their vectorization for no soundness gain."""
    sdfg, state, map_entry = _map_over_body(_conditional_body(), ["p"])
    assert map_body_has_lane_varying_loop(state, map_entry) is False
    assert is_vectorizable_map(state, map_entry, 1) is True


def test_conditional_over_the_tiled_param_is_refused_by_the_branch_gate():
    """The complement of the test above, so "conditionals are allowed" cannot quietly widen into
    "every conditional is allowed": a guard over the param about to be STRIDED is evaluated once per
    tile at the tile base, so lane 0 decides for all W lanes. That one is refused -- and refused by
    the branch gate, not the loop gate."""
    sdfg, state, map_entry = _map_over_body(_conditional_body("i > 0"), ["p"])
    assert map_body_has_lane_varying_loop(state, map_entry) is False
    assert map_body_has_tiled_param_dependent_branch(state, map_entry, ("i",)) is True
    assert is_vectorizable_map(state, map_entry, 1) is False


#: Per kernel, its maps after canon+vec as ``(label, schedule, step of the last dim, holds tile nodes)``, and the
#: variables of the sequential loops that survive. adi's and deriche's sweeps carry a refused map scalar
#: (``__no_vectorize``); lu's lane-uniform reductions inside its sweeps tile by 8 with a remainder map each.
REAL_KERNEL_SHAPES = {
    "adi": (
        [
            ("single_state_body_0_map__no_vectorize", "CPU_Multicore", 1, False),
            ("single_state_body_map__no_vectorize", "CPU_Multicore", 1, False),
        ],
        ["_loop_it_0", "_loop_it_2", "_loop_it_3", "_loop_it_5", "_loop_it_6"],
    ),
    "deriche": (
        [
            ("comp_iout", "CPU_Multicore", 8, True),
            ("comp_iout2", "CPU_Multicore", 8, True),
            ("comp_iout2__tile_main", "CPU_Multicore", 8, True),
            ("comp_iout__tile_main", "CPU_Multicore", 8, True),
            ("single_state_body_0_map__no_vectorize", "CPU_Multicore", 1, False),
            ("single_state_body_1_map__no_vectorize", "CPU_Multicore", 1, False),
        ],
        ["_loop_it_1", "_loop_it_3", "_loop_it_5", "_loop_it_7"],
    ),
    "lu": (
        [
            ("k_loop1", "Sequential", 8, True),
            ("k_loop1__tile_main", "Sequential", 8, True),
            ("k_loop2", "Sequential", 8, True),
            ("k_loop2__tile_main", "Sequential", 8, True),
            ("single_state_body_map", "CPU_Multicore", 1, True),
        ],
        ["_loop_it_0", "_loop_it_1", "_loop_it_2", "_skew_t_0"],
    ),
}


def map_shapes(sdfg: dace.SDFG) -> list[tuple[str, str, object, bool]]:
    """Every map of ``sdfg`` as ``(label, schedule, step of its last dim, whether its body holds tile nodes)``."""
    shapes = []
    for entry, state in sdfg.all_nodes_recursive():
        if not isinstance(entry, dace.nodes.MapEntry):
            continue
        bodies = [n.sdfg for n in state.scope_subgraph(entry).nodes() if isinstance(n, dace.nodes.NestedSDFG)]
        tiled = any(isinstance(n, TILE_NODE_TYPES) for body in bodies for n, _ in body.all_nodes_recursive())
        shapes.append((entry.map.label, entry.map.schedule.name, entry.map.range[-1][2], tiled))
    return sorted(shapes)


@pytest.mark.parametrize("name", ["adi", "deriche", "lu"])
def test_real_kernels_tile_exactly_their_lane_uniform_maps_and_stay_correct(name):
    """End-to-end on the polybench kernels whose maps run sequential sweeps: the maps after canon+vec are exactly
    the expected ones (a sweep's map refused and scalar, a lane-uniform map tiled with its remainder), every sweep
    stays a loop, and every kernel stays value-correct."""
    from dace.transformation.passes.canonicalize.finalize import finalize_for_target

    base, checker = mp.CORPORA["poly"][1](name)
    sd = copy.deepcopy(base)
    mp.apply_config(sd, "canon+vec", mp.cpu_params(4))

    maps, loops = REAL_KERNEL_SHAPES[name]
    assert map_shapes(sd) == maps
    assert (
        sorted(r.loop_variable for r in sd.all_control_flow_regions(recursive=True) if isinstance(r, LoopRegion))
        == loops
    )
    fin = finalize_for_target(copy.deepcopy(sd), "cpu")
    fin.name = f"{name}_inner_loop_gate_test"
    assert bool(checker(fin)), f"{name} must be value-correct"


if __name__ == "__main__":
    test_flat_body_is_accepted()
    for arrays in (["p"], ["p", "q"], ["p", "q", "v"]):
        test_a_lane_uniform_loop_body_is_accepted(arrays)
    test_a_loop_that_reads_its_next_symbol_from_the_data_is_refused()
    test_a_loop_bounded_by_the_tiled_param_is_refused()
    test_conditional_body_is_still_allowed()
    test_conditional_over_the_tiled_param_is_refused_by_the_branch_gate()
    for name in ("adi", "deriche", "lu"):
        test_real_kernels_tile_exactly_their_lane_uniform_maps_and_stay_correct(name)
