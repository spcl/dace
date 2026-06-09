# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Integration tests: ``StageInsideBody`` followed by ``ConvertTaskletsToTileOps``.

Per user direction 2026-06-09: "we should be able to extend tile accesses
and then replace all tasklets with the new libnodes." This file pins the
contract that the two passes compose end-to-end:

1. ``StageInsideBody`` walks every tile-tagged body NSDFG and stages every
   non-transient AccessNode read through a tile (or Scalar) transient
   bridge, rewiring downstream consumers to read from the bridge.

2. ``ConvertTaskletsToTileOps`` then walks the same bodies and replaces
   every recognised tasklet (binary, unary, ternary, in-place RMW
   reduction) with the corresponding tile lib node (``TileBinop`` /
   ``TileUnop`` / ``TileITE`` / ``TileReduce``).

After both passes run, the body NSDFG should have no raw tasklets
remaining for the recognised shapes -- only ``Tile*`` lib nodes between
the bridge AccessNodes.
"""
import dace
from dace.libraries.tileops import TileBinop, TileReduce, TileUnop
from dace.memlet import Memlet
from dace.transformation.passes.vectorization.convert_tasklets_to_tile_ops import (ConvertTaskletsToTileOps)
from dace.transformation.passes.vectorization.stage_inside_body import StageInsideBody


def _build_binop_kernel():
    """Body NSDFG: ``out_t = A[ii] + B[ii]``."""
    sdfg = dace.SDFG("compose_binop")
    sdfg.add_array("A", (8, ), dace.float64, transient=False)
    sdfg.add_array("B", (8, ), dace.float64, transient=False)
    state = sdfg.add_state("s")
    me, mx = state.add_map("k", {"ii": "0:8"})

    inner = dace.SDFG("body")
    inner.add_array("A", (8, ), dace.float64, transient=False)
    inner.add_array("B", (8, ), dace.float64, transient=False)
    inner.add_array("out_t", (1, ), dace.float64, transient=True)
    instate = inner.add_state("body")
    a_inner = instate.add_access("A")
    b_inner = instate.add_access("B")
    t_inner = instate.add_access("out_t")
    tasklet = instate.add_tasklet("body_t", {"_a", "_b"}, {"_o"}, "_o = _a + _b")
    instate.add_edge(a_inner, None, tasklet, "_a", Memlet("A[ii]"))
    instate.add_edge(b_inner, None, tasklet, "_b", Memlet("B[ii]"))
    instate.add_edge(tasklet, "_o", t_inner, None, Memlet("out_t[0]"))

    nsdfg = state.add_nested_sdfg(inner, {"A", "B"}, set(), symbol_mapping={"ii": "ii"})
    a_outer = state.add_access("A")
    b_outer = state.add_access("B")
    state.add_memlet_path(a_outer, me, nsdfg, dst_conn="A", memlet=Memlet("A[0:8]"))
    state.add_memlet_path(b_outer, me, nsdfg, dst_conn="B", memlet=Memlet("B[0:8]"))
    state.add_nedge(nsdfg, mx, Memlet())
    return sdfg, inner


def _build_unop_kernel():
    """Body NSDFG: ``out_t = abs(A[ii])``."""
    sdfg = dace.SDFG("compose_unop")
    sdfg.add_array("A", (8, ), dace.float64, transient=False)
    state = sdfg.add_state("s")
    me, mx = state.add_map("k", {"ii": "0:8"})

    inner = dace.SDFG("body")
    inner.add_array("A", (8, ), dace.float64, transient=False)
    inner.add_array("out_t", (1, ), dace.float64, transient=True)
    instate = inner.add_state("body")
    a_inner = instate.add_access("A")
    t_inner = instate.add_access("out_t")
    tasklet = instate.add_tasklet("body_t", {"_a"}, {"_o"}, "_o = math.abs(_a)")
    instate.add_edge(a_inner, None, tasklet, "_a", Memlet("A[ii]"))
    instate.add_edge(tasklet, "_o", t_inner, None, Memlet("out_t[0]"))

    nsdfg = state.add_nested_sdfg(inner, {"A"}, set(), symbol_mapping={"ii": "ii"})
    a_outer = state.add_access("A")
    state.add_memlet_path(a_outer, me, nsdfg, dst_conn="A", memlet=Memlet("A[0:8]"))
    state.add_nedge(nsdfg, mx, Memlet())
    return sdfg, inner


def _build_reduction_kernel():
    """Body NSDFG: ``Acc[0] = Acc[0] + A[ii]``."""
    sdfg = dace.SDFG("compose_reduction")
    sdfg.add_array("A", (8, ), dace.float64, transient=False)
    sdfg.add_array("Acc", (1, ), dace.float64, transient=False)
    state = sdfg.add_state("s")
    me, mx = state.add_map("k", {"ii": "0:8"})

    inner = dace.SDFG("body")
    inner.add_array("A", (8, ), dace.float64, transient=False)
    inner.add_array("Acc", (1, ), dace.float64, transient=False)
    instate = inner.add_state("body")
    a_inner = instate.add_access("A")
    acc_inner_in = instate.add_access("Acc")
    acc_inner_out = instate.add_access("Acc")
    tasklet = instate.add_tasklet("body_t", {"_acc", "_val"}, {"_acc"}, "_acc = _acc + _val")
    instate.add_edge(a_inner, None, tasklet, "_val", Memlet("A[ii]"))
    instate.add_edge(acc_inner_in, None, tasklet, "_acc", Memlet("Acc[0]"))
    instate.add_edge(tasklet, "_acc", acc_inner_out, None, Memlet("Acc[0]"))

    nsdfg = state.add_nested_sdfg(inner, {"A", "Acc"}, {"Acc"}, symbol_mapping={"ii": "ii"})
    a_outer = state.add_access("A")
    acc_outer_in = state.add_access("Acc")
    acc_outer_out = state.add_access("Acc")
    state.add_memlet_path(a_outer, me, nsdfg, dst_conn="A", memlet=Memlet("A[0:8]"))
    state.add_memlet_path(acc_outer_in, me, nsdfg, dst_conn="Acc", memlet=Memlet("Acc[0]"))
    state.add_memlet_path(nsdfg, mx, acc_outer_out, src_conn="Acc", memlet=Memlet("Acc[0]"))
    return sdfg, inner


def _body_state(inner):
    return next(s for s in inner.states())


def _count(state, cls):
    return sum(1 for n in state.nodes() if isinstance(n, cls))


def test_walker_then_converter_binop_kernel():
    """Compose StageInsideBody + ConvertTaskletsToTileOps on a binop kernel.

    After both passes: the body has a TileBinop reading from tile-shape bridges,
    and the original raw tasklet is gone.
    """
    sdfg, inner = _build_binop_kernel()
    StageInsideBody(widths=(8, )).apply_pass(sdfg, {})
    ConvertTaskletsToTileOps(widths=(8, )).apply_pass(sdfg, {})
    body = _body_state(inner)
    assert _count(body, dace.nodes.Tasklet) == 0, "expected the raw tasklet to be gone"
    assert _count(body, TileBinop) == 1, "expected exactly one TileBinop"


def test_walker_then_converter_unop_kernel():
    """Compose passes on a unary kernel -> TileUnop, no raw tasklets."""
    sdfg, inner = _build_unop_kernel()
    StageInsideBody(widths=(8, )).apply_pass(sdfg, {})
    ConvertTaskletsToTileOps(widths=(8, )).apply_pass(sdfg, {})
    body = _body_state(inner)
    assert _count(body, dace.nodes.Tasklet) == 0
    assert _count(body, TileUnop) == 1


def test_walker_then_converter_reduction_kernel():
    """Compose passes on an in-place RMW reduction -> TileReduce, no raw tasklets."""
    sdfg, inner = _build_reduction_kernel()
    StageInsideBody(widths=(8, )).apply_pass(sdfg, {})
    ConvertTaskletsToTileOps(widths=(8, )).apply_pass(sdfg, {})
    body = _body_state(inner)
    assert _count(body, dace.nodes.Tasklet) == 0
    assert _count(body, TileReduce) == 1


def test_walker_extends_then_converter_replaces_all_recognised_tasklets():
    """The composed pipeline never leaves a raw recognised-shape tasklet in the body.

    Probes the post-pipeline invariant: for every body NSDFG produced by the
    walker, every tasklet with a recognised body shape (binary / unary /
    ternary / in-place RMW) is replaced. Unrecognised shapes are left intact;
    this test covers the recognised classes.
    """
    for builder, expected_node_type in (
        (_build_binop_kernel, TileBinop),
        (_build_unop_kernel, TileUnop),
        (_build_reduction_kernel, TileReduce),
    ):
        sdfg, inner = builder()
        StageInsideBody(widths=(8, )).apply_pass(sdfg, {})
        ConvertTaskletsToTileOps(widths=(8, )).apply_pass(sdfg, {})
        body = _body_state(inner)
        assert _count(body, dace.nodes.Tasklet) == 0, (f"{builder.__name__}: expected no raw tasklets after "
                                                       f"the composed walker+converter pipeline")
        assert _count(
            body,
            expected_node_type) >= 1, (f"{builder.__name__}: expected a {expected_node_type.__name__} after conversion")
