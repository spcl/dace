# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Overlapping boundary writes must keep their happens-before across a scope nesting.

``u[:, 0] = 0; u[:, -1] = 0; u[-1, :] = 1`` (npbench ``cavity_flow``'s lid-driven boundary) overlaps
at the two corners, so the lid assignment is only correct AFTER the column zeroing. Canonicalization
records that as an empty memlet -- an ordering edge -- from the access node the column map writes to
the lid map's entry.

The column map writes ``u`` TWICE, and :func:`~dace.transformation.helpers.nest_state_subgraph` gives
the nested SDFG one connector per data name, so the second write folds onto the first and its outer
edge is dropped. Dropping the transfer alone left the access node it fed producer-less: the ordering
edge survived, anchored on nothing, and the corners came out zero because execution order fell back
to node insertion order. These tests pin the ordering both right at the nesting and end to end
through canonicalize + vectorize.
"""
import copy

import numpy as np
import pytest

import dace
from dace.libraries.tileops._dispatch import detect_host_isa
from dace.sdfg import nodes as nd
from dace.transformation.dataflow.map_for_loop import MapToForLoop
from dace.transformation.passes.canonicalize import canonicalize
from dace.transformation.passes.vectorization.config import VectorizeConfig
from dace.transformation.passes.vectorization.vectorize_cpu_multi_dim import VectorizeCPUMultiDim

N = dace.symbol("N", dtype=dace.int64)
_SIZE = 16


@dace.program
def _boundary(u: dace.float64[N, N]):
    u[0, :] = 0.0
    u[:, 0] = 0.0
    u[:, -1] = 0.0
    u[-1, :] = 1.0


def _reference():
    u = np.full((_SIZE, _SIZE), 7.0, dtype=np.float64)
    u[0, :] = 0.0
    u[:, 0] = 0.0
    u[:, -1] = 0.0
    u[-1, :] = 1.0
    return u


def _point(rng, dim):
    """``True`` iff dimension ``dim`` of ``rng`` is a single index, not a span."""
    begin, end, _ = rng[dim]
    return str(begin) == str(end)


def _classify(memlet):
    """Which boundary region of ``u`` a write memlet covers, if any."""
    rng = memlet.subset
    if rng is None or rng.dims() != 2:
        return None
    if _point(rng, 0) and "N - 1" in str(rng[0][0]):
        return "lid"
    if _point(rng, 1) and str(rng[1][0]) in ("0", "N - 1"):
        return "column"
    return None


def _written_regions(state, node):
    """Boundary regions written by ``node``, looking inside a nested SDFG."""
    regions = set()
    if isinstance(node, nd.NestedSDFG):
        for nested in node.sdfg.all_sdfgs_recursive():
            for nstate in nested.all_states():
                for edge in nstate.edges():
                    if edge.data.is_empty() or not isinstance(edge.dst, nd.AccessNode):
                        continue
                    regions.add(_classify(edge.data))
        return regions - {None}
    for edge in state.out_edges(node):
        if edge.data.is_empty() or not isinstance(edge.dst, nd.AccessNode):
            continue
        regions.add(_classify(edge.data))
    return regions - {None}


def _reaches(state, src, dst):
    seen, work = {src}, [src]
    while work:
        node = work.pop()
        if node is dst:
            return True
        for edge in state.out_edges(node):
            if edge.dst not in seen:
                seen.add(edge.dst)
                work.append(edge.dst)
    return False


def _ordering_target(state, node):
    """Where an ordering edge into ``node``'s unit lands: a map scope is entered at its entry."""
    return state.entry_node(node) if isinstance(node, nd.MapExit) else node


def _boundary_units(sdfg):
    """``(state, column writers, lid writers)`` for the one state holding both."""
    for nested in sdfg.all_sdfgs_recursive():
        for state in nested.all_states():
            columns, lids = [], []
            for node in state.nodes():
                regions = _written_regions(state, node)
                if "column" in regions:
                    columns.append(node)
                if "lid" in regions:
                    lids.append(node)
            if columns and lids:
                return state, columns, lids
    return None, [], []


def _canonical():
    sdfg = _boundary.to_sdfg(simplify=False)
    canonicalize(sdfg, validate=True)
    return sdfg


def _assert_lid_ordered_after_columns(sdfg, phase):
    state, columns, lids = _boundary_units(sdfg)
    assert state is not None, f"{phase}: no state writes both the columns and the lid"
    for lid in lids:
        target = _ordering_target(state, lid)
        ordering = [e for e in state.in_edges(target) if e.data.is_empty()]
        assert ordering, (f"{phase}: the lid scope {lid} lost its ordering edge; nothing keeps it "
                          f"after the column zeroing it overlaps at the corners")
        for column in columns:
            if column is lid:
                continue
            assert _reaches(
                state, column,
                target), (f"{phase}: no happens-before from the column scope {column} to the lid scope {lid}. "
                          f"The ordering edges into the lid start at {[e.src for e in ordering]}, which the "
                          f"column scope no longer reaches, so the corners race.")


def test_canonicalized_boundary_orders_lid_after_columns():
    """The premise: canon really does order the lid behind the columns, via an empty memlet."""
    sdfg = _canonical()
    _assert_lid_ordered_after_columns(sdfg, "canon")


def test_map_to_for_loop_keeps_folded_boundary_write_ordering():
    """Nesting the column map must not strand the access node its folded second write fed."""
    sdfg = _canonical()
    state, columns, lids = _boundary_units(sdfg)
    column_entry = next(state.entry_node(node) for node in columns if isinstance(node, nd.MapExit))
    anchors = [e.src for lid in lids for e in state.in_edges(_ordering_target(state, lid)) if e.data.is_empty()]
    assert any(state.in_degree(anchor) > 0 for anchor in anchors), "premise: the lid ordering is anchored on a write"

    xform = MapToForLoop()
    xform.map_entry = column_entry
    xform.expr_index = 0
    xform._sdfg = sdfg
    xform.state_id = state.parent_graph.node_id(state)
    assert xform.can_be_applied(state, 0, sdfg)
    xform.apply(state, sdfg)
    sdfg.validate()

    _assert_lid_ordered_after_columns(sdfg, "after MapToForLoop")


@pytest.mark.parametrize("target_isa", ["SCALAR", detect_host_isa()])
def test_boundary_corners_survive_canonicalize_vectorize(target_isa):
    """End to end: the corners the lid overwrites must stay 1.0 after canonicalize + vectorize."""
    sdfg = _canonical()
    VectorizeCPUMultiDim(
        VectorizeConfig(widths=(8, ), target_isa=target_isa, remainder_strategy="masked_tail",
                        branch_mode="merge")).apply_pass(sdfg, {})
    sdfg.name = f"{sdfg.name}_{target_isa.lower()}"
    sdfg.validate()

    _assert_lid_ordered_after_columns(sdfg, f"canon_vec[{target_isa}]")

    u = np.full((_SIZE, _SIZE), 7.0, dtype=np.float64)
    sdfg(u=u, N=_SIZE)
    expected = _reference()
    assert u[-1, 0] == 1.0 and u[-1, -1] == 1.0, "the column zeroing overwrote the lid at the corners"
    np.testing.assert_allclose(u, expected, rtol=0, atol=0)


def test_boundary_corners_match_canonical_only():
    """The canon-only path is the control: same numbers, no vectorization involved."""
    sdfg = copy.deepcopy(_canonical())
    sdfg.name = f"{sdfg.name}_canon_only"
    u = np.full((_SIZE, _SIZE), 7.0, dtype=np.float64)
    sdfg(u=u, N=_SIZE)
    np.testing.assert_allclose(u, _reference(), rtol=0, atol=0)
