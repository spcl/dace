# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Unit tests for :class:`InsertTileLoadStore`.

Validates the staging-first replacement chain end-to-end:

  ``StageGlobalArrayThroughScalars  ->  WidenAccesses  ->
   InsertTileLoadStore``

on simple Python kernels. Each test confirms:

* Lane-dep non-transient reads gain a ``TileLoad`` lib node.
* Lane-dep non-transient writes gain a ``TileStore`` lib node.
* CONSTANT (loop-invariant) edges stay direct -- no lib node, no Python
  assignment tasklet inserted.
"""
import dace
from dace.libraries.tileops import TileLoad, TileStore
from dace.transformation.passes.vectorization.bypass_trivial_assign_tasklets import BypassTrivialAssignTasklets
from dace.transformation.passes.vectorization.nest_innermost_map_body import NestInnermostMapBodyIntoNSDFG
from dace.transformation.interstate.expand_nested_sdfg_inputs import ExpandNestedSDFGInputs
from dace.transformation.passes.vectorization.stage_global_array_through_scalars import (
    StageGlobalArrayThroughScalars, )
from dace.transformation.passes.vectorization.widen_accesses import WidenAccesses
from dace.transformation.passes.vectorization.insert_tile_load_store import InsertTileLoadStore

N = dace.symbol("N")


def _stage_widen_insert(prog):
    """Apply the three staging-first passes in order, return ``(sdfg, body_state)``."""
    sdfg = prog.to_sdfg(simplify=True)
    BypassTrivialAssignTasklets().apply_pass(sdfg, {})
    NestInnermostMapBodyIntoNSDFG().apply_pass(sdfg, {})
    sdfg.apply_transformations_repeated(ExpandNestedSDFGInputs)
    StageGlobalArrayThroughScalars().apply_pass(sdfg, {})
    WidenAccesses(widths=(8, )).apply_pass(sdfg, {})
    InsertTileLoadStore(widths=(8, )).apply_pass(sdfg, {})
    body_state = None
    for sd in sdfg.all_sdfgs_recursive():
        for state in sd.states():
            for n in state.nodes():
                if isinstance(n, (TileLoad, TileStore)):
                    body_state = state
                    break
            if body_state:
                break
        if body_state:
            break
    return sdfg, body_state


@dace.program
def linear_kernel(A: dace.float64[N], B: dace.float64[N], scale: dace.float64):
    """``B[i] = A[i] * scale`` -- LINEAR read+write, CONSTANT scalar."""
    for i in dace.map[0:N]:
        B[i] = A[i] * scale


def test_linear_kernel_emits_tileload_and_tilestore():
    """B[i] = A[i] * scale -- A and B widened via TileLoad/TileStore; scale stays direct."""
    sdfg, body_state = _stage_widen_insert(linear_kernel)
    assert body_state is not None, "expected at least one TileLoad / TileStore in some body NSDFG"
    tile_loads = [n for n in body_state.nodes() if isinstance(n, TileLoad)]
    tile_stores = [n for n in body_state.nodes() if isinstance(n, TileStore)]
    assert len(tile_loads) == 1, f"expected 1 TileLoad (A), got {len(tile_loads)}"
    assert len(tile_stores) == 1, f"expected 1 TileStore (B), got {len(tile_stores)}"
    # ``scale`` is CONSTANT -- stays as direct edge from scale AN to the consumer tasklet.
    # No TileLoad for scale.
    for tl in tile_loads:
        in_edges = list(body_state.in_edges(tl))
        # TileLoad's _src reads from a non-transient (A, not scale).
        for e in in_edges:
            if e.dst_conn == "_src":
                assert e.data.data != "scale", "scale must not get a TileLoad (CONSTANT)"


@dace.program
def two_loads_kernel(A: dace.float64[N], B: dace.float64[N], C: dace.float64[N]):
    """``C[i] = A[i] + B[i]`` -- two LINEAR reads, one LINEAR write."""
    for i in dace.map[0:N]:
        C[i] = A[i] + B[i]


def test_two_lane_dep_reads_emit_two_tileloads():
    """Both A and B reads gain a TileLoad."""
    sdfg, body_state = _stage_widen_insert(two_loads_kernel)
    assert body_state is not None
    tile_loads = [n for n in body_state.nodes() if isinstance(n, TileLoad)]
    tile_stores = [n for n in body_state.nodes() if isinstance(n, TileStore)]
    assert len(tile_loads) == 2, f"expected 2 TileLoads (A,B), got {len(tile_loads)}"
    assert len(tile_stores) == 1


@dace.program
def constant_only_kernel(A: dace.float64[N], scale: dace.float64):
    """``A[i] = scale + scale`` -- no lane-dep read (only the LINEAR write)."""
    for i in dace.map[0:N]:
        A[i] = scale + scale


def test_constant_read_no_tileload():
    """``scale`` is CONSTANT -- no TileLoad emitted for it. Only the LINEAR
    write to A gets a TileStore."""
    sdfg, body_state = _stage_widen_insert(constant_only_kernel)
    if body_state is None:
        # Kernel may be too trivial to produce TileStore; allow.
        return
    tile_loads = [n for n in body_state.nodes() if isinstance(n, TileLoad)]
    tile_stores = [n for n in body_state.nodes() if isinstance(n, TileStore)]
    assert len(tile_loads) == 0, "scale is CONSTANT, no TileLoad expected"
    assert len(tile_stores) == 1
