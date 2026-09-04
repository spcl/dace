# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Unit tests for :class:`WidenAccesses` gather-related behaviour.

The CPP per-lane index materialiser (``materialise_per_lane_index_tile``)
was removed 2026-06-14 (user direction: no CPP index tasklets) -- the K-dim
gather index is now built as tile lib nodes by
:meth:`InsertTileLoadStore._stage_index_via_tileops` into a
``(W_d if dep else ONE)`` index tile (see ``test_kdim_broadcasts.py`` and
``test_strided_gather_scatter.py`` for the end-to-end coverage). What remains
here are the WidenAccesses contracts: it widens + seeds per-lane symbols but
does NOT itself materialise the index tile.
"""
import dace


def test_widen_accesses_returns_none_on_empty_sdfg():
    """No tile-tagged maps -> no widening / gather materialisation -> ``None``.

    Folds the prior ``PreparePerLaneIndices`` Pass-level test into the
    unified ``WidenAccesses`` per user direction 2026-06-11.
    """
    from dace.transformation.passes.vectorization.widen_accesses import WidenAccesses
    sdfg = dace.SDFG("empty")
    sdfg.add_state("s")
    assert WidenAccesses(widths=(8, )).apply_pass(sdfg, {}) is None


def test_widen_accesses_does_not_materialise_idx_tile_for_gather_access():
    """Per user direction 2026-06-11: per-lane idx tile materialisation is
    owned by :class:`InsertTileLoadStore` at TileLoad emission time --
    WidenAccesses handles widening + per-lane SYMBOL fanout (step 5), NOT
    tile materialisation.
    """
    from dace.memlet import Memlet as _Memlet
    from dace.transformation.passes.vectorization.widen_accesses import WidenAccesses

    sdfg = dace.SDFG("walker_gather_fixture")
    sdfg.add_array("A", (32, ), dace.float64, transient=False)
    sdfg.add_array("idx", (32, ), dace.int64, transient=False)
    state = sdfg.add_state("s")
    me, mx = state.add_map("k", {"ii": "0:8"})

    inner = dace.SDFG("body_nsdfg")
    inner.add_array("A", (32, ), dace.float64, transient=False)
    inner.add_array("idx", (32, ), dace.int64, transient=False)
    inner.add_array("out_t", (1, ), dace.float64, transient=True)
    instate = inner.add_state("body")
    a_inner = instate.add_access("A")
    t_inner = instate.add_access("out_t")
    tasklet = instate.add_tasklet("ld", {"_a"}, {"_o"}, "_o = _a")
    from dace.subsets import Range as _Range
    from dace.symbolic import pystr_to_symbolic as _to_sym
    instate.add_edge(a_inner, None, tasklet, "_a",
                     _Memlet(data="A", subset=_Range([(_to_sym("idx[ii]"), _to_sym("idx[ii]"), 1)])))
    instate.add_edge(tasklet, "_o", t_inner, None, _Memlet("out_t[0]"))

    nsdfg = state.add_nested_sdfg(inner, {"A", "idx"}, set(), symbol_mapping={"ii": "ii"})
    a_outer = state.add_access("A")
    idx_outer = state.add_access("idx")
    state.add_memlet_path(a_outer, me, nsdfg, dst_conn="A", memlet=_Memlet("A[0:32]"))
    state.add_memlet_path(idx_outer, me, nsdfg, dst_conn="idx", memlet=_Memlet("idx[0:32]"))
    state.add_nedge(nsdfg, mx, _Memlet())

    before_int_arrays = sum(1 for d in inner.arrays.values()
                            if isinstance(d, dace.data.Array) and d.transient and d.dtype == dace.int64)
    WidenAccesses(widths=(8, )).apply_pass(sdfg, {})
    after_int_arrays = sum(1 for d in inner.arrays.values()
                           if isinstance(d, dace.data.Array) and d.transient and d.dtype == dace.int64)
    assert after_int_arrays == before_int_arrays, (
        "WidenAccesses must NOT materialise idx tiles -- InsertTileLoadStore owns that step")
