# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Pass-level tests for ``ResolveMixedDtypeBinops`` on the ternary ``ITE`` blend and the
masked write ``IT``.

Design 6.2 locks a single dtype per tile lib node: ``TileITE``'s ``_t`` / ``_e`` / ``_o``
connectors must share the output dtype (see ``dace.libraries.tileops.nodes.tile_ite``), and a
masked ``TileStore`` writes ``_val`` straight into the destination. ``ResolveMixedDtypeBinops``
already inserted casts before a plain binop tasklet reached ``ConvertTaskletsToTileOps``; its
AST matcher only recognised the 2-input binop / 1-input bare-copy shapes, so a mismatched-dtype
``ITE(...)`` arm or ``IT(...)`` value sailed through uncast. These tests pin the fix: the pass
must insert an explicit cast tasklet for a mismatched arm/value BEFORE the ITE / masked-write
tasklet, exactly as it already does for a plain binop.
"""
import dace
from dace.transformation.passes.vectorization.resolve_mixed_dtype_binops import ResolveMixedDtypeBinops


def _cast_source_dtype(sdfg: dace.SDFG, state: dace.SDFGState, tasklet: dace.nodes.Tasklet, conn: str):
    """Walk one hop upstream of ``tasklet``'s ``conn`` input and return the dtype the connector
    now reads. When a cast was inserted, this is a fresh cast-tasklet's output (register
    transient of the promoted dtype); when nothing changed, it is the original source array."""
    edge = next(e for e in state.in_edges(tasklet) if e.dst_conn == conn)
    return sdfg.arrays[edge.data.data].dtype, edge.src


def _find_cast_tasklet(state: dace.SDFGState, an: dace.nodes.AccessNode):
    """The tasklet writing ``an``, if ``an`` is a fresh transient produced by a cast."""
    producers = [e.src for e in state.in_edges(an) if isinstance(e.src, dace.nodes.Tasklet)]
    return producers[0] if len(producers) == 1 else None


def _build_ite_sdfg(t_dtype, e_dtype, out_dtype, ite_form: str):
    """Single-state SDFG: ``A[0] = ITE(C[0], T[0], E[0])`` (or the Python-ternary spelling),
    with ``T`` / ``E`` / ``A`` given INDEPENDENT dtypes -- the mismatch this pass must resolve
    before the tile converter locks one dtype per ``TileITE``."""
    sdfg = dace.SDFG(f"ite_mixed_dtype_{ite_form}")
    sdfg.add_array("A", shape=(1, ), dtype=out_dtype)
    sdfg.add_array("T", shape=(1, ), dtype=t_dtype)
    sdfg.add_array("E", shape=(1, ), dtype=e_dtype)
    sdfg.add_array("C", shape=(1, ), dtype=dace.bool_)
    state = sdfg.add_state("only", is_start_block=True)
    rT, rE, rC = state.add_access("T"), state.add_access("E"), state.add_access("C")
    wA = state.add_access("A")
    code = ("_o = ITE(_c, _t, _e)" if ite_form == "call" else "_o = _t if _c else _e")
    t = state.add_tasklet("ite_A", {"_c": None, "_t": None, "_e": None}, {"_o": None}, code)
    state.add_edge(rC, None, t, "_c", dace.Memlet("C[0]"))
    state.add_edge(rT, None, t, "_t", dace.Memlet("T[0]"))
    state.add_edge(rE, None, t, "_e", dace.Memlet("E[0]"))
    state.add_edge(t, "_o", wA, None, dace.Memlet("A[0]"))
    return sdfg, state, t


def _assert_arm_cast_to_output_dtype(sdfg, state, tasklet, conn, out_dtype):
    dt, src = _cast_source_dtype(sdfg, state, tasklet, conn)
    assert dt == out_dtype, f"{conn} reads {dt}, expected the output dtype {out_dtype} after the cast"
    cast = _find_cast_tasklet(state, src)
    assert cast is not None, f"{conn} was not routed through a cast tasklet"
    assert f"dace.{out_dtype}" in cast.code.as_string or out_dtype.type.__name__ in cast.code.as_string


def test_ite_call_form_mismatched_then_arm_gets_cast():
    sdfg, state, t = _build_ite_sdfg(t_dtype=dace.int32, e_dtype=dace.float64, out_dtype=dace.float64, ite_form="call")
    count = ResolveMixedDtypeBinops().apply_pass(sdfg, {})
    assert count == 1
    _assert_arm_cast_to_output_dtype(sdfg, state, t, "_t", dace.float64)
    # The else arm already matched the output dtype -- must be left untouched.
    e_dt, _ = _cast_source_dtype(sdfg, state, t, "_e")
    assert e_dt == dace.float64
    assert "ITE(" in t.code.as_string


def test_ite_call_form_both_arms_mismatched_get_cast():
    sdfg, state, t = _build_ite_sdfg(t_dtype=dace.int32, e_dtype=dace.int64, out_dtype=dace.float64, ite_form="call")
    count = ResolveMixedDtypeBinops().apply_pass(sdfg, {})
    assert count == 1
    _assert_arm_cast_to_output_dtype(sdfg, state, t, "_t", dace.float64)
    _assert_arm_cast_to_output_dtype(sdfg, state, t, "_e", dace.float64)


def test_python_ternary_form_mismatched_else_arm_gets_cast():
    sdfg, state, t = _build_ite_sdfg(t_dtype=dace.float64,
                                     e_dtype=dace.int32,
                                     out_dtype=dace.float64,
                                     ite_form="ternary")
    count = ResolveMixedDtypeBinops().apply_pass(sdfg, {})
    assert count == 1
    _assert_arm_cast_to_output_dtype(sdfg, state, t, "_e", dace.float64)


def test_ite_matching_dtypes_are_left_alone():
    sdfg, state, t = _build_ite_sdfg(t_dtype=dace.float64,
                                     e_dtype=dace.float64,
                                     out_dtype=dace.float64,
                                     ite_form="call")
    count = ResolveMixedDtypeBinops().apply_pass(sdfg, {})
    assert count is None
    assert len(state.nodes()) == 5  # C, T, E, A access nodes + the one tasklet -- no cast inserted


def test_masked_write_mismatched_value_gets_cast():
    """``_o = IT(_c, _v)`` (``NormalizeMaskedWriteTasklets`` form): a value dtype that
    differs from the destination must be cast before the masked ``TileStore`` lowering
    (``ConvertTaskletsToTileOps._convert_conditional_write``) copies it in raw."""
    sdfg = dace.SDFG("masked_write_mixed_dtype")
    sdfg.add_array("A", shape=(1, ), dtype=dace.float64)
    sdfg.add_array("V", shape=(1, ), dtype=dace.int32)
    sdfg.add_array("C", shape=(1, ), dtype=dace.bool_)
    state = sdfg.add_state("only", is_start_block=True)
    rV, rC = state.add_access("V"), state.add_access("C")
    wA = state.add_access("A")
    t = state.add_tasklet("masked_A", {"_c": None, "_v": None}, {"_o": None}, "_o = IT(_c, _v)")
    state.add_edge(rC, None, t, "_c", dace.Memlet("C[0]"))
    state.add_edge(rV, None, t, "_v", dace.Memlet("V[0]"))
    state.add_edge(t, "_o", wA, None, dace.Memlet("A[0]"))

    count = ResolveMixedDtypeBinops().apply_pass(sdfg, {})
    assert count == 1
    _assert_arm_cast_to_output_dtype(sdfg, state, t, "_v", dace.float64)
    assert "IT(" in t.code.as_string
